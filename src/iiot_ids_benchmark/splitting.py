"""Assess timestamps before exclusion; keep groups and time ties out of shared splits."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import column_key


@dataclass
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict


def parse_time(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        median = float(numeric.median())
        # Argus epoch timestamps; smaller counters cannot establish a real chronology.
        for unit, lower, upper in (("s", 1e8, 1e11), ("ms", 1e11, 1e14)):
            if lower <= median < upper:
                return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    # Explicit ISO/common date strings only; do not guess ambiguous day/month formats.
    if not values.fillna("").str.match(r"^\d{4}[-/]\d{2}[-/]\d{2}").all():
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(values, format="ISO8601", utc=True, errors="coerce")


def _validate(train, validation, test, y):
    partitions = [set(p) for p in (train, validation, test)]
    if any(partitions[i] & partitions[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Split partitions overlap")
    if min(map(len, partitions)) < 2:
        raise ValueError("Each split requires at least two observations")
    if set(y.loc[train]) != {0, 1}:
        raise ValueError("Training split must contain normal and attack observations")
    if not (y.loc[validation] == 0).any():
        raise ValueError("Validation requires benign observations for FPR calibration")


def split_data(
    dataset, strategy="auto", validation_fraction=0.2, test_fraction=0.2, seed=42, group_column=None
) -> Split:
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Split fractions must be positive and sum to less than one")
    frame, y, traffic = dataset.frame, dataset.y, dataset.traffic
    indices = frame.index.to_numpy()
    metadata = {"requested_strategy": strategy, "warnings": []}
    if strategy not in {"auto", "row", "chronological", "group"}:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if group_column and strategy != "group":
        raise ValueError("group_column requires split: group")
    if strategy == "group":
        lookup = {column_key(c): c for c in frame}
        actual = lookup.get(column_key(group_column)) if group_column else None
        if not actual or frame[actual].isna().any():
            raise ValueError("Group split requires an existing, nonmissing group column")
        groups = frame[actual]
        remaining_pos, test_pos = next(
            GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
                indices, groups=groups
            )
        )
        train_pos, val_pos = next(
            GroupShuffleSplit(
                n_splits=1, test_size=validation_fraction / (1 - test_fraction), random_state=seed
            ).split(remaining_pos, groups=groups.iloc[remaining_pos])
        )
        train, validation, test = (
            indices[remaining_pos[train_pos]],
            indices[remaining_pos[val_pos]],
            indices[test_pos],
        )
        metadata.update(strategy="group", group_column=actual, purged_rows=0)
    else:
        start = parse_time(frame[dataset.columns["starttime"]])
        end = parse_time(frame[dataset.columns["lasttime"]])
        valid_time = bool(start.notna().all() and end.notna().all() and (end >= start).all())
        metadata["time_assessment"] = {
            "start_parse_fraction": float(start.notna().mean()),
            "end_parse_fraction": float(end.notna().mean()),
            "unique_start_times": int(start.nunique()),
            "complete_ordered_intervals": valid_time,
            "group_assessment": "No documented capture/session ID; addresses are attack-specific",
        }
        chrono_error = "timestamps incomplete, ambiguous, or not valid flow intervals"
        if strategy in {"auto", "chronological"} and valid_time and start.nunique() >= 3:
            ordered = start.sort_values(kind="stable")
            val_boundary = ordered.iloc[int(len(frame) * (1 - validation_fraction - test_fraction))]
            test_boundary = ordered.iloc[int(len(frame) * (1 - test_fraction))]
            train = indices[(start < val_boundary) & (end < val_boundary)]
            validation = indices[
                (start >= val_boundary) & (start < test_boundary) & (end < test_boundary)
            ]
            test = indices[start >= test_boundary]
            try:
                _validate(train, validation, test, y)
                metadata.update(
                    strategy="chronological",
                    validation_start=val_boundary.isoformat(),
                    test_start=test_boundary.isoformat(),
                    purged_rows=len(frame) - len(train) - len(validation) - len(test),
                    time_ties="equal StartTime values stay in the same partition",
                )
                return _finish(train, validation, test, metadata, traffic)
            except ValueError as exc:
                chrono_error = str(exc)
        if strategy == "chronological":
            raise ValueError(f"Cannot form chronological split: {chrono_error}")
        if strategy == "auto":
            metadata["warnings"].append(f"Chronological split unavailable: {chrono_error}")
        counts = traffic.value_counts()
        strata = traffic if counts.min() >= 5 else y
        metadata["stratification"] = "Traffic" if strata is traffic else "binary target"
        try:
            remaining, test = train_test_split(
                indices, test_size=test_fraction, random_state=seed, stratify=strata
            )
            train, validation = train_test_split(
                remaining,
                test_size=validation_fraction / (1 - test_fraction),
                random_state=seed,
                stratify=strata.loc[remaining],
            )
        except ValueError as exc:
            raise ValueError(
                f"Insufficient samples for requested stratified splits: {exc}"
            ) from exc
        metadata.update(strategy="row", purged_rows=0)
        metadata["warnings"].append(
            "Row-level splitting may share related sessions/near-duplicates and inflate metrics"
        )
    _validate(train, validation, test, y)
    return _finish(train, validation, test, metadata, traffic)


def _finish(train, validation, test, metadata, traffic):
    metadata["traffic_counts"] = {
        name: traffic.loc[index].value_counts().to_dict()
        for name, index in (("train", train), ("validation", validation), ("test", test))
    }
    categories = set(traffic)
    metadata["missing_categories"] = {
        name: sorted(categories - set(counts))
        for name, counts in metadata["traffic_counts"].items()
    }
    if any(metadata["missing_categories"].values()):
        metadata["warnings"].append("Some categories absent from splits; per-attack claims limited")
    return Split(np.asarray(train), np.asarray(validation), np.asarray(test), metadata)
