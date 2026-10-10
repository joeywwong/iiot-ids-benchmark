"""Strict local CSV loading, bounded uniform sampling and provenance."""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = ("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId")
DEFAULT_LABEL_MAPPING = {
    "normal": 0,
    "normal traffic": 0,
    **{
        name: 1
        for attack in ("dos", "reconnaissance", "command injection", "backdoor")
        for name in (attack, f"{attack} traffic")
    },
}
REPORTED_COUNTS = {"observations": 1194464, "normal": 1107448, "attack": 87016}


def column_key(name: str) -> str:
    """Accept case/whitespace/underscore variants; reject ambiguous duplicate names."""
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def label_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def map_labels(traffic: pd.Series, mapping: dict | None = None) -> pd.Series:
    mapping = DEFAULT_LABEL_MAPPING if mapping is None else mapping
    if not mapping or any(type(v) is not int or v not in (0, 1) for v in mapping.values()):
        raise ValueError("label_mapping must explicitly map strings to integer 0 or 1")
    normalized = {label_key(k): v for k, v in mapping.items()}
    if len(normalized) != len(mapping):
        raise ValueError("label_mapping has duplicate normalized labels")
    values = traffic.map(label_key)
    unknown = sorted(set(values) - normalized.keys())
    if unknown:
        raise ValueError(f"Unknown Traffic labels: {unknown}. Inspect CSV and set label_mapping.")
    return values.map(normalized).astype("int8")


def inspect_schema(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    keys = [column_key(c) for c in header]
    if not header or any(not k for k in keys) or len(set(keys)) != len(keys):
        raise ValueError("CSV needs unique, nonempty column names (including normalized names)")
    lookup = dict(zip(keys, header, strict=True))
    required = (*LEAKAGE_COLUMNS, "Traffic")
    missing = [c for c in required if column_key(c) not in lookup]
    if missing:
        raise ValueError(f"Missing required original dataset columns: {missing}")
    return header, lookup


@dataclass
class Dataset:
    frame: pd.DataFrame
    y: pd.Series
    traffic: pd.Series
    columns: dict[str, str]
    provenance: dict


def load_dataset(path: Path, max_rows: int = 100000, seed: int = 42, mapping=None) -> Dataset:
    """Scan every row; retain the smallest random priorities for a uniform subset.

    The scan validates labels and counts the full file. It is not a head-of-file sample.
    The retained DataFrame index is the zero-based source CSV row number.
    """
    path = Path(path)
    if max_rows < 0:
        raise ValueError("max_rows must be nonnegative (0 means all rows)")
    header, columns = inspect_schema(path)
    traffic_column = columns["traffic"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    rng = np.random.default_rng(seed)
    retained, priorities = None, np.empty(0)
    chunks, total, counts = [], 0, {}
    with pd.read_csv(path, dtype=str, encoding="utf-8-sig", chunksize=50000) as reader:
        for chunk in reader:
            chunk.index = pd.RangeIndex(total, total + len(chunk))
            chunk.columns = header
            map_labels(chunk[traffic_column], mapping)
            for label, count in chunk[traffic_column].value_counts(dropna=False).items():
                counts[str(label)] = counts.get(str(label), 0) + int(count)
            total += len(chunk)
            if max_rows == 0:
                chunks.append(chunk)
            else:
                keys = rng.random(len(chunk))
                retained = chunk if retained is None else pd.concat([retained, chunk])
                priorities = np.concatenate([priorities, keys])
                if len(retained) > max_rows:
                    keep = np.argpartition(priorities, max_rows - 1)[:max_rows]
                    retained = retained.iloc[keep]
                    priorities = priorities[keep]
    if total == 0:
        raise ValueError("CSV contains no observations")
    frame = pd.concat(chunks) if max_rows == 0 else retained
    frame = frame.sort_index()
    traffic = frame[traffic_column].copy()
    y = map_labels(traffic, mapping)
    full_attack = sum(
        count
        for label, count in counts.items()
        if map_labels(pd.Series([label]), mapping).iloc[0] == 1
    )
    actual = {"observations": total, "normal": total - full_attack, "attack": full_attack}
    provenance = {
        "filename": path.name,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "rows_in_file": total,
        "rows_selected": len(frame),
        "full_dataset": len(frame) == total,
        "sampling": "all rows" if len(frame) == total else "uniform random without replacement",
        "sampling_balanced": False,
        "file_traffic_counts": counts,
        "selected_traffic_counts": traffic.value_counts().to_dict(),
        "categories_missing_from_sample": sorted(set(counts) - set(traffic)),
        "observed_counts": actual,
        "author_reported_counts": REPORTED_COUNTS,
        "count_discrepancies": {
            k: {"observed": actual[k], "reported": v}
            for k, v in REPORTED_COUNTS.items()
            if actual[k] != v
        },
        "observed_column_count": len(header),
        "author_reported_feature_count": 41,
        "label_mapping": DEFAULT_LABEL_MAPPING if mapping is None else mapping,
        "observed_dtypes": {c: "string (CSV input)" for c in header},
    }
    return Dataset(frame, y, traffic, columns, provenance)
