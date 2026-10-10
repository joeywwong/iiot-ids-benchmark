"""Training-only feature review and learned preprocessing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import LEAKAGE_COLUMNS, column_key, label_key


def audit_features(frame, y, traffic, group_column=None) -> tuple[list[str], dict]:
    """Review labeled TRAINING data only; this is not a proof of leakage freedom."""
    prohibited = {column_key(c) for c in (*LEAKAGE_COLUMNS, "Traffic")}
    target_names = {
        "target",
        "label",
        "class",
        "attack",
        "isattack",
        "attacktype",
        "attacklabel",
        "binarylabel",
        "binarytarget",
        "trafficlabel",
        "trafficclass",
        "category",
    }
    excluded, warnings, features, diagnostics = {}, [], [], {}
    for name in frame:
        key = column_key(name)
        series = frame[name]
        if key in prohibited:
            excluded[name] = "author-identified identifier/time field or Traffic target"
            continue
        if (group_column and name == group_column) or key in target_names:
            excluded[name] = "split group or additional target-like column"
            continue
        if any(token in key for token in ("label", "target", "attack", "traffic")):
            excluded[name] = "name suggests a target-derived column; review before inclusion"
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if np.array_equal(numeric.to_numpy(), y.to_numpy()) or np.array_equal(
            numeric.to_numpy(), 1 - y.to_numpy()
        ):
            excluded[name] = "exact binary target copy/complement in training data"
            continue
        if series.map(label_key).equals(traffic.map(label_key)):
            excluded[name] = "exact Traffic target copy in training data"
            continue
        unique = int(series.nunique(dropna=True))
        # A bijective low-cardinality encoding can hide a renamed target column.
        complete = series.notna().all()
        if complete and unique == 2 and y.nunique() == 2:
            codes = pd.DataFrame({"value": series, "target": y})
            if codes.groupby("value")["target"].nunique().eq(1).all():
                excluded[name] = "bijective binary target encoding in training data"
                continue
        if complete and unique == traffic.nunique():
            codes = pd.DataFrame({"value": series, "traffic": traffic})
            if codes.groupby("value")["traffic"].nunique().eq(1).all():
                excluded[name] = "bijective Traffic encoding in training data"
                continue
        is_numeric = bool(numeric[series.notna()].notna().all())
        diagnostics[name] = {
            "training_unique_values": unique,
            "training_missing": int(series.isna().sum()),
            "training_nonfinite_numeric": int(np.isinf(numeric.to_numpy()).sum()),
            "numeric_on_training": is_numeric,
        }
        if not is_numeric and unique > 200 and unique / len(series) > 0.5:
            excluded[name] = "high-cardinality nonnumeric field; possible identifier"
            continue
        if unique <= 1:
            excluded[name] = "constant or entirely missing in training"
            continue
        if key in {"id", "index", "unnamed0", "flowid", "sessionid", "recordid", "rowid"} or any(
            token in key for token in ("addr", "ipid", "timestamp", "time", "uuid")
        ):
            excluded[name] = "possible identifier/timestamp; conservative exclusion"
            continue
        if is_numeric and unique / len(series) > 0.95:
            warnings.append(f"{name}: nearly unique numeric values; inspect inference availability")
        features.append(name)
    if not features:
        raise ValueError("No usable predictors remain after training-only leakage review")
    return features, {
        "review_partition": "training only",
        "excluded_columns": excluded,
        "feature_diagnostics": diagnostics,
        "warnings": warnings,
        "caveat": (
            "Heuristic review cannot eliminate all leakage or certify inference availability. "
            "Flow aggregates require completed flows; inspect unexpected columns before claims."
        ),
    }


class FrameCleaner(TransformerMixin, BaseEstimator):
    """Learn numeric/categorical typing on fit rows; convert infinities to missing."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.numeric_columns_ = [
            c for c in X if pd.to_numeric(X[c], errors="coerce")[X[c].notna()].notna().all()
        ]
        self.fit_rows_ = np.asarray(X.index)
        return self

    def transform(self, X):
        result = X.loc[:, self.feature_names_in_].copy()
        for c in result:
            if c in self.numeric_columns_:
                result[c] = (
                    pd.to_numeric(result[c], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .astype(float)
                )
            else:
                result[c] = result[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        return result


def make_preprocessor(scale: bool) -> Pipeline:
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scale", StandardScaler(with_mean=False)))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore", max_categories=64, min_frequency=2, dtype=np.float32
                ),
            ),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline([("clean", FrameCleaner()), ("transform", transform)])
