from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from iiot_ids_benchmark.data import DEFAULT_LABEL_MAPPING, LEAKAGE_COLUMNS, load_dataset, map_labels
from iiot_ids_benchmark.features import audit_features


def test_labels_are_explicit():
    result = map_labels(pd.Series([" Normal Traffic ", "DoS", "BACKDOOR", "command injection"]))
    assert result.tolist() == [0, 1, 1, 1]
    for values in (["new attack"], [None], ["0"], ["benign"]):
        with pytest.raises(ValueError, match="Unknown Traffic"):
            map_labels(pd.Series(values))


@pytest.mark.parametrize(
    "label,expected",
    [
        ("normal", 0),
        ("normal traffic", 0),
        ("dos", 1),
        ("dos traffic", 1),
        ("reconn", 1),
        ("reconnaissance", 1),
        ("reconnaissance traffic", 1),
        ("comminj", 1),
        ("command injection", 1),
        ("command injection traffic", 1),
        ("backdoor", 1),
        ("backdoor traffic", 1),
    ],
)
@pytest.mark.parametrize("variant", [str.lower, str.upper, str.title, lambda s: f"  {s.upper()}  "])
def test_all_supported_label_aliases(label, expected, variant):
    assert map_labels(pd.Series([variant(label)])).tolist() == [expected]


def test_yaml_defaults_match_authoritative_label_mapping():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["label_mapping"] == DEFAULT_LABEL_MAPPING


@pytest.mark.parametrize(
    "mapping",
    [
        {"normal": 2},
        {"normal": True},
        {},
        {"Normal": 0, " normal ": 1},
        {"Normal": 0, " normal ": 0},
        {"normal": "0"},
        {"normal": 0.0},
        {"": 0},
        {"   ": 0},
        {1: 0},
        [],
        [0, 1],
        0,
        "normal",
    ],
)
def test_bad_label_mapping(mapping):
    with pytest.raises(ValueError):
        map_labels(pd.Series(["normal"]), mapping)


def test_custom_mapping_requires_explicit_semantics():
    assert map_labels(pd.Series(["benign", "scan"]), {"benign": 0, "scan": 1}).tolist() == [0, 1]


def test_schema_and_target_exclusions(synthetic_csv):
    data = load_dataset(synthetic_csv, max_rows=0)
    frame = data.frame.copy()
    frame["Label"] = data.y
    frame["Target"] = data.y
    frame["unremarkable_name"] = 1 - data.y
    frame["CopiedTraffic"] = data.traffic
    frame["hidden_code"] = data.traffic.map(
        {"Normal Traffic": "a", "DoS": "b", "Reconnaissance": "c"}
    )
    frame["scaled_code"] = data.y * 10 + 20
    frame["constant"] = "same"
    frame["uuid"] = [f"flow-{n}" for n in range(len(frame))]
    features, audit = audit_features(frame, data.y, data.traffic)
    assert set(features) == {"TotBytes", "TotPkts", "Proto"}
    assert "hidden_code" in audit["excluded_columns"]
    assert "scaled_code" in audit["excluded_columns"]
    assert (
        set(
            (
                *LEAKAGE_COLUMNS,
                "Traffic",
                "Label",
                "Target",
                "unremarkable_name",
                "CopiedTraffic",
                "constant",
                "uuid",
            )
        )
        <= audit["excluded_columns"].keys()
    )


def test_case_variants_and_missing_schema(synthetic_csv, tmp_path):
    frame = pd.read_csv(synthetic_csv).rename(columns={"sIpId": " S_IP_ID ", "Traffic": "TRAFFIC"})
    path = tmp_path / "case.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(path)
    assert data.columns["sipid"] == " S_IP_ID "
    features, _ = audit_features(data.frame, data.y, data.traffic)
    assert " S_IP_ID " not in features and "TRAFFIC" not in features
    frame.drop(columns=["SrcAddr"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        load_dataset(path)


def test_duplicate_normalized_headers(synthetic_csv, tmp_path):
    frame = pd.read_csv(synthetic_csv)
    frame["src_addr"] = "duplicate"
    path = tmp_path / "duplicate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path)


def test_sampling_is_reproducible_uniform_not_head(synthetic_csv):
    data = load_dataset(synthetic_csv, max_rows=60, seed=42)
    again = load_dataset(synthetic_csv, max_rows=60, seed=42)
    assert data.frame.index.equals(again.frame.index)
    assert len(data.frame) == 60 and data.frame.index.max() > 300
    assert data.provenance["rows_in_file"] == 600
    assert data.provenance["observed_counts"] == {"observations": 600, "normal": 480, "attack": 120}
    assert data.provenance["sampling_balanced"] is False
    assert data.provenance["full_dataset"] is False
    assert len(data.provenance["sha256"]) == 64


def test_labels_validated_outside_retained_sample(synthetic_csv, tmp_path):
    frame = pd.read_csv(synthetic_csv)
    frame.loc[599, "Traffic"] = "UNKNOWN"
    path = tmp_path / "unknown.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="UNKNOWN|unknown"):
        load_dataset(path, max_rows=1)


def test_invalid_numeric_values_become_missing():
    from iiot_ids_benchmark.features import FrameCleaner

    frame = pd.DataFrame({"number": ["1", "2", "inf"], "category": ["tcp", None, "udp"]})
    cleaner = FrameCleaner().fit(frame)
    transformed = cleaner.transform(
        pd.DataFrame({"number": ["-inf", "bad"], "category": [None, "new"]})
    )
    assert transformed["number"].isna().all()
    assert pd.isna(transformed["category"].iloc[0])
    assert np.issubdtype(transformed["number"].dtype, np.floating)
