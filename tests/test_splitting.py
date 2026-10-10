import numpy as np
import pandas as pd
import pytest

from iiot_ids_benchmark.data import load_dataset
from iiot_ids_benchmark.splitting import split_data


def assert_disjoint(split):
    a, b, c = map(set, (split.train, split.validation, split.test))
    assert not (a & b or a & c or b & c)


def test_row_split_disjoint_deterministic(synthetic_csv):
    data = load_dataset(synthetic_csv)
    split, again = split_data(data), split_data(data)
    assert_disjoint(split)
    assert np.array_equal(split.train, again.train)
    assert [len(split.train), len(split.validation), len(split.test)] == [360, 120, 120]
    assert split.metadata["strategy"] == "row"
    assert split.metadata["warnings"]
    assert all(not v for v in split.metadata["missing_categories"].values())


def test_chronological_ties_and_cross_boundary_purge(synthetic_csv, tmp_path):
    frame = pd.read_csv(synthetic_csv)
    frame["StartTime"] = 1600000000 + np.arange(len(frame)) // 2
    frame["LastTime"] = frame["StartTime"] + 0.1
    frame.loc[358, "LastTime"] = frame.loc[360, "StartTime"] + 10
    path = tmp_path / "time.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(path)
    split = split_data(data)
    assert_disjoint(split)
    assert split.metadata["strategy"] == "chronological"
    assert split.metadata["purged_rows"] == 1
    assert 358 not in set(split.train) | set(split.validation) | set(split.test)
    a, b, c = (
        set(data.frame.loc[idx, "StartTime"]) for idx in (split.train, split.validation, split.test)
    )
    assert not (a & b or a & c or b & c)


def test_explicit_chronological_does_not_fallback(synthetic_csv):
    data = load_dataset(synthetic_csv)
    with pytest.raises(ValueError, match="chronological"):
        split_data(data, strategy="chronological")


def test_group_partitions_no_shared_groups(synthetic_csv, tmp_path):
    frame = pd.read_csv(synthetic_csv)
    frame["SessionId"] = np.arange(len(frame)) // 10
    path = tmp_path / "groups.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(path)
    split = split_data(data, strategy="group", group_column="session_id")
    assert_disjoint(split)
    groups = [
        set(data.frame.loc[p, "SessionId"]) for p in (split.train, split.validation, split.test)
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert split.metadata["group_column"] == "SessionId"


@pytest.mark.parametrize("val,test", [(0, 0.2), (0.8, 0.3), (0.2, -1)])
def test_invalid_split_fractions(synthetic_csv, val, test):
    with pytest.raises(ValueError, match="fractions"):
        split_data(load_dataset(synthetic_csv), validation_fraction=val, test_fraction=test)
