"""Synthetic flows for correctness checks only; not WUSTL observations."""

import numpy as np
import pandas as pd
import pytest


def synthetic_frame(n=600, seed=7):
    rng = np.random.default_rng(seed)
    labels = np.array(
        ["Normal Traffic"] * (n * 4 // 5)
        + ["DoS"] * (n // 10)
        + ["Reconnaissance"] * (n - n * 4 // 5 - n // 10)
    )
    rng.shuffle(labels)
    attack = labels != "Normal Traffic"
    return pd.DataFrame(
        {
            "StartTime": ["unavailable"] * n,
            "LastTime": ["unavailable"] * n,
            "SrcAddr": ["192.0.2.1"] * n,
            "DstAddr": ["192.0.2.2"] * n,
            "sIpId": rng.integers(0, 1000, n),
            "dIpId": rng.integers(0, 1000, n),
            "TotBytes": np.maximum(0, rng.normal(100 + 200 * attack, 30, n)),
            "TotPkts": rng.integers(1, 20, n) + 15 * attack,
            "Proto": rng.choice(["tcp", "udp"], n),
            "Traffic": labels,
        }
    )


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "synthetic-fixture.csv"
    synthetic_frame().to_csv(path, index=False)
    return path
