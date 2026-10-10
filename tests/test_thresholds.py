import numpy as np
import pytest

from iiot_ids_benchmark.evaluation import evaluate
from iiot_ids_benchmark.thresholds import calibrate_threshold, predict_at_threshold


def test_calibration_uses_only_benign_validation_scores():
    normal = np.arange(100, dtype=float)
    scores = np.r_[normal, [-1000, 1000]]
    labels = np.r_[np.zeros(100), [1, 1]]
    calibration = calibrate_threshold(scores, labels, 0.01)
    baseline = calibrate_threshold(normal, np.zeros(100), 0.01)
    assert calibration == baseline
    assert calibration.threshold == 98 and calibration.validation_fpr == 0.01
    assert calibration.benign_validation_count == 100


def test_ties_and_small_validation_are_conservative():
    scores = np.r_[np.zeros(90), np.ones(10)]
    c = calibrate_threshold(scores, np.zeros(100), 0.01)
    assert c.threshold == 1 and c.validation_fpr == 0
    assert predict_at_threshold([1, 1.01], c.threshold).tolist() == [0, 1]
    c = calibrate_threshold([1, 2, 3], [0, 0, 0], 0.01)
    assert c.allowed_false_positives == 0 and c.threshold == 3


@pytest.mark.parametrize("target", [0, 0.01, 0.1, 0.5, 0.99])
def test_empirical_fpr_never_above_target(target):
    rng = np.random.default_rng(42)
    scores = rng.integers(0, 20, 301)
    c = calibrate_threshold(scores, np.zeros(len(scores)), target)
    assert np.mean(predict_at_threshold(scores, c.threshold)) <= target


@pytest.mark.parametrize(
    "scores,labels,target",
    [
        ([1], [1], 0.01),
        ([np.nan], [0], 0.01),
        ([1], [0], 1),
        ([1], [0], -0.01),
        ([1, 2], [0], 0.01),
        ([1], [2], 0.01),
    ],
)
def test_bad_calibration_inputs(scores, labels, target):
    with pytest.raises(ValueError):
        calibrate_threshold(scores, labels, target)


def test_fpr_correct_denominator_and_undefined_cases():
    metrics, matrix = evaluate([0, 0, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["test_fpr"] == pytest.approx(1 / 3)
    assert matrix.tolist() == [[2, 1], [0, 1]]
    metrics, _ = evaluate([1, 1], [0, 1], [0.1, 0.9])
    assert metrics["test_fpr"] is None and metrics["test_roc_auc"] is None
    metrics, _ = evaluate([0, 0], [0, 0], [0.1, 0.2])
    assert metrics["test_precision"] is None and metrics["test_recall"] is None
    assert metrics["test_pr_auc_average_precision"] is None
