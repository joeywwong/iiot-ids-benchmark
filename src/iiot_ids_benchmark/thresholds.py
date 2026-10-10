"""Conservative empirical FPR calibration with an explicit tie convention."""

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Calibration:
    threshold: float
    target_fpr: float
    validation_fpr: float
    benign_validation_count: int
    allowed_false_positives: int
    convention: str = "predict attack iff score > threshold; ties are benign"

    def to_dict(self):
        return asdict(self)


def calibrate_threshold(scores, labels, target_fpr=0.01) -> Calibration:
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not set(labels) <= {0, 1}:
        raise ValueError("Scores and binary labels must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or not 0 <= target_fpr < 1:
        raise ValueError("Finite scores and target FPR in [0, 1) required")
    benign = scores[labels == 0]
    if not len(benign):
        raise ValueError("Cannot calibrate without benign validation observations")
    allowed = math.floor(target_fpr * len(benign))
    threshold = float(np.sort(benign)[len(benign) - allowed - 1])
    return Calibration(
        threshold, target_fpr, float(np.mean(benign > threshold)), len(benign), allowed
    )


def predict_at_threshold(scores, threshold):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must be finite")
    return (scores > threshold).astype("int8")
