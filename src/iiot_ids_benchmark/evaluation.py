"""Held-out binary metrics; absent denominators are undefined, not perfect scores."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def evaluate(y, predictions, scores):
    y = np.asarray(y)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(n) for n in matrix.ravel())
    both = len(np.unique(y)) == 2
    return {
        "test_precision": ratio(tp, tp + fp),
        "test_recall": ratio(tp, tp + fn),
        "test_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "test_fpr": ratio(fp, fp + tn),
        "test_pr_auc_average_precision": float(average_precision_score(y, scores))
        if both
        else None,
        "test_roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }, matrix


def per_attack_recall(y, traffic, predictions, all_attack_categories):
    predictions = np.asarray(predictions)
    rows = []
    for category in sorted(all_attack_categories):
        mask = (np.asarray(traffic) == category) & (np.asarray(y) == 1)
        count = int(mask.sum())
        detected = int(predictions[mask].sum())
        rows.append(
            {
                "attack": category,
                "test_count": count,
                "detected": detected,
                "recall": ratio(detected, count),
            }
        )
    return rows
