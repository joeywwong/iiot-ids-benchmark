"""Small aggregate artifacts only: no model binaries or observation-level exports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve

OUTPUT_FILES = {
    "metrics.csv",
    "per_attack_recall.csv",
    "confusion_matrices.json",
    "confusion_matrices.png",
    "precision_recall.png",
    "permutation_importance.csv",
    "manifest.json",
}


def check_output(output: Path):
    output = Path(output)
    if output.exists() and any((output / name).exists() for name in OUTPUT_FILES):
        raise ValueError("Output contains an earlier experiment; choose a new --output directory")
    output.mkdir(parents=True, exist_ok=True)


def validation_importance(pipeline, X, y, cap=1000, repeats=3, seed=42, n_jobs=1):
    """Uniform, unbalanced validation sample; permutation of original source columns."""
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(X), size=min(cap, len(X)), replace=False))
    subset_X, subset_y = X.iloc[positions], y.iloc[positions]
    metadata = {
        "partition": "validation",
        "sample_rows": len(positions),
        "attack_prevalence": float(subset_y.mean()),
        "sampling": "uniform without class balancing",
        "repeats": repeats,
        "scoring": "average_precision",
        "seed": seed,
        "caveat": "Correlated predictors can mask importance; predictive association is not causal",
    }
    if subset_y.nunique() < 2:
        metadata["status"] = "undefined: sampled validation rows contain only one class"
        return pd.DataFrame(columns=["feature", "mean_ap_decrease", "std_ap_decrease"]), metadata
    importance = permutation_importance(
        pipeline,
        subset_X,
        subset_y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "mean_ap_decrease": importance.importances_mean,
            "std_ap_decrease": importance.importances_std,
        }
    ).sort_values("mean_ap_decrease", ascending=False)
    metadata["status"] = "measured"
    return table, metadata


def write_reports(output, metrics, per_attack, matrices, scores, y_test, importance, manifest):
    output = Path(output)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(
        per_attack, columns=["model", "attack", "test_count", "detected", "recall"]
    ).to_csv(output / "per_attack_recall.csv", index=False)
    payload = {
        "label_order": ["normal", "attack"],
        "orientation": "rows=true, columns=predicted",
        "models": {name: matrix.tolist() for name, matrix in matrices.items()},
    }
    (output / "confusion_matrices.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    synthetic = manifest["data_kind"] == "synthetic"
    prefix = "SYNTHETIC verification — " if synthetic else ""
    fig, axes = plt.subplots(1, len(matrices), figsize=(12, 3.8))
    for ax, (name, matrix) in zip(axes, matrices.items(), strict=True):
        ConfusionMatrixDisplay(matrix, display_labels=["normal", "attack"]).plot(
            ax=ax, colorbar=False, values_format="d"
        )
        ax.set_title(name.replace("_", " "), fontsize=10)
    fig.suptitle(prefix + "Held-out test confusion matrices (validation-calibrated thresholds)")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    if y_test.nunique() == 2:
        for name, values in scores.items():
            precision, recall, _ = precision_recall_curve(y_test, values)
            ap = average_precision_score(y_test, values)
            ax.plot(recall, precision, label=f"{name.replace('_', ' ')} (AP={ap:.3f})")
        ax.axhline(float(y_test.mean()), color="gray", linestyle="--", label="test prevalence")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Undefined: test contains one class", ha="center")
    ax.set(
        xlabel="Attack recall",
        ylabel="Attack precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=prefix + "Test precision-recall\nDescriptive only; thresholds fixed on validation",
    )
    fig.tight_layout()
    fig.savefig(output / "precision_recall.png", dpi=150)
    plt.close(fig)
    importance.to_csv(output / "permutation_importance.csv", index=False)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
