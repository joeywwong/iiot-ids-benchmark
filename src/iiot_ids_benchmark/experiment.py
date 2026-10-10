"""One explicit load → split → fit → calibrate → evaluate workflow."""

import hashlib
import importlib.metadata
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .data import DEFAULT_LABEL_MAPPING, load_dataset
from .evaluation import evaluate, per_attack_recall
from .features import audit_features
from .models import attack_scores, fit_model, make_models
from .reporting import check_output, validation_importance, write_reports
from .splitting import split_data
from .thresholds import calibrate_threshold, predict_at_threshold


@dataclass
class Config:
    seed: int = 42
    max_rows: int = 100000
    rf_trees: int = 100
    if_trees: int = 100
    n_jobs: int = 1
    target_fpr: float = 0.01
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    split: str = "auto"
    group_column: str | None = None
    importance_rows: int = 1000
    importance_repeats: int = 3
    timing_repeats: int = 3
    label_mapping: dict = field(default_factory=lambda: DEFAULT_LABEL_MAPPING.copy())

    def validate(self):
        for key in (
            "rf_trees",
            "if_trees",
            "importance_rows",
            "importance_repeats",
            "timing_repeats",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.max_rows) is not int or self.max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.n_jobs) is not int or self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        if not 0 <= self.target_fpr < 1:
            raise ValueError("target_fpr must be in [0, 1)")


def run_experiment(data: Path, output: Path, config: Config, data_kind="wustl") -> dict:
    config.validate()
    if data_kind not in {"wustl", "synthetic"}:
        raise ValueError("data_kind must be wustl or synthetic")
    check_output(output)
    print("Loading CSV and auditing labels...", flush=True)
    dataset = load_dataset(data, config.max_rows, config.seed, config.label_mapping)
    split = split_data(
        dataset,
        config.split,
        config.validation_fraction,
        config.test_fraction,
        config.seed,
        config.group_column,
    )
    print(f"Split: {split.metadata['strategy']}; selected {len(dataset.frame)} rows", flush=True)
    features, audit = audit_features(
        dataset.frame.loc[split.train],
        dataset.y.loc[split.train],
        dataset.traffic.loc[split.train],
        split.metadata.get("group_column"),
    )
    X = dataset.frame[features]
    X_train, X_val, X_test = (X.loc[p] for p in (split.train, split.validation, split.test))
    y_train, y_val, y_test = (dataset.y.loc[p] for p in (split.train, split.validation, split.test))
    models = make_models(config.seed, config.rf_trees, config.if_trees, config.n_jobs)
    metrics, attacks, matrices, test_scores, calibrations, fit_details = [], [], {}, {}, {}, {}
    common = {
        "data_kind": data_kind,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "n_test": len(X_test),
        "train_attack_prevalence": float(y_train.mean()),
        "validation_attack_prevalence": float(y_val.mean()),
        "test_attack_prevalence": float(y_test.mean()),
        "target_fpr": config.target_fpr,
    }
    categories = set(dataset.traffic[dataset.y == 1])
    for name, pipeline in models.items():
        print(f"Fitting {name}...", flush=True)
        started = time.perf_counter()
        fit_rows = fit_model(name, pipeline, X_train, y_train)
        training_seconds = time.perf_counter() - started
        calibration = calibrate_threshold(
            attack_scores(name, pipeline, X_val), y_val, config.target_fpr
        )
        # Untimed warmup; subsequent whole-test batches include transformation and scoring.
        scores = attack_scores(name, pipeline, X_test)
        durations = []
        for _ in range(config.timing_repeats):
            started = time.perf_counter()
            attack_scores(name, pipeline, X_test)
            durations.append(time.perf_counter() - started)
        batch_seconds = float(np.median(durations))
        predictions = predict_at_threshold(scores, calibration.threshold)
        measured, matrix = evaluate(y_test, predictions, scores)
        row = {
            "model": name,
            **common,
            "fit_rows": fit_rows,
            "fit_attack_prevalence": 0.0 if name == "isolation_forest" else float(y_train.mean()),
            "threshold": calibration.threshold,
            "validation_fpr": calibration.validation_fpr,
            **measured,
            "training_seconds": training_seconds,
            "batch_inference_seconds": batch_seconds,
            "amortized_ms_per_flow": 1000 * batch_seconds / len(X_test),
            "flows_per_second": len(X_test) / batch_seconds if batch_seconds else None,
        }
        metrics.append(row)
        attacks.extend(
            {"model": name, **r}
            for r in per_attack_recall(
                y_test, dataset.traffic.loc[split.test], predictions, categories
            )
        )
        matrices[name], test_scores[name] = matrix, scores
        calibrations[name] = calibration.to_dict()
        cleaner = pipeline.named_steps["preprocess"].named_steps["clean"]
        fit_details[name] = {
            "fit_rows": fit_rows,
            "normal_only": name == "isolation_forest",
            "numeric_features": cleaner.numeric_columns_,
            "categorical_features": [c for c in features if c not in cleaner.numeric_columns_],
            "estimator_parameters": pipeline.named_steps["model"].get_params(),
        }
    importance, importance_info = validation_importance(
        models["random_forest"],
        X_val,
        y_val,
        config.importance_rows,
        config.importance_repeats,
        config.seed,
        config.n_jobs,
    )
    split_hashes = {
        name: hashlib.sha256(np.sort(index).astype("<i8").tobytes()).hexdigest()
        for name, index in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
    }
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "data_kind": data_kind,
        "config": asdict(config),
        "dataset": dataset.provenance,
        "features": features,
        "feature_review": audit,
        "split": split.metadata,
        "source_row_index_sha256": split_hashes,
        "samples": common,
        "calibration": calibrations,
        "model_fitting": fit_details,
        "permutation_importance": importance_info,
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "iiot-ids-benchmark",
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "PyYAML",
            )
        },
        "hardware": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "timing": {
            "clock": "perf_counter",
            "warmup_batches": 1,
            "repeats": config.timing_repeats,
            "batch_size": len(X_test),
            "aggregation": "median whole-test batch; preprocessing included",
            "caveat": "Approximate offline batch throughput, not live packet latency",
        },
        "limitations": [
            "Supervised models fit attack labels; Isolation Forest selects benign rows with labels",
            "Shared feature audit uses labeled training data; IF pipeline fits benign rows only",
            "Testbed flow data do not establish production or unseen-attack detection performance",
            "Leakage review is heuristic; inspect feature semantics and inference availability",
            "No tuning or threshold selection on test data; test FPR can exceed validation target",
        ],
    }
    write_reports(output, metrics, attacks, matrices, test_scores, y_test, importance, manifest)
    print(f"Aggregate reports written to {output}", flush=True)
    return manifest
