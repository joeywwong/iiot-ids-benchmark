import json

import numpy as np
import pandas as pd
import pytest

from iiot_ids_benchmark.cli import main
from iiot_ids_benchmark.experiment import Config
from iiot_ids_benchmark.models import attack_scores, fit_model, make_models
from iiot_ids_benchmark.reporting import OUTPUT_FILES


def test_preprocessing_fit_only_training_and_if_normal_only():
    X = pd.DataFrame(
        {
            "size": ["1", "2", "3", "400", "500"],
            "proto": ["tcp", "tcp", "udp", "attack-proto", "attack-proto"],
        },
        index=[10, 11, 12, 13, 14],
    )
    y = pd.Series([0, 0, 0, 1, 1], index=X.index)
    held_out = pd.DataFrame(
        {"size": ["999999", "missing"], "proto": ["test-only", "udp"]}, index=[99, 100]
    )
    for name, pipeline in make_models(rf_trees=5, if_trees=5).items():
        count = fit_model(name, pipeline, X, y)
        preprocessor = pipeline.named_steps["preprocess"]
        cleaner = preprocessor.named_steps["clean"]
        expected = [10, 11, 12] if name == "isolation_forest" else list(X.index)
        assert cleaner.fit_rows_.tolist() == expected
        assert count == len(expected)
        encode = preprocessor.named_steps["transform"].named_transformers_["categorical"]
        categories = encode.named_steps["encode"].categories_[0]
        assert "test-only" not in categories
        if name == "isolation_forest":
            assert "attack-proto" not in categories
        numeric = preprocessor.named_steps["transform"].named_transformers_["numeric"]
        before = numeric.named_steps["impute"].statistics_.copy()
        assert before.tolist() == [2 if name == "isolation_forest" else 3]
        assert np.isfinite(attack_scores(name, pipeline, held_out)).all()
        assert np.array_equal(before, numeric.named_steps["impute"].statistics_)
        assert cleaner.fit_rows_.tolist() == expected
        if name == "logistic_regression":
            assert numeric.named_steps["scale"].mean_.tolist() == [181.2]


def test_isolation_forest_attack_score_direction():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"flow": rng.normal(0, 1, 500)})
    y = pd.Series(np.zeros(len(X), dtype=int))
    model = make_models(if_trees=50)["isolation_forest"]
    fit_model("isolation_forest", model, X, y)
    examples = pd.DataFrame({"flow": [0.0, 1000.0]})
    scores = attack_scores("isolation_forest", model, examples)
    assert np.allclose(scores, -model.score_samples(examples))
    assert scores[1] > scores[0]


def test_synthetic_end_to_end_cli(synthetic_csv, tmp_path):
    output = tmp_path / "synthetic-reports"
    main(
        [
            "run",
            "--data",
            str(synthetic_csv),
            "--output",
            str(output),
            "--data-kind",
            "synthetic",
            "--max-rows",
            "0",
            "--rf-trees",
            "8",
            "--if-trees",
            "8",
            "--importance-rows",
            "100",
            "--importance-repeats",
            "1",
            "--timing-repeats",
            "1",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_kind"] == "synthetic"
    assert manifest["features"] == ["TotBytes", "TotPkts", "Proto"]
    assert manifest["samples"]["n_test"] == 120
    assert manifest["dataset"]["full_dataset"] is True
    assert manifest["permutation_importance"]["partition"] == "validation"
    assert manifest["model_fitting"]["isolation_forest"]["fit_rows"] == 288
    metrics = pd.read_csv(output / "metrics.csv")
    assert len(metrics) == 3 and set(metrics["n_test"]) == {120}
    assert (metrics["validation_fpr"] <= 0.01).all()
    assert (metrics["training_seconds"] > 0).all()
    assert (metrics["batch_inference_seconds"] > 0).all()
    assert len(pd.read_csv(output / "per_attack_recall.csv")) == 6
    assert (output / "precision_recall.png").stat().st_size > 1000
    assert (output / "confusion_matrices.png").stat().st_size > 1000
    with pytest.raises(SystemExit) as exc:
        main(["run", "--data", str(synthetic_csv), "--output", str(output)])
    assert exc.value.code == 2


def test_config_override_and_unknown_keys(synthetic_csv, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("unknown_setting: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(synthetic_csv),
                "--config",
                str(config),
                "--output",
                str(tmp_path / "invalid-output"),
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("mapping", ["[]", "0", "{normal: true}"])
def test_cli_rejects_invalid_custom_label_mapping(synthetic_csv, tmp_path, capsys, mapping):
    config = tmp_path / "invalid-mapping.yaml"
    config.write_text(f"label_mapping: {mapping}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--data",
                str(synthetic_csv),
                "--config",
                str(config),
                "--output",
                str(tmp_path / "invalid-mapping-output"),
            ]
        )
    assert exc.value.code == 2
    assert "label_mapping must explicitly map" in capsys.readouterr().err


def test_abbreviated_synthetic_cli_with_yaml_defaults(abbreviated_synthetic_csv, tmp_path):
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    output = tmp_path / "abbreviated-synthetic-reports"
    main(
        [
            "run",
            "--data",
            str(abbreviated_synthetic_csv),
            "--output",
            str(output),
            "--config",
            str(config_path),
            "--data-kind",
            "synthetic",
            "--max-rows",
            "0",
            "--rf-trees",
            "8",
            "--if-trees",
            "8",
            "--importance-repeats",
            "1",
            "--importance-rows",
            "100",
            "--timing-repeats",
            "1",
        ]
    )
    assert set(p.name for p in output.iterdir()) == OUTPUT_FILES
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_kind"] == "synthetic"
    assert manifest["config"]["rf_trees"] == 8  # CLI overrides YAML defaults.
    assert manifest["dataset"]["observed_counts"] == {
        "observations": 600,
        "normal": 480,
        "attack": 120,
    }
    assert set(manifest["dataset"]["file_traffic_counts"]) == {
        "normal",
        "DoS",
        "Reconn",
        "CommInj",
        "Backdoor",
    }
    metrics = pd.read_csv(output / "metrics.csv")
    assert set(metrics["data_kind"]) == {"synthetic"}
    assert set(metrics["n_test"]) == {120}
    assert (metrics["validation_fpr"] <= 0.01).all()
    per_attack = pd.read_csv(output / "per_attack_recall.csv")
    assert len(per_attack) == 12
    assert set(per_attack["attack"]) == {"DoS", "Reconn", "CommInj", "Backdoor"}
    assert (per_attack["test_count"] > 0).all()
    assert per_attack.groupby("model")["test_count"].sum().eq(24).all()


@pytest.mark.parametrize(
    "config",
    [
        Config(rf_trees=0),
        Config(n_jobs=0),
        Config(target_fpr=1),
        Config(max_rows=-1),
        Config(seed=-1),
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        config.validate()
