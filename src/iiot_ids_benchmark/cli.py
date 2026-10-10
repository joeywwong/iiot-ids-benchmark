"""Command line entry point; flags override YAML and built-in defaults."""

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from .experiment import Config, run_experiment


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline binary IIoT intrusion detection benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Train and evaluate three baselines on a local CSV")
    run.add_argument("--data", type=Path, required=True, help="Original local CSV (any filename)")
    run.add_argument("--output", type=Path, default=Path("results"), help="New results directory")
    run.add_argument("--config", type=Path, help="YAML config; omitted keys use built-in defaults")
    for option in (
        "seed",
        "max_rows",
        "rf_trees",
        "if_trees",
        "n_jobs",
        "importance_rows",
        "importance_repeats",
        "timing_repeats",
    ):
        run.add_argument("--" + option.replace("_", "-"), type=int, default=None)
    for option in ("target_fpr", "validation_fraction", "test_fraction"):
        run.add_argument("--" + option.replace("_", "-"), type=float, default=None)
    run.add_argument("--split", choices=["auto", "row", "chronological", "group"])
    run.add_argument(
        "--group-column", help="Verified capture/session column; excluded from features"
    )
    run.add_argument(
        "--data-kind",
        choices=["wustl", "synthetic"],
        default="wustl",
        help="Use synthetic for fixtures; artifacts will be labeled accordingly",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = {}
        names = {f.name for f in fields(Config)}
        if args.config:
            settings = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
            if not isinstance(settings, dict) or set(settings) - names:
                raise ValueError("Config must be a mapping with only documented Config fields")
        for name in names:
            value = getattr(args, name, None)
            if value is not None:
                settings[name] = value
        run_experiment(args.data, args.output, Config(**settings), args.data_kind)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
