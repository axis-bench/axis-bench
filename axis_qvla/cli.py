"""`qvla`: the command line of AXIS-Bench.

Groups: bench (check a benchmark, evaluate a policy, score), data (get and verify downloads against
the release checksum manifest), init (write ./qvla.toml), paper (reproduce the paper's tables from
per-seed logs), labels
(compile, export and check the paper's quality labels and training weights), results (build a
results file from a run, score it, paired contrasts, re-read checks), realrobot (kinematic renderer,
REALALIGN labels, FR3 data and configs), train (render a run spec into a launchable training run),
env (build and check the training and evaluation environments). Each group is imported only when
the parser is built, so importing this module pulls in nothing else.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
import importlib
import sys

_GROUPS = (
    ("axis_qvla.bench.cli", "add_bench_parsers"),
    ("axis_qvla.data.cli", "add_data_parsers"),
    ("axis_qvla.data.init_cli", "add_init_parsers"),
    ("axis_qvla.paper.cli", "add_paper_parsers"),
    ("axis_qvla.signals.cli", "add_signals_parsers"),
    ("axis_qvla.bench.results.cli", "add_results_parsers"),
    ("axis_qvla.realrobot.cli", "add_realrobot_parsers"),
    ("axis_qvla.train.cli", "add_train_parsers"),
    ("axis_qvla.train.env_cli", "add_env_parsers"),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qvla",
        description="AXIS-Bench: a benchmark for quality-aware VLA pretraining.")
    sub = p.add_subparsers(dest="command", required=True)
    for module, fn in _GROUPS:
        getattr(importlib.import_module(module), fn)(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
