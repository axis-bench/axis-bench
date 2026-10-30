"""A quality method of your own: keep the top share of trainable frames by a per-frame score.

docs/tutorials/write-a-quality-method.md walks through this file. The method is composed from the
shipped blocks (axis_qvla.signals.compile.Compose): a Scorer that passes an imported scores file through,
the TopFraction selector and the schedule planner. compile_program checks the composition rules and
compiles the selector's size-matched random control with it.

    python experiments/examples/keep_top.py --scores S.json      the paper's budget (keep 0.5, 20,605 x 64,
                                                                   seed 42), into <cache>/bundles
    python experiments/examples/keep_top.py --scores S.json --toy --steps 6 --batch 2 --out bundles
    python experiments/examples/keep_top.py --scores S.json --seed 43 --name example.keep_top.s43

Prints one line per set of compiled labels written: its name (the program's method; --name, default
example.keep_top, with its random control as <name>.control) and its plan's shape. It records the
name, so a run spec names the labels as `name:example.keep_top`. A compile at another seed takes its
own --name: a name resolves to its latest compile, so compiling seed 43 under the seed-42 name would
re-point every spec that names it.

Offline tier: numpy, pandas (the pretraining data's frame index reads parquet metadata).
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

from axis_qvla.signals.bundle import write_bundle
from axis_qvla.signals.compile import Compose, compile_program
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.methods.filtering import TopFraction
from axis_qvla.signals.planners import SchedulePlanner
from axis_qvla.signals.schema import Budget


@dataclasses.dataclass(frozen=True)
class ImportedValues:
    """Scorer: the per-frame values of an imported scores file, unchanged."""

    def score(self, labels, frames):
        return labels.values()


DEFAULT_NAME = "example.keep_top"


def keep_top(keep: float, name: str = DEFAULT_NAME) -> Compose:
    """Keep the top `keep` of trainable frames by the imported value; draw them with the schedule planner.
    `name` is the method the compiled labels are recorded under; the control's is `<name>.control`."""
    return Compose(method=name, version=1, params={"keep": keep}, scorer=ImportedValues(),
                   selector=TopFraction(keep), planner=SchedulePlanner("drop_top"))


def rowspace(args: argparse.Namespace):
    if args.toy:
        from axis_qvla.signals.testing import toy_rowspace

        return toy_rowspace()
    from axis_qvla.signals.rowspace import paper_rowspaces

    return paper_rowspaces(args.pack, args.corpus)[args.rowspace]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", type=pathlib.Path, required=True,
                    help="a scores file: {'weights': {episode: [one value per frame]}}")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--toy", action="store_true", help="the 23-row toy frame index of axis_qvla.signals.testing")
    where.add_argument("--pretraining-data", dest="pack", metavar="DIR", type=pathlib.Path, default=None,
                       help="a local copy of the pretraining data (default: <corpus>/qvla-v1-corpus of ./qvla.toml)")
    ap.add_argument("--episodes", dest="corpus", metavar="DIR", type=pathlib.Path, default=None,
                    help="its episodes/ directory (default: <pretraining data>/episodes)")
    ap.add_argument("--rowspace", default="libero5k_v2_d8", help="one of the paper's frame indexes")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20605, help="plan steps (default: the paper's, 20,605)")
    ap.add_argument("--batch", type=int, default=64, help="plan batch (default: the paper's, 64)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help=f"the name the compiled labels are recorded under (default {DEFAULT_NAME}); the "
                         f"control's is <name>.control. Give each seed its own, e.g. {DEFAULT_NAME}.s43")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="compiled-labels root (default: <cache>/bundles of ./qvla.toml, where qvla train looks)")
    ap.add_argument("--host", default=None, help="this machine's qvla.toml (default: ./qvla.toml)")
    args = ap.parse_args(argv)
    from axis_qvla.train.names import NameResolutionError, check_labels_name, record_labels_name

    try:
        for name in (args.name, f"{args.name}.control"):
            check_labels_name(name)
    except NameResolutionError as exc:
        print(f"FAIL  --name: {exc}")
        return 2
    if args.out is None or (args.pack is None and not args.toy):
        from axis_qvla.core.hosts import HostProfileError
        from axis_qvla.data.layout import CORPUS, default_dest
        from axis_qvla.data.userconf import user_paths
        from axis_qvla.train.ids import local_bundles_root

        try:
            paths = user_paths(args.host)
        except HostProfileError as exc:
            print(f"FAIL  {exc}")
            return 2
        args.out = args.out or local_bundles_root(paths)
        if args.pack is None and not args.toy:
            args.pack = default_dest(CORPUS, paths)
    if not args.toy and args.corpus is None:
        args.corpus = args.pack / "episodes"
    bundle, controls = compile_program(keep_top(args.keep, args.name), ImportedScores.from_json(args.scores), rowspace(args),
                                       Budget(args.steps, args.batch), args.seed)
    for b in (bundle, *controls):
        record_labels_name(args.out, b.program["method"], write_bundle(b, args.out))
        print(f"OK    {b.program['method']}  plan {tuple(b.plan.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
