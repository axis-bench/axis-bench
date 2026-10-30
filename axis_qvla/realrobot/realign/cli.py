"""One command from phase curves to gated REALALIGN signal artifacts.

OFFLINE TIER. Run under the offline venv:

    # export-keyed curves + the alignment map: builds a metadata-only corpus view
    python -m axis_qvla.realrobot.realign build --out DIR --export-curves DIR --alignment-map FILE --prompts FILE
    # a rendered corpus with its curve store (the renderer writes both)
    python -m axis_qvla.realrobot.realign build --out DIR --roots FILE --curves-store DIR
    # re-run the gates over a built directory (exit 1 when one fails)
    python -m axis_qvla.realrobot.realign gates --out DIR
    # compile the four stage-1 runs (bc, conditioning, awr, filtering) to compiled labels
    python -m axis_qvla.realrobot.realign compile --built DIR --out DIR
    # phase curves of a rendered pretraining-data tree from the prepared simulation demonstrations in
    # qvla-v1-realworld: the attempt store `render labels` reads
    python -m axis_qvla.realrobot.realign phase --rendered DIR --data DIR --out DIR [--tasks N ...]

`--runs` (`build`, `gates`, `compile`) selects the stage-1 runs: a build for `bc conditioning` fits only
their labels (rank weights, tags, ranges), with no phase-balanced AWR fit and no filtering schedule. A
build records its runs in `runs.json`; `gates` and `compile` follow it and refuse a run it does not have.

Exit codes: 0 done; 1 a gate failed (`build` and `gates`, naming the gates; see `gates.json`); 2 a
refusal, naming what was refused: a directory that is not a `build --out` directory, one whose
corpus base does not resolve, a missing or malformed input or a flag combination `build` cannot use
(refused before anything is created), `$AXIS_PRETRAIN_CORPUS_ROOT` set (any command), an `--runs` name
the build does not have, and a build whose gates fail (`compile`: no compiled labels are produced from
it).

Writes, in DIR (only for the selected `--runs`, except the universal `curves/`, `roots.json`,
`ranges.json`, `runs.json` and `gates.json`): `awr_weights_phase_rank.json` (the ranking input,
'conditioning'), `awr_weights_phase.json` (phase-balanced row AWR, 'awr'), `awr_weights_phase_ep.json`
(episode AWR, 'filtering'), `quality_phase.npz`
(pure-conditioning tags from the rank weights, 'conditioning'), `schedule_drop_top.npz` (drop-top 50%
over the episode weights, plan 20000 x 64, seed 0, 'filtering').

The episode weights and the schedule come from the unchanged `axis_qvla.signals` CLIs. Everything that
REALALIGN adds is in this package.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from axis_qvla.signals.methods import awr_episode as baw_ep
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.awr_chunk import CORPUS_ROOT_ENV
from axis_qvla.realrobot.realign.curves import (
    CurveError, load_export_curves, read_curve_store, read_dumped_curve_store, write_curve_store,
)
from axis_qvla.realrobot.realign.gates import PLAN, GateError, run_gates
from axis_qvla.realrobot.realign.ranges import tail_ranges, write_ranges
from axis_qvla.realrobot.realign.tags import build_pure_tags
from axis_qvla.realrobot.realign.view import view_episodes_from_map, write_meta_view
from axis_qvla.realrobot.realign.weights import MIN_FIT_POOL, balanced_weights, rank_weights, write_weights
from axis_qvla.realrobot.realign.runs import (
    RUNS, BUILT, UnbuiltRunError, check_runs_built, recorded_runs, required_files, write_recorded_runs,
)

KEEP_TOP_FRAC = 0.5
SEED = 0
OVERRIDE_REFUSAL = (f"${CORPUS_ROOT_ENV} is set: it overrides the roots index's base in the tag and schedule "
                    f"builders but not in the weights or the fingerprint, so the artifacts would describe two "
                    f"corpora. Unset it.")


def _mode(path: Path) -> int | None:
    """`path`'s stat mode, or None when it does not exist. Any other error (an unreadable parent, say)
    raises: it never reads as absent (pathlib's is_dir swallows it on some Pythons)."""
    try:
        return os.stat(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None


def not_a_build(flag: str, path: Path, *, runs: tuple[str, ...] = RUNS) -> str | None:
    """Why `path` (given as `flag`), a directory with a runs record, is not a `build --out` directory, or
    None when it is one: lacking an artifact `runs` needs, or naming a corpus base that does not resolve
    from where it now sits (a build over `--roots` keeps its corpus by a path relative to the build, so
    it moves with it or not at all)."""
    path = Path(path)
    lacking = [name for name in required_files(runs) if not stat.S_ISREG(_mode(path / name) or 0)]
    if lacking:
        return f"{flag} {path} is not a `realign build --out` directory: it lacks {lacking}"
    base = json.loads((path / "roots.json").read_text(encoding="utf-8")).get("base", ".")
    if not stat.S_ISDIR(_mode(path / base) or 0):
        return (f"{flag} {path}: roots.json's corpus base {base!r} does not resolve from {path}; a build "
                f"over --roots moves together with its corpus")
    return None


def build(out_dir: Path, *, export_curves: Path | None = None, alignment_map: Path | None = None,
          prompts: Path | None = None, roots: Path | None = None, curves_store: Path | None = None,
          min_fit_pool: int = MIN_FIT_POOL, valid_from_data: bool = False,
          runs: tuple[str, ...] = RUNS) -> dict:
    """Build only what `runs` need into `out_dir`, gate them, and return a short summary.

    Every input is read -- so a missing or malformed `--export-curves`/`--alignment-map`/`--prompts` or
    `--roots`/`--curves-store`, or a `runs` name outside `RUNS`, is refused -- before `out_dir` is
    created: a build refused at this point leaves nothing. (A refusal from deeper in the build, e.g.
    a weights pool too small for `--min-fit-pool`, or a failing gate, runs after some artifacts are
    already written.)
    """
    if os.environ.get(CORPUS_ROOT_ENV):
        raise RuntimeError(OVERRIDE_REFUSAL)
    unknown = [r for r in runs if r not in RUNS]
    if unknown:
        raise ValueError(f"--runs names {unknown}, not one of {RUNS}")
    if not runs:
        raise ValueError("--runs must name at least one run")
    if export_curves is not None:
        if alignment_map is None or prompts is None or roots is not None or curves_store is not None:
            raise ValueError("--export-curves needs --alignment-map and --prompts, and excludes --roots/--curves-store")
        if valid_from_data:
            raise ValueError("--valid-from-data reads a --curves-store; it does not apply to --export-curves")
    elif roots is None or curves_store is None:
        raise ValueError("give either --export-curves/--alignment-map/--prompts or --roots/--curves-store")

    if export_curves is not None:
        curves = load_export_curves(export_curves, alignment_map)
        names = {int(k): str(v) for k, v in json.loads(Path(prompts).read_text()).items()}
        view_episodes = view_episodes_from_map(alignment_map, names)
    else:
        curves = (read_dumped_curve_store(curves_store, roots) if valid_from_data
                  else read_curve_store(curves_store))
        roots_doc = json.loads(Path(roots).read_text())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_curve_store(out / "curves", curves)
    if export_curves is not None:
        roots_path = write_meta_view(out, view_episodes)
    else:
        roots_path = out / "roots.json"
        base = (Path(roots).parent / roots_doc.get("base", ".")).resolve()
        roots_doc["base"] = os.path.relpath(base, out.resolve())
        roots_path.write_text(json.dumps(roots_doc))
    corpus = (roots_path.parent / json.loads(roots_path.read_text()).get("base", ".")).resolve()

    write_ranges(out / "ranges.json", tail_ranges(roots_path))

    if "conditioning" in runs:
        write_weights(out / "awr_weights_phase_rank.json", rank_weights(curves), corpus=corpus, balanced=False)

    scales = None
    if "awr" in runs or "filtering" in runs:    # 'filtering''s episode weights are derived FROM the balanced fit
        eps, scales = balanced_weights(curves, min_pool=min_fit_pool)
        write_weights(out / "awr_weights_phase.json", eps, corpus=corpus, balanced=True, scales=scales)

    if "filtering" in runs:
        if baw_ep.main(["--weights", str(out / "awr_weights_phase.json"), "--ranges", str(out / "ranges.json"),
                        "--out", str(out / "awr_weights_phase_ep.json")]) != 0:
            raise RuntimeError("episode AWR weights failed")

    if "conditioning" in runs:
        build_pure_tags(out / "awr_weights_phase_rank.json", roots_path, out / "ranges.json",
                        out / "quality_phase.npz")

    if "filtering" in runs:
        if bis.main(["--mode", "drop_top", "--weights", str(out / "awr_weights_phase_ep.json"),
                     "--roots", str(roots_path), "--ranges", str(out / "ranges.json"),
                     "--out", str(out / "schedule_drop_top.npz"), "--total-steps", str(PLAN[0]),
                     "--batch", str(PLAN[1]), "--seed", str(SEED), "--keep-top-frac", str(KEEP_TOP_FRAC)]) != 0:
            raise RuntimeError("drop-top schedule failed")

    write_recorded_runs(out, runs)

    gates = run_gates(out, runs=runs)
    summary = {"episodes": gates["G1"]["episodes"], "neutral": gates["G2"]["neutral"],
              "n_rows": gates["G3"]["n_rows"], "trainable_rows": gates["G3"]["trainable_rows"]}
    if scales is not None:
        summary["phase_balance"] = scales
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.realrobot.realign", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="phase curves -> gated signal artifacts")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--export-curves", type=Path)
    b.add_argument("--alignment-map", type=Path)
    b.add_argument("--prompts", type=Path, help='JSON {"<task_id>": "<instruction>"}')
    b.add_argument("--roots", type=Path)
    b.add_argument("--curves-store", type=Path)
    b.add_argument("--min-fit-pool", type=int, default=MIN_FIT_POOL,
                   help="per-phase fit-pool minimum; lower it only for a small corpus (it is stamped)")
    b.add_argument("--valid-from-data", action="store_true",
                   help="--curves-store is an AWR --dump-curves store: masks and neutral episodes from the "
                        "pretraining data")
    b.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS),
                   help="which stage-1 runs to build (default: all four, bc conditioning awr filtering). A "
                        "build for a subset does not fit AWR or produce the filtering schedule unless "
                        "'awr'/'filtering' is named")
    g = sub.add_parser("gates", help="re-run the gates over a built directory")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--runs", nargs="+", choices=RUNS,
                   help="which runs to check (default: the build's own recorded runs)")
    bu = sub.add_parser("compile", help="compile the four stage-1 runs (bc, conditioning, awr, filtering) "
                                        "to compiled labels")
    bu.add_argument("--built", type=Path, required=True, help="a `build --out` directory")
    bu.add_argument("--out", type=Path, required=True)
    bu.add_argument("--runs", nargs="+", choices=RUNS,
                   help="which runs to compile (default: the build's own recorded runs)")
    ph = sub.add_parser("phase", help="phase curves of rendered episodes from the published recordings")
    ph.add_argument("--rendered", type=Path, required=True,
                    help="your re-render's DROID-8D view (the directory holding roots.json)")
    ph.add_argument("--data", type=Path, required=True, help="the prepared simulation demonstrations in qvla-v1-realworld (render prepare --out)")
    ph.add_argument("--out", type=Path, required=True, help="the attempt store to write")
    ph.add_argument("--tasks", type=int, nargs="+")
    a = ap.parse_args(argv)
    if os.environ.get(CORPUS_ROOT_ENV):
        print(f"FAIL  {OVERRIDE_REFUSAL}", file=sys.stderr)
        return 2
    if a.cmd in ("gates", "compile"):
        target = a.out if a.cmd == "gates" else a.built
        flag = "--out" if a.cmd == "gates" else "--built"
        # A run the build lacks is named before not_a_build lists the files it lacks.
        try:
            runs = tuple(x for x in RUNS if x in a.runs) if a.runs else recorded_runs(target)
            check_runs_built(target, runs)
        except UnbuiltRunError as exc:
            print(f"FAIL  {flag} {exc}", file=sys.stderr)
            return 2
        refusal = not_a_build(flag, target, runs=runs)
        if refusal:
            print(f"FAIL  {refusal}", file=sys.stderr)
            return 2
    if a.cmd == "gates":
        try:
            report = run_gates(a.out, runs=runs)
        except GateError as exc:
            print(f"FAIL  {exc}; see {a.out / 'gates.json'}", file=sys.stderr)
            return 1
        print(json.dumps({k: v["ok"] for k, v in report.items()}))
        return 0
    if a.cmd == "compile":
        from axis_qvla.realrobot.realign.bundles import write_bundles
        try:
            ids = write_bundles(a.built, a.out, runs=runs)
        except GateError as exc:
            print(f"FAIL  {exc}; no compiled labels are produced from a build whose gates fail "
                  f"(see {a.built / 'gates.json'})", file=sys.stderr)
            return 2
        except ValueError as exc:            # a weights file the row space refuses, H(q) = 0
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2
        print(json.dumps(ids, indent=1))
        return 0
    if a.cmd == "phase":
        from axis_qvla.realrobot.realign.curves import write_attempt_store
        from axis_qvla.realrobot.realign.phase import phase_curves
        try:
            got = phase_curves(a.rendered / "roots.json", a.data, a.tasks)
        except (FileNotFoundError, ValueError) as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2
        write_attempt_store(a.out, got)
        n_curves = sum(c.t0 is not None for c in got.values())
        print(f"{len(got)} episodes: {n_curves} curves, {len(got) - n_curves} neutral -> {a.out}")
        return 0
    try:
        summary = build(a.out, export_curves=a.export_curves, alignment_map=a.alignment_map, prompts=a.prompts,
                        roots=a.roots, curves_store=a.curves_store,
                        min_fit_pool=a.min_fit_pool, valid_from_data=a.valid_from_data,
                        runs=tuple(a.runs))
    except GateError as exc:
        print(f"FAIL  {exc}; see {a.out / 'gates.json'}", file=sys.stderr)
        return 1
    except (CurveError, FileNotFoundError, ValueError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1))
    return 0
