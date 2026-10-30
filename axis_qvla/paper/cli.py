"""`paper tables`: reproducing the paper's tables -- every simulation figure value, Table I and the
in-text numbers.

    python -m axis_qvla.cli paper tables --check                    (the v1.0 Hub results)
    python -m axis_qvla.cli paper tables --results <local pack> --out <dir>

Run it from a checkout: every default (the results map, the paper layout, benchmark, --checksums) is
read from the checkout's conf/, and a tree without it exits 3 saying so.
An hf:// source resolves through --checksums (default conf/release/v1.0.lock.json) and caches under
--cache (default ./.qvla-cache). --out writes the outputs after --check's comparison.

`paper reeval` re-evaluates released checkpoints -- one result, Table I or a figure, optionally one
backbone's part -- and writes the same paper-layout outputs from the user's own results files, blank
where a result was not re-evaluated (axis_qvla.paper.reeval plans it, axis_qvla.paper.reeval_run runs it,
axis_qvla.paper.reeval_tables lays it out). No published value is read.

With the committed results map, expected/MANIFEST.json is pinned (FROZEN_EXPECTED) for a local pack
as the checksum manifest pins it for hf://, so `--check` certifies the outputs against the frozen
expected/, not merely against whatever expected/ the pack carries.

Exit codes: 0 identical / ok; 1 an output differs from expected/; 2 integrity failure (a sha,
a fingerprint, the benchmark, the results map, a results or manifest file the results map names that
the pack lacks, the release checksum manifest, or an expected/MANIFEST.json that is malformed or not
the pinned one); 3 a byte could not be fetched or read.

Core tier: stdlib only. The HF token and endpoint are read here, through the hf_http helpers,
and only for an hf:// source.
"""
from __future__ import annotations

import argparse
import pathlib

from axis_qvla.bench.spec import load_benchmark
from axis_qvla.paper.expected import write_expected
from axis_qvla.paper.fetch import FetchError, ShaMismatch, open_source
from axis_qvla.paper.spec import load_spec
from axis_qvla.paper.t0 import check, rebuild
from axis_qvla.core.canon import sha256_bytes
from axis_qvla.data.hf import endpoint_from_env, token_from_env
from axis_qvla.core.lock import SOURCE_TREE, default_lock, load_lock
from axis_qvla.data.layout import ORG
from axis_qvla.paper.reeval import BACKBONES, FIGURE_UNITS, TABLES

REPO = SOURCE_TREE                    # every default below is the checkout's (axis_qvla.core.lock)
DEFAULT_CELLS = REPO / "experiments" / "paper" / "cells.json"
# The expected/ each committed cell map was frozen with: sha256(cell map bytes) ->
# sha256(expected/MANIFEST.json). Keyed by the map's BYTES, not its path, so a copy of the paper's
# map is pinned too and any other map (a test pack's) reads its manifest unpinned. The value is
# the MANIFEST.json write_expected gives for the 14 golden outputs, the frozen expected/ of
# qvla-v1-results; tests/golden/paper/test_t0_pinning.py ties both digests to the committed files.
FROZEN_EXPECTED = {
    "c92062984f2084a4f031069ad37fad4521f5b4be1ee0ca2ba554536be3d8a624":
        "9f1b03a0314160ad449a56587eec300cfb77e86c11a3beea8ae336601f551719",
}
CHECKOUT = ("`qvla paper tables` runs from a checkout of the repository: `pip install -e .` in it, or "
            "`python -m axis_qvla.cli paper tables ...` from its root (not `pip install .`, which carries no conf/)")
# The v1.0 results on the Hub: what `paper tables` reads unless --results names another pack.
DEFAULT_RESULTS = f"hf://{ORG}/qvla-v1-results@v1.0"


def _cache(arg: str | None) -> pathlib.Path:
    if arg:
        return pathlib.Path(arg)
    path = (pathlib.Path.cwd() / ".qvla-cache").resolve()
    print(f"cache: {path}")
    return path


MISSING_LOCK = ("the release checksum manifest {lock} is not in this checkout, so the published results cannot "
                "be fetched verified from the Hub here: fetch them with `qvla data get results` and pass that copy "
                "as --results <its directory>, or pass --results <a local copy of " + ORG + "/qvla-v1-results>")


def open_results(results: str, *, cache: str | None, lock: str):
    """The one verified opener of a results pack, `paper tables`' and `results contrast`'s: a local
    directory as it is, an hf:// source through the release checksum manifest `lock`, cached under
    `cache` (default ./.qvla-cache). load_lock validates the lock (ValueError -> INTEGRITY, exit 2); a
    bare json.loads let a row without `commit` escape as KeyError, exit 1, which means "differs"."""
    hf = results.startswith("hf://")
    if hf and not pathlib.Path(lock).is_file():
        # Named, not a raw `[Errno 2]` from load_lock: what to run instead is the message.
        raise FetchError(MISSING_LOCK.format(lock=lock))
    return open_source(results, cache=_cache(cache) if hf else None,
                       lock=load_lock(pathlib.Path(lock)) if hf else None,
                       token=token_from_env() if hf else None, endpoint=endpoint_from_env() if hf else None)


def _cmd_tables(args: argparse.Namespace) -> int:
    if not (REPO / "experiments" / "paper").is_dir():           # a non-editable install: exit 3, and say why
        print(f"FETCH  {REPO / 'experiments' / 'paper'} does not exist. {CHECKOUT}.")
        return 3
    try:
        cells_ref = pathlib.Path(args.cells).read_bytes()
        bench = load_benchmark(args.benchmark, conf_dir=pathlib.Path(args.bench_conf_dir) if args.bench_conf_dir else None)
        spec = load_spec()
        source = open_results(args.results, cache=args.cache, lock=args.lock)
        result = rebuild(source, cells_ref=cells_ref, bench=bench, spec=spec)
        if result.problems:
            for p in result.problems:
                print(f"INTEGRITY  {p}")
            return 2
        exit_code = 0
        if args.check:
            # Check BEFORE writing anything -- `--out` must never let a
            # rebuild's own write make the comparison that follows compare against itself.
            diffs = check(result, source, manifest_sha256=FROZEN_EXPECTED.get(sha256_bytes(cells_ref)))
            if diffs:
                print(f"DIFFERS  {diffs[0]}")
                for d in diffs[1:]:
                    print(f"         {d}")
                exit_code = 1
        if args.out:
            write_expected(result.outputs, pathlib.Path(args.out))
            print(f"wrote {len(result.outputs)} outputs + MANIFEST.json to {args.out}")
        if args.check:
            if exit_code == 0:
                print(f"IDENTICAL  {len(result.outputs)} outputs match expected/")
        elif not args.out:
            print(f"rebuilt {len(result.outputs)} outputs; add --check to compare or --out DIR to write them")
        return exit_code
    except ShaMismatch as exc:
        print(f"INTEGRITY  {exc}")
        return 2
    except (FetchError, OSError) as exc:
        print(f"FETCH  {exc}")
        return 3
    except ValueError as exc:
        print(f"INTEGRITY  {exc}")
        return 2
    except Exception as exc:
        # Any exception this CLI does not already have a named handler for
        # must still be a clean exit 2 with a message -- never a bare traceback, whose exit 1
        # from the interpreter is otherwise indistinguishable from this CLI's OWN exit 1
        # ("differs"). BaseException subclasses (SystemExit, KeyboardInterrupt) are untouched.
        print(f"INTEGRITY  unexpected error: {exc}")
        return 2


def _cmd_fig4(args: argparse.Namespace) -> int:
    import json

    from axis_qvla.paper import fig4

    try:
        pe = fig4.load_pe(pathlib.Path(args.corpus))
        data, summary_a = fig4.fig4a(pathlib.Path(args.corpus), pe)
        summary = {"fig4a": summary_a}
        if args.study:
            summary["fig4b"] = fig4.fig4b(pe, json.loads(pathlib.Path(args.study).read_bytes()))
        if args.out:
            out = pathlib.Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "fig4a_sim_pe.csv").write_bytes(data)
            (out / "fig4.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        for r in fig4.REWARDS:
            q = summary_a[r]["quantiles"]
            line = f"{r}: {summary_a[r]['n']} episodes, median P_e {q['0.5']:.4f}"
            if "fig4b" in summary:
                b = summary["fig4b"][r]
                line += f"; rho {b['rho']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] over {b['N']} episodes"
            print(line)
        if args.check:
            problems = fig4.check(summary, json.loads(pathlib.Path(args.expected).read_bytes()))
            for p in problems:
                print(f"DIFFERS  {p}")
            if problems:
                return 1
            print("IDENTICAL  the quality-label distributions"
                  + (" and their rank correlation with human judgments, within tolerance" if args.study else ""))
    except ImportError as exc:
        # Fig. 4(b) needs numpy; Fig. 4(a) alone does not. Without this, the ModuleNotFoundError
        # from `paired_bootstrap`'s `import numpy as np` propagates as an interpreter traceback and
        # exit 1 -- the same code `--check` uses for "the figure moved" (README exit-code contract).
        print(f"FAIL  {exc}; the rank correlation with human judgments needs numpy (pip install -e \".[method]\"), "
              "or pass --study '' to compute the label distributions alone")
        return 2
    except (OSError, KeyError, ValueError) as exc:
        print(f"FAIL  {exc}")
        return 2
    except Exception as exc:
        # As in _cmd_tables: any exception this CLI does not already have a named handler for
        # (a malformed corpus raising IndexError or ZeroDivisionError, say) must still be a clean
        # exit 2 with a message, never a bare traceback whose exit 1 is otherwise indistinguishable
        # from this CLI's own exit 1 ("differs"). BaseException subclasses are untouched.
        print(f"FAIL  unexpected error: {exc}")
        return 2
    return 0


def _unit(args: argparse.Namespace):
    from axis_qvla.paper.reeval import Unit

    kind = next(k for k in ("result", "table", "figure") if getattr(args, k))
    return Unit(kind, getattr(args, kind), args.backbone)


def _cmd_reeval(args: argparse.Namespace) -> int:
    """`paper reeval`: --list, --print, or fetch + evaluate + tabulate here (`_reeval_here`). No
    published value is read: the tables hold the user's runs only."""
    import shlex

    from axis_qvla.paper import reeval
    from axis_qvla.paper.cells import load_cells

    prog = args.paper_prog.rsplit(" ", 1)[0]
    try:
        cells, spec = load_cells(pathlib.Path(args.cells).read_bytes()), load_spec()
        if args.list:
            used = reeval.uses(spec)
            for c in cells.cells:
                print(f"{c.cell}\t{c.backbone}\t{c.run}\t{', '.join(used.get(c.cell, [])) or '-'}")
            return 0
        unit = _unit(args)
        sel = reeval.select(cells, spec, unit)
        steps = reeval.plan(sel, host=args.host)
    except (OSError, ValueError) as exc:
        print(f"FAIL  {exc}")
        return 2
    if args.print_only:
        root, out = _print_roots(args, unit)
        for step in steps:
            print(f"{prog} {shlex.join(step.argv)}")
        print(f"# results files: {root}/out_<run>/results.json, one per result ({len(sel)})")
        print(f"# tables: {out}")
        tail = [*unit.flags(), "--collect", *(["--host", args.host] if args.host else []),
                *(["--out", args.out] if args.out else [])]
        print(f"{prog} paper reeval {shlex.join(tail)}")
        return 0
    return _reeval_here(args, prog, unit, sel, steps, spec)


def _print_roots(args: argparse.Namespace, unit) -> tuple[str, str]:
    """Where the results files and the tables will go, for `--print`: from qvla.toml when there is
    one, else as placeholders, so `--print` needs no profile, no GPU, no network and no data."""
    from axis_qvla.paper.reeval_run import HostError, load_host

    try:
        _, root, runs = load_host(args.host)
    except HostError:
        root, runs = "<r1>", "<runs>"
    return str(root), args.out or f"{runs}/paper-reeval/{unit.slug()}"


def _reeval_here(args: argparse.Namespace, prog: str, unit, sel, steps, spec) -> int:
    """Check the environments before any byte is fetched, run the plan (unless --collect), then write
    the paper-layout outputs from whatever results are there."""
    from axis_qvla.bench.spec import BenchmarkError
    from axis_qvla.paper import reeval_run, reeval_tables

    try:
        host, root, runs = reeval_run.load_host(args.host)
        bench = load_benchmark(args.benchmark,
                               conf_dir=pathlib.Path(args.bench_conf_dir) if args.bench_conf_dir else None)
        out = pathlib.Path(args.out) if args.out else runs / "paper-reeval" / unit.slug()
        if not args.collect:
            python, gap = reeval_run.env_gap(host, [c.backbone for c in sel], prog)
            if gap:
                print(f"FAIL  {gap}")
                return 2
    except (reeval_run.HostError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    if not args.collect:
        rc = reeval_run.run_steps(steps, run_root=root, render_python=python, prog=prog)
        if rc:
            return rc
    try:
        P, missing = reeval_tables.collect(sel, root, bench)
        if not P:
            print(f"FAIL  none of the {len(sel)} results has a results file under {root} yet; missing: "
                  f"{', '.join(missing)}. Run this without --collect to evaluate them.")
            return 2
        outputs = reeval_tables.tables(P, sel, spec)
        write_expected(outputs, out)
    except OSError as exc:
        print(f"FETCH  {exc}")
        return 3
    except ValueError as exc:
        print(f"FAIL  {exc}")
        return 2
    print(reeval_tables.summary(unit, outputs, P))
    print(f"wrote {len(outputs)} files to {out}: {len(P)} of {len(sel)} results re-evaluated"
          + (f"; not yet: {', '.join(missing)}" if missing else ""))
    return 0


class _HiddenChoices(dict):
    """A subcommand map whose hidden aliases resolve (lookup, `in`) but are left out when it is listed:
    argparse lists a subparser action's choices by iterating this map, e.g. in its invalid-choice error."""

    hidden: frozenset[str] = frozenset()

    def __iter__(self):
        return (k for k in dict.__iter__(self) if k not in self.hidden)


def hidden_alias(sub: argparse._SubParsersAction, alias: str, name: str) -> None:
    """`alias` runs what `name` runs and is listed nowhere: not in --help, not in the usage line, not in
    the refusal of a mistyped subcommand, not on the generated CLI page (its generator skips a parser it
    has already seen under another name). What an earlier release called a command keeps working."""
    choices = _HiddenChoices(sub._name_parser_map)
    choices.hidden = frozenset(getattr(sub._name_parser_map, "hidden", ())) | {alias}
    choices[alias] = choices[name]
    sub._name_parser_map = sub.choices = choices
    sub.metavar = "{" + ",".join(a.dest for a in sub._choices_actions) + "}"


def add_paper_parsers(sub: argparse._SubParsersAction) -> None:
    paper = sub.add_parser("paper", help="reproduce the paper's results -- every figure value, the main results "
                                         "and the in-text numbers -- from the published per-seed files")
    ps = paper.add_subparsers(dest="subcommand", required=True)
    t = ps.add_parser("tables", help="rebuild every figure value, the main results and every in-text number; "
                                     "--check compares with the pack's expected/")
    t.add_argument("--results", default=DEFAULT_RESULTS,
                   help=f"a local results pack directory or hf://<org>/<repo>@<rev> (default {DEFAULT_RESULTS})")
    t.add_argument("--out", default=None,
                   help="write the outputs and MANIFEST.json here, after --check's comparison")
    t.add_argument("--check", action="store_true", help="compare with <results>/expected/; exit 1 on any difference")
    t.add_argument("--cache", default=None, help="download directory for hf:// (default ./.qvla-cache)")
    t.add_argument("--results-map", dest="cells", metavar="FILE", default=str(DEFAULT_CELLS),
                   help="the committed results map (default experiments/paper/cells.json)")
    t.add_argument("--checksums", dest="lock", metavar="FILE", default=str(default_lock()),
                   help="the release checksum manifest an hf:// source resolves through (default "
                        "conf/release/v1.0.lock.json)")
    t.add_argument("--benchmark", default="franka_heldout24")
    t.add_argument("--bench-conf-dir", default=None, help=argparse.SUPPRESS)
    t.set_defaults(func=_cmd_tables)

    f = ps.add_parser("label-study", help="the quality-label distributions and their rank correlation with human "
                                          "judgments, from published files")
    f.add_argument("--pretraining-data", dest="corpus", metavar="DIR", required=True,
                   help="a local copy of the pretraining data, qvla-v1-corpus (index/ and signals/v1/ suffice)")
    f.add_argument("--study", default=str(REPO / "experiments" / "paper" / "fig4b_human_study.json"),
                   help="the human-study inputs (per-episode H); pass '' to skip the rank correlation")
    f.add_argument("--expected", default=str(REPO / "experiments" / "paper" / "fig4.expected.json"))
    f.add_argument("--out", default=None, help="write fig4a_sim_pe.csv and fig4.json here")
    f.add_argument("--check", action="store_true", help="compare with --expected; exit 1 on a difference")
    f.set_defaults(func=_cmd_fig4)

    rv = ps.add_parser("reeval", help="re-evaluate released checkpoints -- one result, the main results or a "
                                      "figure -- and write the paper-layout tables from your own runs")
    unit = rv.add_mutually_exclusive_group(required=True)
    unit.add_argument("--result", metavar="ID", help="one published result: its name in the results map "
                                                     "(e.g. 'bc|bare') or its run; --list lists them")
    unit.add_argument("--table", metavar="NAME", help=f"a table of the paper: {', '.join(TABLES)}")
    unit.add_argument("--figure", metavar="NAME", help=f"a figure of the paper: {', '.join(FIGURE_UNITS)}")
    unit.add_argument("--list", action="store_true",
                      help="list the published results with their backbone, run and the tables and figures that read them")
    rv.add_argument("--backbone", choices=BACKBONES, default=None,
                    help="only that backbone's results of the table or figure")
    rv.add_argument("--out", default=None, help="where the tables go (default <runs>/paper-reeval/<unit>)")
    rv.add_argument("--host", default=None, help="a qvla.toml path or a host profile name (default ./qvla.toml)")
    rv.add_argument("--print", dest="print_only", action="store_true",
                    help="print the qvla commands in order instead of running them (for a batch script); "
                         "the last one builds the tables")
    rv.add_argument("--collect", action="store_true",
                    help="evaluate nothing: build the tables from the results already evaluated")
    rv.add_argument("--results-map", dest="cells", metavar="FILE", default=str(DEFAULT_CELLS), help=argparse.SUPPRESS)
    rv.add_argument("--benchmark", default="franka_heldout24", help=argparse.SUPPRESS)
    rv.add_argument("--bench-conf-dir", default=None, help=argparse.SUPPRESS)
    rv.set_defaults(func=_cmd_reeval, paper_prog=paper.prog)
    hidden_alias(ps, "fig4", "label-study")
