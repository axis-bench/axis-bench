"""`results build|score|contrast|reeval`: write qvla.results/v1 files and apply the reporting rules
(docs/reference/reporting-rules.md).

`build` turns one evaluation run into a results file (axis_qvla/bench/results/build.py; the format is
docs/reference/results-format.md). `score` prints each file's all-24 success rate with its
binomial SE and a seed-bootstrap interval (`--per-task`: every task's, with its Wilson score
interval); `bench score FILE...` is the same command. `contrast` prints the paired difference at every training replicate, with
that pair's eval SE, then their mean and sample SD, in percentage points; `contrast --relative
--bc <BC files>` prints the paper's Table I statistic, 100 (SR_A - SR_B) / SR_BC per replicate.
A side of `contrast` may name a published result of the paper instead of a file: `paper:<result>`,
its results-map id as `qvla paper reeval --list` prints it (`paper:'bc|bare'`), which stands for the
results files of that result's Table I replicates, read from a results pack (`--results`; by default
the local copy `qvla data get results` writes, else the v1.0 Hub results `qvla paper tables` reads)
and verified against the results map (axis_qvla.bench.results.paper_refs). Next to files of your own it
stands for the replicates at their training seeds only.
`reeval` checks a re-read of a checkpoint against its published read: a reproducibility check,
exit 1 when the success rates differ by more than the pair's own evaluation error, refused (2)
when the two files name different checkpoints unless `--allow-different-checkpoints`.

Exit codes: 0 ok (for `reeval`: consistent); 1 `reeval` only: not consistent; 2 the content is
refused: a file is not valid results, the rules refuse it, a `paper:` reference names no published
result (or none at your seeds), a published file does not match its recorded digest, or a run cannot
be built (a record the evaluation should have left is missing, or the records cannot be tied
together); 3 an OS error: a file named on the command line (a results file, `--out`) cannot be read
or written, a published file cannot be fetched, or a record that exists cannot be read (a permission
denial, say).

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
import pathlib

from axis_qvla.bench.spec import BenchmarkError, load_benchmark
from axis_qvla.bench.results.format import load_results
from axis_qvla.bench.results.bootstrap import N_BOOT, bootstrap, wilson
from axis_qvla.bench.results.score import VIEWS, contrast, score


def _bench(args: argparse.Namespace):
    return load_benchmark(args.benchmark, conf_dir=pathlib.Path(args.bench_conf_dir) if args.bench_conf_dir else None)


def _load(paths: list[str]):
    return [load_results(pathlib.Path(p).read_bytes()) for p in paths]


def _score_lines(r, bench, args: argparse.Namespace) -> list[str]:
    s = score(r, bench, view=args.view, allow_off_protocol=args.allow_off_protocol)
    flags = "".join(["  (appendix view)" if s.appendix else "",
                     "  [development seeds]" if s.seed_set == "dev" else "",
                     "  OFF-PROTOCOL" if s.off_protocol else ""])
    iv = bootstrap({t: (k, n) for t, k, n in s.tallies}, n_boot=args.bootstrap) if args.bootstrap else None
    ci = f"\t95% CI [{iv.pooled[0]:.2f}, {iv.pooled[1]:.2f}] (seed bootstrap, B={iv.n_boot})" if iv else ""
    lines = [f"{s.served}\t{s.view}\t{s.pct:.2f} ±{s.se:.2f}{ci}\ttasks {s.n_tasks}\trollouts {s.rollouts}"
             f"\tmissing {s.missing}{flags}"]
    if args.per_task:
        lines.append("  per task: successes/rollouts, rate %, 95% Wilson score interval")
        for t, k, n in s.tallies:
            lo, hi = wilson(k, n)
            lines.append(f"  task {t}\t{k}/{n}\t{100 * k / n:.1f}  [{lo:.1f}, {hi:.1f}]")
    return lines


def score_files(args: argparse.Namespace) -> int:
    """`results score` and `bench score FILE...`: one line per file (and per task with --per-task).
    Every file is read and scored before anything is printed, so a refused file leaves no partial
    table behind it."""
    if args.bootstrap < 0:
        print("FAIL  --bootstrap takes a number of resamples >= 0 (0: no pooled interval)")
        return 2
    try:
        bench = _bench(args)
        lines = [line for r in _load(args.files) for line in _score_lines(r, bench, args)]
    except OSError as exc:
        print(f"FAIL  cannot read: {exc}")
        return 3
    except (ValueError, KeyError, TypeError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    print("\n".join(lines))
    return 0


# The defaults of the options `results score` and `bench score` share, in one place: `bench score
# --run-root` refuses each of them given explicitly (the arm-summary table uses none of them).
SCORE_DEFAULTS = {"view": "all24", "bootstrap": N_BOOT, "per_task": False, "allow_off_protocol": False}


def add_score_options(p: argparse.ArgumentParser) -> None:
    """The options `results score` and `bench score` share."""
    p.add_argument("--view", choices=VIEWS, default=SCORE_DEFAULTS["view"],
                   help="all24 (default) is the one reported aggregate; the other views are appendix-only")
    p.add_argument("--bootstrap", type=int, default=SCORE_DEFAULTS["bootstrap"], metavar="B",
                   help=f"seed-bootstrap resamples for the pooled 95%% interval (default {N_BOOT}; 0: none)")
    p.add_argument("--per-task", action="store_true", default=SCORE_DEFAULTS["per_task"],
                   help="also print every task's success and its 95%% Wilson score interval")
    p.add_argument("--allow-off-protocol", action="store_true", default=SCORE_DEFAULTS["allow_off_protocol"],
                   help="score results whose execution deviates from the benchmark's; each is labelled OFF-PROTOCOL")


def _cmd_score(args: argparse.Namespace) -> int:
    return score_files(args)


def _local_results(host: str | None) -> str | None:
    """The local copy of the v1.0 results `qvla data get results` writes, when this machine has one
    (a qvla.toml, and a results map in that copy). No qvla.toml is not an error unless --host named one."""
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data.layout import ORG, default_dest
    from axis_qvla.data.userconf import user_paths

    try:
        paths = user_paths(host)
    except HostProfileError:
        if host is not None:
            raise
        return None
    local = default_dest(f"{ORG}/qvla-v1-results", paths)
    return str(local) if (local / "cells.json").is_file() else None


def _with_paper_refs(args: argparse.Namespace, sides: dict[str, list[str]]) -> dict[str, list]:
    """Each side's results: its files, then what its `paper:` references resolve to (paper_refs).
    Next to files of the user's own, a reference contributes the replicates at their keys only."""
    from axis_qvla.bench.results.paper_refs import describe, is_ref, resolve
    from axis_qvla.bench.results.score import _seed_key

    out = {k: _load([x for x in v if not is_ref(x)]) for k, v in sides.items()}
    refs = [x for v in sides.values() for x in v if is_ref(x)]
    if not refs:
        return out
    from axis_qvla.paper.cells import load_cells
    from axis_qvla.core.lock import default_lock
    from axis_qvla.paper.cli import DEFAULT_CELLS, DEFAULT_RESULTS, open_results
    from axis_qvla.paper.spec import load_spec

    from axis_qvla.bench.results.paper_refs import backbone_of

    keys = {_seed_key(r.header.training_seed) for v in out.values() for r in v} - {None}
    cells = load_cells(pathlib.Path(args.cells or DEFAULT_CELLS).read_bytes())
    table1 = load_spec().table1
    known = {c.backbone for c in cells.cells}
    backbones = {backbone_of(r, known) for v in out.values() for r in v} - {None}
    where: list[str] = []

    def opener():
        where.append(args.results or _local_results(args.host) or DEFAULT_RESULTS)
        return open_results(where[0], cache=args.cache, lock=args.lock or str(default_lock()))

    resolved = resolve(refs, opener, cells, table1, keys or None, backbones)
    print(f"published results: {where[0]}")
    for ref in resolved.values():
        print(f"  {describe(ref)}")
    for k, v in sides.items():
        out[k] += [x for a in v if is_ref(a) for x in resolved[a].results]
    return out


def _cmd_contrast(args: argparse.Namespace) -> int:
    from axis_qvla.bench.results.paper_refs import PaperRefError
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.paper.fetch import FetchError, ShaMismatch

    if bool(args.relative) != bool(args.bc):
        print("FAIL  --relative needs --bc (the plain-BC results per training seed), and --bc needs --relative")
        return 2
    try:
        s = _with_paper_refs(args, {"a": args.a, "b": args.b, "bc": args.bc or []})
        c = contrast(s["a"], s["b"], _bench(args), bc=s["bc"] if args.bc else None,
                     allow_off_protocol=args.allow_off_protocol)
    except ShaMismatch:             # paper_refs names a mismatched results file; anything else, without digests
        print("FAIL  a published file is not the one the paper published; re-fetch it with `qvla data get results`")
        return 2
    except (FetchError, OSError) as exc:
        print(f"FAIL  cannot read: {exc}")
        return 3
    except (PaperRefError, HostProfileError, ValueError, KeyError, TypeError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    unit = "% of plain BC" if c.relative else "pt"
    for seed, d, se in zip(c.replicates, c.diffs, c.eval_se):
        print(f"replicate {seed}: {d:+.2f} {unit} (eval SE of this pair {se:.2f})")
    sd = "n/a" if c.sd is None else f"{c.sd:.2f}"
    print(f"{c.a} - {c.b}: mean {c.mean:+.2f} {unit}, sample SD {sd}, pairs {c.n_pairs} [{c.label}]")
    if c.ft_seed_differs:
        print(f"ft seeds differ in replicate(s) {', '.join(str(s) for s in c.ft_seed_differs)}")
    if not c.claim_allowed:
        print("no claim: fewer than 3 paired training seeds")
    return 0


def _cmd_reeval(args: argparse.Namespace) -> int:
    from axis_qvla.bench.results.reeval import reeval

    try:
        pub, new = _load([args.published, args.new])
        g = reeval(pub, new, _bench(args), view=args.view,
                   allow_different_checkpoints=args.allow_different_checkpoints)
    except OSError as exc:
        print(f"FAIL  cannot read: {exc}")
        return 3
    except (ValueError, KeyError, TypeError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    p, n = g.published, g.new
    print(f"published  {p.served}\t{p.view}\t{p.pct:.2f} ±{p.se:.2f}")
    print(f"re-read    {n.served}\t{n.view}\t{n.pct:.2f} ±{n.se:.2f}")
    checkpoint = {True: "the same (a content digest in common)",
                  False: "DIFFERENT checkpoints, compared anyway (--allow-different-checkpoints): the two "
                         "reads are of two checkpoints",
                  None: "not compared (a file names no content digest)"}[g.same_checkpoint]
    print(f"checkpoint: {checkpoint}")
    print(f"rollouts with a different outcome: {g.flipped} of {g.compared} (per-rollout outcomes are "
          f"reported, not compared)")
    z = "n/a" if g.z is None else f"{g.z:+.2f}"
    verdict = "consistent with" if g.agrees else "NOT consistent with"
    print(f"reproducibility check: {g.diff:+.2f} pt against this pair's tolerance {g.tolerance:.2f} pt "
          f"(1.96 x eval SE {g.se:.2f}, z {z}): {verdict} the published read")
    return 0 if g.agrees else 1


def _cmd_build(args: argparse.Namespace) -> int:
    from axis_qvla.bench.results.build import BuildError, build_results
    from axis_qvla.bench.results.format import dumps_results

    try:
        pre = None if args.pretrain_seed == "none" else int(args.pretrain_seed)
        r = build_results(pathlib.Path(args.run_root), args.arm, _bench(args), ft_seed=args.ft_seed,
                          pretrain_seed=pre, checkpoint_ref=args.checkpoint_ref,
                          log_root=pathlib.Path(args.log_root) if args.log_root else None)
        out = pathlib.Path(args.out)
        tmp = out.with_name(out.name + ".partial")
        tmp.write_bytes(dumps_results(r))
        tmp.replace(out)
    except OSError as exc:      # there, but unreadable; or the output cannot be written
        print(f"FAIL  cannot read or write: {exc}")
        return 3
    except (BuildError, ValueError, KeyError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    seed_set = r.header.bench.get("seed_set")
    notes = "".join([f"  {seed_set.upper()} SEEDS" if seed_set != "test" else "",
                     "  OFF-PROTOCOL" if r.header.execution.get("off_protocol") else ""])
    print(f"{r.header.served}: {len(r.rows)} rows, fingerprint {r.header.eval_fingerprint[:16]}{notes} -> {out}")
    return 0


_RESULTS_HELP = {
    "score": "score results files under the reporting rules: all-24 and per task, with seed-bootstrap "
             "intervals (as `bench score <files>`)",
    "contrast": "contrast two methods' results files, one per training seed on each side: "
                "100 (SR_A - SR_B) / SR_BC per replicate with --relative, else percentage points",
}


def add_results_parsers(sub: argparse._SubParsersAction) -> None:
    res = sub.add_parser("results", help="build qvla.results/v1 files from an evaluation, and score, contrast "
                                         "and re-check them under the reporting rules")
    rs = res.add_subparsers(dest="subcommand", required=True)
    for name in ("score", "contrast"):
        p = rs.add_parser(name, help=_RESULTS_HELP[name])
        p.add_argument("--benchmark", default="franka_heldout24")
        p.add_argument("--bench-conf-dir", default=None, help=argparse.SUPPRESS)
        if name == "score":
            p.add_argument("files", nargs="+")
            add_score_options(p)
            p.set_defaults(func=_cmd_score)
        else:
            ref = (", or paper:<id>: a published result by its `qvla paper reeval --list` id (paper:'bc|bare'), "
                   "standing for that result's replicate group in the main results (its files at every training seed; next "
                   "to your own files, those at your files' seeds)")
            p.add_argument("--a", nargs="+", required=True, help="side A: one results file per training seed" + ref)
            p.add_argument("--b", nargs="+", required=True, help="side B: one results file per training seed" + ref)
            p.add_argument("--bc", nargs="+", default=None,
                           help="with --relative: the plain-BC results file per training seed (the normaliser)" + ref)
            p.add_argument("--relative", action="store_true",
                           help="report 100 (SR_A - SR_B) / SR_BC per replicate (the paper's main results) instead of pt")
            p.add_argument("--allow-off-protocol", action="store_true",
                           help="pair off-protocol results too (all executed alike), labelled OFF-PROTOCOL")
            p.add_argument("--results", default=None,
                           help="where paper:<result> is read: a results pack directory or hf://<org>/<repo>@<rev> "
                                "(default: the local copy `qvla data get results` writes, else the v1.0 Hub results "
                                "`qvla paper tables` reads)")
            p.add_argument("--cache", default=None, help="download directory for hf:// (default ./.qvla-cache)")
            p.add_argument("--host", default=None,
                           help="this machine's qvla.toml, for the local copy of the results (default ./qvla.toml)")
            p.add_argument("--results-map", dest="cells", metavar="FILE", default=None, help=argparse.SUPPRESS)
            p.add_argument("--checksums", dest="lock", metavar="FILE", default=None, help=argparse.SUPPRESS)
            p.set_defaults(func=_cmd_contrast)
    rv = rs.add_parser("reeval", help="is a re-read of a checkpoint consistent with its published read? "
                                      "(a reproducibility check; exit 1 if not)")
    rv.add_argument("--published", required=True, help="the published results file of the checkpoint")
    rv.add_argument("--new", required=True, help="your read of the same checkpoint on the same seeds")
    rv.add_argument("--view", choices=VIEWS, default="all24",
                    help="all24 (default) is the one reported aggregate; the other views are appendix-only")
    rv.add_argument("--allow-different-checkpoints", action="store_true",
                    help="compare two files that name different checkpoint digests anyway (printed); without "
                         "it they are refused: the check applies only to one checkpoint read twice")
    rv.add_argument("--benchmark", default="franka_heldout24")
    rv.add_argument("--bench-conf-dir", default=None, help=argparse.SUPPRESS)
    rv.set_defaults(func=_cmd_reeval)
    b = rs.add_parser("build", help="turn one `qvla bench eval` run into a qvla.results/v1 file")
    b.add_argument("--run-root", required=True, help="the campaign root holding the run's arm_summary.json")
    b.add_argument("--run", dest="arm", required=True, metavar="RUN",
                   help="the run as bench eval filed it (a development read ends in _dev)")
    b.add_argument("--ft-seed", type=int, required=True, help="the fine-tune training seed")
    b.add_argument("--pretrain-seed", required=True, help="the pretraining seed, or 'none' for no pretraining")
    b.add_argument("--checkpoint-ref", default=None,
                   help="how readers can obtain the checkpoint (default: the path the run recorded)")
    b.add_argument("--log-root", default=None, help="where the seed files are now, if moved since the run")
    b.add_argument("--out", required=True)
    b.add_argument("--benchmark", default="franka_heldout24")
    b.add_argument("--bench-conf-dir", default=None, help=argparse.SUPPRESS)
    b.set_defaults(func=_cmd_build)
