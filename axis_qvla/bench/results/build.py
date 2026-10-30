"""From one evaluation on disk to a `qvla.results/v1` file: the public path from `qvla bench eval`
to the scorer.

Why: anyone can score an evaluation of their own under the reporting rules. `build_results` reads
exactly what an evaluation leaves -- the arm's arm_summary.json under the campaign root (it embeds
the run's record as `provenance`), and each task's meta.json and seed files under the log root --
and refuses what it cannot tie together: an arm with errored rollouts or incomplete tasks, seed
files that do not reproduce the summary's fingerprint or are not exactly the declared seeds, server
metadata or execution constants that differ between tasks, another benchmark, fewer tasks or seeds
than the benchmark declares, records that disagree on the benchmark or the seed set. What the header
states is read from those records, never from a name: the served directory is the one the summary
names, the benchmark the one the run record names (the summary's for a record that predates the
field), the seed set the one every record states, serving and execution what the rollouts recorded
(axis_qvla.bench.results.recorded). Training seeds are not on disk, so the caller states them.

A file that exists but cannot be read raises OSError; a record that is missing or cannot be tied
together raises BuildError.

Core tier: stdlib only.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from axis_qvla.bench.results.fingerprint import FingerprintError, units_sha256
from axis_qvla.bench.results.format import Results, ResultsHeader, results_from_run
from axis_qvla.bench.results.recorded import (BuildError, executed_constants,  # noqa: F401 (re-exported)
                                         serve_condition)
from axis_qvla.bench.results.seedfiles import SEED_FILE, SeedFileError, read_served_dir
from axis_qvla.bench.sim.arms import (FAILED_SUMMARY_FILE, SUMMARY_FILE, ArmError, check_seed_set_arm,
                                 out_dir_name, task_dir)
from axis_qvla.bench.spec import BenchmarkError

DRIVER = "axis_qvla/bench/sim/campaign.py"    # the run record's `sources.files` key for the driver


def _json(p: Path) -> dict:
    try:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BuildError(f"no {Path(p).name} at {p}") from None
    except ValueError as exc:
        raise BuildError(f"{p} is not valid JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise BuildError(f"{p} is not a JSON object")
    return doc


def harness(manifest: Mapping | None) -> dict | None:
    if manifest is None:
        return None
    src = manifest.get("sources") or {}
    repo = src.get("repo") or {}
    return {"driver_sha256": (src.get("files") or {}).get(DRIVER), "repo_head": repo.get("head"),
            "repo_dirty": repo.get("dirty")}


def _records(run_root: Path, arm: str) -> tuple[dict, dict | None]:
    """(arm_summary.json, the run record it embeds) of `arm` under `run_root`."""
    out = Path(run_root) / out_dir_name(arm)
    if not (out / SUMMARY_FILE).exists() and (out / FAILED_SUMMARY_FILE).exists():
        raise BuildError(f"{arm}: the run finished with no rollouts ({out / FAILED_SUMMARY_FILE})")
    summary = _json(out / SUMMARY_FILE)
    if summary.get("arm") != arm:
        raise BuildError(f"{out / SUMMARY_FILE} is the summary of run {summary.get('arm')!r}, not {arm!r}")
    manifest = summary.get("provenance")
    if manifest is not None and (not isinstance(manifest, dict) or "error" in manifest):
        raise BuildError(f"{arm}: the run record cannot be read: {manifest!r}"[:300])
    return summary, manifest


def _decl(ident: Mapping) -> str | None:
    # `manifest_sha256` is the declaration digest's name in summaries that predate the package.
    return ident.get("sha256") or ident.get("manifest_sha256")


def _identity(summary: Mapping, manifest: Mapping | None, bench, arm: str) -> tuple[Mapping, str]:
    """(the benchmark identity the run evaluated, where it was read). The run record's is written
    when the run starts; the summary's is written when it is tallied, so under --tally-only it is
    the tally's. Each must be `bench` (a declaration it accepts, its version), and when the run
    record has one, the two must agree, and the record's is the header's. A summary that names no
    benchmark is refused; for a published run, its published results file is the record."""
    if not isinstance(summary.get("benchmark"), Mapping) or _decl(summary["benchmark"]) is None:
        raise BuildError(f"{arm}: the summary names no benchmark, so "
                         f"nothing ties its rollouts to {bench.name}; it cannot be built")
    named = [("the summary", summary["benchmark"])]
    if (manifest or {}).get("benchmark") is not None:
        named.append(("the run record", manifest["benchmark"]))
    for where, ident in named:
        if not isinstance(ident, Mapping) or _decl(ident) not in bench.accepted_declarations() \
                or ident.get("version_sha256") not in (None, bench.version_sha256):
            ident = ident if isinstance(ident, Mapping) else {}
            raise BuildError(f"{arm} was evaluated on benchmark {ident.get('name')!r} (declaration "
                             f"{str(_decl(ident))[:12]}, version {str(ident.get('version_sha256'))[:12]}) by "
                             f"{where}, not {bench.name} ({bench.sha256[:12]}, version {bench.version_sha256[:12]})")
        if ident.get("matches_manifest") is False:
            raise BuildError(f"{arm}: {where} says the run evaluated {len(ident.get('task_ids') or ())} tasks, not "
                             f"the {len(bench.task_ids)} of {bench.name}: not a benchmark result")
    if len(named) == 1:
        return named[0][1], "summary"
    (_, s), (_, r) = named
    for key, get in (("declaration", _decl), ("version_sha256", lambda i: i.get("version_sha256")),
                     ("n_seeds", lambda i: i.get("n_seeds"))):
        if get(s) is not None and get(r) is not None and get(s) != get(r):
            raise BuildError(f"{arm}: the summary and the run record name different benchmarks: {key} "
                             f"{str(get(s))[:16]} (summary) vs {str(get(r))[:16]} (run record)")
    return r, "run_record"


def _seed_set(summary: Mapping, manifest: Mapping | None, metas: Sequence[Mapping], arm: str) -> str:
    """The seed set every record states (a record that predates the field states none)."""
    seed_set = summary.get("seed_set") or "test"
    stated = [("the run record", (manifest or {}).get("seed_set"))] + [
        (f"task {m.get('task_id')}'s meta.json", m.get("seed_set")) for m in metas]
    for where, s in stated:
        if s is not None and s != seed_set:
            raise BuildError(f"{arm}: the summary says seed set {seed_set!r}, {where} says {s!r}")
    try:
        check_seed_set_arm(arm, seed_set)
    except ArmError as exc:
        raise BuildError(f"{arm}, seed set {seed_set}: {exc}") from None
    return seed_set


def _check_seeds(run, bench, seed_set: str) -> int:
    """Every task holds exactly the first n_seeds declared seeds of the set (a missing seed would
    score as a failure, a stray one is not the benchmark's); returns the set's seed offset."""
    try:
        want = {t: set(bench.seeds_for(t, seed_set)[:bench.n_seeds]) for t in bench.task_ids}
        offset = 0 if seed_set == "test" else int(bench.dev["offset"])
    except BenchmarkError as exc:
        raise BuildError(str(exc)) from None
    wrong = []
    for t in bench.task_ids:
        have = {int(SEED_FILE.fullmatch(f.name).group(1)) for f in run.tasks[t]}
        missing, stray = sorted(want[t] - have), sorted(have - want[t])
        if missing or stray:
            wrong.append(f"task {t}: " + "; ".join(
                f"{len(x)} {what} {x[:5]}{'...' if len(x) > 5 else ''}"
                for what, x in (("missing", missing), (f"not {seed_set} seeds", stray)) if x))
    if wrong:
        raise BuildError(f"{run.served}: the seed files are not the benchmark's {bench.n_seeds} {seed_set} "
                         f"seeds per task: " + " | ".join(wrong))
    return offset


def _convention(key: str, var: str, dev: Mapping, manifest: Mapping | None, bench) -> str:
    """What executed: a recorded deviation's value, else the run record's verdict, else the variable
    the workers started with; with none of them (and no deviation recorded), the declared value."""
    if key in dev:
        return dev[key]["executed"]
    ran = ((manifest or {}).get("execution") or {}).get("executed") or {}
    return ran.get(key) or ((manifest or {}).get("env_pins") or {}).get(var) or getattr(bench, key)


def build_results(run_root: Path, arm: str, bench, *, ft_seed: int | None, pretrain_seed: int | None,
                  checkpoint_ref: str | None = None, log_root: Path | None = None) -> Results:
    def is_int(x: object) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if ft_seed is None and pretrain_seed is not None:
        raise BuildError(f"a pretrain seed ({pretrain_seed}) without a fine-tune seed: give both or neither")
    if not (ft_seed is None or is_int(ft_seed)) or not (pretrain_seed is None or is_int(pretrain_seed)):
        raise BuildError(f"training seeds are ints (pretrain may be None): ft {ft_seed!r}, pretrain {pretrain_seed!r}")
    summary, manifest = _records(run_root, arm)
    if summary.get("errored_rollouts"):
        raise BuildError(f"{arm}: {summary['errored_rollouts']} rollouts ended with an `error:` reason, a "
                         f"broken harness rather than a result")
    if summary.get("incomplete_tasks"):
        raise BuildError(f"{arm}: the run left tasks incomplete (task, seed files): {summary['incomplete_tasks']}")
    ident, ident_source = _identity(summary, manifest, bench, arm)
    if summary.get("n_seeds") not in (None, bench.n_seeds):
        raise BuildError(f"{arm} evaluated {summary['n_seeds']} seeds per task, not the {bench.n_seeds} of "
                         f"{bench.name}: not a benchmark result")
    served, root = summary.get("served_arm"), log_root if log_root is not None else summary.get("log_root")
    if not served or root is None or not summary.get("eval_fingerprint"):
        raise BuildError(f"{arm}: the summary names no served_arm, log_root or eval_fingerprint")
    root = Path(root)
    try:
        run = read_served_dir(root, served)
    except SeedFileError as exc:
        if isinstance(exc.__cause__, OSError) and not isinstance(exc.__cause__, FileNotFoundError):
            raise OSError(str(exc)) from exc.__cause__     # there, but unreadable: not a build refusal
        raise BuildError(str(exc)) from None
    missing = [t for t in bench.task_ids if t not in run.tasks]
    if missing:
        raise BuildError(f"{served}: no task directory under {root} for tasks {missing}")
    metas = [_json(task_dir(root / served, t) / "meta.json") for t in bench.task_ids]
    seed_set = _seed_set(summary, manifest, metas, arm)
    offset = _check_seeds(run, bench, seed_set)
    executed = executed_constants(metas, manifest, bench)
    dev = executed.get("deviations") or {}
    header = ResultsHeader(
        bench={"name": bench.name, "version": ident.get("version"), "version_sha256": ident.get("version_sha256"),
               "declaration_sha256": _decl(ident), "identity_source": ident_source, "seed_set": seed_set,
               "units_sha256": units_sha256(bench.task_ids, bench.n_seeds, offset=offset)},
        arm=arm, served=served,
        # None: a policy whose training seeds nobody stated (a server of your own); `contrast`
        # refuses it, `score` does not need it.
        training_seed=None if ft_seed is None else {"pretrain": pretrain_seed, "ft": ft_seed},
        checkpoint={"ref": checkpoint_ref or summary.get("checkpoint"), "pub_digest": None,
                    "eval_digest": ((manifest or {}).get("checkpoint") or {}).get("sha256")},
        serve=serve_condition(metas), execution=executed,
        image_geometry=_convention("image_geometry", "AXIS_EVAL_IMAGE_GEOMETRY", dev, manifest, bench),
        action_space=_convention("action_space", "AXIS_EVAL_ACTION_SPACE", dev, manifest, bench),
        harness=harness(manifest), eval_lock_sha256=None, gpu=(manifest or {}).get("gpu"),
        xla_pin=summary.get("xla_flags") or bench.xla_pin, noise_salt=(manifest or {}).get("noise_salt"),
        eval_fingerprint=summary["eval_fingerprint"])
    try:
        return results_from_run(run, header, task_ids=bench.task_ids)
    except (ValueError, FingerprintError) as exc:
        raise BuildError(str(exc)) from None
