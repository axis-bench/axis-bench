"""The one entrypoint for served-policy rollout evaluation.

Run one task's seeds against a served policy:

    MUJOCO_GL=egl .venv-eval/bin/python -m axis_qvla.bench.sim.cli \
        --task-id 1889 --variant vanilla --host 127.0.0.1 --port 8000 \
        --checkpoint-dir <ckpt> --n-seeds 50 --seeds 1889000,1889007 --dual-sim \
        --log-root <dir> --work-dir <dir> --task-data-root <qvla-v1-bench tasks/>

WHERE PROMPT, SEEDS AND GRIPPER COME FROM. From the tracked benchmark declaration
(`axis_qvla.bench.spec`, `--benchmark franka_heldout24`): its per-task prompt, its declared test
seeds (or, under `--seed-set dev`, its development seeds), its gripper widths, and its layouts
file as the default `--layout-manifest`. The benchmark's layouts file keys "tasks" by id;
`axis_qvla.bench.sim.layout_plan.adapt_layout_manifest` reshapes it before planning (the same
adaptation the campaign driver applies), so the invocation above needs no extra flag for it.

WHAT THE SERVER OWNS. Guidance is a server-side sampling knob (`serve_policy.py
--cfg-conditioning --guidance-scale W`); the websocket carries observations, not sampling
kwargs. The client therefore READS `cfg_conditioning`/`guidance_scale` off the handshake and
names the run directory after them (`<variant>_w<W>_dualsim` when conditioned). Passing
`--cfg-conditioning`/`--guidance-scale` is an ASSERTION: the run is refused if the server
disagrees.

DOMAIN DEFAULT IS `--dual-sim`, the benchmark's evaluation domain. The render-only domain
(`--no-dual-sim`) steps no object physics, so it is not a supported evaluation domain for tasks
whose success condition references an object; selecting it prints a note.

Exit codes: 0 success, 3 UNEVALUABLE (see eval_layout_plan). 3 is distinct because a task with
no fair layout is not a task the policy failed -- collapsing them into one nonzero code would
put "we could not ask the question" and "the answer was no" in the same bucket. 4 ASSET REFUSED
(axis_qvla.bench.sim.assets.EXIT_ASSET_REFUSED): a scene asset was missing under --asset-dir or did not
match ASSETS.lock.json. The run stops at that seed, which gets no seed file; the queue worker and
the campaign driver abort the whole run on it. A named --asset-dir that is not a directory, or an
--asset-checksums that does not load, exits 4 the same way before the server is contacted. (A missing
flag -- refuse with no --asset-dir, http with no --asset-base-url -- is a usage error: exit 1.)
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from axis_qvla.bench import execution
from axis_qvla.bench import spec as bench_spec
from axis_qvla.bench.sim import arms, assets, domain, policy, runner

# At 15 Hz: 1500 steps = 100 s horizon, 45 stable frames = 3 s hold. Each is read from its
# environment variable (axis_qvla.bench.execution declares and checks them). REPLAN is the default the
# dual-sim loop's AXIS_EVAL_REPLAN overrides; meta.json records the value executed
# (domain.executed_replan), and the --diagnostics MP4 is written at fps / the executed replan.
# Seed files carry no replan, so their bytes do not change.
FPS = float(os.environ.get("PI05_FPS", "15"))
REPLAN = int(os.environ.get("PI05_REPLAN", "8"))
MAX_ROLLOUT_STEPS = int(os.environ.get("MAX_ROLLOUT_STEPS", "1500"))
N_STABLE_FRAMES = int(os.environ.get("N_STABLE_FRAMES", "45"))

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint-dir", required=True, help="for provenance in meta.json")
    ap.add_argument("--benchmark", default=bench_spec.DEFAULT_NAME,
                    help="tracked benchmark declaration (conf/benchmarks/axis-heldout-v1/<name>.json) that "
                         "supplies the task's prompt, test seeds, gripper widths and layouts "
                         "file.")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="first N of the task's declared seeds, in declared order "
                         "(default: the benchmark's n_seeds)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated SUBSET of this task's declared seeds of --seed-set (its "
                         "test seeds by default), for a "
                         "work queue that splits one task across workers. Filtered against the "
                         "declared list, so an undeclared seed is refused rather than silently "
                         "evaluated. Seed order stays the declaration's, so a subset run agrees "
                         "seed-for-seed with the full run; meta.json then describes only that "
                         "subset and the driver recomputes the task tally from the seed files.")
    ap.add_argument("--seed-set", choices=bench_spec.SEED_SETS, default="test",
                    help="which declared seeds --n-seeds/--seeds select from: the test seeds (default) "
                         "or the development seeds (tuning and checkpoint selection only); a dev run's "
                         "--variant (or --run-name) must end in _dev")
    ap.add_argument("--run-name", dest="arm_name", default=None, metavar="RUN_NAME",
                    help="override the derived run directory name (the held-out sweeps use "
                         "hx_<task>_<run>)")
    ap.add_argument("--cfg-conditioning", action=argparse.BooleanOptionalAction, default=None,
                    help="ASSERT that the server does (or, --no-cfg-conditioning, does not) "
                         "append the quality tag '\\nQuality: <q>'. Unset: read off the server.")
    ap.add_argument("--guidance-scale", type=float, default=None,
                    help="ASSERT the w the server was launched with. Unset: read off the "
                         "server. Never sent -- the server owns its sampler.")
    ap.add_argument("--eval-layouts", action=argparse.BooleanOptionalAction, default=True,
                    help="generate a fresh, novel layout per seed (render domain)")
    ap.add_argument("--dual-sim", action=argparse.BooleanOptionalAction, default=True,
                    help="DEFAULT. Physics dynamics with photorealistic observations; implies "
                         "--eval-layouts. --no-dual-sim selects the render-only domain, which "
                         "does NOT hold objects -- see the warning this prints.")
    ap.add_argument("--diagnostics", action="store_true",
                    help="per-seed read-only diagnostic sidecar; no effect on scoring")
    ap.add_argument("--layout-manifest", type=Path, default=None,
                    help="default: the benchmark's layouts file")
    ap.add_argument("--render-root", type=Path, default=REPO_ROOT / "third_party" / "axis-render",
                    help="axis-render checkout the render half reads (default third_party/axis-render)")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--log-root", type=Path, required=True, help="root under which <run>/task_<id>/ is written")
    ap.add_argument("--task-data-root", type=Path, required=True,
                    help="root holding task_<id>/ scene directories (the tasks/ directory of qvla-v1-bench)")
    ap.add_argument("--asset-dir", type=Path, default=None,
                    help="local directory holding every scene mesh and texture (qvla-v1-bench "
                         "assets/). Required under assets.fetch 'refuse', the benchmark default.")
    ap.add_argument("--asset-fetch", choices=assets.FETCH_MODES, default=None,
                    help="default: the benchmark's assets.fetch ('refuse')")
    ap.add_argument("--asset-base-url", default=None,
                    help="only with --asset-fetch http. There is no default URL.")
    ap.add_argument("--asset-checksums", dest="asset_lock", metavar="FILE", type=Path, default=None,
                    help="ASSETS.lock.json, the asset checksum manifest: verify every scene file's sha256 on "
                         "first open (qvla bench eval passes the benchmark's own)")
    return ap


def warn_if_unsupported_domain(domain: str, report=None) -> None:
    """Note that the render-only domain is not a supported evaluation domain for object tasks.

    `render` steps no object physics, so a success condition on an object's position or rotation
    is not evaluated there. Not an exception: the domain serves tasks whose success condition
    references no object.
    """
    if domain != "render":
        return
    (report or (lambda m: print(m, file=sys.stderr)))(
        "NOTE: --no-dual-sim selects the render-only domain, which is not a supported evaluation "
        "domain for tasks whose success condition references an object. The benchmark evaluates "
        "under --dual-sim (the default)."
    )


def _require_dir(path: Path, flag: str, what: str, hint: str) -> None:
    """Refuse a missing root up front, by flag name; unchecked, that surfaces as one error record
    per seed. Only a missing path is 'missing': any other OSError (EACCES) is reported as
    unreadable."""
    try:
        is_dir = Path(path).stat().st_mode & 0o170000 == 0o040000
    except FileNotFoundError:
        is_dir = False
    except OSError as exc:
        raise SystemExit(f"{what} {path} cannot be read ({type(exc).__name__}); check {flag}") from None
    if not is_dir:
        raise SystemExit(f"{what} {path} is not a directory: pass {flag} <{hint}>")


def _domain_of(args) -> str:
    if args.dual_sim:
        return "dualsim"
    return "render" if args.eval_layouts else "physics"


@dataclasses.dataclass(frozen=True)
class _TaskSource:
    """One task's eval facts plus WHERE they came from, so meta.json can say which."""

    prompt: str
    seeds: tuple[int, ...]
    open_w: float
    closed_w: float
    n_seeds_default: int
    layouts_default: Path
    benchmark: dict              # Benchmark.identity()
    manifest: None               # recorded as task_manifest: null
    asset_fetch_default: str     # the declaration's assets.fetch
    seed_set: str = "test"
    execution_declared: dict | None = None   # axis_qvla.bench.execution.declared(benchmark)


def resolve_task_source(args) -> _TaskSource:
    """The task's prompt, seeds, gripper widths and layouts file, from the benchmark declaration."""
    seed_set = getattr(args, "seed_set", "test")
    try:
        bench = bench_spec.load_benchmark(args.benchmark)
        entry = bench.task(args.task_id)
        seeds = bench.seeds_for(args.task_id, seed_set)
    except bench_spec.BenchmarkError as exc:
        raise SystemExit(f"--benchmark {args.benchmark}: {exc}") from None
    return _TaskSource(
        prompt=entry.prompt, seeds=seeds, open_w=entry.open_w, closed_w=entry.closed_w,
        n_seeds_default=bench.n_seeds, layouts_default=bench.layouts_path,
        benchmark=bench.identity(), manifest=None,
        asset_fetch_default=assets.bench_asset_fetch(bench.declaration_path),
        seed_set=seed_set, execution_declared=execution.declared(bench),
    )


def _parse_seed_subset(text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        return [int(s) for s in text.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"--seeds must be comma-separated integers, got {text!r}") from None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.guidance_scale not in (None, 0.0) and args.cfg_conditioning is not True:
        raise SystemExit(
            "--guidance-scale requires --cfg-conditioning (an untagged uncond branch makes "
            "w a no-op)"
        )
    if args.dual_sim and not args.eval_layouts:
        # Dual-sim applies a generated layout to BOTH scenes; the shipped-seed physics path
        # has no render half. Refuse rather than silently ignore the flag.
        raise SystemExit("--dual-sim requires --eval-layouts (it is a render-domain path)")

    source = resolve_task_source(args)
    # The directory this run writes is named after --run-name or --variant; refused before the
    # server is contacted unless that name says which seed set it holds (axis_qvla.bench.sim.arms).
    try:
        arms.check_seed_set_arm(args.arm_name or args.variant, source.seed_set)
    except arms.ArmError as exc:
        raise SystemExit(f"task {args.task_id}, --seed-set {source.seed_set}: {exc}") from None
    layout_manifest = args.layout_manifest or source.layouts_default
    render_root, work_root, log_root, task_data_root = args.render_root, args.work_dir, args.log_root, args.task_data_root
    dom = _domain_of(args)
    warn_if_unsupported_domain(dom)
    # What this process executes, read now: a spelling the rollout would refuse or silently ignore
    # stops the task here, before the server is contacted, instead of inside the first seed.
    try:
        ran = execution.executed(os.environ, domain=dom)
    except ValueError as exc:
        raise SystemExit(f"task {args.task_id}: {exc}") from None
    try:
        replan = domain.executed_replan(dom, REPLAN)
    except ValueError as exc:
        raise SystemExit(f"task {args.task_id}: {exc}") from None
    asset_fetch = args.asset_fetch or source.asset_fetch_default
    if asset_fetch == "refuse" and args.asset_dir is None:
        raise SystemExit("assets.fetch is 'refuse': pass --asset-dir <directory holding "
                         "qvla-v1-bench assets/>. There is no network fallback.")
    if asset_fetch == "http" and not args.asset_base_url:
        raise SystemExit("--asset-fetch http needs --asset-base-url; there is no default URL")
    # A NAMED asset dir or lock that cannot be used is a refused asset tree, not a usage error: the
    # same exit 4 a refusal mid-run gives, so the queue aborts the arm instead of retrying. Checked
    # here because inside a scene build a bad lock would fail every seed as an error record.
    from axis_qvla.bench.sim.asset_guard import check_asset_inputs

    try:
        check_asset_inputs(args.asset_dir, args.asset_lock)
    except assets.AssetFetchRefused as exc:
        print(f"ABORT: scene assets refused for task {args.task_id}: {exc}", file=sys.stderr)
        return assets.EXIT_ASSET_REFUSED
    asset_policy = {"asset_dir": str(args.asset_dir) if args.asset_dir else None,
                    "asset_fetch": asset_fetch, "asset_base_url": args.asset_base_url,
                    "asset_lock": str(args.asset_lock) if args.asset_lock else None}
    if dom != "physics":
        _require_dir(render_root, "--render-root", "render root",
                     "an axis-render checkout: the third_party/axis-render submodule")
    _require_dir(task_data_root, "--task-data-root", "task data root",
                 "the tasks/ directory of a local axisrobotics/qvla-v1-bench@v1.0 copy")

    n_seeds = args.n_seeds if args.n_seeds is not None else source.n_seeds_default
    try:
        seeds = runner.resolve_seeds(source.seeds, n_seeds, only=_parse_seed_subset(args.seeds),
                                     seed_set=source.seed_set)
    except runner.RunnerError as exc:
        raise SystemExit(f"task {args.task_id}: {exc}") from None

    # The server is contacted BEFORE layouts are planned because the arm directory -- which
    # an unevaluable record is written under -- is named after the server's conditioning.
    # Connecting costs no rollouts; an unevaluable task still evaluates zero seeds.
    client, server_metadata = policy.connect_served_policy(args.host, args.port)
    print(f"connected to {args.host}:{args.port} | server_metadata={server_metadata}")
    cond = policy.resolve_conditioning(
        server_metadata, cfg_conditioning=args.cfg_conditioning,
        guidance_scale=args.guidance_scale,
    )

    arm = args.arm_name or arms.served_arm_name(
        args.variant, cfg_conditioning=cond.cfg_conditioning,
        guidance_scale=cond.guidance_scale, domain=dom,
    )
    log_dir = arms.task_dir(log_root / arm, args.task_id)

    # Resolve EVERY layout before the first rollout: an unevaluable task must cost zero
    # rollouts and must not leave a half-populated directory.
    layouts: dict = {}
    if dom in ("render", "dualsim"):
        from axis_qvla.bench.sim.layout_plan import (
            EXIT_UNEVALUABLE, adapt_layout_manifest, plan_layouts, write_unevaluable,
        )
        from axis_qvla.bench.sim.layouts import UnsupportedTaskError

        # The benchmark's layouts file keys "tasks" by id; plan_layouts reads a list. A no-op on
        # a list-shaped file (a worker's already adapted --layout-manifest).
        layout_manifest = adapt_layout_manifest(layout_manifest, log_dir)
        try:
            layouts = plan_layouts(args.task_id, seeds, manifest=layout_manifest,
                                   task_data_root=task_data_root)
        except UnsupportedTaskError as exc:
            log_dir.mkdir(parents=True, exist_ok=True)
            out = write_unevaluable(
                log_dir, task_id=args.task_id, arm=arm, manifest=layout_manifest,
                exc=exc, seeds=seeds,
            )
            print(f"UNEVALUABLE task {args.task_id} ({arm}): {exc}", file=sys.stderr)
            print(f"recorded at {out}; no seeds were evaluated and no fallback was applied.",
                  file=sys.stderr)
            return EXIT_UNEVALUABLE

    def checker_factory(initial_state):
        from axis_qvla.bench.sim.checker import make_stable_checker

        return make_stable_checker(args.task_id, initial_state, n_stable=N_STABLE_FRAMES,
                                   task_data_root=task_data_root)

    rollout_fn = domain.served_rollout_fn(
        dom,
        client=client,
        task_dir=task_data_root / f"task_{args.task_id}",
        task_id=args.task_id,
        prompt=source.prompt,
        common=dict(
            prompt=source.prompt, open_w=source.open_w, closed_w=source.closed_w,
            checker_factory=checker_factory, fps=FPS, replan=replan,
            max_steps=MAX_ROLLOUT_STEPS,
        ),
        layouts=layouts or None,
        render_root=render_root,
        work_dir=work_root / arm / f"task_{args.task_id}",
        diag_dir=log_dir if args.diagnostics else None,
        assets=asset_policy,
    )

    from axis_qvla.bench.sim.task_data import task_data_record

    run_spec = runner.RunSpec(
        task_id=args.task_id, arm=arm, seeds=seeds, log_dir=log_dir,
        provenance={
            "variant": args.variant,
            "cfg_conditioning": bool(cond.cfg_conditioning),
            "guidance_scale": float(cond.guidance_scale),
            "conditioning_source": cond.source,
            "server_metadata": server_metadata,
            "server": f"{args.host}:{args.port}",
            "checkpoint_dir": args.checkpoint_dir,
            "benchmark": source.benchmark,
            "task_manifest": source.manifest,
            "prompt": source.prompt,
            "open_w": source.open_w, "closed_w": source.closed_w,
            "fps": FPS, "replan": replan, "n_stable_frames": N_STABLE_FRAMES,
            "max_steps": MAX_ROLLOUT_STEPS, "checker": "real:CheckerManager",
            "eval_layouts": dom in ("render", "dualsim"),
            "domain": dom,
            "layout_manifest": str(layout_manifest) if layouts else None,
            # The exact arrangement each seed ran on. Without it a fresh-layout number is not
            # reproducible: the sampler is deterministic only relative to a given
            # eval_layouts.py, and its constraints have already changed once.
            "layouts": {str(s): layouts[s].to_json() for s in sorted(layouts)},
            # Which task files scored this result, by content, and the root as a logical id.
            "task_data": task_data_record(task_data_root, args.task_id),
            "assets": {"fetch": asset_fetch, "asset_dir": asset_policy["asset_dir"],
                       "base_url": args.asset_base_url, "lock": asset_policy["asset_lock"]},
            "seed_set": source.seed_set,
            # What this process executes against what the benchmark declares: the execution
            # variables it was started with and every constant that differs from the declaration.
            "execution_env": execution.set_vars(os.environ),
            "execution_deviations": execution.deviations(ran, source.execution_declared),
        },
    )

    try:
        runner.run_seeds(
            run_spec, rollout_fn,
            extra_fn=lambda s: {
                "guidance_scale": float(cond.guidance_scale),
                "cfg_conditioning": bool(cond.cfg_conditioning),
                "eval_layouts": dom in ("render", "dualsim"),
                "domain": dom,
                "base_variant_index": (
                    int(layouts[s].base_variant_index) if layouts else -1
                ),
            },
        )
    except assets.AssetFetchRefused as exc:
        # A broken eval input, not a policy outcome: no seed file, meta.json stays complete=false.
        print(f"ABORT: scene asset refused for task {args.task_id} ({arm}): {exc}", file=sys.stderr)
        return assets.EXIT_ASSET_REFUSED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
