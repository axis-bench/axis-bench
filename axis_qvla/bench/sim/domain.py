"""Bind a sim domain to a `rollout_fn(seed) -> SeedResult`, importing its backend lazily.

Three domains, and they are NOT three views of one number:

    physics  bare-physics rollout on the task's SHIPPED scene variant for that seed. Cheapest;
             a shipped variant can also appear in that task's TRAINING set, so it is a
             development domain, not the benchmark's.
    render   fresh generated layout per seed in the render domain -- the visual domain pi0.5
             was fine-tuned on. Same object set, resampled XY, novelty and difficulty parity
             enforced against every training layout, so success measures competence not recall.
    dualsim  physics dynamics stepped while observations are rendered photorealistically. The
             benchmark's evaluation domain.

WHY THE IMPORTS ARE INSIDE THE FUNCTIONS. The physics backend needs the offline environment and
the render backends need the eval environment; neither can import the other's stack. A
module-level import would make this file unloadable in one of the two environments that must
both load it. The venv-isolation ratchet pins that.

Offline tier at module scope: stdlib plus first-party modules under the import root.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from axis_qvla.bench.sim.arms import DOMAINS
from axis_qvla.bench.sim.record import SeedResult


class DomainError(ValueError):
    """A domain was requested that cannot be built as described."""


def _as_seed_result(seed: int, res: object) -> SeedResult:
    """Adapt a backend's result object to the recorded schema.

    Every backend returns something with these four attributes, but they are different
    classes from different packages. Converting here means the schema is enforced once,
    at the boundary, rather than trusted at four call sites.
    """
    try:
        return SeedResult(
            seed=int(seed),
            success=bool(res.success),          # type: ignore[attr-defined]
            n_steps=int(res.n_steps),           # type: ignore[attr-defined]
            sim_time=float(res.sim_time),       # type: ignore[attr-defined]
            reason=str(res.reason),             # type: ignore[attr-defined]
        )
    except AttributeError as exc:
        raise DomainError(
            f"rollout backend returned {type(res).__name__}, which is missing "
            f"{exc}. A result that cannot be recorded must not be scored."
        ) from None


def diagnostic_path(diag_dir: Path | None, domain: str, seed: int) -> Path | None:
    """Where one seed's diagnostic artifact goes -- and it is not the same KIND per domain.

    The render and dual-sim backends take a `trace_path` and write a JSON sidecar of actuator types,
    command-tracking gap and checker deltas. The physics backend takes a `video_path` and hands it
    straight to `imageio.mimwrite`, which infers the container from the suffix. A `.json` name would
    have imageio write an MP4 into it. The extension is load-bearing, so it is chosen where the
    backend is bound rather than by the caller, who would have to know which domain wants which.
    """
    if diag_dir is None:
        return None
    if domain not in DOMAINS:
        raise DomainError(f"domain must be one of {DOMAINS}, got {domain!r}")
    suffix = "mp4" if domain == "physics" else "diag.json"
    return diag_dir / f"seed_{int(seed)}.{suffix}"


def executed_replan(domain: str, replan: int) -> int:
    """The replan `domain`'s backend executes when handed `replan` -- what a run record must say.

    Only the dual-sim loop honours AXIS_EVAL_REPLAN (`axis_qvla.bench.sim.replan.effective_replan`); the
    render and physics loops consume `replan` as passed. meta.json recorded the PI05_REPLAN
    default even when the override changed what ran, so a replan=4 run was filed as replan=8.
    A malformed override raises ValueError here, before any rollout, instead of in every seed.
    """
    if domain not in DOMAINS:
        raise DomainError(f"domain must be one of {DOMAINS}, got {domain!r}")
    if domain == "dualsim":
        from axis_qvla.bench.sim.replan import effective_replan

        return effective_replan(int(replan))
    return int(replan)


def served_rollout_fn(
    domain: str,
    *,
    client: object,
    task_dir: Path,
    task_id: int,
    prompt: str,
    common: Mapping[str, object],
    layouts: Mapping[int, object] | None = None,
    render_root: Path | None = None,
    work_dir: Path | None = None,
    diag_dir: Path | None = None,
    assets: Mapping[str, object] | None = None,
) -> Callable[[int], SeedResult]:
    """`rollout_fn` for a served policy (pi0.5 over the websocket client).

    `common` carries the knobs every backend takes identically (prompt, gripper calibration,
    checker factory, fps, replan, max_steps) -- passed through rather than re-listed, so a
    backend gaining a shared knob does not need a change here.

    `diag_dir` is a DIRECTORY, not a path function: what the diagnostic artifact is called
    depends on the domain (see `diagnostic_path`), and a caller that supplied the filename
    would be deciding something only this module knows.

    `assets` is the asset policy `{asset_dir, asset_fetch, asset_base_url, asset_lock}`, forwarded
    unchanged to every backend.
    """
    akw = dict(assets or {})
    if domain not in DOMAINS:
        raise DomainError(f"domain must be one of {DOMAINS}, got {domain!r}")

    if domain in ("render", "dualsim"):
        if not layouts:
            raise DomainError(
                f"domain {domain!r} needs a resolved layout per seed. Layouts are planned "
                f"before the server is contacted so an unevaluable task costs zero rollouts."
            )
        if render_root is None or work_dir is None:
            raise DomainError(f"domain {domain!r} needs render_root and work_dir")

    def _diag(seed: int) -> Path | None:
        return diagnostic_path(diag_dir, domain, seed)

    if domain == "dualsim":
        def run(seed: int) -> SeedResult:
            from axis_qvla.bench.sim.rollout_dualsim import rollout_pi05_dualsim

            return _as_seed_result(seed, rollout_pi05_dualsim(
                task_dir, client, init_state_seed=seed, task_id=task_id,
                task_name=prompt, render_root=render_root, work_dir=work_dir,
                eval_layout=_layout_for(layouts, seed), trace_path=_diag(seed), **common, **akw,
            ))
        return run

    if domain == "render":
        def run(seed: int) -> SeedResult:
            from axis_qvla.bench.sim.rollout_render import rollout_pi05_render

            return _as_seed_result(seed, rollout_pi05_render(
                task_dir, client, init_state_seed=seed, task_id=task_id,
                task_name=prompt, render_root=render_root, work_dir=work_dir,
                eval_layout=_layout_for(layouts, seed), trace_path=_diag(seed), **common, **akw,
            ))
        return run

    def run(seed: int) -> SeedResult:
        from axis_qvla.bench.sim.rollout_physics import rollout_pi05_policy

        return _as_seed_result(seed, rollout_pi05_policy(
            task_dir, client, init_state_seed=seed, video_path=_diag(seed), **common, **akw,
        ))
    return run


def local_rollout_fn(
    *,
    policy_factory: Callable[[], Callable],
    task_dir: Path,
    object_body: str,
    goal_xyz: object,
    norm_stats: object,
    checker_factory: Callable,
    max_steps: int,
) -> Callable[[int], SeedResult]:
    """`rollout_fn` for a local torch checkpoint in the physics domain.

    `policy_factory()` is called once PER SEED, not once per sweep: the returned `policy_fn`
    owns an action-chunk buffer, and reusing one across seeds leaks the tail of the previous
    rollout's chunk into the first steps of the next.
    """
    def run(seed: int) -> SeedResult:
        from axis_replay_native.rollout import rollout_policy

        return _as_seed_result(seed, rollout_policy(
            task_dir=task_dir,
            policy_fn=policy_factory(),
            init_state_seed=int(seed),
            object_body=object_body,
            goal_xyz=goal_xyz,
            norm_stats=norm_stats,
            checker_factory=checker_factory,
            max_steps=max_steps,
        ))
    return run


def _layout_for(layouts: Mapping[int, object] | None, seed: int) -> object:
    """Fail by name when a seed has no planned layout.

    A `KeyError: 17` from inside a rollout is indistinguishable from a bug in the backend;
    the planner not covering a seed is a different problem with a different fix.
    """
    if layouts is None or seed not in layouts:
        raise DomainError(
            f"no planned layout for seed {seed}. Every seed must be resolved before the "
            f"sweep starts -- see eval_layout_plan.plan_layouts."
        )
    return layouts[seed]
