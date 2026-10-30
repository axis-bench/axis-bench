"""The execution constants of an evaluation: declared by the benchmark version, enforced at launch.

Why: fps, replan, the step budget and the stability window are part of what a success rate
means, yet the rollout client reads each from an environment variable with a default, the
dual-sim loop and the controller read six more, and the campaign driver passes its caller's
whole environment to every server and worker. A shell that exported PI05_REPLAN=4 for an
experiment and forgot it would file a replan-4 number under the benchmark's name. Each knob has a
declared value (the version's execution.json, and the declaration's two conventions);
`deviations` compares what a run would execute with it. `qvla bench eval` and the campaign driver
refuse a deviation unless the run says --off-protocol, and every record names what executed:
run_manifest.json carries the `run_record`, and every task's meta.json the variables set and the
deviations its rollouts executed.

Core tier: stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping

from axis_qvla.bench.sim.arms import DOMAINS, DUAL_SIM  # the sim domains, named in one place

# Environment variable -> (execution key, the rollout code's own default when it is unset).
# tests/unit/bench/test_execution_knobs.py reads every environment access of the axis_qvla/bench
# package, in every spelling, so a new knob, or a changed default, fails a test instead of
# changing results.
ENV: dict[str, tuple[str, str | None]] = {
    "PI05_FPS": ("fps", "15"),
    "PI05_REPLAN": ("replan", "8"),
    "AXIS_EVAL_REPLAN": ("replan", None),           # the dual-sim loop's override of PI05_REPLAN
    "MAX_ROLLOUT_STEPS": ("max_steps", "1500"),
    "N_STABLE_FRAMES": ("n_stable_frames", "45"),
    "AXIS_EVAL_GRIP_STATE": ("grip_state", "closedness"),
    "AXIS_EVAL_EEF_IK": ("eef_ik", "single"),
    "AXIS_EVAL_TABLE_TEXTURE": ("table_texture", None),
    "AXIS_EVAL_ACTION_SPACE": ("action_space", "droid8"),
    "AXIS_EVAL_IMAGE_GEOMETRY": ("image_geometry", "profile"),
}
# Read by the package but outcome-neutral: diagnostics, or (QVLA_EVAL_REEXEC) CLI process
# bookkeeping that never reaches a rollout.
DIAGNOSTIC_ENV = ("AXIS_EVAL_PROVENANCE", "AXIS_ROLLOUT_VIDEO_EVERY", "QVLA_EVAL_REEXEC")
NUMERIC = ("fps", "replan", "max_steps", "n_stable_frames")
KEYS = ("fps", "replan", "max_steps", "n_stable_frames", "grip_state", "eef_ik", "table_texture",
        "action_space", "image_geometry")
# The two conventions the declaration itself (not execution.json) states, by the variable that
# carries each to the rollout. Every launch exports them, so an unset variable never silently
# means the code default (a letterboxed frame).
CONVENTION_ENV = {"AXIS_EVAL_IMAGE_GEOMETRY": "image_geometry", "AXIS_EVAL_ACTION_SPACE": "action_space"}
# The domain of every rollout the campaign driver starts (its workers pass --dual-sim).
DRIVER_DOMAIN = DUAL_SIM
# The GL platform every rollout worker renders through, set by every launch and never inherited
# from the caller's shell: mujoco's and robosuite's EGL contexts refuse (ImportError) a
# PYOPENGL_PLATFORM other than egl, so a shell that exported osmesa would error every rollout.
GL_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}

# How each reader turns its variable into what executes, mirrored here so the verdict is the
# rollout's own. Like the readers, a variable set to "" is set (only AXIS_EVAL_REPLAN treats a
# blank value as unset). A value the reader would refuse is refused here by name, and so is a
# spelling it would silently read as its default (AXIS_EVAL_GRIP_STATE=finger runs closedness).
_FLOAT = ("PI05_FPS",)                                           # float(raw)
_INT = ("PI05_REPLAN", "MAX_ROLLOUT_STEPS", "N_STABLE_FRAMES")   # int(raw)
_MODES = {                     # raw.strip().lower(), compared with one mode; empty means the default
    "AXIS_EVAL_GRIP_STATE": ("closedness", "fingers"),
    "AXIS_EVAL_EEF_IK": ("single", "iterate"),
    "AXIS_EVAL_IMAGE_GEOMETRY": ("profile", "raw"),
}
_EXACT = {"AXIS_EVAL_ACTION_SPACE": ("droid8", "joint9", "eef7")}   # compared as given; else it raises
# AXIS_EVAL_TABLE_TEXTURE: any non-empty value is the texture path, used as given.


def _read(var: str, env: Mapping[str, str]) -> object:
    key, default = ENV[var]
    raw = env.get(var, default)

    def refused(why: str) -> ValueError:
        return ValueError(f"{key}: {var}={raw!r} {why}")

    if var in _FLOAT:
        try:
            x = float(raw)
        except ValueError:
            raise refused("is not a number (the rollout reads it with float())") from None
        return int(x) if x.is_integer() else x
    if var in _INT:
        try:
            return int(raw)
        except ValueError:
            raise refused("is not an integer (the rollout reads it with int())") from None
    if var in _MODES:
        mode = raw.strip().lower() or default
        if mode not in _MODES[var]:
            raise refused(f"is not one of {', '.join(_MODES[var])}: the rollout would run {default!r}")
        return mode
    if var in _EXACT:
        if raw not in _EXACT[var]:
            raise refused(f"is not one of {', '.join(_EXACT[var])}, spelled exactly: the rollout refuses it")
        return raw
    return raw or None


def _replan_override(env: Mapping[str, str]) -> int | None:
    """AXIS_EVAL_REPLAN as the dual-sim loop reads it (axis_qvla.bench.sim.replan.effective_replan)."""
    raw = env.get("AXIS_EVAL_REPLAN")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"replan: AXIS_EVAL_REPLAN={raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(f"replan: AXIS_EVAL_REPLAN={value} must be >= 1: the dual-sim loop refuses it")
    return value


def executed(env: Mapping[str, str], *, domain: str = DRIVER_DOMAIN) -> dict[str, object]:
    """What a rollout of `domain` started under `env` executes, key by key. AXIS_EVAL_REPLAN
    overrides PI05_REPLAN in the dual-sim loop only (axis_qvla.bench.sim.domain.executed_replan); every
    other knob is read as its reader reads it. A domain other than dual-sim is not the declared
    protocol of a dual-sim benchmark in the first place; meta.json records the domain beside this."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    out = {key: _read(var, env) for var, (key, _) in ENV.items() if var != "AXIS_EVAL_REPLAN"}
    override = _replan_override(env) if domain == DUAL_SIM else None
    if override is not None:
        out["replan"] = override
    return out


def declared(b) -> dict[str, object] | None:
    """The benchmark's value for every key, or None when its directory has no execution.json."""
    if b is None or not b.execution:
        return None
    out: dict[str, object] = {k: int(b.execution[k]) for k in NUMERIC}
    for key in ("grip_state", "eef_ik"):
        out[key] = str(b.execution[key]).strip().lower()
    out["table_texture"] = b.execution["table_texture"] or None
    out["action_space"] = str(b.action_space)
    out["image_geometry"] = str(b.image_geometry).strip().lower()
    return out


def convention_env(b) -> dict[str, str]:
    """The environment every launch of `b` exports: its two declared conventions."""
    return {var: str(getattr(b, key)) for var, key in CONVENTION_ENV.items()}


def deviations(executed_values: Mapping[str, object],
               declared_values: Mapping[str, object] | None) -> dict[str, dict]:
    """{key: {"executed", "declared"}} for every key the two disagree on; {} with no declaration."""
    if declared_values is None:
        return {}
    return {k: {"executed": executed_values.get(k), "declared": declared_values.get(k)}
            for k in KEYS if executed_values.get(k) != declared_values.get(k)}


def set_vars(env: Mapping[str, str]) -> dict[str, str]:
    """The execution variables `env` sets, for the record."""
    return {k: env[k] for k in ENV if k in env}


def run_record(env: Mapping[str, str], b, *, off_protocol: bool, domain: str = DRIVER_DOMAIN) -> dict:
    """What a run started under `env` executes against what `b` declares, as a run record stores
    it. `off_protocol` is whether the run was allowed to deviate (--off-protocol)."""
    decl = declared(b)
    ran = executed(env, domain=domain)
    return {"declared": decl, "executed": ran, "deviations": deviations(ran, decl),
            "off_protocol": bool(off_protocol), "env": set_vars(env)}


def _source(key: str, env: Mapping[str, str]) -> str:
    named = [f"{v}={env[v]!r}" for v, (k, _) in ENV.items() if k == key and v in env]
    if named:
        return "set by " + ", ".join(named)
    unset = " / ".join(v for v, (k, _) in ENV.items() if k == key)
    return f"{unset} unset: the code default"


def refusal(dev: Mapping[str, dict], env: Mapping[str, str]) -> str:
    lines = [f"{k}: this run would execute {v['executed']!r}, the benchmark declares {v['declared']!r} "
             f"({_source(k, env)})" for k, v in sorted(dev.items())]
    return ("; ".join(lines) + ". Unset the variable, or pass --off-protocol to run anyway: the run is "
            "then recorded as off-protocol, with every deviation, in run_manifest.json and each task's "
            "meta.json.")
