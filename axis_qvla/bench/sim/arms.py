"""Where one eval arm's per-seed results are written, and why the name is load-bearing.

An "arm" is one (policy, sampling setting, sim domain) combination. Its directory name is the
only thing separating result sets that must never be averaged together: a fresh-layout number
and a shipped-variant number answer different questions, and four guidance scales share ONE
checkpoint so they would otherwise collide in a single directory.

The served-policy harness (`axis_qvla.bench.sim.cli`) names arms `<variant>[_w<W>][_fresh|_dualsim]`
under one root, reproduced here EXACTLY, spelling included. These names are not cosmetic --
published results already sit under them and the result tooling globs for them, so "tidying"
the scheme silently orphans prior results rather than raising. New behaviour gets a new suffix;
existing spellings are frozen.

Offline tier: stdlib only, so arm names can be computed and tested without a GPU, a simulator,
or either heavyweight venv.
"""
from __future__ import annotations

from pathlib import Path

# The sim domain an arm ran in. These are not interchangeable views of one number:
#   physics -- bare-physics rollout on the task's shipped scene variant for that seed
#   render  -- fresh generated layout, render domain (the visual domain pi0.5 trained on)
#   dualsim -- physics dynamics stepped, observations rendered photorealistically
DUAL_SIM = "dualsim"
DOMAINS = ("physics", "render", DUAL_SIM)
# The arm-name suffix of a development-seed read (axis_qvla.bench.spec SEED_SETS: "test", "dev").
DEV_SUFFIX = "_dev"
# The prefix of an arm's run directory under a campaign root (`out_dir_name`), and the summary the
# driver writes there when the queue drains: SUMMARY_FILE, or FAILED_SUMMARY_FILE for 0 rollouts.
OUT_PREFIX = "out_"
SUMMARY_FILE, FAILED_SUMMARY_FILE = "arm_summary.json", "arm_summary.FAILED.json"


class ArmError(ValueError):
    """An arm was described in a way that cannot name a directory unambiguously."""


def served_arm_name(
    variant: str,
    *,
    cfg_conditioning: bool = False,
    guidance_scale: float = 0.0,
    domain: str = "render",
) -> str:
    """Arm directory name for the served-policy (pi0.5) harness.

    The names match the published runs' served directories; `test_arms.py` pins them.

    Guidance is inference-only, so every scale reuses one checkpoint and would otherwise land in the
    same `<variant>/` directory. Un-guided runs keep the bare variant name; tagged runs always carry
    the scale -- including `w0`, which is tagged-but-unguided and is NOT the same policy as
    untagged.

    `_fresh` and `_dualsim` are suffixes, so shipped-seed results keep their own directories and
    can never be averaged with fresh-layout ones by a tool that globs for `<variant>/`.
    """
    if domain not in DOMAINS:
        raise ArmError(f"domain must be one of {DOMAINS}, got {domain!r}")
    if guidance_scale != 0.0 and not cfg_conditioning:
        raise ArmError(
            "guidance_scale requires cfg_conditioning -- with no tagged conditional branch "
            "the guidance weight is a no-op, so the name would claim a setting that did not "
            "take effect"
        )

    arm = f"{variant}_w{guidance_scale:g}" if cfg_conditioning else variant
    if domain == DUAL_SIM:
        return f"{arm}_dualsim"
    return f"{arm}_fresh" if domain == "render" else arm


def seed_set_arm(arm: str, seed_set: str) -> str:
    """The arm a read of `seed_set` is filed under. A development read's name ends in DEV_SUFFIX
    (added when missing) and a test read's never does, so one result directory never holds seed
    files of both sets and a development number is never tallied or scored as a test number. The
    served directory is then named from it as usual (`served_arm_name`)."""
    if seed_set == "dev":
        return arm if arm.endswith(DEV_SUFFIX) else arm + DEV_SUFFIX
    if seed_set != "test":
        raise ArmError(f"seed set must be 'test' or 'dev', got {seed_set!r}")
    if arm.endswith(DEV_SUFFIX):
        raise ArmError(f"run {arm!r} ends in {DEV_SUFFIX!r}, which names development reads; a test-seed "
                       f"read must be filed under another name")
    return arm


def check_seed_set_arm(arm: str, seed_set: str) -> None:
    """Raise ArmError unless `arm` already names a read of `seed_set` (`seed_set_arm` would leave it
    unchanged), in either direction: for a name the caller gave and cannot have rewritten -- the
    driver's --run, the rollout's --variant or --run-name."""
    if seed_set_arm(arm, seed_set) != arm:
        raise ArmError(f"run {arm!r} would hold development seeds, so it must end in {DEV_SUFFIX!r} "
                       f"({arm}{DEV_SUFFIX}): a development read is never filed under a test run's name")


def check_arm_name(arm: str) -> None:
    """Raise ArmError unless `arm` is one path component. Every record of an arm is written under
    a directory named from it (out_<run>/, q_<run>/, work_<run>/), so a `/`, a `\\` or a `..` would
    write outside the campaign root."""
    if not arm or "/" in arm or "\\" in arm or ".." in arm:
        raise ArmError(f"run {arm!r} must be one path component: no '/', '\\' or '..', since its records "
                       f"are written under directories named from it")


def out_dir_name(arm: str) -> str:
    """The directory under a campaign root that holds one arm's records: run_manifest.json, the
    worker summaries and, once the queue drains, arm_summary.json. The driver writes it, and
    `bench score` and `results build` read it, by this one name."""
    check_arm_name(arm)
    return f"{OUT_PREFIX}{arm}"


def task_dir(arm_root: Path, task_id: int) -> Path:
    """Per-task subdirectory of an arm directory."""
    return arm_root / f"task_{int(task_id)}"
