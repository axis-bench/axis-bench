"""Arm names are load-bearing: they are the only thing keeping incomparable results apart.

These tests pin the exact spellings against the legacy implementation. The legacy
formula is transcribed here rather than imported, because importing `rollout_eval` pulls in
mujoco and torch -- the point of `axis_qvla.bench.sim.arms` being stdlib-only is that naming can be
checked without either heavyweight venv.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from axis_qvla.bench.sim.arms import (
    ArmError,
    served_arm_name,
    task_dir,
)


def _legacy_served_arm_name(
    variant: str, *, cfg_conditioning: bool, guidance_scale: float,
    eval_layouts: bool, dual_sim: bool = False,
) -> str:
    """Transcribed verbatim from the archived legacy evaluation harness's arm-naming logic;
    the transcription is what keeps the equivalence tested."""
    arm = f"{variant}_w{guidance_scale:g}" if cfg_conditioning else variant
    if dual_sim:
        return f"{arm}_dualsim"
    return f"{arm}_fresh" if eval_layouts else arm


VARIANTS = ("vanilla", "filt_bin", "filt_top70", "cfg", "bc_rnn")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("domain,eval_layouts,dual_sim", [
    ("physics", False, False),
    ("render", True, False),
    ("dualsim", True, True),
])
def test_served_arm_name_matches_legacy_unguided(variant, domain, eval_layouts, dual_sim):
    assert served_arm_name(variant, domain=domain) == _legacy_served_arm_name(
        variant, cfg_conditioning=False, guidance_scale=0.0,
        eval_layouts=eval_layouts, dual_sim=dual_sim,
    )


@pytest.mark.parametrize("w", [0.0, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("domain,eval_layouts,dual_sim", [
    ("physics", False, False),
    ("render", True, False),
    ("dualsim", True, True),
])
def test_served_arm_name_matches_legacy_guided(w, domain, eval_layouts, dual_sim):
    assert served_arm_name(
        "cfg", cfg_conditioning=True, guidance_scale=w, domain=domain,
    ) == _legacy_served_arm_name(
        "cfg", cfg_conditioning=True, guidance_scale=w,
        eval_layouts=eval_layouts, dual_sim=dual_sim,
    )


def test_tagged_w0_differs_from_untagged():
    """w=0 tagged is NOT the same policy as untagged -- the names must not collide."""
    assert served_arm_name("cfg", cfg_conditioning=True, guidance_scale=0.0) != \
        served_arm_name("cfg", cfg_conditioning=False)


def test_domains_never_share_a_name():
    """A fresh-layout and a shipped-variant number must not land in one directory."""
    names = {served_arm_name("vanilla", domain=d) for d in ("physics", "render", "dualsim")}
    assert len(names) == 3


def test_guidance_without_conditioning_is_refused():
    """Naming a run w=2 when no conditional branch exists would claim a setting that
    never took effect."""
    with pytest.raises(ArmError, match="cfg_conditioning"):
        served_arm_name("cfg", cfg_conditioning=False, guidance_scale=2.0)


def test_unknown_domain_is_refused():
    with pytest.raises(ArmError, match="domain"):
        served_arm_name("vanilla", domain="teleport")


def test_task_dir_shape():
    assert task_dir(Path("/a/b"), 1889) == Path("/a/b/task_1889")


def test_an_arms_run_directory_is_named_once_for_its_writer_and_readers(tmp_path):
    """The driver writes out_<arm>/ and `bench score` and `results build` read it: one name."""
    from axis_qvla.bench.results.summaries import read_summaries
    from axis_qvla.bench.sim.arms import out_dir_name

    assert out_dir_name("probe_dev") == "out_probe_dev"
    (tmp_path / out_dir_name("probe")).mkdir()
    (tmp_path / out_dir_name("probe") / "arm_summary.json").write_text('{"arm": "probe", "rollouts": 1}')
    assert [r.arm for r in read_summaries(tmp_path)] == ["probe"]


@pytest.mark.parametrize("arm", ["x/../../elsewhere", "a/b", "a\\b", "..", "a..b", ""])
def test_an_arm_is_one_path_component(arm):
    """Every record of an arm is written under a directory named from it, so a separator or a
    `..` would put it outside the campaign root."""
    from axis_qvla.bench.sim.arms import ArmError, check_arm_name, out_dir_name

    with pytest.raises(ArmError, match="must be one path component"):
        check_arm_name(arm)
    with pytest.raises(ArmError):
        out_dir_name(arm)


def test_the_driver_refuses_an_arm_that_is_not_one_path_component(tmp_path, capsys):
    from axis_qvla.bench.sim import campaign

    with pytest.raises(SystemExit) as exc:
        campaign.main(["--run", "x/../../elsewhere", "--config", "cfg", "--checkpoint", str(tmp_path),
                       "--tasks", "806", "--n-seeds", "1", "--gpus", "0", "--openpi-dir", str(tmp_path),
                       "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"), "--dry-run"])
    assert exc.value.code == 2 and "must be one path component" in capsys.readouterr().err
    assert not (tmp_path / "r1").exists() and not (tmp_path / "elsewhere").exists()
