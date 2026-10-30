"""CLI argument contract: the flag combinations that must be refused before any GPU work.

Only the parse-and-validate half is exercised here. Everything past the validation gate
contacts a policy server and a simulator, and is covered by the unit tests of the pieces it
wires together.
"""
from __future__ import annotations

import pytest

from axis_qvla.bench.sim.cli import _domain_of, build_parser, main


ROOTS = ["--log-root", "/logs", "--work-dir", "/work", "--task-data-root", "/tasks"]


def _args(*extra):
    return build_parser().parse_args(
        ["--task-id", "1889", "--variant", "vanilla", "--checkpoint-dir", "/ckpt", *ROOTS, *extra]
    )


def test_the_rollout_cli_reads_its_task_from_the_benchmark_only(capsys):
    with pytest.raises(SystemExit) as exc:
        _args("--manifest", "m.json")
    assert exc.value.code == 2 and "--manifest" in capsys.readouterr().err


def test_fresh_layouts_are_still_the_default():
    """Fresh layouts are the default, run in the dual-sim domain (test_dual_sim_is_the_default)."""
    assert _args().eval_layouts is True


def test_no_eval_layouts_selects_physics():
    """--no-eval-layouts must also drop dual-sim, which is a render-domain path."""
    assert _domain_of(_args("--no-dual-sim", "--no-eval-layouts")) == "physics"


def test_dual_sim_selects_dualsim():
    assert _domain_of(_args("--dual-sim")) == "dualsim"


def test_dual_sim_wins_over_eval_layouts():
    assert _domain_of(_args("--dual-sim", "--eval-layouts")) == "dualsim"


def test_guidance_without_conditioning_is_refused():
    """w on an untagged uncond branch is a no-op; naming the run w=2 would be a lie."""
    with pytest.raises(SystemExit, match="cfg-conditioning"):
        main(["--task-id", "1", "--variant", "cfg", "--checkpoint-dir", "/c", *ROOTS,
              "--guidance-scale", "2.0"])


def test_dual_sim_without_layouts_is_refused():
    """The shipped-seed physics path has no render half — refuse, don't ignore the flag."""
    with pytest.raises(SystemExit, match="dual-sim"):
        main(["--task-id", "1", "--variant", "vanilla", "--checkpoint-dir", "/c", *ROOTS,
              "--dual-sim", "--no-eval-layouts"])


def test_guidance_zero_without_conditioning_is_allowed():
    """w=0 untagged is the ordinary unguided run, not a mismatch."""
    assert _args("--guidance-scale", "0.0").guidance_scale == 0.0


def test_arm_name_override_is_available():
    """The held-out sweeps write hx_<task>_<arm>, which is not a derivable variant name."""
    assert _args("--run-name", "hx_1889_ki").arm_name == "hx_1889_ki"


def test_arm_name_defaults_to_derived():
    assert _args().arm_name is None


def test_task_id_and_checkpoint_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--variant", "vanilla"])


def test_n_seeds_is_unset_by_default_so_the_benchmark_decides():
    """A benchmark declares n_seeds=50; a parser default would quietly evaluate another count of
    them under the benchmark's name."""
    assert _args().n_seeds is None


def test_the_roots_are_required_and_the_render_root_is_the_submodule(capsys):
    from axis_qvla.bench.sim.cli import REPO_ROOT

    assert _args().render_root == REPO_ROOT / "third_party" / "axis-render"
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--task-id", "1889", "--variant", "vanilla", "--checkpoint-dir", "/ckpt"])
    err = capsys.readouterr().err
    assert exc.value.code == 2 and all(f in err for f in ("--work-dir", "--log-root", "--task-data-root"))


def test_benchmark_defaults_to_the_tracked_declaration():
    from axis_qvla.bench.spec import DEFAULT_NAME

    assert _args().benchmark == DEFAULT_NAME == "franka_heldout24"


def test_conditioning_flags_are_unset_by_default():
    """Unset means 'read it off the server'; False means 'assert the server is untagged'.
    A store_true flag cannot tell those apart, which is why both are tri-state."""
    a = _args()
    assert a.cfg_conditioning is None and a.guidance_scale is None
    assert _args("--no-cfg-conditioning").cfg_conditioning is False
    assert _args("--cfg-conditioning", "--guidance-scale", "0").cfg_conditioning is True


def test_seeds_subset_is_accepted_as_a_comma_list():
    assert _args("--seeds", "806000,806003").seeds == "806000,806003"


# --- domain default: dual-sim is the evaluation domain ---------------------------

def test_dual_sim_is_the_default():
    """Dual-sim, the benchmark's evaluation domain, is the default."""
    assert _domain_of(_args()) == "dualsim"


def test_no_dual_sim_selects_the_render_domain():
    assert _domain_of(_args("--no-dual-sim")) == "render"


def test_no_eval_layouts_still_selects_physics():
    assert _domain_of(_args("--no-dual-sim", "--no-eval-layouts")) == "physics"


def test_render_domain_notes_that_it_is_not_an_evaluation_domain():
    from axis_qvla.bench.sim.cli import warn_if_unsupported_domain

    seen = []
    warn_if_unsupported_domain("render", seen.append)
    assert seen and "not a supported evaluation domain" in seen[0]
    assert "--dual-sim" in seen[0]


def test_dualsim_does_not_warn():
    from axis_qvla.bench.sim.cli import warn_if_unsupported_domain

    seen = []
    warn_if_unsupported_domain("dualsim", seen.append)
    assert seen == []


def test_physics_domain_does_not_warn():
    """The note is specific to the render-only domain."""
    from axis_qvla.bench.sim.cli import warn_if_unsupported_domain

    seen = []
    warn_if_unsupported_domain("physics", seen.append)
    assert seen == []
