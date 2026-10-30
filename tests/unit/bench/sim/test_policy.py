"""Policy construction and the guidance safety check.

The guidance check is the load-bearing one: it is the only thing standing between a
mislaunched server and w=2 rollouts written silently into the w=0 directory.
"""
from __future__ import annotations

import pytest

from axis_qvla.bench.sim.policy import PolicyError, check_served_guidance


def test_matching_guidance_passes():
    check_served_guidance(
        {"cfg_conditioning": True, "guidance_scale": 2.0},
        cfg_conditioning=True, guidance_scale=2.0,
    )


def test_unguided_server_and_client_agree():
    check_served_guidance({}, cfg_conditioning=False, guidance_scale=0.0)


def test_weight_mismatch_is_refused():
    """The failure this exists for: results land in a directory naming the wrong w."""
    with pytest.raises(PolicyError, match="guidance mismatch"):
        check_served_guidance(
            {"cfg_conditioning": True, "guidance_scale": 2.0},
            cfg_conditioning=True, guidance_scale=0.0,
        )


def test_conditioning_mismatch_is_refused():
    with pytest.raises(PolicyError, match="guidance mismatch"):
        check_served_guidance(
            {"cfg_conditioning": False, "guidance_scale": 0.0},
            cfg_conditioning=True, guidance_scale=0.0,
        )


def test_absent_metadata_is_treated_as_unguided():
    """A server that publishes nothing is not evidence of guidance being on."""
    check_served_guidance(None, cfg_conditioning=False, guidance_scale=0.0)
    with pytest.raises(PolicyError):
        check_served_guidance(None, cfg_conditioning=True, guidance_scale=1.0)


def test_error_names_both_sides():
    """The operator has to know which side to relaunch."""
    with pytest.raises(PolicyError) as exc:
        check_served_guidance(
            {"cfg_conditioning": True, "guidance_scale": 3.0},
            cfg_conditioning=True, guidance_scale=1.5,
        )
    msg = str(exc.value)
    assert "3.0" in msg and "1.5" in msg and "serve_policy.py" in msg


# --- resolve_conditioning: the handshake is the source of record, flags are an assertion ----

def test_unset_flags_take_the_servers_conditioning():
    from axis_qvla.bench.sim.policy import resolve_conditioning

    c = resolve_conditioning({"cfg_conditioning": True, "guidance_scale": 1.5})
    assert (c.cfg_conditioning, c.guidance_scale, c.source) == (True, 1.5, "server")


def test_unset_flags_on_a_bare_server_read_as_unconditioned():
    """Absent keys are the plain policy -- openpi publishes cfg_conditioning only for a
    tagged serve -- and must not be read as 'unknown, refuse'."""
    from axis_qvla.bench.sim.policy import resolve_conditioning

    c = resolve_conditioning({"ckpt": "x"})
    assert (c.cfg_conditioning, c.guidance_scale, c.source) == (False, 0.0, "server")


def test_a_matching_assertion_is_recorded_as_asserted():
    from axis_qvla.bench.sim.policy import resolve_conditioning

    c = resolve_conditioning({"cfg_conditioning": True, "guidance_scale": 0.0},
                             cfg_conditioning=True, guidance_scale=0.0)
    assert c.source == "asserted" and c.cfg_conditioning is True


def test_a_negative_assertion_against_a_tagged_server_is_refused():
    """`--no-cfg-conditioning` must keep its teeth: deriving from the server is a default,
    not a way to make a wrong flag harmless."""
    from axis_qvla.bench.sim.policy import resolve_conditioning

    with pytest.raises(PolicyError, match="mismatch"):
        resolve_conditioning({"cfg_conditioning": True, "guidance_scale": 0.0},
                             cfg_conditioning=False)


def test_a_partial_assertion_checks_only_what_was_said():
    """Asserting the scale alone borrows the served conditioning; asserting conditioning
    alone borrows the served scale. Either half can still be wrong on its own."""
    from axis_qvla.bench.sim.policy import resolve_conditioning

    meta = {"cfg_conditioning": True, "guidance_scale": 2.0}
    c = resolve_conditioning(meta, guidance_scale=2.0)
    assert (c.cfg_conditioning, c.guidance_scale, c.source) == (True, 2.0, "asserted")
    with pytest.raises(PolicyError, match="mismatch"):
        resolve_conditioning(meta, guidance_scale=0.0)
    with pytest.raises(PolicyError, match="mismatch"):
        resolve_conditioning(meta, cfg_conditioning=False)


def test_the_served_policy_module_reaches_no_torch_backend():
    """The served path must reach no torch backend. The torch policies are reached only through a
    local-policy backend, which this release does not ship. So an import of a train package, libs
    or torch in policy.py could only tie the served path to the local backend it exists to keep
    apart."""
    import ast
    import pathlib

    import axis_qvla.bench.sim.policy as served

    tree = ast.parse(pathlib.Path(served.__file__).read_text())
    names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    pkg = served.__name__.split(".")[0]
    assert not any(m.startswith((f"{pkg}.train", "libs")) or m == "torch" for m in names)
