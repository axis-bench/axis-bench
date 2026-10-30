"""Launching a reference server as a script must not let its own directory shadow the `lerobot`
package it imports.

Running `python <dir>/<script>.py` sets sys.path[0] to <dir> -- the SCRIPT'S OWN directory --
ahead of every installed package, including anything on PYTHONPATH. A module directly under
axis_qvla/bench/servers/ named like a third-party package a server imports then IS that import:
`from lerobot.policies.factory import ...` resolves to the sibling file before Python ever
looks at site-packages, and the server dies with `ModuleNotFoundError: No module named
'lerobot.policies'; 'lerobot' is not a package` before it binds a port (`check-server` then
reports a plain connection refusal, with nothing pointing at the cause).

This runs the EXACT command `axis_qvla.bench.backends` renders for each LeRobot-based reference
server -- SmolVLA and X-VLA -- as a real subprocess, cwd=REPO (the cwd `bench eval` launches
from). `build_predictor` does `import torch` immediately before the vulnerable
`from lerobot.policies.factory import ...` line, and the repository's own CI `test` job
installs NEITHER torch nor lerobot (openpi-client supplies websockets/msgpack/numpy, which is
why this directory's other tests -- which never call `build_predictor` with a real policy type
-- pass there). So this test must not need a real torch or a real lerobot anywhere on the
system, in either direction: it must still catch the shadow when both are genuinely absent (CI),
and it must not accidentally pass or fail because THIS box's interpreter happens to carry a real
torch (it does) or a real lerobot (it does not).

Both are planted as minimal STUB packages on PYTHONPATH -- a directory Python checks after
sys.path[0] (the script's own directory) but before site-packages. That ordering is exactly what
makes the test catch a shadowing module: when sys.path[0] holds a sibling `lerobot.py`, it
wins the lookup over the PYTHONPATH stub and the shadow signature appears; when it holds no such
file, resolution falls through to the stub.
`torch` is never shadowed by anything in axis_qvla/bench/servers/ either way, so its stub exists
purely to get the subprocess past `import torch` without needing a real one. Each stub prints
its own marker on import, so a pass proves BOTH resolved to the planted packages -- not to
whatever torch/lerobot, real or missing, happens to sit on this interpreter's own site-packages.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from axis_qvla.bench import backends

REPO = Path(__file__).resolve().parents[4]
TORCH_MARKER = "stub-torch-loaded"
LEROBOT_MARKER = "stub-lerobot-factory-loaded"
SHADOW_SIGNATURE = "'lerobot' is not a package"


def _plant_stubs(root: Path) -> None:
    """A minimal `torch` (just importable) and a minimal `lerobot` (`policies/factory.py`, no
    `modeling_*` submodule beneath it). The missing submodule is deliberate: a server that
    resolves `lerobot` correctly gets PAST the factory import (proving the resolution, via its
    marker) and only then fails on the next, intentionally-absent submodule -- an ordinary
    `ModuleNotFoundError` naming that submodule, never the "not a package" shadow signature.
    """
    (root / "torch").mkdir()
    (root / "torch" / "__init__.py").write_text(f"print({TORCH_MARKER!r})\n")

    policies = root / "lerobot" / "policies"
    policies.mkdir(parents=True)
    (root / "lerobot" / "__init__.py").write_text("")
    (policies / "__init__.py").write_text("")
    (policies / "factory.py").write_text(
        f"print({LEROBOT_MARKER!r})\n"
        "def make_pre_post_processors(*a, **kw):\n"
        "    raise NotImplementedError('stub')\n"
    )


@pytest.mark.parametrize("name, template", [
    ("smolvla", backends.SMOLVLA_SERVER_CMD),
    ("xvla", backends.XVLA_SERVER_CMD),
])
def test_rendered_server_command_resolves_lerobot_to_the_planted_package(name, template, tmp_path):
    _plant_stubs(tmp_path)
    argv = template.format(server_python=sys.executable, ckpt="/does/not/exist", port=0).split()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO)])}
    result = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=90)

    assert SHADOW_SIGNATURE not in result.stderr, (
        f"the {name} launch command ({argv[1]}) shadowed the real `lerobot` package with a "
        f"sibling file under axis_qvla/bench/servers/ instead of resolving to the installed "
        f"package:\n{result.stderr}"
    )
    assert TORCH_MARKER in result.stdout, (
        f"the {name} server never reached the planted torch stub -- it crashed before even the "
        f"`import torch` line, so this result says nothing about the shadow.\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert LEROBOT_MARKER in result.stdout, (
        f"the {name} server never reached the planted package's factory module -- it did not "
        f"resolve `lerobot` to the stub at all.\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
