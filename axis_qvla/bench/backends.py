"""The three reference policy servers `qvla bench eval --backend` can start, in one table.

Why: the eval command's choices, the launcher's host roots and ports and the campaign driver's
server templates and serving checks all read this one table. A policy that is not one of these is
evaluated with `--policy HOST:PORT` against a server you started yourself
(docs/reference/policy-server.md); nothing here is needed for that.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses

QUALITY_TAG = 5
# X-VLA soft-prompt row for the Franka sim: an UNTRAINED row of lerobot/xvla-base (rows 10-17 are
# the base's trained embodiments), the same on every arm so pretrain, finetune and serving agree.
XVLA_DOMAIN_ID = 20
DEFAULT_BACKEND = "openpi"


@dataclasses.dataclass(frozen=True)
class ReferenceServer:
    name: str
    logs_root: str          # host root the rollout logs land under
    server_root: str        # host root the server runs in (its working directory)
    base_port: int
    server_cmd: str | None  # template with {server_python} {ckpt} {port}; None = openpi's serve_policy argv
    can_guide: bool         # implements classifier-free guidance (a non-zero --guidance-scale)
    xla: bool               # a JAX server: the benchmark's XLA pin applies; uses the host's "online" venv
    # The host ROOT key (the host profile's [roots]) holding this backend's server interpreter
    # when `xla` is false; unused (None) when `xla` is true, which uses the host's "online" venv
    # instead. A declared field, not a literal re-spelled at the call site.
    server_venv_root: str | None
    hf_env: bool             # whether the LeRobot/HF cache env (HF_HUB_OFFLINE, HF_HOME, ...) applies
    short_label: str         # the backbone's proper name alone, for an inline refusal sentence
    label: str               # short_label plus where its server lives, for the --backend help listing


SMOLVLA_SERVER_CMD = (
    "{server_python} axis_qvla/bench/servers/lerobot_server.py --policy-type smolvla "
    "--ckpt {ckpt} --host 127.0.0.1 --port {port}"
)
XVLA_SERVER_CMD = (
    "{server_python} axis_qvla/bench/servers/xvla.py --policy-type xvla "
    f"--ckpt {{ckpt}} --host 127.0.0.1 --port {{port}} --domain-id {XVLA_DOMAIN_ID}"
)
REFERENCE_SERVERS: dict[str, ReferenceServer] = {
    "openpi": ReferenceServer("openpi", "bench", "openpi", 34000, None, True, True, None, False,
                              "π0.5", "π0.5, openpi's serve_policy.py"),
    "smolvla": ReferenceServer("smolvla", "smolvla_bench", "smolvla_bench", 35000, SMOLVLA_SERVER_CMD,
                               False, False, "smolvla_venv", True,
                               "SmolVLA", "SmolVLA, axis_qvla/bench/servers/lerobot_server.py"),
    "xvla": ReferenceServer("xvla", "smolvla_bench", "smolvla_bench", 38000, XVLA_SERVER_CMD,
                            False, False, "smolvla_venv", True,
                            "X-VLA", "X-VLA, axis_qvla/bench/servers/xvla.py (domain id 20)"),
}
BACKENDS = tuple(REFERENCE_SERVERS)

# The `qvla env build <name>` that builds each reference server's environment.
ENVIRONMENTS = {"openpi": "pi05", "smolvla": "smolvla", "xvla": "xvla"}
