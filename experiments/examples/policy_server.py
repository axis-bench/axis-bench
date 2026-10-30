"""A policy server of your own: the smallest server `qvla bench eval --policy` can evaluate.

docs/tutorials/connect-your-own-model.md walks through this file. It speaks qvla.policy_server/1
(docs/reference/policy-server.md): on connect, one handshake frame; then, for each observation the
client sends, one reply holding an (8, 8) action chunk. This policy holds the arm still with the
gripper open, so it never succeeds; replace `act` with your model.

    python experiments/examples/policy_server.py --port 8000
    python experiments/examples/policy_server.py --port 0 --port-file port.txt   # any free port, written to the file

Policy-server tier: numpy, websockets, openpi-client (its msgpack_numpy codec).
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.server import serve

HANDSHAKE = {
    "protocol": "qvla.policy_server/1",
    "supports_client_noise": True,     # required: the server takes the noise the client sends
    "action_horizon": 8,               # the noise tensor's shape; this policy ignores the noise
    "action_dim": 8,
    "image_geometry": {"base_0_rgb": "raw", "left_wrist_0_rgb": "raw"},
    "state_order": ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper"],
    "action_units": "rad_s",
    "action_fps": 15,
    "action_dim_out": 8,
}


def act(obs: dict) -> np.ndarray:
    """Eight rows of [7 joint velocities in rad/s, gripper closedness]: hold still, gripper open."""
    assert obs["state"].shape == (8,) and obs["base_0_rgb"].shape == (360, 640, 3)
    return np.zeros((8, 8), dtype=np.float32)


def handler(ws) -> None:
    ws.send(msgpack_numpy.packb(HANDSHAKE))
    for message in ws:
        obs = msgpack_numpy.unpackb(message)
        obs.pop("__noise__")            # the sampler's noise: a stochastic policy starts from it
        ws.send(msgpack_numpy.packb({"actions": act(obs)}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--port-file", type=pathlib.Path, default=None,
                    help="write the port here once listening (with --port 0, a free port)")
    a = ap.parse_args(argv)
    with serve(handler, a.host, a.port, compression=None, max_size=None) as server:
        port = server.socket.getsockname()[1]
        if a.port_file is not None:
            a.port_file.write_text(f"{port}\n")
        print(f"serving qvla.policy_server/1 on {a.host}:{port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
