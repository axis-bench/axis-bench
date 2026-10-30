# Connect your own model

Reported results belong to a fixed **benchmark version**: a benchmark is frozen once numbers are
reported on it. New tasks, new layouts or a new execution constant make a new version (a new directory
under `conf/benchmarks/`), so a reported number keeps its meaning. v1.0
ships one version, `axis-heldout-v1`: 24 held-out tasks, 50 seeds each. Compare your results only with
results of the same version ([the benchmark](../reference/verification.md#the-benchmark)).

Any policy can be evaluated on the 24 held-out tasks if it runs behind a small WebSocket server that speaks
the benchmark's protocol. On connect, the server sends a handshake. For each observation (two camera frames,
the 8-D state and the instruction) it replies with a chunk of actions. The whole contract is on
[Policy server](../reference/policy-server.md).

You need a GPU with EGL, the evaluation environment (`qvla env build eval`) and what your model needs.

## 1. Write a server

`experiments/examples/policy_server.py` is a complete server. This policy holds the arm still; replace `act`
with your model:

```python file=experiments/examples/policy_server.py
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
```

Start it with the evaluation environment's interpreter, which has its packages (numpy, websockets and
openpi's client; `qvla env build eval` builds it), and leave it running:

<!-- ci
CI runs the server with its own interpreter, which has those packages, behind the evaluation
environment's path:

```bash ci
mkdir -p .venv-eval/bin
printf '#!/bin/sh\nexec python "$@"\n' > .venv-eval/bin/python
chmod +x .venv-eval/bin/python
```
-->

```bash ci background=1
.venv-eval/bin/python experiments/examples/policy_server.py --port 8000
```

## 2. Check it

`check-server` runs from any interpreter: it runs itself in the evaluation environment, as the
evaluation does. It connects, reads the handshake and sends a few observations shaped as the
evaluation sends them:

```bash ci
qvla bench check-server --policy 127.0.0.1:8000 --timeout 60
```

```text output
…: conformant (qvla.policy_server/1)
```

## 3. Get the benchmark

```bash setup
qvla data get bench
```

<!-- ci
CI fetches nothing here; it writes a `qvla.toml` whose roots sit under the current directory:

```bash ci
qvla init --root srv
```

```text output
OK    wrote …/qvla.toml (data under …/srv)
```
-->

## 4. Evaluate

```bash gpu
qvla bench eval --policy 127.0.0.1:8000 --run my_model
```

<!-- ci
`--print` prints the command and starts nothing:

```bash ci
qvla bench eval --policy 127.0.0.1:8000 --run my_model --print
```

```text output
# evaluates: the running server(s) 127.0.0.1:8000
# results file: …/out_my_model/results.json
```
-->

It runs the 24 tasks with 50 seeds each, writes the results file and prints the all-24 success rate with a
per-task table in the layout of the paper's main results. If the server runs on another machine,
`--policy` takes its address.
What to report: [reporting rules](../reference/reporting-rules.md).
