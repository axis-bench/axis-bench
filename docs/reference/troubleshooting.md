# Troubleshooting

Find the error or the symptom below, then follow its fix: each section says what the problem means
and what to do. A quoted message is the part of the
refusal that does not change; `…` stands for the parts that do (a path, an id). If yours is not
here, open an issue on this repository with the command, the full output and `qvla bench check`'s.

## Rendering fails on a machine without a GPU

MuJoCo renders offscreen through EGL or OSMesa, picked by `MUJOCO_GL` and `PYOPENGL_PLATFORM`. EGL
needs a GPU driver, or Mesa's EGL on a CPU; OSMesa is a software renderer that needs neither (on
Debian and Ubuntu, `apt-get install libosmesa6`). On a machine without a GPU, run the real-robot
renderer with `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`, as the CI `render` job does: it is slower,
and its output passes the same hard gates. `qvla bench eval` sets `MUJOCO_GL=egl` for its rollout
workers (`axis_qvla/bench/execution.py`), so an evaluation needs a machine with EGL.

## Caches fill your home directory

Each environment is built in its own checkout (`qvla env build <name>`; `--print` prints the
commands), and the package installer and the Hugging Face client cache under `~/.cache` by default: a training
environment and its base models take tens of GB. On a machine with a small home quota, point the
caches at a large disk before building: `UV_CACHE_DIR` (uv), `PIP_CACHE_DIR` (pip) and `HF_HOME`
(the Hugging Face client). Everything `qvla` downloads or computes goes under the roots of your
`qvla.toml` (`<corpus>`, `<cache>`), so under your home directory only if a root says so;
`qvla bench eval` also points `HF_HOME`, `OPENPI_DATA_HOME` and the JAX compilation cache at
`<cache>`.

## π0.5 training: `--fsdp-devices` and the GPUs you have

<!-- quote: axis_qvla/train/generate.py -->
```text
--fsdp-devices … does not divide the global batch …
```

A π0.5 run shards its train state over `fsdp_devices` GPUs; the paper's pretraining preset sets 8.
The number of GPUs the launch sees (`CUDA_VISIBLE_DEVICES`) must be a multiple of it, and it must
divide the global batch. On a machine with fewer GPUs, pass `--fsdp-devices <your GPU count>` to
`qvla train` (or `qvla train pretrain|finetune`): the global batch, and so what the run trains, stays the
same. A smoke run uses `--fsdp-devices 1`.

## Resuming a π0.5 run after a crash

<!-- quote: axis_qvla/train/cli.py -->
```text
… already exists. The pinned openpi fork refuses to train into an existing checkpoint directory unless it resumes
```

Run `qvla train <spec> --resume` again: the run continues in `<runs>/<run id>/train/` from the step
after its last saved checkpoint, with the configuration the run recorded (a spec that now gives another
configuration is refused). If the run died before its first checkpoint, `qvla train` says so instead:

<!-- quote: axis_qvla/train/cli.py -->
```text
… already exists and holds no saved step: a launch stopped before its first checkpoint
```

There is nothing to resume: remove that directory yourself (`qvla train` keeps every run directory)
and run `qvla train <spec>` again.

## A SmolVLA or X-VLA relaunch is refused

<!-- quote: axis_qvla/train/cli.py -->
```text
… already exists. LeRobot refuses to train into an existing output_dir when resume is false
```

LeRobot would refuse at launch, after the GPUs are allocated, so `qvla train` refuses first. Remove the
named directory yourself to start over.

## The smoke fixture's statistics are refused

<!-- quote: axis_qvla/train/smoke.py -->
```text
… holds other statistics than this fixture's …
… it is never replaced: remove it yourself and build again
```

The smoke run specs name their normalisation statistics by digest, so the file is written once. A
rebuild from other inputs computes other statistics; remove the named file (and the fixture) and build
again.

## The SmolVLA or X-VLA environment lost the policy-server client after `uv sync`

<!-- quote: axis_qvla/train/envs.py -->
```text
missing …
import …: …
```

The reference policy servers' client (`openpi-client`) pins numpy < 2, so the LeRobot environment
installs it without its dependencies and then adds the packages it needs (msgpack, websockets and
dm-tree, with dm-tree's absl-py and wrapt) at pinned versions. A later `uv sync` in `third_party/lerobot`
removes them again, and the environment's check reports them missing, or the import of
`openpi_client.websocket_client_policy` failing. Rebuild it with `qvla env build smolvla` (SmolVLA and
X-VLA share it), which installs them again after its own `uv sync`.

## `qvla data get` without the checksum manifest

<!-- quote: axis_qvla/data/get.py -->
```text
… does not exist, so nothing can be fetched verified
```

`qvla data get` and `qvla data verify` check every file against `conf/release/v1.0.lock.json`,
which a release checkout carries; without it nothing is fetched. `qvla paper tables` still runs
from any local snapshot of the results repository, for example
`hf download axisrobotics/qvla-v1-results --repo-type dataset --revision v1.0 --local-dir <dir>`:
the committed results map pins every byte it reads
([from a local snapshot](data.md#from-a-local-snapshot)).

## `check-server` times out before the handshake completes

A server that compiles its model at the first request (π0.5, X-VLA) can take minutes to answer.
`--timeout` bounds the port wait, the WebSocket handshake and every reply: raise it, for example
`--timeout 600`. A server that connects but stays silent, or answers with text or bytes that are
not msgpack, fails the handshake check within `--timeout`; a refused TCP connection is reported as
unreachable ([policy-server protocol](policy-server.md#conformance)).

## `qvla paper tables` exits 3 after `pip install .`

<!-- quote: axis_qvla/paper/cli.py -->
```text
`qvla paper tables` runs from a checkout of the repository
```

It reads its defaults (the results map, the table layout, the benchmark declaration, the checksum
manifest) from the checkout. Install it editable (`pip install -e .`) or run
`python -m axis_qvla.cli paper tables ...` from the repository root.
