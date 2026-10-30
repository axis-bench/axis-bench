# Policy-server protocol

`qvla bench eval` drives the simulator and asks a **policy server** for actions over a WebSocket,
in the protocol `qvla.policy_server/1`. A policy can be evaluated if its server speaks it: start your
server, check it with `qvla bench check-server`, then run
`qvla bench eval --policy HOST:PORT --run <name>` ([connect your own model](../tutorials/connect-your-own-model.md)
walks through it). The three backbones of the paper ship as reference servers (below).

A first server needs the three required keys of the [handshake](#handshake), the
[request](#request) it receives and the [reply](#reply) it sends; the rest of this page is the
detail behind them.

## Transport

- One WebSocket connection per rollout batch; binary frames, no compression, no size limit.
- Every frame is MessagePack. NumPy arrays and scalars use the `openpi_client.msgpack_numpy`
  extension ([wire format of arrays](#wire-format-of-arrays)). Nothing is pickled.
- The client sends no keep-alive pings, because a server may be busy compiling for minutes on its
  first request. A server must not require them.
- Strictly request/reply: the client sends one observation and waits for its reply.
- Optional: `GET /healthz` answers `200 OK` (the reference LeRobot server does).

## Handshake

On connect, the server sends the handshake, a map, as its first frame, before anything else.

| key | required | type | meaning |
|---|---|---|---|
| `supports_client_noise` | yes | `true` | the server uses the noise the client sends (below). The client refuses a server without it. |
| `action_horizon` | yes | int > 0 | rows of the noise tensor the sampler takes |
| `action_dim` | yes | int > 0 | columns of that tensor |
| `cfg_conditioning` | no | bool, default `false` | the server conditions every prompt (e.g. a quality tag) |
| `guidance_scale` | no | float, default `0` | classifier-free guidance weight `w`; non-zero needs `cfg_conditioning` |
| `quality_tag` | no | int | the tag a conditioned server appends |
| `beta` | no | float | `1 + w`, recorded for the results file |
| `checkpoint_dir`, `config_name` | no | str | what is being served, recorded in the run record |
| `protocol` | no | `"qvla.policy_server/1"` | if present, must be this string |
| `image_geometry` | no | map camera key → str | the transform the server applies to that camera's frame before its model sees it: `raw` (none), `center_crop224` or `letterbox224`. The simulator client always sends raw 640 × 360 frames; the real-world line's client sends 320 × 180 frames ([real world](../tutorials/real-world-your-workspace.md)) |
| `state_order` | no | list of str | the `state` components, in order; for the simulator `["q0", …, "q6", "gripper"]` |
| `action_units` | no | str | units of action columns 0–6; `"rad_s"` for the simulator (joint velocities in rad/s) |
| `action_fps` | no | number | the rate actions are executed at; `15` |
| `action_dim_out` | no | int | the action columns meant to be executed: at least 8, at most `action_dim` |
| `prompt_suffix` | no | str | text the server appends to every prompt (a conditioned server's tag, e.g. `"\nQuality: 5"`) |

The last six keys state the serving contract. They are optional. `check-server` records the ones a
server sends (its report's `contract` line) and fails an unknown camera or transform in
`image_geometry`, and any other key whose value is not the simulator's: for example
`action_units: "normalized"`. State `image_geometry` so that a mismatch with the geometry the
checkpoint was trained with is visible: a policy trained on 224 × 224 centre crops and served raw
frames still answers, wrongly.

The handshake's `action_horizon` × `action_dim` is the shape of the **noise** tensor the client
sends, not of the reply: the π0.5
reference server announces `(10, 32)`, the SmolVLA reference `(50, 32)`; both reply with 10 × 8
actions.

The handshake decides the result directory: a run `A` served with `cfg_conditioning` true and
weight `w` is filed as `A_w<w>_dualsim`, otherwise as `A_dualsim`. The client reads the sampling
settings from the handshake (a flag passed to the client is only an assertion,
refused when the server disagrees).

## Request

A map with exactly these keys:

| key | type | content |
|---|---|---|
| `base_0_rgb` | uint8 `(360, 640, 3)`, HWC, RGB | the front camera, raw as rendered (the benchmark declares `image_geometry: raw`); the server applies the transform its handshake states |
| `left_wrist_0_rgb` | uint8 `(360, 640, 3)` | the wrist camera |
| `state` | float32 `(8,)` | joint angles q0..q6 of the Franka arm in rad, then gripper closedness in [0, 1] (0 open, 1 closed; the simulator's finger width normalised by its 0–0.04 m range) |
| `prompt` | str | the task instruction (`task_name_canonical` of the benchmark's tasks file), untagged: a conditioned server adds its tag |
| `__noise__` | float32 `(action_horizon, action_dim)` | the sampler's initial noise for this call |

`__noise__` is a pure function of (task id, seed, call index within the rollout): standard normal
draws from `numpy.random.Generator(Philox(SeedSequence([0x41584953, task, seed, call])))`
(`axis_qvla/bench/sim/inference_noise.py`). A server must remove it from the observation before its own
input transforms and use it as the sampler's starting noise; a deterministic policy may ignore it.
This makes a rollout independent of which server answered it and of how many requests came
before.

A conditioned server appends its tag to the prompt itself (the references append
`"\nQuality: <tag>"`) and must refuse a prompt that already carries one.

## Reply

A map with:

- `actions`: a 2-D float array `(H, D)`, `H` ≥ 8 and `D` ≥ 8. Columns 0–6 are joint
  **velocities in rad/s** for joints 1–7; column 7 is gripper closedness. Columns past 8 and rows
  past 8 are ignored.
- optionally `server_timing`: a map of timings in ms, recorded and otherwise ignored.

## How the client executes a reply

At 15 Hz, for the first 8 rows: the joint command is integrated, `q_cmd += v / 15`, clipped to the
joint limits and held between steps (a persistent position target, not a re-read of the measured
position); the gripper closes when column 7 ≥ 0.5 and opens otherwise. After 8 steps the client
renders a new observation and sends the next request (call index + 1). An episode lasts at most
1,500 steps (100 s) and succeeds when the task's checker holds for 45 consecutive frames (3 s).
These constants are the benchmark's declared execution constants
(`conf/benchmarks/<version>/execution.json`). `qvla bench eval` refuses to start a run whose
environment would execute anything else (an exported `PI05_REPLAN`, say) unless it is given
`--off-protocol`; the run then records every deviation in its `run_manifest.json` and in each
task's `meta.json`, and the results file `qvla bench eval` writes at the end says
`off_protocol: true` ([results format](results-format.md)).

**Simulation units are not the real-robot units.** A fine-tuned real-world run declares
`action_units: "droid_normalized_joint_velocity_command"`: DROID's normalised joint-velocity command in
[−1, 1] (controller target `q + 0.2·a`) and an absolute gripper position, at 15 Hz. `check-server`
fails that value, since it judges a server against the simulator; `qvla realrobot fr3 check
--handshake` holds the handshake to the run instead ([real world](../tutorials/real-world-your-workspace.md)).
Each line's servers speak its own units: the simulator evaluates simulation servers, and the robot runs
real-world ones.

## Errors

On any failure the server sends one **text** frame holding the error and closes the connection with
code 1011. The client raises; the rollout is recorded with an `error:` reason, and the scorer scores
only runs whose rollouts all completed.

## Conformance

```
qvla bench check-server --policy HOST:PORT [--task <id>] [--calls 3] [--report check.json]
```

It checks the handshake (the required keys; `image_geometry` against the known transforms;
`state_order`, `action_units`, `action_fps` and `action_dim_out` against the simulator's values;
`prompt_suffix` for its type only) and, for each call, that the reply is a 2-D float array with at
least 8 rows and 8 columns and only finite values (exit 1 otherwise); it reports the stated contract
and whether the gripper column stays in [0, 1]. A server that does not complete the handshake
within `--timeout`, or answers with text or bytes that are not msgpack, fails the handshake check.
One request is sent twice: a different action shape or an error fails, a different value is only
reported (GPU kernels need not repeat bitwise). `--timeout` bounds the port wait, the WebSocket
handshake and every reply, including the first; raise it for a server that compiles for minutes on
its first request (π0.5, X-VLA). Exit 2 means the server could not be reached at the TCP level, or
the arguments are wrong.

## Reference servers

| backbone | `qvla bench eval --backend` | server | noise shape |
|---|---|---|---|
| π0.5 | `openpi` | openpi's `serve_policy.py` (`third_party/openpi`), given `--policy.config=<file>.json`: `bench eval --from-run <run id>` renders that file from your own trained run, and `bench eval --checkpoint-id <id>`, or the openpi config name of a `cells.json` `eval_command` as `--config`, from the paper run that trained the checkpoint | (10, 32) |
| SmolVLA | `smolvla` | `axis_qvla/bench/servers/lerobot_server.py --policy-type smolvla` | (50, 32) |
| X-VLA | `xvla` | `axis_qvla/bench/servers/xvla.py --policy-type xvla --domain-id 20` | from the model |

`axis_qvla/bench/servers/lerobot_server.py --policy-type mock` serves zeros with no model and is what the
protocol tests run against.

`bench eval` reads `<cache>/openpi/big_vision/paligemma_tokenizer.model`: it starts the π0.5 server
with `OPENPI_DATA_HOME` at the host's `<cache>/openpi`, where openpi downloads the PaliGemma tokenizer
at its first start, and a start without the file fails naming the file and the root.
`qvla data get base-weights` stores it under `<cache>/checkpoints/qvla-v1-base-weights/openpi/big_vision/`
(`qvla train` sets `OPENPI_DATA_HOME` to that copy's `openpi/`); link it across on an offline machine.

## Wire format of arrays

NumPy arrays and scalars use the `openpi_client.msgpack_numpy` extension. An array packs as a
4-entry map whose **keys are msgpack BIN, not str** -- `b"__ndarray__"` (value `true`), `b"data"` (the array's raw bytes, C order, via `.tobytes()`),
`b"dtype"` (a str, numpy's `.dtype.str`, e.g. `"<f4"` little-endian float32, `"|u1"` uint8) and
`b"shape"` (an array of ints). A NumPy scalar packs as a 3-entry map with the same BIN keys
and no `shape`: `b"__npgeneric__"` (value `true`), `b"data"` (the scalar's `.item()`, a plain
msgpack number or bool) and `b"dtype"` (its `.dtype.str`). Nothing is pickled.

A `(1, 1)` `uint8` array holding `5`, packed alone, is 45 bytes:
`84 c4 0b 5f5f6e6461727261795f5f c3 c4 04 64617461 c4 01 05 c4 05 6474797065 a3 7c7531 c4 05 7368617065 92 01 01`
-- `84` (fixmap, 4 pairs), `c4 0b` + 11 bytes (BIN `__ndarray__`), `c3` (`true`), `c4 04` + 4
bytes (BIN `data`), `c4 01 05` (BIN, the array's one data byte), `c4 05` + 5 bytes (BIN
`dtype`), `a3 7c7531` (STR `"|u1"`), `c4 05` + 5 bytes (BIN `shape`), `92 01 01` (array `[1, 1]`).
