# Real world: your workspace

The real-world line pretrains π0.5 on 19,250 simulation demonstrations (663 tasks) re-rendered as your
Franka FR3 workspace's camera sees them, then fine-tunes it on demonstrations from your robot. This
tutorial runs it for your workspace, from your camera's calibration to a policy server for your robot: describe the workspace in a
profile, check the profile against your camera, re-render the episodes for it, compute their quality
labels, pretrain, record your own demonstrations, fine-tune and serve the policy. The last section gives
the serving contract your robot's client follows.

**What you need.**
- the simulation demonstrations the real-world line pretrains on, `axisrobotics/qvla-v1-realworld`
  (42 GB: 663 tasks, 19,250 episodes);
- CPUs with an OpenGL backend (`MUJOCO_GL=egl` or `osmesa`) to re-render them, and 8 GPUs to train each run;
- a Franka with a Robotiq 2F-85 gripper on DROID's control stack, with DROID's camera set: two exterior
  cameras and a wrist camera.

Install with the `realrobot` extra and write `qvla.toml` ([install](../install.md)); `<runs>` below is its
`runs` root. Every step reads what the step before it wrote, under the roots of `qvla.toml`, and a step
that needs another environment names its `qvla env build` command
([environments](../install.md#environments)). First fetch the data:

```bash setup
qvla data get realworld
```

<!-- ci
CI fetches nothing: a fixture lays out stand-ins for what steps 1 to 4 and step 6 write, and `qvla init`
puts the roots of `qvla.toml` beside them:

```bash ci
python -m tests.docs.fixtures realworld-local roots/cache
qvla init --root roots
```

```text output
OK    wrote …/qvla.toml (data under …/roots)
```

The render steps run in the renderer's environment with software GL, on a six-episode copy of the data:

```bash ci-render
python -m tests.docs.fixtures realworld-data roots/corpus/qvla-v1-realworld
python -m tests.docs.fixtures workspace-photo exterior.png
qvla init --root roots
```

```text output
OK    wrote …/qvla.toml (data under …/roots)
```
-->

## 1. Describe your workspace

A workspace profile holds what the renders must share with your camera: the exterior camera's pose and
field of view, the table and the background. The exterior camera is the one the policy will see. This
writes one from your camera's calibration: the extrinsic as OpenCV's `solvePnP` gives it
(`x_cam = R x_base + t`: `R` row by row, then `t` in metres) and the vertical focal length in pixels of a
720-pixel-high image:

```bash setup
qvla realrobot workspace my_workspace.toml --base-to-camera <r11> <r12> <r13> <r21> <r22> <r23> <r31> <r32> <r33> <t1> <t2> <t3> --fy <fy>
```

<!-- ci
```bash ci
qvla realrobot workspace my_workspace.toml --base-to-camera 0 -1 0 -0.4 0 -0.9 0.9 0 -0.4 0.1 0.3 1.0 --fy 520
```

```text output
workspace my_workspace -> my_workspace.toml (camera pose and field of view; set its table, background and fixtures as measured)
```
-->

The rest of `my_workspace.toml` starts as the paper's reference workspace
(`conf/workspaces/fr3_reference.toml`, every field commented). Edit it so the renders look like your
workspace: positions are metres in the robot base frame (x forward, y left, z up from the table top where
the base stands), and an image is a path inside the profile's directory. Where each part of the rendered
scene is set:

| to change | edit |
|---|---|
| the exterior camera's pose and field of view | `[camera]` `pos`, `quat`, `fovy` (the command above sets them) |
| the table's height, position and size | `[table]` `height`, `center`, `size` |
| the table top's colour, or a photo of it | `[table.surface]` `rgb`, `texture = "<image>.png"`, `texture_repeat` |
| the wall behind the table and the floor | `[background]` `wall_distance`, `wall_half_width`, `wall_height`; `[background.wall]` and `[background.floor]` `rgb`, `texture` |
| how matte or shiny each surface is, and how much it varies per episode | each surface's `material`, `material_caps`, `emission`, `rgb_jitter`, `material_jitter` |
| boxes on the table that your camera sees (mounting plates, rails) | `[[fixtures]]` `name`, `pos` (x and y in the base frame, z above the table top), `half`, `material`, with its colour in `[materials.<material>]` |
| the lighting | `LIGHT` and `HEADLIGHT` in `axis_qvla/realrobot/render/rig.py` |
| the objects' meshes and textures | the scenes `tasks/task_<T>/scenes/variant_<I>.xml` and the mesh layouts `tabletopgen/<L>/` of the render input, `<cache>/derived/realworld/source/`, which step 3 lays out from the real-world data ([its layout](../reference/data.md#the-real-world-data)) |
| where the objects start and how they move | each episode's recorded trajectory, `tasks/task_<T>/scenes/attempt_<A>.json`; the objects follow it on your table top |

## 2. Check it against your camera

This draws the outlines of the profile's table and robot over a frame of your exterior camera, with the
robot posed at the joint positions it had when the frame was taken (DROID's `joint_positions`):

```bash gpu
qvla realrobot check-camera --profile my_workspace.toml --image exterior.png --joints <q0> <q1> <q2> <q3> <q4> <q5> <q6>
```

<!-- ci
The stand-in frame was rendered at the home pose, which `--joints` defaults to:

```bash ci-render
qvla realrobot workspace my_workspace.toml
qvla realrobot check-camera --profile my_workspace.toml --image exterior.png
```

```text output
overlay of exterior.png on workspace my_workspace -> overlay.png
```
-->

`overlay.png` is your frame with the rendered table edge (red) and robot (green) outlined, beside a
50/50 blend of the two images. Adjust the profile until both outlines sit on the real table and arm.

## 3. Re-render the pretraining data

This renders every task for your profile and writes its training view: 7 joint angles and the gripper's
closedness as the state, 7 joint velocities (rad/s, clipped to ±1) and the closedness as the action, at
15 Hz:

```bash gpu
qvla realrobot rerender --profile my_workspace.toml
```

<!-- ci
```bash ci-render
qvla realrobot rerender --profile my_workspace.toml
```

```text output
rendered tasks [9001] -> …/render
training view: 1 tasks written, 0 kept -> …/d8
```
-->

Each task is written whole or not at all, so the same command run again after an interruption renders only
the tasks not yet rendered. `--tasks` renders a selection, and several processes with disjoint `--tasks`
lists render into one layout.

## 4. Compute the quality labels

The labels align to the rendered frames, so they are computed after the re-render. This computes each
episode's phase curve on exactly its rendered frames, checks the render, and compiles the labels of the line's two
pretraining runs, `bc` and `conditioning`:

```bash gpu
qvla realrobot labels
```

<!-- ci
Six episodes that make no contact have one quality bin, so CI labels `bc` only:

```bash ci-render
qvla realrobot labels --runs bc
```

```text output
all hard gates pass; …
compiled labels realrobot.bc -> …
```
-->

It prints `all hard gates pass` once the render follows the recorded hand and objects
([how results are verified](../reference/verification.md#real-robot)). Run again after an interruption, it
keeps the phase curves already computed.

## 5. Pretrain

The line has three configurations: quality conditioning (`cond`), plain BC (`bc`) and fine-tuning with no
pretraining (`only`). This writes the two stage-1 run specs, `bc` and `cond`, and trains each from
π0.5-base:

```bash gpu
qvla realrobot pretrain
```

<!-- ci
`--print` writes the run specs and prints the training commands:

```bash ci
qvla realrobot pretrain --print
```

```text output
OK    pi05.pretrain.realrobot_bc -> …
OK    pi05.pretrain.realrobot_cond -> …
qvla train …/pi05.pretrain.realrobot_cond.toml --host …
```
-->

Stage 1 is 20,000 steps at batch 64, sharded over 8 GPUs: LoRA on both experts with the vision tower
frozen, centre-cropped 224 images, an action horizon of 15, and photometric augmentation of the rendered
frames. A run writes its checkpoints under `<runs>/<run id>/train/`.

## 6. Record and convert your demonstrations

Record demonstrations with DROID's data collection on your robot: the two exterior cameras and the wrist
camera, the joint positions and the gripper, and actions in DROID's joint-velocity action space at 15 Hz.
Annotate each episode's instruction in `aggregated-annotations-030724.json` in the recordings directory,
`{"<id>": {"language_instruction1": "<instruction>"}}` with the `<id>` of each episode's
`metadata_<id>.json`. This converts them with openpi's DROID converter and writes the fine-tuning set
(one task per instruction) and its norm statistics:

```bash gpu
qvla realrobot demos <your DROID recordings>
```

<!-- ci
`--print` prints the converter's command; `--converted` starts from a dataset the converter wrote, here a
three-episode stand-in:

```bash ci
qvla realrobot demos roots/droid_recording --print
python -m tests.docs.fixtures droid-recording converted
qvla realrobot demos --converted converted --replace
```

```text output
…/convert_droid_data_to_lerobot.py local/fr3_demonstrations …
3 episodes / 62 frames of 2 instructions -> …/demos
norm statistics of demos (fr3_droid) -> …
```
-->

Your demonstrations are the fine-tuning set.

## 7. Fine-tune

This writes the three fine-tuning run specs and trains each: `bc` and `cond` start from their stage-1 runs'
checkpoints, `only` from π0.5-base with no pretraining, the control:

```bash gpu
qvla realrobot finetune
```

<!-- ci
```bash ci
qvla realrobot finetune --print
```

```text output
OK    pi05.finetune.realrobot_bc -> …
OK    pi05.finetune.realrobot_cond -> …
OK    pi05.finetune.realrobot_only -> …
qvla train …/pi05.finetune.realrobot_only.toml --host …
```
-->

Stage 2 is 16,000 steps at batch 64 from the stage-1 checkpoint, the vision tower trainable. `cond` appends
`"\nQuality: 5"` to every prompt (with tag dropout during fine-tuning) and is served with that
tag.

## 8. Serve

This serves a fine-tuned run's final checkpoint on port 8000 and checks the server against the run and
your demonstrations:

```bash gpu
qvla realrobot serve cond
```

<!-- ci
`--print` writes the serve config and the drive kit, and prints the server's command:

```bash ci
qvla realrobot serve cond --print
```

```text output
serve config of pi05.finetune.realrobot_cond -> …/serve.json
drive kit -> …/pi05.finetune.realrobot_cond/drive (an optional client for a DROID control stack: docs Real robot > Connect your robot)
…serve_policy.py --port=8000 --quality-tag=5 …
```
-->

It prints `ALL PASS` and serves until Ctrl+C. `bc` and `only` are served the same way.

## Connect your robot

The server speaks the [policy-server protocol](../reference/policy-server.md) with the FR3 runs' serving
contract. Its handshake states `action_units: "droid_normalized_joint_velocity_command"`,
`action_fps: 15`, `action_horizon: 15` and `action_dim_out: 8`. Your robot's client sends, per request:

| key | content |
|---|---|
| `base_0_rgb` | the exterior camera's RGB frame, uint8: recorded at 1280 × 720 and resized to 320 × 180 with Pillow's bicubic resize, as openpi's DROID converter stores the training frames (the server crops the centre square and resizes it to 224 × 224 itself) |
| `left_wrist_0_rgb` | the wrist camera's frame, resized the same way |
| `state` | float32 `(8,)`: the joint positions `q0` … `q6` in rad, then the gripper position in [0, 1] (0 open) |
| `prompt` | the task instruction; the `cond` run's server appends its quality tag itself |

The reply's `actions` hold 15 rows of 8 columns. Columns 0–6 are DROID's normalised joint-velocity command
in [−1, 1] (the controller's joint target is `q + 0.2·a`) and column 7 the absolute gripper position
command in [0, 1] (0 open). The client executes the first 8 rows at 15 Hz, one per step, then sends the
next observation.

The DROID drive kit `qvla realrobot serve` writes beside the run, `<runs>/<run id>/drive/`, is an optional
client for a DROID control stack (`--replace-kit` refreshes it after you update the checkout).
