# Getting the data

Download the v1.0 data and checkpoints from the Hugging Face organisation `axisrobotics`, at tag
`v1.0`. The base models are the exception: those come from their publishers
([base weights](#base-weights)).

| your goal | what to get |
|---|---|
| reproduce the paper's numbers | `results` |
| evaluate a policy | `bench`; for a paper checkpoint, also `checkpoints/<run>` |
| train with your own quality method | parts of `corpus`, and `base-weights` |
| relabel with a new reward | `traces-sample` |
| run it on a real robot | `realworld`, and `base-weights --include pi05_base` |

```
qvla data get bench                                # finetune demos, tasks, scene assets, layouts
qvla data get results                              # per-seed rollout logs, cells.json, expected/
qvla data get corpus --include 'index/*' --include 'signals/v1/*' --include 'norm_stats/*'
qvla data get traces-sample                        # the raw traces of 21 tasks, for relabelling
qvla data get realworld                            # the simulation demonstrations the real-world line pretrains on: recordings and scenes, to re-render
qvla data get checkpoints/<run> --with-init        # a paper checkpoint and the one it started from
qvla data get base-weights --include pi05_droid       # pi0.5's base model and tokenizer, from their publisher
qvla data verify [--local <dir> --repo axisrobotics/<repo>]
```

`qvla data get` downloads the tagged files and checks each one
([how results are verified](verification.md#downloads)); an interrupted `get` resumes.
`checkpoints/<x>` takes a result name, a run or a checkpoint id from `experiments/paper/cells.json`.

## Where files go

`--dest DIR` fills `DIR` with the repository's own layout. Without `--dest`, `get` uses the roots of
`qvla.toml` in the working directory (`--host <file>.toml` picks another file) and puts each
repository where `qvla train` looks for it (`axis_qvla/data/layout.py`):

| what | local directory |
|---|---|
| `qvla-v1-corpus`, `qvla-v1-bench`, `qvla-v1-results`, `qvla-v1-realworld` | `<corpus>/<repository>/`, files at their repository paths |
| the per-task pretraining data | `<corpus>/qvla-v1-corpus/episodes/` |
| the normalisation statistics: fine-tuning set, pretraining data | `<corpus>/qvla-v1-bench/norm_stats/heldout24_qual_v2/`, `<corpus>/qvla-v1-corpus/norm_stats/libero5k_v2_d8/` |
| a checkpoint (`qvla-v1-pi05`, `-smolvla`, `-xvla`) | `<cache>/checkpoints/<repository>/<path>/` |
| the base weights | `<cache>/checkpoints/qvla-v1-base-weights/`: openpi's download cache under `openpi/`, a Hugging Face cache under `hf/` |
| anything computed here: compiled labels built on this machine, the merged LeRobot copy of the fine-tuning demos | `<cache>/bundles/<id>/`, `<cache>/derived/<name>/` |

`qvla bench eval` reads the benchmark where `qvla data get bench` puts it: `--asset-dir` defaults to
`<corpus>/qvla-v1-bench/assets` and `--task-data-root` to `<corpus>/qvla-v1-bench/tasks`. After
`qvla data get checkpoints/<run>`, a checkpoint is `<cache>/checkpoints/<repository>/<path>` (the
`<local copy of ...>` placeholders of a `cells.json` `eval_command`).

### The merged LeRobot copy

A SmolVLA or X-VLA run reads its frames from one LeRobot dataset merged from the per-task data, under
`<cache>/derived/<name>/`. `qvla train` builds it with the LeRobot fork's merge the first time a run needs
it, before the run trains, and binding refuses a copy that is not the one the run's frame index names. To
rebuild one, remove it and run `qvla train` again.

`<corpus>` and `<cache>` are the roots of `qvla.toml`; `qvla init` writes `qvla.toml` with every root
under one directory ([install](../install.md)). What your machine computes goes under its own roots
(`<cache>`, `<runs>`, `<r1>`), beside the downloaded repositories. The roots and environments `qvla.toml` names:

| root or venv | what it is | who reads it |
|---|---|---|
| `corpus` | the downloaded repositories | `qvla data get`, `qvla train` |
| `cache` | checkpoints, base weights and everything computed on this machine | `qvla data get`, `qvla train` |
| `runs` | training runs, `<runs>/<run id>/` | `qvla train` |
| `r1` | the evaluation root: each run's summary and queue | `bench eval` |
| `rollout_logs` | every evaluation's per-seed rollout logs, for every backend; `qvla init` writes `<r1>/rollout_logs` | `bench eval` |
| `bench` | this checkout | `bench eval` |
| `eval` | the evaluation's asset cache | `bench eval` |
| `[venv] eval` | the evaluation environment, `<checkout>/.venv-eval`, whose interpreter runs the driver and the rollout workers | `bench eval`, `bench check-server`, `paper reeval` |
| `openpi` | the openpi checkout, `third_party/openpi` | the π0.5 reference server |
| `[venv] online` | the π0.5 environment, `third_party/openpi/.venv` | the π0.5 reference server |
| `smolvla_bench` | the checkout whose LeRobot server a SmolVLA or X-VLA evaluation starts | `bench eval --backend smolvla` or `xvla` |
| `smolvla_venv` | the LeRobot environment, `third_party/lerobot/.venv` (its `bin/python` serves) | `bench eval --backend smolvla` or `xvla` |
| `hf_cache` | the `HF_HOME` holding the LeRobot base models, read offline | `bench eval --backend smolvla` or `xvla` |

## The benchmark inputs

`conf/benchmarks/axis-heldout-v1/franka_heldout24.json` declares the benchmark: tasks, seeds,
conventions and inputs, each pinned as `repo@v1.0`. Nothing is fetched implicitly. `bench eval` reads
two inputs:

- a local copy of `axisrobotics/qvla-v1-bench@v1.0` (`<bench>`): its `assets/` holds the scene meshes
  and textures, and its `tasks/` the per-task `task.json` and `checker_config.json`. `qvla data get
  bench` puts it at `<corpus>/qvla-v1-bench/`, or in the directory `--dest` names;
- the renderer, the submodule `third_party/axis-render` (`git submodule update --init
  third_party/axis-render`), at `--render-root`, by default `third_party/axis-render`. `bench eval`
  puts `<render-root>/src` on the rollout workers' `PYTHONPATH`.

The renderer reads the Polyhaven table-texture bank from its own tree,
`third_party/axis-render/assets/table_textures/polyhaven/downloaded_256/`. `bench eval` links each file of
`<bench>/assets/textures/polyhaven/downloaded_256/` there before any server starts; a file already there,
copy or link, is left as it is. The renderer's own textures and robot meshes ship with
`third_party/axis-render`.

## Base weights

The base models a run starts from are downloaded from their publishers:
`qvla data get base-weights` fetches the π0.5 bases (`pi05_droid`, `pi05_base`) and the PaliGemma
tokenizer from Google Cloud Storage (`gs://openpi-assets`, `gs://big_vision`), and the LeRobot bases
(`smolvlm2_500m`, `smolvla_base` with the SmolVLM2 weights it loads, `xvla_base` with its tokenizer
`facebook/bart-large`) from the Hugging Face Hub ([how results are verified](verification.md#downloads)).
An interrupted `get` resumes. `--include <base>` (repeatable) fetches only the bases
named in `conf/recipes/bases.toml`; without it, all five, about 31 GB. The π0.5 bases are under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms); the LeRobot bases are Apache-2.0.

## From a local snapshot

`qvla data get results --dest <dir>` is the verified way to get the results repository.
`qvla paper tables` also runs from any local snapshot of it, for example one downloaded with the
Hugging Face CLI or copied from a machine with network access:

```
hf download axisrobotics/qvla-v1-results --repo-type dataset --revision v1.0 --local-dir <dir>
qvla paper tables --results <dir> --check
```

What `--check` compares: [how results are verified](verification.md#the-papers-tables).

## Quality labels

The pretraining data, `axisrobotics/qvla-v1-corpus`, holds 5,000 simulated Franka demonstrations of
731 tasks, one LeRobot dataset per task, with the DROID 8-D state and action (seven joints and the
gripper; `d8` in identifiers). Of its 1,906,469 frames, 1,319,784 are trainable: idle stretches and the
last 10 frames of every kept stretch are left out.

Quality labels say how well each part of a demonstration went. They are computed from each attempt's
recorded simulation state (its trace), not from the video, for two rewards, object-centric and
approach-aware:

- the reward, on every frame: task progress, plus a term for how close the hand is to the object;
- the phase curve, for the approach-aware reward: it splits an episode at the start of stable contact
  into an approach phase and a manipulation phase;
- training weights: the AWR weight `w = min(exp(10 Δ), 2)`, with `Δ` the reward's change across a
  10-frame window;
- quality tags: a quintile bin from 1 to 5, which a conditioned prompt carries as
  `"\nQuality: q"`.

The published labels are under `signals/v1/` of the pretraining data, one directory per reward and
level, and the frame indexes they are computed over under `index/`:

| path | holds |
|---|---|
| `signals/v1/<reward>/<level>/awr_weights.json` | the training weight of every frame (`<reward>`: `object_centric` or `approach_aware`; `<level>`: `episode` or `chunk`) |
| `signals/v1/<reward>/<level>/tags.npz` | the quality tags |
| `signals/v1/<reward>/<level>/schedule_drop_top50.npz` | the filtering schedule that keeps the top-ranked half of the frames |
| `signals/v1/lerobot/` | the same labels as the files the SmolVLA and X-VLA runs read |
| `index/` | the frame indexes: the episode ranges and roots of the pretraining data, and the scaling subsets |

`axis_qvla/labels/` recomputes the per-frame labels from the raw recordings; the
[traces sample](#the-traces-sample) is enough to run it end to end.

### The label constants

The numbers a quality method builds on, with the part of the labels each belongs to and the module
attribute that holds it. A conditioned run drops the whole condition from a row with probability
`DROP_WHOLE` and, independently, the quality field with probability `DROP_COMPONENT`, so a share `TAGGED_FRACTION`
of the trainable rows carries a tag.

<!-- label-constants:begin -->
| constant | value | part | meaning | where |
|---|---|---|---|---|
| `W_PROGRESS` | 0.8 | reward | weight of task progress in the reward | `axis_qvla.labels.reward.W_PLACE_V2` |
| `W_CONTACT` | 0.2 | reward | weight of the hand-object contact shaping term | `axis_qvla.labels.reward.W_CONTACT_V2` |
| `CONTACT_TANH_K` | 5.0 | reward | contact shaping is 1 - tanh(k * hand-object distance in m) | `axis_qvla.labels.reward.CONTACT_TANH_K` |
| `PHASE_LAMBDA` | 0.5 | phase curve | height of the approach band | `axis_qvla.labels.phase.LAMBDA` |
| `PHASE_BONUS` | 0.1 | phase curve | step at the start of the manipulation band | `axis_qvla.labels.phase.BONUS` |
| `CONTACT_MIN_RUN` | 3 | phase curve | frames of continuous target contact that make stable contact | `axis_qvla.labels.phase.MIN_RUN` |
| `EDGE_FRAMES` | 5 | phase curve | contact within this many frames of either end is not a phase boundary | `axis_qvla.labels.phase.EDGE_FRAMES` |
| `MIN_APPROACH_M` | 0.02 | phase curve | an approach shorter than this (m) is no approach | `axis_qvla.labels.phase.MIN_APPROACH_M` |
| `TARGET_CONTACT` | `"contact_target_finger"`, `"contact_target_eef"` | phase curve | target contact by a finger OR the hand/palm | `axis_qvla.labels.phase.TARGET_CONTACT` |
| `AWR_TAU` | 10.0 | training weights | w = min(exp(tau * Delta), cap) | `axis_qvla.signals.blocks.awr.DEFAULT_TAU` |
| `AWR_CAP` | 2.0 | training weights | upper bound of w | `axis_qvla.signals.blocks.awr.DEFAULT_CAP` |
| `AWR_HORIZON` | 10 | training weights | Delta_t = V(min(t + H - 1, T - 1)) - V(t) | `axis_qvla.signals.blocks.awr.DEFAULT_HORIZON` |
| `MIN_WINDOW_FRAMES` | 3 | training weights | a phase-clamped window shorter than this has Delta = 0; longer ones are rescaled by (H - 1) / span | `axis_qvla.signals.methods.awr_chunk.MIN_WINDOW_FRAMES` |
| `N_BINS` | 5 | quality tags | quality tags are global quintiles | `axis_qvla.signals.blocks.schedule.N_BINS` |
| `NEUTRAL_Q` | 3 | quality tags | tag of an episode whose weights are all 1 (unscored) | `axis_qvla.signals.blocks.schedule.NEUTRAL_Q` |
| `DROP_WHOLE` | 0.15 | conditioning | probability the whole condition is dropped from a row | `axis_qvla.signals.blocks.tags.DROP_WHOLE` |
| `DROP_COMPONENT` | 0.05 | conditioning | probability the quality field is dropped, independently | `axis_qvla.signals.blocks.tags.DROP_COMPONENT` |
| `TAGGED_FRACTION` | 0.8075 | conditioning | share of trainable rows that carry a tag (80.75%) | `axis_qvla.signals.blocks.tags.TAGGED_FRACTION` |
| `PROMPT_MARKER` | `"\nQuality: "` | conditioning | text appended to the instruction before the tag | `axis_qvla.signals.methods.cfg.PROMPT_MARKER` |
<!-- label-constants:end -->

### Rules the equations leave implicit

- The phase curve's stable contact is at least 3 consecutive frames of target contact by a finger or
  the palm. It splits an episode by when stable contact starts and how far the hand approached
  ([phases](#phases)).
- Before `w = min(exp(10 Δ), 2)`, approach-aware windows are clamped to their phase: a clamped window
  shorter than 3 frames has `Δ = 0`, and a longer one is rescaled by `(H − 1) / span`. Each phase's
  `Δ` is then scaled so that both phases reach the cap at the pooled rate.
- The quality tags evaluate `1[Δ > 0]` as `w > 1` on the float32 weight (`Δ > 2^-24 / τ`). Keep the weights
  float32: a tiny positive advantage lands on exactly 1.0 and does not count.
- Ties in every ranking break by position in the pretraining data, so a rebuild gives the same bins.
- An episode whose weights are all 1 (unscored) gets the neutral tag 3.

### Phases

With `T` the episode's length:

| stable contact starts | approach | phases |
|---|---|---|
| in [5, T − 5] | at least 0.02 m before it | two bands: approach, then manipulation |
| in [5, T − 5] | shorter (the hand began engaged) | all manipulation from frame 0 |
| before frame 5 | any | all manipulation from frame 0 |
| never, or after T − 5 | the hand closed in by at least 0.02 m | all approach, normalised by the closest approach |
| never, or after T − 5 | less | unscored (weight 1 everywhere) |

## Compiled labels

A quality method writes one compiled-labels directory: `manifest.json` and one `.npy` file per array.

| field | meaning |
|---|---|
| `plan` | int64 `[steps, global_batch]`: the frames every optimizer step draws (always present) |
| `weights` | float32 `[n_rows]` loss weights, NaN on frames no plan may draw; with `reducer` `global_ratio` (openpi) or `per_rank_ratio` (LeRobot); absent means `mean` |
| `cond`, `vocab` | uint8 `[steps, global_batch, fields]` codes for the drawn frames (0 = field dropped); `vocab` maps a code to the exact text appended to the instruction (`"\nQuality: 5"`) and lists the pretraining instructions |
| `serve` | serve conditions: fields, the unconditional fields when guided, and `beta` |
| `rowspace` | the frame index (episode table, trainable spans, the index files it writes back) |
| `requires` | consumer capabilities: `plan`, `loss_weights`, `cond_prompt`, `guidance` |
| `program`, `program_sha`, `inputs`, `diagnostics` | the method's declared identity, the digests of what it read, summary numbers |
| `legacy` | the label files a training fork reads: for the paper's methods the files their runs read; for a composed method the LeRobot files that train it |

```bash
qvla labels export <compiled labels dir> --format lerobot-legacy --out <dir>    # files a LeRobot recipe reads
qvla labels check <compiled labels root>                               # verify and re-export every listed file
qvla labels build-paper --pretraining-data <qvla-v1-corpus> --episodes <qvla-v1-corpus>/episodes --out <dir>
```

`build-paper` rebuilds the paper's compiled labels from the published per-task pretraining data,
`qvla-v1-corpus/episodes`. Compile into a directory of your own, such as `<cache>/bundles`; the
downloaded `qvla-v1-corpus` stays as published. What these commands check:
[how results are verified](verification.md#quality-labels-and-compiled-labels).

## The traces sample

`qvla-v1-corpus/traces_sample/` (`qvla data get traces-sample`) holds the raw recordings behind 21
pretraining tasks, enough to run the label pipeline end to end. It holds one task per instruction verb
family (the instruction's first word, lower-cased; the task with the fewest pretraining episodes, ties
to the smaller id), plus the tasks the renderer tests read (`required` in the manifest), and every
pretraining episode of each of those tasks.

```
traces_sample/
  traces/task_<id>_attempt_<aid>.jsonl             one recorded trace (5 Hz state events) per episode
  goals_per_attempt/task_<id>_attempt_<aid>.json   the task checkers' goal record of that attempt
  data/task_<id>/checker_config.json               the task's checker configuration
  corpus/task_<id>/{meta,data}/...                 the task's published corpus files (no video)
  manifest.json                                    every file; the tasks, families and episodes
```

Keep `traces/` beside `data/`: a position-delta goal without per-axis bounds takes them from
`data/task_<id>/checker_config.json` next to the trace's directory, and a trace without that file
raises `axis_qvla.labels.reward.AxisBoundsError`.

To label one episode, take a task and an episode from the manifest:

```python
import json, pathlib
from axis_qvla.labels.pipeline import label_episode, trace_states

sample = pathlib.Path("qvla-v1-corpus/traces_sample")
task = json.loads((sample / "manifest.json").read_text())["tasks"][0]
ep = task["episodes"][0]
name = f"task_{task['task_id']}_attempt_{ep['attempt_id']}"
t = sample / "traces" / f"{name}.jsonl"
goals = json.loads((sample / "goals_per_attempt" / f"{name}.json").read_text())
cols = label_episode(trace_states(t), goals, n_rows=ep["length"], trace_path=t)
# cols["reward"], cols["reward_valid"], cols["success"], cols["is_anchor"], cols["observation.body_pose"], ...
```

The result is that episode's rows in `corpus/task_<id>/data/`. The contact flags come from the task's
scene, passed as `contact_model=axis_qvla.labels.contacts.ContactModel(<scene.xml>)`; the pretraining data's
own per-frame labels hold them for every episode.

## The real-world data

`qvla data get realworld` fetches `axisrobotics/qvla-v1-realworld`, the simulation demonstrations the
real-world line pretrains on (42.29 GB in 70,784 files: 663 tasks, 19,250 episodes recorded in 8,294
scenes; CC BY 4.0; the Panda meshes are MuJoCo Menagerie's, Apache-2.0):

| path | holds |
|---|---|
| `manifest.json` | every task: its instruction, scene variants, mesh layouts, and each episode's attempt id, its scene variant and `variant_from`: `recorded` or `resolver`, how the episode's scene variant was obtained from its attempt record |
| `tasks/task_T/scenes/variant_I.xml` | a scene the task's episodes were recorded in |
| `tasks/task_T/scenes/attempt_A.json` | the episode as the renderer reads it: the recorded packed trajectory |
| `tasks/task_T/traces/task_T_attempt_A.jsonl` | the recorded trace: every joint and body per recorded row |
| `tasks/task_T/goals_per_attempt/task_T_attempt_A.json` | the task checkers' goal record |
| `tasks/task_T/data/task_T/checker_config.json` | the task's checker configuration |
| `tabletopgen/L.tar` | the object meshes and textures of one layout |
| `robots/franka_emika_panda/` | the Panda meshes the scenes name |

T is a task id, A an attempt id, I a scene variant and L a mesh layout.

The real-robot runs in the paper fine-tune on a public Hugging Face dataset, downloaded from its
publisher: `ZhixuLi/tasl-fr3-10task-pbc-v2`, 392 segments (62,680 frames) of 10 Franka FR3 tasks as
LeRobot v2.1 with centre-cropped 224 × 224 frames. Its segments are the ones the idle filter keeps from
the 250 teleoperated recordings (66,463 frames, 15 fps, stored as video) of
`Litian2002/tasl-fr3-10task-250ep-video`. `experiments/realrobot/line.toml` and
`experiments/realrobot/fr3_data.toml` name both by these ids:

```
hf download ZhixuLi/tasl-fr3-10task-pbc-v2 --repo-type dataset --local-dir <dir>
hf download Litian2002/tasl-fr3-10task-250ep-video --repo-type dataset --local-dir <dir>
```

On your own robot, your demonstrations are the fine-tuning set
([real-world tutorial](../tutorials/real-world-your-workspace.md#6-record-and-convert-your-demonstrations)).

## The checksum manifest

`conf/release/v1.0.lock.json` lists every published file of every repository with its size, sha256
and git blob id at tag `v1.0`. `qvla data get` and `qvla data verify` read it.
How every download is checked: [how results are verified](verification.md#downloads).
