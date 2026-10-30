# Write a quality method

The **pretraining data** is what every VLA of the paper pretrains on: 5,000 simulated Franka
demonstrations of 731 tasks, published as `axisrobotics/qvla-v1-corpus`. Each task is one LeRobot
dataset, with the DROID 8-D state and action (seven joints and the gripper; `d8` in identifiers).
**Quality labels** say how well each part of a demonstration went, computed from each attempt's
recorded simulation state rather than from the video: the task-progress reward on every frame, the
phase curve of the approach-aware reward, training weights from the reward's change over a
10-frame window, and quality tags from 1 to 5 that a conditioned prompt carries. The published labels
are under `signals/v1/` of the pretraining data ([quality labels](../reference/data.md#quality-labels)).

A **frame index** gives every frame of a dataset a fixed position, so that quality labels, weights and
training plans all address the same frames, and says which frames a run may draw: its trainable
frames. The pretraining data's index drops idle stretches and the last 10 frames of every kept
stretch: of its 1,906,469 frames, 1,319,784 are trainable. **Compiled labels** are a quality method's
training decisions, saved before training: which frames a run draws, how much each counts and what
each prompt says. A quality method trains nothing itself; it writes compiled labels, one directory
with a manifest and plain `.npy` arrays, over one frame index
([compiled labels](../reference/data.md#compiled-labels)).

This tutorial writes a new method from the shipped blocks (keep the top share of trainable frames by a
per-frame score) and compiles it on the paper's pretraining data. It is the only part of training you
write: [Pretrain](pretrain.md) and [Fine-tune and evaluate](fine-tune-and-evaluate.md) then run it
through the benchmark's protocol, one command each.

**What you get.** Your method's compiled labels and those of its size-matched control, stored under
their names (`example.keep_top`, `example.keep_top.control`), which the next step names.

**What you need.** The `method` extra and a `qvla.toml` in the directory you run from
([Install](../install.md) writes it), a CPU, and about 165 MB of index, label and metadata files.

| step | command | what it produces |
|---|---|---|
| [the method](#the-method) | `experiments/examples/keep_top.py` | a composition of a scorer, a selector and a planner |
| [1. get the labels](#1-get-the-pretraining-datas-labels) | `qvla data get corpus --include …` | the frame index, the paper's training weights and the episode metadata under `<corpus>/qvla-v1-corpus/` |
| [2. compile it](#2-compile-it) | `python experiments/examples/keep_top.py --scores …` | `example.keep_top` and `example.keep_top.control` |
| 3. train it | [Pretrain](pretrain.md) | your method's pretraining run |

## The method

The example reads one value per frame from a scores file, keeps the top share of the trainable frames
and draws them with a schedule. `experiments/examples/keep_top.py` composes three blocks: a scorer that
passes an imported scores file through, the `TopFraction` selector and the schedule planner.

```python file=experiments/examples/keep_top.py
@dataclasses.dataclass(frozen=True)
class ImportedValues:
    """Scorer: the per-frame values of an imported scores file, unchanged."""

    def score(self, labels, frames):
        return labels.values()


DEFAULT_NAME = "example.keep_top"


def keep_top(keep: float, name: str = DEFAULT_NAME) -> Compose:
    """Keep the top `keep` of trainable frames by the imported value; draw them with the schedule planner.
    `name` is the method the compiled labels are recorded under; the control's is `<name>.control`."""
    return Compose(method=name, version=1, params={"keep": keep}, scorer=ImportedValues(),
                   selector=TopFraction(keep), planner=SchedulePlanner("drop_top"))
```

Compiling the method also compiles its size-matched control, a random selection of as many frames,
so a result can be told apart from training on fewer frames, and checks the rules the blocks compose
by ([how results are verified](../reference/verification.md#quality-labels-and-compiled-labels)).
Other blocks change other things:
- a weighter sets a loss weight per frame (the paper's weighting methods);
- a conditioner adds a quality tag to the prompt (the paper's conditioning methods).

The built-in methods are in `axis_qvla/signals/methods/`, one module per family; the format they write is
in [compiled labels](../reference/data.md#compiled-labels).

## 1. Get the pretraining data's labels

The example scores frames by the paper's approach-aware training weights. If the directory you run
from has no `qvla.toml` yet, `qvla init` writes one:

```bash data
qvla init
```

<!-- ci
```bash ci
qvla init --root roots
```
-->

Fetch the training weights with the frame index and the episode metadata, about 165 MB, into
`<corpus>/qvla-v1-corpus/`, where the pretraining data goes:

```bash data timeout=1800
qvla data get corpus --include 'index/*' --include 'signals/v1/approach_aware/chunk/*' --include 'episodes/*/meta/episodes/*' --include 'episodes/*/meta/tasks.parquet'
```

<!-- ci
A toy scores file stands in:

```bash ci
python -m tests.docs.fixtures toy-scores toy_scores.json
```

```text output
OK    toy-scores -> toy_scores.json
```
-->

## 2. Compile it

```bash data timeout=1800
python experiments/examples/keep_top.py --scores srv/corpus/qvla-v1-corpus/signals/v1/approach_aware/chunk/awr_weights.json
```

```text output
OK    example.keep_top  plan (20605, 64)
OK    example.keep_top.control  plan (20605, 64)
```

<!-- ci
The same on the 23-frame toy frame index, in seconds:

```bash ci
python experiments/examples/keep_top.py --scores toy_scores.json --toy --keep 0.3 --steps 6 --batch 2
```

```text output
OK    example.keep_top  plan (6, 2)
OK    example.keep_top.control  plan (6, 2)
```

A per-seed compile under its own name, as [Pretrain](pretrain.md) does for another training replicate:

```bash ci
python experiments/examples/keep_top.py --scores toy_scores.json --toy --keep 0.3 --steps 6 --batch 2 --seed 43 --name example.keep_top.s43
```

```text output
OK    example.keep_top.s43  plan (6, 2)
OK    example.keep_top.s43.control  plan (6, 2)
```
-->

It compiles at the paper's π0.5 budget (20,605 steps of 64, keeping half of the trainable frames) and
stores both under `<cache>/bundles`, where training looks for them, by the names it prints. It reads
the pretraining data from `<corpus>/qvla-v1-corpus/`; `srv/corpus` is the `<corpus>` root `qvla init`
writes by default, so with another root, give the scores file's path under yours.

A scores file is a JSON map from each episode (`task_<id>--<episode index>`) to one value per frame,
the format of the pretraining data's `signals/v1/**/awr_weights.json`.
[Relabel with a new reward](relabel-with-a-new-reward.md) writes one.

Next: [Pretrain](pretrain.md).
