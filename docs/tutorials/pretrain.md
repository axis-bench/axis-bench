# Pretrain

A **run spec** is one TOML file that says what one training run is; in identifiers a run is an `arm`,
the name it was published under. Its tables are `[arm]` (the run's id, backbone, stage and track),
`[init]` (the checkpoint it starts from: a published checkpoint, a base preset `base:<name>` or a prior
run's own checkpoint `run:<run id>`), `[data]` (the frame index it trains on), `[signal]` (the compiled
labels it binds), `[norm]` (the normalisation statistics) and `[recipe]` (a preset of `conf/recipes/`,
the seed and the overrides the run needs).

For your method, `qvla train pretrain` writes it: the run spec of the paper's
pretraining for a backbone, at that backbone's paper budget, with your [compiled
labels](write-a-quality-method.md) named by name, under `<runs>/<run id>/`, and trains it there in
the backbone's environment. The roots (`<corpus>`, `<cache>`, `<runs>`) come from `qvla.toml` in the
directory you run it from.

**What you get.** Your method's pretraining run, `<runs>/pi05.pretrain.example.keep_top/`, and the
command that fine-tunes it.

**What you need.** GPUs, the backbone's environment ([install](../install.md#environments)), the
pretraining data and your method's compiled labels ([Write a quality method](write-a-quality-method.md)).

| step | command | what it produces |
|---|---|---|
| [1. get the pretraining data](#1-get-the-pretraining-data) | `qvla data get corpus` | the pretraining data under `<corpus>/qvla-v1-corpus/`, about 36 GB |
| [2. pretrain](#2-pretrain) | `qvla train pretrain --labels example.keep_top --backbone pi05` | the run `pi05.pretrain.example.keep_top` |
| 3. fine-tune and evaluate it | [Fine-tune and evaluate](fine-tune-and-evaluate.md) | its all-24 success rate |

## 1. Get the pretraining data

About 36 GB, nearly all of it the per-task episodes that π0.5 reads and the merged LeRobot copy
that SmolVLA and X-VLA read:

```bash gpu
qvla data get corpus
```

<!-- ci
```bash ci
qvla init --root roots
```
-->

## 2. Pretrain

```bash gpu
qvla train pretrain --labels example.keep_top --backbone pi05
```

<!-- ci
`--print` writes the run spec and prints the command that trains it, reading no data:

```bash ci
qvla train pretrain --labels example.keep_top --backbone pi05 --print
```

```text output
OK    wrote …/roots/runs/pi05.pretrain.example.keep_top/spec.toml
OK    pi05.pretrain.example.keep_top: …/roots/runs/pi05.pretrain.example.keep_top/spec.toml
qvla train …/roots/runs/pi05.pretrain.example.keep_top/spec.toml --host …/qvla.toml
NEXT  qvla train finetune --from pi05.pretrain.example.keep_top
```

Another training replicate:

```bash ci
qvla train pretrain --labels example.keep_top --backbone pi05 --seed 43 --print
```

```text output
NEXT  qvla train finetune --from pi05.pretrain.example.keep_top.s43
```

A replicate from labels compiled under a per-seed name, under the method's run name (here `mine`, so
that it does not meet the run the block above wrote):

```bash ci
qvla train pretrain --labels mine.s43 --backbone pi05 --run mine --seed 43 --print
```

```text output
NEXT  qvla train finetune --from pi05.pretrain.mine.s43
```
-->

It checks the run's inputs before any GPU time, fetches the backbone's base weights if they are not
there yet, and trains. If the backbone's environment is not built, it stops and names the command
that builds it (`qvla env build pi05`). When it finishes it prints the next command,
`qvla train finetune --from pi05.pretrain.example.keep_top`: the run's id is
`<backbone>.pretrain.<labels name>` (`--run NAME` gives it another name).

- `--backbone smolvla` or `--backbone xvla` trains the same labels on the other backbones. That
  holds for the example, which only selects frames. A method that sets loss weights or quality tags
  compiles once more for those two backbones, which weigh or tag the frames their own sampler draws,
  under another name, as the paper's weighting methods do (`axis_qvla/signals/methods/awr.py`;
  [compiled labels](../reference/data.md#compiled-labels)); pass that name to `--labels`. Training
  checks that the backbone can train the labels and names what it lacks.
- `--seed N` pretrains another training replicate. The default is the paper's first, 42 for π0.5 and 0
  for SmolVLA and X-VLA; another seed gives the run id `.s<N>`
  (`--seed 43`: `pi05.pretrain.example.keep_top.s43`). A claim about your method needs three,
  paired with the paper's ([Fine-tune and evaluate](fine-tune-and-evaluate.md#4-compare-with-plain-bc)).
  A method whose labels involve randomness, like the size-matched control, compiles its labels once
  per training seed under a per-seed name,
  `python experiments/examples/keep_top.py --scores … --seed 43 --name example.keep_top.s43`, and
  pretrains them under the method's run name,
  `qvla train pretrain --labels example.keep_top.s43 --backbone pi05 --run example.keep_top --seed 43`,
  which makes `pi05.pretrain.example.keep_top.s43`. The compile writes the control alongside, as
  `example.keep_top.s43.control`, which pretrains the same way with `--run example.keep_top.control`.
- `--print` writes the run spec and prints the one command that trains it, for a SLURM job, without
  training.
- Pretrain the size-matched control the same way, so your result can be contrasted with it:
  `qvla train pretrain --labels example.keep_top.control --backbone pi05` makes the run
  `pi05.pretrain.example.keep_top.control`, which [Fine-tune and evaluate](fine-tune-and-evaluate.md)
  takes through the same two commands as yours.

Next: [Fine-tune and evaluate](fine-tune-and-evaluate.md).
