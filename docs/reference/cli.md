# CLI

The complete index of `qvla` commands, including the tools a `qvla` group forwards to, generated
from their argument parsers. Each entry gives the command's one-line help and its usage; run a
command with `--help` for what each option means. A forwarded tool's usage line names its module
(`python -m axis_qvla.realrobot.render source ...`); `qvla realrobot render source ...` runs the same
command. To choose a command for a task, start from the [tutorials](../index.md).

## `qvla bench`

the held-out benchmark: check the declaration, evaluate a checkpoint, read scores

### `qvla bench check`

print the benchmark identity; exit 1 on any inconsistency

```text
qvla bench check [-h] [--benchmark BENCHMARK] [--pretraining-prompts FILE]
```

### `qvla bench versions`

list the benchmark versions under conf/benchmarks/ and their hashes

```text
qvla bench versions [-h]
```

### `qvla bench eval`

evaluate ONE checkpoint on the benchmark (the campaign driver, fully composed)

```text
qvla bench eval [-h] [--checkpoint CHECKPOINT] [--checkpoint-id ID] [--from-run RUN_ID] [--verify-checkpoint] [--config CONFIG] [--policy HOST:PORT] [--run RUN] [--serving {bare,conditioned}] [--asset-dir ASSET_DIR] [--render-root RENDER_ROOT] [--task-data-root TASK_DATA_ROOT] [--guidance-scale GUIDANCE_SCALE] [--backend {openpi,smolvla,xvla}] [--benchmark BENCHMARK] [--host HOST] [--gpus GPUS] [--workers-per-gpu WORKERS_PER_GPU] [--batch BATCH] [--mem-fraction MEM_FRACTION] [--dry-run] [--tally-only] [--seed-set {test,dev}] [--off-protocol] [--ft-seed SEED] [--pretrain-seed SEED] [--print]
```

### `qvla bench score`

score results files (all-24 and per-task, with seed-bootstrap intervals), or list the run summaries under a campaign root

```text
qvla bench score [-h] [--run-root RUN_ROOT] [--run RUN] [--benchmark BENCHMARK] [--view {all24,informative_20,usable_12}] [--bootstrap B] [--per-task] [--allow-off-protocol] [files ...]
```

### `qvla bench check-server`

check that a running policy server speaks the protocol `eval` uses (docs/reference/policy-server.md)

```text
qvla bench check-server [-h] --policy POLICY [--benchmark BENCHMARK] [--task TASK] [--calls CALLS] [--timeout TIMEOUT] [--report REPORT] [--host HOST]
```

## `qvla data`

get and verify the benchmark's data against the release checksum manifest

### `qvla data verify`

compare the checksum manifest with the Hub (no download) or a local tree

```text
qvla data verify [-h] [--checksums FILE] [--local LOCAL] [--repo REPO]
```

### `qvla data get`

download a published data set, verified file by file against the release checksum manifest

```text
qvla data get [-h] [--include GLOB] [--with-init] [--serving {pi05,smolvla,xvla}] [--dest DEST] [--host HOST] [--checksums FILE] [--results-map FILE] target
```

## `qvla init`

write ./qvla.toml: where this machine keeps the data, runs and environments

```text
qvla init [-h] [--root DIR] [--force] [--gpus N]
```

## `qvla paper`

reproduce the paper's results -- every figure value, the main results and the in-text numbers -- from the published per-seed files

### `qvla paper tables`

rebuild every figure value, the main results and every in-text number; --check compares with the pack's expected/

```text
qvla paper tables [-h] [--results RESULTS] [--out OUT] [--check] [--cache CACHE] [--results-map FILE] [--checksums FILE] [--benchmark BENCHMARK]
```

### `qvla paper label-study`

the quality-label distributions and their rank correlation with human judgments, from published files

```text
qvla paper label-study [-h] --pretraining-data DIR [--study STUDY] [--expected EXPECTED] [--out OUT] [--check]
```

### `qvla paper reeval`

re-evaluate released checkpoints -- one result, the main results or a figure -- and write the paper-layout tables from your own runs

```text
qvla paper reeval [-h] (--result ID | --table NAME | --figure NAME | --list) [--backbone {pi05,smolvla,xvla}] [--out OUT] [--host HOST] [--print] [--collect]
```

## `qvla labels`

compile, export and check the paper's quality labels and training weights

### `qvla labels build-paper`

compile every paper run's labels from the published pretraining data

```text
qvla labels build-paper [-h] --pretraining-data DIR --episodes DIR --out OUT [--only ONLY]
```

### `qvla labels export`

write compiled labels' legacy files, byte-equal to what the paper's runs read

```text
qvla labels export [-h] --format {openpi-legacy,lerobot-legacy} --out OUT COMPILED
```

### `qvla labels check`

verify a compiled-labels tree against the paper's declaration and re-export every file the checksum manifest lists from it, byte for byte

```text
qvla labels check [-h] COMPILED
```

## `qvla results`

build qvla.results/v1 files from an evaluation, and score, contrast and re-check them under the reporting rules

### `qvla results score`

score results files under the reporting rules: all-24 and per task, with seed-bootstrap intervals (as `bench score <files>`)

```text
qvla results score [-h] [--benchmark BENCHMARK] [--view {all24,informative_20,usable_12}] [--bootstrap B] [--per-task] [--allow-off-protocol] files [files ...]
```

### `qvla results contrast`

contrast two methods' results files, one per training seed on each side: 100 (SR_A - SR_B) / SR_BC per replicate with --relative, else percentage points

```text
qvla results contrast [-h] [--benchmark BENCHMARK] --a A [A ...] --b B [B ...] [--bc BC [BC ...]] [--relative] [--allow-off-protocol] [--results RESULTS] [--cache CACHE] [--host HOST]
```

### `qvla results reeval`

is a re-read of a checkpoint consistent with its published read? (a reproducibility check; exit 1 if not)

```text
qvla results reeval [-h] --published PUBLISHED --new NEW [--view {all24,informative_20,usable_12}] [--allow-different-checkpoints] [--benchmark BENCHMARK]
```

### `qvla results build`

turn one `qvla bench eval` run into a qvla.results/v1 file

```text
qvla results build [-h] --run-root RUN_ROOT --run RUN --ft-seed FT_SEED --pretrain-seed PRETRAIN_SEED [--checkpoint-ref CHECKPOINT_REF] [--log-root LOG_ROOT] --out OUT [--benchmark BENCHMARK]
```

## `qvla realrobot`

real-robot steps (workspace … serve) and tools: kinematic renderer, REALALIGN labels, FR3 data and configs

### `qvla realrobot render`

python -m axis_qvla.realrobot.render

#### `qvla realrobot render source`

a source_episode.json from a recorded trace

```text
python -m axis_qvla.realrobot.render source [-h] --trace TRACE --scene-dir SCENE_DIR --scene-xml SCENE_XML --task-id TASK_ID --attempt-id ATTEMPT_ID --prompt PROMPT --out OUT
```

#### `qvla realrobot render prepare`

the render input tree of the simulation demonstrations in qvla-v1-realworld

```text
python -m axis_qvla.realrobot.render prepare [-h] [--data DATA] [--out OUT] [--tasks TASKS [TASKS ...]] [--host HOST]
```

#### `qvla realrobot render render`

render source episodes to LeRobot v3 tasks

```text
python -m axis_qvla.realrobot.render render [-h] [--episodes EPISODES [EPISODES ...]] [--data DATA] [--tasks TASKS [TASKS ...]] [--robots ROBOTS] [--gripper-xml GRIPPER_XML] [--unlocked-gripper] [--profile PROFILE] --out OUT
```

#### `qvla realrobot render droid8d`

the DROID-8D training view of a render

```text
python -m axis_qvla.realrobot.render droid8d [-h] --src SRC --out OUT
```

#### `qvla realrobot render labels`

attempt-keyed phase curves -> the pretraining data's curve store

```text
python -m axis_qvla.realrobot.render labels [-h] --pretraining-data CORPUS [--attempt-curves ATTEMPT_CURVES] [--export-curves EXPORT_CURVES] [--alignment-map ALIGNMENT_MAP] --out OUT
```

#### `qvla realrobot render gates`

the hard gates -> render_gates.json

```text
python -m axis_qvla.realrobot.render gates [-h] --out OUT [--droid8d DROID8D] [--curves CURVES] [--attempt-curves ATTEMPT_CURVES]
```

#### `qvla realrobot render report`

reported metrics against a reference render of the same attempt

```text
python -m axis_qvla.realrobot.render report [-h] --task TASK --episode EPISODE --ref-task REF_TASK --ref-episode REF_EPISODE --out OUT
```

#### `qvla realrobot render overlay`

the outlines of the rendered table and robot on a real camera frame

```text
python -m axis_qvla.realrobot.render overlay [-h] [--profile PROFILE] --image IMAGE --scene SCENE [--robots ROBOTS] --out OUT [--joints JOINTS JOINTS JOINTS JOINTS JOINTS JOINTS JOINTS]
```

### `qvla realrobot realign`

python -m axis_qvla.realrobot.realign

#### `qvla realrobot realign build`

phase curves -> gated signal artifacts

```text
python -m axis_qvla.realrobot.realign build [-h] --out OUT [--export-curves EXPORT_CURVES] [--alignment-map ALIGNMENT_MAP] [--prompts PROMPTS] [--roots ROOTS] [--curves-store CURVES_STORE] [--min-fit-pool MIN_FIT_POOL] [--valid-from-data] [--runs {bc,conditioning,awr,filtering} [{bc,conditioning,awr,filtering} ...]]
```

#### `qvla realrobot realign gates`

re-run the gates over a built directory

```text
python -m axis_qvla.realrobot.realign gates [-h] --out OUT [--runs {bc,conditioning,awr,filtering} [{bc,conditioning,awr,filtering} ...]]
```

#### `qvla realrobot realign compile`

compile the four stage-1 runs (bc, conditioning, awr, filtering) to compiled labels

```text
python -m axis_qvla.realrobot.realign compile [-h] --built BUILT --out OUT [--runs {bc,conditioning,awr,filtering} [{bc,conditioning,awr,filtering} ...]]
```

#### `qvla realrobot realign phase`

phase curves of rendered episodes from the published recordings

```text
python -m axis_qvla.realrobot.realign phase [-h] --rendered RENDERED --data DATA --out OUT [--tasks TASKS [TASKS ...]]
```

### `qvla realrobot realdata`

python -m axis_qvla.realrobot.realdata

#### `qvla realrobot realdata idle-ranges`

the idle filter: kept segments, optionally the exported dataset

```text
python -m axis_qvla.realrobot.realdata idle-ranges [-h] --dataset DATASET --out OUT [--export EXPORT] [--grip-guard GRIP_GUARD] [--grip-delta GRIP_DELTA]
```

#### `qvla realrobot realdata tail`

tail-10 chunk-start ranges (openpi filter_dict_path file)

```text
python -m axis_qvla.realrobot.realdata tail [-h] --dataset DATASET --out OUT [--n N]
```

#### `qvla realrobot realdata crop224`

rebuild a letterboxed dataset's images as centre-crop-224

```text
python -m axis_qvla.realrobot.realdata crop224 [-h] --dataset DATASET --recordings RECORDINGS --out OUT [--window WINDOW] [--episodes EPISODES [EPISODES ...]]
```

#### `qvla realrobot realdata training-set`

a DROID-layout recording in the per-task layout the π0.5 loader reads

```text
python -m axis_qvla.realrobot.realdata training-set [-h] --dataset DATASET --out OUT [--tail TAIL]
```

### `qvla realrobot fr3`

python -m axis_qvla.realrobot.fr3

#### `qvla realrobot fr3 check`

the line record, its ArmSpecs, presets, a server's handshake, a norm file

```text
python -m axis_qvla.realrobot.fr3 check [-h] [--line LINE] [--runs RUNS] [--presets] [--run RUN] [--handshake HANDSHAKE] [--dataset DATASET] [--norm NORM] [--embodiment {fr3,realalign_sim,fr3_droid}]
```

#### `qvla realrobot fr3 norm`

norm statistics over a training set, in the model's state order

```text
python -m axis_qvla.realrobot.fr3 norm [-h] --dataset DATASET --embodiment {fr3,realalign_sim,fr3_droid} [--out OUT] [--host HOST]
```

#### `qvla realrobot fr3 local-runs`

run specs that name this machine's real-world inputs by content

```text
python -m axis_qvla.realrobot.fr3 local-runs [-h] [--host HOST] [--line LINE] [--runs RUNS [RUNS ...]]
```

#### `qvla realrobot fr3 serve-config`

the openpi config a policy server of a real-world run is started with

```text
python -m axis_qvla.realrobot.fr3 serve-config [-h] --run RUN [--out OUT] [--host HOST]
```

## `qvla train`

train a run spec in its backbone's environment (--print: the command only)

```text
qvla train [-h] [--host HOST] [--smoke] [--no-bind] [--fsdp-devices N] [--resume] [--print | --render] run_spec
```

### `qvla train smoke-fixture`

build the 2-task x 3-episode fixture

```text
qvla train smoke-fixture [-h] [--host HOST] --episodes DIR
```

### `qvla train merge-check`

check that a plain merge accepts these task directories

```text
qvla train merge-check [-h] tasks [tasks ...]
```

### `qvla train pretrain`

pretrain a backbone on your compiled labels with the paper's budget

```text
qvla train pretrain [-h] --labels NAME --backbone {pi05,smolvla,xvla} [--run NAME] [--seed N] [--host HOST] [--fsdp-devices N] [--print]
```

### `qvla train finetune`

fine-tune a pretraining run with the paper's recipe on the 24 held-out tasks

```text
qvla train finetune [-h] --from RUN_ID [--cond] [--run NAME] [--host HOST] [--fsdp-devices N] [--print]
```

## `qvla env`

build and check the training and evaluation environments

### `qvla env build`

build an environment from its pinned lock, then check it

```text
qvla env build [-h] [--print] [--rebuild] environment
```

### `qvla env check`

check a built environment against its pins

```text
qvla env check [-h] [--python PYTHON] [--checkout CHECKOUT] environment
```
