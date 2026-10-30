# Results format

Each evaluated run has one JSON results file (schema `qvla.results/v1`): its header records what was
measured and how, and its rows record the individual rollouts, one per rollout. The bytes are
canonical JSON (sorted keys, one-space indent, UTF-8, one trailing newline).

- `build`: turn a run of `qvla bench eval` into a results file again ([below](#building-a-results-file)).
- `score`: score results files, all-24 and per task.
- `contrast`: compare two methods, one results file per training seed on each side.
- `reeval`: check a re-read of a checkpoint against its published read.

```
qvla results build --run-root <campaign root> --run <run> --ft-seed <n> --pretrain-seed <n|none> \
    [--checkpoint-ref <ref>] [--log-root <dir>] --out <run>.json
qvla results score <run>.json ...
qvla results contrast --a <A files> --b <B files>
qvla results reeval --published <published>.json --new <your read>.json
```

How the scorer reads these files: [reporting rules](reporting-rules.md).

## Building a results file

`qvla bench eval` writes its run's results file at the end, `<r1>/out_<run>/results.json`, with the
training seeds it was given. `qvla results build` turns a run's summary and per-seed rollout files into a
results file again, for example with other seeds stated. It reads what `qvla bench eval` left on disk and
nothing else:

- `out_<run>/arm_summary.json` under the campaign root. It names the served directory, the log
  root, the eval fingerprint and the benchmark, and embeds the run's `run_manifest.json` as
  `provenance`.
- each task's `meta.json` and `seed_<n>.json` files under `<log root>/<served>/task_<id>/`. Pass
  `--log-root` if the seed files have moved since the run.

Training seeds are not on disk, so you pass them when you build the file: `--ft-seed`, and
`--pretrain-seed` (`none` for a run with no pretraining). `--checkpoint-ref` says how a reader can obtain the checkpoint;
the default is the path the run recorded.

It refuses a run it cannot tie to one benchmark, one seed set and one served policy
([how results are verified](verification.md#evaluation-runs)).

## Where a run's files land

Per-seed rollout logs land under the `rollout_logs` root of `qvla.toml` (`qvla init` writes
`<r1>/rollout_logs/`), the same for every backend: `bench eval` passes that one directory to every
worker, and the tally reads it. Under it, each task's `meta.json` and `seed_<n>.json` files are in
`<served>/task_<id>/`. The run's summary, `out_<run>/arm_summary.json`, and its results file,
`out_<run>/results.json`, are under the `r1` root of `qvla.toml` ([where files go](data.md#where-files-go)).

## Header

| key | content |
|---|---|
| `bench.name`, `bench.version` | the declaration and its benchmark version |
| `bench.version_sha256` | the version's content hash (`qvla bench versions`) |
| `bench.declaration_sha256` | sha256 of the declaration file the run evaluated |
| `bench.identity_source` | where the benchmark identity was read: `run_record` or `summary` |
| `bench.units_sha256` | digest of the declared (task, seed) work units of the seed set |
| `bench.seed_set` | `test`, or `dev` for the development seeds |
| `arm`, `served` | the run, and the directory its rollouts were written under |
| `training_seed` | `{pretrain: int or null, ft: int}` |
| `checkpoint` | `ref` (how to obtain it), `eval_digest` (the evaluated directory's content digest), `pub_digest` (the published file set's digest; `null` in a file `build` writes) |
| `serve` | `fields` (the conditioning, e.g. `{"quality": "5"}`), `uncond_fields` (`{}` when guided, else `null`), `beta` (`1 + w`) |
| `execution` | `fps`, `replan`, `max_steps`, `n_stable_frames`, `off_protocol`, and `deviations` when the run executed something other than the benchmark declares |
| `image_geometry`, `action_space`, `xla_pin`, `noise_salt` | the conventions the run used |
| `harness` | `driver_sha256`, `repo_head`, `repo_dirty` |
| `gpu`, `eval_lock_sha256` | the machine's GPUs; reserved |
| `eval_fingerprint` | sha256 over every seed file's name and bytes, tasks ascending |

The benchmark identity is the run record's, written when the run started. When a run record names
none, the identity is the summary's, which a `--tally-only` writes at tally time. `serve` comes from the policy server's own handshake as every
task recorded it. `image_geometry` and `action_space` are what
executed: a recorded deviation's value, else the run record's, else the declared one.

`execution` holds the constants the rollouts recorded. `deviations` maps each key that differs
from the benchmark's `execution.json` to `{executed, declared}`. It joins three sources:

- the run record (the environment the workers started under);
- each rollout's `meta.json`;
- the recorded constants themselves.

`off_protocol` is `true` when there is any deviation, and also when the run was started with
`--off-protocol`, whether or not anything deviated. A development read (`bench.seed_set: dev`)
carries the development units digest, so it scores as a development read.

## Rows

`{task, seed, success, n_steps, sim_time, reason}`, sorted by `(task, seed)`, one per seed file.
`reason` is `success`, `timeout`, or an `error:` string.

## Names in released files

- **A result** (`cell` in `cells.json`, such as `bc|bare` or `bc|cond`): one checkpoint, served one way,
  evaluated once on the 24 × 50 benchmark. The suffix says how it is served: `bare` without a quality
  tag, `cond` with one. `qvla data get checkpoints/<result>` takes this name.
- **An evaluation run** (`run` in `cells.json`, such as `r1bc_r2`): the name `bench eval --run` files an
  evaluation under, and a results file's `arm`. `served` (such as `r1bc_r2_dualsim`) is the directory
  its rollouts are written under.
- **A training run id** (such as `pi05.finetune.<name>` or `pi05.pretrain.keep_top`):
  `<backbone>.<stage>.<name>`, one training run, declared by its run spec. A result's `lineage.arm`
  names the run that trained its checkpoint.
- **A backbone** (`pi05`, `smolvla`, `xvla`): `backbone` in `cells.json` and the first part of a
  training run id. `bench eval --backend` names its reference server: `openpi` for π0.5, `smolvla`,
  `xvla`.
- **A frame index** (`rowspace` in a run spec's `[data]`, in `--rowspace` and in
  `conf/recipes/rowspaces.toml`, such as `libero5k_v2_d8`): the fixed position of every frame of a
  dataset, and which of those frames a run may draw.
- **Compiled labels** (`bundle` in a run spec's `[signal]` and under `<cache>/bundles/`): a quality
  method's decision, written before training: the frame each sample of each step draws, an optional
  loss weight per frame and an optional prompt tag.
