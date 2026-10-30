# Fine-tune and evaluate

After pretraining, everything follows the benchmark protocol: the paper's fine-tuning recipe on the
demonstrations of the 24 held-out tasks, then 24 tasks × 50 seeds, reported as the all-24 success rate
(the mean of the 24 tasks' rates) with each task's rate. One command each.

Fine-tuning is **bare** by default: the policy fine-tunes and runs without quality conditioning. If
your method produces conditioning labels, `--cond` fine-tunes with the quality tag instead, as the run
`<backbone>.finetune.<name>.cond`, and the evaluation runs it both bare and conditioned: two results,
filed as `<backbone>.finetune.<name>.cond_b` and `<backbone>.finetune.<name>.cond_c`.

**What you get.** Your method's fine-tuned run and its results file, with the all-24 success rate and
the per-task table printed.

**What you need.** GPUs, the backbone's environment, the evaluation environment and a GPU with EGL
([install](../install.md#environments)), and a finished pretraining run ([Pretrain](pretrain.md)).

| step | command | what it produces |
|---|---|---|
| [1. get the fine-tuning data](#1-get-the-fine-tuning-data) | `qvla data get bench` | the held-out tasks' demonstrations, scenes and layouts under `<corpus>/qvla-v1-bench/` |
| [2. fine-tune](#2-fine-tune) | `qvla train finetune --from pi05.pretrain.example.keep_top` | the run `pi05.finetune.example.keep_top` |
| [3. evaluate](#3-evaluate) | `qvla bench eval --from-run pi05.finetune.example.keep_top` | its results file, the all-24 success rate and the per-task table |
| [4. compare with plain BC](#4-compare-with-plain-bc) | `qvla results contrast --a … --b paper:'bc|bare' --bc paper:'bc|bare' --relative` | your method relative to plain BC, per training seed |

## 1. Get the fine-tuning data

```bash setup
qvla data get bench
```

<!-- ci
The setup block's twin: the steps below need a planned pretraining run, which `--print` writes
without data, so the pretraining of the page before is planned here.

```bash ci
qvla init --root roots
qvla train pretrain --labels example.keep_top --backbone pi05 --print
qvla train pretrain --labels example.keep_top --backbone pi05 --seed 43 --print
```

```text output
NEXT  qvla train finetune --from pi05.pretrain.example.keep_top
```
-->

## 2. Fine-tune

With the run id your pretraining printed:

```bash gpu
qvla train finetune --from pi05.pretrain.example.keep_top
```

<!-- ci
`--print` writes the fine-tune's run spec and prints the command that trains it, before the
pretraining has run:

```bash ci
qvla train finetune --from pi05.pretrain.example.keep_top --print
qvla train finetune --from pi05.pretrain.example.keep_top --cond --print
qvla train finetune --from pi05.pretrain.example.keep_top.s43 --print
```

```text output
OK    pi05.finetune.example.keep_top: …/roots/runs/pi05.finetune.example.keep_top/spec.toml
NEXT  qvla bench eval --from-run pi05.finetune.example.keep_top
OK    pi05.finetune.example.keep_top.cond: …/roots/runs/pi05.finetune.example.keep_top.cond/spec.toml
NEXT  qvla bench eval --from-run pi05.finetune.example.keep_top.cond
SEEDS pretrain 43, fine-tune 42 (the paper's pi0.5 fine-tunes, pretrain seed -> fine-tune seed: 42 -> 42; 43 -> 42 (bc43: 43); 44 -> 42); `qvla results contrast` pairs this run on pretrain seed 43
NEXT  qvla bench eval --from-run pi05.finetune.example.keep_top.s43
```
-->

It starts from the pretraining run's final checkpoint and trains the paper's fine-tune for that
backbone; it refuses a pretraining run that has not finished. When it finishes it prints the next
command. A run fine-tuned with `--cond` has the id `<backbone>.finetune.<name>.cond`
(`pi05.finetune.example.keep_top.cond`).

Fine-tune the size-matched control the same way:
`qvla train finetune --from pi05.pretrain.example.keep_top.control`.

## 3. Evaluate

```bash gpu
qvla bench eval --from-run pi05.finetune.example.keep_top
```

<!-- ci
`--print` prints the command that runs the evaluation and where its results file lands, and starts
nothing; a `--cond` run is evaluated twice:

```bash ci
qvla bench eval --from-run pi05.finetune.example.keep_top --print
qvla bench eval --from-run pi05.finetune.example.keep_top.cond --print
```

```text output
# results file: …/roots/r1/out_pi05.finetune.example.keep_top/results.json
… bench eval --from-run pi05.finetune.example.keep_top
# results file: …/roots/r1/out_pi05.finetune.example.keep_top.cond_b/results.json
# results file: …/roots/r1/out_pi05.finetune.example.keep_top.cond_c/results.json
… bench eval --from-run pi05.finetune.example.keep_top.cond
```
-->

It serves the fine-tuned checkpoint on the backbone's reference server, runs the 24 tasks with 50
seeds each, writes the results file `<r1>/out_<run id>/results.json` and prints the all-24 success
rate with each task's rate. If the evaluation environment is not built, it stops and names the
command that builds it (`qvla env build eval`). `--print` prints the one command that runs the
evaluation, for a SLURM job.

Evaluate the control the same way, `qvla bench eval --from-run pi05.finetune.example.keep_top.control`,
so the two results can be contrasted.

## 4. Compare with plain BC

The paper's main results report each method relative to plain behavioural cloning:
`100 (SR_A − SR_B) / SR_BC`, in % of plain BC, paired on the pretraining seed. This fetches the paper's published results, then contrasts
your results file with the paper's plain-BC π0.5 run, `bc|bare` (`svla_bc|bare` and `xvla_bc|bare`
for SmolVLA and X-VLA; `qvla paper reeval --list` lists every published result). Quote the
`paper:'…'` name: unquoted, the `|` in it is a shell pipe.

```bash gpu
qvla data get results
qvla results contrast --a <r1>/out_pi05.finetune.example.keep_top/results.json --b paper:'bc|bare' --bc paper:'bc|bare' --relative
```

<!-- ci
The same contrast on the contrast tests' fixture pack (2 tasks × 3 seeds), with stand-in results
files for the method and its control:

```bash ci
python -m tests.docs.fixtures method-results r1
qvla results contrast --a r1/out_pi05.finetune.example.keep_top/results.json --b paper:'bc|bare' --bc paper:'bc|bare' --relative --results tests/golden/paper/fixture_pack/pack --results-map tests/golden/paper/fixture_pack/paper_fixture_cells.json --benchmark fixture_bench --bench-conf-dir tests/golden/paper/fixture_pack/conf
```
-->

```text output
  paper:bc|bare = … (seed 42)
replicate 42: … % of plain BC (eval SE of this pair …)
pi05.finetune.example.keep_top - …: mean … % of plain BC, sample SD n/a, pairs 1 [descriptive, n_seeds=1; …]
no claim: fewer than 3 paired training seeds
```

One training seed gives a descriptive result, not a claim. A claim needs three replicates: pretrain
with `--seed 42`, `--seed 43` and `--seed 44` (0, 1 and 2 for SmolVLA and X-VLA;
`qvla train pretrain --labels example.keep_top
--backbone pi05 --seed 43` makes `pi05.pretrain.example.keep_top.s43`), fine-tune and evaluate each,
and pass the three results files together to `--a`. Each is paired with the paper's replicate at the
same pretraining seed. The paper's fine-tunes use these seeds, and yours takes the fine-tune seed of
the "every other method" row at its pretraining seed:

| paper fine-tunes | pretrain seed | fine-tune seed |
|---|---|---|
| pi0.5 plain BC (`bc`, `bc43`, `bc44`) | 42, 43, 44 | 42, 43, 42 |
| pi0.5, every other method | 42, 43, 44 | 42 |
| SmolVLA and X-VLA, every method | 0, 1, 2 | 0, 1, 2 |

A pi0.5 replicate at pretrain seed 43 pairs with `bc43`, whose fine-tune seed is 43, so the contrast
records that pair's seeds with `ft seeds differ in replicate(s) 43`. The statistics are in the
[reporting rules](../reference/reporting-rules.md#comparing-two-methods).

Contrast your method with its size-matched control the same way, so a gain can be told apart from
training on fewer frames:

```bash gpu
qvla results contrast --a <r1>/out_pi05.finetune.example.keep_top/results.json --b <r1>/out_pi05.finetune.example.keep_top.control/results.json --bc paper:'bc|bare' --relative
```

<!-- ci
```bash ci
qvla results contrast --a r1/out_pi05.finetune.example.keep_top/results.json --b r1/out_pi05.finetune.example.keep_top.control/results.json --bc paper:'bc|bare' --relative --results tests/golden/paper/fixture_pack/pack --results-map tests/golden/paper/fixture_pack/paper_fixture_cells.json --benchmark fixture_bench --bench-conf-dir tests/golden/paper/fixture_pack/conf
```

```text output
pi05.finetune.example.keep_top - pi05.finetune.example.keep_top.control: mean … % of plain BC, …
```
-->
