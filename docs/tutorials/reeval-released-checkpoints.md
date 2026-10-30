# Re-evaluate the released checkpoints

Every result of the paper comes with its fine-tuned checkpoint. `qvla paper reeval` takes one result, or
the main results or one figure of the paper, fetches what it needs (the benchmark data, the base weights the
backbone's reference server loads, each checkpoint), serves each checkpoint on its backbone's reference
server, evaluates it on the 24 held-out tasks with 50 seeds each, and writes that table or figure in the
paper's layout from your own runs. A result you did not rerun shows a dash (—) in the main results and
an empty field in a figure's data.

You need a GPU with EGL and the `eval` extra ([Install](../install.md)).

## Build the environments

The evaluation runs in the `eval` environment, and each backbone's reference server in that backbone's.
Build `eval` and the environment of each backbone you rerun (`pi05`, `smolvla` or `xvla`) once:

```bash setup
qvla env build eval
qvla env build pi05
```

<!-- ci
`--print` prints the build commands instead of running them.

```bash ci
qvla env build eval --print
qvla env build pi05 --print
```

```text output
# Benchmark evaluation in simulation: …
# pi0.5 training and serving: …
```
-->

If one is missing, `qvla paper reeval` names the command that builds it before it fetches anything.

## The main results

The main results (each method relative to plain BC) on π0.5:

```bash gpu
qvla paper reeval --table main-results --backbone pi05
```

<!-- ci
```bash ci
qvla paper reeval --table main-results --backbone pi05 --print
```

```text output
qvla data get bench
qvla data get base-weights --serving pi05
qvla bench eval --checkpoint-id … --serving bare --run r1bc_r2 …
# tables: …/paper-reeval/main-results-pi05
qvla paper reeval --table main-results --backbone pi05 --collect
```
-->

It prints the main results with your π0.5 results; the cells of the other backbones show a dash (—).
Without `--backbone` it reruns every backbone of the table. The table lands in
`<runs>/paper-reeval/main-results-pi05`, under the `runs` root of your `qvla.toml` (`--out DIR` for
another directory). `--figure` does the same for the data of one figure:

| `--figure` | what it shows |
|---|---|
| `headline` | the headline result |
| `random-vs-filtered` | random selection against approach-aware filtering at the same half-data budget |
| `methods` | weighting, top-50% filtering and conditioning |
| `scaling` | scaling with pretraining data size |
| `conditioning` | the conditioning stage and the inference context |

## One result

A result is named as the results map names it: the method's row and the mode it is served in, such as
`bc|bare` (the BC row, served bare). `--list` prints every result with its backbone, the run it was published
under and the units that read it:

```bash ci
qvla paper reeval --list
```

```text output
bc|bare	pi05	r1bc_r2	main-results, headline, methods, scaling, conditioning
bc|cond	pi05	r1bccond	conditioning
```

Re-evaluate one of them (quote the name: `|` is a pipe to the shell):

```bash gpu
qvla paper reeval --result 'bc|bare'
```

<!-- ci
```bash ci
qvla paper reeval --result 'bc|bare' --print
```
-->

It prints the result's all-24 success rate from your run. Each result's results file goes under the `r1`
root of your `qvla.toml`, in the name of the run the paper published it under.

## On a cluster

`--print` prints the commands in order instead of running them, for a batch script; the last one builds
the tables from the results that are there:

```text output
qvla data get bench
qvla data get base-weights --serving pi05
qvla data get checkpoints/…
qvla bench eval --checkpoint-id … --serving bare --run r1bc_r2 …
qvla paper reeval --result 'bc|bare' --collect
```

A result that is already evaluated is skipped, so running the command again after an interruption resumes
it.

What the numbers mean and how to compare them: [reporting rules](../reference/reporting-rules.md).
