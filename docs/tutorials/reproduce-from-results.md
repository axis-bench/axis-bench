# Reproduce the paper from published results

The paper reports 123 results: π0.5, SmolVLA and X-VLA, each pretrained with one quality method,
fine-tuned on the 24 held-out tasks and evaluated on 50 seeds per task. The per-seed rollout logs of every
result are published. `qvla paper tables` recomputes from them every number the paper derives from its
evaluation rollouts and checks them against the published outputs: the main results (each method relative
to plain BC, per VLA), the data of the figures (the headline result, random selection against filtering at
the same data budget, the methods, scaling with pretraining data size, conditioning), the success rates in
the text and a results page.

You need a CPU and the `reproduce` extra ([Install](../install.md)).

## Recompute the tables

```bash data timeout=1800
qvla paper tables --check
```

```text output
IDENTICAL  14 outputs match expected/
```

<!-- ci
The same command on the two-result fixture this repository ships, with its own benchmark declaration and
expected/.

```bash ci
qvla paper tables --results tests/golden/paper/fixture_pack/pack --results-map tests/golden/paper/fixture_pack/paper_fixture_cells.json --benchmark fixture_bench --bench-conf-dir tests/golden/paper/fixture_pack/conf --check
```

```text output
IDENTICAL  13 outputs match expected/
```
-->

It reads the v1.0 results from the Hub and keeps the download in `./.qvla-cache` (`--cache DIR` for
another directory). Add `--out DIR` to keep the outputs in `DIR`: `tables/table1.md`, `figures/*.csv`,
`claims.json` and `results_page.html`.
[How the check works](../reference/verification.md#the-papers-tables).

## Quality labels and the human study

The quality-label distributions and their rank correlation with human judgments come from the pretraining
data's labels and the human study, not from the rollouts. `qvla paper label-study` recomputes them; the
correlation needs numpy, which the `method` extra installs:

```bash data timeout=1800
qvla data get corpus --include 'index/*' --include 'signals/v1/*/chunk/awr_weights.json' --dest corpus
qvla paper label-study --pretraining-data corpus --check
```

<!-- ci
CI fetches no pretraining data, so it checks that the command exists.

```bash ci
qvla paper label-study --help
```

```text output
usage: qvla paper label-study …
```
-->

`--out DIR` keeps `fig4a_sim_pe.csv` and `fig4.json`.
[How the check works](../reference/verification.md#quality-labels-and-the-human-study).

## Next

- [Re-evaluate the released checkpoints](reeval-released-checkpoints.md): measure a result again on your GPU.
- Compare results files of your own the way the main results do: [reporting rules](../reference/reporting-rules.md#comparing-two-methods).
