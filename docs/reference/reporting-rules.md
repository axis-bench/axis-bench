# Reporting rules

Use these rules to score one checkpoint and to compare methods. They define the all-24 aggregate,
the uncertainty of one evaluation, and paired comparisons across training seeds. One read of one
checkpoint is a noisy number, and the paper's findings are differences between such numbers, so how
numbers are aggregated and compared is part of the benchmark. The scorer (`qvla bench score`,
`qvla results score|contrast|reeval`) applies these rules:

- **One aggregate.** all-24 is the mean, over the 24 tasks, of each task's success rate over its 50 test seeds.
- **Subsets are views.** The two declared subsets are appendix views; all-24 is the default.
- **An interval describes one read**: one checkpoint over the benchmark's seeds.
- **Comparisons are paired** by training seed, with at least three pairs for a claim; the paper
  reports them relative to plain behavioural cloning.
- **Seeds have roles.** Tune and select on the development seeds; report the test seeds.

Results files are described in [results format](results-format.md).

## The one aggregate: all-24

**all-24** is the number to report: compute each of the benchmark's 24 tasks' success rate over its
50 test seeds, then take the mean of the 24 rates. Every table and every comparison uses it unless it says
otherwise.

- A declared (task, seed) with no rollout counts as a failure, so a partial read scores as the full
  read with those rollouts failed.
- The scorer scores files whose rollouts all completed, and refuses a file with any rollout whose
  `reason` starts `error:`.

```
qvla bench score <run>.json [--per-task]       # all-24 ± SE, a 95% seed-bootstrap interval; per task, Wilson
qvla results score <run>.json --view usable_12  # an appendix view
```

One read's evaluation uncertainty is reported three ways:

- `±` is the binomial SE of the mean of the per-task rates.
- The 95 % interval of the all-24 number is a seed bootstrap of that mean. Each task's seeds are
  resampled with replacement, with B = 10,000 resamples drawn from `random.Random(7)`, and the
  bounds are percentiles. `--bootstrap 0` turns it off.
- `--per-task` gives each task's 95 % **Wilson score** interval,
  `(p̂ + z²/2n ∓ z·sqrt(p̂(1 − p̂)/n + z²/4n²)) / (1 + z²/n)` with z = 1.959964. At 0/n and n/n,
  where several tasks sit, the binomial SE and a bootstrap both collapse to a point (every
  resample of 0/50 is 0/50); the Wilson interval does not (0/50 gives [0, 7.1] %, 1/50 gives
  [0.4, 10.5] %).

All three describe that one read of that one checkpoint over the benchmark's seeds. The SE and the
bootstrap take each task's observed rate as its distribution, so a task at 0 % or 100 % adds no
spread to them. Each belongs to its own read; the variation between training runs is what a
[comparison of replicates](#comparing-two-methods) measures.

### Subsets are appendix views

The benchmark declares two subsets (`subsets` in the declaration):

- `informative_20` drops the four tasks where the π0.5 results sit at the floor or the ceiling
  (1426, 1435, 1436, 1459);
- `usable_12` is a twelve-task subset.

`--view` prints either one, flagged `(appendix view)`; the default is all-24. Report all-24.

## Comparing two methods

A contrast pairs the two sides by **training replicate**. The key is the pretraining seed, or the
fine-tune seed for a run without pretraining. For each replicate it reports:

- the paired difference of the two reads' all-24 rates, in percentage points;
- the evaluation SE **of that pair**, `sqrt(se_a² + se_b²)`, from the two reads' per-task binomial
  SEs.

Then it reports the mean and the **sample** SD of the per-replicate differences:

```
qvla results contrast --a <A file per replicate> --b <B file per replicate> [--relative --bc <BC files>]

# what it prints (the numbers are illustrative):
replicate 42: +3.12 pt (eval SE of this pair 1.49)
replicate 43: +1.04 pt (eval SE of this pair 1.47)
replicate 44: +2.30 pt (eval SE of this pair 1.50)
a42+a43+a44 - b42+b43+b44: mean +2.15 pt, sample SD 1.05, pairs 3 [paired, n_seeds=3]
```

Report the per-replicate differences and their eval SEs, then the mean and sample SD. An evaluation
error belongs to the contrast it was computed for; compute it for each pair.

- With fewer than 3 replicates the result is labelled `descriptive, n_seeds=k`, and the contrast
  prints `no claim`.
- The contrast records a fine-tune seed that differs inside a pair.
- `--relative --bc <files>` reports the statistic of the paper's main results instead,
  R = 100 (SR_A − SR_B) / SR_BC per replicate. Its eval SE comes from the delta method and
  includes the BC read's own error:

      var R = g² (se_A² + se_B² − 2 cov_AB) + h² se_BC² + 2 g h (cov_A,BC − cov_B,BC),
      g = 100 / SR_BC,   h = −100 (SR_A − SR_B) / SR_BC²

  Separate reads are independent (covariance 0). A side that is the BC read itself (the same seed
  files, as when B is the plain-BC run) has covariance se_BC² with it.

Training seeds dominate the variation between methods, so methods are compared across training
replicates, each one training run.

## Development seeds

The benchmark also declares development seeds (`dev_seeds`), disjoint from every task's test
seeds. Tune and select checkpoints on them: `qvla bench eval --seed-set dev` files the read under
`<run>_dev`. A development read scores on the development seeds, labelled `[development seeds]`.
Report test reads, of a checkpoint fixed before its test read.

## What the scorer pools

The scorer pools and compares reads of one benchmark version, one seed set and one protocol:

- **One benchmark version.** It scores results of the version they name (`bench.version_sha256`), on
  that version's files as released; it refuses a results file of another version.
- **One seed set.** A contrast pairs test reads with test reads, or development reads with
  development reads; it refuses a mix.
- **The benchmark's protocol.** It reads an off-protocol result (an execution constant or convention
  other than the benchmark's `execution.json`, a recorded deviation, or a run started with
  `--off-protocol`) only with `--allow-off-protocol`, and labels each `OFF-PROTOCOL`. A contrast
  pairs reads executed alike.
- **Trained quality tags.** A conditioned read carries a trained quality bin. Tag 0 is the trainer's
  "untagged" sentinel, so a server conditioned on it is refused.
- **One read per replicate.** A contrast pairs replicates by training seed and refuses the same
  checkpoint twice on one side: a re-read is a reproducibility check of its replicate.
