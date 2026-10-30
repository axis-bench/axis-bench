# How results are verified

Every number AXIS-Bench reports is computed from files that are checked before they are used. The
checks run inside the commands, so nothing on this page is a step you run by hand; the tutorials link
here instead of repeating it.

## Downloads

`conf/release/v1.0.lock.json`, the checksum manifest, lists every published file of every repository with
its size, sha256 and git blob id at tag `v1.0`. `qvla data get` checks every file it downloads against
it: it keeps the files that match (exit 1 on one that does not), and an interrupted `get` resumes.
`qvla data verify` checks the hosted files without downloading them; `qvla data verify --local <dir>
--repo axisrobotics/<repo>` checks a local copy. What your machine computes goes beside the downloaded
repositories, so a local copy stays checkable as published.

The base weights come from their publishers. `conf/recipes/bases.lock.json` pins each Google Cloud
Storage object's generation, each Hub repository's commit, and every file's size and sha256;
`qvla data get base-weights` keeps only matching files.

## The paper's tables

`qvla paper tables --check` rebuilds every number the paper derives from the evaluation rollouts. The
quality-label distributions and their rank correlation with human judgments come from the labels and
the human study; [quality labels and the human study](#quality-labels-and-the-human-study) below covers
them.

It fetches the per-seed rollout logs of the 123 results and recomputes each run's `eval_fingerprint`. It
also checks that every `results/` file and run manifest (`runs/`) that `cells.json` references exists and
matches its recorded sha256 (and size, where one is recorded), and checks the task set against the
benchmark declaration. From a local pack, a missing or mismatched referenced file is an integrity failure
(exit 2). Over `hf://` every file is read through the checksum manifest: a path the checksum manifest does
not list, or a digest that disagrees with it, is an integrity failure (exit 2), and a listed path the Hub
does not serve is a fetch error (exit 3). It then byte-compares its outputs with the published
`expected/`, whose manifest the results map's own digest pins.

It needs nothing beyond the standard library and gives byte-identical outputs on Python 3.11, 3.12 and
3.13. Its defaults (the results map, the table layout, the benchmark declaration and the checksum
manifest) are read from the checkout's `conf/` and `experiments/`, so run it from a checkout installed
editable (`pip install -e .`), or run `python -m axis_qvla.cli paper tables` from the repository root.

## Quality labels and the human study

`qvla paper label-study --check` compares the quality-label distributions, the per-reward summary of
`P_e` (the share of an episode's frames whose training weight exceeds 1), exactly with
`experiments/paper/fig4.expected.json`. For the rank correlation with human judgments it holds
Spearman's ρ to 1e-12 and the ends of the 10,000-draw paired bootstrap interval to 1e-9 of the expected
values (`numpy.random.default_rng(42)`, episodes in the study file's order, each resample with the
48-index auxiliary draw of `experiments/paper/fig4b_human_study.json`). It exits 1 on a difference.

## Quality labels and compiled labels

A compiled-labels directory is named by the sha256 of its `manifest.json`, which describes every array by
dtype, shape and digest: equal ids mean identical labels, on any machine. A frame index is named by the
sha256 of its record. You refer to both by name; `qvla train` resolves a name to its id when the run binds
and records the id in the run.

Compiling checks the composition rules: one producer per output, a plan inside the frame index's
trainable frames, codes from the vocabulary, and a guided condition only with an unconditional branch.
A constant quality label is refused: labels that give every unit the same quality bin carry no
conditioning signal.

The published quality labels and training weights are listed in `conf/release/SIGNALS.lock.json`. Before
using a label file, the LeRobot fork's loader checks it against the lock: a file at a locked path must hold
that path's locked bytes, and locked bytes load only as their own kind and over their own frame index. Bytes
the lock does not know load with an `unlocked` warning, so new labels can be used.

## Training runs

Every input of a run is named by its content, so a run spec identifies the whole run. Before it writes
anything, `qvla train` binds the compiled labels: it checks them against the run's frame index, budget and
backbone, and lists every problem at once (a frame index that is not the run's, a capability the backbone
lacks, a plan that does not fit the budget, an input that is missing or holds other bytes), before any GPU
time. A run directory that already holds another run is refused; nothing is overwritten.

## The benchmark

A benchmark version is frozen once numbers are reported on it: new tasks, layouts or execution constants
make a new version, so a reported number keeps its meaning. A version's identity is one
sha256 over its files, excluding `VERSION.json`, which records that identity and each file's digest.
`qvla bench versions` lists the versions and whether each matches its record; `qvla bench check` prints the
benchmark's identity and checks that the version is consistent. The scorer compares results of one version
only.

The asset store checks every scene file it opens against `conf/benchmarks/axis-heldout-v1/ASSETS.lock.json`.
The render shell reads the Polyhaven table-texture bank from the renderer,
`third_party/axis-render/assets/table_textures/polyhaven/downloaded_256/`. `qvla bench eval` links the bank
there from `<bench>/assets/textures/polyhaven/downloaded_256/*` and checks that directory against
`ASSETS.lock.json` before any server starts: the same file names both ways and every sha256.

## Evaluation runs

A run is tied to one benchmark, one seed set and one served policy. Its `eval_fingerprint` is the sha256
over every seed file's name and bytes, tasks ascending, and a results file records it with the digests of
the declaration, the work units and the driver ([results format](results-format.md#header)). Building a
results file refuses (exit 2) errored or missing rollouts, seed files that do not reproduce the
fingerprint or are not exactly the declared seeds, and server metadata, execution constants or domains that
differ between tasks. It also refuses (exit 2):

- a conditioned server whose handshake publishes no quality tag, tag 0 (the trainer's "untagged"
  sentinel) or a tag that is not a positive integer: no serve fields describe what such a server did.
  Serve bare, or a trained quality bin;
- a run evaluated on a benchmark other than `--benchmark`'s (another declaration, another version, or
  another task set), a run whose summary names no benchmark, and a run whose summary and run record name
  different benchmarks.

Checkpoints fetched with `qvla data get` are verified file by file on download.

The scorer computes every interval from the rollouts alone, with a fixed seed, so one results file always
gives the same intervals. How each interval is computed: [reporting rules](reporting-rules.md#the-one-aggregate-all-24).

## Real robot

`qvla realrobot render gates` writes `render_gates.json` and exits 1 when any hard gate fails: the render
must follow the recorded hand and objects within the declared tolerances before its labels are trusted.
`qvla realrobot fr3 check` holds a run's norm statistics, dataset and policy-server handshake to the FR3
embodiment the run was trained for; it prints `ALL PASS`, or one `FAIL` line per difference.

## Exit codes

| command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `qvla paper tables --check` | identical | differs | integrity failure | I/O or fetch error |
| `qvla paper label-study --check` | identical | differs | an input cannot be read, or the human-study correlation has no numpy | |
| `qvla data get` | verified | a file does not match (not kept) | the request or the local files are wrong | the Hub or a publisher could not be reached, or answered with an HTTP error (e.g. 401/403/404 for a private repository without a token) |
| `qvla data verify` | every file matches | a file differs | the checksum manifest, the local copy or the Hub cannot be read | |
| `qvla results build` | written | | refused (above) | OS error |
| `qvla results reeval` | consistent | not consistent | refused | OS error |
| `qvla paper reeval` | every step ran, or was done already | | refused before anything is fetched | a step's state cannot be read |
| `qvla train` | trained, printed or rendered | | refused before training | a missing base could not be fetched |
| `qvla bench eval` | evaluated and its results file written, or printed | | refused before any server starts, or the run refused as a results file | the results file cannot be read or written |
| `qvla realrobot labels` | the labels are compiled | a gate fails (the render's or the labels') | refused input | |
| `qvla realrobot render gates` | every gate passes | a gate fails | | |
| `qvla realrobot fr3 check` | `ALL PASS` | a `FAIL` line | | |

Any other exit code is passed on: the training's or the merge's by `qvla train`, the evaluation driver's by
`qvla bench eval`, and the first failing step's by `qvla paper reeval`, which stops there (130 on Ctrl-C).
