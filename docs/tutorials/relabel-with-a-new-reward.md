# Relabel with a new reward

The pretraining data publishes per-frame labels: poses, the task-progress reward, its validity
(`reward_valid`) and success. The label pipeline recomputes them from the raw recordings
([quality labels](../reference/data.md#quality-labels)), so a new reward can be computed from the same geometry
and turned into training weights. The recordings behind 21 pretraining tasks are published as the
traces sample: a task for every instruction verb, and every episode of each.

**What you get.** A scores file: one training weight per frame of every sampled episode, computed from
a reward of your own, in the format a quality method reads.

**What you need.** The `method` extra and the traces sample (`qvla data get traces-sample`). A reward
over the sample reads the recorded poses of the hand and the objects.

| step | command | what it produces |
|---|---|---|
| 1. [write the reward](#a-new-reward) | `experiments/examples/relabel.py` | a reward per render row of each episode |
| 2. [run it](#run-it) | `qvla data get traces-sample --dest corpus`, then `python experiments/examples/relabel.py --traces corpus/traces_sample --out reach_weights.json` | `reach_weights.json`; one line per episode, then the file's header |
| 3. use it | a scores file over the pretraining episodes, which `experiments/examples/keep_top.py --scores` reads ([write a quality method](write-a-quality-method.md)) | your method's compiled labels |

## A new reward

`experiments/examples/relabel.py` labels every sampled episode with the pipeline, then scores it
with a reward of its own: how close the hand is to the task's object, on every render row.

```python file=experiments/examples/relabel.py
def reach_reward(cols: dict, target: str) -> np.ndarray:
    """The new reward: 1 - tanh(5 d), d the hand-to-object distance in m, on every render row."""
    names = list(cols["body_names"])
    pose = cols["observation.body_pose"]                      # (rows, bodies, 7): xyz, then a quaternion
    d = np.linalg.norm(pose[:, names.index(HAND_BODY), :3] - pose[:, names.index(target), :3], axis=1)
    return (1.0 - np.tanh(5.0 * d)).astype(np.float64)
```

It then turns each episode's reward into per-frame training weights as the paper's are made
([quality labels](../reference/data.md#quality-labels)) and writes them as a scores file, which
`experiments/examples/keep_top.py --scores` reads.

For a contact-based reward over the whole pretraining data, start from the pretraining data's own per-frame
labels (`qvla data get corpus`), which hold the contact columns, as the paper's training weights do.

## Run it

On the published traces sample. It prints one line per episode (its key, its rows, the published
reward's mean and the new reward's mean), then the scores file's header:

```bash data timeout=1800
qvla data get traces-sample --dest corpus
python experiments/examples/relabel.py --traces corpus/traces_sample --out reach_weights.json
```

```text output
{"cap": 2.0, "horizon": 10, "n_episodes": 164, "n_neutral_rows": 5983, "n_rows": 64182, "reward_id": "example.reach", "tau": 10.0, "version": 1}
```

<!-- ci
The same on a synthetic sample of one task and two episodes:

```bash ci
python -m tests.docs.fixtures traces-sample sample
python experiments/examples/relabel.py --traces sample --out reach_weights.json
```

```text output
task_1--0  80 rows  published 0.6410  reach 0.7551
{"cap": 2.0, "horizon": 10, "n_episodes": 2, "n_neutral_rows": 62, "n_rows": 160, "reward_id": "example.reach", "tau": 10.0, "version": 1}
```
-->

Next: a scores file over the pretraining episodes is what
[Write a quality method](write-a-quality-method.md) reads with `--scores`.
