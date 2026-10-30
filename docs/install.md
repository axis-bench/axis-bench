# Install

One way, for every goal: clone the repository with its submodules, install it editable with the extra of
your goal, and write the `qvla.toml` that tells every command where its data goes.

```bash setup
git clone --recurse-submodules https://github.com/axis-bench/axis-bench
cd axis-bench
pip install -e ".[reproduce]"
```

The extra is your goal's; `reproduce` is the one for reproducing the paper's tables:

| your goal | extra |
|---|---|
| reproduce the paper's tables | `reproduce` |
| re-evaluate the released checkpoints, evaluate your own model | `eval` |
| write a quality method, pretrain and fine-tune with it | `method` |
| run it on a real robot | `realrobot` |

`qvla init` writes `./qvla.toml` with every data root under `./srv` (`--root DIR` for another
directory); `qvla bench check` then prints the benchmark it found:

```bash ci
qvla init
qvla bench check
```

```text output
OK    wrote …/qvla.toml (data under …/srv)
name             franka_heldout24
n_tasks          24
```

## Environments

Reproducing the tables and writing a quality method need nothing more. Evaluation, each training backbone
and the real-robot tools run in an environment of their own; build it once with one command. A command
that needs an environment you have not built names the command that builds it.

| your goal | build |
|---|---|
| re-evaluate a checkpoint, evaluate your own model | `qvla env build eval` |
| pretrain, fine-tune or serve π0.5 | `qvla env build pi05` |
| pretrain, fine-tune or serve SmolVLA | `qvla env build smolvla` |
| pretrain, fine-tune or serve X-VLA | `qvla env build xvla` |
| re-render the simulation demonstrations for your workspace | `qvla env build render` |
| convert your real-world demonstrations | `qvla env build droid_convert` |

SmolVLA and X-VLA share one environment, so either command builds both.

Next: [choose a goal](index.md).
