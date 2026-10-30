"""The pi0.5 reference server's config, rendered from the paper run that trained the checkpoint.

Why: the pinned openpi fork registers no training config by name. Its serve_policy.py loads the
JSON TrainConfig description `qvla train` renders for a run (`--policy.config=<file>.json`), so the
config a paper checkpoint is served with is its run's, rendered for serving. A paper checkpoint
names its run through the checkpoint id cells.json records (experiments/paper/runs); a policy you
trained yourself is served with the config file you rendered, passed as `--config <file>.json`.

The paper's own eval commands (cells.json `eval_command`) name the openpi config each checkpoint
was trained under (`--config <name>`, the lock's `config`). One name can be the config of several
paper checkpoints (seed replicates), so a name resolves to a checkpoint id -- the one checkpoint
the lock gives that name, else the one of the result whose run is the command's `--run` -- and is
served exactly as that `--checkpoint-id` is.

Core tier: stdlib only (axis_qvla.train and axis_qvla.paper are core tier too; they are imported inside the
functions so that building the CLI parser imports nothing from them).
"""
from __future__ import annotations

from pathlib import Path

from axis_qvla.bench.sim.arms import out_dir_name
from axis_qvla.core.canon import dumps_canonical

SERVE_CONFIG = "openpi_config.json"
OPENPI_BACKBONE = "pi05-openpi"


class ServeConfigError(ValueError):
    """No paper arm trained the checkpoint, or the arm is not one openpi serves."""


def render_serve_config(checkpoint_id: str, paths) -> bytes:
    """The bytes of the serving config of the paper arm that trained `checkpoint_id` (a cells.json
    `checkpoint.id`). `paths` is the machine's axis_qvla.core.paths.Paths.

    Every refusal of the training package (no arm or two arms name the checkpoint, an arm file that
    does not read, an arm the generator does not render) is a ValueError or a LookupError there; it
    is raised here as a ServeConfigError naming the checkpoint id, so `bench eval` prints it and
    starts nothing."""
    from axis_qvla.train.armspec import load_armspec
    from axis_qvla.train.generate import openpi_config
    from axis_qvla.train.paper import find_arm_by_checkpoint
    from axis_qvla.train.recipes import load_recipe

    try:
        spec = load_armspec(find_arm_by_checkpoint(checkpoint_id))
    except (LookupError, ValueError, OSError) as exc:
        raise ServeConfigError(f"--checkpoint-id {checkpoint_id}: {exc}") from None
    if spec.backbone != OPENPI_BACKBONE:
        raise ServeConfigError(f"--checkpoint-id {checkpoint_id} was trained by {spec.id}, a {spec.backbone} "
                               f"run; only pi0.5 runs are served by openpi (--backend openpi)")
    # serve=True: the config a server loads, with no training bindings (a pi0.5 finetune names no
    # bundle; its plan is compiled only when it trains).
    try:
        cfg = openpi_config(spec, load_recipe(spec.preset), paths, serve=True)
    except (LookupError, ValueError) as exc:
        raise ServeConfigError(f"--checkpoint-id {checkpoint_id} ({spec.id}): {exc}") from None
    return dumps_canonical(cfg)


def checkpoint_for_config(config: str, arm: str | None) -> str | None:
    """The paper checkpoint id `--config <config>` names, or None when `config` is not the openpi
    config of any paper pi0.5 run (conf/release/paper_configs.lock.json): a config file, or a name
    some other openpi line registers, which the server is given as it is.

    A name that is the config of one checkpoint resolves to it. A name shared by several (seed
    replicates of one recipe) resolves through the paper cell whose run is `arm` -- the `--run` of
    that cell's eval_command -- to the checkpoint that cell names; with no such cell it is refused,
    naming the checkpoints, since serving one of them in place of another would file its rollouts
    under the wrong arm."""
    from axis_qvla.paper.cells import load_cells
    from axis_qvla.paper.cli import DEFAULT_CELLS
    from axis_qvla.train.parity import load_lock

    try:
        rows = load_lock()["pi05"].values()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ServeConfigError(f"--config {config}: the recorded paper configurations cannot be read ({exc})") from None
    candidates = sorted(r["checkpoint"] for r in rows if r.get("config") == config)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    try:
        cells = load_cells(DEFAULT_CELLS.read_bytes()).cells
    except (OSError, ValueError) as exc:
        raise ServeConfigError(f"--config {config} is the paper config of {len(candidates)} checkpoints, and "
                               f"the results map that tells them apart cannot be read ({exc})") from None
    named = sorted({c.checkpoint["id"] for c in cells if c.run == arm and c.checkpoint.get("id") in candidates})
    if len(named) == 1:
        return named[0]
    raise ServeConfigError(f"--config {config} is the paper config of {len(candidates)} checkpoints "
                           f"({', '.join(candidates)}), and no paper result of theirs has the run --run {arm!r}: "
                           f"pass --checkpoint-id <one of them> instead of --config")


def config_path(run_root: Path, arm: str) -> Path:
    """Where `bench eval` writes the config it hands the server: beside the arm's run manifest, in
    the directory the driver files the arm under (axis_qvla.bench.sim.arms.out_dir_name)."""
    return Path(run_root) / out_dir_name(arm) / "serve" / SERVE_CONFIG
