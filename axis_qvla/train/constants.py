"""The training budgets and row counts every paper arm ran with, by name: the registry pattern of
axis_qvla.labels.constants; a constants page is generated from it (tests/docs/constants_page.py).

Core tier: stdlib only.
"""
from __future__ import annotations

from axis_qvla.labels.constants import Constant

PAPER = "conf/recipes/tracks/paper.toml"
ROWSPACES = "conf/recipes/rowspaces.toml"
NOT_IN_PAPER = "implementation (not in the paper)"


def _budget(name: str, backbone: str, stage: str, key: str, value: int, meaning: str) -> Constant:
    return Constant(name, value, "App. A", meaning, PAPER, f"budgets/{backbone}/{stage}/{key}")


CONSTANTS: tuple[Constant, ...] = (
    Constant("PRETRAINING_FRAMES", 1_906_469, NOT_IN_PAPER, "frames of the 5,000-demonstration pretraining data "
             "(libero5k_v2_d8)", ROWSPACES, "libero5k_v2_d8/n_rows"),
    Constant("FINETUNE_FRAMES", 86_912, NOT_IN_PAPER, "trainable (ranges-cut) frames of the held-out fine-tuning "
             "set of 354 demonstrations (heldout24_ft)", ROWSPACES, "heldout24_ft/n_rows"),
    _budget("PI05_PRETRAIN_STEPS", "pi05-openpi", "pretrain", "steps", 20_605, "pi0.5 pretraining optimizer steps"),
    _budget("PI05_PRETRAIN_BATCH", "pi05-openpi", "pretrain", "global_batch", 64, "pi0.5 pretraining global batch"),
    _budget("PI05_FINETUNE_STEPS", "pi05-openpi", "finetune", "steps", 13_580, "pi0.5 finetuning optimizer steps"),
    _budget("PI05_FINETUNE_BATCH", "pi05-openpi", "finetune", "global_batch", 32, "pi0.5 finetuning global batch"),
    _budget("SMOLVLA_PRETRAIN_STEPS", "smolvla-lerobot", "pretrain", "steps", 20_621, "SmolVLA pretraining optimizer steps"),
    _budget("SMOLVLA_PRETRAIN_BATCH", "smolvla-lerobot", "pretrain", "global_batch", 64, "SmolVLA pretraining global batch"),
    _budget("SMOLVLA_FINETUNE_STEPS", "smolvla-lerobot", "finetune", "steps", 6_790, "SmolVLA finetuning optimizer steps"),
    _budget("SMOLVLA_FINETUNE_BATCH", "smolvla-lerobot", "finetune", "global_batch", 64, "SmolVLA finetuning global batch"),
    _budget("XVLA_PRETRAIN_STEPS", "xvla-lerobot", "pretrain", "steps", 20_621, "X-VLA pretraining optimizer steps"),
    _budget("XVLA_PRETRAIN_BATCH", "xvla-lerobot", "pretrain", "global_batch", 64, "X-VLA pretraining global batch"),
    _budget("XVLA_FINETUNE_STEPS", "xvla-lerobot", "finetune", "steps", 6_790, "X-VLA finetuning optimizer steps"),
    _budget("XVLA_FINETUNE_BATCH", "xvla-lerobot", "finetune", "global_batch", 64, "X-VLA finetuning global batch"),
    Constant("PI05_PRETRAIN_WARMUP", 2_060, "App. A", "pi0.5 pretraining warmup steps, then a constant learning rate",
             "conf/recipes/pi05.pretrain.full.paper.toml", "fields/lr_schedule.warmup_steps"),
    Constant("PI05_FINETUNE_WARMUP", 500, "App. A", "pi0.5 finetuning warmup steps, then a constant learning rate",
             "conf/recipes/pi05.finetune.lora.paper.toml", "fields/lr_schedule.warmup_steps"),
    Constant("SMOKE_STEPS", 20, NOT_IN_PAPER, "steps of a smoke run (a check of the training path, not a result)",
             "axis_qvla.train.smoke", "STEPS"),
    Constant("SMOKE_BATCH", 8, NOT_IN_PAPER, "global batch of a smoke run, on one GPU", "axis_qvla.train.smoke", "BATCH"),
)
