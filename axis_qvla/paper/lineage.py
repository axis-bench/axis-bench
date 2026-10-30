"""Per-cell lineage in cells.json: what each evaluated checkpoint was trained from, and how its
evaluated directory is identified.

Why: T0 proves every number comes from published rollouts; it does not say which training run
produced the checkpoint behind a cell. `lineage` records it: the paper ArmSpec that trained it
(`arm`, a run id under experiments/paper/runs/), the training config, its command-line overrides
(paths relative to the training data root), the step, the normalisation-statistics digest, the base
model, and the initial checkpoint (id, published ref, digest, seed). The checkpoint itself is
identified by its published file set (`checkpoint.pub_digest`).

`replicate_table` renders each π0.5 cell's seeds, training config and initial checkpoint as a
Markdown table.

Core tier: stdlib only.
"""
from __future__ import annotations

import re

LINEAGE_KEYS = ("train_config", "train_overrides", "step", "norm_stats_sha256", "base", "init", "arm")
INIT_KEYS = ("id", "ref", "pub_digest", "training_seed", "train_config", "train_overrides", "step",
             "norm_stats_sha256", "base")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# The paper run spec that trained the result's checkpoint (experiments/paper/runs/): every result is a
# fine-tune, so its arm is a finetune arm of one of the three backbones.
_ARM = re.compile(r"(pi05|smolvla|xvla)\.finetune\.[A-Za-z0-9_.-]+")


def _block_problems(where: str, block: dict) -> list[str]:
    out = []
    o = block["train_overrides"]
    if not (isinstance(o, list) and all(isinstance(x, str) for x in o)):
        out.append(f"{where}.train_overrides: a list of strings")
    if not (isinstance(block["step"], str) and block["step"]):
        out.append(f"{where}.step: a non-empty string")
    n = block["norm_stats_sha256"]
    if n is not None and not (isinstance(n, str) and _HEX64.fullmatch(n)):
        out.append(f"{where}.norm_stats_sha256: null or 64 hex")
    for k in ("train_config", "base"):
        if block[k] is not None and not isinstance(block[k], str):
            out.append(f"{where}.{k}: null or a string")
    return out


def lineage_problems(i: int, lin: object) -> list[str]:
    where = f"cells[{i}].lineage"
    if not isinstance(lin, dict) or set(lin) != set(LINEAGE_KEYS):
        return [f"{where}: needs exactly {list(LINEAGE_KEYS)}"]
    out = _block_problems(where, lin)
    if not (isinstance(lin["arm"], str) and _ARM.fullmatch(lin["arm"])):
        out.append(f"{where}.arm: the id of the finetune run spec that trained the checkpoint")
    init = lin["init"]
    if init is None:
        return out
    if not isinstance(init, dict) or set(init) != set(INIT_KEYS):
        return out + [f"{where}.init: null or exactly {list(INIT_KEYS)}"]
    if not (isinstance(init["ref"], str) and isinstance(init["pub_digest"], str)
            and _HEX64.fullmatch(init["pub_digest"])):
        out.append(f"{where}.init: needs a ref and a 64-hex pub_digest")
    return out + _block_problems(f"{where}.init", init)


def _cell(name: str) -> str:
    """A cell name as a Markdown table cell: its `|` (bc|bare) escaped, or it splits the row."""
    return "`" + name.replace("|", "\\|") + "`"


def replicate_table(cells: list[dict]) -> str:
    """Markdown rows for every π0.5 cell: its seeds, its training config and its initial checkpoint."""
    rows = ["| cell | pretrain seed | FT seed | FT config | initial checkpoint |", "|---|---|---|---|---|"]
    for c in cells:
        if c["backbone"] != "pi05":
            continue
        init, seed = c["lineage"]["init"], c["training_seed"]
        pre = seed["pretrain"] if seed["pretrain"] is not None else "—"
        first = f"`{init['id']}`" if init else "—"
        rows.append(f"| {_cell(c['cell'])} | {pre} | {seed['ft']} | `{c['lineage']['train_config']}` | {first} |")
    return "\n".join(rows)
