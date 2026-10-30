"""The run specs a quality method's author never writes: its pretraining and its fine-tune.

Why: the benchmark protocol fixes everything after the method.
- The pretraining is the paper's for that backbone (experiments/examples/runs/<backbone>.pretrain.keep_top.toml),
  with the method's compiled labels named by name.
- The fine-tune is the paper's own fine-tune run spec for that backbone, started from the pretraining
  run: BC's plain one, or with `--cond` the tagged one the paper uses for conditioned pretrains.
Writing either reads no data, so `--print` plans both on any machine. What the labels decide
(the budget, and for SmolVLA / X-VLA the overrides their exported files require) is completed by
`complete_from_labels` when the run is bound on the machine that has them. A key the spec sets
itself is kept, and binding refuses it if it disagrees.

Core tier at import; loading compiled labels imports numpy inside the functions.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import types

from axis_qvla.core.paths import Paths
from axis_qvla.train import ids, names, runs
from axis_qvla.train.armspec import ArmSpec, dumps_armspec, load_armspec, loads_armspec
from axis_qvla.train.paper import RUNS_DIR
from axis_qvla.train.recipes import load_recipe

FAMILIES = {"pi05": "pi05-openpi", "smolvla": "smolvla-lerobot", "xvla": "xvla-lerobot"}
EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "examples" / "runs"
# The paper's fine-tune of each backbone: BC's (no tag), and the tagged one of its conditioned pretrains.
FINETUNE_TEMPLATES = {
    ("pi05-openpi", False): "pi05/finetune/pi05_axis_heldout_qual_v2_bc__qual_v2_bc.toml",
    ("pi05-openpi", True): "pi05/finetune/pi05_axis_heldout_qual_v2_cfgt__night12_cfgpt.toml",
    ("smolvla-lerobot", False): "smolvla/finetune/mvft_smolvla_bc.toml",
    ("smolvla-lerobot", True): "smolvla/finetune/mvft_smolvla_cfgphase.toml",
    ("xvla-lerobot", False): "xvla/finetune/mvft_xvla_bc.toml",
    ("xvla-lerobot", True): "xvla/finetune/mvft_xvla_cfgphase.toml",
}
# The pretraining seed of each backbone's first Table I replicate, the default of `qvla train pretrain
# --seed` (the shipped examples' [recipe] seed). The paper's replicates: 42/43/44 for pi0.5, 0/1/2 for
# SmolVLA and X-VLA (experiments/paper/paper.json table1.replicates).
PAPER_SEED = {"pi05": 42, "smolvla": 0, "xvla": 0}
# The paper's pretraining budget every backbone's method pretraining is held to (20,605 x 64).
PAPER_PRETRAIN = "pi05.pretrain.full.paper"
# The paper's schedule-replaying SmolVLA pretrains draw the schedule's 64 per step on one GPU
# (experiments/paper/runs/smolvla/pretrain/mv_smolvla_drop_top_*.toml); X-VLA's keep the preset's world.
SCHEDULE_WORLD = {"smolvla-lerobot": {"batch_size": 64, "parallelism.dp_replicate": 1}}
# The token budget the paper's tagged LeRobot runs reserve for the quality suffix.
TAG_TOKEN_BUDGET = {"smolvla-lerobot": 48, "xvla-lerobot": 50}
# The weighting block of the paper's SmolVLA / X-VLA AWR pretrains.
AWR_WEIGHTING = {"sample_weighting.type": "axis_flat", "sample_weighting.epsilon": 1e-06,
                 "sample_weighting.head_mode": "sparse", "sample_weighting.kappa": 0.01}


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")      # a run id is a directory name under <runs>


class MethodRunError(ValueError):
    """The labels or the pretraining run cannot make the protocol's run."""


def seed_suffix(family: str, seed: int | None) -> str:
    """`.s<seed>` for a pretraining seed other than the paper's first replicate's (PAPER_SEED), else
    "": two replicates of one method never share a run directory, and the id says which it is."""
    return "" if seed is None or family not in PAPER_SEED or seed == PAPER_SEED[family] else f".s{seed}"


def pretrain_run_id(family: str, run: str, seed: int | None = None) -> str:
    """<family>.pretrain.<run>[.s<seed>] (seed_suffix)."""
    return f"{family}.pretrain.{run}{seed_suffix(family, seed)}"


_LABEL = {"pi05-openpi": "pi0.5", "smolvla-lerobot": "SmolVLA", "xvla-lerobot": "X-VLA"}


def _cell_name(arm: ArmSpec) -> str:
    """The paper cell a fine-tune spec fills (`bc43` of "bc43|bare"), else its run id."""
    cells = arm.paper.get("cells")
    return cells[0].split("|", 1)[0] if isinstance(cells, list) and cells and isinstance(cells[0], str) else arm.id


def paper_finetune_seed(backbone: str, pretrain_seed: int) -> tuple[int, str]:
    """The fine-tune seed the paper's convention gives a pretraining at `pretrain_seed`, and the
    paper's seed table it comes from, read from the paper's own fine-tune run specs of `backbone`
    (experiments/paper/runs/<family>/finetune: each one's [recipe] seed against its [paper]
    pretrain_seed), not from a table kept here. The table maps each pretrain seed the paper trained
    to the fine-tune seed its fine-tunes use, naming the paper cells that use another one: pi0.5
    42 -> 42, 43 -> 42 (bc43: 43), 44 -> 42; SmolVLA and X-VLA 0 -> 0, 1 -> 1, 2 -> 2. At a pretrain
    seed the paper trained, the seed most of its fine-tunes use (the smaller one on a tie); at another,
    the rule every paper pretrain seed follows."""
    family = next(f for f, b in FAMILIES.items() if b == backbone)
    seen: dict[int, dict[int, list[str]]] = {}
    for f in sorted((RUNS_DIR / family / "finetune").glob("*.toml")):
        arm = load_armspec(f)
        pre = arm.paper.get("pretrain_seed")
        if isinstance(pre, int) and not isinstance(pre, bool):
            seen.setdefault(pre, {}).setdefault(arm.seed, []).append(_cell_name(arm))
    if not seen:
        raise MethodRunError(f"no paper fine-tune of {backbone} records its pretrain seed under {RUNS_DIR / family}")
    mode = {p: min(c, key=lambda s: (-len(c[s]), s)) for p, c in seen.items()}
    label = _LABEL[backbone]

    def entry(p: int) -> str:
        other = [f"{', '.join(sorted(set(seen[p][s])))}: {s}" for s in sorted(seen[p]) if s != mode[p]]
        return f"{p} -> {mode[p]}" + (f" ({'; '.join(other)})" if other else "")

    table = f"the paper's {label} fine-tunes, pretrain seed -> fine-tune seed: {'; '.join(map(entry, sorted(seen)))}"
    if pretrain_seed in seen:
        return mode[pretrain_seed], table
    if all(m == p for p, m in mode.items()):
        return pretrain_seed, f"{table}; any other pretrain seed -> itself"
    if len(set(mode.values())) == 1:
        s = next(iter(mode.values()))
        return s, f"{table}; any other pretrain seed -> {s}"
    raise MethodRunError(f"the paper's {label} fine-tunes follow no one seed rule ({mode}); pretrain at one of "
                         f"its seeds ({', '.join(map(str, sorted(seen)))})")


def finetune_run_id(pretrain_id: str, *, cond: bool, run: str | None = None) -> str:
    family, stage, rest = (pretrain_id.split(".", 2) + ["", ""])[:3]
    if not _RUN_ID.fullmatch(pretrain_id) or stage != "pretrain" or family not in FAMILIES or not rest:
        raise MethodRunError(f"{pretrain_id} is not a pretraining run id (<backbone>.pretrain.<name>)")
    return f"{family}.finetune.{run or rest}" + (".cond" if cond else "")


def lerobot_overrides(backbone: str, bundle, recipe) -> dict:
    """The [recipe.overrides] a LeRobot pretrain on `bundle` needs, read from the files it exports."""
    import numpy as np

    from axis_qvla.signals.lerobot_files import record_meta
    from axis_qvla.train.bind import _lerobot_records
    from axis_qvla.train.lerobot_cfg import WEIGHTS_FIELD

    mapped, problems = _lerobot_records(bundle)
    if problems:
        raise MethodRunError("; ".join(problems))
    over = {}
    for field, rec in mapped:
        meta = record_meta(rec) or {}
        if field == "axis_schedule_path":
            over.update({k: v for k, v in (("axis_expected_mode", meta.get("mode")),
                                           ("axis_expected_reward", meta.get("reward_id"))) if v is not None})
            over.update(SCHEDULE_WORLD.get(backbone, {}))
        elif field == "axis_quality_path":
            if meta.get("reward_id") is not None:
                over["axis_expected_reward"] = meta["reward_id"]
            over["axis_quality_token_budget"] = TAG_TOKEN_BUDGET[backbone]
            if not bool((np.asarray(bundle.legacy_arrays.get("tags", ())) == 0).any()):
                over["axis_quality_unconditional"] = False
        elif field == WEIGHTS_FIELD:
            over.update({**AWR_WEIGHTING, "sample_weighting.extra_params.expected_frames": int(bundle.rowspace.n_rows)})
    if int(bundle.budget.steps) != recipe.steps:
        over["steps"] = int(bundle.budget.steps)
    return over


def pretrain_spec(family: str, labels: str, *, run: str | None = None, seed: int | None = None) -> ArmSpec:
    """The paper's pretraining of `family` on the compiled labels named `labels`, at training seed
    `seed` (default the paper's first replicate's, PAPER_SEED). Reads no data. The seed is the
    replicate key `results contrast` pairs on (decision 5: replicates pair on the PRETRAIN seed);
    the fine-tune takes the seed the paper's convention gives it (paper_finetune_seed) and records
    this one (eval.json)."""
    if family not in FAMILIES:
        raise MethodRunError(f"backbone {family!r} is not one of {', '.join(FAMILIES)}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise MethodRunError(f"--seed must be an integer in [0, 2**32), got {seed!r}")
    template = load_armspec(EXAMPLES / f"{family}.pretrain.keep_top.toml")
    return dataclasses.replace(template, id=pretrain_run_id(family, run or labels, seed),
                               bundle=f"{names.LABEL_PREFIX}{labels}", overrides=types.MappingProxyType({}),
                               seed=template.seed if seed is None else seed)


def complete_from_labels(spec: ArmSpec, paths: Paths, *, smoke: bool) -> ArmSpec:
    """`spec`, its labels resolved to content, with what the labels decide: a real run's plan must be
    the paper's budget (refused with the compile flags otherwise), and a SmolVLA / X-VLA run gains
    every override its labels' exported files require that it does not set itself."""
    from axis_qvla.signals.bundle import load_bundle
    from axis_qvla.train.bind import _first_present

    recipe = load_recipe(spec.preset)
    bundle = load_bundle(_first_present(ids.bundle_dirs(paths, spec.bundle)))
    steps, batch = int(bundle.budget.steps), int(bundle.budget.global_batch)
    pi05 = spec.backbone == "pi05-openpi"
    # Every backbone's method pretraining replays a plan of the paper's pretraining budget (pi0.5's
    # preset; a LeRobot run trains the plan's steps, so its preset's own count is not the budget).
    want = recipe.steps if pi05 else load_recipe(PAPER_PRETRAIN).steps
    if not smoke and (batch != recipe.global_batch or steps != want):
        raise MethodRunError(f"{spec.id}: its labels were compiled for {steps} steps x {batch}; the paper trains it "
                             f"for {want} x {recipe.global_batch}: compile them again with --steps "
                             f"{want} --batch {recipe.global_batch} (a --smoke run trains a prefix of any plan)")
    if pi05:
        return spec
    derived = lerobot_overrides(spec.backbone, bundle, recipe)
    return dataclasses.replace(spec, overrides=types.MappingProxyType({**derived, **spec.overrides}))


def finetune_spec(pretrain: ArmSpec, *, cond: bool, run: str | None = None) -> ArmSpec:
    """The paper's fine-tune of the pretraining run `pretrain` on the 24 held-out tasks, at the seed the
    paper's convention gives that pretraining seed (paper_finetune_seed). Reads no data. A `run` name
    keeps the pretraining's `.s<seed>` suffix, so two replicates' fine-tunes never share a directory."""
    if pretrain.stage != "pretrain":
        raise MethodRunError(f"{pretrain.id} is a {pretrain.stage} run; fine-tune a pretraining run")
    suffix = seed_suffix(pretrain.id.split(".", 1)[0], pretrain.seed)
    if run is not None and suffix and not run.endswith(suffix):
        run += suffix
    ft_seed, _ = paper_finetune_seed(pretrain.backbone, pretrain.seed)
    t = load_armspec(RUNS_DIR / FINETUNE_TEMPLATES[(pretrain.backbone, cond)])
    # Every input by name, and no [data] asset_id: the recipe's data.repo_id is the same dataset
    # (axis_qvla.train.generate.openpi_asset_id falls back to it).
    return dataclasses.replace(
        t, id=finetune_run_id(pretrain.id, cond=cond, run=run), init=f"run:{pretrain.id}", paper_ref=None,
        paper=types.MappingProxyType({}), rowspace=names.published_name(t.rowspace, names.PUBLISHED_ROWSPACES),
        norm=None if t.norm is None else names.published_name(t.norm, names.PUBLISHED_NORMS),
        bundle=None if t.bundle is None else names.published_labels_name(t.bundle),
        data=types.MappingProxyType({k: v for k, v in t.data.items() if k != "asset_id"}), seed=ft_seed)


def write_spec(paths: Paths, spec: ArmSpec) -> pathlib.Path:
    """<runs>/<id>/spec.toml; an existing one with other contents is refused, never replaced. The text
    is read back as a run spec first, so a name that makes no valid run (an id with a `/`, labels
    named with capitals) is refused before anything is written."""
    text = dumps_armspec(spec)
    loads_armspec(text, source=f"the run spec for {spec.id!r}")
    dest = ids.run_root(paths, spec.id) / runs.PLANNED_SPEC
    try:
        old = dest.read_text(encoding="utf-8")
    except FileNotFoundError:
        old = None
    if old is not None and old != text:
        raise MethodRunError(f"{dest} already holds another run spec: pass --run <another name>, or remove "
                             f"{dest.parent} yourself")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dest)
    return dest
