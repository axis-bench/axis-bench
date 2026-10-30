"""Where the real-world line keeps what it builds on this machine: one fixed name per directory.

Why: each step of the real-world tutorial reads what the step before it wrote, and `fr3 local-runs`
reads one exact layout. Here every directory has a fixed name under the roots of
qvla.toml. A step finds its inputs without a flag, and a stored path never names this machine:

    <corpus>/qvla-v1-realworld/              the published simulation demonstrations (`qvla data get realworld`)
    <cache>/derived/realworld/source/        the render input tree (`render prepare`)
                              render/        the re-render: one LeRobot task per task id
                              d8/            its DROID-8D training view
                              attempts/      phase curves by attempt (`realign phase`)
                              curves/        the curve store keyed by the render's episodes (`render labels`)
                              built/         the stage-1 runs' labels (`realign build`)
                              lerobot/       openpi's DROID converter output (its HF_LEROBOT_HOME)
                              demos/         your demonstrations in the training layout (`realdata training-set`)
                              runs/          the run specs `fr3 local-runs` writes
    <cache>/derived/norm/norm_<embodiment>/<asset id>/norm_stats.json   each stage's norm statistics
    <cache>/bundles/                         compiled labels and frame indexes
    <runs>/<run id>/                         a run: train/ (openpi's checkpoints), serve.json, drive/

Core tier at import (stdlib and the core-tier axis_qvla.train.ids and axis_qvla.data.layout). The line record and
the recipes are read inside `Layout.norm_file`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from axis_qvla.data.layout import default_dest
from axis_qvla.train import ids

REPO = Path(__file__).resolve().parents[2]
LINE = REPO / "experiments" / "realrobot" / "line.toml"
ROBOTS = REPO / "third_party" / "axis-render" / "assets" / "robots"   # robotiq_2f85/ at the locked commit
REALWORLD = "qvla-v1-realworld"
LOCAL = "realworld"
CONVERTED = "local/fr3_demonstrations"       # openpi's converter REPO_NAME: its dataset is lerobot/<CONVERTED>
SERVE_CONFIG = "serve.json"
# Each stage's norm statistics and the data they describe: (artifact name, data directory, embodiment).
NORMS = {"pretrain": ("norm_realalign_sim", "d8", "realalign_sim"),
         "finetune": ("norm_fr3_droid", "demos", "fr3_droid")}
STAGE_OF = {emb: stage for stage, (_name, _data, emb) in NORMS.items()}


def run_id(stage: str, name: str) -> str:
    """`cond` -> `pi05.<stage>.realrobot_cond`; a full run id is kept as it is."""
    return name if name.startswith("pi05.") else f"pi05.{stage}.realrobot_{name}"


def _sub(name: str) -> property:
    return property(lambda self: self.base / name, doc=f"<cache>/derived/realworld/{name}")


@dataclasses.dataclass(frozen=True)
class Layout:
    paths: object       # axis_qvla.core.paths.Paths
    base: Path          # <cache>/derived/realworld

    source, render, d8, attempts, curves, built, lerobot, demos, runs_dir = (
        _sub(n) for n in ("source", "render", "d8", "attempts", "curves", "built", "lerobot", "demos", "runs"))

    @property
    def data(self) -> Path:
        return default_dest(REALWORLD, self.paths)

    @property
    def converted(self) -> Path:
        return self.lerobot / CONVERTED

    @property
    def bundles(self) -> Path:
        return ids.local_bundles_root(self.paths)

    def run_spec(self, rid: str) -> Path:
        return self.runs_dir / f"{rid}.toml"

    def run_dir(self, rid: str) -> Path:
        return ids.run_root(self.paths, rid)

    def checkpoint(self, rid: str) -> Path:
        return ids.lerobot_output_dir(self.paths, rid) / "final"

    def serve_config(self, rid: str) -> Path:
        return self.run_dir(rid) / SERVE_CONFIG

    def drive_kit(self, rid: str) -> Path:
        return self.run_dir(rid) / "drive"

    def norm_file(self, stage: str) -> Path:
        """The norm statistics `fr3 local-runs` reads for every run of `stage`. There is one file per
        stage, at the asset id the runs' training config renders (axis_qvla.train.generate.openpi_asset_id).
        ValueError when the stage's runs disagree on it."""
        from axis_qvla.realrobot.fr3.line import load_line, run_spec
        from axis_qvla.train.generate import openpi_asset_id
        from axis_qvla.train.recipes import load_recipe, resolved_fields

        line = load_line(LINE)
        specs = [run_spec(line, r) for r in sorted(line.runs)]
        assets = {str(openpi_asset_id(s, resolved_fields(load_recipe(s.preset), s))) for s in specs if s.stage == stage}
        if len(assets) != 1:
            raise ValueError(f"the {stage} runs of {LINE.name} render {len(assets)} norm asset ids "
                             f"({', '.join(sorted(assets))}); one is expected")
        return ids.local_derived(self.paths, f"norm/{NORMS[stage][0]}") / assets.pop() / "norm_stats.json"


def layout(host: str | None = None) -> Layout:
    """This machine's layout, from `--host` or ./qvla.toml (axis_qvla.data.userconf). HostProfileError without one."""
    from axis_qvla.data.userconf import user_paths

    paths = user_paths(host)
    return Layout(paths, ids.local_derived(paths, LOCAL))
