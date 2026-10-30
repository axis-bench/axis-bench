"""Which stage-1 runs a build was produced with, and what each one needs.

OFFLINE TIER: stdlib only.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = ("bc", "conditioning", "awr", "filtering")

# What a build of every run writes beside curves/, runs.json and gates.json.
BUILT = ("roots.json", "ranges.json", "awr_weights_phase_rank.json", "awr_weights_phase.json",
         "awr_weights_phase_ep.json", "quality_phase.npz", "schedule_drop_top.npz")

# What each run reads beside roots.json and ranges.json. 'filtering' compiles from the episode weights;
# its drop-top schedule is gated as an artifact of its own.
RUN_FILES = {
    "conditioning": ("awr_weights_phase_rank.json", "quality_phase.npz"),
    "awr": ("awr_weights_phase.json",),
    "filtering": ("awr_weights_phase_ep.json", "schedule_drop_top.npz"),
}
RUNS_RECORD = "runs.json"   # written by every build: {"runs": [...]}


class UnbuiltRunError(ValueError):
    """A directory without a build's runs record, or `runs` naming a run the record does not list."""


def required_files(runs: tuple[str, ...]) -> tuple[str, ...]:
    """BUILT, filtered to what `runs` need (roots.json and ranges.json always), in BUILT's order."""
    needed = {"roots.json", "ranges.json"}
    for run in runs:
        needed.update(RUN_FILES.get(run, ()))
    return tuple(name for name in BUILT if name in needed)


def recorded_runs(built: Path) -> tuple[str, ...]:
    """The runs `built` was produced with, read from its `runs.json`."""
    try:
        doc = json.loads((Path(built) / RUNS_RECORD).read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        raise UnbuiltRunError(f"{built}: no runs.json; give a `realign build --out` directory") from None
    return tuple(r for r in RUNS if r in doc["runs"])


def write_recorded_runs(built: Path, runs: tuple[str, ...]) -> None:
    """Write `built`'s `runs.json`, normalised to `RUNS`' order."""
    (Path(built) / RUNS_RECORD).write_text(json.dumps({"runs": [r for r in RUNS if r in runs]}))


def check_runs_built(built: Path, runs: tuple[str, ...]) -> None:
    """Raise `UnbuiltRunError` unless `built`'s `runs.json` lists every run of `runs`."""
    recorded = recorded_runs(built)
    missing = [r for r in runs if r not in recorded]
    if missing:
        raise UnbuiltRunError(
            f"{built}: --runs names {missing}, which this build does not include "
            f"(it was built with {list(recorded)})")
