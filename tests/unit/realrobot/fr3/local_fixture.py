"""A local real-world layout over the realign driver's synthetic rendered corpus, for the run-spec tests and
the tutorial's CI blocks: d8/, built/, compiled labels, norm statistics, and a demonstrations set written
from a DROID-layout recording by `realdata training-set`."""
from __future__ import annotations

import json
from pathlib import Path

from axis_qvla.core.paths import Paths
from axis_qvla.realrobot.fr3 import norm
from axis_qvla.realrobot.realdata.training_set import write_training_set
from axis_qvla.realrobot.realign.bundles import write_bundles
from axis_qvla.realrobot.realign.cli import build
from tests.unit.realrobot.realdata.test_training_set import write_droid_recording
from tests.unit.realrobot.realign.test_driver import _dumped_roots_fixture
from tests.unit.train._arms import host

# The recorded demonstrations: "stack the blocks" twice (50 and 40 frames) and "press the button" once (60),
# which become task_2 and task_1. Tail 10 leaves 120 trainable frames, more than one global batch of 64.
DEMO_LENGTHS, DEMO_TASKS = (50, 60, 40), (0, 1, 0)


def _add_droid8d_columns(corpus: Path) -> None:
    """Give the synthetic corpus the DROID-8D view's 8-wide state and action, and its roots.json, as
    `render droid8d` writes them, so its norm statistics are computed on it."""
    import numpy as np
    import pyarrow.parquet as pq

    from axis_qvla.realrobot.realdata.training_set import _fsl
    tasks = sorted(p for p in corpus.glob("task_*") if p.is_dir())
    for task in tasks:
        for f in sorted((task / "data").rglob("*.parquet")):
            t = pq.read_table(f)
            rng = np.random.default_rng(len(t))
            t = t.append_column("observation.state", _fsl(rng.normal(size=(len(t), 8))))
            pq.write_table(t.append_column("action", _fsl(rng.uniform(-1, 1, (len(t), 8)))), f)
    roots = {p.name.split("_")[1]: p.name for p in tasks}
    (corpus / "roots.json").write_text(json.dumps({"version": 2, "base": ".", "roots": roots}))


def build_cache(cache: Path, recording: Path, *, runs=("bc", "conditioning")) -> None:
    """The tutorial's local layout in the cache root `cache`, with `runs`' compiled labels; `recording` is
    where the DROID-layout demonstrations are written first."""
    cache = Path(cache)
    rw = cache / "derived" / "realworld"
    roots, store, _tid, _n = _dumped_roots_fixture(rw / "fx")
    _add_droid8d_columns(rw / "fx" / "corpus")
    (rw / "d8").symlink_to("fx/corpus", target_is_directory=True)
    # The synthetic corpus's six episodes fill per-phase fit pools of a few hundred frames.
    build(rw / "built", roots=roots, curves_store=store, valid_from_data=True, runs=tuple(runs), min_fit_pool=100)
    write_bundles(rw / "built", cache / "bundles", runs=tuple(runs))
    norm.write_norm(cache / "derived" / "norm" / "norm_realalign_sim" / "qvla-v1-realworld" / "norm_stats.json",
                    norm.compute_norm(rw / "d8", "realalign_sim"))
    write_training_set(write_droid_recording(recording, DEMO_LENGTHS, DEMO_TASKS), rw / "demos")
    norm.write_norm(cache / "derived" / "norm" / "norm_fr3_droid" / "fr3_demonstrations" / "norm_stats.json",
                    norm.compute_norm(rw / "demos", "fr3_droid"))


def build_local(root: Path, *, runs=("bc", "conditioning")) -> tuple[Path, Paths]:
    """(the qvla.toml, its Paths) with the tutorial's local layout under <cache>/derived/realworld/."""
    toml, paths = host(Path(root))
    build_cache(paths.cache, Path(root) / "droid", runs=runs)
    return toml, paths
