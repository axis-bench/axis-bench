"""The smoke fixture: two corpus tasks of three episodes each, for a 20-step run per backbone.

    qvla train smoke-fixture [--host <qvla.toml>] --episodes <dir holding task_2013/ and task_2232/>

builds everything in this machine's local data (axis_qvla.data.layout.local_dest), never inside a
mirrored repository: in <cache>/derived/smoke/, the DROID 8-D form of both tasks (corpus_d8/), their
roots and ranges index (index/, written by the corpus index builders, axis_qvla.data.index) and one
ArmSpec per backbone (runs/); the row space and one SignalBundle per backbone in <cache>/bundles/
(written by the signal compiler's writer, axis_qvla.signals.bundle); and pi0.5 norm stats over the
fixture's frames in
<cache>/derived/norm/smoke_2x3/. The input is the per-task corpus in its published layout (a 9-D
state: seven joints and two finger widths), converted by axis_qvla.data.convert_droid_actions; the
videos are linked to the input's, never copied. A feature whose `names` differ between the two
tasks (`observation.body_pose` names each scene's own bodies) keeps no names in the fixture's copy
(axis_qvla.train.plain_copy.clear_differing_names), so the plain merge, which needs equal features,
accepts them; the build refuses if any other difference remains (axis_qvla.train.plain_copy).

The pi0.5 bundle exercises every consumer path at once: a 20 x 8 plan of openpi's row sampler
over the trainable rows, a loss weight in [0.5, 2] on every trainable row (reducer global_ratio),
and a quality code per sample, 80.75% tagged. The SmolVLA / X-VLA bundle carries the same plan and
weights (reducer per_rank_ratio) and the legacy row and weight files the LeRobot line reads. The
merged LeRobot copy (smoke/lerobot_merged/) is built in the LeRobot environment with
`python -m lerobot.scripts.axis_merge_datasets`.

Every arm is on the `smoke` track (conf/recipes/tracks/smoke.toml): its presets declare the
budget, 20 steps of a global batch of 8 on one GPU -- pi0.5 with its train state on one device
(`fsdp_devices = 1`), SmolVLA and X-VLA with a per-rank batch of 8 on a world of one -- so a smoke
fits one GPU per backbone without an arm overriding its batch.

The fixture is built beside its destination and moved there only when it is complete; an existing
fixture is refused, never replaced, and the norm stats are written once: a rebuild that computes
other statistics is refused while the file the earlier arms name by digest is there. The fixture
checks the training path end to end; the numbers it trains to mean nothing.

Core tier at import; numpy, pandas and the signal compiler are imported inside the functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import types

from axis_qvla.core.canon import dumps_canonical, sha256_bytes
from axis_qvla.core.canonical import canonical_name
from axis_qvla.core.paths import Paths
from axis_qvla.train import ids
from axis_qvla.train.armspec import ArmSpec, dumps_armspec
from axis_qvla.train.plain_copy import clear_differing_names, merge_problems

SMOKE_TASKS = (2013, 2232)
SMOKE_ROWSPACE = "smoke_2x3"
# Recorded in the fixture's row space, so part of its id: the module keeps the development root.
SMOKE_SOURCE = canonical_name("train.smoke") + ": corpus tasks 2013 and 2232, three episodes each"
SMOKE_DIR = "smoke"
STEPS = 20
BATCH = 8                    # the smoke track's global batch
TAG_FRACTION = 0.8075
ASSET_ID = "Devon018/Franka-Datasets-v2"
ARMS = ("pi05", "smolvla", "xvla")
# The weighting block of the paper's SmolVLA / X-VLA AWR arms; expected_frames is the fixture's.
_LEROBOT_WEIGHTING = {"sample_weighting.type": "axis_flat", "sample_weighting.epsilon": 1e-06,
                      "sample_weighting.head_mode": "sparse", "sample_weighting.kappa": 0.01,
                      "sample_weighting.progress_path": None}
_RAW_STATE = 9


def fixture_weights(n_rows: int, rows, *, seed: int = 1):
    """A weight in [0.5, 2] on every trainable row, NaN on every other."""
    import numpy as np

    w = np.full(n_rows, np.nan, dtype=np.float32)
    w[np.asarray(rows)] = np.random.default_rng(seed).uniform(0.5, 2.0, len(rows)).astype(np.float32)
    return w


def fixture_cond(*, steps: int = STEPS, batch: int = BATCH, seed: int = 2):
    """A code in 1..5 per sample, dropped to 0 (untagged) with probability 1 - TAG_FRACTION."""
    import numpy as np

    rng = np.random.default_rng(seed)
    codes = rng.integers(1, 6, size=(steps, batch, 1))
    codes[rng.random((steps, batch, 1)) >= TAG_FRACTION] = 0
    return codes.astype(np.uint8)


def norm_stats(state, actions) -> dict:
    """openpi's norm_stats.json body over the fixture's 8-D state and action columns."""
    import numpy as np

    def one(x):
        x = np.asarray(x, dtype=np.float64)
        return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist()}

    return {"norm_stats": {"state": one(state), "actions": one(actions)}}


def _arm(**kw) -> ArmSpec:
    for k in ("overrides", "data", "paper"):
        kw[k] = types.MappingProxyType(dict(kw.get(k) or {}))
    return ArmSpec(**kw)


def lerobot_overrides(n_rows: int) -> dict:
    """A LeRobot smoke arm's [recipe.overrides]: the paper's AWR weighting over the fixture's frames."""
    return {"axis_expected_frames": n_rows, "sample_weighting.extra_params.expected_frames": n_rows,
            **_LEROBOT_WEIGHTING}


def fixture_arms(rowspace: str, bundles: dict[str, str], norm: str, n_rows: int) -> dict[str, ArmSpec]:
    """The three smoke ArmSpecs over row space `rowspace`, with `bundles` {"pi05" | "lerobot": id}."""
    over = lerobot_overrides(n_rows)
    return {
        "pi05": _arm(id="pi05.pretrain.smoke", backbone="pi05-openpi", stage="pretrain", track="smoke",
                     init="base:pi05_droid", rowspace=rowspace, bundle=bundles["pi05"], norm=norm,
                     preset="pi05.pretrain.full.smoke", seed=42, overrides={}, data={"asset_id": ASSET_ID}),
        "smolvla": _arm(id="smolvla.pretrain.smoke", backbone="smolvla-lerobot", stage="pretrain", track="smoke",
                        init="base:smolvlm2_500m", rowspace=rowspace, bundle=bundles["lerobot"], norm=None,
                        preset="smolvla.pretrain.smoke", seed=0, overrides=over),
        "xvla": _arm(id="xvla.pretrain.smoke", backbone="xvla-lerobot", stage="pretrain", track="smoke",
                     init="base:xvla_base", rowspace=rowspace, bundle=bundles["lerobot"], norm=None,
                     preset="xvla.pretrain.smoke", seed=0, overrides=over),
    }


def fixture_bundles(rs) -> dict:
    """The pi0.5 and the LeRobot SignalBundle over the fixture row space `rs`."""
    import numpy as np

    from axis_qvla.signals.planners import PermutationPlanner
    from axis_qvla.signals.rowspace import trainable_rows
    from axis_qvla.signals.schema import BARE, Budget, CondVocab, LegacyFile, SignalBundle

    rows = trainable_rows(rs)
    budget = Budget(STEPS, BATCH)
    plan = PermutationPlanner("openpi-rowsampler").plan(rs, None, None, budget, 0)
    weights = fixture_weights(rs.n_rows, rows)

    def program(name):
        p = {"method": f"qvla.train.smoke.{name}", "version": 1, "params": {"steps": STEPS, "batch": BATCH}}
        return p, sha256_bytes(dumps_canonical(p))

    p, sha = program("pi05")
    vocab = CondVocab(("quality",), {"quality": {q: f"\nQuality: {q}" for q in range(1, 6)}}, rs.prompts)
    pi05 = SignalBundle(program=p, program_sha=sha, rowspace=rs, budget=budget, seed=0, plan=plan, weights=weights,
                        reducer="global_ratio", cond=fixture_cond(), vocab=vocab, serve=(BARE,),
                        requires=frozenset({"plan", "loss_weights", "cond_prompt"}))
    p, sha = program("lerobot")
    legacy = (LegacyFile("lerobot-legacy", None, "signals/smoke/lerobot_rows.npz", "npz",
                         {"$o": [["rows", {"$array": "legacy.rows"}]]}),
              LegacyFile("lerobot-legacy", None, "signals/smoke/lerobot_awr_flat_smoke.npz", "npz",
                         {"$o": [["weights", {"$array": "weights"}]]}))
    lerobot = SignalBundle(program=p, program_sha=sha, rowspace=rs, budget=budget, seed=0, plan=plan, weights=weights,
                           reducer="per_rank_ratio", serve=(BARE,), requires=frozenset({"plan", "loss_weights"}),
                           legacy=legacy, legacy_arrays={"rows": np.asarray(rows, dtype=np.int64)})
    return {"pi05": pi05, "lerobot": lerobot}


def state_width(task_dir: pathlib.Path) -> int:
    """The width of a task's `observation.state`, as its meta/info.json declares it."""
    info = json.loads((task_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    try:
        (width,) = info["features"]["observation.state"]["shape"]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{task_dir}/meta/info.json declares no 1-D observation.state") from None
    return int(width)


def droid8(src: pathlib.Path, dst: pathlib.Path) -> None:
    """`src`, a task in the 9-D layout, converted to the DROID 8-D one at `dst` (videos linked)."""
    from axis_qvla.data.convert_droid_actions import convert_dataset

    width = state_width(src)
    if width != _RAW_STATE:
        raise ValueError(f"{src}: observation.state is {width}-D; the fixture converts the pretraining data's "
                         f"own {_RAW_STATE}-D layout (seven joints and two finger widths) to DROID 8-D")
    convert_dataset(src, dst)


def write_once(path: pathlib.Path, data: bytes) -> None:
    """Write `data` at `path` unless it is there already with these bytes; other bytes are refused,
    since arms name the file by its digest."""
    try:
        old = path.read_bytes()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        tmp.write_bytes(data)
        tmp.replace(path)
        return
    if old != data:
        raise ValueError(f"{path} holds other statistics than this fixture's (sha256 "
                         f"{hashlib.sha256(old).hexdigest()[:12]}, this build's {hashlib.sha256(data).hexdigest()[:12]}); "
                         f"run specs name it by digest, so it is never replaced: remove it yourself and build again")


def _check_inputs(episodes: pathlib.Path, dest: pathlib.Path) -> None:
    """Refuse, before anything is written, a missing task and an existing fixture."""
    for tid in SMOKE_TASKS:
        info = episodes / f"task_{tid}" / "meta" / "info.json"
        if not info.is_file():
            raise FileNotFoundError(f"--episodes {episodes}: no task_{tid}/meta/info.json; give the "
                                    f"pretraining episodes directory holding task_{SMOKE_TASKS[0]}/ and "
                                    f"task_{SMOKE_TASKS[1]}/")
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; remove it yourself to build the fixture again")


def build_fixture(paths: Paths, episodes: pathlib.Path) -> dict[str, pathlib.Path]:
    """Build the fixture (module docstring); {"pi05" | "smolvla" | "xvla": its ArmSpec file}."""
    import numpy as np
    import pandas as pd

    from axis_qvla.data.index import ranges as ranges_index
    from axis_qvla.data.index.roots import roots_payload, select_roots
    from axis_qvla.signals.bundle import rowspace_id, write_bundle, write_rowspace
    from axis_qvla.signals.rowspace import build_rowspace

    episodes, dest = pathlib.Path(episodes), ids.local_derived(paths, SMOKE_DIR)
    _check_inputs(episodes, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo = pathlib.Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.build-"))
    try:
        corpus = repo / "corpus_d8"
        tasks = [corpus / f"task_{tid}" for tid in SMOKE_TASKS]
        for task in tasks:
            droid8(episodes / task.name, task)
        clear_differing_names(tasks)
        unmergeable = merge_problems(tasks)
        if unmergeable:
            raise ValueError(f"{episodes}: the tasks cannot form one merged LeRobot copy: " + "; ".join(unmergeable))
        index = repo / "index"
        index.mkdir()
        roots_file, ranges_file = index / "roots.json", index / "ranges.json"
        report = select_roots(corpus)
        if sorted(int(t) for t in report.roots) != list(SMOKE_TASKS):
            raise ValueError(f"{episodes}: the index builder kept tasks {sorted(report.roots)} "
                             f"(unusable: {list(report.unusable)})")
        roots_file.write_text(json.dumps(roots_payload(report, src_root=corpus, out=roots_file), indent=1))
        ranges_file.write_text(json.dumps(ranges_index.build(corpus)["ranges"]))
        rs = build_rowspace(name=SMOKE_ROWSPACE, corpus=corpus, roots_file=roots_file, ranges_file=ranges_file,
                            roots_path="index/roots.json", ranges_path="index/ranges.json", source=SMOKE_SOURCE)
        frames = pd.concat([pd.read_parquet(p) for t in SMOKE_TASKS
                            for p in sorted((corpus / f"task_{t}" / "data").rglob("*.parquet"))])
        stats = json.dumps(norm_stats(np.stack(frames["observation.state"].to_numpy()),
                                      np.stack(frames["action"].to_numpy())), indent=2).encode()
        norm_file = ids.local_derived(paths, f"norm/{SMOKE_ROWSPACE}") / ASSET_ID / "norm_stats.json"
        write_once(norm_file, stats)
        root = ids.local_bundles_root(paths)
        write_rowspace(rs, root / "rowspaces")
        bundles = {k: "sha256:" + write_bundle(b, root) for k, b in fixture_bundles(rs).items()}
        arms = fixture_arms(f"{rs.name}#sha256:{rowspace_id(rs)}", bundles,
                            f"{SMOKE_ROWSPACE}#sha256:{hashlib.sha256(stats).hexdigest()}", rs.n_rows)
        (repo / "runs").mkdir()
        for name, spec in arms.items():
            (repo / "runs" / f"{name}.toml").write_text(dumps_armspec(spec))
        os.rename(repo, dest)
    except BaseException:
        shutil.rmtree(repo, ignore_errors=True)
        raise
    return {name: dest / "runs" / f"{name}.toml" for name in ARMS}


def main(argv: list[str]) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data.userconf import user_paths

    p = argparse.ArgumentParser(prog="qvla train smoke-fixture", description="build the 2-task x 3-episode fixture")
    p.add_argument("--host", default=None, help="this machine's qvla.toml (default: ./qvla.toml, then $AXIS_HOST)")
    p.add_argument("--episodes", metavar="DIR", type=pathlib.Path, required=True,
                   help="the pretraining episodes directory holding task_2013/ and task_2232/")
    a = p.parse_args(argv)
    try:
        arms = build_fixture(user_paths(a.host), a.episodes)
    except (HostProfileError, ids.IdError, OSError, ValueError) as exc:
        print(f"FAIL  {exc}")
        return 2
    for name, path in arms.items():
        print(f"OK    {name}: {path}")
    return 0
