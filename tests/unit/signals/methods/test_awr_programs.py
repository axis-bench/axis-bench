"""The AWR built-ins write the bytes the legacy builders write, and an unscorable episode is declared."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from axis_qvla.labels.corpus import CorpusLabels
from axis_qvla.signals.blocks.awr import AwrWeightError
from axis_qvla.signals.bundle import export_legacy
from axis_qvla.signals.imported import ImportedScores
from axis_qvla.signals.legacy import LegacyTarget
from axis_qvla.signals.methods import awr_chunk, awr_episode
from axis_qvla.signals.methods import schedule as bis
from axis_qvla.signals.methods.awr import AwrChunk, AwrEpisode, AwrImported
from axis_qvla.signals.program import Unscorable
from axis_qvla.signals.schema import Budget
from axis_qvla.signals.views import lerobot as bls
from tests.unit.signals.toy_corpus import EPISODES, write_toy_corpus


def _target(kind, path, fmt="openpi-legacy") -> LegacyTarget:
    return LegacyTarget(lock_id=f"toy.{kind}", path=path, kind=kind, sha256="", format=fmt, generator="toy",
                        generator_sha256=None, roots_form="v2", numerics=None, tolerance=None, template_name=None)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """The toy corpus plus the per-frame label columns the AWR builders read.

    The toy episodes are 4-8 frames, so the object-centric tests use horizon 3: at the default 10 every
    window reaches the invalid last frame and every weight is neutral, which proves nothing. The pose
    columns are float32, as the corpus stores them: two bodies (`plate_0`, then `cup_0` at the
    origin) and a hand closing on `cup_0` by 3 cm a frame; `task_800--1` touches it from frame 1
    (the manipulation branch), `task_1000--0` over its last 3 frames, the others never (approach).
    """
    monkeypatch.delenv(bis.CORPUS_ROOT_ENV, raising=False)
    t = write_toy_corpus(tmp_path)
    rng = np.random.default_rng(1)
    bodies = np.asarray([0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32)
    for tid, eps, _prompt in EPISODES:
        rows = []
        for ep, _from, n, _attempt in eps:
            reward = np.cumsum(rng.uniform(0, 0.05, n))
            touch = {(800, 1): range(1, n), (1000, 0): range(n - 3, n)}.get((tid, ep), ())
            for f in range(n):
                rows.append({"episode_index": ep, "frame_index": f, "reward": float(reward[f]),
                             "reward_valid": f < n - 1, "contact_target_finger": f in touch, "contact_target_eef": False,
                             "observation.body_pose": bodies,
                             "observation.eef_pose": np.asarray([0.30 - 0.03 * f, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                                                np.float32)})
        d = tmp_path / "corpus" / f"task_{tid}" / "data" / "chunk-000"
        d.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(d / "file-000.parquet")
    return t


def _names_table(path) -> str:
    from axis_qvla.data import episode_names as en

    keys = [f"task_{tid}--{ep}" for tid, eps, _p in EPISODES for ep, *_ in eps]
    return en.write_table({k: (["plate_0", "cup_0"], "cup_0") for k in keys}, path, about="toy")


def _cli_chunk(toy, out, *extra, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["awr_chunk", "--index", str(toy.roots), "--out", str(out), *extra])
    assert awr_chunk.main() == 0


def test_object_centric_chunk_weights_are_the_builders_bytes(toy, tmp_path, monkeypatch):
    cli = tmp_path / "cli.json"
    _cli_chunk(toy, cli, "--reward", "v2", "--horizon", "3", monkeypatch=monkeypatch)
    prog = AwrChunk("v2", "openpi-rowsampler", "global_ratio", phase_balanced=False, horizon=3,
                    exports=(_target("awr_weights", "signals/toy/awr_weights.json"),))
    b = prog.compile(CorpusLabels(toy.root / "corpus"), toy.rs, Budget(4, 2), 42)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    assert ours.read_bytes() == cli.read_bytes()
    assert b.reducer == "global_ratio" and b.diagnostics["n_unscored_episodes"] == 0.0
    assert json.loads(cli.read_text())["n_neutral_rows"] < 23      # the toy windows are not all neutral


def test_an_unscorable_episode_is_neutral_by_declaration_or_raises(toy, tmp_path, monkeypatch):
    """The toy meta has no body_names, so the approach-aware scorer cannot bind any object."""
    cli = tmp_path / "cli.json"
    _cli_chunk(toy, cli, "--reward", "phase", "--binding", "names", monkeypatch=monkeypatch)
    prog = AwrChunk("phase", "openpi-rowsampler", "global_ratio", phase_balanced=False,
                    exports=(_target("awr_weights", "signals/toy/awr_weights.json"),))
    b = prog.compile(CorpusLabels(toy.root / "corpus"), toy.rs, Budget(4, 2), 42)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    assert ours.read_bytes() == cli.read_bytes()
    assert b.diagnostics["n_unscored_episodes"] == 4.0
    with pytest.raises(Unscorable):
        AwrChunk("phase", "openpi-rowsampler", "global_ratio", phase_balanced=False, on_unscorable="raise").compile(
            CorpusLabels(toy.root / "corpus"), toy.rs, Budget(4, 2), 42)


def test_phase_balancing_refuses_a_pool_too_small_to_fit():
    curve = np.linspace(0.0, 1.0, 40)
    with pytest.raises(AwrWeightError, match="fit pools too small"):
        awr_chunk.phase_balance([(0, 800, 0, curve, np.ones(40, bool), 20)], tau=10.0, cap=2.0, horizon=10)


def test_episode_weights_are_the_builders_bytes(toy, tmp_path, monkeypatch):
    chunk = tmp_path / "chunk.json"
    _cli_chunk(toy, chunk, "--reward", "v2", "--horizon", "3", monkeypatch=monkeypatch)
    cli = tmp_path / "ep.json"
    assert awr_episode.main(["--weights", str(chunk), "--ranges", str(toy.ranges), "--out", str(cli)]) == 0
    prog = AwrEpisode("openpi-rowsampler", "global_ratio",
                      exports=(_target("awr_weights", "signals/toy/awr_weights_ep.json"),))
    b = prog.compile(ImportedScores.from_json(chunk), toy.rs, Budget(4, 2), 42)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    assert ours.read_bytes() == cli.read_bytes()


def test_imported_weights_export_the_lerobot_flat_view(toy, tmp_path):
    cli = tmp_path / "flat.npz"
    assert bls.main(["flat", "--roots", str(toy.roots), "--ranges", str(toy.ranges), "--weights", str(toy.weights),
                     "--out", str(cli)]) == 0
    prog = AwrImported("lerobot-axisrowsampler", "per_rank_ratio",
                       exports=(_target("lerobot_awr_flat", "signals/toy/flat.npz", fmt="lerobot-legacy"),))
    b = prog.compile(ImportedScores.from_json(toy.weights), toy.rs, Budget(4, 2), 0)
    [ours] = export_legacy(b, "lerobot-legacy", tmp_path / "o")
    assert ours.read_bytes() == cli.read_bytes()
    assert b.reducer == "per_rank_ratio" and "loss_weights" in b.requires


def test_episode_weights_refuse_a_source_from_another_corpus(toy, tmp_path, monkeypatch):
    chunk = tmp_path / "chunk.json"
    _cli_chunk(toy, chunk, "--reward", "v2", monkeypatch=monkeypatch)
    doc = json.loads(chunk.read_text())
    doc["provenance"]["corpus_fingerprint"] = "sha256:" + "0" * 32
    chunk.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="other pretraining data"):
        AwrEpisode("openpi-rowsampler", "global_ratio").compile(ImportedScores.from_json(chunk), toy.rs, Budget(4, 2), 42)


def test_the_names_table_is_restricted_to_the_row_space_and_pinned(toy, tmp_path):
    from axis_qvla.data import episode_names as en

    table = {k: (["link0", "cup_0"], "cup_0") for k in ("task_800--0", "task_800--1", "task_800--2", "task_1000--0")}
    full, short = tmp_path / "names.json", tmp_path / "short.json"
    digest = en.write_table(table, full, about="toy")
    en.write_table({k: v for k, v in table.items() if k != "task_1000--0"}, short, about="toy")
    labels = CorpusLabels(toy.root / "corpus", names_table=full, names_sha256=digest)
    assert labels.names(toy.rs) == table
    assert labels.digests(toy.rs) == {"corpus_fingerprint": toy.rs.corpus_fingerprint, "episode_body_names": digest}
    with pytest.raises(ValueError, match="does not digest"):
        CorpusLabels(toy.root / "corpus", names_table=full, names_sha256="0" * 64).names(toy.rs)
    with pytest.raises(ValueError, match="1 episodes of toy are not in"):
        CorpusLabels(toy.root / "corpus", names_table=short).names(toy.rs)
    assert CorpusLabels(toy.root / "corpus").names(toy.rs) is None


def test_a_target_the_program_cannot_write_is_refused(toy):
    from axis_qvla.signals.exports import awr_json_file
    from axis_qvla.signals.legacy import LegacyError
    from axis_qvla.signals.methods.bc import BC

    scores = ImportedScores.from_json(toy.weights)
    with pytest.raises(LegacyError, match="writes none"):
        AwrImported("openpi-rowsampler", "global_ratio", exports=(_target("awr_weights", "signals/toy/w.json"),)).compile(
            scores, toy.rs, Budget(4, 2), 42)
    with pytest.raises(LegacyError, match="weighs no rows"):
        BC("lerobot-axisrowsampler", exports=(_target("lerobot_awr_flat", "signals/toy/f.npz", fmt="lerobot-legacy"),)
           ).compile(None, toy.rs, Budget(4, 2), 0)
    with pytest.raises(LegacyError, match="not a float32 value"):
        awr_json_file(_target("awr_weights", "signals/toy/w.json"), {"weights": {"task_800--0": [0.1]}}, name="awr")


@pytest.mark.parametrize("kwargs,match", [
    (dict(scorer="v3"), "scorer"), (dict(scorer="phase", on_unscorable="skip"), "on_unscorable"),
    (dict(scorer="v2"), "phase balancing"), (dict(scorer="phase", binding="mover"), "names binding")])
def test_an_awr_chunk_refuses_what_no_paper_arm_computed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AwrChunk(convention="openpi-rowsampler", loss_reducer="global_ratio", **kwargs)


def test_the_names_table_binds_the_object_the_meta_does_not_name(toy, tmp_path, monkeypatch):
    """The toy meta has no body_names; the names table binds `cup_0` in every episode, exactly as the
    builder's --names-map does."""
    table = tmp_path / "names.json"
    digest = _names_table(table)
    cli = tmp_path / "cli.json"
    _cli_chunk(toy, cli, "--reward", "phase", "--binding", "names", "--names-map", str(table), "--names-sha256", digest,
               "--horizon", "3", monkeypatch=monkeypatch)
    prog = AwrChunk("phase", "openpi-rowsampler", "global_ratio", phase_balanced=False, horizon=3,
                    exports=(_target("awr_weights", "signals/toy/awr_weights.json"),))
    b = prog.compile(CorpusLabels(toy.root / "corpus", names_table=table, names_sha256=digest), toy.rs, Budget(4, 2), 42)
    [ours] = export_legacy(b, "openpi-legacy", tmp_path / "o")
    assert ours.read_bytes() == cli.read_bytes()
    assert b.diagnostics["n_unscored_episodes"] == 0.0 and b.inputs["episode_body_names"] == digest
    doc = json.loads(cli.read_text())
    assert doc["provenance"]["params"]["binding"] == "names"
    assert any(w != 1.0 for ws in doc["weights"].values() for w in ws)      # bound, scored, not all neutral


def test_a_pose_column_that_is_not_float32_is_refused_by_name(toy, tmp_path):
    """The approach-aware distances are float32, as the corpus stores them; a float64 copy of a pose
    column would be scored in another precision, so it is refused, naming the column."""
    table = tmp_path / "names.json"
    digest = _names_table(table)
    f = toy.root / "corpus" / "task_800" / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(f)
    df["observation.body_pose"] = [np.asarray(v, np.float64) for v in df["observation.body_pose"]]
    df.to_parquet(f)
    with pytest.raises(TypeError, match="'observation.body_pose' holds \\['float64'\\], not float32"):
        AwrChunk("phase", "openpi-rowsampler", "global_ratio", phase_balanced=False, horizon=3).compile(
            CorpusLabels(toy.root / "corpus", names_table=table, names_sha256=digest), toy.rs, Budget(4, 2), 42)


def test_the_object_centric_reward_refuses_a_names_table_it_would_not_read(toy, tmp_path):
    table = tmp_path / "names.json"
    _names_table(table)
    with pytest.raises(ValueError, match="binds no object"):
        AwrChunk("v2", "openpi-rowsampler", "global_ratio", phase_balanced=False).compile(
            CorpusLabels(toy.root / "corpus", names_table=table), toy.rs, Budget(4, 2), 42)


def test_a_manifest_carries_the_fingerprint_it_is_given():
    from axis_qvla.signals.blocks.awr import manifest_payload, neutral_episode

    fp = "sha256:" + "ab" * 16
    doc = manifest_payload(iter([neutral_episode(800, 0, 3)]), reward_id="v2", tau=10.0, cap=2.0, horizon=10,
                           corpus_fingerprint=fp)
    assert list(doc)[-1] == "provenance" and doc["provenance"]["corpus_fingerprint"] == fp
    assert doc["provenance"]["params"] == {"cap": 2.0, "horizon": 10, "tau": 10.0}
    assert "provenance" not in manifest_payload(iter([neutral_episode(800, 0, 3)]), reward_id="v2", tau=10.0,
                                                cap=2.0, horizon=10)
