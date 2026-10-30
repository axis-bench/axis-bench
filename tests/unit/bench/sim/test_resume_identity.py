"""A resumed arm runs what the run whose seed files it keeps ran, for reference servers as for --policy.

The queue of an arm keeps every unit an earlier run finished, so a second `bench eval` of the arm
runs only what is left and then writes its own run_manifest.json and arm_summary.json over the
earlier seed files. Under the default arm name (<checkpoint dir basename>_<serving>) every step-13579
pi0.5 checkpoint is `13579_bare`, so "another checkpoint under the same arm" is the ordinary mistake:
it must be refused before the queue or the record is touched, naming what differs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from axis_qvla.bench import launch
from axis_qvla.bench.sim import campaign as drv
from tests.unit.bench.sim.driver_stubs import FakeProc, driver_stack


def _checkpoint(root: Path, weights: bytes) -> Path:
    """A checkpoint directory named like a paper step (13579), holding `weights`."""
    ck = root / "13579"
    (ck / "params").mkdir(parents=True)
    (ck / "params" / "w.bin").write_bytes(weights)
    return ck


def _argv(tmp_path: Path, arm: str, ck: Path | str, *extra: str, config: str = "cfg") -> list[str]:
    return ["--run", arm, "--config", config, "--checkpoint", str(ck), "--tasks", "806,1889",
            "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
            "--openpi-dir", str(tmp_path), "--n-seeds", "2", "--gpus", "0", "--workers-per-gpu", "1",
            "--task-data-root", str(tmp_path / "tasks"), *extra]


def _snapshot(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def first_run(tmp_path, monkeypatch):
    """Checkpoint A evaluated under its default arm name; returns what a second run needs."""
    seen = driver_stack(monkeypatch, tmp_path)
    ck_a = _checkpoint(tmp_path / "a", b"weights A")
    ck_b = _checkpoint(tmp_path / "b", b"weights B")
    arm = launch.filed_arm(launch.EvalRequest(checkpoint=str(ck_a), config="cfg"))
    assert arm == launch.filed_arm(launch.EvalRequest(checkpoint=str(ck_b), config="cfg")) == "13579_bare"
    assert drv.main(_argv(tmp_path, arm, ck_a, "--serving", "bare")) == 0
    out, queue = tmp_path / "r1" / f"out_{arm}", tmp_path / "r1" / f"q_{arm}"
    return {"seen": seen, "arm": arm, "a": ck_a, "b": ck_b, "out": out, "queue": queue,
            "records": _snapshot(out), "units": _snapshot(queue), "workers": len(seen["workers"])}


def _refused(tmp_path, run: dict, argv: list[str], match: str) -> None:
    with pytest.raises(SystemExit, match=match) as exc:
        drv.main(argv)
    assert "Use a new --run" in str(exc.value)
    # Refused before anything is touched: no worker, the queue and every record as they were.
    assert len(run["seen"]["workers"]) == run["workers"]
    assert _snapshot(run["queue"]) == run["units"]
    assert _snapshot(run["out"]) == run["records"]


def test_another_checkpoint_under_the_default_arm_name_is_refused(tmp_path, first_run):
    run = first_run
    old = json.loads(run["records"]["run_manifest.json"])["checkpoint"]["sha256"]
    _refused(tmp_path, run, _argv(tmp_path, run["arm"], run["b"], "--serving", "bare"),
             rf"checkpoint: 'sha256 {old}' -> 'sha256 [0-9a-f]{{64}}'\)\. The digest covers every file under "
             rf"the directory, download caches and training state included \(1 files, 9 bytes recorded; "
             rf"1 files, 9 bytes now\)")


def test_another_config_is_refused(tmp_path, first_run):
    run = first_run
    _refused(tmp_path, run, _argv(tmp_path, run["arm"], run["a"], "--serving", "bare", config="other_cfg"),
             r"config: 'cfg' -> 'other_cfg'")


def test_another_serving_or_backend_is_refused(tmp_path, first_run):
    run = first_run
    _refused(tmp_path, run, _argv(tmp_path, run["arm"], run["a"], "--serving", "conditioned"),
             r"serving\.serving: 'bare' -> 'conditioned'")
    _refused(tmp_path, run, _argv(tmp_path, run["arm"], run["a"], "--serving", "bare", "--backend", "smolvla"),
             r"serving\.backend: 'openpi' -> 'smolvla'")


def test_a_rendered_config_is_compared_by_its_bytes_not_its_path(tmp_path, monkeypatch):
    """A paper checkpoint's server config is a rendered file: the same bytes at the same path
    resume; other bytes are another config, however the file is named."""
    driver_stack(monkeypatch, tmp_path)
    ck = _checkpoint(tmp_path / "a", b"weights A")
    cfg = tmp_path / "serve.json"
    cfg.write_text('{"name": "one"}')
    assert drv.main(_argv(tmp_path, "probe", ck, "--serving", "bare", config=str(cfg))) == 0
    assert drv.main(_argv(tmp_path, "probe", ck, "--serving", "bare", config=str(cfg))) == 0
    cfg.write_text('{"name": "two"}')
    with pytest.raises(SystemExit, match=r"config: 'sha256 [0-9a-f]{64}' -> 'sha256 [0-9a-f]{64}'"):
        drv.main(_argv(tmp_path, "probe", ck, "--serving", "bare", config=str(cfg)))


def test_the_same_run_resumes(tmp_path, first_run):
    """A genuine resume -- same checkpoint bytes, config, serving and benchmark -- is not refused,
    even from a moved copy of the checkpoint: the path is not the model."""
    run = first_run
    moved = tmp_path / "moved"
    run["a"].rename(moved)
    assert drv.main(_argv(tmp_path, run["arm"], moved, "--serving", "bare")) == 0


def test_a_policy_record_is_not_resumed_by_a_reference_run(tmp_path, monkeypatch):
    driver_stack(monkeypatch, tmp_path)
    out = tmp_path / "r1" / "out_probe"
    out.mkdir(parents=True)
    recorded = {"serving": {"external": ["127.0.0.1:9000"], "handshakes": [{"cfg_conditioning": False}]}}
    (out / "run_manifest.json").write_text(json.dumps(recorded))
    with pytest.raises(SystemExit, match="records a --policy run under this name"):
        drv.main(_argv(tmp_path, "probe", _checkpoint(tmp_path, b"w"), "--serving", "bare"))
    assert json.loads((out / "run_manifest.json").read_text()) == recorded


@pytest.mark.parametrize("record", ["{not json", '["a list"]'])
def test_an_unreadable_record_refuses_a_reference_resume(tmp_path, monkeypatch, record):
    driver_stack(monkeypatch, tmp_path)
    out = tmp_path / "r1" / "out_probe"
    out.mkdir(parents=True)
    (out / "run_manifest.json").write_text(record)
    with pytest.raises(SystemExit, match="cannot verify"):
        drv.main(_argv(tmp_path, "probe", _checkpoint(tmp_path, b"w"), "--serving", "bare"))
    assert (out / "run_manifest.json").read_text() == record


def test_the_compared_keys_are_recorded_by_every_real_run(tmp_path, first_run, monkeypatch):
    """Every RESUME_IDENTITY key is present in a real run's record -- for a local checkpoint and
    for one named by reference alike -- so no compared key reads absent on both sides."""
    m = json.loads(first_run["records"]["run_manifest.json"])
    assert m["checkpoint"]["identity"] == "content"
    assert all(drv._resume_value(m, k) is not drv._MISSING for k in drv.RESUME_IDENTITY)
    out = tmp_path / "hub"
    assert drv.main(_argv(tmp_path, "hub", "orgA/smolvla_finetune", *SMOLVLA, "--out-dir", str(out),
                          "--queue-root", str(tmp_path / "q_hub"))) == 0
    m = json.loads((out / "run_manifest.json").read_text())
    assert all(drv._resume_value(m, k) is not drv._MISSING for k in drv.RESUME_IDENTITY)


# --- a checkpoint the server resolves by name (a Hub id, a URL) -------------------------------------

SMOLVLA = ("--serving", "bare", "--backend", "smolvla")


def _hub_run(tmp_path, ref: str, arm: str | None = None) -> tuple[str, Path]:
    """`--checkpoint <ref>` under its default arm name (or `arm`); returns (arm, out dir)."""
    arm = arm or launch.filed_arm(launch.EvalRequest(checkpoint=ref, config="cfg", backend="smolvla"))
    assert drv.main(_argv(tmp_path, arm, ref, *SMOLVLA)) == 0
    return arm, tmp_path / "r1" / f"out_{arm}"


def test_a_reference_is_recorded_by_its_exact_string_and_says_so(tmp_path, monkeypatch):
    driver_stack(monkeypatch, tmp_path)
    _, out = _hub_run(tmp_path, "orgA/smolvla_finetune@0123abc")
    ck = json.loads((out / "run_manifest.json").read_text())["checkpoint"]
    assert ck == {"path": "orgA/smolvla_finetune@0123abc", "ref": "orgA/smolvla_finetune@0123abc",
                  "identity": "name", "revision": "0123abc", "pinned": True}
    ck = drv.checkpoint_record("gs://openpi-assets/checkpoints/pi05_libero", external=False)
    assert (ck["identity"], ck["revision"], ck["pinned"]) == ("name", None, False)
    assert drv.checkpoint_record(str(tmp_path / "absent" / "13579"), external=False)["error"] == "missing"


def test_two_hub_ids_under_the_same_default_arm_are_refused(tmp_path, monkeypatch):
    """The re-review's probe: orgA's and orgB's `smolvla_finetune` share the default arm
    `smolvla_finetune_bare`; orgB's run must not file orgA's seeds under its name."""
    seen = driver_stack(monkeypatch, tmp_path)
    arm, out = _hub_run(tmp_path, "orgA/smolvla_finetune")
    assert arm == "smolvla_finetune_bare"
    run = {"seen": seen, "workers": len(seen["workers"]), "queue": tmp_path / "r1" / f"q_{arm}", "out": out}
    run["records"], run["units"] = _snapshot(out), _snapshot(run["queue"])
    _refused(tmp_path, run, _argv(tmp_path, arm, "orgB/smolvla_finetune", *SMOLVLA),
             r"checkpoint: 'reference orgA/smolvla_finetune' -> 'reference orgB/smolvla_finetune'\)\. "
             r"--checkpoint orgA/smolvla_finetune names no revision, so the same name can serve other "
             r"weights later: pin one, e\.g\. `hf download orgA/smolvla_finetune --revision <commit>")


def test_the_same_hub_id_at_another_revision_is_refused(tmp_path, monkeypatch):
    driver_stack(monkeypatch, tmp_path)
    arm, _ = _hub_run(tmp_path, "orgA/smolvla_finetune@rev1", arm="probe")
    with pytest.raises(SystemExit, match=r"checkpoint: 'reference orgA/smolvla_finetune@rev1' -> "
                                         r"'reference orgA/smolvla_finetune@rev2'") as exc:
        drv.main(_argv(tmp_path, arm, "orgA/smolvla_finetune@rev2", *SMOLVLA))
    assert "names no revision" not in str(exc.value)          # both are pinned: no advice to pin


def test_a_local_directory_and_a_hub_id_are_two_models(tmp_path, monkeypatch):
    driver_stack(monkeypatch, tmp_path)
    ck = _checkpoint(tmp_path / "a", b"weights A")
    assert drv.main(_argv(tmp_path, "probe", ck, *SMOLVLA)) == 0
    with pytest.raises(SystemExit, match=r"checkpoint: 'sha256 [0-9a-f]{64}' -> 'reference orgA/13579'"):
        drv.main(_argv(tmp_path, "probe", "orgA/13579", *SMOLVLA))
    with pytest.raises(SystemExit, match=r"checkpoint: 'sha256 [0-9a-f]{64}' -> <absent>\)\. This run "
                                         r"identifies no checkpoint: "):
        drv.main(_argv(tmp_path, "probe", tmp_path / "gone" / "13579", *SMOLVLA))


def test_the_same_unpinned_hub_id_resumes_with_a_warning(tmp_path, monkeypatch):
    driver_stack(monkeypatch, tmp_path)
    said: list[str] = []
    monkeypatch.setattr(drv, "log", said.append)
    arm, _ = _hub_run(tmp_path, "orgA/smolvla_finetune", arm="probe")
    assert any(m.startswith("WARNING --checkpoint orgA/smolvla_finetune names no revision") for m in said)
    said.clear()
    assert drv.main(_argv(tmp_path, arm, "orgA/smolvla_finetune", *SMOLVLA)) == 0
    assert any(m.startswith("WARNING resuming ") and "names no revision" in m for m in said)


def test_a_record_that_identifies_no_checkpoint_refuses_a_resume(tmp_path, monkeypatch):
    """A record with neither a digest nor a reference -- a run whose --checkpoint was not there, or
    a record from before references were recorded -- never reads as the same model, even against
    another such record."""
    driver_stack(monkeypatch, tmp_path)
    gone = tmp_path / "gone" / "13579"
    assert drv.main(_argv(tmp_path, "probe", gone, "--serving", "bare")) == 0
    m = json.loads((tmp_path / "r1" / "out_probe" / "run_manifest.json").read_text())
    assert m["checkpoint"] == {"path": str(gone), "error": "missing"}
    with pytest.raises(SystemExit, match=r"checkpoint: <absent> -> <absent>\)\. Neither run identifies a "
                                         r"checkpoint: "):
        drv.main(_argv(tmp_path, "probe", gone, "--serving", "bare"))
    ck = _checkpoint(tmp_path / "a", b"weights A")
    with pytest.raises(SystemExit, match=r"<absent> -> 'sha256 [0-9a-f]{64}'\)\. The earlier record identifies "
                                         r"no checkpoint: "):
        drv.main(_argv(tmp_path, "probe", ck, "--serving", "bare"))


def test_a_run_that_started_no_rollout_says_its_record_can_be_removed(tmp_path, monkeypatch):
    """A first attempt that died before any rollout (its queue claimed nothing) left a record and
    no seed file: the corrected retry is refused all the same, and told what to remove."""
    driver_stack(monkeypatch, tmp_path)
    out = tmp_path / "r1" / "out_probe"
    out.mkdir(parents=True)
    (out / "run_manifest.json").write_text(json.dumps({"checkpoint": {"path": "x", "error": "missing"}}))
    ck = _checkpoint(tmp_path / "a", b"weights A")
    with pytest.raises(SystemExit, match=r"That run started no rollout \(its queue records no claimed unit\), "
                                         rf"so nothing of it would be kept: remove {out / 'run_manifest.json'}"):
        drv.main(_argv(tmp_path, "probe", ck, "--serving", "bare"))
    (out / "run_manifest.json").unlink()
    assert drv.main(_argv(tmp_path, "probe", ck, "--serving", "bare")) == 0


def test_a_refused_resume_leaves_the_earlier_runs_layout_copy(tmp_path, monkeypatch):
    """The workers' layout copy is written after the resume check: a resume refused (here, another
    checkpoint) leaves the copy the earlier run recorded, byte for byte."""
    from axis_qvla.bench.spec import load_benchmark

    driver_stack(monkeypatch, tmp_path)
    layouts = tmp_path / "layouts.json"
    layouts.write_bytes(load_benchmark().layouts_path.read_bytes())
    ck_a, ck_b = _checkpoint(tmp_path / "a", b"A"), _checkpoint(tmp_path / "b", b"B")
    assert drv.main(_argv(tmp_path, "probe", ck_a, "--serving", "bare", "--layout-manifest", str(layouts))) == 0
    copy = tmp_path / "r1" / "out_probe" / "layout_manifest.harness.json"
    before = copy.read_bytes()
    recorded = json.loads((copy.parent / "run_manifest.json").read_text())["layout_manifest"]["sha256"]
    assert recorded == hashlib.sha256(before).hexdigest()
    doc = json.loads(layouts.read_text())
    doc["edited"] = True
    layouts.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="checkpoint: 'sha256"):
        drv.main(_argv(tmp_path, "probe", ck_b, "--serving", "bare", "--layout-manifest", str(layouts)))
    assert copy.read_bytes() == before


def test_a_relative_checkpoint_is_one_absolute_directory_for_the_record_and_the_server(tmp_path, monkeypatch):
    """The driver digests --checkpoint in its working directory and starts the server in the
    server's checkout: a relative path is made absolute once, so both read the same directory."""
    driver_stack(monkeypatch, tmp_path)
    started: list[str] = []
    monkeypatch.setattr(drv, "start_server", lambda gpu, port, cfg, ckpt, *a, **k: started.append(ckpt) or FakeProc())
    work = tmp_path / "work"
    ck = _checkpoint(work, b"weights A")
    monkeypatch.chdir(work)
    argv = _argv(tmp_path, "probe", "13579", "--serving", "bare")
    argv[argv.index("--openpi-dir") + 1] = str(tmp_path / "server_checkout")
    assert drv.main(argv) == 0
    m = json.loads((tmp_path / "r1" / "out_probe" / "run_manifest.json").read_text())
    assert started == [str(ck)] and m["checkpoint"]["path"] == str(ck)
    assert m["checkpoint"]["identity"] == "content" and m["checkpoint"]["n_files"] == 1
    assert drv.local_checkpoint("orgA/smolvla_finetune") == "orgA/smolvla_finetune"   # a reference stays as given
