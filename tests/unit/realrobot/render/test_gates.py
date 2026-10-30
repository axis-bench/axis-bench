"""Each hard gate catches what it exists to catch, on a writer-built task with a consistent sidecar."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.realign.curves import PhaseCurve, write_curve_store
from axis_qvla.realrobot.render import droid8d, gates, labels, profile, resample, rig, writer
from axis_qvla.realrobot.render.report import joint_jumps


def _consistent_episode(n_knots=4, attempt=7001, separations=(0.098, 0.018), times=None):
    """Post-step rows 0.2 s apart, or rows at `times` (a step-count recording, opened one period early)."""
    kt = 0.0 + (np.arange(n_knots) + 1) * 0.2 if times is None else np.asarray(times, np.float64)
    lead = 0.0 if times is None else float(kt[0]) - rig.SOURCE_ROW_DT
    n_knots = len(kt)
    t_all = np.r_[lead, kt]
    grid, rows = resample.frame_grid(kt)
    T = len(grid)
    arm = np.tile(np.asarray(rig.PIXELMATCH_HOME), (n_knots, 1))
    u = np.linspace(1.0, 0.0, n_knots)
    state = np.tile(np.asarray(rig.PIXELMATCH_HOME, np.float32), (T, 1))
    action = np.concatenate([resample.zoh(t_all, np.vstack([arm[:1], arm]), grid),
                             resample.zoh(t_all, np.r_[255 * (1 - u[:1]), 255 * (1 - u)][:, None], grid)], 1)
    ee = np.zeros((T, 7), np.float32)
    ee[:, 3] = 1.0
    pinch = np.array([0.0, 0.0, 0.145])
    c2w = {k: np.zeros((T, 16), np.float32) for k in rig.CAMERA_KEYS}
    cam = profile.load_profile().camera
    R = np.asarray(rig.quat_to_mat(cam.quat))
    front = np.eye(4)
    front[:3, :3], front[:3, 3] = R, cam.pos
    c2w["third_person"][:] = front.reshape(-1)
    c2w["wrist"][:] = np.eye(4).reshape(-1)
    side = {"attempt_id": attempt, "n_knots": n_knots, "row_times": kt.tolist(), "lead_time": lead, "frames": T,
            "knot_rows": rows.tolist(), "source_tcp": [pinch.tolist()] * n_knots,
            "openness": u.tolist(), "pinch_in_base": pinch.tolist(),
            "gripper": {"open_separation": separations[0], "closed_separation": separations[1], "open_ctrl": 0.0,
                        "closed_ctrl": 255.0}, "workspace": "fr3_reference"}
    return writer.EpisodeData(attempt, state, action.astype(np.float32), ee, c2w, side)


def _writer(root, prompt="x"):
    """A task writer for the reference workspace, the profile kept beside the render as the renderer keeps it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / profile.FILE).write_text(profile.REFERENCE.read_text())
    return writer.TaskWriter(root, 9001, prompt, cameras=profile.load_profile().cameras, workspace="fr3_reference")


def _black(T):
    return ({k: np.zeros((rig.HEIGHT, rig.WIDTH, 3), np.uint8) for k in ("third_person", "wrist")} for _ in range(T))


@pytest.fixture
def task(tmp_path):
    w = _writer(tmp_path / "out")
    ep = _consistent_episode()
    frames = ({k: np.zeros((rig.HEIGHT, rig.WIDTH, 3), np.uint8) for k in ("third_person", "wrist")} for _ in range(12))
    w.add_episode(ep, frames)
    return w.close()


def test_every_hard_gate_passes_on_a_consistent_task(task):
    cameras = profile.load_profile().cameras
    for gate in (gates.gate_structure(task), gates.gate_intrinsics(task, cameras), gates.gate_home(task),
                 gates.gate_tcp(task), gates.gate_gripper(task, open_separation=0.098, closed_separation=0.018)):
        assert gate["ok"], gate
    assert gates.gate_tcp(task)["episodes"][0]["convergence_knot"] == 1


def _rewrite(task, column, fn):
    p = task / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(p)
    width = t.schema.field(column).type.list_size
    a = np.asarray(t.column(column).combine_chunks().flatten()).reshape(-1, width).copy()
    fn(a)
    pq.write_table(t.set_column(t.schema.get_field_index(column), column, writer._fsl(a)), p)


def test_the_home_gate_catches_a_moved_first_frame(task):
    _rewrite(task, "observation.state", lambda a: a.__setitem__((1, 0), a[1, 0] + 1e-3))
    assert not gates.gate_home(task)["ok"]


def test_the_home_gate_holds_frame_2_of_a_post_step_recording(task):
    _rewrite(task, "observation.state", lambda a: a.__setitem__((2, 0), a[2, 0] + 1e-3))
    assert not gates.gate_home(task)["ok"]


def test_the_home_gate_reads_the_frames_up_to_the_first_recorded_row(tmp_path):
    """A step-count recording 0.62 s long: its rows land on frames 2, 5, 8 and 11, frame 2 is 0.02 s past
    row 0 and on its way to row 1, so frames 0 and 1 hold the home and frame 2 need not."""
    w = _writer(tmp_path / "out")
    w.add_episode(_consistent_episode(times=[0.0, 0.2, 0.4, 0.62]), _black(12))
    task = w.close()
    for gate in (gates.gate_structure(task), gates.gate_home(task),
                 gates.gate_gripper(task, open_separation=0.098, closed_separation=0.018)):
        assert gate["ok"], gate
    _rewrite(task, "observation.state", lambda a: a.__setitem__((2, 0), a[2, 0] + 1e-3))
    assert gates.gate_home(task)["ok"]
    _rewrite(task, "observation.state", lambda a: a.__setitem__((1, 0), a[1, 0] + 1e-3))
    res = gates.gate_home(task)
    assert not res["ok"] and any("frames 0..1" in f for f in res["failures"]), res


def test_the_structure_gate_catches_knot_rows_off_the_recorded_times(tmp_path):
    w = _writer(tmp_path / "out")
    w.add_episode(_consistent_episode(times=[0.0, 0.2, 0.4, 0.62]), _black(12))
    task = w.close()
    _edit_sidecar(task, lambda d: d.update(knot_rows=[2, 5, 8, 10]))
    res = gates.gate_structure(task)
    assert not res["ok"] and any("recorded times" in f for f in res["failures"]), res


def test_the_intrinsics_gate_catches_a_changed_focal(task):
    _rewrite(task, "observation.cameras.wrist.intrinsics", lambda a: a.__setitem__((0, 0), 300.0))
    assert not gates.gate_intrinsics(task, profile.load_profile().cameras)["ok"]


def test_the_gripper_gate_catches_a_shifted_switch_and_wrong_extremes(task):
    assert gates.gate_gripper(task, open_separation=0.098, closed_separation=0.018)["ok"]
    assert not gates.gate_gripper(task, open_separation=0.0983, closed_separation=0.018)["ok"]   # 0.3 mm off
    assert not gates.gate_gripper(task, open_separation=0.098, closed_separation=0.0149)["ok"]
    _rewrite(task, "action", lambda a: a.__setitem__((5, 7), a[4, 7]))
    assert not gates.gate_gripper(task, open_separation=0.098, closed_separation=0.018)["ok"]


def _edit_sidecar(task, fn):
    side = task / "meta" / "axis_episodes" / "episode_000000.json"
    doc = json.loads(side.read_text())
    fn(doc)
    side.write_text(json.dumps(doc))


def test_the_gripper_gate_catches_an_inverted_ctrl_range(task):
    _edit_sidecar(task, lambda d: d["gripper"].update(open_ctrl=255.0, closed_ctrl=0.0))
    res = gates.gate_gripper(task, open_separation=0.098, closed_separation=0.018)
    assert not res["ok"] and any("ctrl range" in f for f in res["failures"]), res


def test_the_intrinsics_gate_catches_a_moved_exterior_camera(task):
    _rewrite(task, "observation.cameras.third_person.camera_to_world", lambda a: a.__setitem__((slice(None), 3), a[:, 3] + 0.01))
    res = gates.gate_intrinsics(task, profile.load_profile().cameras)
    assert not res["ok"] and any("exterior camera" in f for f in res["failures"]), res


def test_the_gates_check_the_exterior_pose_against_the_profile_kept_beside_the_render(task):
    moved = profile.REFERENCE.read_text().replace("pos = [1.0, -0.13, 0.55]", "pos = [1.05, -0.13, 0.55]")
    (task.parent / profile.FILE).write_text(moved)
    with pytest.raises(gates.GateError, match="intrinsics"):
        gates.run_gates(task.parent, open_separation=0.098, closed_separation=0.018)
    res = json.loads((task.parent / "render_gates.json").read_text())["tasks"]["9001"]["intrinsics"]
    assert any("exterior camera" in f for f in res["failures"]), res


def test_gates_on_a_directory_without_tasks_say_so_before_reading_a_profile(tmp_path):
    with pytest.raises(FileNotFoundError, match="no task_"):
        gates.run_gates(tmp_path)


def _tcp_task(tmp_path, n_knots, off):
    """A task whose source TCP sits `off[k]` metres above the rendered pinch at knot k (0 elsewhere)."""
    ep = _consistent_episode(n_knots=n_knots)
    tcp = [[0.0, 0.0, 0.145 + off.get(k, 0.0)] for k in range(n_knots)]
    ep = dataclasses.replace(ep, sidecar={**ep.sidecar, "source_tcp": tcp})
    w = _writer(tmp_path / "out")
    w.add_episode(ep, _black(3 * n_knots))
    return w.close()


def test_knots_the_arm_cannot_reach_are_reported_when_the_ik_accepted_the_episode(tmp_path):
    """Two of ten knots 55 mm off the source TCP: 80% of knots are within the IK's 2 cm, so the episode
    passes, and the two knots over the 3 mm convergence threshold are reported."""
    task = _tcp_task(tmp_path, 10, {6: 0.055, 7: 0.055})
    res = gates.gate_tcp(task)
    assert res["ok"], res
    assert res["episodes"][0]["convergence_knot"] == 1 and res["episodes"][0]["reach_pct"] == pytest.approx(80.0)
    rep = gates.tcp_reported(task)[0]
    assert rep["knots_over_gate_after"] == 2 and rep["max_mm_after"] == pytest.approx(55.0, abs=1e-3)
    assert rep["median_mm_after"] == pytest.approx(0.0, abs=1e-3)


def test_the_tcp_gate_fails_an_episode_the_ik_would_have_refused(tmp_path):
    """Five of ten knots 55 mm off: 50% within 2 cm, under the IK's 60% minimum."""
    res = gates.gate_tcp(_tcp_task(tmp_path, 10, {k: 0.055 for k in range(5, 10)}))
    assert not res["ok"] and res["episodes"][0]["reach_pct"] == pytest.approx(50.0)
    assert any("50.0% of knots within 2 cm" in f for f in res["failures"]), res


def test_the_tcp_gate_checks_the_convergence_knot_comes_by_knot_8(tmp_path):
    """Knots 1..8 are 10 mm off: within the IK's 2 cm, over the 3 mm convergence threshold, so the TCP
    converges at knot 9 and only the convergence rule fails."""
    res = gates.gate_tcp(_tcp_task(tmp_path, 10, {k: 0.010 for k in range(1, 9)}))
    assert not res["ok"] and res["episodes"][0]["convergence_knot"] == 9, res
    assert res["episodes"][0]["reach_pct"] == pytest.approx(100.0) and res["n_failures"] == 1


def test_the_largest_arm_joint_jump_between_frames_is_reported(task):
    _rewrite(task, "observation.state", lambda a: a.__setitem__((7, 3), a[7, 3] + 0.5))
    jump = joint_jumps(task)[0]
    assert jump["max_rad"] == pytest.approx(0.5, abs=1e-6) and jump["frame"] in (7, 8) and jump["joint"] == 3


def test_run_gates_reports_the_tcp_errors_and_joint_jumps_without_gating_them(tmp_path):
    task = _tcp_task(tmp_path, 10, {6: 0.055})
    rep = gates.run_gates(task.parent, open_separation=0.098, closed_separation=0.018)
    assert rep["tasks"]["9001"]["tcp"]["ok"]
    assert rep["reported"]["tcp"]["9001"]["0"]["knots_over_gate_after"] == 1
    assert rep["reported"]["joint_jump"]["9001"]["0"]["max_rad"] == pytest.approx(0.0)
    on_disk = json.loads((task.parent / "render_gates.json").read_text())
    assert on_disk["reported"]["tcp"]["9001"]["0"]["max_mm_after"] == pytest.approx(55.0, abs=1e-3)


def test_the_structure_gate_catches_a_missing_attempt_id(task):
    p = task / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(p)
    pq.write_table(t.drop_columns(["attempt_id"]), p)
    assert not gates.gate_structure(task)["ok"]


def test_the_structure_gate_catches_a_sidecar_of_another_attempt(task):
    side = task / "meta" / "axis_episodes" / "episode_000000.json"
    doc = json.loads(side.read_text())
    doc["attempt_id"] = 7999
    side.write_text(json.dumps(doc))
    res = gates.gate_structure(task)
    assert not res["ok"] and any("7999" in f for f in res["failures"]), res


def test_the_structure_gate_catches_an_attempt_rendered_twice(tmp_path):
    w = _writer(tmp_path / "out")
    for _ in range(2):
        w.add_episode(_consistent_episode(), _black(12))
    res = gates.gate_structure(w.close())
    assert not res["ok"] and any("more than once" in f for f in res["failures"]), res


def test_the_structure_gate_holds_the_task_to_the_workspace_kept_beside_the_render(task):
    info = task / "meta" / "info.json"
    doc = json.loads(info.read_text())
    info.write_text(json.dumps({**doc, "workspace_profile": "my_lab"}))
    assert gates.gate_structure(task)["failures"] == [
        "info.json workspace_profile 'my_lab'; the render keeps workspace fr3_reference (workspace.toml)"]
    info.write_text(json.dumps(doc))
    _edit_sidecar(task, lambda side: side.update(workspace="my_lab"))
    assert gates.gate_structure(task)["failures"] == [
        "episode 0: sidecar workspace 'my_lab'; the render keeps workspace fr3_reference (workspace.toml)"]


def test_the_structure_gate_catches_a_missing_video(task):
    (task / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4").unlink()
    assert not gates.gate_structure(task)["ok"]


def test_run_gates_writes_the_report_and_raises_on_failure(task):
    _rewrite(task, "observation.state", lambda a: a.__setitem__((0, 0), 0.0))
    with pytest.raises(gates.GateError, match="home"):
        gates.run_gates(task.parent, open_separation=0.098, closed_separation=0.018)
    assert json.loads((task.parent / "render_gates.json").read_text())["tasks"]["9001"]["home"]["ok"] is False


@pytest.fixture
def two_attempts(tmp_path):
    """Two equal-length attempts, rendered, viewed as DROID-8D, with distinct attempt-keyed curves."""
    w = _writer(tmp_path / "render", "push the red box")
    for aid in (7001, 7002):
        w.add_episode(_consistent_episode(n_knots=8, attempt=aid), _black(24))
    w.close()
    droid8d.build_corpus(tmp_path / "render", tmp_path / "d8")
    curves = {(9001, 7001): PhaseCurve(np.linspace(0.0, 1.0, 24, dtype=np.float32), np.ones(24, bool), 6),
              (9001, 7002): PhaseCurve(np.r_[np.zeros(12), np.full(12, 0.3)].astype(np.float32), np.ones(24, bool), 12)}
    labels.write_attempt_store(tmp_path / "attempts", curves)
    return tmp_path, curves


def test_the_label_tag_gate_passes_on_a_store_keyed_through_attempt_ids(two_attempts):
    root, curves = two_attempts
    labels.render_curve_store(root / "d8", curves, root / "store")
    res = gates.gate_label_tag(root / "d8", root / "store", curves)
    assert res["ok"] and "tags_not_built" not in res and res["rows"] == 48, res


def test_the_label_tag_gate_catches_curves_attached_by_position(two_attempts):
    root, curves = two_attempts
    # the classic mis-attachment: episode 0 gets the curve of the attempt rendered second
    write_curve_store(root / "swapped", {(9001, 0): curves[(9001, 7002)], (9001, 1): curves[(9001, 7001)]})
    res = gates.gate_label_tag(root / "d8", root / "swapped", curves)
    assert not res["ok"], res


def test_a_one_episode_corpus_is_compared_on_ranges_and_weights_when_no_tags_can_be_built(tmp_path):
    """One episode has one quality bin, which the pinned tag builder refuses (its entropy diagnostic is
    0/0) for the render and the view alike: parity holds, and the reason is recorded."""
    w = _writer(tmp_path / "render", "push the red box")
    w.add_episode(_consistent_episode(n_knots=8), _black(24))
    w.close()
    droid8d.build_corpus(tmp_path / "render", tmp_path / "d8")
    curves = {(9001, 7001): PhaseCurve(np.linspace(0.0, 1.0, 24, dtype=np.float32), np.ones(24, bool), 6)}
    labels.render_curve_store(tmp_path / "d8", curves, tmp_path / "store")
    res = gates.gate_label_tag(tmp_path / "d8", tmp_path / "store", curves)
    assert res["ok"] and "constant" in res["tags_not_built"], res
    bad = {(9001, 7001): PhaseCurve(np.zeros(24, np.float32), np.ones(24, bool), 6)}
    assert not gates.gate_label_tag(tmp_path / "d8", tmp_path / "store", bad)["ok"]   # weights still compared


def test_the_label_tag_gate_fails_by_name_on_a_mistyped_curves_path(two_attempts):
    """A `--curves` path that does not exist (or holds no store) must fail the gate by name, not raise:
    `read_curve_store`'s `CurveError` is caught inside `gate_label_tag`, same as every other mismatch it
    reports in `failures`."""
    root, curves = two_attempts
    res = gates.gate_label_tag(root / "d8", root / "no-such-curves-dir", curves)
    assert not res["ok"] and any("curve store" in f for f in res["failures"]), res


def test_run_gates_fails_the_label_tag_gate_by_name_and_still_writes_the_report(two_attempts):
    """The CLI scenario this guards: a mistyped `--curves` path must still produce `render_gates.json`
    with `label_tag` recorded as failed, and `run_gates` must raise the named `GateError` -- never an
    uncaught `CurveError` traceback."""
    root, curves = two_attempts
    labels.write_attempt_store(root / "attempts", curves)
    with pytest.raises(gates.GateError, match="label_tag"):
        gates.run_gates(root / "render", droid8d_root=root / "d8", render_store=root / "no-such-curves-dir",
                        attempt_curves_dir=root / "attempts")
    report = json.loads((root / "render" / "render_gates.json").read_text())
    assert report["label_tag"]["ok"] is False


def test_the_gates_command_fails_by_name_on_a_mistyped_curves_path(two_attempts, capsys):
    from axis_qvla.realrobot.render.cli import main
    root, curves = two_attempts
    labels.write_attempt_store(root / "attempts", curves)
    rc = main(["gates", "--out", str(root / "render"), "--droid8d", str(root / "d8"),
              "--curves", str(root / "no-such-curves-dir"), "--attempt-curves", str(root / "attempts")])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err
    report = json.loads((root / "render" / "render_gates.json").read_text())
    assert report["label_tag"]["ok"] is False


def test_the_gates_command_refuses_the_label_tag_flags_given_apart(two_attempts, capsys):
    from axis_qvla.realrobot.render.cli import main
    root, _ = two_attempts
    assert main(["gates", "--out", str(root / "render"), "--droid8d", str(root / "d8")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL ") and "--droid8d, --curves and --attempt-curves" in err
    assert not (root / "render" / "render_gates.json").exists()


def test_the_label_tag_gate_names_an_episode_without_a_curve(two_attempts):
    root, curves = two_attempts
    write_curve_store(root / "partial", {(9001, 0): curves[(9001, 7001)]})
    res = gates.gate_label_tag(root / "d8", root / "partial", curves)
    assert not res["ok"] and any("episode 1" in f for f in res["failures"]), res


@pytest.mark.parametrize("field", ["t0", "valid"])
def test_the_label_tag_gate_catches_a_store_curve_whose_t0_or_mask_is_not_the_attempts(two_attempts, field):
    """`realign build` reads t0 and the mask (balanced weights); the rank file and tags do not."""
    root, curves = two_attempts
    c = curves[(9001, 7001)]
    wrong = dataclasses.replace(c, t0=1) if field == "t0" else dataclasses.replace(c, valid=np.r_[False, c.valid[1:]])
    write_curve_store(root / "wrong", {(9001, 0): wrong, (9001, 1): curves[(9001, 7002)]})
    res = gates.gate_label_tag(root / "d8", root / "wrong", curves)
    assert not res["ok"] and any(field in f for f in res["failures"]), res


@pytest.mark.parametrize("message", ["episode row blocks are not strictly ascending across the corpus",
                                     "row 99 is outside the corpus of 48 rows",
                                     "H(q) is zero: the same words, raised from somewhere else"])
def test_a_tag_builder_refusal_other_than_constant_quality_fails_the_gate(two_attempts, monkeypatch, message):
    """Both corpora refused with the same text is NOT parity: only the entropy diagnostic's H(q) = 0
    refusal, recognised by the frame that raised it, may pass -- never by its message."""
    import axis_qvla.realrobot.realign.tags as tags

    def refuse(*_a, **_k):
        raise ValueError(message)

    root, curves = two_attempts
    labels.render_curve_store(root / "d8", curves, root / "store")
    monkeypatch.setattr(tags, "build_pure_tags", refuse)
    res = gates.gate_label_tag(root / "d8", root / "store", curves)
    assert not res["ok"] and "tags_not_built" not in res and any(message in f for f in res["failures"]), res


def test_the_gates_cli_exits_0_on_a_passing_render_and_1_on_a_failing_one(tmp_path, capsys):
    from axis_qvla.cli import main
    w = _writer(tmp_path / "out")
    w.add_episode(_consistent_episode(separations=(rig.GRIPPER_OPEN_SEPARATION_M, rig.GRIPPER_CLOSED_SEPARATION_M)),
                  _black(12))
    task = w.close()
    assert main(["realrobot", "render", "gates", "--out", str(tmp_path / "out")]) == 0
    _rewrite(task, "observation.state", lambda a: a.__setitem__((0, 0), 0.0))
    assert main(["realrobot", "render", "gates", "--out", str(tmp_path / "out")]) == 1
    assert "home" in capsys.readouterr().err
    assert json.loads((tmp_path / "out" / "render_gates.json").read_text())["tasks"]["9001"]["home"]["ok"] is False


def test_the_report_says_whether_the_gripper_was_the_locked_model(task):
    """The gripper gate checks pad separations, which another 2F-85-shaped model can match: the report
    and the command say whether the render used the locked model."""
    report = gates.run_gates(task.parent, open_separation=0.098, closed_separation=0.018)
    assert report["gripper_lock"] == {"locked": False, "commits": []}
    for side in sorted((task / "meta" / "axis_episodes").glob("episode_*.json")):
        doc = json.loads(side.read_text())
        doc["gripper"]["lock_commit"] = "7" * 40
        side.write_text(json.dumps(doc))
    report = gates.run_gates(task.parent, open_separation=0.098, closed_separation=0.018)
    assert report["gripper_lock"] == {"locked": True, "commits": ["7" * 40]}
    assert json.loads((task.parent / "render_gates.json").read_text())["gripper_lock"]["locked"] is True


def test_the_gates_command_says_on_stderr_what_a_pass_did_not_certify(tmp_path, capsys):
    from axis_qvla.realrobot.render.cli import main
    w = _writer(tmp_path / "render", "push the red box")
    w.add_episode(_consistent_episode(n_knots=8, separations=(rig.GRIPPER_OPEN_SEPARATION_M,
                                                              rig.GRIPPER_CLOSED_SEPARATION_M)), _black(24))
    w.close()
    droid8d.build_corpus(tmp_path / "render", tmp_path / "d8")
    curves = {(9001, 7001): PhaseCurve(np.linspace(0.0, 1.0, 24, dtype=np.float32), np.ones(24, bool), 6)}
    labels.write_attempt_store(tmp_path / "attempts", curves)
    labels.render_curve_store(tmp_path / "d8", curves, tmp_path / "store")
    assert main(["gates", "--out", str(tmp_path / "render"), "--droid8d", str(tmp_path / "d8"),
                 "--curves", str(tmp_path / "store"), "--attempt-curves", str(tmp_path / "attempts")]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("all hard gates pass")
    assert "label_tag compared ranges and weights only" in err and "H(q) = 0" in err
    assert "not the locked 2F-85" in err
