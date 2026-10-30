"""The serve check: what a policy server's handshake declares against what its finetune run trained on."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_qvla.realrobot.fr3 import checks, line
from tests.unit.realrobot.fr3.fr3_sets import edit_info, fr3_set, letterbox_frame

REPO = Path(__file__).resolve().parents[4]
LN = line.load_line(REPO / "experiments" / "realrobot" / "line.toml")
ARM = "pi05.finetune.realrobot_cond"
CROP = {"base_0_rgb": "center_crop224", "left_wrist_0_rgb": "center_crop224"}
LETTERBOX = {"base_0_rgb": "letterbox224", "left_wrist_0_rgb": "letterbox224"}


def _set(tmp_path):
    """The run's finetune set as the paper's is: centre-cropped 224 frames."""
    return fr3_set(tmp_path / "fr3_392")


def _handshake(**over):
    h = {"supports_client_noise": True, "action_horizon": 15, "action_dim": 32,
         "image_geometry": dict(CROP),
         "state_order": ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "gripper"],
         "action_units": "droid_normalized_joint_velocity_command", "action_fps": 15, "action_dim_out": 8,
         "prompt_suffix": "\nQuality: 5", "guidance_scale": 0.0}
    h.update(over)
    return h


def test_a_conforming_handshake_passes(tmp_path):
    assert checks.check_serve(_handshake(), LN, ARM, _set(tmp_path)) == []


@pytest.mark.parametrize("key, value, word", [
    ("image_geometry", {"base_0_rgb": "letterbox224", "left_wrist_0_rgb": "center_crop224"}, "geometry"),
    ("state_order", ["gripper", "q0", "q1", "q2", "q3", "q4", "q5", "q6"], "state order"),
    ("action_units", "rad_s", "units"),
    ("action_fps", 10, "fps"),
    ("prompt_suffix", "", "prompt"),
    ("guidance_scale", 1.5, "guidance"),
    ("action_dim_out", 7, "action dim"),
    ("action_horizon", 10, "action horizon"),
])
def test_each_mismatch_is_refused(tmp_path, key, value, word):
    assert any(word in p for p in checks.check_serve(_handshake(**{key: value}), LN, ARM, _set(tmp_path)))


def test_a_camera_without_a_declared_geometry_is_named(tmp_path):
    problems = checks.check_serve(_handshake(image_geometry={"base_0_rgb": "center_crop224"}), LN, ARM,
                                  _set(tmp_path))
    assert problems == ["image left_wrist_0_rgb: the handshake declares no geometry for it"]


def test_a_handshake_without_the_serving_keys_names_each_missing_one(tmp_path):
    bare = {"supports_client_noise": True, "action_horizon": 15, "action_dim": 32}
    problems = checks.check_serve(bare, LN, ARM, _set(tmp_path))
    assert {p.split()[-1] for p in problems} >= {"image_geometry", "state_order", "action_units"}


def test_a_check_server_report_is_read_for_its_handshake(tmp_path):
    report = {"endpoint": "localhost:8000", "ok": False, "handshake": _handshake()}
    assert checks.check_serve(report, LN, ARM, _set(tmp_path)) == []


def test_the_bare_finetunes_serve_without_the_tag(tmp_path):
    root = _set(tmp_path)
    assert checks.check_serve(_handshake(prompt_suffix=""), LN, "pi05.finetune.realrobot_bc", root) == []
    assert checks.check_serve(_handshake(), LN, "pi05.finetune.realrobot_only", root) != []


def test_only_finetune_runs_are_served(tmp_path):
    assert checks.check_serve(_handshake(), LN, "pi05.pretrain.realrobot_cond", _set(tmp_path)) == [
        "pi05.pretrain.realrobot_cond is not served: only finetune runs name a serve condition"]


def test_a_run_the_record_does_not_hold_is_named(tmp_path):
    assert checks.check_serve(_handshake(), LN, "pi05.finetune.other", _set(tmp_path)) == [
        "pi05.finetune.other is not a run of the line record"]


def test_the_geometry_is_the_one_measured_on_the_paper_style_centre_cropped_set(tmp_path):
    """The paper's finetune set is centre-cropped: a centre-crop server passes, a letterbox one is named
    on both cameras, whatever marker files the set holds or lacks."""
    root = _set(tmp_path)
    assert checks.check_serve(_handshake(), LN, ARM, root) == []
    assert checks.check_serve(_handshake(image_geometry=LETTERBOX), LN, ARM, root) == [
        f"image {cam}: the server applies geometry 'letterbox224' to raw frames; the training frames were made "
        f"with 'center_crop224'" for cam in ("base_0_rgb", "left_wrist_0_rgb")]


def test_a_letterboxed_set_is_not_the_records_finetune_set(tmp_path):
    root = fr3_set(tmp_path / "lb", letterbox_frame)
    problems = checks.check_serve(_handshake(image_geometry=LETTERBOX), LN, ARM, root)
    assert problems == [f"{root}: its frames measure as letterbox224; {ARM} trains on center_crop224 frames "
                        f"(the record's stage 2, ZhixuLi/tasl-fr3-10task-pbc-v2)"]


def test_a_dataset_directory_that_is_not_there_is_named(tmp_path):
    problems = checks.check_serve(_handshake(), LN, ARM, tmp_path / "nowhere")
    assert len(problems) == 1 and problems[0].startswith(f"{tmp_path / 'nowhere'}: cannot read the fr3 training set")


def test_a_readable_set_that_is_not_the_embodiments_is_named_and_the_rest_still_checked(tmp_path):
    root = edit_info(fr3_set(tmp_path / "w"), lambda i: {**i, "features": {**i["features"],
                                                                          "actions": {"dtype": "float32", "shape": [7]}}})
    problems = checks.check_serve(_handshake(action_fps=10), LN, ARM, root)
    assert problems == [f"{root}: not an fr3 training set: actions has shape [7], the fr3 embodiment's is [8]",
                        "action fps 10, the finetune data has 15"]
    (tmp_path / "e" / "meta").mkdir(parents=True)
    (tmp_path / "e" / "meta" / "info.json").write_text("{}")
    assert checks.check_serve(_handshake(), LN, ARM, tmp_path / "e")[0].startswith(
        f"{tmp_path / 'e'}: cannot read the fr3 training set")


@pytest.mark.parametrize("key, value, named", [
    ("action_fps", 15.9, "action fps 15.9, the finetune data has 15"),
    ("action_fps", "15", "action_fps is str '15', not a number"),
    ("action_fps", True, "action_fps is bool True, not a number"),
    ("action_dim_out", "8", "action_dim_out is str '8', not an integer"),
    ("action_dim_out", 8.0, "action_dim_out is float 8.0, not an integer"),
    ("action_horizon", 15.7, "action_horizon is float 15.7, not an integer"),
    ("guidance_scale", "0", "guidance_scale is str '0', not a number"),
    ("action_units", 1, "action_units is int 1, not a string"),
    ("state_order", "q0 q1 q2 q3 q4 q5 q6 gripper", "state_order is str 'q0 q1 q2 q3 q4 q5 q6 gripper', not a list of names"),
    ("image_geometry", {"base_0_rgb": 224, "left_wrist_0_rgb": "center_crop224"},
     "image_geometry is dict {'base_0_rgb': 224, 'left_wrist_0_rgb': 'center_crop224'}, not a map of camera to geometry"),
])
def test_values_are_compared_as_typed_never_coerced(tmp_path, key, value, named):
    """A 15.9 Hz server is not a 15 Hz one, and a string is not a number."""
    assert named in checks.check_serve(_handshake(**{key: value}), LN, ARM, _set(tmp_path))


def test_the_action_horizon_is_required(tmp_path):
    h = _handshake()
    del h["action_horizon"]
    assert checks.check_serve(h, LN, ARM, _set(tmp_path)) == ["the handshake does not declare action_horizon"]


def test_an_absent_guidance_scale_is_the_protocols_default_of_zero(tmp_path):
    """docs/reference/policy-server.md: guidance_scale is optional, default 0. The FR3 serve conditions
    are both at 0, so its absence is that default, not a guess; a declared non-zero one is named."""
    h, root = _handshake(), _set(tmp_path)
    del h["guidance_scale"]
    assert checks.check_serve(h, LN, ARM, root) == []
    assert any(p.startswith("guidance 1.5") for p in checks.check_serve(_handshake(guidance_scale=1.5), LN, ARM, root))


def test_a_camera_the_client_does_not_send_is_named(tmp_path):
    geo = {**CROP, "right_wrist_0_rgb": "center_crop224"}
    assert checks.check_serve(_handshake(image_geometry=geo), LN, ARM, _set(tmp_path)) == [
        "image right_wrist_0_rgb: a camera the client does not send (it sends base_0_rgb, left_wrist_0_rgb)"]


def test_an_unreadable_stage_2_embodiment_is_a_named_problem(tmp_path, monkeypatch):
    from axis_qvla.realrobot.fr3 import serve_check
    from axis_qvla.realrobot.fr3.embodiment import EmbodimentError

    def unreadable(name):
        raise EmbodimentError(f"{name}.toml: cannot read (OSError: gone)")

    monkeypatch.setattr(serve_check, "load_embodiment", unreadable)
    assert checks.check_serve(_handshake(), LN, ARM, _set(tmp_path)) == ["fr3.toml: cannot read (OSError: gone)"]


def _demos(tmp_path):
    """Your demonstrations in the training layout (`realdata training-set`): the fr3_droid set."""
    from axis_qvla.realrobot.realdata.training_set import write_training_set
    from tests.unit.realrobot.realdata.test_training_set import write_droid_recording
    write_training_set(write_droid_recording(tmp_path / "rec"), tmp_path / "demos")
    return tmp_path / "demos"


def test_a_server_of_your_demonstrations_is_checked_against_their_declared_geometry(tmp_path):
    """fr3_droid frames keep their 16:9 geometry; training and serving crop the centre square, as the
    embodiment declares, so nothing is measured from the pixels."""
    demos = _demos(tmp_path)
    bare = "pi05.finetune.realrobot_only"
    assert checks.check_serve(_handshake(prompt_suffix=""), LN, bare, demos, "fr3_droid") == []
    assert checks.check_serve(_handshake(), LN, ARM, demos, "fr3_droid") == []
    assert checks.check_serve(_handshake(image_geometry=LETTERBOX, action_horizon=10), LN, ARM, demos, "fr3_droid") == [
        *(f"image {cam}: the server applies geometry 'letterbox224' to raw frames; the training frames were made "
          f"with 'center_crop224'" for cam in ("base_0_rgb", "left_wrist_0_rgb")),
        f"action horizon 10, {ARM} trained 15"]
    assert checks.check_serve(_handshake(), LN, bare, demos, "fr3_droid") == [
        f"prompt suffix '\\nQuality: 5', {bare} serves '' (bare)"]


def test_a_set_that_is_not_your_demonstrations_training_layout_is_named(tmp_path):
    root = _set(tmp_path)              # the paper's v2.1 set, not a training layout
    problems = checks.check_serve(_handshake(), LN, ARM, root, "fr3_droid")
    assert problems[0].startswith(f"{root}: cannot read the fr3_droid training set")
    demos = _demos(tmp_path)
    info = demos / "task_1" / "meta" / "info.json"
    doc = json.loads(info.read_text())
    doc["features"]["observation.images.wrist"]["shape"] = [720, 1280, 3]
    info.write_text(json.dumps(doc))
    assert checks.check_serve(_handshake(), LN, ARM, demos, "fr3_droid") == [
        f"{demos}: not an fr3_droid training set: task_1: observation.images.wrist has shape [720, 1280, 3], "
        f"the fr3_droid embodiment's is [180, 320, 3]"]
