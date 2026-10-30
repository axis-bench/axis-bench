"""The request openpi's DROID example client sends once edited, and the config the server is started with."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from axis_qvla.bench.protocol import OBS_KEYS, STATE_ORDER
from axis_qvla.realrobot.fr3 import cli, droid_client
from axis_qvla.realrobot.fr3.embodiment import load_embodiment
from axis_qvla.realrobot.fr3.line import load_line
from axis_qvla.realrobot.fr3.local_runs import LINE, write_local_runs
from axis_qvla.realrobot.realdata import lerobot_v21
from axis_qvla.realrobot.realdata.training_set import write_training_set
from axis_qvla.train import envs
from axis_qvla.train.armspec import load_armspec
from axis_qvla.train.generate import openpi_config
from axis_qvla.train.recipes import load_recipe
from tests.unit.realrobot.fr3.local_fixture import build_local
from tests.unit.realrobot.realdata.test_training_set import FEATURES
from tests.unit.train._arms import host

FR3_DROID = load_embodiment("fr3_droid")
RECORDED_HW = FR3_DROID.image_recorded[::-1]        # (720, 1280)
STORED_HW = FR3_DROID.image_stored[::-1]            # (180, 320)
CONVERTER = envs.REPO / envs.load_env("openpi").checkout / "examples" / "droid" / "convert_droid_data_to_lerobot.py"


# openpi examples/droid/convert_droid_data_to_lerobot.py:32-34, verbatim: how the converter stores every
# camera frame of a recording (lines 128-134, at (320, 180), after flipping the recorded BGR to RGB).
def converter_resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def _raw_bgr(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (*RECORDED_HW, 3), dtype=np.uint8)


def _recording(root: Path, ext: np.ndarray, wrist: np.ndarray, q: np.ndarray, g: np.ndarray) -> Path:
    """A one-frame DROID-layout recording holding the converter's stored frames and this state."""
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 15, "chunks_size": 1000, "robot_type": "panda", "features": FEATURES,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "stack the blocks"}) + "\n")
    vec = lambda a: pa.array([list(a)], pa.list_(pa.float32()))   # noqa: E731
    table = pa.table({"exterior_image_1_left": [{"bytes": lerobot_v21.encode_png(ext), "path": None}],
                      "wrist_image_left": [{"bytes": lerobot_v21.encode_png(wrist), "path": None}],
                      "joint_position": vec(q), "gripper_position": vec(g), "actions": vec(np.zeros(8)),
                      "frame_index": [0], "episode_index": [0], "task_index": [0]})
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")
    (root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["stack the blocks"],
                                                              "length": 1}) + "\n")
    return root


def test_the_request_is_the_protocol_s_keys_with_stored_frames_and_the_model_s_state():
    ext = np.zeros((*RECORDED_HW, 3), np.uint8)
    req = droid_client.request(ext, ext, np.arange(7, dtype=np.float64), np.array([0.5]), "stack the blocks")
    assert tuple(req) == OBS_KEYS and req["base_0_rgb"].shape == (*STORED_HW, 3)
    assert req["state"].dtype == np.float32
    np.testing.assert_array_equal(req["state"], np.float32([0, 1, 2, 3, 4, 5, 6, 0.5]))
    assert tuple(FR3_DROID.state_model) == STATE_ORDER[:7] + ("gripper",)


def test_the_client_s_keys_and_sizes_are_the_protocol_s_and_fr3_droid_s():
    assert droid_client.RECORDED == FR3_DROID.image_recorded and droid_client.STORED == FR3_DROID.image_stored
    assert droid_client.OBS_KEYS == OBS_KEYS


def test_a_raw_frame_reaches_the_server_as_the_training_set_stores_it(tmp_path):
    """One raw camera frame through the converter into a training set, and through `request`: the same
    pixels, and the same state as training-set writes for the same joint and gripper positions."""
    bgr_ext, bgr_wrist = _raw_bgr(0), _raw_bgr(1)
    q, g = np.float64([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]), np.float64([0.25])
    # The converter and the client both flip the camera's BGR to RGB first (converter 128-134, main.py 216-219).
    stored = {k: converter_resize_image(bgr[..., ::-1], droid_client.STORED)
              for k, bgr in (("exterior_image_1_left", bgr_ext), ("wrist_image_left", bgr_wrist))}
    rec = _recording(tmp_path / "rec", stored["exterior_image_1_left"], stored["wrist_image_left"], q, g)
    # What training-set hands the video encoder: the recording's PNG cells, decoded.
    ds = lerobot_v21.Dataset.open(rec)
    t = ds.read_episode(0, list(stored))
    trained = {k: lerobot_v21.decode_png(t.column(k)[0].as_py()["bytes"]) for k in stored}
    write_training_set(rec, tmp_path / "demos", tail=0)
    state = pq.read_table(tmp_path / "demos" / "task_1" / "data" / "chunk-000" / "file-000.parquet")

    req = droid_client.request(bgr_ext[..., ::-1], bgr_wrist[..., ::-1], q, g, "stack the blocks")
    for key, column in (("base_0_rgb", "exterior_image_1_left"), ("left_wrist_0_rgb", "wrist_image_left")):
        assert req[key].dtype == np.uint8 and req[key].shape == (*STORED_HW, 3)
        np.testing.assert_array_equal(req[key], trained[column])
    np.testing.assert_array_equal(req["state"], np.float32(state.column("observation.state")[0].as_py()))
    np.testing.assert_array_equal(req["state"], np.concatenate([q, g]).astype(np.float32))


def test_a_frame_at_the_stored_size_passes_unchanged_and_any_other_size_is_refused():
    small = converter_resize_image(_raw_bgr(2), droid_client.STORED)
    np.testing.assert_array_equal(droid_client.frame(small, "exterior"), small)
    ok = np.zeros((*RECORDED_HW, 3), np.uint8)
    letterboxed = np.zeros((224, 224, 3), np.uint8)     # the stock client's resize_with_pad(..., 224, 224)
    for exterior, wrist, camera, got in ((letterboxed, ok, "exterior", "224 x 224"),
                                         (ok, np.zeros((360, 640, 3), np.uint8), "wrist", "640 x 360")):
        with pytest.raises(ValueError) as exc:
            droid_client.request(exterior, wrist, np.zeros(7), np.zeros(1), "stack the blocks")
        msg = str(exc.value)
        assert msg.startswith(f"the {camera} camera's frame is {got} uint8;")
        assert "1280 x 720 (as recorded) or 320 x 180 (as stored)" in msg
        assert "as captured, after its BGR-to-RGB flip, not its resize_with_pad(..., 224, 224) output" in msg


def _body(fn: ast.FunctionDef, *, drop_docstring_and_pil_import: bool = False) -> list[str]:
    body = fn.body
    if drop_docstring_and_pil_import:
        body = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                and not (isinstance(s, ast.ImportFrom) and s.module == "PIL")]
    return [ast.dump(fn.args), *map(ast.dump, body)]


def _parsed(fn) -> ast.FunctionDef:
    return ast.parse(inspect.getsource(fn)).body[0]


def test_droid_client_resizes_with_the_frozen_converter_function():
    """droid_client.resize_image is the frozen copy, less its docstring and its function-local Pillow import."""
    assert _body(_parsed(droid_client.resize_image), drop_docstring_and_pil_import=True) == \
        _body(_parsed(converter_resize_image))


@pytest.mark.skipif(not CONVERTER.is_file(),
                    reason=f"{CONVERTER.relative_to(envs.REPO)} is not here: third_party/openpi is not checked out "
                           f"(the CI checks it out)")
def test_the_frozen_converter_function_is_the_pinned_fork_s():
    tree = ast.parse(CONVERTER.read_text(encoding="utf-8"))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resize_image"]
    assert len(fns) == 1 and _body(fns[0]) == _body(_parsed(converter_resize_image))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resize_image"]
    sizes = {ast.literal_eval(c.args[1]) for c in calls}
    assert len(calls) == 3 and sizes == {droid_client.STORED}


def test_the_chunk_is_the_runs_horizon_and_width():
    line = load_line(LINE)
    assert droid_client.CHUNK == (line.stage2["action_horizon"], len(FR3_DROID.action_stored))
    assert droid_client.OPEN_LOOP <= droid_client.CHUNK[0]


def _served(tmp_path, capsys, run="pi05.finetune.realrobot_only"):
    toml, paths = build_local(tmp_path)
    spec = write_local_runs(paths, load_line(LINE), runs=[run])[run]
    assert cli.main(["serve-config", "--run", str(spec), "--out", str(tmp_path / "serve.json"),
                     "--host", str(toml)]) == 0
    assert capsys.readouterr().out.strip() == f"serve config of {run} -> {tmp_path / 'serve.json'}"
    return json.loads((tmp_path / "serve.json").read_text()), spec, paths


def test_a_served_fine_tune_declares_the_client_s_action_units(tmp_path, capsys):
    cfg, _spec, _paths = _served(tmp_path, capsys)
    assert cfg["exp_name"] == "serve" and cfg["model"]["action_horizon"] == droid_client.CHUNK[0]
    assert cfg["data"]["action_units"] == FR3_DROID.action_units
    assert cfg["data"]["image_center_crop"] is True


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        out |= _flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


def test_the_serve_config_and_the_training_config_differ_only_in_what_serving_does_not_read(tmp_path, capsys):
    """Every image and action convention (the crop, the action units and horizon, the model) is the
    training config's: the two differ in the mode and the compiled labels a training run is bound to."""
    from axis_qvla.realrobot.fr3.local_runs import load_local_run

    served, spec_path, paths = _served(tmp_path, capsys)
    spec = load_local_run(spec_path, paths)
    trained = openpi_config(spec, load_recipe(spec.preset), paths, bundle="sha256:" + "0" * 64,
                            bundle_dir=str(tmp_path / "bundle"))
    s, t = _flat(served), _flat(trained)
    assert set(s) == set(t)
    assert {k for k in s if s[k] != t[k]} == {"exp_name", "data.bundle", "data.bundle_id", "identity.bundle"}
    for key in ("data.image_center_crop", "data.action_units", "model.action_horizon", "model.action_dim"):
        assert s[key] == t[key]


def test_serve_config_refuses_a_run_spec_naming_local_inputs(tmp_path, capsys):
    toml, _paths = host(tmp_path)
    shipped = Path(LINE).parent / "runs" / "pi05.finetune.realrobot_only.toml"
    assert cli.main(["serve-config", "--run", str(shipped), "--out", str(tmp_path / "serve.json"),
                     "--host", str(toml)]) == 2
    assert "local-runs" in capsys.readouterr().err and not (tmp_path / "serve.json").exists()


def test_the_module_prints_the_request_two_camera_frames_make_and_refuses_a_padded_frame(tmp_path, capsys):
    rgb = np.random.default_rng(3).integers(0, 256, (*RECORDED_HW, 3), np.uint8)
    Image.fromarray(rgb).save(tmp_path / "exterior.png")
    Image.fromarray(rgb[::-1]).save(tmp_path / "wrist.png")
    assert droid_client.main([str(tmp_path / "exterior.png"), str(tmp_path / "wrist.png")]) == 0
    assert capsys.readouterr().out == ("base_0_rgb 320 x 180 (from 1280 x 720), left_wrist_0_rgb 320 x 180 "
                                       "(from 1280 x 720), state 8 float32\n")
    Image.fromarray(rgb[:224, :224]).save(tmp_path / "padded.png")
    assert droid_client.main([str(tmp_path / "padded.png"), str(tmp_path / "wrist.png")]) == 2
    assert "the exterior camera's frame is 224 x 224 uint8" in capsys.readouterr().err


def test_the_copied_file_runs_on_its_own(tmp_path):
    """On the robot's computer only droid_client.py is copied: it runs as a script with numpy and Pillow and
    nothing of this repository."""
    import shutil
    import subprocess
    import sys

    copy = tmp_path / "droid_client.py"
    shutil.copy(inspect.getsourcefile(droid_client), copy)
    assert "from axis" not in copy.read_text() and "import axis" not in copy.read_text()
    rgb = np.random.default_rng(4).integers(0, 256, (*RECORDED_HW, 3), np.uint8)
    Image.fromarray(rgb).save(tmp_path / "exterior.png")
    Image.fromarray(rgb).save(tmp_path / "wrist.png")
    r = subprocess.run([sys.executable, "droid_client.py", "exterior.png", "wrist.png"], cwd=tmp_path,
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("base_0_rgb 320 x 180 (from 1280 x 720), left_wrist_0_rgb 320 x 180")
