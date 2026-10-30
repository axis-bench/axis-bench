"""DROID-layout demonstrations in the π0.5 loader's per-task layout: model-order state, recorded actions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.fr3.embodiment import load_embodiment
from axis_qvla.realrobot.realdata import lerobot_v21, training_set
from axis_qvla.realrobot.realdata.cli import main
from axis_qvla.realrobot.realdata.training_set import DROID_COLUMNS, write_training_set

# openpi's examples/droid/convert_droid_data_to_lerobot.py writes its frames at 320 x 180.
H, W = 180, 320
FEATURES = {"exterior_image_1_left": {"dtype": "image", "shape": [H, W, 3], "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": [H, W, 3], "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": [7], "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": [1], "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": [8], "names": ["actions"]}}
PROMPTS = ("stack the blocks", "press the button")


def droid_frame(episode: int, k: int, camera: int) -> np.ndarray:
    """A smooth frame with a square that moves each step, so a lossy video keeps it and frames stay apart."""
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([x * 255 // W, y * 255 // H, np.full_like(x, 60 + 40 * episode + 80 * camera)], axis=-1)
    x0 = (24 * k) % (W - 40)
    img[40:80, x0:x0 + 40] = (240, 240 - 200 * camera, 30)
    return img.astype(np.uint8)


def write_droid_recording(root: Path, lengths=(20, 24, 18), tasks=(0, 1, 0)) -> Path:
    """A DROID-layout LeRobot v2.1 recording: episode e has lengths[e] frames of instruction PROMPTS[tasks[e]]."""
    root = Path(root)
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 15, "chunks_size": 1000, "robot_type": "panda", "features": FEATURES,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}))
    (root / "meta" / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": p}) + "\n"
                                                       for i, p in enumerate(PROMPTS)))
    eps = []
    for e, (n, t) in enumerate(zip(lengths, tasks)):
        rng = np.random.default_rng(e)
        ext, wrist = ([{"bytes": lerobot_v21.encode_png(droid_frame(e, k, cam)), "path": None} for k in range(n)]
                      for cam in (0, 1))
        vec = lambda a: pa.array(list(a), pa.list_(pa.float32()))   # noqa: E731
        table = pa.table({"exterior_image_1_left": ext, "wrist_image_left": wrist,
                          "joint_position": vec(rng.normal(size=(n, 7)).astype(np.float32)),
                          "gripper_position": vec(np.float32([[(k + 1) / (n + 1)] for k in range(n)])),
                          "actions": vec(rng.uniform(-1, 1, (n, 8)).astype(np.float32)),
                          "frame_index": np.arange(n), "episode_index": np.full(n, e), "task_index": np.full(n, t)})
        path = root / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        eps.append({"episode_index": e, "tasks": [PROMPTS[t]], "length": n})
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eps))
    return root


def _video_frames(path: Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def test_a_droid_recording_becomes_one_training_task_per_instruction(tmp_path):
    src = write_droid_recording(tmp_path / "rec")
    got = write_training_set(src, tmp_path / "demos")
    assert got == {"tasks": 2, "episodes": 3, "frames": 62}
    roots = json.loads((tmp_path / "demos" / "roots.json").read_text())
    assert roots["roots"] == {"1": "task_1", "2": "task_2"}
    # task_1 is "press the button" (recorded episode 1), task_2 "stack the blocks" (recorded episodes 0 and 2)
    tasks = pq.read_table(tmp_path / "demos" / "task_2" / "meta" / "tasks.parquet").to_pandas()
    assert list(tasks.index) == ["stack the blocks"]
    # Every episode, whole: state = [joint_position, gripper_position], action = actions, bit for bit.
    for task, episode, recorded in (("task_1", 0, 1), ("task_2", 0, 0), ("task_2", 1, 2)):
        t = pq.read_table(next((tmp_path / "demos" / task / "data").rglob("*.parquet"))).to_pandas()
        got = t[t["episode_index"] == episode].sort_values("frame_index")
        rec = pq.read_table(src / "data" / "chunk-000" / f"episode_{recorded:06d}.parquet").to_pandas()
        want = np.concatenate([np.stack(rec["joint_position"]), np.stack(rec["gripper_position"])], axis=1)
        np.testing.assert_array_equal(np.stack(got["observation.state"]), want)
        np.testing.assert_array_equal(np.stack(got["action"]), np.stack(rec["actions"]))
    ranges = json.loads((tmp_path / "demos" / "ranges.json").read_text())
    assert ranges == {"task_1--0": [[0, 14]], "task_2--0": [[0, 10]], "task_2--1": [[0, 8]]}


def test_a_recorded_frame_reaches_the_training_set_unchanged(tmp_path):
    """Train/serve parity: state is [joint_position, gripper_position] in the model's order and action the
    recorded 8 values, bit for bit; the pixels are the recorded ones up to h264's loss."""
    src = write_droid_recording(tmp_path / "rec")
    write_training_set(src, tmp_path / "demos")
    task = tmp_path / "demos" / "task_2"
    data = pq.read_table(task / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    episodes = pq.read_table(task / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    rec = pq.read_table(src / "data" / "chunk-000" / "episode_000002.parquet").to_pandas()   # task_2's episode 1
    rows = data[data["episode_index"] == 1].sort_values("frame_index")
    k = 7
    state, action = np.stack(rows["observation.state"])[k], np.stack(rows["action"])[k]
    assert state.dtype == np.float32 and action.dtype == np.float32
    np.testing.assert_array_equal(state, np.concatenate([rec["joint_position"][k], rec["gripper_position"][k]]))
    np.testing.assert_array_equal(action, rec["actions"][k])
    assert load_embodiment("fr3_droid").state_model == tuple(f"q{i}" for i in range(7)) + ("gripper",)

    row = int(episodes.loc[episodes["episode_index"] == 1, "dataset_from_index"].iloc[0]) + k
    assert int(rows["index"].iloc[k]) == row
    for key, column in (("third_person", "exterior_image_1_left"), ("wrist", "wrist_image_left")):
        video = _video_frames(task / "videos" / f"observation.images.{key}" / "chunk-000" / "file-000.mp4")
        assert video.shape == (len(data), H, W, 3)
        want = lerobot_v21.decode_png(rec[column][k]["bytes"]).astype(np.int16)
        err = np.abs(video[row].astype(np.int16) - want)
        # h264 at crf 23 over yuv420p: a smooth frame comes back within a few levels on average.
        assert err.mean() < 2.0, (key, err.mean())
        # ...and it is that frame, not a neighbour: along the square's band the next frame is far off.
        band = slice(40, 80)
        off = np.abs(video[row + 1, band].astype(np.int16) - want[band]).mean()
        assert off > 4 * err[band].mean(), (key, off, err[band].mean())
    info = json.loads((task / "meta" / "info.json").read_text())
    assert info["features"]["observation.images.third_person"]["shape"] == [H, W, 3] and info["fps"] == 15


def test_the_fixture_and_the_embodiment_declare_the_droid_converters_frames():
    emb = load_embodiment("fr3_droid")
    assert emb.image_stored == (W, H) == training_set.STORED and emb.fps == 15
    assert emb.image_keys == ("observation.images.third_person", "observation.images.wrist")


def test_a_recording_without_the_droid_columns_is_refused_by_name(tmp_path):
    src = write_droid_recording(tmp_path / "rec")
    info = json.loads((src / "meta" / "info.json").read_text())
    info["features"].pop("wrist_image_left")
    (src / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(lerobot_v21.RealdataError, match="wrist_image_left"):
        write_training_set(src, tmp_path / "demos")
    assert "wrist_image_left" in DROID_COLUMNS and not (tmp_path / "demos").exists()


@pytest.mark.parametrize("episode", [0, 2])
def test_a_recording_at_another_frame_size_is_refused_by_name(tmp_path, episode):
    """The fr3_droid runs train and serve on the DROID converter's 320 x 180 frames."""
    src = write_droid_recording(tmp_path / "rec")
    f = src / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    t = pq.read_table(f)
    big = [{"bytes": lerobot_v21.encode_png(np.zeros((360, 640, 3), np.uint8)), "path": None}] * len(t)
    pq.write_table(t.set_column(t.schema.get_field_index("wrist_image_left"), "wrist_image_left", pa.array(big)), f)
    with pytest.raises(lerobot_v21.RealdataError, match=rf"episode {episode}: wrist_image_left frames are 640 x 360; "
                                                         r"the fr3_droid runs train on 320 x 180"):
        write_training_set(src, tmp_path / "demos")
    if episode == 0:
        assert not (tmp_path / "demos").exists()


def test_the_command_writes_the_set_and_refuses_an_existing_output(tmp_path, capsys):
    src = write_droid_recording(tmp_path / "rec")
    out = tmp_path / "demos"
    assert main(["training-set", "--dataset", str(src), "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip() == f"3 episodes / 62 frames of 2 instructions -> {out}"
    assert main(["training-set", "--dataset", str(src), "--out", str(out), "--tail", "5"]) == 2
    assert capsys.readouterr().err.startswith("FAIL  ")
    assert main(["training-set", "--dataset", str(src), "--out", str(tmp_path / "t5"), "--tail", "5"]) == 0
    assert json.loads((tmp_path / "t5" / "ranges.json").read_text())["task_1--0"] == [[0, 19]]
