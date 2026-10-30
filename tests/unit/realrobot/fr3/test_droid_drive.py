"""The patched DROID client, with no robot: a fake RobotEnv and a fake policy server."""
from __future__ import annotations

import builtins
import csv
import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from axis_qvla.realrobot.fr3 import droid_drive as dd
from axis_qvla.realrobot.fr3.droid_client import OBS_KEYS, frame

FR3_HANDSHAKE = {"action_units": "droid_normalized_joint_velocity_command", "action_fps": 15,
                 "action_horizon": 15, "action_dim_out": 8}


def _bgra(seed):
    return np.random.default_rng(seed).integers(0, 256, (720, 1280, 4), dtype=np.uint8)


def _rgb(bgra):
    return np.ascontiguousarray(bgra[..., :3][..., ::-1])


class FakeEnv:
    def __init__(self, *a, **kw):
        self.images = {"111_left": _bgra(1), "111_right": _bgra(9), "222_left": _bgra(2), "333_left": _bgra(3)}
        self.steps, self.resets = [], 0

    def get_observation(self):
        return {"image": dict(self.images), "robot_state": {"joint_positions": [0.1] * 7, "gripper_position": 0.2}}

    def step(self, action):
        self.steps.append(np.asarray(action))

    def reset(self):
        self.resets += 1


class FakePolicy:
    def __init__(self, *a, chunk=(15, 8)):
        self.requests, self.chunk = [], chunk

    def get_server_metadata(self):
        return dict(FR3_HANDSHAKE)

    def infer(self, req):
        self.requests.append(req)
        out = np.full(self.chunk, 0.1)
        out[:, -1] = 0.7
        return {"actions": out}


def test_actions_are_binarised_and_clipped_as_openpis_client_does():
    np.testing.assert_array_equal(dd.to_command([2.0, -3.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.51]), [1, -1, 0.2, 0, 0, 0, 0, 1])
    assert dd.to_command([0.0] * 7 + [0.5])[-1] == 0.0


def test_a_rollout_asks_every_eight_steps_with_the_runs_request():
    env, policy, sleeps = FakeEnv(), FakePolicy(), []
    t = iter(np.arange(0, 100, 0.01))
    assert dd.rollout(env, policy, "stack the blocks", "111", "333", max_steps=20,
                      clock=lambda: next(t), sleep=sleeps.append) == 20
    assert len(policy.requests) == 3 and len(env.steps) == 20
    req = policy.requests[0]
    assert tuple(req) == OBS_KEYS and req["base_0_rgb"].shape == (180, 320, 3)
    np.testing.assert_array_equal(req["base_0_rgb"], frame(_rgb(env.images["111_left"]), "exterior"))
    assert req["state"].dtype == np.float32 and req["state"].shape == (8,) and req["prompt"] == "stack the blocks"
    assert all(a[-1] == 1.0 for a in env.steps) and sleeps and all(s > 0 for s in sleeps)


def test_a_reply_of_another_shape_is_refused():
    with pytest.raises(ValueError, match=r"\(10, 8\)"):
        dd.rollout(FakeEnv(), FakePolicy(chunk=(10, 8)), "x", "111", "333", max_steps=2, sleep=lambda s: None)


class _NonFinitePolicy(FakePolicy):
    def infer(self, req):
        out = super().infer(req)
        out["actions"][0, 0] = np.nan
        return out


def test_a_non_finite_reply_is_refused_and_the_robot_is_not_commanded():
    env = FakeEnv()
    with pytest.raises(ValueError, match="non-finite"):
        dd.rollout(env, _NonFinitePolicy(), "x", "111", "333", max_steps=2, sleep=lambda s: None)
    assert env.steps == []


def test_the_exterior_camera_is_the_one_nearest_the_reference():
    env = FakeEnv()
    frames = {s: dd.lens(env.get_observation(), s) for s in ("111", "222")}
    ref = frame(frames["222"], "exterior")
    assert dd.pick_camera(frames, ref)[0] == "222"
    with pytest.raises(ValueError, match="--external-camera"):
        dd.pick_camera({"111": frames["111"], "222": frames["111"]}, ref)


def test_a_server_that_is_not_an_fr3_one_is_refused():
    with pytest.raises(ValueError, match="action_units"):
        dd.check_handshake({**FR3_HANDSHAKE, "action_units": "joint_velocity_rad_s_clipped_1"})


def test_instruction_asks_again_for_an_empty_or_out_of_range_answer(monkeypatch):
    answers = iter(["", "9", "2"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert dd._instruction(["press the button", "stack the blocks"]) == "stack the blocks"


def test_instruction_accepts_freeform_words_instead_of_a_number(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "open the drawer")
    assert dd._instruction(["press the button"]) == "open the drawer"


def test_main_drives_one_rollout_and_records_it(tmp_path, monkeypatch):
    env = FakeEnv()
    droid, robot_env = types.ModuleType("droid"), types.ModuleType("droid.robot_env")
    robot_env.RobotEnv = lambda **kw: env
    client, wcp = types.ModuleType("openpi_client"), types.ModuleType("openpi_client.websocket_client_policy")
    wcp.WebsocketClientPolicy = FakePolicy
    for name, mod in {"droid": droid, "droid.robot_env": robot_env, "openpi_client": client,
                      "openpi_client.websocket_client_policy": wcp}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    kit = tmp_path / "kit"
    kit.mkdir()
    Image.fromarray(frame(_rgb(env.images["111_left"]), "exterior")).save(kit / "reference_exterior.png")
    (kit / "drive.json").write_text(json.dumps({"run": "r", "instructions": ["stack the blocks"], "reference": "reference_exterior.png"}))
    answers = iter(["1", "y", "n"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    monkeypatch.chdir(tmp_path)
    assert dd.main(["--remote-host", "10.0.0.5", "--kit", str(kit), "--max-timesteps", "3",
                    "--exterior-camera-ids", "111", "222", "--wrist-camera-id", "333"]) == 0
    (out,) = (tmp_path / "results").iterdir()
    rows = list(csv.reader(out.open()))
    assert rows == [["run", "remote_host", "instruction", "success", "steps", "camera"],
                     ["r", "10.0.0.5", "stack the blocks", "1.0", "3", "111"]]
