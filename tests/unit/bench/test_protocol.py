"""axis_qvla.bench.protocol: the handshake and reply rules, and their agreement with the code that
actually sends and executes them (read from source, so no simulator is imported)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import axis_qvla.bench.sim as _bench_sim
from axis_qvla.bench import protocol

# A local alias, not a dotted attribute chain on the package import: the release cut's post-cut
# check resolves a renamed dotted name by walking the package's AST, which can see the `sim`
# subpackage itself but not a dunder attribute nothing in source assigns. The local alias is the
# idiom already used by test_rollout_dualsim_names.py and test_task_data_root.py in this directory.
SIM = Path(_bench_sim.__file__).resolve().parent
GOOD = {"supports_client_noise": True, "action_horizon": 10, "action_dim": 32}


def test_the_reference_handshakes_pass():
    from axis_qvla.bench.servers.lerobot_server import build_metadata

    assert protocol.handshake_problems(GOOD) == []
    assert protocol.handshake_problems(build_metadata(checkpoint_dir="c", config_name="x", quality_tag=5)) == []
    assert protocol.handshake_problems({**GOOD, "protocol": protocol.PROTOCOL}) == []


@pytest.mark.parametrize("md, needle", [
    ({**GOOD, "supports_client_noise": False}, "supports_client_noise"),
    ({k: v for k, v in GOOD.items() if k != "action_dim"}, "action_dim"),
    ({**GOOD, "action_horizon": True}, "action_horizon"),          # a bool is not an int
    ({**GOOD, "action_horizon": 0}, "action_horizon"),
    ({**GOOD, "quality_tag": True}, "quality_tag"),
    ({**GOOD, "guidance_scale": 1.0}, "cfg_conditioning"),          # guidance needs a tagged branch
    ({**GOOD, "protocol": "other/9"}, "protocol"),
    ([1, 2], "not a map"),
])
def test_bad_handshakes_are_named(md, needle):
    problems = protocol.handshake_problems(md)
    assert problems and any(needle in p for p in problems), problems


FULL = {**GOOD, "image_geometry": {"base_0_rgb": "raw", "left_wrist_0_rgb": "raw"},
        "state_order": list(protocol.STATE_ORDER), "action_units": "rad_s", "action_fps": 15,
        "action_dim_out": 8, "prompt_suffix": "\nQuality: 5"}


def test_a_handshake_that_states_the_simulator_contract_passes():
    assert protocol.handshake_problems(FULL) == []
    crop = {**FULL, "image_geometry": {"base_0_rgb": "center_crop224", "left_wrist_0_rgb": "letterbox224"}}
    assert protocol.handshake_problems(crop) == []          # what the server does to the raw frame
    assert protocol.contract(crop) == {k: crop[k] for k in protocol.CONTRACT_KEYS}
    assert protocol.contract(GOOD) == {}


@pytest.mark.parametrize("md, needle", [
    ({**FULL, "action_units": "normalized"}, "rad/s"),
    ({**FULL, "action_units": 1}, "action_units"),
    ({**FULL, "image_geometry": {"base_0_rgb": "resize99"}}, "not one of"),
    ({**FULL, "image_geometry": {"top": "raw"}}, "does not send"),
    ({**FULL, "image_geometry": "raw"}, "map each camera"),
    ({**FULL, "state_order": ["gripper", "q0", "q1", "q2", "q3", "q4", "q5", "q6"]}, "state_order"),
    ({**FULL, "action_fps": 10}, "15 Hz"),
    ({**FULL, "action_dim_out": 7}, "action_dim_out"),
    ({**FULL, "action_dim_out": 64}, "action_dim_out"),
    ({**FULL, "prompt_suffix": 5}, "prompt_suffix"),
])
def test_a_contract_that_is_not_the_simulators_is_named(md, needle):
    problems = protocol.handshake_problems(md)
    assert problems and any(needle in p for p in problems), problems


def test_reply_rules():
    ok = {"actions": np.zeros((10, 8), np.float32)}
    assert protocol.reply_problems(ok, replan=8) == []
    assert protocol.reply_problems({"actions": np.zeros((10, 9), np.float64)}, replan=8) == []
    assert "at least 8" in protocol.reply_problems({"actions": np.zeros((4, 8), np.float32)}, replan=8)[0]
    assert "dim" in protocol.reply_problems({"actions": np.zeros((10, 7), np.float32)}, replan=8)[0]
    assert "2-D" in protocol.reply_problems({"actions": np.zeros((10,), np.float32)}, replan=8)[0]
    assert "floating" in protocol.reply_problems({"actions": np.zeros((10, 8), np.int32)}, replan=8)[0]
    assert "no 'actions'" in protocol.reply_problems({"x": 1}, replan=8)[0]


def test_the_documented_wire_format_decodes_without_openpi_client():
    """docs/reference/policy-server.md's ndarray extension, reimplemented from the page's own
    words and nothing else -- msgpack and numpy, no openpi_client. Proves the page is accurate
    (the described format decodes into an array `reply_problems` accepts) and that the same four
    names as plain msgpack STR keys do not decode as an array: real wire bytes carry them only as
    msgpack BIN."""
    import msgpack

    actions = np.arange(80, dtype=np.float32).reshape(10, 8)
    wire = msgpack.packb({"actions": {b"__ndarray__": True, b"data": actions.tobytes(),
                                       b"dtype": actions.dtype.str, b"shape": list(actions.shape)}})
    decoded = msgpack.unpackb(wire, raw=False)["actions"]

    # the page's current wording: BIN keys, dtype.str, raw C-order bytes.
    rebuilt = np.ndarray(buffer=decoded[b"data"], dtype=np.dtype(decoded[b"dtype"]),
                         shape=tuple(decoded[b"shape"]))
    assert np.array_equal(rebuilt, actions)
    assert protocol.reply_problems({"actions": rebuilt}, replan=8) == []

    # the page's old wording: plain string keys. Real wire bytes never carry them that way.
    assert "__ndarray__" not in decoded and b"__ndarray__" in decoded
    assert "no 'actions'" in protocol.reply_problems({"actions": decoded}, replan=8)[0]


def test_the_documented_scalar_format_is_the_client_packers():
    """The page's NumPy-scalar map -- three BIN keys, `__npgeneric__`, `data` (the `.item()`) and
    `dtype`, no `shape` -- is byte for byte what openpi_client packs."""
    import msgpack
    from openpi_client import msgpack_numpy

    for x in (np.float32(1.5), np.int64(-3), np.bool_(True)):
        page = msgpack.packb({b"__npgeneric__": True, b"data": x.item(), b"dtype": x.dtype.str})
        assert msgpack_numpy.packb(x) == page and page[0] == 0x83     # a 3-entry fixmap
        assert msgpack_numpy.unpackb(page) == x


def test_the_key_groups_cover_every_declared_key_except_supports_client_noise():
    """CONDITIONING_KEYS, SHAPE_KEYS, CONTRACT_KEYS and IDENTITY_KEYS (axis_qvla.bench.sim.campaign's
    declared_differences groups the same keys this way for comparison) are a second spelling of
    the keys handshake_problems recognizes: the required action_horizon/action_dim, every key
    _OPTIONAL type-checks, and image_geometry/state_order (handled specially in
    _contract_problems rather than through _OPTIONAL). Pinned here so the two spellings cannot
    silently drift -- a key added to one and not the other would pass unnoticed otherwise."""
    recognized = {"action_horizon", "action_dim", "image_geometry", "state_order"} | set(protocol._OPTIONAL)
    grouped = (set(protocol.CONDITIONING_KEYS) | set(protocol.SHAPE_KEYS) |
               set(protocol.CONTRACT_KEYS) | set(protocol.IDENTITY_KEYS))
    assert grouped == recognized
    assert "supports_client_noise" not in grouped   # the one required key with nothing to compare


def test_parse_endpoint():
    assert protocol.parse_endpoint("127.0.0.1:8000") == ("127.0.0.1", 8000)
    assert protocol.parse_endpoint("gpu-node.example:34001") == ("gpu-node.example", 34001)
    for bad in ("8000", "host:", ":80", "host:x", "host:0", "host:70000"):
        with pytest.raises(ValueError):
            protocol.parse_endpoint(bad)


def _assigned(path: Path) -> dict[str, object]:
    """Module-level NAME = literal (and A, B = 1, 2) assignments of a source file."""
    out: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                out[t.id] = value
            elif isinstance(t, ast.Tuple) and isinstance(value, tuple):
                out.update({n.id: v for n, v in zip(t.elts, value) if isinstance(n, ast.Name)})
    return out


def _env_default(path: Path, var: str) -> str | None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == var):
            return ast.literal_eval(node.args[1]) if len(node.args) > 1 else None
    raise AssertionError(f"{path.name} reads no {var}")


def test_the_constants_are_the_ones_the_client_executes():
    from openpi_client.websocket_client_policy import NOISE_KEY

    render = _assigned(SIM / "rollout_render.py")
    assert (protocol.RAW_WIDTH, protocol.RAW_HEIGHT) == (render["RENDER_W"], render["RENDER_H"])
    assert set(render["CAMERA_TO_OBS_KEY"].values()) == set(protocol.OBS_KEYS) - {"state", "prompt"}
    assert int(_env_default(SIM / "cli.py", "PI05_REPLAN")) == protocol.DEFAULT_REPLAN
    assert float(_env_default(SIM / "cli.py", "PI05_FPS")) == protocol.ACTION_FPS
    assert protocol.NOISE_KEY == NOISE_KEY
    tree = ast.parse((SIM / "rollout_dualsim.py").read_text(encoding="utf-8"))
    widths = next(ast.literal_eval(n.value.func.value) for n in tree.body if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", "") == "_MIN_ACTION_WIDTH")
    assert widths["droid8"] == protocol.MIN_ACTION_DIM
