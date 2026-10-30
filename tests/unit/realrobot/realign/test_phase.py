"""Phase curves of rendered episodes from the published recordings: one per attempt, on its frames."""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.realrobot.realign import cli, curves, phase
from axis_qvla.realrobot.realign.view import ViewEpisode, write_meta_view
from tests.unit.realrobot.realworld_fixture import PROMPT, write_realworld

FRAMES = 36                                     # 12 recorded rows at 5 Hz


def _render_view(tmp_path):
    return write_meta_view(tmp_path / "d8", [ViewEpisode(9001, 0, FRAMES, 7001, PROMPT),
                                             ViewEpisode(9001, 1, FRAMES, 7002, PROMPT)])


def test_every_rendered_attempt_gets_a_curve_of_its_length(tmp_path):
    write_realworld(tmp_path / "pub")
    got = phase.phase_curves(_render_view(tmp_path), tmp_path / "pub")
    assert sorted(got) == [(9001, 7001), (9001, 7002)]
    for c in got.values():
        assert c.curve.dtype == np.float32 and c.curve.shape == (FRAMES,) and c.valid.shape == (FRAMES,)
        assert (c.t0 is None) == (not c.valid.any())
        assert np.all((c.curve[c.valid] >= 0.0) & (c.curve[c.valid] <= 1.0))


def _box_below_the_last_hand_pose(trace):
    """Rewrite a fixture trace so its box waits 0.1 m below where the hand ends: the hand approaches it."""
    import json
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    x, y, z = rows[-2]["body_pose"]["franka/panda_hand"]["pos"]
    for r in rows[:-1]:
        r["body_pose"]["box_1"]["pos"] = [x, y, z - 0.1]
        r["joint_qpos"]["box_1_free"][:3] = [x, y, z - 0.1]
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_each_episode_gets_the_approach_curve_of_its_own_attempt(tmp_path):
    from axis_qvla.labels.phase import LAMBDA
    from axis_qvla.realrobot.render import data
    write_realworld(tmp_path / "pub")
    _box_below_the_last_hand_pose(tmp_path / "pub" / data.trace_rel(9001, 7001))
    view = write_meta_view(tmp_path / "d8", [ViewEpisode(9001, 0, FRAMES, 7002, PROMPT),
                                             ViewEpisode(9001, 1, FRAMES, 7001, PROMPT)])
    got = phase.phase_curves(view, tmp_path / "pub")
    approach, other = got[(9001, 7001)], got[(9001, 7002)]
    assert approach.curve.shape == (FRAMES,) and other.curve.shape == (FRAMES,)
    assert approach.t0 == FRAMES and approach.valid.any()          # no contact: all approach
    assert approach.curve[0] == 0.0 and np.isclose(approach.curve.max(), LAMBDA)
    assert np.all(np.diff(approach.curve[approach.valid]) >= 0)
    assert other.t0 is None and not other.valid.any()


def test_an_episode_whose_render_is_not_its_recordings_length_is_refused(tmp_path, capsys):
    """The render puts recorded row k on frame rint((t_k - t_0) * 15) + 2 and ends on the last row's frame,
    so an episode of 12 rows at 5 Hz is 36 frames; any other length is not this recording's render."""
    write_realworld(tmp_path / "pub")
    view = write_meta_view(tmp_path / "d8", [ViewEpisode(9001, 0, FRAMES, 7001, PROMPT),
                                             ViewEpisode(9001, 1, FRAMES + 3, 7002, PROMPT)])
    with pytest.raises(ValueError, match=f"task 9001 attempt 7002: rendered {FRAMES + 3} frames; its recording's "
                                         f"last row is frame {FRAMES - 1}"):
        phase.phase_curves(view, tmp_path / "pub")
    assert cli.main(["phase", "--rendered", str(view.parent), "--data", str(tmp_path / "pub"),
                     "--out", str(tmp_path / "attempts")]) == 2
    assert capsys.readouterr().err.startswith("FAIL  task 9001 attempt 7002: rendered")


@pytest.mark.parametrize("task, attempt, named", [
    (9002, 7001, "task 9002 is not in the qvla-v1-realworld data"),
    (9001, 7999, "task 9001 attempt 7999 is not in the qvla-v1-realworld data"),
])
def test_a_rendered_episode_the_data_does_not_hold_is_named(tmp_path, capsys, task, attempt, named):
    write_realworld(tmp_path / "pub")
    view = write_meta_view(tmp_path / "d8", [ViewEpisode(task, 0, FRAMES, attempt, PROMPT)])
    assert cli.main(["phase", "--rendered", str(view.parent), "--data", str(tmp_path / "pub"),
                     "--out", str(tmp_path / "attempts")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and named in err


def test_an_attempt_without_a_reward_leaf_is_neutral(tmp_path):
    import json
    from axis_qvla.realrobot.render import data
    write_realworld(tmp_path / "pub")
    g = tmp_path / "pub" / data.goals_rel(9001, 7002)
    g.write_text(json.dumps({**json.loads(g.read_text()), "leaves": []}))
    c = phase.phase_curves(_render_view(tmp_path), tmp_path / "pub")[(9001, 7002)]
    assert c.t0 is None and not c.valid.any() and not c.curve.any()


def test_the_command_writes_an_attempt_store(tmp_path, capsys):
    write_realworld(tmp_path / "pub")
    roots = _render_view(tmp_path)
    assert cli.main(["phase", "--rendered", str(roots.parent), "--data", str(tmp_path / "pub"),
                     "--out", str(tmp_path / "attempts")]) == 0
    assert sorted(curves.read_attempt_store(tmp_path / "attempts")) == [(9001, 7001), (9001, 7002)]
    assert "2 episodes:" in capsys.readouterr().out


def test_the_command_refuses_data_without_its_manifest(tmp_path, capsys):
    write_realworld(tmp_path / "pub")
    (tmp_path / "pub" / "manifest.json").unlink()
    roots = _render_view(tmp_path)
    assert cli.main(["phase", "--rendered", str(roots.parent), "--data", str(tmp_path / "pub"),
                     "--out", str(tmp_path / "attempts")]) == 2
    assert "manifest.json" in capsys.readouterr().err and not (tmp_path / "attempts").exists()


def test_the_command_refuses_tasks_with_no_rendered_episodes(tmp_path, capsys):
    write_realworld(tmp_path / "pub")
    roots = _render_view(tmp_path)
    assert cli.main(["phase", "--rendered", str(roots.parent), "--data", str(tmp_path / "pub"),
                     "--out", str(tmp_path / "attempts"), "--tasks", "9001", "9002"]) == 2
    err = capsys.readouterr().err
    assert "FAIL" in err and "[9002]" in err and not (tmp_path / "attempts").exists()
