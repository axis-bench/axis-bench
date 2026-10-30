"""The goal extraction's pure parts; the checker classes are imported only when a record is extracted."""
from __future__ import annotations

import subprocess
import sys

from axis_qvla.labels import goals


def test_helpers():
    assert goals._wxyz_to_xyzw([1, 2, 3, 4]) == [2, 3, 4, 1]
    assert goals._tracked_body({"bodyName": "", "objName": "cup_0"}) == "cup_0"
    assert goals._resolve_key("cup_0", {"sample/cup_0": 1}) == "sample/cup_0"


def test_a_stable_window_is_the_longest_still_tail_that_lasts_long_enough():
    steps = [{"sim_time": 0.2 * i, "body_pose": {"cup": {"pos": [0.0 if i >= 4 else 0.1 * i, 0, 0]}}} for i in range(10)]
    passing = [i >= 3 for i in range(10)]
    assert goals._find_stable_window(steps, "cup", passing, max_diameter=0.005, min_seconds=0.5) == (4, 9)
    assert goals._find_stable_window(steps, "cup", passing, max_diameter=0.005, min_seconds=5.0) is None


def test_importing_the_module_does_not_import_the_checkers():
    code = "import sys, axis_qvla.labels.goals; print('axis_replay_native' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
