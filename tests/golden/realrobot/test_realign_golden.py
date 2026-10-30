"""REALALIGN artifacts on a deterministic synthetic delivery, pinned by array digests.

Arrays, not file bytes: the tag artifact stamps its builder's source digest, which changes with the
package's import root by design. Regenerate realign_golden.json only for an
intended change:  python -c "from tests.golden.realrobot.test_realign_golden import write_expected; write_expected()"
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from axis_qvla.realrobot.realign.cli import build
from tests.unit.realrobot.realign.test_driver import _export_curves

EXPECTED = Path(__file__).with_name("realign_golden.json")


def _digests(tmp: Path) -> dict:
    export_dir, amap, prompts = _export_curves(tmp)
    out = tmp / "out"
    build(out, export_curves=export_dir, alignment_map=amap, prompts=prompts)
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    d = {"tag": h(np.load(out / "quality_phase.npz")["tag"]),
         "schedule_drop_top": h(np.load(out / "schedule_drop_top.npz")["rows"])}
    for name in ("awr_weights_phase_rank.json", "awr_weights_phase.json", "awr_weights_phase_ep.json"):
        w = json.loads((out / name).read_text())["weights"]
        d[name] = h(np.concatenate([np.asarray(w[k], np.float32) for k in sorted(w)]))
    return d


def write_expected() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        EXPECTED.write_text(json.dumps(_digests(Path(tmp)), indent=1, sort_keys=True) + "\n")


def test_realign_artifacts_match_their_golden_digests(tmp_path):
    assert _digests(tmp_path) == json.loads(EXPECTED.read_text())
