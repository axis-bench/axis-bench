"""The results files a method author's `qvla bench eval --from-run` would write, over the contrast tests'
fixture pack (fixture_pack/: 2 tasks x 3 seeds): shared by tests/golden/paper/test_contrast_paper_refs.py
and the docs' CI stand-ins (tests/docs/fixtures.py), so neither imports the other's privates."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

from axis_qvla.bench.results.format import ResultsRow, dumps_results, load_results

FIX = Path(__file__).resolve().parent / "fixture_pack"
BC_FILE = "results/s2/r1fbc_dualsim.json"
# The fixture's plain BC (r1fbc) succeeds on 2/3 of task 806 and 1/3 of task 1050: SR_BC = 1/2.
MINE = {806: [True, True, True], 1050: [False, True, True]}          # SR = 5/6: +66.67 % of plain BC


def user_results(path: Path, *, arm: str, pretrain: int, ft: int, outcomes=MINE) -> Path:
    """The fixture pack's BC file at `path`, as run `arm` with these training seeds and per-task outcomes."""
    bc = load_results((FIX / "pack" / BC_FILE).read_bytes())
    rows = tuple(ResultsRow(t, t * 1000 + i, ok, 100, 6.6, "success" if ok else "timeout")
                 for t in sorted(outcomes) for i, ok in enumerate(outcomes[t]))
    h = dataclasses.replace(bc.header, arm=arm, served=arm, training_seed={"pretrain": pretrain, "ft": ft},
                            checkpoint={"ref": f"runs/{arm}/train/final", "pub_digest": None,
                                        "eval_digest": None},
                            eval_fingerprint=hashlib.sha256(arm.encode()).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps_results(dataclasses.replace(bc, header=h, rows=rows)))
    return path
