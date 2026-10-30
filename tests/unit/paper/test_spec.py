"""experiments/paper/paper.json: the paper's layout reads exactly the committed cells and is validated at load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_qvla.paper.spec import DEFAULT_PATH, SpecError, load_spec

REPO = Path(__file__).resolve().parents[3]


def test_the_layout_reads_exactly_the_cells_of_the_committed_map():
    cells = {c["cell"] for c in json.loads((REPO / "experiments" / "paper" / "cells.json").read_text())["cells"]}
    assert load_spec().cells() == cells and len(cells) == 123


def test_the_layout_has_the_papers_figures_table_and_sections():
    s = load_spec()
    assert [f.key for f in s.figures] == ["fig4c", "fig5", "fig6", "fig7"]
    assert [len(f.bars) for f in s.figures] == [8, 36 + 6, 18, 23]
    assert [r["label"] for r in s.table1["rows"]] == ["Episode cond. − BC", "Cond.: episode − chunk",
                                                       "Filter: chunk − episode", "Tagged − plain: untagged",
                                                       "Tagged − plain: tag 5"]
    assert {v: [r["seed"] for r in rs] for v, rs in s.table1["replicates"].items()} == {
        "pi05": [42, 43, 44], "smolvla": [0, 1, 2], "xvla": [0, 1, 2]}
    assert [x["id"] for x in s.sections] == ["Fig. 1", "IV.B", "V.B", "V.C", "V.D", "V.E", "V.F"]


def _write(tmp_path: Path, mutate) -> Path:
    doc = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    mutate(doc)
    p = tmp_path / "layout.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def _set(path, value):
    def f(doc):
        d = doc
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return f


@pytest.mark.parametrize("mutate,match", [
    (_set(["schema"], "qvla.paper_layout/1"), "schema must be"),
    (_set(["figures", 1, "bars", 0, "vla"], "openvla"), "unknown VLA"),
    (_set(["figures", 1, "bars", 0, "label"], "episode, chunk"), "without commas"),
    (_set(["table1", "rows", 0, "a"], "nobody"), "needs id, label and roles"),
    (_set(["fig1", "row"], "absent"), "fig1 names Table I row"),
    (_set(["claims", 0, "kind"], "median"), "kind must be one of"),
    (_set(["claims", 1, "id"], "IV.B.aa_filter_minus_random.smolvla.episode"), "duplicate claim ids"),
    (_set(["sections", 0, "figures"], ["fig9"]), "unknown figures"),
])
def test_a_malformed_layout_is_a_named_error(tmp_path, mutate, match):
    with pytest.raises(SpecError, match=match):
        load_spec(_write(tmp_path, mutate))


def test_a_table1_replicate_must_name_every_role(tmp_path):
    def drop(doc):
        del doc["table1"]["replicates"]["xvla"][2]["cells"]["plain_tag5"]
    with pytest.raises(SpecError, match="each naming every role"):
        load_spec(_write(tmp_path, drop))
