"""claims.json: every in-text number and count of Secs. IV.B and V.B-V.F, by kind."""
from __future__ import annotations

import json

from axis_qvla.paper.claims import compute, dump
from axis_qvla.paper.rates import diff, exact, mean, rel, sr
from axis_qvla.paper.spec import load_spec
from axis_qvla.paper.table1 import compute as table1
from tests.unit.paper.paper_world import lineage, pooled

SPEC = load_spec()
IDX = {c: (i, role) for reps in SPEC.table1["replicates"].values() for i, r in enumerate(reps)
       for role, c in r["cells"].items()}


def _succ(cell: str) -> int:
    if cell in IDX:
        i, role = IDX[cell]
        return {"bc": 600, "cond_ep": (612, 600, 588)[i], "cond_chunk": 606, "filter_chunk": 600,
                "filter_ep": 594, "tagged_untagged": 610, "plain_untagged": 604, "plain_tag5": 590}[role]
    return 500 + (len(cell) * 13) % 89


def _claims(P=None) -> dict:
    P = P if P is not None else pooled(_succ, SPEC)
    return compute(P, table1(P, SPEC, lineage(SPEC)), SPEC)


def test_differences_ratios_and_rates_are_unrounded_with_the_printed_text():
    P, c = pooled(_succ, SPEC), _claims()
    d = c["IV.B.aa_filter_minus_random.smolvla.episode"]
    want = diff(P, "svla_drop_ph_ep|bare", "svla_rand_ep|bare")      # exact, rounded once
    assert abs(want - (sr(P, "svla_drop_ph_ep|bare") - sr(P, "svla_rand_ep|bare"))) < 1e-12
    assert d["value"] == want and d["text"] == f"{want:+.2f}" and d["cells"] == ["svla_drop_ph_ep|bare", "svla_rand_ep|bare"]
    r = c["V.B.cond_relative_to_bc.pi05"]
    assert r["value"] == rel(P, "cfg_ph_ep|cond", "bc|bare") and r["dp"] == 2
    assert c["Fig5.caption.bc_sr.pi05"]["value"] == sr(P, "bc|bare")


def test_signs_count_exact_ties_and_keep_each_pair():
    c = _claims()["V.F.cond_vs_bc_seed_level"]
    assert (c["positive"], c["zero"], c["negative"], c["missing"]) == (3, 3, 3, 0)
    s = _claims()["V.C.conditioning_episode_vs_chunk"]
    assert len(s["pairs"]) == 6 and s["positive"] + s["zero"] + s["negative"] == 6
    assert all(set(p) == {"label", "cells", "diff", "sign"} for p in s["pairs"])


def test_table_i_aggregates_average_the_nine_vla_seed_values():
    P = pooled(_succ, SPEC)
    t = table1(P, SPEC, lineage(SPEC))
    c = compute(P, t, SPEC)
    vals = [x for v in SPEC.vlas for x in t.entries[("cond_episode_minus_chunk", v.key)].exact]
    assert c["V.F.cond_episode_minus_chunk_mean"]["value"] == float(mean(vals))
    tp = c["V.F.tagged_vs_plain_seed_level"]
    assert (tp["positive"], tp["zero"], tp["negative"]) == (15, 0, 3)     # 588 - 590 < 0 at each third seed
    vm = c["V.F.tagged_vs_plain_vla_means"]
    assert (vm["positive"], vm["negative"]) == (6, 0) and len(vm["values"]) == 6


def test_replicate_signs_compare_two_roles_at_each_seed():
    c = _claims()["V.F.add_tag5_after_tagged_ft.xvla"]
    assert [p["sign"] for p in c["pairs"]] == [1, -1, -1] and c["pairs"][0]["label"] == "seed 0"


def test_means_and_a_missing_cell():
    P = pooled(_succ, SPEC)
    m = _claims(P)["V.B.method_means.xvla"]["groups"]
    assert m["weighting"]["value"] == float(100 * mean([exact(P, x) for x in m["weighting"]["cells"]]))
    del P["svla_rand_ep|bare"]
    c = compute(P, table1(P, SPEC, lineage(SPEC)), SPEC)
    assert c["IV.B.aa_filter_minus_random.smolvla.episode"]["value"] is None
    assert c["IV.B.aa_filter_minus_random.smolvla.episode"]["text"] == "—"


def test_dump_is_canonical_json_with_every_claim():
    P = pooled(_succ, SPEC)
    t = table1(P, SPEC, lineage(SPEC))
    doc = json.loads(dump(compute(P, t, SPEC), t, SPEC, n_cells=len(P)))
    assert doc["schema"] == "qvla.paper_claims/1" and doc["n_cells"] == len(P)
    assert set(doc["claims"]) == {c["id"] for c in SPEC.claims}
    assert set(doc["table1"]["vlas"]) == {"pi05", "smolvla", "xvla"}
