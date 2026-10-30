"""Every simulation number the paper prints, pinned to T0's outputs at the paper's precision.

The expected values below are copied from the paper text (Fig. 1; Fig. 4(c) and
Sec. IV.B; Fig. 5 and its caption; Sec. V.B-V.F; Fig. 6 and 7; Table I), never from T0. Each T0
value must round to the printed digits: |T0 - printed| <= half a unit of the last printed digit.
Fig. 4(c) and Fig. 6 print no bar labels; their numbers are the in-text ones of IV.B, V.C and V.E.
The outputs are rebuilt in-process from the golden per-task tallies (test_paper_golden), which
the real-pack T0 run reproduces byte for byte.
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from tests.golden.paper.test_paper_golden import rebuild

OUT = rebuild()
CLAIMS = json.loads(OUT["claims.json"])["claims"]
TABLE1 = json.loads(OUT["claims.json"])["table1"]["vlas"]
POOLED_N = json.loads(OUT["claims.json"])["n_cells"]
VLAS = ("π0.5", "SmolVLA", "X-VLA")


def close(value: float, printed: str) -> bool:
    dp = len(printed.split(".")[1]) if "." in printed else 0
    return abs(value - float(printed)) <= 0.5 * 10 ** -dp + 1e-9


def _csv(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(OUT[name].decode("utf-8"))))


def _bar(name: str, vla: str, group: str, label: str) -> dict:
    hits = [r for r in _csv(name) if (r["vla"], r["group"], r["label"]) == (vla, group, label)]
    assert len(hits) == 1, (name, vla, group, label)
    return hits[0]


# ---- Fig. 1 ---------------------------------------------------------------------------------
FIG1 = {"π0.5": "+3.9", "SmolVLA": "+23.4", "X-VLA": "+5.6"}


@pytest.mark.parametrize("vla", VLAS)
def test_fig1_headline_gains(vla):
    assert close(float(_bar("figures/fig1.csv", vla, "Episode cond. − BC", "mean over seeds")["rel"]), FIG1[vla])


# ---- Fig. 5 (36 bars, p. 5) and its caption -------------------------------------------------
SETTINGS = ("object-centric / episode", "object-centric / chunk", "approach-aware / episode", "approach-aware / chunk")
FIG5 = {
    "π0.5": {"weighting": ("-1.3", "-8.9", "-4.8", "+1.2"), "filtering": ("-13.4", "-6.2", "-12.3", "-12.3"),
             "conditioning": ("-7.8", "-1.0", "+1.0", "-5.1")},
    "SmolVLA": {"weighting": ("+14.8", "+4.9", "+34.4", "+9.8"), "filtering": ("-26.2", "+39.3", "0.0", "+37.7"),
                "conditioning": ("+80.3", "+29.5", "+73.8", "+16.4")},
    "X-VLA": {"weighting": ("+10.4", "+2.9", "0.0", "+8.7"), "filtering": ("+9.7", "-8.4", "+11.7", "+13.3"),
              "conditioning": ("-8.7", "-20.7", "+8.1", "+0.3")},
}
FIG5_CASES = [(v, m, s, p) for v, ms in FIG5.items() for m, ps in ms.items() for s, p in zip(SETTINGS, ps)]
CAPTION = {"plain BC": {"π0.5": "57.42", "SmolVLA": "5.08", "X-VLA": "25.75"},
           "FT-only control": {"π0.5": "8.83", "SmolVLA": "2.50", "X-VLA": "6.17"}}


@pytest.mark.parametrize("vla,method,setting,printed", FIG5_CASES)
def test_fig5_bars(vla, method, setting, printed):
    assert close(float(_bar("figures/fig5.csv", vla, method, setting)["rel"]), printed)


def test_fig5_has_all_36_combinations():
    assert len(FIG5_CASES) == 36
    assert sum(r["group"] in ("weighting", "filtering", "conditioning") for r in _csv("figures/fig5.csv")) == 36


@pytest.mark.parametrize("group,vla", [(g, v) for g in CAPTION for v in VLAS])
def test_fig5_caption_success_rates(group, vla):
    label = "reference" if group == "plain BC" else "no AXIS pretraining"
    assert close(float(_bar("figures/fig5.csv", vla, group, label)["sr"]), CAPTION[group][vla])


# ---- Fig. 7 (p. 6) ----------------------------------------------------------------------------
FIG7 = {"π0.5": ("+0.6", "-0.6", "-0.7", "+0.6", "+0.7", "+1.0"),
        "SmolVLA": ("+6.6", "+16.4", "+23.0", "-24.6", "+36.1", "+73.8"),
        "X-VLA": ("-0.6", "+6.8", "-5.5", "-0.3", "+10.0", "+8.1")}
FIG7_BARS = [(g, i) for g in ("BC pretraining / tagged FT", "conditioning / plain FT", "conditioning / tagged FT")
             for i in ("untagged inference", "tag-5 inference")]


@pytest.mark.parametrize("vla,k", [(v, k) for v in VLAS for k in range(6)])
def test_fig7_bars(vla, k):
    group, inference = FIG7_BARS[k]
    assert close(float(_bar("figures/fig7.csv", vla, group, inference)["rel"]), FIG7[vla][k])


def test_fig7_holds_the_18_bars_the_3_references_and_2_in_text_values():
    """The paper's Fig. 7 has 6 bars per VLA and the plain-BC band; two more rows carry
    the Sec. V.D values for BC pretraining / plain FT with tag 5 (X-VLA has none) and say so."""
    rows = _csv("figures/fig7.csv")
    bars = [r for r in rows if (r["group"], r["label"]) in FIG7_BARS]
    refs = [r for r in rows if (r["group"], r["label"]) == ("BC pretraining / plain FT", "untagged inference")]
    extra = [r for r in rows if r not in bars and r not in refs]
    assert (len(bars), len(refs), len(rows)) == (18, 3, 23)
    assert all("reference" in r["note"] for r in refs)
    assert [(r["vla"], r["note"]) for r in extra] == [(v, "Sec. V.D only; not a Fig. 7 bar") for v in ("π0.5", "SmolVLA")]


# ---- Table I (p. 6) -----------------------------------------------------------------------------
TABLE_I = {"cond_minus_bc": (("+3.86", "5.83"), ("+23.42", "45.42"), ("+5.62", "5.39")),
           "cond_episode_minus_chunk": (("-0.90", "8.52"), ("+24.71", "32.64"), ("+11.36", "15.16")),
           "filter_chunk_minus_episode": (("-1.90", "3.56"), ("+6.63", "28.34"), ("+5.59", "8.05")),
           "tagged_minus_plain_untagged": (("+0.36", "4.30"), ("+9.58", "12.31"), ("+7.86", "6.81")),
           "tagged_minus_plain_tag5": (("+0.93", "4.28"), ("+62.29", "31.29"), ("+8.93", "1.95"))}


@pytest.mark.parametrize("row,i", [(r, i) for r in TABLE_I for i in range(3)])
def test_table1_entries(row, i):
    e = TABLE1[("pi05", "smolvla", "xvla")[i]]["entries"][row]
    m, sd = TABLE_I[row][i]
    assert close(e["mean"], m) and close(e["sd"], sd)
    text = OUT["tables/table1.md"].decode()
    assert f"{m.replace('-', '−')} ± {sd}" in text                     # the paper's minus sign


def test_table1_seeds_are_the_papers():
    assert TABLE1["pi05"]["entries"]["cond_minus_bc"]["seeds"] == [42, 43, 44]
    assert TABLE1["smolvla"]["entries"]["cond_minus_bc"]["seeds"] == TABLE1["xvla"]["entries"]["cond_minus_bc"]["seeds"] == [0, 1, 2]


# ---- In-text numbers: IV.B, V.B-V.E (percentage points, relative %, success rates) --------------
PRINTED = {
    "IV.B.aa_filter_minus_random.smolvla.episode": "1.08", "IV.B.aa_filter_minus_random.xvla.episode": "5.75",
    "IV.B.aa_filter_minus_random.smolvla.chunk": "3.08", "IV.B.aa_filter_minus_random.xvla.chunk": "2.58",
    "IV.B.oc_filter_minus_random.smolvla.chunk": "3.17", "IV.B.oc_filter_minus_random.xvla.chunk": "-3.00",
    "V.B.cond_relative_to_bc.pi05": "1.02", "V.B.cond_relative_to_bc.smolvla": "73.77",
    "V.B.cond_relative_to_bc.xvla": "8.09",
    "V.C.aa_conditioning_episode_minus_chunk.pi05": "3.50", "V.C.aa_conditioning_episode_minus_chunk.smolvla": "2.92",
    "V.C.aa_conditioning_episode_minus_chunk.xvla": "2.00", "V.C.oc_conditioning_episode_minus_chunk.pi05": "-3.92",
    "V.C.oc_filtering_chunk_minus_episode.xvla": "-4.67", "V.C.random_chunk_minus_episode.xvla": "3.58",
    "V.D.tagged_minus_plain_ft.tag5.pi05": "0.25", "V.D.tagged_minus_plain_ft.tag5.smolvla": "5.00",
    "V.D.tagged_minus_plain_ft.tag5.xvla": "2.17", "V.D.tagged_minus_plain_ft.untagged.pi05": "0.83",
    "V.D.tagged_minus_plain_ft.untagged.smolvla": "0.67", "V.D.tagged_minus_plain_ft.untagged.xvla": "4.00",
    "V.D.add_tag5.after_plain_ft.smolvla": "-2.42", "V.D.add_tag5.after_tagged_ft.smolvla": "1.92",
    "V.D.add_tag5.after_plain_ft.xvla": "1.33", "V.D.add_tag5.after_tagged_ft.xvla": "-0.50",
    "V.D.bc_pretraining.tagged_ft_tag5_minus_plain.pi05": "-0.33",
    "V.D.bc_pretraining.tagged_ft_tag5_minus_plain.smolvla": "0.83",
    "V.D.bc_pretraining.tagged_ft_tag5_minus_plain.xvla": "1.75",
    "V.D.bc_pretraining.tag5_only_minus_plain.pi05": "0.92", "V.D.bc_pretraining.tag5_only_minus_plain.smolvla": "-0.67",
    "V.E.sr.pi05.5K.conditioning": "58.00", "V.E.sr.pi05.5K.bc": "57.42",
    "V.E.sr.smolvla.5K.conditioning": "8.83", "V.E.sr.smolvla.5K.bc": "5.08",
    "V.E.sr.xvla.2.5K.conditioning": "20.75", "V.E.sr.xvla.2.5K.bc": "21.50",
    "V.F.cond_episode_minus_chunk_mean": "11.72", "V.F.filter_chunk_minus_episode_mean": "3.44",
}


@pytest.mark.parametrize("claim", sorted(PRINTED))
def test_in_text_numbers(claim):
    assert close(CLAIMS[claim]["value"], PRINTED[claim]), (claim, CLAIMS[claim]["value"])


def test_approach_aware_chunk_weighting_gains_over_bc():   # "(+0.67 and +0.50 pp)", "across models"
    c = CLAIMS["V.C.aa_chunk_weighting_vs_bc"]
    assert close(c["pairs"][0]["diff"], "0.67") and close(c["pairs"][1]["diff"], "0.50") and c["positive"] == 3


def test_figure_4c_bars_lie_on_its_axis():                   # the panel's axis runs from -40 to 60
    assert all(-40 <= float(r["rel"]) <= 60 for r in _csv("figures/fig4c.csv"))


# ---- Counts and directions (V.B, V.C, V.E, V.F) -------------------------------------------------
def _counts(claim: str) -> tuple[int, int, int]:
    c = CLAIMS[claim]
    assert c["missing"] == 0
    return c["positive"], c["zero"], c["negative"]


def test_counts_the_paper_states():
    assert _counts("V.B.xvla_filtering_vs_conditioning") == (4, 0, 0)        # "in all four matched settings"
    assert _counts("V.C.conditioning_episode_vs_chunk") == (5, 0, 1)         # "five of six"
    assert _counts("V.C.filtering_chunk_vs_episode") == (4, 1, 1)            # "four, ties in one, below in one"
    assert _counts("V.C.oc_weighting_episode_vs_chunk") == (3, 0, 0)         # "across all three models"
    assert [p["sign"] for p in CLAIMS["V.C.aa_weighting_episode_vs_chunk"]["pairs"]] == [-1, 1, -1]
    assert _counts("V.F.cond_vs_bc_seed_level") == (6, 1, 2)                 # "six of nine ... one tie and two negative"
    assert _counts("V.F.tagged_vs_plain_vla_means") == (6, 0, 0)             # "All six VLA-inference comparisons"
    assert _counts("V.F.tagged_vs_plain_seed_level") == (15, 0, 3)           # "15 of 18 seed-level differences"
    assert _counts("V.F.add_tag5_after_tagged_ft.xvla") == (0, 0, 3)         # removing the tag helps X-VLA at all three
    assert _counts("V.F.add_tag5_after_tagged_ft.smolvla") == (1, 0, 2)      # adding it helps SmolVLA in one, hurts in two
    for row in ("cond_episode_minus_chunk", "filter_chunk_minus_episode"):   # "hold in SmolVLA and X-VLA, reverse for π0.5"
        assert [v["sign"] for v in CLAIMS[f"V.F.vla_means.{row}"]["values"]] == [-1, 1, 1]


def test_scaling_directions():                                               # Sec. V.E
    assert [p["sign"] for p in CLAIMS["V.E.conditioning_vs_bc.pi05"]["pairs"]] == [-1, -1, 1]
    assert [p["sign"] for p in CLAIMS["V.E.conditioning_vs_bc.smolvla"]["pairs"]] == [1, 1, 1]
    assert [p["sign"] for p in CLAIMS["V.E.conditioning_vs_bc.xvla"]["pairs"]] == [1, -1, 1]


def test_larger_pretraining_datasets_improve_success():               # Sec. V.E, Fig. 6
    rows = _csv("figures/fig6.csv")
    for vla in VLAS:
        for group in ("plain BC", "conditioning"):
            series = [float(_bar("figures/fig6.csv", vla, group, size)["sr"]) for size in ("1K", "2.5K", "5K")]
            assert series == sorted(series) and len(set(series)) == 3, (vla, group, series)
    assert len(rows) == 18


def test_the_seed_43_tie_is_a_tie_in_the_csv_too():                       # Sec. V.F "one tie"
    r = _bar("figures/fig1.csv", "π0.5", "Episode cond. − BC", "seed 43")
    assert r["sr"] == r["sr_bc"] and float(r["rel"]) == 0.0 == 100 * (float(r["sr"]) / float(r["sr_bc"]) - 1)


def test_claims_json_keeps_ascii_numbers_for_machines():
    """claims.json is data: its `text` parses with float(); the paper's minus sign is for the page."""
    assert CLAIMS["IV.B.oc_filter_minus_random.xvla.chunk"]["text"] == "-3.00"


def test_method_rankings():                                                  # Sec. V.B
    m = {v: {g: x["value"] for g, x in CLAIMS[f"V.B.method_means.{v}"]["groups"].items()} for v in ("pi05", "smolvla", "xvla")}
    assert min(m["pi05"]["conditioning"], m["pi05"]["weighting"]) > m["pi05"]["filtering"]
    assert abs(m["pi05"]["conditioning"] - m["pi05"]["weighting"]) < 0.5        # "similar means"
    assert max(m["smolvla"], key=m["smolvla"].get) == "conditioning"
    assert min(m["xvla"]["filtering"], m["xvla"]["weighting"]) > m["xvla"]["conditioning"]


def test_every_cell_is_one_full_read_of_24_tasks_by_50_layouts():           # Sec. V.A: "1,200 trials"
    assert POOLED_N == 123
    from tests.golden.paper.test_paper_golden import GOLD
    pooled = json.loads((GOLD / "pooled_r1.json").read_text())
    assert all(v["tot"] == 1200 and v["n_tasks"] == 24 for v in pooled.values())
