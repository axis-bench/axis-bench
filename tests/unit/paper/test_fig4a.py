"""Fig. 4(a): an episode's P_e is the exact share of its trainable rows with weight > 1.

Also: `paper label-study`'s CLI wiring (stdlib-only tests, so they run in the `t0` CI job, which installs
no numpy -- the missing-numpy test below is the real scenario there, not a simulation) and the
privacy ratchet over the committed `experiments/paper/fig4b_human_study.json`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

from axis_qvla.cli import build_parser
from axis_qvla.paper import fig4

REPO = Path(__file__).resolve().parents[3]
STUDY_PATH = REPO / "experiments" / "paper" / "fig4b_human_study.json"

# A 3-episode study whose keys match _corpus() below, for the CLI tests that need `--study`.
_STUDY_EPISODES = [{"task_id": 806, "episode_index": 0, "H": 0.1, "in_correlation_set": True},
                    {"task_id": 806, "episode_index": 1, "H": 0.6, "in_correlation_set": True},
                    {"task_id": 1050, "episode_index": 0, "H": 0.9, "in_correlation_set": True}]


def _corpus(tmp_path):
    w = {"task_806--0": [1.0, 1.5, 0.5, 2.0, 1.0], "task_806--1": [1.2, 1.2, 1.2], "task_1050--0": [1.0, 0.9]}
    ranges = {"task_806--0": [[0, 2], [3, 5]], "task_806--1": [[0, 3]], "task_1050--0": [[0, 2]]}
    for reward, rel in fig4.REWARDS.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"weights": w, "reward_id": reward}))
    (tmp_path / fig4.RANGES).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / fig4.RANGES).write_text(json.dumps(ranges))
    return tmp_path


def test_pe_counts_rows_over_the_half_open_ranges():
    pe = fig4.episode_pe({"task_806--0": [1.0, 1.5, 0.5, 2.0, 1.0]}, {"task_806--0": [[0, 2], [3, 5]]})
    assert pe == {(806, 0): 2 / 4}        # rows 0, 1, 3, 4; weights > 1 at rows 1 and 3


def test_fig4a_writes_sorted_exact_rows_and_a_summary(tmp_path):
    data, summary = fig4.fig4a(_corpus(tmp_path))
    assert data.decode().splitlines() == ["task_id,episode_index,P_object_centric,P_approach_aware",
                                          "806,0,0.5,0.5", "806,1,1.0,1.0", "1050,0,0.0,0.0"]
    assert summary["object_centric"]["n"] == 3 and summary["sha256"] == hashlib.sha256(data).hexdigest()
    assert summary["object_centric"]["quantiles"]["0.5"] == 0.5


def test_check_names_what_differs(tmp_path):
    _, summary = fig4.fig4a(_corpus(tmp_path))
    good = {"fig4a": summary, "tolerance": {"rho": 1e-12, "ci": 1e-9}}
    assert fig4.check({"fig4a": summary}, good) == []
    bad = json.loads(json.dumps(good))
    bad["fig4a"]["sha256"] = "0" * 64
    assert fig4.check({"fig4a": summary}, bad) == ["fig4a.sha256: " + summary["sha256"] + " != " + "0" * 64]


def test_check_flags_a_nan_rho_or_ci_end():
    """A failed bootstrap (every resample's rho undefined, say) must never read as 'matches': NaN
    makes a naive `> tol` comparison False on both sides, so it has to be caught explicitly."""
    nan = float("nan")
    good_b = {"object_centric": {"N": 3, "rho": 0.5, "ci95": [0.1, 0.9]},
              "approach_aware": {"N": 3, "rho": 0.5, "ci95": [0.1, 0.9]}}
    expected = {"fig4b": good_b, "tolerance": {"rho": 1e-12, "ci": 1e-9}}

    bad_rho = copy.deepcopy(good_b)
    bad_rho["object_centric"]["rho"] = nan
    assert fig4.check({"fig4b": bad_rho}, expected) != []

    bad_ci = copy.deepcopy(good_b)
    bad_ci["object_centric"]["ci95"] = [nan, 0.9]
    assert fig4.check({"fig4b": bad_ci}, expected) != []


def _fig4_args(**kw):
    defaults = {"study": "", "out": None, "check": False, "expected": str(REPO / "experiments" / "paper" / "fig4.expected.json")}
    defaults.update(kw)
    argv = ["paper", "label-study", "--pretraining-data", defaults.pop("corpus")]
    if defaults["study"] is not None:
        argv += ["--study", defaults["study"]]
    if defaults["out"]:
        argv += ["--out", defaults["out"]]
    if defaults["check"]:
        argv += ["--check"]
    argv += ["--expected", defaults["expected"]]
    return build_parser().parse_args(argv)


def test_cli_writes_outputs_and_check_round_trips(tmp_path, capsys):
    """`--study ''` computes Fig. 4(a) alone (stdlib only); `--out` writes both files; `--check`
    exits 0 on a match and 1 -- never a traceback -- on a real mismatch."""
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    args = _fig4_args(corpus=str(corpus), out=str(out))
    assert args.func(args) == 0
    data, summary_a = fig4.fig4a(corpus)
    assert (out / "fig4a_sim_pe.csv").read_bytes() == data
    saved = json.loads((out / "fig4.json").read_text())
    assert "fig4b" not in saved and saved["fig4a"] == summary_a

    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"fig4a": summary_a, "tolerance": {"rho": 1e-12, "ci": 1e-9}}))
    ok = _fig4_args(corpus=str(corpus), check=True, expected=str(expected))
    assert ok.func(ok) == 0
    assert "IDENTICAL" in capsys.readouterr().out

    bad_doc = json.loads(expected.read_text())
    bad_doc["fig4a"]["sha256"] = "0" * 64
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bad_doc))
    differs = _fig4_args(corpus=str(corpus), check=True, expected=str(bad))
    assert differs.func(differs) == 1
    assert "DIFFERS" in capsys.readouterr().out


def test_a_malformed_corpus_is_exit_2_not_exit_1(tmp_path, capsys):
    """A row index past a weights list's end (a mismatched signals version, say) raises IndexError,
    which must not land on exit 1 -- the code `--check` uses for "differs" -- but on exit 2, an
    integrity failure, like every other named exception _cmd_fig4 catches."""
    corpus = _corpus(tmp_path)
    ranges = json.loads((corpus / fig4.RANGES).read_text())
    ranges["task_1050--0"] = [[0, 5]]          # task_1050--0's weights list has only 2 entries
    (corpus / fig4.RANGES).write_text(json.dumps(ranges))
    args = _fig4_args(corpus=str(corpus))
    assert args.func(args) == 2
    assert capsys.readouterr().out.startswith("FAIL")


def test_missing_numpy_is_exit_2_with_an_actionable_message(tmp_path, monkeypatch, capsys):
    """On a base install (no numpy), Fig. 4(b) fails importing it; `_cmd_fig4` must turn that into
    one `FAIL` line and exit 2 -- the README's integrity code -- never an interpreter traceback at
    exit 1, the same code `--check` uses for "the figure moved". `sys.modules["numpy"] = None` makes
    `import numpy` raise exactly the error an uninstalled numpy would, on an interpreter that does
    have it."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    corpus = _corpus(tmp_path)
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"schema": "qvla.human_study/1",
                                 "params": {"n_boot": 2, "seed": 1, "aux_draws_per_resample": 0},
                                 "episodes": _STUDY_EPISODES}))
    args = _fig4_args(corpus=str(corpus), study=str(study))
    assert args.func(args) == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL") and "numpy" in out and 'pip install -e ".[method]"' in out and "--study" in out


def test_the_published_study_file_holds_only_the_declared_keys():
    """Privacy ratchet on the one place the human study is published: a closed key set, top level
    and per episode -- no judge, video, attempt, interval or free text, ever. A later rebuild that
    adds a column ("for debugging", say) must fail here, not ship silently: `experiments/paper/*`
    is both `include` and `verbatim` in the public allowlist, so it would go out unrenamed."""
    doc = json.loads(STUDY_PATH.read_bytes())
    assert set(doc) == {"H", "correlation_set", "episodes", "params", "schema", "study"}
    assert set(doc["params"]) == {"aux_draws_per_resample", "c_min", "min_records", "n_boot", "record_rule", "seed"}
    assert doc["episodes"], "no episodes"
    for ep in doc["episodes"]:
        assert set(ep) == {"task_id", "episode_index", "H", "n_qualifying_judges", "in_correlation_set"}
