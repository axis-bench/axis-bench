"""expected/ round-trips, and a difference is reported at its first cell, claim or line."""
from __future__ import annotations

import json
from pathlib import Path

from axis_qvla.paper.expected import EXPECTED_SCHEMA, compare_expected, write_expected

GOLD = Path(__file__).resolve().parents[2] / "golden" / "paper"


def _outputs() -> dict[str, bytes]:
    return {"pooled_r1.json": (GOLD / "pooled_r1.json").read_bytes(),
            "claims.json": (GOLD / "claims.json").read_bytes(),
            "results_page.md": b"line one\nline two\n",
            "tables/table1.tex": b"% Table I\n"}


def test_round_trip_is_clean(tmp_path: Path):
    write_expected(_outputs(), tmp_path / "expected")
    man = json.loads((tmp_path / "expected" / "MANIFEST.json").read_text())
    assert man["schema"] == EXPECTED_SCHEMA and sorted(man["files"]) == sorted(_outputs())
    assert compare_expected(_outputs(), tmp_path / "expected") == []


def test_pooled_difference_names_the_first_cell(tmp_path: Path):
    write_expected(_outputs(), tmp_path / "e")
    P = json.loads((GOLD / "pooled_r1.json").read_text())
    P["awr_v2_row|bare"]["suc"] += 1
    out = dict(_outputs(), **{"pooled_r1.json": json.dumps(P, indent=1).encode()})
    problems = compare_expected(out, tmp_path / "e")
    assert problems == ["pooled_r1.json differs: first differing cell 'awr_v2_row|bare'"]


def test_claims_difference_names_the_first_claim(tmp_path: Path):
    write_expected(_outputs(), tmp_path / "e")
    C = json.loads((GOLD / "claims.json").read_text())
    C["claims"]["V.D.tagged_minus_plain_ft.tag5.pi05"]["value"] += 1e-9
    out = dict(_outputs(), **{"claims.json": json.dumps(C).encode()})
    assert compare_expected(out, tmp_path / "e") == [
        "claims.json differs: first differing claim 'V.D.tagged_minus_plain_ft.tag5.pi05'"]
    C = json.loads((GOLD / "claims.json").read_text())
    C["table1"]["vlas"]["xvla"]["entries"]["cond_minus_bc"]["sd"] += 1e-9
    out = dict(_outputs(), **{"claims.json": json.dumps(C).encode()})
    assert compare_expected(out, tmp_path / "e") == ["claims.json differs: table1 (Table I per-seed values)"]


def test_text_difference_names_the_line_and_order_is_fixed(tmp_path: Path):
    write_expected(_outputs(), tmp_path / "e")
    out = dict(_outputs(), **{"results_page.md": b"line one\nline 2\n", "tables/table1.tex": b"% changed\n"})
    assert compare_expected(out, tmp_path / "e") == [
        "results_page.md differs: first differing line 2", "tables/table1.tex differs: first differing line 1"]


def test_missing_extra_and_stray_files(tmp_path: Path):
    write_expected(_outputs(), tmp_path / "e")
    (tmp_path / "e" / "stray.txt").write_text("x")
    out = dict(_outputs())
    del out["tables/table1.tex"]
    out["figures/fig5.csv"] = b"a\n"
    assert compare_expected(out, tmp_path / "e") == [
        "figures/fig5.csv: produced but not in expected/",
        "tables/table1.tex: in expected/ but not produced",
        "expected/stray.txt is not listed in MANIFEST.json"]


def test_a_stray_part_file_in_a_local_pack_is_reported_not_silently_skipped(tmp_path: Path):
    """`_PART_FILE` names `axis_qvla.data.hf.download`'s own temp-file pattern in ITS cache
    directory. A LOCAL pack's expected/ is never touched by `download` -- nothing there ever
    creates a `.<name>.part-<pid>` file -- so one turning up there is an ordinary stray that
    must DIFFER (exit 1), never something the walk quietly steps over."""
    write_expected(_outputs(), tmp_path / "e")
    (tmp_path / "e" / ".pooled_r1.json.part-4242").write_text("leftover")
    assert compare_expected(_outputs(), tmp_path / "e") == [
        "expected/.pooled_r1.json.part-4242 is not listed in MANIFEST.json"]


def test_a_part_file_is_skipped_only_when_the_caller_says_this_is_the_download_cache(tmp_path: Path):
    """The skip exists for one place only: `axis_qvla.paper.fetch.HfSource`'s download cache, where
    a killed-mid-transfer fetch can legitimately leave one of these behind. `skip_part_files`
    is how `axis_qvla.paper.t0.check` says "this expected_dir is that cache, not a local pack"."""
    write_expected(_outputs(), tmp_path / "e")
    (tmp_path / "e" / ".pooled_r1.json.part-4242").write_text("leftover")
    assert compare_expected(_outputs(), tmp_path / "e", skip_part_files=True) == []
