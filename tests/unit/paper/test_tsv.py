"""Per-task TSV rows: string task order, and the 6-column writer and reader."""
from __future__ import annotations

from pathlib import Path

from axis_qvla.paper.tsv import TSV_HEADER, read_tsv, rows_for_run, tallies_by_run, write_tsv

FP = "a" * 64


def test_rows_are_in_task_string_order():
    rows = rows_for_run("r1x", "r1x_dualsim", {806: (1, 2), 1050: (2, 2), 953: (0, 2)}, FP)
    assert [r.task for r in rows] == [1050, 806, 953]


def test_write_then_read_round_trips(tmp_path: Path):
    rows = rows_for_run("r1x", "r1x_dualsim", {806: (1, 2), 1050: (2, 2)}, FP)
    data = write_tsv(rows)
    assert data.startswith((TSV_HEADER + "\n").encode()) and data.endswith(b"\n")
    (tmp_path / "t.tsv").write_bytes(data)
    assert read_tsv(tmp_path / "t.tsv") == rows


def test_tallies_keep_first_appearance_order():
    rows = rows_for_run("b", "b_dualsim", {806: (1, 2)}, FP) + rows_for_run("a", "a_dualsim", {807: (0, 2)}, FP)
    assert list(tallies_by_run(rows)) == ["b", "a"]
