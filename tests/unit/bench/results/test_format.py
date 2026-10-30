"""qvla.results/v1: built only from a run that reproduces its fingerprint; canonical round-trip."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from axis_qvla.bench.results.fingerprint import eval_fingerprint
from axis_qvla.bench.results.format import dumps_results, load_results, results_from_run
from axis_qvla.bench.results.seedfiles import read_served_dir
from tests.unit.bench.results.fixtures import TASKS, header, make_served


def _run(tmp_path: Path, served: str = "armx_dualsim", outcomes=None):
    make_served(tmp_path / "logs", served, outcomes or {806: [True, False, True], 1050: [False, False, True]})
    return read_served_dir(tmp_path / "logs", served)


def test_round_trip_is_byte_stable(tmp_path: Path):
    run = _run(tmp_path)
    r = results_from_run(run, header(eval_fingerprint(run, TASKS)), task_ids=TASKS)
    assert [(x.task, x.seed, x.success) for x in r.rows][:3] == [(806, 806000, True), (806, 806001, False), (806, 806002, True)]
    data = dumps_results(r)
    assert load_results(data) == r and dumps_results(load_results(data)) == data
    assert data.endswith(b"\n") and b'"schema": "qvla.results/v1"' in data


def test_refuses_a_fingerprint_mismatch(tmp_path: Path):
    run = _run(tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        results_from_run(run, header("f" * 64), task_ids=TASKS)


def test_refuses_a_served_mismatch(tmp_path: Path):
    run = _run(tmp_path)
    h = header(eval_fingerprint(run, TASKS), served="otherarm_dualsim")
    with pytest.raises(ValueError, match="header is for"):
        results_from_run(run, h, task_ids=TASKS)


def test_load_rejects_unsorted_rows(tmp_path: Path):
    run = _run(tmp_path)
    r = results_from_run(run, header(eval_fingerprint(run, TASKS)), task_ids=TASKS)
    bad = dataclasses.replace(r, rows=tuple(reversed(r.rows)))
    with pytest.raises(ValueError, match="sorted"):
        load_results(dumps_results(bad))
