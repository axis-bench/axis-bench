"""`qvla data get base-weights --serving <backbone>`: only what that backbone's reference policy
server reads at start (pi0.5: the PaliGemma tokenizer, not the 12 GB parameters), still through the
checksum-list fetch; and `--serving` is refused for any other target."""
from __future__ import annotations

import pytest

from axis_qvla.cli import main
from axis_qvla.data import bases


def test_serving_lock_keeps_only_what_a_server_reads():
    lock = bases.load()
    pi = bases.serving_lock(lock, "pi05")
    assert list(pi["bases"]) == ["pi05_base"] and [b["dest"] for b in pi["bases"]["pi05_base"]] == ["openpi/big_vision"]
    assert sum(f["size"] for b in pi["bases"]["pi05_base"] for f in b["files"]) < 10_000_000
    assert list(bases.serving_lock(lock, "smolvla")["bases"]) == ["smolvlm2_500m"]
    xvla = bases.serving_lock(lock, "xvla")["bases"]
    assert list(xvla) == ["xvla_base"] and xvla["xvla_base"] == lock["bases"]["xvla_base"]   # the full base
    with pytest.raises(bases.BasesError):
        bases.serving_lock(lock, "octo")


def test_a_serving_base_the_checksum_list_lacks_is_refused():
    lock = {**bases.load(), "bases": {"pi05_base": [{"source": "gs://x", "dest": "openpi/x", "files": []}]}}
    with pytest.raises(bases.BasesError, match="no pi05_base block openpi/big_vision"):
        bases.serving_lock(lock, "pi05")


def test_get_base_weights_serving_fetches_only_that(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(bases, "fetch", lambda lock, names, dest, **kw: seen.update(
        names=list(names), bases=list(lock["bases"]),
        dests=[b["dest"] for b in lock["bases"]["pi05_base"]]) or (0, 0))
    assert main(["data", "get", "base-weights", "--serving", "pi05", "--dest", str(tmp_path)]) == 0
    assert seen == {"names": ["pi05_base"], "bases": ["pi05_base"], "dests": ["openpi/big_vision"]}


def test_serving_is_for_base_weights_only(capsys):
    assert main(["data", "get", "bench", "--serving", "pi05"]) == 2 and "--serving" in capsys.readouterr().out
