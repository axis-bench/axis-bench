"""`qvla data get`: every byte comes from the locked commit and matches its locked sha256; a get
resumes; a mismatch leaves nothing behind; no lock means the documented local route."""
from __future__ import annotations

import hashlib
import http.server
import json
import threading
import urllib.parse

import pytest

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.cli import main
from axis_qvla.data import get, layout

COMMIT = "c" * 40
FILES = {("axisrobotics/qvla-v1-bench", "tasks/task_806/task.json"): b'{"t": 806}',
         ("axisrobotics/qvla-v1-bench", "assets/a.stl"): b"solid a",
         ("axisrobotics/qvla-v1-pi05", "ft/x/13579/params/w.bin"): b"weights",
         ("axisrobotics/qvla-v1-pi05", "pretrain/p/20604/params/w.bin"): b"pretrain weights"}


def _lock(files=FILES) -> dict:
    rows = [{"repo": r, "path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest(),
             "git_oid": hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()} for (r, p), b in sorted(files.items())]
    return {"schema": "qvla.release_lock/1", "release": "v1.0", "files": rows,
            "repos": {"axisrobotics/qvla-v1-bench": {"type": "dataset", "tag": "v1.0", "commit": COMMIT},
                      "axisrobotics/qvla-v1-pi05": {"type": "model", "tag": "v1.0", "commit": COMMIT}}}


CELLS = {"cells": [{"cell": "bc|bare", "run": "r1x", "checkpoint": {
    "id": "pi05/ft/x", "ref": "axisrobotics/qvla-v1-pi05@v1.0:ft/x/13579"},
    "lineage": {"init": {"id": "pi05/pretrain/p", "ref": "axisrobotics/qvla-v1-pi05@v1.0:pretrain/p/20604"}}},
    {"cell": "scratch|bare", "run": "r1scratch", "checkpoint": {    # no lineage.init: trained from scratch
        "id": "pi05/ft/scratch", "ref": "axisrobotics/qvla-v1-pi05@v1.0:ft/scratch/13579"}}]}


class _Hub(http.server.BaseHTTPRequestHandler):
    served: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        for (repo, p), data in self.served.items():
            prefix = "/datasets/" if repo.endswith(("bench", "realworld")) else "/"
            if path == f"{prefix}{repo}/resolve/{COMMIT}/{p}":
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def hub(monkeypatch):
    _Hub.served = dict(FILES)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{srv.server_address[1]}")
    yield _Hub
    srv.shutdown()
    srv.server_close()


def _files(tmp_path):
    lock, cells = tmp_path / "lock.json", tmp_path / "cells.json"
    lock.write_bytes(dumps_canonical(_lock()))
    cells.write_text(json.dumps(CELLS))
    return ["--checksums", str(lock), "--results-map", str(cells)]


def test_a_dataset_is_fetched_verified_and_a_second_get_fetches_nothing(hub, tmp_path, capsys):
    dest = tmp_path / "bench"
    assert main(["data", "get", "bench", "--dest", str(dest), *_files(tmp_path)]) == 0
    assert (dest / "tasks" / "task_806" / "task.json").read_bytes() == b'{"t": 806}'
    assert "2 fetched, 0 already present" in capsys.readouterr().out
    assert main(["data", "get", "bench", "--dest", str(dest), *_files(tmp_path)]) == 0
    assert "0 fetched, 2 already present" in capsys.readouterr().out


def test_include_selects_part_of_a_repo(hub, tmp_path):
    dest = tmp_path / "bench"
    assert main(["data", "get", "bench", "--include", "tasks/*", "--dest", str(dest), *_files(tmp_path)]) == 0
    assert (dest / "tasks" / "task_806" / "task.json").is_file() and not (dest / "assets").exists()


def test_a_locked_file_the_hub_404s_on_is_exit_3(hub, tmp_path, capsys):
    """A file the release lock pins but the Hub no longer serves at that commit is an HTTP 404,
    exit 3 -- never read as a digest mismatch (1) or a configuration error (2)."""
    del hub.served[("axisrobotics/qvla-v1-bench", "assets/a.stl")]
    dest = tmp_path / "bench"
    assert main(["data", "get", "bench", "--dest", str(dest), *_files(tmp_path)]) == 3
    out = capsys.readouterr().out
    assert "FAIL" in out and "404" in out
    assert not (dest / "assets" / "a.stl").exists()


def test_bytes_that_disagree_with_the_lock_are_refused_and_not_kept(hub, tmp_path, capsys):
    hub.served[("axisrobotics/qvla-v1-bench", "assets/a.stl")] = b"solid B"   # same size, other bytes
    dest = tmp_path / "bench"
    assert main(["data", "get", "bench", "--dest", str(dest), *_files(tmp_path)]) == 1
    assert "differs from the release checksum manifest" in capsys.readouterr().out and not (dest / "assets" / "a.stl").exists()


def test_a_checkpoint_resolves_through_the_cell_map_with_its_initial_checkpoint(hub, tmp_path):
    dest = tmp_path / "ck"
    assert main(["data", "get", "checkpoints/r1x", "--with-init", "--dest", str(dest), *_files(tmp_path)]) == 0
    assert (dest / "ft" / "x" / "13579" / "params" / "w.bin").read_bytes() == b"weights"
    assert (dest / "pretrain" / "p" / "20604" / "params" / "w.bin").read_bytes() == b"pretrain weights"


def test_resolve_names_cells_runs_and_checkpoint_ids_and_refuses_the_rest():
    for key in ("bc|bare", "r1x", "pi05/ft/x"):
        [sel] = get.resolve(f"checkpoints/{key}", CELLS, with_init=False)
        assert (sel.repo, sel.prefix) == ("axisrobotics/qvla-v1-pi05", "ft/x/13579/")
    assert get.resolve("traces-sample", CELLS, with_init=False)[0].prefix == "traces_sample/"
    for bad in ("weights", "base-weights", "checkpoints/", "checkpoints/nope"):
        with pytest.raises(get.GetError):
            get.resolve(bad, CELLS, with_init=False)


def test_with_init_notes_a_cell_with_no_lineage_init(capsys):
    """--with-init on a cell trained from scratch fetches only the checkpoint and says why: silence
    would read as a bug (the init just never arrived), not as "there was never one to fetch"."""
    [sel] = get.resolve("checkpoints/r1scratch", CELLS, with_init=True)
    assert sel.repo == "axisrobotics/qvla-v1-pi05"
    out = capsys.readouterr().out
    assert "r1scratch" in out and "lineage.init" in out
    # with_init=False asks for nothing extra, so there is nothing to note either
    get.resolve("checkpoints/r1scratch", CELLS, with_init=False)
    assert capsys.readouterr().out == ""


def test_ckpt_accepts_the_same_ref_grammar_as_layout_checkpoint_dest():
    """One grammar (axis_qvla.data.layout.parse_ref) for both readers, checkpoint_dest and get._ckpt:
    the tag is optional, and a path that is empty, all slashes or leaving the repository is
    refused rather than resolving to the whole repository root or above it."""
    tagged = get._ckpt("axisrobotics/qvla-v1-pi05@v1.0:ft/x/13579", "x")
    tagless = get._ckpt("axisrobotics/qvla-v1-pi05:ft/x/13579/", "x")
    assert tagged == tagless == get.Selection("axisrobotics/qvla-v1-pi05", "ft/x/13579/", "x")
    for bad in ("axisrobotics/qvla-v1-pi05@v1.0", "axisrobotics/qvla-v1-pi05@v1.0:",
                "axisrobotics/qvla-v1-pi05@v1.0:/", "axisrobotics/qvla-v1-pi05@v1.0://",
                "axisrobotics/qvla-v1-pi05@v1.0:../x", "axisrobotics/qvla-v1-pi05:/abs"):
        with pytest.raises(get.GetError):
            get._ckpt(bad, "x")
        with pytest.raises(layout.LayoutError):
            layout.parse_ref(bad)


QVLA_TOML = """launcher = "local"
gpus = 0
requires_secrets = []

[roots]
corpus = "{d}/corpus"
cache = "{d}/cache"
scratch = "{d}/scratch"
runs = "{d}/runs"
"""


def test_without_dest_files_go_where_the_layout_puts_them(hub, tmp_path, monkeypatch):
    """The places training and evaluation read (axis_qvla.data.layout), from ./qvla.toml."""
    monkeypatch.delenv("AXIS_HOST", raising=False)
    (tmp_path / "qvla.toml").write_text(QVLA_TOML.format(d=tmp_path))
    flags = _files(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["data", "get", "bench", "--include", "tasks/*", *flags]) == 0
    assert (tmp_path / "corpus" / "qvla-v1-bench" / "tasks" / "task_806" / "task.json").is_file()
    assert main(["data", "get", "checkpoints/r1x", *flags]) == 0
    assert (tmp_path / "cache" / "checkpoints" / "qvla-v1-pi05" / "ft" / "x" / "13579" / "params" / "w.bin").is_file()


def test_without_a_lock_the_local_route_is_printed(tmp_path, capsys):
    assert main(["data", "get", "results", "--dest", str(tmp_path), "--checksums", str(tmp_path / "absent.json")]) == 2
    out = capsys.readouterr().out
    assert "does not exist" in out and "qvla paper tables --results <dir> --check" in out
    assert "`hf download axisrobotics/qvla-v1-results --repo-type dataset --revision v1.0" in out
    assert "huggingface-cli" not in out      # exits 1 under the huggingface_hub the `offline` extra requires



def test_the_real_world_target_is_its_own_repository():
    (s,) = get.resolve("realworld", {}, with_init=False)
    assert (s.repo, s.prefix) == ("axisrobotics/qvla-v1-realworld", "")
    assert "qvla-v1-realworld" in layout.DATASET_REPOS


def test_qvla_data_get_realworld_fetches_its_repository_verified(hub, tmp_path, capsys):
    """The real-world data through the checksum manifest: one file first, as the renderer's error asks, then the rest."""
    repo = "axisrobotics/qvla-v1-realworld"
    files = {(repo, "manifest.json"): b'{"schema": "qvla.realworld/1"}',
             (repo, "tasks/task_9001/scenes/attempt_7001.json"): b'{"attempt_id": 7001}'}
    hub.served = {**FILES, **files}
    lock = _lock({**FILES, **files})
    lock["repos"][repo] = {"type": "dataset", "tag": "v1.0", "commit": COMMIT}
    (tmp_path / "lock.json").write_bytes(dumps_canonical(lock))
    dest = tmp_path / "rw"
    args = ["--dest", str(dest), "--checksums", str(tmp_path / "lock.json")]
    assert main(["data", "get", "realworld", "--include", "manifest.json", *args]) == 0
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == ["manifest.json"]
    assert "realworld: 1 fetched, 0 already present" in capsys.readouterr().out
    assert main(["data", "get", "realworld", *args]) == 0
    assert (dest / "tasks/task_9001/scenes/attempt_7001.json").read_bytes() == b'{"attempt_id": 7001}'
    assert "realworld: 1 fetched, 1 already present" in capsys.readouterr().out


def test_the_real_world_data_lands_in_the_corpus_root(tmp_path):
    class P:
        corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert layout.default_dest("axisrobotics/qvla-v1-realworld", P) == tmp_path / "corpus" / "qvla-v1-realworld"
