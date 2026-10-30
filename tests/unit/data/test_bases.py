"""`qvla data get base-weights`: each base comes from its publisher at the locked GCS generation or
Hub commit, every file matches the checksum list before it is kept, a get resumes, and a mismatch
or an unknown base is refused with nothing left behind."""
from __future__ import annotations

import hashlib
import http.server
import socket
import struct
import threading
import tomllib
import urllib.parse

import pytest

from axis_qvla.core.canon import dumps_canonical
from axis_qvla.cli import main
from axis_qvla.data import bases
from axis_qvla.train.recipes import RECIPES_DIR

REV = "a" * 40
GEN = "1757354377569924"
PARAMS = {"_METADATA": b'{"tree": 1}', "ocdbt.process_0/d/0f": b"weights" * 100}
TOKENIZER = b"sentencepiece"
HUB = {"config.json": b'{"type": "smolvla"}', "model.safetensors": b"tensors" * 50}
VREV = "b" * 40
VLM = {"config.json": b'{"model_type": "smolvlm"}'}     # the backbone smolvla_base's config names
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def _lock() -> dict:
    def rows(files, gen=None):
        return [{"name": n, "size": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                 **({"generation": gen} if gen else {})} for n, b in sorted(files.items())]
    return {"schema": bases.SCHEMA, "bases": {
        "pi05_droid": [
            {"source": "gs://openpi-assets/checkpoints/pi05_droid/params",
             "dest": "openpi/openpi-assets/checkpoints/pi05_droid/params", "files": rows(PARAMS, GEN)},
            {"source": "gs://big_vision", "dest": "openpi/big_vision",
             "files": rows({"paligemma_tokenizer.model": TOKENIZER}, GEN)}],
        "smolvla_base": [
            {"source": "hf:lerobot/smolvla_base", "revision": REV,
             "dest": f"hf/hub/models--lerobot--smolvla_base/snapshots/{REV}", "files": rows(HUB)},
            {"source": f"hf:{VLM_REPO}", "revision": VREV,
             "dest": f"hf/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/{VREV}",
             "files": rows(VLM)}]}}


class _Upstream(http.server.BaseHTTPRequestHandler):
    """GCS objects at one generation (`/<bucket>/<object>?generation=`) and Hub files at one commit
    (`/<repo>/resolve/<rev>/<path>`); every request is recorded."""
    served: dict = {}
    requests: list = []
    reset: str | None = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.requests.append(self.path)
        if urllib.parse.unquote(self.path) == self.reset:      # half the body, then a reset
            data = self.served[self.reset]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[: len(data) // 2])
            self.wfile.flush()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
            return
        data = self.served.get(urllib.parse.unquote(self.path))
        self.send_response(200 if data is not None else 404)
        self.send_header("Content-Length", str(len(data or b"")))
        self.end_headers()
        self.wfile.write(data or b"")


def _served() -> dict:
    out = {f"/openpi-assets/checkpoints/pi05_droid/params/{n}?generation={GEN}": b for n, b in PARAMS.items()}
    out[f"/big_vision/paligemma_tokenizer.model?generation={GEN}"] = TOKENIZER
    out.update({f"/lerobot/smolvla_base/resolve/{REV}/{n}": b for n, b in HUB.items()})
    out.update({f"/{VLM_REPO}/resolve/{VREV}/{n}": b for n, b in VLM.items()})
    return out


@pytest.fixture
def upstream(monkeypatch):
    _Upstream.served, _Upstream.requests, _Upstream.reset = _served(), [], None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setenv("HF_ENDPOINT", url)
    monkeypatch.setattr(bases, "GCS_ENDPOINT", url)
    yield _Upstream
    srv.shutdown()
    srv.server_close()


def _get(tmp_path, *args):
    lock = tmp_path / "bases.lock.json"
    lock.write_bytes(dumps_canonical(_lock()))
    return main(["data", "get", "base-weights", "--dest", str(tmp_path / "bw"), "--checksums", str(lock), *args])


def _left(tmp_path):
    return sorted(p.relative_to(tmp_path / "bw").as_posix() for p in (tmp_path / "bw").rglob("*") if p.is_file())


def test_every_base_lands_in_the_layout_runs_read_verified(upstream, tmp_path, capsys):
    assert _get(tmp_path) == 0
    bw = tmp_path / "bw"
    assert (bw / "openpi/openpi-assets/checkpoints/pi05_droid/params/ocdbt.process_0/d/0f").read_bytes() == PARAMS["ocdbt.process_0/d/0f"]
    assert (bw / "openpi/big_vision/paligemma_tokenizer.model").read_bytes() == TOKENIZER
    hub = bw / "hf/hub/models--lerobot--smolvla_base"
    assert (hub / "snapshots" / REV / "model.safetensors").read_bytes() == HUB["model.safetensors"]
    assert (hub / "refs" / "main").read_text() == REV      # what an offline from_pretrained("lerobot/smolvla_base") reads
    out = capsys.readouterr().out
    assert "pi05_droid: 3 fetched, 0 already present" in out and "smolvla_base: 3 fetched, 0 already present" in out


def test_the_urls_pin_the_gcs_generation_and_the_hub_commit(upstream, tmp_path):
    assert _get(tmp_path) == 0
    assert all(r.endswith(f"?generation={GEN}") for r in upstream.requests if "/resolve/" not in r)
    assert sorted(r for r in upstream.requests if "/resolve/" in r) == \
        [f"/{VLM_REPO}/resolve/{VREV}/config.json", f"/lerobot/smolvla_base/resolve/{REV}/config.json",
         f"/lerobot/smolvla_base/resolve/{REV}/model.safetensors"]


def test_a_second_get_fetches_nothing_and_a_removed_file_is_fetched_again(upstream, tmp_path, capsys):
    assert _get(tmp_path) == 0
    capsys.readouterr()
    upstream.requests.clear()
    assert _get(tmp_path) == 0
    assert upstream.requests == [] and "pi05_droid: 0 fetched, 3 already present" in capsys.readouterr().out
    (tmp_path / "bw/openpi/big_vision/paligemma_tokenizer.model").unlink()
    assert _get(tmp_path, "--include", "pi05_droid") == 0
    assert upstream.requests == [f"/big_vision/paligemma_tokenizer.model?generation={GEN}"]


def test_include_names_bases_and_a_base_brings_the_backbone_it_loads(upstream, tmp_path):
    """smolvla_base's config loads SmolVLM2's weights (vlm_model_name, load_vlm_weights), so the base
    carries that snapshot too, as xvla_base carries BART's tokenizer."""
    assert _get(tmp_path, "--include", "smolvla_base") == 0
    hub = tmp_path / "bw/hf/hub"
    assert _left(tmp_path) == sorted(
        [f"hf/hub/models--lerobot--smolvla_base/snapshots/{REV}/{n}" for n in HUB]
        + [f"hf/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/{VREV}/{n}" for n in VLM]
        + ["hf/hub/models--lerobot--smolvla_base/refs/main",
           "hf/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main"])
    assert (hub / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/refs/main").read_text() == VREV


def test_the_shipped_smolvla_base_carries_the_smolvlm2_snapshot_it_loads():
    lock = bases.load()
    vlm = [b for b in lock["bases"]["smolvla_base"] if b["source"] == f"hf:{VLM_REPO}"]
    assert vlm == lock["bases"]["smolvlm2_500m"]


def test_other_bytes_of_the_same_size_are_refused_and_not_kept(upstream, tmp_path, capsys):
    upstream.served[f"/lerobot/smolvla_base/resolve/{REV}/config.json"] = b'{"type": "smolvlb"}'
    assert _get(tmp_path, "--include", "smolvla_base") == 1
    out = capsys.readouterr().out
    want = hashlib.sha256(HUB["config.json"]).hexdigest()
    got = hashlib.sha256(b'{"type": "smolvlb"}').hexdigest()
    assert f"smolvla_base: hf/hub/models--lerobot--smolvla_base/snapshots/{REV}/config.json: sha256 {got} != expected {want}" in out
    assert "config.json" not in " ".join(_left(tmp_path))      # nor its temporary file


def test_a_truncated_file_is_refused_by_its_size(upstream, tmp_path, capsys):
    upstream.served[f"/big_vision/paligemma_tokenizer.model?generation={GEN}"] = TOKENIZER[:-1]
    assert _get(tmp_path, "--include", "pi05_droid") == 1
    assert f"pi05_droid: openpi/big_vision/paligemma_tokenizer.model: size {len(TOKENIZER) - 1} != expected " \
           f"{len(TOKENIZER)}" in capsys.readouterr().out
    assert not any("paligemma_tokenizer" in p for p in _left(tmp_path))


def test_an_unknown_base_is_refused_naming_the_bases(upstream, tmp_path, capsys):
    assert _get(tmp_path, "--include", "pi05_drod") == 2
    assert "no base 'pi05_drod': the bases are pi05_droid, smolvla_base" in capsys.readouterr().out
    assert upstream.requests == []


def test_a_connection_reset_mid_download_names_its_base_and_url_and_is_exit_3(upstream, tmp_path, capsys):
    upstream.reset = f"/lerobot/smolvla_base/resolve/{REV}/model.safetensors"
    assert _get(tmp_path, "--include", "smolvla_base") == 3
    out = capsys.readouterr().out
    assert "FAIL  smolvla_base: connection lost reading http://127.0.0.1:" in out and upstream.reset in out
    assert not any("model.safetensors" in p for p in _left(tmp_path))


def test_an_unreachable_file_names_its_base_and_url(upstream, tmp_path, capsys):
    del upstream.served[f"/lerobot/smolvla_base/resolve/{REV}/model.safetensors"]
    assert _get(tmp_path, "--include", "smolvla_base") == 3
    out = capsys.readouterr().out
    assert "FAIL  smolvla_base: HTTP 404" in out and f"/lerobot/smolvla_base/resolve/{REV}/model.safetensors" in out


def test_the_checksum_list_covers_exactly_the_bases_runs_name():
    lock = bases.load()
    declared = {k for k, v in tomllib.loads((RECIPES_DIR / "bases.toml").read_text()).items() if isinstance(v, dict)}
    assert set(lock["bases"]) == declared
    for name, blocks in lock["bases"].items():
        for b in blocks:
            assert b["dest"] == bases.dest_dir(b["source"], b.get("revision")) and b["files"], name


@pytest.mark.slow
@pytest.mark.network
def test_the_smallest_real_base_downloads_from_its_publisher_and_verifies(tmp_path, monkeypatch):
    """smolvlm2_500m (about 2.0 GB) from the Hugging Face Hub at its locked commit."""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    lock = bases.load()
    fetched, present = bases.fetch(lock, ["smolvlm2_500m"], tmp_path, hf_endpoint="https://huggingface.co")
    assert (fetched, present) == (sum(len(b["files"]) for b in lock["bases"]["smolvlm2_500m"]), 0)
