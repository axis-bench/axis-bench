"""The stdlib Hub client against a local HTTP fixture: pagination, verified downloads, and a
token that goes only where it belongs.

T0 and `release verify` run with nothing installed, so this client replaces huggingface_hub for
them. The redirect test matters most: the Hub redirects LFS downloads to a CDN host, and a token
forwarded there is both a leak and (for S3) a rejected request.
"""

from __future__ import annotations

import email.message
import hashlib
import http.server
import json
import socket
import struct
import threading
import urllib.parse
import urllib.request

import pytest

from axis_qvla.data import hf as hh

FILES = {"a.txt": b"hello", "dir/b.bin": b"x" * 10}
BLOB_A = hashlib.sha1(b"blob 5\0hello").hexdigest()
SHA_B = hashlib.sha256(b"x" * 10).hexdigest()
SHA_HELLO = hashlib.sha256(b"hello").hexdigest()
SHA_CDN = hashlib.sha256(b"cdn bytes").hexdigest()


class _Hub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, headers: dict | None = None) -> None:
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv = self.server
        srv.seen.append((self.path, self.headers.get("Authorization")))
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/api/datasets/o/r/tree/main":
            if "cursor=2" not in u.query:
                page = [{"type": "directory", "path": "dir", "oid": "0" * 40, "size": 0},
                        {"type": "file", "path": "a.txt", "oid": BLOB_A, "size": 5}]
                nxt = f'<{srv.base}/api/datasets/o/r/tree/main?recursive=true&expand=true&cursor=2>; rel="next"'
                return self._send(200, json.dumps(page).encode(), {"Link": nxt})
            page = [{"type": "file", "path": "dir/b.bin", "oid": "1" * 40, "size": 10,
                     "lfs": {"oid": SHA_B, "size": 10, "pointerSize": 130}}]
            return self._send(200, json.dumps(page).encode())
        if u.path == "/api/datasets/o/r/tree/badlink":
            page = [{"type": "file", "path": "a.txt", "oid": BLOB_A, "size": 5}]
            nxt = f'<{srv.other}/api/datasets/o/r/tree/main?recursive=true&cursor=2>; rel="next"'
            return self._send(200, json.dumps(page).encode(), {"Link": nxt})
        if u.path == "/incomplete":
            body = b'{"a": 1}'
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/reset":            # half the body, then the connection is reset
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b"x" * 500)
            self.wfile.flush()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
            return
        if u.path == "/api/datasets/o/r/revision/v1.0":
            return self._send(200, json.dumps({"sha": "c" * 40}).encode())
        if u.path == "/api/datasets/o/r/revision/unauthorized":
            return self._send(401, b"")
        if u.path.startswith("/datasets/o/r/resolve/"):
            rev, _, path = u.path[len("/datasets/o/r/resolve/"):].partition("/")
            path = urllib.parse.unquote(path)
            if path == "redir/cdn.bin":
                return self._send(302, b"", {"Location": f"{srv.other}/cdn/x"})
            if path == "redir/same.bin":
                return self._send(302, b"", {"Location": f"/datasets/o/r/resolve/{rev}/a.txt"})
            if path in FILES:
                return self._send(200, FILES[path])
        if u.path == "/cdn/x":
            return self._send(200, b"cdn bytes")
        return self._send(404, b"not found")


def _serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    srv.seen, srv.base = [], f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def hub():
    a, b = _serve(), _serve()
    a.other, b.other = b.base, a.base
    yield a, b
    for s in (a, b):
        s.shutdown()
        s.server_close()


REPO = hh.HfRepo("o/r", "dataset")


def test_resolve_url_quotes_the_path_and_prefixes_datasets():
    assert hh.resolve_url(REPO, "v1.0", "d/a b.json", endpoint="https://h/") == \
        "https://h/datasets/o/r/resolve/v1.0/d/a%20b.json"
    assert hh.resolve_url(hh.HfRepo("o/m", "model"), "main", "x", endpoint="https://h") == \
        "https://h/o/m/resolve/main/x"


def test_list_tree_follows_link_pagination_and_skips_directories(hub):
    a, _ = hub
    assert hh.list_tree(REPO, "main", token=None, endpoint=a.base) == [
        hh.TreeEntry("a.txt", 5, BLOB_A, None), hh.TreeEntry("dir/b.bin", 10, "1" * 40, SHA_B)]


def test_revision_commit(hub):
    a, _ = hub
    assert hh.revision_commit(REPO, "v1.0", token=None, endpoint=a.base) == "c" * 40


def test_the_token_is_sent_only_when_given(hub):
    a, _ = hub
    url = hh.resolve_url(REPO, "main", "a.txt", endpoint=a.base)
    assert hh.http_get(url, token=None) == b"hello"
    assert hh.http_get(url, token="secret-token") == b"hello"
    assert [auth for _, auth in a.seen] == [None, "Bearer secret-token"]


def test_a_redirect_to_another_host_drops_the_token_and_a_same_host_redirect_keeps_it(hub, tmp_path):
    a, cdn = hub
    hh.download(REPO, "main", "redir/cdn.bin", tmp_path / "c.bin", expected_sha256=SHA_CDN,
                token="secret-token", endpoint=a.base)
    assert cdn.seen == [("/cdn/x", None)]
    a.seen.clear()
    hh.download(REPO, "main", "redir/same.bin", tmp_path / "s.bin", expected_sha256=SHA_HELLO,
                token="secret-token", endpoint=a.base)
    assert [auth for _, auth in a.seen] == ["Bearer secret-token", "Bearer secret-token"]
    assert (tmp_path / "s.bin").read_bytes() == b"hello"


def test_download_verifies_the_sha256_and_leaves_no_partial_file(hub, tmp_path):
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match="sha256"):
        hh.download(REPO, "main", "dir/b.bin", tmp_path / "out" / "b.bin", expected_sha256="0" * 64,
                    token=None, endpoint=a.base)
    assert list((tmp_path / "out").iterdir()) == []          # neither the file nor its partial
    got = hh.download(REPO, "main", "dir/b.bin", tmp_path / "out" / "b.bin", expected_sha256=SHA_B,
                      token=None, endpoint=a.base)
    assert got.read_bytes() == b"x" * 10


def test_download_requires_expected_sha256(tmp_path):
    """No None path that renames unverified bytes -- expected_sha256 is required."""
    with pytest.raises(TypeError):
        hh.download(REPO, "main", "a.txt", tmp_path / "a.bin", token=None, endpoint="http://x")


def test_download_rejects_an_explicit_none_sha256_instead_of_skipping_verification(hub, tmp_path):
    """The old signature let `expected_sha256=None` mean 'skip the check' -- that silently
    renamed unverified bytes into place. Passing None must now fail closed, not succeed."""
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match="sha256"):
        hh.download(REPO, "main", "a.txt", tmp_path / "a.bin", expected_sha256=None,
                    token=None, endpoint=a.base)
    assert not (tmp_path / "a.bin").exists()


def test_download_checks_size_when_the_caller_knows_it(hub, tmp_path):
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match="size"):
        hh.download(REPO, "main", "a.txt", tmp_path / "a.bin", expected_sha256=SHA_HELLO,
                    expected_size=999, token=None, endpoint=a.base)
    assert not (tmp_path / "a.bin").exists()
    got = hh.download(REPO, "main", "a.txt", tmp_path / "a.bin", expected_sha256=SHA_HELLO,
                      expected_size=5, token=None, endpoint=a.base)
    assert got.read_bytes() == b"hello"


def test_errors_name_the_status_and_never_the_token(hub):
    a, _ = hub
    with pytest.raises(hh.HfHttpError) as exc:
        hh.http_get(f"{a.base}/nope", token="secret-token")
    assert "404" in str(exc.value) and "secret-token" not in str(exc.value)


def test_env_helpers(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hh.endpoint_from_env() == "https://huggingface.co" and hh.token_from_env() is None
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    assert hh.endpoint_from_env() == "https://hf-mirror.com"


def test_autouse_fixture_hides_a_real_token_and_pins_a_loopback_endpoint():
    """tests/unit/conftest.py's autouse fixture must already have run by the time this
    body executes, so a real HF_TOKEN exported on the machine that runs the suite
    never reaches a test that forgets to override it itself. Run with HF_TOKEN=dummy exported in
    the shell to prove the outer environment cannot leak through -- this test takes no monkeypatch
    of its own, so it sees exactly what the autouse fixture left behind."""
    assert hh.token_from_env() is None
    assert hh.endpoint_from_env() == "http://127.0.0.1:9"


def test_same_host_auth_pins_scheme_and_host_not_just_host():
    """An https->http (or a different-host) redirect at the SAME netloc-looking host
    must not carry the token -- only an identical scheme+host redirect may."""
    req = urllib.request.Request("http://h.example/x")
    req.add_unredirected_header("Authorization", "Bearer t")
    handler = hh._SameHostAuth()
    hdrs = email.message.Message()
    scheme_change = handler.redirect_request(req, None, 302, "Found", hdrs, "https://h.example/y")
    assert scheme_change.unredirected_hdrs.get("Authorization") is None
    same = handler.redirect_request(req, None, 302, "Found", hdrs, "http://h.example/z")
    assert same.unredirected_hdrs.get("Authorization") == "Bearer t"
    host_change = handler.redirect_request(req, None, 302, "Found", hdrs, "http://other.example/y")
    assert host_change.unredirected_hdrs.get("Authorization") is None


def test_list_tree_refuses_a_pagination_link_that_points_off_endpoint(hub):
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match="off-endpoint"):
        hh.list_tree(REPO, "badlink", token=None, endpoint=a.base)


def test_an_incomplete_read_becomes_an_hfhttperror_not_a_raw_traceback(hub):
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match="incomplete"):
        hh.http_get(f"{a.base}/incomplete", token=None)


def test_a_connection_reset_mid_body_is_a_network_error_naming_the_url(hub, tmp_path):
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match=f"{a.base}/reset") as exc:
        hh.download_url(f"{a.base}/reset", tmp_path / "f.bin", expected_sha256="0" * 64, label="f")
    assert not isinstance(exc.value, hh.HfDigestMismatch)
    assert list(tmp_path.iterdir()) == []


def test_a_body_shorter_than_its_content_length_is_an_incomplete_download_not_a_mismatch(hub, tmp_path):
    a, _ = hub
    with pytest.raises(hh.HfHttpError, match=f"incomplete .*{a.base}/incomplete") as exc:
        hh.download_url(f"{a.base}/incomplete", tmp_path / "f.json", expected_sha256="0" * 64,
                        expected_size=1000, label="f")
    assert not isinstance(exc.value, hh.HfDigestMismatch)
    assert list(tmp_path.iterdir()) == []


def test_a_partial_a_killed_download_left_is_removed_when_the_file_is_fetched_again(hub, tmp_path):
    a, _ = hub
    (tmp_path / ".a.bin.part-1").write_bytes(b"x" * 100)       # another process, since killed
    (tmp_path / ".other.part-1").write_bytes(b"y")             # another file's: not this fetch's to remove
    hh.download(REPO, "main", "a.txt", tmp_path / "a.bin", expected_sha256=SHA_HELLO, token=None, endpoint=a.base)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".other.part-1", "a.bin"]


def test_a_mismatch_prints_both_full_digests(hub, tmp_path):
    a, _ = hub
    with pytest.raises(hh.HfDigestMismatch) as exc:
        hh.download(REPO, "main", "a.txt", tmp_path / "a.bin", expected_sha256="0" * 64, token=None, endpoint=a.base)
    assert f"sha256 {SHA_HELLO} != expected {'0' * 64}" in str(exc.value)
