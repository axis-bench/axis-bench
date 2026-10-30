"""A stdlib Hugging Face Hub client: resolve URLs, verified downloads, tree listings, revisions.

CORE TIER (stdlib only). T0 (`qvla paper tables --results hf://...`) and `qvla release verify`
must run on a machine with nothing installed, so neither may import huggingface_hub; this is
the small part of the Hub's REST API they need, and nothing more.

AUTH. A token is sent only when one is given, and only to the endpoint's own SCHEME AND host: the
resolve endpoint redirects LFS files to a CDN/S3 URL that must not receive it (S3 also rejects a
request carrying two auth mechanisms), and an https->http downgrade at the same host must drop it
too. No error message ever contains the token.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_ENDPOINT = "https://huggingface.co"
_NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="next"')
_HEX40 = re.compile(r"[0-9a-f]{40}")


class HfHttpError(RuntimeError):
    """The Hub answered with an error, could not be reached, or served the wrong bytes."""


class HfDigestMismatch(HfHttpError):
    """`download` fetched the file in full but its size or sha256 does not match what was
    asked for. A typed subclass so a caller can tell "the bytes are wrong"
    from "the Hub could not be reached" by type, never by parsing this exception's message."""


@dataclasses.dataclass(frozen=True)
class HfRepo:
    repo_id: str
    repo_type: str


@dataclasses.dataclass(frozen=True)
class TreeEntry:
    path: str
    size: int
    git_oid: str
    lfs_sha256: str | None


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Forward Authorization on a redirect only when BOTH the scheme and the host are unchanged.

    Pinning host alone would still leak the token on an https->http downgrade at the same host
    (or, in principle, any redirect the Hub could point at that shares a netloc but not the
    scheme the token was meant for).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        old_u, new_u = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        same = (old_u.scheme, old_u.netloc) == (new_u.scheme, new_u.netloc)
        if new is not None and auth and same:
            new.add_unredirected_header("Authorization", auth)
        return new


_OPENER = urllib.request.build_opener(_SameHostAuth)


def endpoint_from_env() -> str:
    return (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def token_from_env() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def resolve_url(repo: HfRepo, rev: str, path: str, *, endpoint: str) -> str:
    prefix = "datasets/" if repo.repo_type == "dataset" else ""
    return (f"{endpoint.rstrip('/')}/{prefix}{repo.repo_id}/resolve/"
            f"{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(path)}")


def _api(repo: HfRepo, endpoint: str, what: str, rev: str) -> str:
    return (f"{endpoint.rstrip('/')}/api/{repo.repo_type}s/{repo.repo_id}/{what}/"
            f"{urllib.parse.quote(rev, safe='')}")


def _open(url: str, token: str | None, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": "axis-qvla/1.0"})
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise HfHttpError(f"HTTP {exc.code} {exc.reason} for {url}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise HfHttpError(f"cannot reach {url}: {type(exc).__name__}: {getattr(exc, 'reason', '')}") from None


def _read(resp, url: str, size: int | None = None) -> bytes:
    """`resp.read()`, but a connection that dies mid-body (cut short, reset, timed out) becomes an
    HfHttpError naming `url`, not a raw http.client or socket traceback."""
    try:
        return resp.read() if size is None else resp.read(size)
    except http.client.IncompleteRead as exc:
        raise HfHttpError(f"incomplete read ({len(exc.partial)} of "
                          f"{len(exc.partial) + (exc.expected or 0)} bytes) of {url}") from None
    except OSError as exc:
        raise HfHttpError(f"connection lost reading {url}: {type(exc).__name__}: {exc}") from None


def http_get(url: str, *, token: str | None, timeout: float = 60.0) -> bytes:
    with _open(url, token, timeout) as resp:
        return _read(resp, url)


def download(repo: HfRepo, rev: str, path: str, dest: pathlib.Path, *, expected_sha256: str,
             token: str | None, endpoint: str, expected_size: int | None = None) -> pathlib.Path:
    """`download_url` of `path` in `repo` at `rev`."""
    return download_url(resolve_url(repo, rev, path, endpoint=endpoint), dest, expected_sha256=expected_sha256,
                        expected_size=expected_size, token=token, label=f"{repo.repo_id}@{rev}:{path}")


def download_url(url: str, dest: pathlib.Path, *, expected_sha256: str, expected_size: int | None = None,
                 token: str | None = None, label: str) -> pathlib.Path:
    """Stream to a temporary file beside `dest`, verify, then rename; a refused download leaves no
    partial file. `expected_sha256` is required, so nothing unverified is renamed into place; a size
    the caller knows (a tree listing has it) is checked too, for a clearer message on a truncation.
    `label` names the file in a refusal. A partial of this file left by a killed process is
    removed first."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(dest.parent):
        if name.startswith(f".{dest.name}.part-"):
            (dest.parent / name).unlink()
    tmp = dest.with_name(f".{dest.name}.part-{os.getpid()}")
    h = hashlib.sha256()
    n = 0
    try:
        with _open(url, token, 60.0) as resp, open(tmp, "wb") as fh:
            for chunk in iter(lambda: _read(resp, url, 1 << 20), b""):
                h.update(chunk)
                fh.write(chunk)
                n += len(chunk)
            served = resp.headers.get("Content-Length")
        if served is not None and served.isdigit() and n < int(served):
            raise HfHttpError(f"incomplete download of {url}: {n} of {served} bytes")
        if expected_size is not None and n != expected_size:
            raise HfDigestMismatch(f"{label}: size {n} != expected {expected_size}")
        if h.hexdigest() != expected_sha256:
            raise HfDigestMismatch(f"{label}: sha256 {h.hexdigest()} != expected {expected_sha256}")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def list_tree(repo: HfRepo, rev: str, *, token: str | None, endpoint: str) -> list[TreeEntry]:
    """Every file of `repo` at `rev`, following the Hub's pagination. A Link that leaves the
    endpoint is refused: the token would follow it."""
    ep = urllib.parse.urlsplit(endpoint)
    url: str | None = _api(repo, endpoint, "tree", rev) + "?recursive=true"
    out: list[TreeEntry] = []
    while url:
        with _open(url, token, 120.0) as resp:
            page = json.loads(_read(resp, url))
            link = resp.headers.get("Link", "")
        for item in page:
            if item["type"] == "file":
                lfs = item.get("lfs")
                out.append(TreeEntry(path=item["path"], size=int(item["size"]), git_oid=item["oid"],
                                     lfs_sha256=lfs["oid"] if lfs else None))
        m = _NEXT.search(link)
        if not m:
            break
        nxt = urllib.parse.urljoin(url, m.group(1))
        nu = urllib.parse.urlsplit(nxt)
        if (nu.scheme, nu.netloc) != (ep.scheme, ep.netloc):
            raise HfHttpError(f"list_tree: pagination Link points off-endpoint: {nxt}")
        url = nxt
    return sorted(out, key=lambda e: e.path)


def revision_commit(repo: HfRepo, rev: str, *, token: str | None, endpoint: str) -> str:
    doc = json.loads(http_get(_api(repo, endpoint, "revision", rev), token=token))
    sha = doc.get("sha") if isinstance(doc, dict) else None
    if not isinstance(sha, str) or not _HEX40.fullmatch(sha):
        raise HfHttpError(f"{repo.repo_id}@{rev}: the revision endpoint returned no commit sha")
    return sha
