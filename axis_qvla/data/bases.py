"""`qvla data get base-weights`: the base models a run starts from, fetched from their publishers.

Why: AXIS-Bench hosts no base weights. Each base downloads from where it is published -- the pi0.5
params and the PaliGemma tokenizer from Google Cloud Storage, the LeRobot bases from the Hugging
Face Hub -- and every file is checked against conf/recipes/bases.lock.json, which pins each GCS
object's generation, each Hub repository's commit, and every file's size and sha256. The bytes land
in the local copy every run reads (axis_qvla.data.layout.base_weights_dest): openpi's download cache
under openpi/ (a pi0.5 run's OPENPI_DATA_HOME) and a Hugging Face cache under hf/ (a LeRobot run's
HF_HOME, with HF_HUB_OFFLINE=1). A file is renamed into place only once verified
(axis_qvla.data.hf.download_url), and a file already present with the locked bytes is not fetched
again, so an interrupted get resumes.

Core tier: stdlib only.
"""
from __future__ import annotations

import json
import pathlib
import urllib.parse
from collections.abc import Callable, Iterable, Mapping

from axis_qvla.core.canon import sha256_file
from axis_qvla.core.lock import SOURCE_TREE
from axis_qvla.data.hf import HfDigestMismatch, HfHttpError, HfRepo, download_url, resolve_url

LOCK = SOURCE_TREE / "conf" / "recipes" / "bases.lock.json"
SCHEMA = "qvla.bases_lock/1"
GCS_ENDPOINT = "https://storage.googleapis.com"


class BasesError(ValueError):
    """The checksum list is unreadable, or names no such base."""


def load(path: pathlib.Path = LOCK) -> dict:
    try:
        doc = json.loads(pathlib.Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise BasesError(f"cannot read the base weights' checksum list {path}: {exc}") from None
    if doc.get("schema") != SCHEMA:
        raise BasesError(f"{path}: schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def dest_dir(source: str, revision: str | None) -> str:
    """Where a source's files go inside the local copy: openpi's cache path for a gs:// URL
    (`openpi/<bucket>/<object path>`), a Hub cache snapshot for `hf:<repo>`."""
    if source.startswith("gs://"):
        return "openpi/" + source.removeprefix("gs://").rstrip("/")
    org, name = source.removeprefix("hf:").split("/")
    return f"hf/hub/models--{org}--{name}/snapshots/{revision}"


def file_url(block: Mapping, row: Mapping, *, hf_endpoint: str, gcs_endpoint: str) -> str:
    """The publisher's URL of one file, at the locked generation or commit."""
    source = block["source"]
    if source.startswith("gs://"):
        bucket, _, prefix = source.removeprefix("gs://").partition("/")
        obj = f"{prefix}/{row['name']}" if prefix else row["name"]
        return f"{gcs_endpoint.rstrip('/')}/{bucket}/{urllib.parse.quote(obj)}?generation={row['generation']}"
    return resolve_url(HfRepo(source.removeprefix("hf:"), "model"), block["revision"], row["name"],
                       endpoint=hf_endpoint)


def select(lock: Mapping, names: Iterable[str]) -> list[str]:
    """`names` (every base when empty), each one the checksum list has."""
    names = list(names) or list(lock["bases"])
    unknown = [n for n in names if n not in lock["bases"]]
    if unknown:
        raise BasesError(f"no base {', '.join(map(repr, unknown))}: the bases are {', '.join(lock['bases'])}")
    return names


# What a reference policy server reads from the base-weights copy when it starts, beside the
# fine-tuned checkpoint it serves: base -> the block destinations it needs (None: every block).
# pi0.5 reads only the PaliGemma tokenizer (OPENPI_DATA_HOME); SmolVLA the SmolVLM2 backbone its
# config names; X-VLA its whole base, the BART tokenizer included (HF_HOME, offline), so all of it is
# fetched.
SERVING: dict[str, tuple[tuple[str, str | None], ...]] = {
    "pi05": (("pi05_base", "openpi/big_vision"),),
    "smolvla": (("smolvlm2_500m", None),),
    "xvla": (("xvla_base", None),),
}


def serving_lock(lock: Mapping, backbone: str) -> dict:
    """`lock` cut down to what a `backbone` reference server reads (SERVING): `qvla data get
    base-weights --serving pi05` fetches the 4 MB tokenizer, not the 12 GB parameters. Every file
    kept is still checked against the checksum list by `fetch`."""
    if backbone not in SERVING:
        raise BasesError(f"--serving {backbone!r}: one of {', '.join(SERVING)}")
    out = {}
    for base, dest in SERVING[backbone]:
        blocks = [b for b in lock["bases"].get(base, ()) if dest is None or b["dest"] == dest]
        if not blocks:
            raise BasesError(f"--serving {backbone}: the base weights' checksum list has no {base}"
                             + (f" block {dest}" if dest else ""))
        out[base] = blocks
    return {**lock, "bases": out}


def fetch(lock: Mapping, names: Iterable[str], dest: pathlib.Path, *, hf_endpoint: str,
          gcs_endpoint: str = GCS_ENDPOINT, report: Callable[[str], None] = print) -> tuple[int, int]:
    """Download and verify the files of each base in `names` into `dest`: (fetched, already present).
    A Hub source also gets the `refs/main` its snapshot is found by offline."""
    fetched = present = 0
    for base in select(lock, names):
        n0, p0 = fetched, present
        for block in lock["bases"][base]:
            for row in block["files"]:
                out = pathlib.Path(dest) / block["dest"] / row["name"]
                if out.is_file() and out.stat().st_size == row["size"] and sha256_file(out) == row["sha256"]:
                    present += 1
                    continue
                url = file_url(block, row, hf_endpoint=hf_endpoint, gcs_endpoint=gcs_endpoint)
                try:
                    download_url(url, out, expected_sha256=row["sha256"], expected_size=row["size"],
                                 label=f"{base}: {block['dest']}/{row['name']}")
                except HfDigestMismatch:
                    raise
                except HfHttpError as exc:
                    raise HfHttpError(f"{base}: {exc}") from None
                fetched += 1
            if block["source"].startswith("hf:"):
                ref = (pathlib.Path(dest) / block["dest"]).parent.parent / "refs" / "main"
                ref.parent.mkdir(parents=True, exist_ok=True)
                ref.write_text(block["revision"])
        report(f"{base}: {fetched - n0} fetched, {present - p0} already present -> {dest}")
    return fetched, present
