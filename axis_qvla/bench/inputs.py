"""What `bench eval` checks on disk before any server starts: the asset checksum manifest and the
Polyhaven texture bank under the render root.

The benchmark declares assets.fetch 'refuse', and with --asset-checksums every scene file the asset
store opens is checked against the benchmark's OWN version's checksum manifest (asset_lock_for(b):
ASSETS_LOCK for axis-heldout-v1, else <version dir>/ASSETS.lock.json). The checksum-manifest flag is
therefore always passed, and a checkout without the checksum manifest (or with one that is not a
JSON object with a list of file rows) is refused: passing it only when the file existed let a run
evaluate unchecked scenes. `refusal` takes that same checksum-manifest path as an argument rather
than defaulting to ASSETS_LOCK, so the pre-launch gate and the workers always check the one checksum
manifest the declaration actually uses.
The Polyhaven bank is never opened through the asset store -- the AXIS-Render shell reads it
straight from <render tree>/assets/table_textures/polyhaven/downloaded_256/, where the render tree
is the checkout the workers import (axis_qvla.bench.launch.render_tree) -- so `bench eval` compares
that directory with the checksum manifest's bank rows: the same names in both directions and every
file's sha256, or the run is refused. Entries are followed like the render shell follows them: a
copy or a symlink to the bytes the checksum manifest lists passes (docs/reference/verification.md#the-benchmark); a subdirectory, a dangling link or
anything else that does not open as a regular file is a problem.

Core tier: stdlib only. The bank is listed with os.scandir; an OSError refuses the run: an
unreadable directory never reads as an empty or a matching one.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

# REPO_ROOT from axis_qvla.core.hosts: a local Path(__file__).parents[1] would name axis/.
from axis_qvla.core.hosts import REPO_ROOT

ASSETS_LOCK = REPO_ROOT / "conf" / "benchmarks" / "axis-heldout-v1" / "ASSETS.lock.json"
BANK_LOCK_PREFIX = "assets/textures/polyhaven/downloaded_256/"
BANK_RENDER_REL = Path("assets") / "table_textures" / "polyhaven" / "downloaded_256"
_SHOWN = 5


def asset_lock_for(b) -> Path:
    """The asset lock of `b`'s version: ASSETS_LOCK for the default version (a test may point that
    constant elsewhere), else <version dir>/ASSETS.lock.json."""
    from axis_qvla.bench.spec import CONF_DIR   # noqa: PLC0415 - spec imports nothing from here

    own = Path(b.conf_dir) / "ASSETS.lock.json"
    return ASSETS_LOCK if Path(b.conf_dir).resolve() == Path(CONF_DIR).resolve() else own


def path_argv(asset_dir: Path | None, render_root: Path | None, task_data_root: Path | None,
              *, asset_lock: Path) -> list[str]:
    """The path flags `bench eval` hands the campaign driver, which forwards them to every worker:
    --asset-dir, --render-root and --task-data-root when given, and always --asset-checksums, the
    checksum manifest of the benchmark's own version (`asset_lock_for(b)`), which the caller names:
    no default, so the checksum manifest the workers check is never one the caller did not choose.
    `--k=v` tokens, like the rest of the composed argv."""
    given = (("asset-dir", asset_dir), ("render-root", render_root), ("task-data-root", task_data_root))
    return [f"--{k}={v.absolute()}" for k, v in given if v is not None] + [f"--asset-checksums={asset_lock}"]


def default_bench_dirs(paths) -> tuple[Path, Path]:
    """The assets/ and tasks/ of the local qvla-v1-bench copy, where `qvla data get bench` puts it
    (axis_qvla.data.layout.default_dest): what `bench eval` reads when --asset-dir / --task-data-root are
    not given, so a user who fetched the benchmark never types them. `paths` is an
    axis_qvla.core.paths.Paths."""
    from axis_qvla.data.layout import BENCH, default_dest   # noqa: PLC0415 - kept off import time

    root = default_dest(BENCH, paths)
    return root / "assets", root / "tasks"


def bank_source(asset_dir: Path) -> Path:
    """Where the bank's files sit in a qvla-v1-bench assets/ copy: the checksum manifest's bank rows
    (BANK_LOCK_PREFIX) are repository paths, and `asset_dir` is the repository's assets/."""
    return Path(asset_dir) / BANK_LOCK_PREFIX[len("assets/"):]


def link_bank(render_tree: Path, asset_dir: Path, lock_path: Path) -> list[str]:
    """Symlink each bank file the checksum manifest lists from the assets copy into the render tree,
    where the render shell reads it (BANK_RENDER_REL), and return the names linked. Why: copying the
    bank by hand would be the one step between `qvla data get bench` and a run. A name
    already under the render tree is left as it is, copy or link, and `bank_problems` then judges it;
    a file the copy lacks is not linked, so the refusal that follows names it. Nothing else is
    written. An unreadable checksum manifest links nothing (`refusal` names it); an unreadable
    directory raises OSError, never reads as empty."""
    try:
        lock = json.loads(Path(lock_path).read_bytes())
    except (OSError, ValueError):
        return []
    if lock_shape_problem(lock):
        return []
    names = sorted(n for n in (str(r.get("path", ""))[len(BANK_LOCK_PREFIX):] for r in lock["files"]
                               if str(r.get("path", "")).startswith(BANK_LOCK_PREFIX)) if n)
    src, dst = bank_source(asset_dir).absolute(), Path(render_tree) / BANK_RENDER_REL
    linked = []
    for n in names:
        if os.path.lexists(dst / n) or not (src / n).is_file():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        os.symlink(src / n, dst / n)
        linked.append(n)
    return linked


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_shape_problem(lock: object) -> str | None:
    """A lock that is not an object with a list of object rows: a problem line, never a traceback."""
    if not isinstance(lock, Mapping):
        return "ASSETS.lock.json is not a JSON object"
    files = lock.get("files")
    if not isinstance(files, list):
        return "ASSETS.lock.json has no files list"
    bad = [i for i, r in enumerate(files) if not isinstance(r, Mapping)]
    return f"ASSETS.lock.json files row {bad[0]} is not an object" if bad else None


def bank_problems(render_root: Path, lock: object) -> list[str]:
    """Every way <render_root>'s bank differs from the checksum manifest's bank rows; [] when identical."""
    shape = lock_shape_problem(lock)
    if shape:
        return [shape]
    rows = {str(r.get("path", ""))[len(BANK_LOCK_PREFIX):]: r for r in lock["files"]
            if str(r.get("path", "")).startswith(BANK_LOCK_PREFIX)}
    if not rows:
        return [f"ASSETS.lock.json has no rows under {BANK_LOCK_PREFIX}"]
    bank = Path(render_root) / BANK_RENDER_REL
    try:
        with os.scandir(bank) as it:
            found = {e.name: e.is_file() for e in it}
    except FileNotFoundError:
        return [f"{bank} does not exist: copy <bench>/{BANK_LOCK_PREFIX}* there "
                "(docs/reference/data.md#the-benchmark-inputs)"]
    except OSError as exc:
        return [f"{bank} cannot be read ({type(exc).__name__})"]
    out = [f"{n}: not a regular file" for n in sorted(found) if not found[n]]
    out += [f"{n}: in the checksum manifest, not under the render root" for n in sorted(set(rows) - set(found))]
    out += [f"{n}: under the render root, not in the checksum manifest" for n in sorted(n for n in found if found[n] and n not in rows)]
    for n in sorted(n for n in rows if found.get(n)):
        try:
            same = _sha256(bank / n) == rows[n].get("sha256")
        except OSError as exc:
            out.append(f"{n}: cannot be read ({type(exc).__name__})")
            continue
        out += [] if same else [f"{n}: sha256 differs from the checksum manifest"]
    return out


def refusal(render_tree: Path, lock_path: Path, *, dual_sim: bool) -> str | None:
    """Why `bench eval` must not start, or None: `lock_path` (the BENCHMARK'S OWN asset checksum
    manifest -- asset_lock_for(b), never a hidden default) is missing, unreadable or malformed, or
    (for the dual-sim render half) the bank under `render_tree` -- launch.render_tree, the checkout
    the workers import -- is not that checksum manifest's bank. A caller that checked a different
    version's checksum manifest here than it hands the workers (`path_argv`'s `asset_lock=`) would
    pass this gate on the wrong scenes, or refuse a run whose actual checksum manifest is fine."""
    lock_path = Path(lock_path)
    try:
        lock = json.loads(lock_path.read_bytes())
    except FileNotFoundError:
        return (f"the asset checksum manifest (ASSETS.lock.json) {lock_path} is missing: every scene file is checked "
                "against it. Refused before any server starts.")
    except (OSError, ValueError) as exc:
        return f"the asset checksum manifest {lock_path} cannot be read ({type(exc).__name__}). Refused before any server starts."
    shape = lock_shape_problem(lock)
    if shape:
        return f"the asset checksum manifest {lock_path}: {shape}. Refused before any server starts."
    probs = bank_problems(Path(render_tree), lock) if dual_sim else []
    if not probs:
        return None
    shown = "; ".join(probs[:_SHOWN]) + (f"; ... {len(probs) - _SHOWN} more" if len(probs) > _SHOWN else "")
    return (f"the Polyhaven bank under the render root is not the bank the checksum manifest lists "
            f"({len(probs)} problem(s)): {shown}. Refused before any server starts.")
