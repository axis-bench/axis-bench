"""Where this machine keeps AXIS-Bench's data and environments: a user `qvla.toml`.

Why: the repository ships no machine profiles. A user writes one with `qvla init` (or copies
qvla.example.toml to qvla.toml and sets its roots); it holds `[roots]` (corpus, cache, scratch,
runs plus the evaluation roots), `[venv]` and `gpus`, so every command builds its paths through one
`axis_qvla.core.paths.Paths` and a stored relative path means the same thing on every machine.

Resolution, first match wins: `--host <path>.toml`; `--host <name>` (`<name>.toml` in
$AXIS_HOSTS_DIR); ./qvla.toml; $AXIS_HOST.

`qvla init` writes one (`write_user_config`).

Core tier: stdlib only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from axis_qvla.core.hosts import HostProfile, HostProfileError, load_host
from axis_qvla.core.paths import Paths

USER_CONFIG = "qvla.toml"


def _is_file(p: Path) -> bool:
    """Path.is_file(), but an untraversable parent directory (`is_file` swallows ENOENT but
    RAISES PermissionError rather than quietly returning False) becomes a named HostProfileError
    instead of an uncaught exception -- EACCES must never read as 'absent'."""
    try:
        return p.is_file()
    except OSError as exc:
        raise HostProfileError(
            f"cannot access {p}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable (e.g. not mode 000)."
        ) from None


def load_profile(host: str | None, *, cwd: Path | None = None) -> HostProfile:
    cwd = Path.cwd() if cwd is None else Path(cwd)
    if host and host.endswith(".toml"):
        p = Path(host) if Path(host).is_absolute() else cwd / host
        if not _is_file(p):
            raise HostProfileError(f"--host {host}: no such file")
        return load_host(p.name[: -len(".toml")], hosts_dir=p.parent)
    if host:
        return load_host(host)
    p = cwd / USER_CONFIG
    if _is_file(p):
        return load_host(p.name[: -len(".toml")], hosts_dir=p.parent)
    if os.environ.get("AXIS_HOST"):
        return load_host(None)
    raise HostProfileError(f"no {USER_CONFIG} in {cwd} and no --host: run `qvla init` here to write one "
                           f"(or copy qvla.example.toml to {USER_CONFIG} and set its [roots])")


def user_paths(host: str | None = None, *, cwd: Path | None = None) -> Paths:
    return Paths(load_profile(host, cwd=cwd))


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = "srv"
# What `qvla init` writes, in qvla.example.toml's order: (key, "root" | "checkout", relative path).
# Every data root sits under one directory and every checkout path under this checkout.
# A release test keeps the keys equal to the shipped example's qvla.example.toml.
INIT_ROOTS = (("corpus", "root", "corpus"), ("cache", "root", "cache"), ("scratch", "root", "scratch"),
              ("runs", "root", "runs"), ("r1", "root", "r1"), ("rollout_logs", "root", "r1/rollout_logs"),
              ("bench", "checkout", ""),
              ("openpi", "checkout", "third_party/openpi"), ("eval", "root", "eval"),
              ("smolvla_bench", "checkout", ""), ("smolvla_venv", "checkout", "third_party/lerobot/.venv"),
              ("hf_cache", "root", "hf"))
INIT_VENVS = (("eval", ".venv-eval"), ("online", "third_party/openpi/.venv"))


class InitError(ValueError):
    """`qvla init` would replace an existing qvla.toml, or was given an invalid value."""


def init_text(root: Path, checkout: Path, *, gpus: int = 1) -> str:
    """The qvla.toml `qvla init` writes: absolute roots, each path a TOML (JSON-compatible) string."""
    def at(base: Path, rel: str) -> str:
        return json.dumps(str(base / rel if rel else base))

    lines = ["# Written by `qvla init`: the data roots under one directory, the checkout paths under this",
             "# checkout. Edit a path to move that root.", f"gpus = {int(gpus)}", "", "[roots]"]
    lines += [f"{k} = {at(root if base == 'root' else checkout, rel)}" for k, base, rel in INIT_ROOTS]
    lines += ["", "[venv]"] + [f"{k} = {at(checkout, rel)}" for k, rel in INIT_VENVS]
    return "\n".join(lines) + "\n"


def write_user_config(cwd: Path, root: Path | str = DEFAULT_ROOT, *, force: bool = False, gpus: int = 1,
                      checkout: Path = REPO) -> Path:
    """Write `<cwd>/qvla.toml` with every data root under `root` (relative to `cwd`) and create those
    directories. The text is loaded as a host profile before it replaces anything, so a profile
    every command can read is the only thing ever written."""
    cwd = Path(cwd)
    root = Path(os.path.abspath(root if Path(root).is_absolute() else cwd / root))
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 0:
        raise InitError(f"--gpus must be a non-negative integer, got {gpus!r}")
    dest = cwd / USER_CONFIG
    if _is_file(dest) and not force:
        raise InitError(f"{dest} already exists; pass --force to replace it")
    tmp = cwd / f".qvla-init-{os.getpid()}.toml"
    tmp.write_text(init_text(root, Path(checkout), gpus=gpus), encoding="utf-8")
    try:
        load_host(tmp.name[: -len(".toml")], hosts_dir=cwd)
        for _, base, rel in INIT_ROOTS:
            if base == "root":
                (root / rel).mkdir(parents=True, exist_ok=True)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest
