"""A qvla.toml for the step commands' tests, and the real-world tutorial's local layout under its cache root."""
from __future__ import annotations

from pathlib import Path


def qvla_toml(root: Path) -> Path:
    """`root`/qvla.toml: the four data roots under `root`/roots, openpi's checkout and its venv (never built)."""
    roots = {k: root / "roots" / k for k in ("corpus", "cache", "scratch", "runs")}
    roots["openpi"] = root / "openpi"
    body = "\n".join(f'{k} = "{v}"' for k, v in roots.items())
    path = root / "qvla.toml"
    path.write_text(f'gpus = 1\n\n[roots]\n{body}\n\n[venv]\nonline = "{root / "openpi" / ".venv"}"\n',
                    encoding="utf-8")
    return path


def local_layout(tmp_path: Path):
    """(the qvla.toml path, its Layout) with d8/, built/, demos/, compiled labels and both norm files built
    (tests/unit/realrobot/fr3/local_fixture.py)."""
    from axis_qvla.realrobot.layout import layout
    from tests.unit.realrobot.fr3.local_fixture import build_cache

    toml = qvla_toml(tmp_path)
    lay = layout(str(toml))
    build_cache(lay.paths.cache, tmp_path / "droid")
    return str(toml), lay
