"""The gripper model the renderer reads, and the lock that pins it.

Render-tier package, stdlib only. The target gripper is MuJoCo Menagerie's Robotiq 2F-85 at one pinned
Menagerie commit (menagerie.lock.json). axis-render ships that directory, with its LICENSE, at
assets/robots/robotiq_2f85/; a Menagerie checkout at the same commit works too. The user passes the
directory that holds robotiq_2f85/ (`--robots`), and every file under it is checked against the lock
before a model is built from it; nothing is fetched. The arm is the recorded scene's own Panda: the
published render shares its kinematics exactly (FK of the published joints reproduces the published
end-effector pose to 1e-7 m).

    python -m axis_qvla.realrobot.render.assets lock --menagerie DIR --commit SHA > menagerie.lock.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GRIPPER_DIR = "robotiq_2f85"
GRIPPER_MJCF = f"{GRIPPER_DIR}/2f85.xml"
LOCK = Path(__file__).with_name("menagerie.lock.json")
SCHEMA = "qvla.menagerie_lock/1"
REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"


class AssetError(ValueError):
    """A Menagerie checkout that is not the locked one."""


def _files(root: Path) -> list[str]:
    base = Path(root) / GRIPPER_DIR
    if not base.is_dir():
        raise AssetError(f"{root}: not a Menagerie checkout (no {GRIPPER_DIR}/)")
    return sorted(p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file())


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_lock(root: Path, commit: str) -> dict:
    return {"schema": SCHEMA, "repository": REPOSITORY, "commit": commit,
            "files": {rel: _sha(Path(root) / rel) for rel in _files(root)}}


def check_menagerie(root: Path, lock: dict | None = None) -> Path:
    """The 2F-85 MJCF path inside `root`, after every locked file matched; raises AssetError otherwise."""
    lock = lock if lock is not None else json.loads(LOCK.read_text(encoding="utf-8"))
    have, want = set(_files(root)), set(lock["files"])
    problems = [f"missing {r}" for r in sorted(want - have)] + [f"unlocked {r}" for r in sorted(have - want)]
    problems += [f"{r}: sha256 differs" for r in sorted(want & have) if _sha(Path(root) / r) != lock["files"][r]]
    if problems:
        raise AssetError(f"{root} is not the locked Menagerie (commit {lock['commit'][:12]}): {problems[:5]}")
    return Path(root) / GRIPPER_MJCF


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.realrobot.render.assets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lock", help="print the lock of a Menagerie checkout")
    lk.add_argument("--menagerie", type=Path, required=True)
    lk.add_argument("--commit", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(build_lock(a.menagerie, a.commit), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
