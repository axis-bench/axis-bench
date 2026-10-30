"""No module directly under axis_qvla/bench/servers/ may be named like a top-level package that
ANY server module in that directory imports.

Why this ratchet, not the venv-tier one: running `python axis_qvla/bench/servers/<x>.py` as a script
puts that directory at sys.path[0], ahead of every installed package. A sibling file sharing a
third-party import's name then IS that import -- `lerobot.py` shadowed the real `lerobot`
package for every server in the directory (not just itself), and the servers died at start with
`ModuleNotFoundError: No module named 'lerobot.policies'; 'lerobot' is not a package`.
tests/ratchets/test_venv_tiers.py checks which FRAMEWORKS a tier may import, by scope; it does
not (and should not -- the import here is deliberately function-local) check whether a file's
OWN NAME collides with one of them. This ratchet checks the name collision directly, by AST,
over every scope (module- or function-level import both create the hazard equally).
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVERS_DIR = REPO / "axis_qvla" / "bench" / "servers"


def _server_modules() -> list[Path]:
    return sorted(p for p in SERVERS_DIR.glob("*.py") if p.name != "__init__.py")


def _imported_top_level_names(path: Path) -> set[str]:
    """Every top-level package name the module imports, at ANY scope. `ast.walk` (not
    `tree.body`) on purpose: the hazard this guards against does not care whether the import is
    lazy -- a function-local `from lerobot...` is shadowed exactly the same way a module-level
    one would be."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_there_are_server_modules_to_check():
    """Guards the rest: a bad glob would make the collision check vacuous."""
    assert len(_server_modules()) >= 2


def test_no_server_module_shares_its_name_with_a_package_any_server_imports():
    modules = _server_modules()
    stems = {p.stem for p in modules}
    imported: set[str] = set()
    for p in modules:
        imported |= _imported_top_level_names(p)

    collisions = stems & imported
    assert not collisions, (
        f"{sorted(collisions)} name both a file directly under axis_qvla/bench/servers/ and a "
        f"top-level package some server in that directory imports. Launching any file in "
        f"the directory as a script puts the directory itself at sys.path[0], so the colliding "
        f"file shadows the real package for every server there, not just itself -- the import "
        f"fails with \"'<name>' is not a package\". Rename the file (e.g. lerobot.py -> "
        f"lerobot_server.py)."
    )
