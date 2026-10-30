"""T0 and the scorer read directories only through os.scandir and never consult a clock or the
environment."""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BANNED_MODULES = {"glob", "datetime", "time"}
BANNED_ATTRS = {"listdir", "glob", "iterdir", "rglob", "environ", "getenv"}
# axis_qvla/paper modules off the T0 path, each with why. `paper reeval`'s runner starts `bench eval`
# children (their environment is this one plus PYTHONPATH) and times each; `paper tables` never
# imports it (tests/golden/paper/test_t0_version_independence.py stops its walk there too).
NOT_T0 = {"reeval_run.py"}


def _modules() -> list[Path]:
    out = []
    for pkg in (("paper",), ("bench", "results")):   # T0, and the scorer
        out += sorted(p for p in (REPO / "axis_qvla").joinpath(*pkg).glob("*.py")
                      if not (pkg == ("paper",) and p.name in NOT_T0))
    out.append(REPO / "axis_qvla" / "core" / "ltr.py")    # the scorer's left-to-right sums
    assert len(out) >= 20, "the ratchet must see the T0 and scorer modules"
    return out


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bad += [a.name for a in n.names if a.name.split(".")[0] in BANNED_MODULES]
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] in BANNED_MODULES:
            bad.append(n.module)
        elif isinstance(n, ast.Attribute) and n.attr in BANNED_ATTRS:
            bad.append(f".{n.attr}")
        elif (isinstance(n, ast.Attribute) and n.attr == "exists" and isinstance(n.value, ast.Attribute)
              and n.value.attr == "path" and isinstance(n.value.value, ast.Name) and n.value.value.id == "os"):
            bad.append("os.path.exists")
    return bad


def test_no_banned_listing_clock_or_environment_calls():
    offenders = {p.relative_to(REPO).as_posix(): v for p in _modules() if (v := _violations(p))}
    assert offenders == {}


def test_the_ratchet_catches_each_banned_form(tmp_path: Path):
    src = ("import glob\nimport os\nfrom datetime import date\nimport time\n"
           "os.listdir('.'); os.environ['X']; os.path.exists('x'); p.iterdir(); p.rglob('*'); p.glob('*')\n")
    f = tmp_path / "m.py"; f.write_text(src)
    assert sorted(_violations(f)) == sorted(["glob", "datetime", "time", ".listdir", ".environ", "os.path.exists",
                                             ".iterdir", ".rglob", ".glob"])
