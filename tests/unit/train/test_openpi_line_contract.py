"""Every openpi config the generator renders has exactly the keys, and the JSON types, that the
openpi line's config reader requires: at every level, for every pi0.5 paper arm and every run spec of
the real-world line (`qvla realrobot fr3 local-runs`), as a training run, a smoke run and a policy server
render it.

Source of truth for the line's side: `src/openpi/training/axis_config.py` of the openpi line, read by
AST (the line runs in its own environment, so it is never imported here). Its key sets `_TOP`,
`_DATA` and `_IDENTITY` are what `train_config_from_dict` requires, and `_DATA_OPTIONAL` names the
data keys a config may leave out. Its `_TYPES` table gives every scalar's JSON type per level, and
the line's own test (`axis_config_keys_test.py::test_the_type_tables_cover_exactly_the_key_sets`)
holds that table equal to those sets and to the dataclass fields the line builds. The line refuses a config whose keys
differ at any level, so a key added on one side and not the other fails here, before any run or
server is started.

The line is the checked-out submodule `third_party/openpi`, at its pin. A checkout without the
submodule reads the same file from a nested openpi repository at the commit its submodule table
pins for the line, never at a branch. Finding neither fails, it never skips.
"""
from __future__ import annotations

import ast
import copy
import pathlib
import subprocess
import tomllib

import pytest

from axis_qvla.train.armspec import load_armspec
from axis_qvla.train.generate import openpi_config
from axis_qvla.train.paper import paper_arm_files
from axis_qvla.train.recipes import load_recipe
from tests.unit.train._arms import H, host

REPO = pathlib.Path(__file__).resolve().parents[3]
LINE_FILE = "src/openpi/training/axis_config.py"
GROUPS = ("data", "identity", "model", "lr_schedule", "optimizer", "weight_loader")
JSON = {"str": lambda v: isinstance(v, str), "bool": lambda v: isinstance(v, bool),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, int | float) and not isinstance(v, bool), None: lambda v: v is None}


def _line_source(rel: str) -> str:
    sub = REPO / "third_party" / "openpi" / rel
    if sub.is_file():
        return sub.read_text()
    # No submodule: the pin is the checkout's submodule table's.
    pins = REPO / "conf" / "release" / "submodules.toml"
    if not pins.is_file():
        pytest.fail(f"no openpi line to check the generator against: third_party/openpi/{rel} is missing "
                    "(the submodule is not checked out)", pytrace=False)
    commit = tomllib.loads(pins.read_text())["openpi"]["commit"]
    r = subprocess.run(["git", "-C", str(REPO / "openpi"), "show", f"{commit}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode:
        pytest.fail(f"no openpi line to check the generator against: neither third_party/openpi/{rel} nor "
                    f"the pinned {commit[:12]}:{rel} in ./openpi ({r.stderr.strip()[:200]})", pytrace=False)
    return r.stdout


def line_contract(source: str) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, tuple]]]:
    """(level -> the keys the reader requires, level -> the keys it also reads when given, level ->
    {scalar key: JSON types}) from the line's axis_config.py. Levels are `config` (the top) and
    `config.<group>`."""
    sets, types, optional = {}, {}, {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name, value = node.target.id, node.value
        else:
            continue
        if name in ("_TOP", "_DATA", "_IDENTITY"):
            sets[name] = set(ast.literal_eval(value.args[0]))
        elif name == "_DATA_OPTIONAL":
            optional["config.data"] = set(ast.literal_eval(value))
        elif name == "_TYPES":
            for level, table in zip(value.keys, value.values, strict=True):
                types[level.value] = {k.value: tuple(None if isinstance(e, ast.Constant) else e.id for e in v.elts)
                                      for k, v in zip(table.keys, table.values, strict=True)}
    missing = sorted({"_TOP", "_DATA", "_IDENTITY"} - set(sets)) + ([] if types else ["_TYPES"])
    if missing:
        pytest.fail(f"{LINE_FILE} of the openpi line defines no {missing}", pytrace=False)
    required = {"config": sets["_TOP"], "config.data": sets["_DATA"], "config.identity": sets["_IDENTITY"]}
    for level, table in types.items():
        required.setdefault(level, set(table) - optional.get(level, set()))
    return required, optional, types


def differences(cfg: dict, required: dict[str, set[str]], types: dict[str, dict[str, tuple]],
                optional: dict[str, set[str]] | None = None) -> list[str]:
    """Every way `cfg` is not a config the line reads."""
    optional = OPTIONAL if optional is None else optional
    def at(level: str):
        return cfg if level == "config" else cfg.get(level.split(".", 1)[1])

    out = []
    for level, keys in required.items():
        d = at(level)
        if not isinstance(d, dict):
            out.append(f"{level}: not an object")
        elif set(d) - keys - optional.get(level, set()) or keys - set(d):
            out.append(f"{level}: the generator writes {sorted(set(d) - keys - optional.get(level, set()))} that the "
                       f"line does not read and omits {sorted(keys - set(d))} that it requires")
    for level, table in types.items():
        d = at(level)
        for key, kinds in table.items():
            if isinstance(d, dict) and key in d and not any(JSON[k](d[key]) for k in kinds):
                out.append(f"{level}.{key}: {d[key]!r} is not {' or '.join(str(k) for k in kinds)}")
    return out


REQUIRED, OPTIONAL, TYPES = line_contract(_line_source(LINE_FILE))


def _pi05_arms():
    return [s for s in (load_armspec(p) for p in paper_arm_files()) if s.backbone == "pi05-openpi"]


def test_the_lines_tables_agree_with_each_other():
    assert set(TYPES["config"]) == REQUIRED["config"] - set(GROUPS)
    assert set(TYPES) == {"config", *(f"config.{g}" for g in GROUPS)}
    for group in GROUPS:
        level = f"config.{group}"
        assert set(TYPES[level]) == REQUIRED[level] | OPTIONAL.get(level, set()), group
    assert "ranges_path" in REQUIRED["config.data"]


@pytest.mark.parametrize("mode", ["train", "smoke", "serve"])
def test_every_pi05_paper_arm_renders_a_config_the_line_reads(tmp_path, mode):
    _, paths = host(tmp_path)
    arms = _pi05_arms()
    assert len(arms) == 59
    problems = {}
    for spec in arms:
        kw = {"serve": True} if mode == "serve" else {"smoke": mode == "smoke"}
        if mode != "serve" and spec.bundle is None:  # a finetune's bundle is compiled when it is bound
            kw.update(bundle=f"sha256:{H}", bundle_dir=str(tmp_path / "bundles" / H))
        found = differences(openpi_config(spec, load_recipe(spec.preset), paths, **kw), REQUIRED, TYPES)
        if found:
            problems[spec.id] = found
    assert problems == {}


@pytest.fixture(scope="module")
def real_world_runs(tmp_path_factory):
    """(Paths, {run id: run spec}) of every run spec `fr3 local-runs` writes over the tutorial's local layout."""
    from axis_qvla.realrobot.fr3 import local_runs
    from axis_qvla.realrobot.fr3.line import load_line
    from tests.unit.realrobot.fr3.local_fixture import build_local

    _toml, paths = build_local(tmp_path_factory.mktemp("rw"), runs=("bc", "conditioning", "awr", "filtering"))
    line = load_line(local_runs.LINE)
    written = local_runs.write_local_runs(paths, line)
    assert sorted(written) == sorted(line.runs)
    return paths, {r: local_runs.load_local_run(p, paths) for r, p in written.items()}


@pytest.mark.parametrize("mode", ["train", "smoke", "serve"])
def test_every_real_world_run_spec_renders_a_config_the_line_reads(real_world_runs, tmp_path, mode):
    paths, specs = real_world_runs
    problems = {}
    for run_id, spec in specs.items():
        kw = {"serve": True} if mode == "serve" else {"smoke": mode == "smoke"}
        if mode != "serve" and spec.bundle is None:  # a finetune's bundle is compiled when it is bound
            kw.update(bundle=f"sha256:{H}", bundle_dir=str(tmp_path / "bundles" / H))
        found = differences(openpi_config(spec, load_recipe(spec.preset), paths, **kw), REQUIRED, TYPES)
        if found:
            problems[run_id] = found
    assert problems == {}


def test_a_key_on_one_side_only_is_reported(tmp_path):
    _, paths = host(tmp_path)
    spec = next(s for s in _pi05_arms() if s.stage == "finetune")
    cfg = openpi_config(spec, load_recipe(spec.preset), paths, serve=True)
    assert differences(cfg, REQUIRED, TYPES) == []
    added = copy.deepcopy(cfg)
    added["data"]["extra"] = "x"
    assert differences(added, REQUIRED, TYPES) == [
        "config.data: the generator writes ['extra'] that the line does not read and omits [] that it requires"]
    dropped = copy.deepcopy(cfg)
    del dropped["data"]["ranges_path"]
    assert differences(dropped, REQUIRED, TYPES) == [
        "config.data: the generator writes [] that the line does not read and omits ['ranges_path'] that it requires"]
    line_only = {**REQUIRED, "config.model": REQUIRED["config.model"] | {"new_field"}}
    assert differences(cfg, line_only, TYPES)[0].startswith("config.model: the generator writes []")
    retyped = copy.deepcopy(cfg)
    retyped["data"]["smoke"] = "false"
    assert differences(retyped, REQUIRED, TYPES) == ["config.data.smoke: 'false' is not bool"]
    with_optional = copy.deepcopy(cfg)
    with_optional["data"] |= {"sim_image_aug": True, "action_units": "droid_normalized_joint_velocity_command"}
    assert differences(with_optional, REQUIRED, TYPES) == []
    with_optional["data"]["sim_image_aug"] = "yes"
    assert differences(with_optional, REQUIRED, TYPES) == ["config.data.sim_image_aug: 'yes' is not bool"]
