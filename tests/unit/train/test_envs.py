"""The tier environments: pinned locks, the numpy split, and a probe that reports a built venv."""
from __future__ import annotations

import pathlib
import sys
import textwrap

import pytest

from axis_qvla.train.envs import (TIERS, EnvError, EnvSpec, UrlPin, check_built, check_closure, check_lock,
                             closure_diff, eval_marker, load_env, main)
from axis_qvla.train.envs import (_closure_lock_txt, _closure_uv_lock, _extras_of, _no_deps_paths, _pip_overrides,
                             check_editable_flags)


def _shipped(name: str) -> pathlib.Path:
    """`name` at the checkout root (`.github/ci-constraints.txt`, `pyproject.toml`), else among the
    release overrides or templates, where a checkout may keep it."""
    from axis_qvla.train.envs import REPO

    public = REPO.joinpath("conf", "release", "public")
    for candidate in (REPO / name, public / "overrides" / name, public / f"{name}.tmpl"):
        if candidate.is_file():
            return candidate
    pytest.fail(f"{name} is neither at the root of {REPO} nor among its release's public files", pytrace=False)


def test_the_two_tiers_are_declared():
    openpi, lerobot = load_env("openpi"), load_env("lerobot")
    assert openpi.python == "3.11" and openpi.versions["numpy"].startswith("1.")
    assert lerobot.python == "3.12" and lerobot.versions["numpy"].startswith("2.")
    assert openpi.versions["lerobot"] == "0.4.4" and lerobot.versions["lerobot"] == "0.6.2"
    client = [c for c in lerobot.build if "openpi-client" in c]
    assert len(client) == 1 and "--no-deps" in client[0]
    assert all(len(s.lock_sha256) == 64 for s in (openpi, lerobot))


# axis-render's own declared runtime dependencies (conf/envs/eval.in): installed --no-deps by
# both the eval tier's build and the CI test job, so nothing in ci-constraints.txt ever resolves
# them -- they are pinned directly in eval.in/eval.lock.txt instead, at the versions the paper's
# evaluations ran in, not at whatever ci-constraints.txt would otherwise pick.
RENDERER_OWN_DEPS = frozenset({"robosuite", "dm-control", "imageio", "imageio-ffmpeg"})


def _ci_pins() -> dict[str, str]:
    import re

    ci_constraints = _shipped(".github/ci-constraints.txt")
    pins: dict[str, str] = {}
    for line in ci_constraints.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, _, ver = line.partition("==")
            pins[re.sub(r"[-_.]+", "-", name).lower()] = ver
    return pins


def test_the_offline_tier_is_this_checkouts_lock_at_the_ci_versions():
    from axis_qvla.train.envs import REPO

    pins = _ci_pins()
    spec = load_env("offline")
    assert spec.checkout == "." and check_lock(spec) == []
    locked = dict(line.split("==") for line in (REPO / spec.lock).read_text().split() if "==" in line)
    assert all(pins.get(n) == v for n, v in locked.items())       # every pin is the CI's own version
    assert all(locked[n] == v for n, v in spec.versions.items())
    assert "pillow" in locked


def test_the_eval_tier_is_the_ci_test_jobs_closure_plus_the_renderers_own_pins():
    """eval = everything the public CI test job's own closure resolves (`.[offline,test]` plus
    openpi-client and the asset store's httpx/pydantic), at the CI's exact versions, PLUS
    axis-render's four declared runtime dependencies (RENDERER_OWN_DEPS) and whatever THEY pull in
    that ci-constraints.txt never had to resolve -- the renderer is installed --no-deps, same as
    the CI test job, so none of that closure is in the constraints file. Every package this lock
    shares with ci-constraints.txt must still equal the CI's own pin: the shared base never
    silently drifts just because the renderer joined the tier."""
    from axis_qvla.train.envs import REPO

    pins = _ci_pins()
    spec = load_env("eval")
    assert spec.checkout == "." and check_lock(spec) == []
    locked = dict(line.split("==") for line in (REPO / spec.lock).read_text().split() if "==" in line)
    assert all(pins[n] == v for n, v in locked.items() if n in pins)   # shared base, never drifted
    assert all(locked[n] == v for n, v in spec.versions.items())        # the sampled table matches
    assert RENDERER_OWN_DEPS <= set(locked)                             # the renderer's own deps made it in
    renderer_only = set(locked) - set(pins)
    assert not RENDERER_OWN_DEPS & set(pins)                            # the CI job never pinned them
    assert RENDERER_OWN_DEPS <= renderer_only


def test_unknown_tiers_and_keys_are_refused(tmp_path):
    with pytest.raises(EnvError, match="not one of"):
        load_env("blender")
    (tmp_path / "openpi.toml").write_text('schema = "qvla.env/1"\ntier = "openpi"\n')
    with pytest.raises(EnvError, match="exactly the keys"):
        load_env("openpi", root=tmp_path)


def _minimal_toml(**overrides) -> str:
    fields = dict(schema='"qvla.env/1"', tier='"openpi"', description='""', checkout='"co"',
                  python='"3.11"', lock='"uv.lock"', lock_sha256='"' + "0" * 64 + '"', build="[]")
    fields.update(overrides)
    body = "\n".join(f"{k} = {v}" for k, v in fields.items())
    return f"{body}\n[versions]\nnumpy = \"1.0.0\"\n[gate]\nimports = [\"json\"]\n"


def test_load_env_refuses_a_malformed_lock_sha256(tmp_path):
    """A tier file's values are type-checked, not just its key set."""
    (tmp_path / "openpi.toml").write_text(_minimal_toml(lock_sha256='"not-hex"'))
    with pytest.raises(EnvError, match="lock_sha256"):
        load_env("openpi", root=tmp_path)


def test_load_env_refuses_a_non_string_versions_value(tmp_path):
    toml = _minimal_toml().replace('numpy = "1.0.0"', "numpy = 126")
    (tmp_path / "openpi.toml").write_text(toml)
    with pytest.raises(EnvError, match="versions"):
        load_env("openpi", root=tmp_path)


def test_load_env_refuses_a_non_list_gate_imports(tmp_path):
    toml = _minimal_toml().replace('imports = ["json"]', 'imports = "json"')
    (tmp_path / "openpi.toml").write_text(toml)
    with pytest.raises(EnvError, match="gate.imports"):
        load_env("openpi", root=tmp_path)


def _spec(**kw) -> EnvSpec:
    base = dict(tier="openpi", description="", checkout="co", python=".".join(map(str, sys.version_info[:2])),
                lock="uv.lock", lock_sha256="0" * 64, build=(), versions={}, imports=("json",))
    base.update(kw)
    return EnvSpec(**base)


def test_the_lock_is_checked_by_digest(tmp_path):
    (tmp_path / "co").mkdir()
    (tmp_path / "co" / "uv.lock").write_bytes(b"lock")
    import hashlib

    assert check_lock(_spec(lock_sha256=hashlib.sha256(b"lock").hexdigest()), tmp_path / "co") == []
    assert "pins" in check_lock(_spec(), tmp_path / "co")[0]
    assert "submodule" in check_lock(_spec(checkout="absent"))[0]


def test_a_built_interpreter_is_probed():
    assert check_built(_spec(), sys.executable) == []
    problems = check_built(_spec(python="2.7", imports=("no_such_module",)), sys.executable)
    assert len(problems) == 2


def test_check_built_reads_the_python_version_and_the_gate_imports_only():
    spec = _spec(versions={"pytest": "0.0"})     # a [versions] row the environment does not have
    assert check_built(spec, sys.executable) == []


def test_a_bad_python_path_is_a_clean_problem_never_a_traceback(tmp_path):
    """A missing or non-executable --python must not crash check_built."""
    missing = tmp_path / "no-such-python"
    problems = check_built(_spec(), missing)
    assert len(problems) == 1 and str(missing) in problems[0] and "FileNotFoundError" in problems[0]

    not_executable = tmp_path / "not-a-python"
    not_executable.write_text("not a script")
    problems = check_built(_spec(), not_executable)
    assert len(problems) == 1 and str(not_executable) in problems[0]


def test_qvla_train_env_check_is_clean_end_to_end_on_a_bad_python(capsys):
    """`qvla train env check ... --python /no/such/python`: a FAIL line and exit 1, never a raised
    exception reaching main()."""
    assert main(["check", "openpi", "--python", "/no/such/python"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "Traceback" not in out


def test_the_extras_carry_pillow_and_the_render_lock():
    """The public pyproject's extras: Pillow in `offline` (the real-robot data tools decode PNG),
    and `render` pinned exactly as the renderer's own lock."""
    import tomllib

    from axis_qvla.train.envs import REPO

    pyproject = _shipped("pyproject.toml")
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    assert "pillow>=12.2.0" in extras["offline"]
    assert sorted(extras["offline"]) == sorted((REPO / "conf/envs/offline.in").read_text().split())
    lock = [ln for ln in (REPO / "axis_qvla/realrobot/render/render.lock.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert sorted(extras["render"]) == sorted(ln for ln in lock if not ln.startswith("pytest=="))


def test_the_eval_input_carries_the_offline_and_test_extras():
    """conf/envs/eval.in is the environment the CI test job builds with `.[offline,test]`, so every
    requirement of either extra is one of its lines: a package added to the `offline` extra (SciPy,
    for the label pipeline) reaches the eval lock too, instead of only the CI's."""
    import tomllib

    from axis_qvla.train.envs import REPO

    extras = tomllib.loads(_shipped("pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert set(extras["offline"]) | set(extras["test"]) <= set((REPO / "conf/envs/eval.in").read_text().split())


def test_the_offline_and_eval_tiers_declare_the_same_numpy(tmp_path):
    """The two tiers' numpy pin is one declared value, not two hand-typed copies that could drift
    apart: both trace to the same ci-constraints.txt line
    (test_the_offline_and_eval_tiers_are_this_checkouts_locks_at_the_ci_versions already proves
    each equals it separately); this asserts the two tiers can never disagree with each other even
    if that transitivity were ever broken."""
    assert load_env("offline").versions["numpy"] == load_env("eval").versions["numpy"]


def test_the_tier_versions_table_agrees_with_the_lock():
    """[versions] is documentation, not the source `check_closure` reads -- this proves it never
    silently drifts from what the lock (or, for offline/eval, the CI-compiled .lock.txt test above
    already covers) actually resolves. Runs over whichever of the two
    uv.lock tiers' submodules this checkout actually has -- asserting at least one is present keeps
    the loop from silently covering nothing."""
    from axis_qvla.train.envs import REPO

    env = {"python_full_version": "3.99.99", "sys_platform": sys.platform, "platform_machine": "x86_64",
          "implementation_name": "cpython", "platform_python_implementation": "CPython"}
    # python_full_version is set absurdly high so every <3.x/>=3.x fork used by the handful of
    # [versions] names we actually check resolves to its "current" branch; none of
    # numpy/jax/torch/lerobot/datasets/transformers/accelerate/draccus is itself forked by
    # python_full_version in either lock (torch is forked by sys_platform instead, set correctly
    # above), so this does not change their resolved versions -- it only avoids this test needing
    # a second, real built interpreter of its own.
    specs = [load_env(t) for t in ("openpi", "lerobot")]
    present = [s for s in specs if (REPO / s.checkout).is_dir()]
    assert present, "neither third_party/openpi nor third_party/lerobot is checked out here"
    for spec in present:
        closure = _closure_uv_lock(spec, REPO / spec.checkout, pathlib.Path(sys.executable), env)
        for name, version in spec.versions.items():
            assert closure.get(name.replace("_", "-").lower()) == version, (spec.tier, name)


def _env(**kw) -> dict[str, str]:
    base = {"python_full_version": "3.11.14", "sys_platform": "linux", "platform_machine": "x86_64",
           "implementation_name": "cpython", "platform_python_implementation": "CPython"}
    base.update(kw)
    return base


def test_eval_marker_handles_the_forms_our_locks_use():
    env = _env()
    assert eval_marker("python_full_version < '3.13'", env) is True
    assert eval_marker("python_full_version >= '3.13'", env) is False
    assert eval_marker("sys_platform == 'linux'", env) is True
    assert eval_marker("sys_platform != 'linux'", env) is False
    assert eval_marker("'linux' in sys_platform", env) is True
    assert eval_marker("implementation_name == 'pypy'", env) is False
    assert eval_marker("platform_python_implementation != 'PyPy' and sys_platform == 'win32'", env) is False
    assert eval_marker("(platform_machine == 'aarch64' and sys_platform == 'linux') or sys_platform == 'darwin'",
                       env) is False
    assert eval_marker("(platform_machine != 'aarch64' and sys_platform == 'linux') or "
                       "(sys_platform != 'darwin' and sys_platform != 'linux')", env) is True


def test_eval_marker_refuses_what_it_does_not_recognize():
    env = _env()
    with pytest.raises(EnvError, match="unrecognized variable"):
        eval_marker("os_name == 'posix'", env)
    with pytest.raises(EnvError):
        eval_marker("not sys_platform == 'linux'", env)       # negation: not a form either lock uses
    with pytest.raises(EnvError):
        eval_marker("python_full_version", env)                # not a comparison at all


def test_closure_diff_reports_each_direction_separately():
    expected = {"a": "1.0", "b": "2.0", "c": "3.0"}
    got = {"a": "1.0", "b": "9.9", "d": "4.0"}
    out = closure_diff(expected, got)
    assert "missing c==3.0" in out
    assert "extra d==4.0" in out
    assert any(x.startswith("b 9.9, the lock resolves 2.0") for x in out)
    assert len(out) == 3
    assert closure_diff(expected, dict(expected)) == []


def test_closure_diff_exempts_named_packages_on_both_sides():
    expected = {"a": "1.0", "client": "0.1.0"}
    got = {"a": "1.0", "client": "9.9.9"}
    assert closure_diff(expected, got, frozenset({"client"})) == []
    assert closure_diff({"a": "1.0"}, {"a": "1.0", "client": "9.9.9"}, frozenset({"client"})) == []


def test_build_line_parsing_matches_the_lerobot_tier():
    lerobot = load_env("lerobot")
    assert _extras_of(lerobot.build) == ("training", "smolvla", "xvla", "test")
    assert _no_deps_paths(lerobot.build) == ["../openpi/packages/openpi-client"]
    assert _pip_overrides(lerobot.build) == {"msgpack": "1.1.0", "websockets": "15.0.1", "dm-tree": "0.1.9",
                                             "absl-py": "2.5.1", "wrapt": "2.5.1"}   # dm-tree's own two
    assert _extras_of(load_env("openpi").build) == ()
    assert _no_deps_paths(load_env("openpi").build) == []
    assert _pip_overrides(load_env("openpi").build) == {}


def test_a_shared_e_flag_across_paths_is_caught_and_one_per_path_passes():
    """`-e . a b` applies `-e` to `.` only -- uv (like pip) treats `a` and `b` as regular,
    non-editable installs even though the flag reads as if it covered all three. The check
    catches that shape from the config text alone."""
    shared = ("uv pip install --python .venv/bin/python --no-deps -e . a b",)
    problems = check_editable_flags(shared)
    assert len(problems) == 1 and "1 -e flag(s) for 3 path(s)" in problems[0]

    one_each = ("uv pip install --python .venv/bin/python --no-deps -e . -e a -e b",)
    assert check_editable_flags(one_each) == []

    # No -e at all is a deliberate, fully non-editable install (the lerobot tier's own
    # openpi-client line) -- not a problem regardless of how many paths it names.
    none = ("uv pip install --python .venv/bin/python --no-deps a b",)
    assert check_editable_flags(none) == []

    # A line with no pip/uv install at all (uv venv, uv sync) is not scanned.
    assert check_editable_flags(("uv venv --python 3.11 .venv",)) == []


def test_no_shipped_tier_shares_one_e_flag_across_paths():
    """Sweeps every tier's build for the same mistake, not just eval's: a config-only regression
    here would otherwise ship silently until someone next builds the tier by hand."""
    for tier in TIERS:
        assert check_editable_flags(load_env(tier).build) == [], tier


_MINI_PYPROJECT = (
    '[project]\nname = "mini"\nversion = "0.1.0"\nrequires-python = ">=3.11"\ndependencies = ["six"]\n\n'
    '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n')
_MINI_LOCK = textwrap.dedent('''\
    version = 1
    revision = 3
    requires-python = ">=3.11"

    [manifest]
    members = ["mini"]

    [[package]]
    name = "mini"
    version = "0.1.0"
    source = { editable = "." }
    dependencies = [
        { name = "six" },
        { name = "dummy-a", marker = "python_full_version >= '3.13'" },
        { name = "dummy-b", marker = "python_full_version < '3.13'" },
    ]

    [[package]]
    name = "six"
    version = "1.16.0"
    source = { registry = "https://pypi.org/simple" }

    [[package]]
    name = "dummy-a"
    version = "9.9.9"
    source = { registry = "https://pypi.org/simple" }

    [[package]]
    name = "dummy-b"
    version = "8.8.8"
    source = { registry = "https://pypi.org/simple" }
    ''')


def test_closure_uv_lock_resolves_markers_and_the_editable_project(tmp_path):
    """The closure comes from `uv.lock` itself (via `uv export`, offline -- `--frozen` never
    touches the network), not from a hand-typed sample -- including the editable root project
    (whose export line has no version text of its own) and a python_full_version fork
    (dummy-a/dummy-b), exactly the dm-tree/nvidia-*-cu12 shape either real lock uses."""
    (tmp_path / "pyproject.toml").write_text(_MINI_PYPROJECT)
    (tmp_path / "uv.lock").write_text(_MINI_LOCK)
    spec = _spec(checkout=str(tmp_path), lock="uv.lock", build=())
    env = _env(python_full_version="3.11.14")
    closure = _closure_uv_lock(spec, tmp_path, pathlib.Path(sys.executable), env)
    assert closure == {"mini": "0.1.0", "six": "1.16.0", "dummy-b": "8.8.8"}

    closure_new = _closure_uv_lock(spec, tmp_path, pathlib.Path(sys.executable), env | {"python_full_version": "3.13.0"})
    assert closure_new == {"mini": "0.1.0", "six": "1.16.0", "dummy-a": "9.9.9"}


def test_check_closure_catches_a_planted_missing_extra_and_wrong_version(tmp_path, monkeypatch):
    """End to end through check_closure: a hand-built venv whose Python version and gate imports
    pass must still fail here if it carries the wrong dependency set."""
    (tmp_path / "pyproject.toml").write_text(_MINI_PYPROJECT)
    (tmp_path / "uv.lock").write_text(_MINI_LOCK)
    spec = _spec(checkout=str(tmp_path), lock="uv.lock", build=(), python=".".join(map(str, sys.version_info[:2])))

    import axis_qvla.train.envs as envs_mod

    def fake_probe(python, names, mods):
        return {"python": spec.python, "versions": {}, "imports": {},
               "python_full_version": "3.11.14", "sys_platform": "linux", "platform_machine": "x86_64",
               "implementation_name": "cpython", "platform_python_implementation": "CPython",
               # six is missing, dummy-b is at the wrong version, "stray" is extra; mini (the
               # editable project, exempt by nothing here -- it is not --no-deps'd) must also match
               # or it would show up as "missing" too.
               "distributions": {"mini": "0.1.0", "dummy-b": "7.0.0", "stray": "1.0.0"}}

    monkeypatch.setattr(envs_mod, "_probe", fake_probe)
    problems = check_closure(spec, pathlib.Path(sys.executable), tmp_path)
    assert "missing six==1.16.0" in problems
    assert "extra stray==1.0.0" in problems
    assert any(p.startswith("dummy-b 7.0.0, the lock resolves 8.8.8") for p in problems)
    assert len(problems) == 3


def test_the_droid_convert_tier_is_its_compiled_lock_with_one_cv2():
    """openpi's DROID converter's environment: every package pinned, the .in's own pins kept, LeRobot at
    the converter's commit, and OpenCV only as the headless build LeRobot asks for, so one `cv2` decodes
    the recordings."""
    from axis_qvla.train.envs import REPO

    spec = load_env("droid_convert")
    assert spec.checkout == "." and check_lock(spec) == []
    lock = (REPO / spec.lock).read_text().splitlines()
    locked = dict(line.split("==") for line in lock if "==" in line)
    wanted = [line for line in (REPO / "conf/envs/droid_convert.in").read_text().splitlines() if line.strip()]
    pins = (w.split("==") for w in wanted if "==" in w)
    assert all(locked[n] == v or locked[n].startswith(v + "+") for n, v in pins)     # torch 2.7.1 -> 2.7.1+cpu
    assert [w for w in wanted if " @ " in w] == [line for line in lock if " @ " in line] == [
        "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"]
    assert "opencv-python-headless" in locked and "opencv-python" not in locked
    assert all(locked.get(n) == v or n == "lerobot" for n, v in spec.versions.items())


def _url_pinned(tmp_path, monkeypatch, commit):
    """check_closure of a .lock.txt tier pinning `b` by URL, against a built b recording `commit`."""
    import axis_qvla.train.envs as envs_mod

    (tmp_path / "x.lock.txt").write_text("a==1.0\nb @ git+https://example.org/b@0123abc\n")
    spec = _spec(lock="x.lock.txt", checkout=".")
    monkeypatch.setattr(envs_mod, "_probe", lambda python, names, mods: {
        "python_full_version": "3.11.14", "sys_platform": "linux", "platform_machine": "x86_64",
        "implementation_name": "cpython", "platform_python_implementation": "CPython",
        "distributions": {"a": "1.0", "b": "0.1.0"}, "commits": {"a": None, "b": commit}})
    return check_closure(spec, pathlib.Path(sys.executable), tmp_path)


def test_a_url_pin_matches_only_its_commit(tmp_path, monkeypatch):
    assert _url_pinned(tmp_path, monkeypatch, "0123abc") == []
    assert _closure_lock_txt(tmp_path / "x.lock.txt")["b"] == UrlPin("git+https://example.org/b@0123abc", "0123abc")
    assert _url_pinned(tmp_path, monkeypatch, "fedcba9") == [
        "b is installed from commit fedcba9, the lock pins commit 0123abc (git+https://example.org/b@0123abc)"]


def test_a_url_pin_without_a_recorded_commit_fails(tmp_path, monkeypatch):
    assert _url_pinned(tmp_path, monkeypatch, None) == [
        "b is installed from no VCS commit (no direct_url.json vcs_info), the lock pins commit 0123abc "
        "(git+https://example.org/b@0123abc)"]


def test_a_missing_url_pin_names_its_url(tmp_path):
    (tmp_path / "x.lock.txt").write_text("b @ git+https://example.org/b@0123abc\n")
    assert closure_diff(_closure_lock_txt(tmp_path / "x.lock.txt"), {}) == [
        "missing b @ git+https://example.org/b@0123abc"]


def test_the_probe_ignores_an_egg_info_in_the_callers_working_directory(tmp_path, monkeypatch):
    """`pip install -e .` leaves `<name>.egg-info` in the checkout root; a probe that ran `python -c`
    there would list it as an installed distribution (`-c` puts the working directory on sys.path)
    and report a correct environment as carrying an extra package."""
    import json
    import subprocess

    from axis_qvla.train import envs

    egg = tmp_path / "stray_probe_pkg.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: stray-probe-pkg\nVersion: 9.9.9\n")
    monkeypatch.chdir(tmp_path)
    plain = subprocess.run([sys.executable, "-c", envs._PROBE, "[]", "[]"], capture_output=True, text=True, check=True)
    assert "stray-probe-pkg" in json.loads(plain.stdout.splitlines()[-1])["distributions"]   # the trap is real
    assert "stray-probe-pkg" not in envs._probe(pathlib.Path(sys.executable), [], [])["distributions"]
    assert check_built(_spec(), sys.executable) == []


def test_a_relative_python_is_the_callers_and_keeps_its_venv(tmp_path, monkeypatch):
    """`--python .venv-eval/bin/python` names a path under the caller's directory; the probe runs
    elsewhere, so the path is made absolute first, without following the venv's symlink."""
    import subprocess

    from axis_qvla.train import envs

    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp_path / ".venv-x")], check=True)
    monkeypatch.chdir(tmp_path)
    got = envs._probe(pathlib.Path(".venv-x/bin/python"), [], [])
    assert got["python"] == ".".join(map(str, sys.version_info[:2]))
    assert check_built(_spec(), pathlib.Path(".venv-x/bin/python")) == []
    prefix = subprocess.run([".venv-x/bin/python", "-c", "import sys; print(sys.prefix)"], capture_output=True,
                            text=True, check=True).stdout.strip()
    assert pathlib.Path(prefix).resolve() == (tmp_path / ".venv-x").resolve()   # the venv, not the base interpreter
    assert pathlib.Path(".venv-x/bin/python").absolute() == tmp_path / ".venv-x" / "bin" / "python"
