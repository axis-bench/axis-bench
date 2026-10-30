"""Training environments: which lock builds each one, how, and what a built one must report.

Training runs in a backbone's own environment, never in this package's interpreter: openpi for pi0.5
(Python 3.11, numpy 1.x, JAX) and LeRobot for SmolVLA and X-VLA (Python >= 3.12, numpy 2.x, torch).
The two cannot share one interpreter, and the reference servers' openpi-client pins numpy < 2 while
LeRobot needs numpy 2 -- so it is installed into the LeRobot environment without its dependencies.
The `offline` environment (signals, labels, data tooling) and the `eval` environment (simulation
evaluation: MuJoCo, the renderer, the policy-server client) are built from this checkout's own lock
files at the versions the public CI runs; the `droid_convert` environment (openpi's DROID converter,
for the real-world tutorial's demonstrations) from this checkout's own lock too. The render environment is the real-robot renderer's own
(axis_qvla/realrobot/render/render.lock.txt, its `render` extra). conf/envs/<environment>.toml pins the
environment's checkout's lock file by digest, lists the build commands (run in the checkout), the
versions the built environment must report and the modules that must import.

    qvla env build <name> [--print] [--rebuild]       build it (--print: print the build commands)
    qvla env check <name> [--python <venv python>] [--checkout <dir>]

`--checkout` names the environment's checkout when it is not at its submodule path (a development
worktree); the lock digest is checked there.

`check` runs `check_lock` (the checkout's lock is the pinned bytes), `check_built` (the Python
version and the gate imports) and `check_closure` (the built interpreter's entire
installed-distribution set against everything the lock resolves for its own Python and platform:
missing, extra and version-different packages, each reported). `[versions]` lists the versions that
matter for the docs; tests/unit/train/test_envs.py keeps it equal to the lock.

For the two `uv.lock` tiers, the closure is `uv export`'s own resolution (so "evaluating markers as
uv would" is literally uv doing it, not an approximation; dev dependencies included, since neither
tier's `uv sync` line passes `--no-dev`), narrowed to this tier's `--extra` flags (parsed from
`build`); the handful of remaining environment markers -- python_full_version, sys_platform,
platform_machine, implementation_name, platform_python_implementation, the only five that appear
in either lock once extras are resolved -- are evaluated here, by `eval_marker`, a small parser
over exactly that vocabulary that refuses -- EnvError, never a silent true/false -- any marker it
was not built to understand. Local or editable installs a tier's `build` adds outside the lock
(openpi-client, the renderer, this
checkout's own package via `-e .`) are exempt from the closure on both sides: nothing in the lock
ever resolves them, and the gate imports are what actually check they are there. Explicit
`name==version` pins a build adds after its main lock step (no `--no-deps`, e.g. the three
msgpack/websockets/dm-tree lines) override whatever the lock's own closure would otherwise expect.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[2]
ENVS_DIR = REPO / "conf" / "envs"
TIERS = ("openpi", "lerobot", "offline", "eval", "droid_convert", "render")
_KEYS = {"schema", "tier", "description", "checkout", "python", "lock", "lock_sha256", "build", "versions", "gate"}
# The names `qvla env build` takes: a backbone, or a tier no backbone names. SmolVLA and X-VLA train
# in one environment. `render` is the real-robot renderer's; `eval` is qvla.toml's `[venv] eval` (which
# a profile written before that name declares as `render`, read as `eval`: axis_qvla.core.hosts).
ENV_NAMES = {"pi05": "openpi", "smolvla": "lerobot", "xvla": "lerobot", "eval": "eval", "render": "render",
             "offline": "offline", "droid_convert": "droid_convert"}
BACKBONE_TIER = {"pi05-openpi": "openpi", "smolvla-lerobot": "lerobot", "xvla-lerobot": "lerobot"}
# Where each tier's build puts its interpreter, relative to this checkout: `uv sync` makes the
# checkout's .venv, and `uv venv <dir>` makes <dir> (tests/unit/train/test_env_cli.py checks both).
VENV_DIRS = {"openpi": "third_party/openpi/.venv", "lerobot": "third_party/lerobot/.venv",
             "eval": ".venv-eval", "offline": ".venv-offline", "droid_convert": ".venv-droid-convert",
             "render": ".venv-render"}
# A qvla.toml that keeps a tier's environment elsewhere names it here: (table, key).
PROFILE_VENVS = {"openpi": ("venv", "online"), "lerobot": ("roots", "smolvla_venv"),
                 "eval": ("venv", "eval"), "offline": ("venv", "offline")}
_PROBE = ("import importlib, importlib.metadata as m, json, platform, sys\n"
          "names, mods = json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
          "out = {'python': '.'.join(map(str, sys.version_info[:2])), 'versions': {}, 'imports': {},\n"
          "       'python_full_version': platform.python_version(), 'sys_platform': sys.platform,\n"
          "       'platform_machine': platform.machine(), 'implementation_name': sys.implementation.name,\n"
          "       'platform_python_implementation': platform.python_implementation(), 'distributions': {},\n"
          "       'commits': {}}\n"
          "for d in m.distributions():\n"
          "    try: out['distributions'][d.metadata['Name']] = d.version\n"
          "    except Exception: continue\n"
          "    try: out['commits'][d.metadata['Name']] = json.loads(d.read_text('direct_url.json'))['vcs_info']['commit_id']\n"
          "    except Exception: out['commits'][d.metadata['Name']] = None\n"
          "for n in names:\n"
          "    try: out['versions'][n] = m.version(n)\n"
          "    except m.PackageNotFoundError: out['versions'][n] = None\n"
          "for mod in mods:\n"
          "    try: importlib.import_module(mod); out['imports'][mod] = 'ok'\n"
          "    except Exception as e: out['imports'][mod] = f'{type(e).__name__}: {e}'\n"
          "print(json.dumps(out))\n")
# The only variables either lock's exported requirements use once extras are resolved (see the
# module docstring).
_MARKER_VARS = {"python_full_version", "sys_platform", "platform_machine", "implementation_name",
               "platform_python_implementation"}
_CMP_OPS = ("==", "!=", "<=", ">=", "<", ">")
_TOKEN_RE = re.compile(r"\s*(?:(\(|\))|(==|!=|<=|>=|<|>)|'([^']*)'|\"([^\"]*)\"|([A-Za-z_][A-Za-z0-9_]*))")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class EnvError(ValueError):
    """A tier's environment spec is malformed, or a built environment does not match it."""


@dataclasses.dataclass(frozen=True)
class EnvSpec:
    tier: str
    description: str
    checkout: str
    python: str
    lock: str
    lock_sha256: str
    build: tuple[str, ...]
    versions: dict[str, str]
    imports: tuple[str, ...]


def load_env(tier: str, *, root: pathlib.Path = ENVS_DIR) -> EnvSpec:
    if tier not in TIERS:
        raise EnvError(f"environment {tier!r} is not one of {TIERS}")
    path = root / f"{tier}.toml"
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EnvError(f"{path}: {exc}") from None
    if doc.get("schema") != "qvla.env/1" or doc.get("tier") != tier or set(doc) != _KEYS:
        raise EnvError(f"{path}: needs schema qvla.env/1, tier {tier!r} (the environment) and exactly the keys "
                       f"{sorted(_KEYS)}")
    if not isinstance(doc.get("lock_sha256"), str) or not _HEX64.fullmatch(doc["lock_sha256"]):
        raise EnvError(f"{path}: lock_sha256 must be 64 lowercase hex characters, got {doc.get('lock_sha256')!r}")
    versions = doc.get("versions")
    if not isinstance(versions, dict) or not all(isinstance(v, str) for v in versions.values()):
        raise EnvError(f"{path}: every [versions] value must be a string")
    imports = doc.get("gate", {}).get("imports") if isinstance(doc.get("gate"), dict) else None
    if not isinstance(imports, list) or not all(isinstance(m, str) for m in imports):
        raise EnvError(f"{path}: gate.imports must be a list of strings")
    return EnvSpec(tier=tier, description=doc["description"], checkout=doc["checkout"], python=doc["python"],
                   lock=doc["lock"], lock_sha256=doc["lock_sha256"], build=tuple(doc["build"]),
                   versions=dict(versions), imports=tuple(imports))


def tier_of(name: str) -> str:
    """The tier `qvla env build <name>` builds."""
    if name not in ENV_NAMES:
        raise EnvError(f"environment {name!r} is not one of {', '.join(ENV_NAMES)}")
    return ENV_NAMES[name]


def venv_dir(tier: str, profile=None) -> pathlib.Path:
    """The tier's environment: the one qvla.toml names (PROFILE_VENVS), else where its build puts it."""
    table, key = PROFILE_VENVS.get(tier, (None, None))
    got = None
    if profile is not None and table is not None:
        got = (profile.venvs if table == "venv" else profile.roots).get(key)
    return pathlib.Path(got) if got is not None else REPO / VENV_DIRS[tier]


def env_python(tier: str, profile=None) -> pathlib.Path:
    return venv_dir(tier, profile) / "bin" / "python"


def checkout(tier: str, profile=None) -> pathlib.Path:
    """The directory a tier's commands run in: its checkout (qvla.toml's `openpi` root for pi0.5)."""
    if tier == "openpi" and profile is not None and "openpi" in profile.roots:
        return pathlib.Path(profile.roots["openpi"])
    return REPO / load_env(tier).checkout


def build_hint(tier: str) -> str:
    return f"qvla env build {next(n for n, t in ENV_NAMES.items() if t == tier)}"


def missing_problem(tier: str, profile=None) -> str | None:
    """Why the tier's interpreter cannot run (not built, or unreadable), naming the command that builds it."""
    python = env_python(tier, profile)
    try:
        os.stat(python)
    except (FileNotFoundError, NotADirectoryError):
        return f"the {tier} environment is not built ({python} is not there): run `{build_hint(tier)}`"
    except OSError as exc:
        return f"cannot read {python} ({exc.__class__.__name__})"
    return None


def check_lock(spec: EnvSpec, checkout: pathlib.Path | None = None) -> list[str]:
    path = (checkout if checkout is not None else REPO / spec.checkout) / spec.lock
    try:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return [f"{path}: {exc.__class__.__name__} (is the {spec.checkout} submodule checked out?)"]
    return [] if got == spec.lock_sha256 else [f"{path} has sha256 {got}, the {spec.tier} environment pins {spec.lock_sha256}"]


def _norm(name: str) -> str:
    """A package name, insensitive to the `-`/`_`/`.`/case spelling PyPI, uv and
    importlib.metadata each pick independently (`PyYAML` vs `pyyaml`, `huggingface_hub` vs
    `huggingface-hub`)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _probe(python: pathlib.Path, names: list[str], mods: list[str]) -> dict:
    """Run `_PROBE` in `python`. Raises `OSError` if `python` cannot even be started (missing,
    not executable, a directory -- the caller's job is to turn that into a clean problem line,
    never a traceback); raises `EnvError` if it started but exited non-zero. It runs in an empty
    working directory: `python -c` puts the working directory first on sys.path, so a checkout root
    holding the `*.egg-info` an editable install writes would read as an installed distribution. A
    relative `python` (`.venv-eval/bin/python`) is taken relative to the caller's directory."""
    # absolute(), not resolve(): a venv's bin/python is a symlink, and following it would leave the venv.
    python = pathlib.Path(python).absolute()
    with tempfile.TemporaryDirectory(prefix="qvla-env-probe-") as cwd:
        r = subprocess.run([str(python), "-c", _PROBE, json.dumps(names), json.dumps(mods)],
                           capture_output=True, text=True, cwd=cwd)
    if r.returncode:
        raise EnvError(f"{python}: probe failed: {r.stderr.strip()[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def check_built(spec: EnvSpec, python: pathlib.Path) -> list[str]:
    """The built interpreter's Python version and the gate imports. Package versions are the
    closure's (`check_closure`); a `python` that cannot be started is one problem line."""
    try:
        got = _probe(python, [], list(spec.imports))
    except OSError as exc:
        return [f"{python}: {exc.__class__.__name__}: {exc} (is this a built interpreter?)"]
    except EnvError as exc:
        return [str(exc)]
    out = [] if got["python"] == spec.python else [f"python {got['python']}, the environment needs {spec.python}"]
    out += [f"import {m}: {s}" for m, s in got["imports"].items() if s != "ok"]
    return out


# --- the marker evaluator: python_full_version / sys_platform / platform_machine only ----------

def _tokenize_marker(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise EnvError(f"marker {text!r}: cannot tokenize {text[pos:]!r}")
        pos = m.end()
        if m.group(1):
            tokens.append(("PAREN", m.group(1)))
        elif m.group(2):
            tokens.append(("OP", m.group(2)))
        elif m.group(3) is not None:
            tokens.append(("STRING", m.group(3)))
        elif m.group(4) is not None:
            tokens.append(("STRING", m.group(4)))
        else:
            word = m.group(5)
            tokens.append(("KW", word) if word in ("and", "or", "in") else ("IDENT", word))
    return tokens


def _parse_marker_or(tokens: list[tuple[str, str]], i: int):
    node, i = _parse_marker_and(tokens, i)
    while i < len(tokens) and tokens[i] == ("KW", "or"):
        rhs, i = _parse_marker_and(tokens, i + 1)
        node = ("or", node, rhs)
    return node, i


def _parse_marker_and(tokens: list[tuple[str, str]], i: int):
    node, i = _parse_marker_atom(tokens, i)
    while i < len(tokens) and tokens[i] == ("KW", "and"):
        rhs, i = _parse_marker_atom(tokens, i + 1)
        node = ("and", node, rhs)
    return node, i


def _parse_marker_atom(tokens: list[tuple[str, str]], i: int):
    if i < len(tokens) and tokens[i] == ("PAREN", "("):
        node, i = _parse_marker_or(tokens, i + 1)
        if i >= len(tokens) or tokens[i] != ("PAREN", ")"):
            raise EnvError(f"marker: expected ')' in {tokens!r}")
        return node, i + 1
    return _parse_marker_comparison(tokens, i)


def _parse_marker_comparison(tokens: list[tuple[str, str]], i: int):
    if i >= len(tokens):
        raise EnvError("marker: unexpected end of expression")
    kind, value = tokens[i]
    if kind == "STRING":
        if i + 2 >= len(tokens) or tokens[i + 1] != ("KW", "in") or tokens[i + 2][0] != "IDENT":
            raise EnvError(f"marker: expected \"<string> in <var>\" at {tokens[i:]!r}")
        return ("in", value, tokens[i + 2][1]), i + 3
    if kind == "IDENT":
        if i + 2 >= len(tokens) or tokens[i + 1][0] != "OP" or tokens[i + 2][0] != "STRING":
            raise EnvError(f"marker: expected \"<var> <op> <string>\" at {tokens[i:]!r}")
        return ("cmp", value, tokens[i + 1][1], tokens[i + 2][1]), i + 3
    raise EnvError(f"marker: unrecognized token {tokens[i]!r}")


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in v.split("."))
    except ValueError:
        raise EnvError(f"marker: cannot compare python_full_version to {v!r}") from None


def _eval_marker_node(node, env: dict[str, str]) -> bool:
    kind = node[0]
    if kind == "and":
        return _eval_marker_node(node[1], env) and _eval_marker_node(node[2], env)
    if kind == "or":
        return _eval_marker_node(node[1], env) or _eval_marker_node(node[2], env)
    if kind == "in":
        _, literal, var = node
        if var not in _MARKER_VARS:
            raise EnvError(f"marker: unrecognized variable {var!r} (knows only {sorted(_MARKER_VARS)})")
        return literal in env[var]
    if kind == "cmp":
        _, var, op, val = node
        if var not in _MARKER_VARS:
            raise EnvError(f"marker: unrecognized variable {var!r} (knows only {sorted(_MARKER_VARS)})")
        if op not in _CMP_OPS:
            raise EnvError(f"marker: unrecognized operator {op!r}")
        a, b = (_version_tuple(env[var]), _version_tuple(val)) if var == "python_full_version" else (env[var], val)
        return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    raise EnvError(f"marker: unrecognized node {node!r}")


def eval_marker(text: str, env: dict[str, str]) -> bool:
    """A PEP 508 environment marker, restricted to the vocabulary our two locks' own exported,
    extras-resolved requirements use (see the module docstring and `_MARKER_VARS`),
    ==/!=/</<=/>/>=, string `in` containment, `and`/`or`, parentheses. `env` must carry every
    variable in `_MARKER_VARS`. Any other variable, operator or shape is refused with `EnvError`,
    never silently treated as true or false."""
    tokens = _tokenize_marker(text)
    node, i = _parse_marker_or(tokens, 0)
    if i != len(tokens):
        raise EnvError(f"marker: trailing tokens in {text!r}: {tokens[i:]!r}")
    return _eval_marker_node(node, env)


# --- the lock closure: what the lock resolves, for this --python's own interpreter and extras ---

def _extras_of(build: tuple[str, ...]) -> tuple[str, ...]:
    """The `--extra <name>` tokens of a build's `uv sync` line, in order."""
    extras: list[str] = []
    for line in build:
        tokens = line.split()
        extras += [tokens[i + 1] for i, t in enumerate(tokens) if t == "--extra" and i + 1 < len(tokens)]
    return tuple(extras)


def _pip_install_lines(build: tuple[str, ...]) -> list[list[str]]:
    """Tokenized `uv pip install ...` lines of a build (never the `uv sync`/`uv venv` ones)."""
    return [line.split() for line in build if "pip" in line.split() and "install" in line.split()]


def _no_deps_paths(build: tuple[str, ...]) -> list[str]:
    """Local/editable paths a `--no-deps` step installs: not in the lock, so the closure cannot
    and should not expect them -- the gate imports are what actually check they are there."""
    out: list[str] = []
    for tokens in _pip_install_lines(build):
        if "--no-deps" not in tokens:
            continue
        skip_value = False
        for tok in tokens:
            if skip_value:
                skip_value = False
                continue
            if tok == "--python":
                skip_value = True
            elif tok in ("uv", "pip", "install", "-e") or tok.startswith("-"):
                continue
            elif "==" not in tok:
                out.append(tok)
    return out


def _editable_path_counts(tokens: list[str]) -> tuple[int, int]:
    """(number of `-e` flags, number of local/path positional arguments) in one tokenized
    `pip`/`uv pip install` line. The two must be equal whenever any `-e` is present: uv (like
    pip) applies `-e` only to the single path immediately following it, so `-e . a b` (one
    flag, three paths) reads as if it covered all three but installs `a` and `b` as regular,
    non-editable wheels (a built venv's `direct_url.json` says which). A line with NO `-e` at all
    is a deliberate, fully non-editable install (the `lerobot` tier's own openpi-client line) and
    is not a problem."""
    n_e = sum(1 for t in tokens if t == "-e")
    n_paths = 0
    skip_value = False
    for tok in tokens:
        if skip_value:
            skip_value = False
            continue
        if tok == "--python":
            skip_value = True
        elif tok in ("uv", "pip", "install", "-e") or tok.startswith("-"):
            continue
        elif "==" not in tok:
            n_paths += 1
    return n_e, n_paths


def check_editable_flags(build: tuple[str, ...]) -> list[str]:
    """Every local path a build's `pip`/`uv pip install` lines name has its OWN `-e`, or none
    of them do. See `_editable_path_counts` for the exact failure this catches."""
    problems = []
    for tokens in _pip_install_lines(build):
        n_e, n_paths = _editable_path_counts(tokens)
        if n_e and n_e != n_paths:
            problems.append(f"{n_e} -e flag(s) for {n_paths} path(s): {' '.join(tokens)!r}")
    return problems


def _pip_overrides(build: tuple[str, ...]) -> dict[str, str]:
    """Explicit `name==version` pins a build installs with no `--no-deps` (e.g. the
    msgpack/websockets/dm-tree line): these replace whatever the lock's own closure expects."""
    out: dict[str, str] = {}
    for tokens in _pip_install_lines(build):
        if "--no-deps" in tokens:
            continue
        for tok in tokens:
            if "==" in tok and not tok.startswith("-"):
                name, _, version = tok.partition("==")
                out[_norm(name)] = version
    return out


def _guess_package_name(path: str, checkout: pathlib.Path) -> str:
    """The distribution name a local/editable `path` (relative to `checkout`) installs as: read
    from its own pyproject.toml when reachable from here (not always -- a submodule path is empty
    until the submodule is checked out), else the path's own last component,
    which is how every name this module ever needs to guess (openpi-client, axis-render) actually
    spells its package name."""
    target = checkout if path == "." else checkout / path
    pyproject = target / "pyproject.toml"
    if not pyproject.is_file() and path == ".":
        pyproject = REPO / "conf" / "release" / "public" / "pyproject.toml.tmpl"
    if pyproject.is_file():
        try:
            return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["name"]
        except (tomllib.TOMLDecodeError, KeyError):
            pass
    return pathlib.PurePosixPath(checkout).name if path == "." else pathlib.PurePosixPath(path).name


def _exempt_names(build: tuple[str, ...], checkout: pathlib.Path) -> set[str]:
    return {_norm(_guess_package_name(p, checkout)) for p in _no_deps_paths(build)}


def _lock_packages(lock_path: pathlib.Path) -> tuple[dict[str, str], dict[str, str]]:
    """({normalized name: version}, {normalized editable path: normalized name}) of every
    `[[package]]` a uv.lock declares. A package's `name`/`version` fields are always authoritative,
    unlike the text `uv export` prints for a git/URL or editable requirement, which carries no
    clean version string of its own -- so every name this reads comes from here, never from
    export's own text, and an editable requirement's PATH (the only thing export prints for one)
    is resolved to a name through the second map."""
    doc = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    versions, editable_paths = {}, {}
    for p in doc.get("package", ()):
        norm = _norm(p["name"])
        versions[norm] = p["version"]
        editable = p.get("source", {}).get("editable")
        if editable is not None:
            editable_paths[str(pathlib.PurePosixPath(editable))] = norm
    return versions, editable_paths


def _closure_uv_lock(spec: EnvSpec, checkout: pathlib.Path, python: pathlib.Path, env: dict[str, str]
                     ) -> dict[str, str]:
    """{normalized name: version} a uv.lock tier's own `uv export` resolves for this tier's
    `--extra` flags (dev dependencies included: neither tier's `build` passes `--no-dev` to
    `uv sync`, so this does not pass it to `uv export` either), with the remaining environment
    markers evaluated by `eval_marker`."""
    cmd = ["uv", "export", "--frozen", "--format", "requirements-txt", "--no-hashes",
           "--no-annotate", "--no-header", "--python", str(python)]
    for extra in _extras_of(spec.build):
        cmd += ["--extra", extra]
    try:
        r = subprocess.run(cmd, cwd=checkout, capture_output=True, text=True)
    except OSError as exc:
        raise EnvError(f"`uv export` in {checkout}: {exc.__class__.__name__}: {exc}") from None
    if r.returncode:
        raise EnvError(f"`uv export` in {checkout} failed: {r.stderr.strip()[-500:]}")
    versions, editable_paths = _lock_packages(checkout / spec.lock)
    out: dict[str, str] = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pinned, _, marker = line.partition(" ; ")
        if marker and not eval_marker(marker, env):
            continue
        if pinned.startswith("-e "):
            # No parseable version in this text at all: the lock's own editable entry is the
            # only place one exists.
            key = str(pathlib.PurePosixPath(pinned[3:].strip()))
            if key not in editable_paths:
                raise EnvError(f"{checkout}: `uv export` names editable {pinned[3:]!r}, not in {spec.lock}")
            norm = editable_paths[key]
            out[norm] = versions[norm]
        elif " @ " in pinned:
            # A git/URL requirement: same, no version in this text, so the lock's own field is
            # the only one there is.
            norm = _norm(pinned.split(" @ ", 1)[0].strip())
            if norm not in versions:
                raise EnvError(f"{checkout}: `uv export` names {pinned!r}, not in {spec.lock}'s own packages")
            out[norm] = versions[norm]
        else:
            # A plain `name==version` line: this fork's own version, never the lock's single
            # `[[package]].version` field, which does not disambiguate when uv.lock carries more
            # than one version of the same name across resolution forks (dm-tree, the nvidia-*-cu12
            # family) -- the marker already picked the one fork this line belongs to.
            name, _, version = pinned.partition("==")
            norm = _norm(name.strip())
            if norm not in versions:
                raise EnvError(f"{checkout}: `uv export` names {name.strip()!r}, not in {spec.lock}'s own packages")
            out[norm] = version
    return out


@dataclasses.dataclass(frozen=True)
class UrlPin:
    """A lock line `name @ git+<repository>@<commit>`: it pins a commit, which an installed
    distribution records in its PEP 610 direct_url.json (`vcs_info.commit_id`), not a version."""
    url: str
    commit: str


def _closure_lock_txt(path: pathlib.Path) -> dict[str, str | UrlPin]:
    """{normalized name: version, or the UrlPin of a direct-URL line} of every pinned line of a plain
    `.lock.txt` tier."""
    out: dict[str, str | UrlPin] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, _, version = line.partition("==")
            out[_norm(name.strip())] = version.strip()
        elif " @ " in line:
            name, _, url = line.partition(" @ ")
            out[_norm(name)] = UrlPin(url.strip(), url.strip().rsplit("@", 1)[-1])
    return out


def closure_diff(expected: dict[str, str | UrlPin], got: dict[str, str | None],
                 exempt: frozenset[str] = frozenset()) -> list[str]:
    """missing/extra/version-different between a lock's resolved closure and a built environment's
    full installed-distribution set, each reported separately; `exempt` names are dropped from
    both sides first (local/editable installs the lock never resolves). An extra package always
    fails: a lock-built environment is not supposed to carry anything the lock did not put there. For a
    UrlPin, `got` holds the installed distribution's recorded commit (None when it has no VCS
    direct_url.json), and only the pinned commit matches."""
    expected = {n: v for n, v in expected.items() if n not in exempt}
    got = {n: v for n, v in got.items() if n not in exempt}
    def pin(n: str) -> str:
        e = expected[n]
        return f"{n} @ {e.url}" if isinstance(e, UrlPin) else f"{n}=={e}"

    out = [f"missing {pin(n)}" for n in sorted(set(expected) - set(got))]
    out += [f"extra {n}=={got[n]}" for n in sorted(set(got) - set(expected))]
    for n in sorted(set(expected) & set(got)):
        e = expected[n]
        if isinstance(e, UrlPin) and got[n] != e.commit:
            installed = f"commit {got[n]}" if got[n] else "no VCS commit (no direct_url.json vcs_info)"
            out.append(f"{n} is installed from {installed}, the lock pins commit {e.commit} ({e.url})")
        elif not isinstance(e, UrlPin) and got[n] != e:
            out.append(f"{n} {got[n]}, the lock resolves {e}")
    return out


def check_closure(spec: EnvSpec, python: pathlib.Path, checkout: pathlib.Path | None = None) -> list[str]:
    """The built `--python`'s complete installed-distribution set against everything `spec`'s lock
    resolves for its own interpreter and platform -- not a sample. A `python` that cannot be
    started at all is a single clean problem line, never a traceback; a `uv export`/lock-parsing
    failure (a malformed lock, an unresolvable editable, an unrecognized marker form) is reported
    the same way, rather than raising past this function."""
    root = checkout if checkout is not None else REPO / spec.checkout
    try:
        got = _probe(python, [], [])
    except OSError as exc:
        return [f"{python}: {exc.__class__.__name__}: {exc} (is this a built interpreter?)"]
    except EnvError as exc:
        return [str(exc)]
    env = {k: got[k] for k in _MARKER_VARS}
    try:
        if pathlib.Path(spec.lock).name == "uv.lock":
            expected = _closure_uv_lock(spec, root, python, env)
        else:
            expected = _closure_lock_txt(root / spec.lock)
    except (EnvError, OSError) as exc:
        return [str(exc)]
    expected.update(_pip_overrides(spec.build))
    installed: dict[str, str | None] = {_norm(n): v for n, v in got["distributions"].items()}
    commits = {_norm(n): c for n, c in got.get("commits", {}).items()}
    installed.update({n: commits.get(n) for n, e in expected.items() if isinstance(e, UrlPin) and n in installed})
    return closure_diff(expected, installed, frozenset(_exempt_names(spec.build, root)))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="qvla train env", description="training environments")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="print the commands that build an environment")
    pl.add_argument("tier", choices=TIERS, metavar="environment", help=f"one of {', '.join(TIERS)}")
    ck = sub.add_parser("check", help="check a built environment against its pins")
    ck.add_argument("tier", choices=TIERS, metavar="environment", help=f"one of {', '.join(TIERS)}")
    ck.add_argument("--python", type=pathlib.Path, required=True)
    ck.add_argument("--checkout", type=pathlib.Path, default=None)
    a = p.parse_args(argv)
    try:
        spec = load_env(a.tier)
    except EnvError as exc:
        print(f"FAIL  {exc}")
        return 2
    if a.cmd == "plan":
        print(f"# {spec.description}")
        print(f"cd {spec.checkout}")
        for cmd in spec.build:
            print(cmd)
        return 0
    problems = check_lock(spec, a.checkout) + check_built(spec, a.python) + check_closure(spec, a.python, a.checkout)
    for x in problems:
        print(f"DIFF  {x}")
    print(f"{'FAIL' if problems else 'OK'}    {spec.tier} environment at {a.python}")
    return 1 if problems else 0
