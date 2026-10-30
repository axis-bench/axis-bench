"""Compose the campaign driver's argv and environment from a benchmark and a host.

A benchmark user has a checkpoint and a config name. Everything else the driver needs is a
fact about the benchmark declaration (task list, seeds, layouts, conventions, XLA pin) or
about the machine (campaign root, rollout-log root, the two interpreters, the render package),
and both are already written down -- in `conf/benchmarks/axis-heldout-v1/<name>.json` and
the host profile (`qvla.toml`). Typing them again per invocation is how the campaign ended up with
34 hand-composed argv lines in untracked bash. `compose` is the one place they are joined.

Core tier: stdlib only, so `bench eval --dry-run` renders on any interpreter.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from axis_qvla.bench import execution
from axis_qvla.bench.backends import BACKENDS as REFERENCE_BACKENDS, DEFAULT_BACKEND, ENVIRONMENTS, REFERENCE_SERVERS
from axis_qvla.bench.protocol import parse_endpoint
from axis_qvla.bench.sim.arms import ArmError, check_arm_name, seed_set_arm
from axis_qvla.bench.spec import DEFAULT_NAME, Benchmark
from axis_qvla.core.hosts import EVAL_VENV, HostProfile

REPO_ROOT = Path(__file__).resolve().parents[2]

SERVINGS = ("bare", "conditioned")
# Where each reference server's pieces live on a host, by ROOT NAME (axis_qvla.bench.backends). They
# are the layout the paper's evaluations used, so a `bench eval` of a paper checkpoint writes where
# its published run did. SmolVLA and X-VLA share the LeRobot checkout and venv; only the server
# script and the port block differ.
BACKENDS = REFERENCE_BACKENDS
_BACKEND_ROOTS = {k: {"logs": s.logs_root, "openpi_dir": s.server_root, "base_port": s.base_port}
                  for k, s in REFERENCE_SERVERS.items()}
# The driver's --log-root: it hands this one root to every worker and rollout and tallies it, so
# the seed files land here whichever checkout the workers import (axis_qvla.bench.sim.campaign). A profile
# that declares the `rollout_logs` root (qvla init writes <root>/r1/rollout_logs) puts every backend's
# there, beside the out_<run>/ directories; one that does not keeps the layout the paper's evaluations
# wrote, <backend's logs root>/ROLLOUT_LOGS_REL, so a re-tally of those runs finds their seed files.
ROLLOUT_LOGS_ROOT = "rollout_logs"
ROLLOUT_LOGS_REL = "eval_results/slb_pi05/rollout_logs"


class LaunchError(RuntimeError):
    """The host lacks a root or venv the requested backend needs."""


@dataclasses.dataclass(frozen=True)
class EvalRequest:
    checkpoint: str | None
    config: str | None
    arm: str | None = None
    serving: str = "bare"
    guidance_scale: float = 0.0
    backend: str = DEFAULT_BACKEND
    benchmark: str = DEFAULT_NAME
    gpus: str | None = None
    workers_per_gpu: int = 4
    batch: int = 10
    mem_fraction: str = "0.5"
    tally_only: bool = False
    # axis-render checkout; its src/ goes on the workers' PYTHONPATH. None: this checkout's
    # third_party/axis-render.
    render_root: str | None = None
    # Running policy servers to evaluate (HOST:PORT). Non-empty: nothing is started, and the
    # checkpoint/config/serving/backend fields describe nothing that runs.
    policy: tuple[str, ...] = ()
    # Which declared seeds: "test", or "dev" (filed under <arm>_dev, axis_qvla.bench.sim.arms).
    seed_set: str = "test"
    # Run although the environment changes a declared execution constant; the driver records it.
    off_protocol: bool = False


@dataclasses.dataclass(frozen=True)
class Launch:
    arm: str
    argv: list[str]
    env: dict[str, str]
    # The interpreter the driver must run under: the host's evaluation environment ([venv] eval),
    # which carries the dual-sim client and openpi_client. The worker interpreter is the same one.
    python: Path


def default_arm_name(checkpoint: str, serving: str) -> str:
    return f"{Path(checkpoint).name}_{serving}"


def _profile_file(host: HostProfile) -> str:
    """The file to tell someone to edit: where this profile was read from (./qvla.toml, an explicit
    --host path, or a named profile; axis_qvla.data.userconf.load_profile), never a guess from the name
    alone. A profile built by hand has no file and is named instead."""
    return host.source()


def _root(host: HostProfile, key: str) -> Path:
    root = host.roots.get(key)
    if root is None:
        raise LaunchError(
            f"host {host.name!r} declares no root {key!r} (has: {', '.join(sorted(host.roots))}). "
            f"Add it to {_profile_file(host)} [roots]."
        )
    return root


def run_root(host: HostProfile) -> Path:
    """The campaign root: out_<arm>/ directories, run manifests and summaries."""
    return _root(host, "r1")


def rollout_log_root(host: HostProfile, legacy_root: str) -> Path:
    """Where the per-seed rollout files go: the declared `rollout_logs` root, else the paper layout
    under `legacy_root` (the backend's logs root)."""
    declared = host.roots.get(ROLLOUT_LOGS_ROOT)
    return declared if declared is not None else _root(host, legacy_root) / ROLLOUT_LOGS_REL


def _venv(host: HostProfile, key: str) -> Path:
    venv = host.venvs.get(key)
    if venv is None:
        raise LaunchError(
            f"host {host.name!r} declares no venv {key!r} (has: {', '.join(sorted(host.venvs)) or 'none'}). "
            f"Add it to {_profile_file(host)} [venv]."
        )
    return venv


def eval_python(host: HostProfile) -> Path:
    """The host's evaluation-environment interpreter ([venv] eval; a profile's legacy `render` key is
    read as it), which `bench eval`, its driver and its workers run under."""
    return _venv(host, EVAL_VENV) / "bin" / "python"


render_python = eval_python     # the name callers used before the key was `eval`


def interpreter_problem(python: Path) -> str | None:
    """Why `python` cannot be run, or None: it is not there (an environment not built yet), or a
    parent cannot be read -- named as such, never read as absent. An explicit stat, not
    Path.is_file: on newer Pythons that is os.path.isfile, which answers False for EACCES too."""
    try:
        mode = Path(python).stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return "is not built"
    except OSError as exc:
        return f"cannot be read ({type(exc).__name__})"
    return None if mode & 0o170000 == 0o100000 else "is not built"


def _server_python(server, host: HostProfile) -> Path:
    """The interpreter a reference server runs under: the online venv for the JAX server, else the
    host root its table row names (axis_qvla.bench.backends)."""
    if server.xla:
        return _venv(host, "online") / "bin" / "python"
    return _root(host, server.server_venv_root) / "bin" / "python"


def missing_env(req: EvalRequest, host: HostProfile) -> str | None:
    """Why the reference server this run starts cannot start, naming the one command that builds its
    environment; None for `--policy` (nothing is started) and a tally."""
    if req.policy or req.tally_only:
        return None
    server = REFERENCE_SERVERS[req.backend]
    py = _server_python(server, host)
    gap = interpreter_problem(py)
    if gap is None:
        return None
    return (f"the {server.short_label} server's environment ({py}) {gap}: build it with `qvla env build "
            f"{ENVIRONMENTS[req.backend]}`, or point {_profile_file(host)} at it. Refused before any server starts.")


def _base_weights(host: HostProfile) -> Path:
    """The base-weights copy `qvla data get base-weights` fills (axis_qvla.data.layout), under the cache root."""
    from axis_qvla.core.paths import Paths
    from axis_qvla.data.layout import base_weights_dest

    return base_weights_dest(Paths(host))


def _openpi_home(host: HostProfile) -> dict[str, str]:
    """OPENPI_DATA_HOME for the pi0.5 server: the base-weights copy's openpi/ once it holds the
    PaliGemma tokenizer the server reads at start (`qvla data get base-weights --serving pi05` puts it
    there; `qvla train` reads the same copy), else the host's own cache (HostProfile.env)."""
    from axis_qvla.bench.sim.campaign import OPENPI_TOKENIZER

    if "cache" not in host.roots:
        return {}
    home = _base_weights(host) / "openpi"
    # An explicit stat, not Path.is_file, which can answer False for EACCES (interpreter_problem).
    try:
        there = (home / OPENPI_TOKENIZER).stat().st_mode & 0o170000 == 0o100000
    except (FileNotFoundError, NotADirectoryError):
        there = False
    except OSError as exc:
        raise LaunchError(f"{home / OPENPI_TOKENIZER} cannot be read ({type(exc).__name__})") from None
    return {"OPENPI_DATA_HOME": str(home)} if there else {}


def _hf_home(host: HostProfile) -> Path:
    """HF_HOME for a LeRobot server: the host's hf_cache, else the base-weights copy's hf/ that
    `qvla data get base-weights --serving smolvla|xvla` fills; a host with neither root is told so."""
    if "hf_cache" in host.roots or "cache" not in host.roots:
        return _root(host, "hf_cache")
    return _base_weights(host) / "hf"


def render_tree(host: HostProfile, render_root: str | Path | None) -> Path:
    """The axis-render checkout the rollout workers IMPORT (its src/ goes on their PYTHONPATH), so the
    tree they render from: the render shell reads the Polyhaven bank relative to that package.
    --render-root when given, else this checkout's third_party/axis-render. The one resolver for the
    PYTHONPATH and the bank check (axis_qvla.bench.inputs)."""
    return Path(render_root) if render_root else REPO_ROOT / "third_party" / "axis-render"


def _worker_env(host: HostProfile, b: Benchmark, req: EvalRequest) -> dict[str, str]:
    """The eval-convention env every worker needs, whichever branch composed the launch: the
    benchmark's declared conventions (axis_qvla.bench.execution.convention_env) and the render tree on
    PYTHONPATH. ONE function, so a pin added later cannot land on only the reference-server branch
    and silently miss `--policy` runs -- the train/serve convention-mismatch class this project
    has already been bitten by. The execution constants are not exported: `bench eval` and the
    driver check the environment this env is merged into against the declaration
    (axis_qvla.bench.execution.run_record), for both branches alike."""
    return {
        **host.env,
        **execution.GL_ENV,
        **execution.convention_env(b),
        "PYTHONPATH": f"{REPO_ROOT}:{render_tree(host, req.render_root) / 'src'}",
        "AXIS_REPLAY_ASSET_CACHE_DIR": str(_root(host, "eval") / "asset_cache"),
    }


def _arm(req: EvalRequest, arm: str) -> str:
    """The arm the run is filed under: `arm`, named for the request's seed set."""
    try:
        check_arm_name(arm)
        return seed_set_arm(arm, req.seed_set)
    except ArmError as exc:
        raise LaunchError(str(exc)) from None


def filed_arm(req: EvalRequest) -> str:
    """The name a reference-server run is filed under (its out_<run>/ directory under `run_root`):
    --run, else <checkpoint dir basename>_<serving>, named for the request's seed set. `compose`
    names the run with it, and `bench eval` puts the server's rendered config beside it."""
    return _arm(req, req.arm or default_arm_name(req.checkpoint, req.serving))


def _protocol_flags(req: EvalRequest) -> list[str]:
    """The seed set and the off-protocol allowance, for either branch."""
    return ((["--seed-set=dev"] if req.seed_set == "dev" else [])
            + (["--off-protocol"] if req.off_protocol else []))


def compose(req: EvalRequest, host: HostProfile, b: Benchmark) -> Launch:
    """Every driver argument, from the declaration and the host; nothing typed twice."""
    if req.policy:
        return _compose_external(req, host, b)
    if req.serving not in SERVINGS:
        raise LaunchError(f"serving must be one of {SERVINGS}, got {req.serving!r}")
    if req.backend not in BACKENDS:
        raise LaunchError(f"backend must be one of {BACKENDS}, got {req.backend!r}")
    roots = _BACKEND_ROOTS[req.backend]
    arm = filed_arm(req)
    render_py = eval_python(host)
    # The table's fields, not the backend's name: `xla` picks the online venv and the benchmark's
    # XLA pin (a torch server needs neither); `server_venv_root` names the host ROOT key holding
    # the torch server's interpreter, independently of whether it is the XLA backend.
    server = REFERENCE_SERVERS[req.backend]
    server_py = _server_python(server, host)
    xla_pin = (b.xla_pin or "-") if server.xla else "-"   # a torch server: the XLA pin is meaningless
    gpus = req.gpus or ",".join(str(i) for i in range(host.gpus))

    # `--k=v`, never `--k v`: the XLA pin's value begins with `--`, which argparse would read
    # as the next option under the two-token form. Same rendering as the spec launcher.
    pairs = [
        ("run", arm),
        ("config", req.config),
        ("checkpoint", req.checkpoint),
        ("benchmark", req.benchmark),
        ("tasks", str(b.declaration_path)),
        ("n-seeds", str(b.n_seeds)),
        ("layout-manifest", str(b.layouts_path)),
        ("run-root", str(run_root(host))),
        ("log-root", str(rollout_log_root(host, roots["logs"]))),
        ("openpi-dir", str(_root(host, roots["openpi_dir"]))),
        ("server-python", str(server_py)),
        ("worker-python", str(render_py)),
        ("xla-pin", xla_pin),
        ("serving", req.serving),
        ("backend", req.backend),
        ("guidance-scale", f"{float(req.guidance_scale):g}"),
        ("gpus", gpus),
        ("workers-per-gpu", str(req.workers_per_gpu)),
        ("batch", str(req.batch)),
        ("base-port", str(roots["base_port"])),
        ("mem-fraction", req.mem_fraction),
    ]
    argv = [f"--{k}={v}" for k, v in pairs]
    if req.tally_only:
        argv.append("--tally-only")
    argv += _protocol_flags(req)

    env = _worker_env(host, b, req)
    if server.xla:
        env.update(_openpi_home(host))
    if server.hf_env:
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(_hf_home(host)),
        })
    return Launch(arm=arm, argv=argv, env=env, python=render_py)


def _compose_external(req: EvalRequest, host: HostProfile, b: Benchmark) -> Launch:
    """`--policy`: the driver attaches to running servers. No server interpreter, checkout or port
    block is needed; the workers still render, so the evaluation environment and the eval roots are."""
    try:
        for p in req.policy:
            parse_endpoint(p)
    except ValueError as exc:
        raise LaunchError(str(exc)) from None
    if not req.arm:
        raise LaunchError("--policy needs --run: results are filed under it")
    arm = _arm(req, req.arm)
    render_py = eval_python(host)
    gpus = req.gpus or ",".join(str(i) for i in range(host.gpus))
    pairs = [
        ("run", arm), ("benchmark", req.benchmark), ("tasks", str(b.declaration_path)),
        ("n-seeds", str(b.n_seeds)), ("layout-manifest", str(b.layouts_path)),
        ("run-root", str(run_root(host))), ("log-root", str(rollout_log_root(host, "bench"))),
        ("worker-python", str(render_py)),
        ("xla-pin", "-"),   # the workers run no JAX; the server pins its own XLA flags
        ("gpus", gpus), ("workers-per-gpu", str(req.workers_per_gpu)), ("batch", str(req.batch)),
        *[("policy", p) for p in req.policy],
    ]
    if req.checkpoint:
        pairs.append(("checkpoint", req.checkpoint))
    argv = [f"--{k}={v}" for k, v in pairs] + (["--tally-only"] if req.tally_only else []) + _protocol_flags(req)
    env = _worker_env(host, b, req)
    return Launch(arm=arm, argv=argv, env=env, python=render_py)


_ROOT_HINTS = {
    "--asset-dir": "`qvla data get bench` puts the benchmark's assets there",
    "--task-data-root": "`qvla data get bench` puts the benchmark's tasks there",
    "--render-root": "the renderer is the third_party/axis-render submodule "
                     "(`git submodule update --init third_party/axis-render`)",
}


def missing_root(render_root: Path | None, task_data_root: Path | None, *, dual_sim: bool,
                 asset_dir: Path | None = None) -> str | None:
    """Why a run cannot start, naming the flag and how to get the directory; None when every root it
    needs is a directory. The render root (default third_party/axis-render) is needed only by the
    render half (dual_sim). Only a missing path reads as missing; any other OSError is reported as
    unreadable."""
    if task_data_root is None:
        return ("--task-data-root is required: the tasks/ directory of a local qvla-v1-bench copy "
                "(`qvla data get bench`). Refused before any server starts.")
    need = [("--task-data-root", task_data_root)]
    if asset_dir is not None:
        need.insert(0, ("--asset-dir", asset_dir))
    if dual_sim:
        need.insert(0, ("--render-root", render_root or REPO_ROOT / "third_party" / "axis-render"))
    for flag, root in need:
        try:
            if Path(root).stat().st_mode & 0o170000 == 0o040000:
                continue
            why = "is not a directory"
        except FileNotFoundError:
            why = "does not exist"
        except OSError as exc:
            why = f"cannot be read ({type(exc).__name__})"
        return f"{flag} {root} {why}: {_ROOT_HINTS[flag]}, or pass {flag}. Refused before any server starts."
    return None
