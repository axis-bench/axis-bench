"""Run one evaluation as a seed-level work queue against persistent servers: one server per GPU for
the whole run, and (task, seed) units handed out as workers free up, so no worker waits on a peer (a
per-task wave spends ~40% of its GPU time waiting at the barrier).

Every child is a Popen with a real pid, and servers are terminated by pid, never by a `ps | grep`
pattern a waiter's own argv could contain.

SAFETY: never `pkill -f` on a shared box, and never kill a process this driver did not start.

Prerequisites, all enforced or asserted:
  * the server must advertise `supports_client_noise` (openpi >= the noise commit) -- checked by a
    real warm-up inference, not by grepping the log for "listening" and sleeping;
  * the determinism env must be set identically on servers and workers (see --xla-pin).

Runs in the eval environment (conf/envs/eval.toml) and imports openpi_client inside the functions
that need it; module scope is stdlib only so `bench eval` can import it from any interpreter.

ONE LOG ROOT. The driver resolves `--log-root` once (absolute) and hands it to every queue worker,
which passes it to every rollout CLI invocation; the tally reads the same path, so the files a run
writes are the files it tallies, whichever checkout the workers import.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from axis_qvla.bench import checkpoint_digest, execution
from axis_qvla.bench.backends import (BACKENDS, DEFAULT_BACKEND, QUALITY_TAG, REFERENCE_SERVERS,  # noqa: F401
                                 SMOLVLA_SERVER_CMD, XVLA_DOMAIN_ID, XVLA_SERVER_CMD)
from axis_qvla.bench.protocol import (CONDITIONING_KEYS, CONTRACT_KEYS, IDENTITY_KEYS, SHAPE_KEYS,
                                 handshake_problems, parse_endpoint)
from axis_qvla.bench.spec import DEFAULT_NAME as DEFAULT_BENCHMARK, SEED_SETS
from axis_qvla.bench.sim.arms import DUAL_SIM, FAILED_SUMMARY_FILE, SUMMARY_FILE, ArmError, check_arm_name, out_dir_name
from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED
from axis_qvla.bench.sim.sources import rollout_sources
from axis_qvla.bench.sim.workqueue import Unit, WorkQueue

REPO_ROOT = Path(__file__).resolve().parents[3]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [queue-driver] {msg}", flush=True)


# --- serving protocol -------------------------------------------------------------------------

SERVINGS = ("bare", "conditioned")
_LEROBOT_SERVER_CMDS = {k: s.server_cmd for k, s in REFERENCE_SERVERS.items() if s.server_cmd}


@dataclasses.dataclass(frozen=True)
class Protocol:
    """The three strings the driver actually passes on, derived from (serving, backend, G)."""

    serving: str | None
    backend: str
    guidance_scale: float
    serve_flags: str
    worker_flags: str
    server_cmd: str | None


def serving_protocol(serving: str, backend: str = DEFAULT_BACKEND, guidance_scale: float = 0.0) -> Protocol:
    """`conditioned` = server tags every prompt with quality tag 5, client passes
    --cfg-conditioning so the result directory carries `_w<G>`; `bare` = neither.

    The LeRobot servers (SmolVLA, X-VLA) can only tag, not guide, so a non-zero guidance scale
    there is refused rather than silently served at 0 under a `_w<G>` name that claims otherwise.
    """
    if serving not in SERVINGS:
        raise ValueError(f"serving must be one of {SERVINGS}, got {serving!r}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    g = float(guidance_scale)
    if serving == "bare" and g != 0.0:
        raise ValueError("guidance_scale requires serving=conditioned; a bare policy has no "
                         "tagged branch for the guidance weight to act on")
    if not REFERENCE_SERVERS[backend].can_guide and g != 0.0:
        raise ValueError(f"the {REFERENCE_SERVERS[backend].short_label} server cannot apply guidance; "
                         f"only --guidance-scale 0 is supported for backend={backend}")
    cond = serving == "conditioned"
    worker = f"--cfg-conditioning --guidance-scale {g:g}" if cond else ""
    if backend in _LEROBOT_SERVER_CMDS:
        server_cmd = _LEROBOT_SERVER_CMDS[backend] + (f" --quality-tag {QUALITY_TAG}" if cond else "")
        return Protocol(serving, backend, g, "", worker, server_cmd)
    serve = f"--quality-tag {QUALITY_TAG} --guidance-scale {g:g}" if cond else ""
    return Protocol(serving, backend, g, serve, worker, None)


def resolve_protocol(a) -> Protocol:
    """The protocol `--serving` (default bare) and `--backend` derive."""
    try:
        return serving_protocol(a.serving or "bare", a.backend, a.guidance_scale)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def external_protocol(a) -> Protocol:
    """--policy: the servers exist already, so nothing on this command line describes how they
    serve; their handshakes do (read by external_handshakes before the run is recorded)."""
    given = [flag for flag, on in (("--serving", a.serving is not None), ("--backend", a.backend != DEFAULT_BACKEND),
                                   ("--guidance-scale", float(a.guidance_scale) != 0.0),
                                   ("--config", a.config is not None)) if on]
    if given:
        raise SystemExit(f"--policy names running servers; {', '.join(given)} describe a server this "
                         f"driver starts")
    try:
        a.endpoints = [parse_endpoint(p) for p in a.policy]
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    return Protocol(None, "external", 0.0, "", "", None)


_MISSING = object()   # a declared key absent from a handshake; distinct from any real value (incl. None)


def _present(v: object) -> str:
    """How a handshake value reads in a refusal message: the sentinel for "absent" renders as the
    word, never its own repr (`<object object at 0x...>`)."""
    return "<absent>" if v is _MISSING else repr(v)


def declared_differences(old: dict, new: dict) -> dict[str, tuple[object, object]]:
    """Every declared handshake key (axis_qvla.bench.protocol's CONDITIONING_KEYS, SHAPE_KEYS,
    CONTRACT_KEYS, IDENTITY_KEYS) where `old` and `new` disagree, as {key: (old_value, new_value)}
    -- raw values, including the `_MISSING` sentinel for an absent key (render with `_present`,
    never the sentinel's own repr). A key stated on one side and absent on the other counts as a
    change. Never includes an undeclared extra key (the protocol permits one unchecked, and it
    may be volatile: a start time, a pid, a build stamp) or an endpoint (serving.external; a
    restarted server may move host or port) -- the ONE comparison both `external_handshakes`
    (every server in a run must agree) and `refuse_handshake_mismatch` (a resumed run must still
    agree with what an earlier one recorded) apply."""
    keys = (*CONDITIONING_KEYS, *SHAPE_KEYS, *CONTRACT_KEYS, *IDENTITY_KEYS)
    return {k: (old.get(k, _MISSING), new.get(k, _MISSING)) for k in keys
            if old.get(k, _MISSING) != new.get(k, _MISSING)}


def external_handshakes(endpoints: list[tuple[str, int]], *, timeout_s: float) -> list[dict]:
    """Every server's handshake, checked against the protocol, before anything is recorded or run.
    One arm is one policy served one way: every server must agree on every declared key
    (`declared_differences` against the first), not only conditioning -- workers are spread
    round-robin over the endpoints, so two servers serving two different models under the same
    conditioning would otherwise split one arm's seed files between them."""
    mds = []
    for host, port in endpoints:
        md = wait_ready(host, port, timeout_s, require_noise=True)
        problems = handshake_problems(md)
        if problems:
            raise SystemExit(f"{host}:{port} is not a conformant policy server: {'; '.join(problems)} "
                             f"(qvla bench check-server --policy {host}:{port})")
        mds.append(md)
    for (host, port), md in zip(endpoints[1:], mds[1:]):
        diffs = declared_differences(mds[0], md)
        if diffs:
            shown = "; ".join(f"{k}: {_present(o)} vs {_present(n)}" for k, (o, n) in diffs.items())
            h0, p0 = endpoints[0]
            raise SystemExit(f"{h0}:{p0} and {host}:{port} disagree ({shown}): one run is one "
                             f"policy served one way")
    return mds


def external_served(arm: str, handshakes: list[dict]) -> str:
    """The directory the rollouts write under, derived from what the servers announced -- the same
    source the rollout client names it from (axis_qvla.bench.sim.cli)."""
    from axis_qvla.bench.sim.arms import served_arm_name  # noqa: PLC0415
    from axis_qvla.bench.sim.policy import served_conditioning  # noqa: PLC0415

    cond, g = served_conditioning(handshakes[0])
    return served_arm_name(arm, cfg_conditioning=cond, guidance_scale=g, domain=DUAL_SIM)


def recorded_handshakes(out_dir: Path) -> list[dict]:
    """A --tally-only of an external arm: the handshakes the run recorded name its directory."""
    rec = recorded_run_manifest(Path(out_dir)) or {}
    hs = (rec.get("serving") or {}).get("handshakes")
    if not hs:
        raise SystemExit(f"{Path(out_dir) / 'run_manifest.json'} records no server handshake, so this "
                         f"--policy run's result directory cannot be named; re-run it")
    return hs


def refuse_handshake_mismatch(out_dir: Path, handshakes: list[dict], checkpoint_sha256: str | None) -> None:
    """A resumed --policy run: refuse before the queue is reused or run_manifest.json is
    overwritten unless the server's declared identity, conditioning and shape still match an
    EARLIER real run of this arm. Compares index [0] with `declared_differences`, plus the
    checkpoint content digest from the run records when both have one; `external_handshakes`
    already requires every server in ONE run to agree on the same keys, so comparing [0] against
    [0] here is sufficient -- it is not a weaker check than comparing every server.

    Also refuses outright when the recorded run of this arm was NOT a --policy run: a bare
    reference arm and a plain external arm both resolve to the same served directory
    (`<arm>_dualsim`), so resuming one as the other would compare nothing and could silently mix a
    reference run's seed files with an external one's, or vice versa; and when that run's record
    is unreadable, corrupt, or (despite being external) recorded no handshake at all -- each is
    real information about the earlier run, never read as "nothing to resume"."""
    p = Path(out_dir) / "run_manifest.json"
    prev = recorded_run_manifest(p.parent)
    if prev is None:
        return   # no earlier real run of this arm at all -- nothing to resume
    if "error" in prev:
        raise SystemExit(
            f"cannot verify that this resume talks to the server the earlier run recorded: "
            f"{prev['error']}; fix or remove the record, or use a new --run"
        )
    if not (prev.get("serving") or {}).get("external"):
        raise SystemExit(
            f"{p} records a reference-server run under this name, not --policy; resuming it here "
            f"would compare nothing against a server, and could silently mix its seed files with "
            f"an external run's. Use a new --run."
        )
    prev_hs_list = (prev.get("serving") or {}).get("handshakes")
    if not prev_hs_list:
        raise SystemExit(
            f"cannot verify that this resume talks to the server the earlier run recorded: {p} "
            f"is a --policy run with no recorded handshake; fix or remove the record, or use a "
            f"new --run"
        )
    prev_hs, now_hs = prev_hs_list[0], handshakes[0]

    diffs = declared_differences(prev_hs, now_hs)

    old_sha = (prev.get("checkpoint") or {}).get("sha256")
    if old_sha and checkpoint_sha256 and old_sha != checkpoint_sha256:
        diffs["checkpoint.sha256"] = (old_sha, checkpoint_sha256)

    if diffs:
        shown = "; ".join(f"{k}: {_present(o)} -> {_present(n)}" for k, (o, n) in diffs.items())
        raise SystemExit(
            f"{p} recorded a different server under this name ({shown}). Resuming would reuse the "
            f"queue and overwrite that manifest for a server this run never verified against it. "
            f"Use a new --run."
        )
    if not any(k in prev_hs or k in now_hs for k in IDENTITY_KEYS):
        log(f"{p}: neither handshake states {'/'.join(IDENTITY_KEYS)}; resume protection covers "
            f"conditioning and shape only")


# --- units ------------------------------------------------------------------------------------

def build_units(task_ids: list[int], n_seeds: int, seed_source, seed_set: str = "test") -> list[Unit]:
    """One unit per (task, seed), in a FIXED canonical order.

    Order is canonical because the arm fingerprint is a hash over the result files; a set- or
    dict-ordered unit list would make two identical runs differ in their queue file alone.
    """
    units: list[Unit] = []
    for tid in task_ids:
        try:
            seeds = list(seed_source(tid))[:n_seeds]
        except StopIteration as e:
            # _task_meta raises a bare StopIteration for an unknown task, which surfaces as an
            # unreadable traceback after the queue is half built. Name the task instead.
            raise SystemExit(
                f"task {tid} is not in the layout/seed manifest this driver resolved. A task "
                f"absent from the manifest fails later as an empty error:build_scene: message."
            ) from e
        if len(seeds) < n_seeds:
            raise SystemExit(
                f"task {tid} declares {len(seeds)} {seed_set} seeds, need {n_seeds}; a short task would "
                "silently shrink the denominator for this run only"
            )
        units.extend(Unit(int(tid), int(s)) for s in seeds)
    return units


def default_seed_source(tid: int, benchmark: str | None = None, seed_set: str = "test"):
    """Seeds come from the tracked benchmark declaration (conf/benchmarks/<version>/<name>.tasks.json,
    and its dev_seeds for a development run): a clean clone builds the same unit list the paper
    ran."""
    from axis_qvla.bench.spec import DEFAULT_NAME, load_benchmark  # noqa: PLC0415

    return load_benchmark(benchmark or DEFAULT_NAME).seeds_for(tid, seed_set)


# --- servers ----------------------------------------------------------------------------------

# What openpi's pi0.5 server fetches into its download cache when it builds the tokenizer
# (gs://big_vision/paligemma_tokenizer.model, cached at <OPENPI_DATA_HOME>/big_vision/...). The
# cache is the host's declared <cache>/openpi (axis_qvla.core.hosts), never a home directory.
OPENPI_TOKENIZER_URL = "gs://big_vision/paligemma_tokenizer.model"
OPENPI_TOKENIZER = OPENPI_TOKENIZER_URL.removeprefix("gs://")


def openpi_asset_problem(env_base: dict) -> str | None:
    """Why an openpi server may fail to start, or None: its tokenizer is not in the download cache
    it reads (OPENPI_DATA_HOME), so it must download it, which an offline machine cannot. The line
    names the missing file and the root, so a failed start says what to put where."""
    root = env_base.get("OPENPI_DATA_HOME")
    where = f"OPENPI_DATA_HOME={root}" if root else "OPENPI_DATA_HOME is unset (openpi's default ~/.cache/openpi)"
    path = Path(root or "~/.cache/openpi").expanduser() / OPENPI_TOKENIZER
    try:
        path.stat()
        return None
    except FileNotFoundError:
        why = "is not there"
    except OSError as exc:
        why = f"cannot be read ({type(exc).__name__})"
    return (f"the pi0.5 server's tokenizer {path} {why} ({where}): the server downloads "
            f"{OPENPI_TOKENIZER_URL} there at start, which an offline machine cannot; put the file "
            f"at that path (the base-weights copy carries one under openpi/{OPENPI_TOKENIZER})")


def config_file_problem(config: str | None) -> str | None:
    """Why a `--config <file>.json` cannot be handed to a server, or None (a config NAME is the
    server checkout's to resolve). Checked before any record, queue or server: a server given a
    missing file exits at start."""
    if not config or not config.endswith(".json"):
        return None
    try:
        mode = Path(config).stat().st_mode
    except FileNotFoundError:
        why = "does not exist"
    except OSError as exc:
        why = f"cannot be read ({type(exc).__name__})"
    else:
        if stat.S_ISREG(mode):
            return None
        why = "is not a file"
    return (f"--config {config} {why}; refused before any record, queue or server (`bench eval "
            f"--checkpoint-id` renders a paper checkpoint's config)")


def config_sha256(config: str | None) -> str | None:
    """sha256 of a `--config <file>.json`, recorded beside its path: the file is what the servers
    loaded. None for a config name, or a file that is not there (a tally after it was removed)."""
    if not config or not config.endswith(".json"):
        return None
    try:
        return sha256_file(Path(config))
    except FileNotFoundError:
        return None


def server_cwd(env_base: dict) -> str:
    """The directory `start_server` runs the server in (its Popen cwd): the server checkout,
    `--openpi-dir`. The argv's relative interpreter and script resolve against it, so every
    record of what the server ran must resolve against it too, never against this checkout."""
    return env_base.get("OPENPI_DIR", ".")


def server_command(gpu: int, port: int, cfg_name: str, ckpt: str, serve_flags: list[str],
                   env_base: dict, server_cmd: str | None = None) -> list[str]:
    """The argv `start_server` launches. `run_manifest` probes its first element, so the
    recorded server versions are the launched interpreter's: splitting the unformatted template
    instead recorded '{server_python}', an error record, for every LeRobot backend."""
    server_python = env_base.get("SERVER_PYTHON", sys.executable)
    if server_cmd:
        return shlex.split(server_cmd.format(gpu=gpu, port=port, ckpt=ckpt, config=cfg_name,
                                             server_python=server_python))
    return [
        server_python,
        "scripts/serve_policy.py",
        f"--port={port}",
        *serve_flags,
        "policy:checkpoint",
        f"--policy.config={cfg_name}",
        f"--policy.dir={ckpt}",
    ]


def start_server(gpu: int, port: int, cfg_name: str, ckpt: str, serve_flags: list[str],
                 out_dir: Path, env_base: dict, mem_fraction: str,
                 server_cmd: str | None = None) -> subprocess.Popen:
    """Launch one policy server on `gpu`.

    `server_cmd` lets the same driver serve a DIFFERENT backbone. SmolVLA already shares this
    driver's rollout client (axis_qvla.bench.sim.cli); only the server binary differs, so a template is
    the whole difference between a pi0.5 arm and a SmolVLA arm. Without it the queue's scheduling
    win -- which SmolVLA needs just as much, since eval_schedule pins 24 tasks 1:1 to 24 slots with
    no work stealing -- would be available to only one of the two backbones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **env_base,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": mem_fraction,
    }
    cmd = server_command(gpu, port, cfg_name, ckpt, serve_flags, env_base, server_cmd)
    logf = (out_dir / f"server_gpu{gpu}.log").open("w")
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=server_cwd(env_base))
    log(f"server gpu{gpu} port {port} pid {p.pid}")
    return p


def wait_ready(host: str, port: int, timeout_s: float, require_noise: bool,
               proc: "subprocess.Popen | None" = None) -> dict:
    """Readiness is a REAL inference, not a log grep plus sleep.

    A grep on our own log tells us our process printed something; it does not tell us the client
    will reach OUR server. If the port is already held, a client would attach to the incumbent, so
    readiness is checked by a real inference against this server.

    `proc` lets this FAIL FAST: a server that has already exited can never become ready, so
    polling its port for the full timeout is pure waste: without this check, a bad config name
    can leave every server dead at startup while the driver cycles a full 600 s timeout per GPU
    with the box idle at 0%.

    The port is probed with a bounded websocket open BEFORE the client is constructed, because
    `WebsocketClientPolicy.__init__` retries forever (`while True: ... sleep(5)`) with no
    timeout of its own. A websocket open, not a bare TCP connect: a server's websocket library
    logs a connection closed before its HTTP request as a failed opening handshake, with a
    traceback, so a bare probe printed one into the user's server log every poll. The first version of this function checked `proc.poll()` once, then
    entered that constructor and never came back: the same bad-config-name failure hung the
    driver for 57 minutes a second time, with all eight servers already dead, until something
    sent it SIGTERM.
    """
    from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: PLC0415
    from websockets.sync.client import connect as ws_connect  # noqa: PLC0415

    deadline = time.time() + timeout_s
    last: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise SystemExit(
                f"server on port {port} exited rc={proc.returncode} before becoming ready; "
                f"it cannot recover -- read its server_gpu*.log for the traceback"
            )
        try:
            # Entered, never used bare (websockets 17.1 deprecates the bare return value).
            with ws_connect(f"ws://{host}:{port}", open_timeout=5.0, compression=None, max_size=None,
                            ping_interval=None):
                pass
        except Exception as e:  # noqa: BLE001 - not listening yet, or not past its handshake yet
            last = e
            time.sleep(3)
            continue
        try:
            client = WebsocketClientPolicy(host=host, port=port)
            md = client.get_server_metadata() or {}
            if require_noise and not md.get("supports_client_noise"):
                raise SystemExit(
                    f"server on port {port} does not advertise supports_client_noise; it would "
                    "silently fall back to its per-process key stream and produce order-dependent "
                    "results under this run's name"
                )
            return md
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 - server not up yet
            last = e
            time.sleep(3)
    raise SystemExit(f"server on port {port} not ready after {timeout_s:.0f}s: {last}")


# --- finalisation -----------------------------------------------------------------------------

def result_dir_name(arm: str, worker_flags: str) -> str:
    """The directory the rollout script writes under, derived the way IT derives it.

    A conditioned arm is served as `<arm>_w<guidance>_dualsim`, not `<arm>_dualsim`: the rollout
    script appends the guidance scale whenever --cfg-conditioning is set (a tagged-but-unguided
    policy is not the untagged one). The name is therefore derived through the same function the
    rollout script uses (`served_arm_name`), from the flags the server was launched with.
    """
    from axis_qvla.bench.sim.arms import served_arm_name  # noqa: PLC0415

    flags = worker_flags.split()
    cond = "--cfg-conditioning" in flags
    g = 0.0
    if "--guidance-scale" in flags:
        g = float(flags[flags.index("--guidance-scale") + 1])
    return served_arm_name(arm, cfg_conditioning=cond, guidance_scale=g, domain=DUAL_SIM)


def recompute_meta(log_root: Path, served: str, task_ids: list[int], n_seeds: int) -> dict:
    """Rebuild each task's tally from its seed files.

    A subset invocation writes a meta.json describing only its own seeds, so the last batch to
    finish would otherwise leave a meta claiming e.g. 10/10 for a 50-seed task. Scoring reads
    these, so they must describe the whole task or nothing.
    """
    from axis_qvla.bench.sim.record import read_seeds  # noqa: PLC0415

    per_task, incomplete, errored = {}, [], 0
    for tid in task_ids:
        d = log_root / served / f"task_{tid}"
        seeds = read_seeds(d) if d.exists() else []
        # read_seeds returns dicts (attribute access would read every task as 0/N).
        n_succ = sum(1 for r in seeds if bool(r.get("success")))
        # A rollout that dies in scene build writes a seed file with reason="error:..." and the
        # process still exits 0. Those are counted apart from failures, so a harness error is
        # reported as an error rather than scored as a policy outcome.
        errored += sum(1 for r in seeds if str(r.get("reason", "")).startswith("error:"))
        per_task[tid] = {"n_success": n_succ, "n_seeds": len(seeds)}
        if len(seeds) != n_seeds:
            incomplete.append((tid, len(seeds)))
    return {"per_task": per_task, "incomplete": incomplete, "errored": errored}


def fingerprint(log_root: Path, served: str, task_ids: list[int]) -> str:
    """sha256 over every seed file, in sorted order -- the arm's eval identity.

    It identifies the exact bytes a score was computed from: the same seed files always give the
    same fingerprint, and any changed file changes it. It is not a reproducibility test. Two
    evaluations of one checkpoint differ in some rollouts -- the renderer's multisample colour
    resolve differs by +-1 LSB between renders, and the closed loop amplifies that -- so their
    fingerprints differ; whether they agree is a question about their success rates
    (`results reeval`, docs/reference/reporting-rules.md).

    The files are exactly the tally's (`record.seed_file_names`: `seed_<int>.json`), so a
    diagnostics sidecar can move neither this digest nor the reported number. Per task: the
    names in sorted order, each name's bytes then the file's bytes, no separator.
    """
    from axis_qvla.bench.sim.record import seed_file_names  # noqa: PLC0415

    h = hashlib.sha256()
    for tid in sorted(task_ids):
        d = log_root / served / f"task_{tid}"
        for name in seed_file_names(d):
            h.update(name.encode())
            h.update((d / name).read_bytes())
    return h.hexdigest()


# --- run manifest -----------------------------------------------------------------------------
#
# Everything a reader needs to say WHICH model, WHICH code and WHICH machine produced a number,
# by content rather than by label. Without it, two checkpoints under one name, or one checkpoint
# served two ways, would be indistinguishable on disk.
#
# `sources.files` digests every file of the driver's checkout on the rollout path, computed from the
# import graph (axis_qvla.bench.sim.sources.rollout_sources). `sources.server` digests the script the
# server runs, in the checkout it runs from (`sources.openpi` is that checkout's git state): for
# SmolVLA and X-VLA that is `smolvla_bench`, which on a worker box can be a different checkout from
# the driver's, so the driver's copy of the same relative path would be a plausible digest of a file
# that did not run. Nothing compares these digests across runs, so the key set may grow with the
# code.

PINNED_ENV = (
    "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "MUJOCO_GL", "PYOPENGL_PLATFORM",
    "AXIS_EVAL_IMAGE_GEOMETRY", "AXIS_EVAL_ACTION_SPACE", "AXIS_EVAL_TABLE_TEXTURE", "HF_HUB_OFFLINE", "HF_HOME",
    "PI05_FPS", "PI05_REPLAN", "AXIS_EVAL_REPLAN", "MAX_ROLLOUT_STEPS", "N_STABLE_FRAMES",
    "AXIS_EVAL_GRIP_STATE", "AXIS_EVAL_EEF_IK",
)


def worker_env_base(env_base: dict) -> dict:
    """The environment every rollout worker starts from: the caller's, with the GL platform pinned
    (axis_qvla.bench.execution.GL_ENV). `env_pins` records this, so the record names the GL the
    workers rendered through rather than whatever the caller's shell held."""
    return {**env_base, **execution.GL_ENV}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_tree(root: Path) -> dict:
    """Content digest of a checkpoint directory: sha256 over (relative path, file sha256) of every
    file, in path-part order -- `axis_qvla.bench.checkpoint_digest.tree_digest`, the one walk the
    published digest and `--verify-checkpoint` use too (links followed; a directory that cannot be
    listed, a dangling link or a cycle raises CheckpointDigestError, never shrinks the digest).

    Full content on purpose. A size/mtime digest would call a re-saved checkpoint with different
    weights the same model, which is the mistake this exists to make impossible. Cost: one
    streaming read of the checkpoint, once per arm (tens of seconds against an hour of rollouts).
    """
    return {"path": str(root), **checkpoint_digest.tree_digest(Path(root))}


# A --checkpoint a server resolves by name rather than reads from this machine: a Hub repository id,
# `<org>/<name>`, or a URL (`gs://...`). The LeRobot servers hand the string to `from_pretrained`
# and openpi's loader takes a gs:// URL, so such a run is legitimate, but it has no bytes here to
# digest. An `@<revision>` suffix is recorded as given, for a server that resolves one; none of the
# reference servers does, so for them a revision is pinned by a local copy at it (`_pin_advice`).
_REFERENCE_URL = re.compile(r"^[a-z][a-z0-9+.-]*://\S+$")
_REFERENCE_HUB = re.compile(r"^(?P<repo>[A-Za-z0-9][\w.-]*/[\w.-]+)(?:@(?P<revision>\S+))?$")


def local_checkpoint(checkpoint: str | None) -> str | None:
    """`checkpoint` made absolute when it is a path on this machine, else as given (a reference).
    Done once, before anything reads it: the record digests it in this process's working directory
    and the server loads it from its own (`server_cwd`), so a relative path would name two
    directories, or be digested nowhere and served from somewhere."""
    if checkpoint and Path(checkpoint).exists():
        return str(Path(checkpoint).absolute())
    return checkpoint


def checkpoint_record(checkpoint: str | None, *, external: bool) -> dict:
    """What identifies the model a run serves, for run_manifest.json `checkpoint`:

      - a local directory: its content digest, `identity: "content"` (digest_tree; a file that is
        not a directory is refused, as no reference server loads one);
      - a reference a server resolves by name (`<org>/<name>[@<revision>]`, or a URL): the exact
        string as given, `identity: "name"`, with the `revision` it pins, or `revision: null` and
        `pinned: false` when it names none -- the same name can then serve other weights later;
      - neither (a local path that is not there): an error record, which identifies nothing;
      - under --policy with no --checkpoint: a record that the server, not this run, names it.

    A directory that cannot be digested is refused here, before any record, queue or server."""
    if checkpoint and Path(checkpoint).exists():
        try:
            return {**digest_tree(Path(checkpoint)), "identity": "content"}
        except checkpoint_digest.CheckpointDigestError as exc:
            raise SystemExit(f"--checkpoint cannot be digested, so nothing could say which model ran: {exc}. "
                             f"Refused before run_manifest.json, the queue or any server") from None
    hub = _REFERENCE_HUB.match(checkpoint or "")
    if hub or _REFERENCE_URL.match(checkpoint or ""):
        revision = hub.group("revision") if hub else None
        return {"path": checkpoint, "ref": checkpoint, "identity": "name", "revision": revision,
                "pinned": revision is not None}
    if external:
        return {"path": checkpoint, "external": True}
    return {"path": checkpoint, "error": "missing"}


def _pin_advice(rec: object) -> str | None:
    """How to give an unpinned reference an identity that cannot move, or None."""
    if not isinstance(rec, dict) or rec.get("identity") != "name" or rec.get("pinned"):
        return None
    ref = rec.get("ref")
    hub = _REFERENCE_HUB.match(str(ref))
    if hub:
        return (f"--checkpoint {ref} names no revision, so the same name can serve other weights later: pin "
                f"one, e.g. `hf download {hub.group('repo')} --revision <commit> --local-dir <dir>` and "
                f"--checkpoint <dir>, which is then identified by its bytes")
    return (f"--checkpoint {ref} names no revision: copy it to a local directory and pass that, which is "
            f"then identified by its bytes")


def git_state(d: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True,
                              timeout=60)

    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"path": str(d), "error": head.stderr.strip()[:200]}
        status = run("status", "--porcelain", "--untracked-files=no").stdout
        diff = run("diff", "HEAD").stdout
        return {
            "path": str(d), "head": head.stdout.strip(), "dirty": bool(status.strip()),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if status.strip() else None,
            "remote": run("remote", "get-url", "origin").stdout.strip() or None,
        }
    except (OSError, subprocess.SubprocessError) as e:
        return {"path": str(d), "error": repr(e)[:200]}


def python_versions(python: str) -> dict:
    """Framework versions as the named interpreter sees them. CUDA is hidden from the probe so it
    can never touch a GPU a live server is using."""
    code = (
        "import json,sys,importlib\n"
        "out={'python':sys.version.split()[0]}\n"
        "for m in ('jax','jaxlib','torch','numpy','mujoco','lerobot','openpi_client'):\n"
        "    try: out[m]=importlib.import_module(m).__version__\n"
        "    except Exception: out[m]=None\n"
        "try:\n"
        "    import torch; out['cudnn']=torch.backends.cudnn.version(); out['cuda']=torch.version.cuda\n"
        "except Exception: pass\n"
        "print(json.dumps(out))"
    )
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=240,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
        lines = r.stdout.strip().splitlines()
        if r.returncode == 0 and lines:
            return json.loads(lines[-1])
        return {"error": (r.stderr or r.stdout)[-300:]}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {"error": repr(e)[:200]}


def gpu_info() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        rows = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        return {"gpus": rows} if r.returncode == 0 else {"error": r.stderr[-200:]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": repr(e)[:200]}


def run_manifest(a, env_base: dict, units: list[Unit], *, probe: bool = True) -> dict:
    """The record of what one invocation runs: checkpoint, code, serving, benchmark, host.

    `main` writes it under out_dir BEFORE the first server starts, so even an arm that fails at
    startup leaves a record of what it tried to run. A real run writes run_manifest.json;
    --tally-only writes tally_manifest.json and --dry-run (built without the version and GPU
    probes) dry_run_manifest.json, so neither replaces the record of the run that wrote the seed
    files. arm_summary.json embeds it too (a tally embeds the run's own record as `provenance` and
    this one as `tally_provenance`)."""
    import socket  # noqa: PLC0415

    repo = REPO_ROOT
    flags = a.worker_flags.split()
    layout = None
    if "--layout-manifest" in flags:
        lp = Path(flags[flags.index("--layout-manifest") + 1])
        pending = getattr(a, "layout_manifest_bytes", None)
        if pending is not None and Path(getattr(a, "layout_manifest", "")) == lp:
            # The adapted copy this run will write after its resume check (main): its digest by bytes.
            layout = {"path": str(lp), "sha256": hashlib.sha256(pending).hexdigest()}
        else:
            layout = {"path": str(lp), "sha256": sha256_file(lp) if lp.exists() else None}
        declared = getattr(a, "layout_manifest_declared", None)
        if declared is not None and Path(declared) != lp:
            # The workers read an adapted copy (see harness_layout_manifest); record BOTH, so
            # a reader can tie the run to the tracked declaration by digest.
            dp = Path(declared)
            layout["declared"] = {"path": str(dp),
                                  "sha256": sha256_file(dp) if dp.exists() else None}
    try:
        from axis_qvla.bench.sim.inference_noise import NOISE_SALT  # noqa: PLC0415
        salt = NOISE_SALT
    except Exception:  # noqa: BLE001 - the worker refuses to run without it anyway
        salt = None
    uh = hashlib.sha256()
    for u in units:
        uh.update(f"{u.task_id}:{u.seed}\n".encode())
    t0 = time.time()
    external = bool(getattr(a, "policy", None))
    ckpt = checkpoint_record(a.checkpoint, external=external)
    m = {
        "arm": a.arm, "config": a.config, "config_sha256": config_sha256(a.config),
        "checkpoint": ckpt, "checkpoint_digest_s": round(time.time() - t0, 1),
        "serving": {"serving": getattr(a, "serving", None),
                    "backend": getattr(a, "backend", None),
                    "guidance_scale": getattr(a, "guidance_scale", None),
                    "serve_flags": a.serve_flags, "worker_flags": a.worker_flags,
                    "server_cmd": a.server_cmd, "mem_fraction": a.mem_fraction,
                    "workers_per_gpu": a.workers_per_gpu, "batch": a.batch,
                    "stale_after_s": a.stale_after_s,
                    "external": list(a.policy) if external else None,
                    "handshakes": getattr(a, "external_handshakes", None)},
        "benchmark": getattr(a, "benchmark_identity", None),
        "seed_set": getattr(a, "seed_set", "test"),
        # axis_qvla.bench.execution.run_record: what the workers execute against the declaration; None
        # for a tally, which executes nothing.
        "execution": getattr(a, "execution_record", None),
        "noise_salt": salt, "units_sha256": uh.hexdigest(), "n_units": len(units),
        "layout_manifest": layout,
        "log_root": str(a.log_root) if getattr(a, "log_root", None) is not None else None,
        "env_pins": {k: worker_env_base(env_base).get(k) for k in PINNED_ENV},
        "sources": {
            "repo": git_state(repo),
            # Under --policy no openpi checkout is used (the server exists already); probing
            # --openpi-dir's default would record a checkout the run never touched.
            "openpi": {"external": True} if external else git_state(Path(a.openpi_dir)),
            "files": {rel: (sha256_file(repo / rel) if (repo / rel).exists() else None)
                      for rel in rollout_sources(repo)},
            "server": {"external": list(a.policy)} if external else launched_server_script(a, env_base),
        },
        "host": socket.gethostname(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if probe:
        m["versions"] = {"worker": python_versions(a.worker_python),
                         "server": {"external": True} if external else python_versions(launched_server_python(a, env_base))}
        m["gpu"] = gpu_info()
    return m


def launched_server_python(a, env_base: dict) -> str:
    """The interpreter `start_server` launches for this run: the first element of the same
    argv, resolved against the server's working directory (OPENPI_DIR) as Popen resolves a
    relative program path."""
    exe = _launched_argv(a, env_base)[0]
    if "/" in exe and not Path(exe).is_absolute():
        return str(Path(server_cwd(env_base)) / exe)
    return exe


def launched_server_script(a, env_base: dict) -> dict:
    """The script `start_server` runs for this run, digested in the checkout it runs from.

    The second element of the same argv, resolved against the server's cwd as the interpreter
    resolves it (an absolute path ignores the root). `sha256` is None when that checkout holds no
    such file -- the containing directory may be entirely absent on a worker box's checkout, not
    just the file -- never this checkout's copy of the same relative path; an unreadable file is
    an error record, not a silent None (an unreadable directory must never read as an empty or a
    matching one).
    """
    argv = _launched_argv(a, env_base)
    root = server_cwd(env_base)
    script = argv[1] if len(argv) > 1 else None
    rec: dict = {"root": root, "script": script, "sha256": None}
    if script is None or script.startswith("-"):
        return rec   # e.g. `python -m pkg`: no script path on disk to digest
    try:
        rec["sha256"] = sha256_file(Path(root) / script)
    except FileNotFoundError:
        pass
    except OSError as exc:
        rec["error"] = f"{exc.__class__.__name__}: {exc}"[:200]
    return rec


def _launched_argv(a, env_base: dict) -> list[str]:
    return server_command(0, 0, a.config, a.checkpoint, a.serve_flags.split(), env_base,
                          a.server_cmd)


def refuse_execution_change(out_dir: Path, record: dict) -> None:
    """A resumed run must execute what the earlier run of this arm executed: the queue keeps that
    run's finished seed files, so two executions would be tallied as one arm while run_manifest.json
    named only the second. Compares the `executed` constants of the earlier run_manifest.json's
    execution record; an earlier record without one predates the record and is not compared, and
    an unreadable record is never read as "nothing to resume"."""
    prev = recorded_run_manifest(Path(out_dir))
    if prev is None:
        return
    if "error" in prev:
        raise SystemExit(f"cannot verify that this resume executes what the earlier run did: {prev['error']}; "
                         f"fix or remove the record, or use a new --run")
    before = (prev.get("execution") or {}).get("executed")
    if before is None:
        return
    now = record.get("executed") or {}
    changed = {k: (before.get(k), now.get(k)) for k in sorted({*before, *now}) if before.get(k) != now.get(k)}
    if changed:
        shown = "; ".join(f"{k}: {o!r} -> {n!r}" for k, (o, n) in changed.items())
        raise SystemExit(f"{Path(out_dir) / 'run_manifest.json'} recorded another execution under this name "
                         f"({shown}); resuming would tally both under one run. Use a new --run.")


def recorded_run_manifest(out_dir: Path) -> dict | None:
    """What the run that wrote the seed files recorded, for a --tally-only summary: None when it
    left no run_manifest.json, an error record when the file cannot be read (never a silent None)."""
    p = Path(out_dir) / "run_manifest.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return {"error": f"cannot read {p}: {type(exc).__name__}: {exc}"[:300]}
    return doc if isinstance(doc, dict) else {"error": f"{p} is not a JSON object"}


# What a resumed run must share with the run whose seed files the queue keeps, as dotted keys of
# run_manifest.json: the model (`checkpoint`: a local directory by its content digest, a reference
# by its exact string), the config the servers loaded, how they served it, and the benchmark
# version the seeds come from. The checkpoint's and the config file's PATHS are not compared (a
# moved root is the same model); their sha256 is. Under --policy the server's handshake names the
# model (refuse_handshake_mismatch), and --checkpoint is an optional record of it.
RESUME_IDENTITY = ("checkpoint", "config", "serving.serving", "serving.backend",
                   "serving.guidance_scale", "serving.serve_flags", "serving.server_cmd",
                   "benchmark.name", "benchmark.version_sha256")


def _resume_value(m: dict, key: str) -> object:
    if key == "config":
        # A rendered config file is identified by its bytes, a config name by itself.
        return f"sha256 {m['config_sha256']}" if m.get("config_sha256") else m.get("config", _MISSING)
    if key == "checkpoint":
        # A local directory by its content, a reference by its exact string; neither is _MISSING,
        # which never equals anything, itself included (resume_differences).
        ck = m.get("checkpoint")
        if not isinstance(ck, dict):
            return _MISSING
        if ck.get("sha256"):
            return f"sha256 {ck['sha256']}"
        return f"reference {ck['ref']}" if ck.get("ref") else _MISSING
    head, _, rest = key.partition(".")
    block = m.get(head)
    return block.get(rest, _MISSING) if isinstance(block, dict) else _MISSING


def resume_differences(prev: dict, new: dict, *, external: bool) -> dict[str, tuple[object, object]]:
    """Every RESUME_IDENTITY key on which an earlier run record and this run's manifest disagree,
    as {key: (old, new)}; a key stated on one side and absent on the other counts as a change. A
    reference run's checkpoint that neither record identifies is a change too: absent and absent
    are never the same model. Under --policy the checkpoint is compared only when both runs
    identify one (the server's handshake names the model there)."""
    out = {}
    for key in RESUME_IDENTITY:
        old, now = _resume_value(prev, key), _resume_value(new, key)
        if key == "checkpoint" and _MISSING in (old, now):
            if not external:
                out[key] = (old, now)
            continue
        if old != now:
            out[key] = (old, now)
    return out


def _resume_notes(prev: dict, new: dict, diffs: dict) -> list[str]:
    """What a refused resume's message adds about the checkpoint, after the keys that differ."""
    if "checkpoint" not in diffs:
        return []
    old, now = diffs["checkpoint"]
    notes = []
    if _MISSING in (old, now):
        side = ("the earlier record identifies no checkpoint" if now is not _MISSING else
                "this run identifies no checkpoint" if old is not _MISSING else "neither run identifies a checkpoint")
        notes.append(f"{side}: a run is identified by a local checkpoint "
                     f"directory's content digest or by the exact reference (a Hub id or URL) a server "
                     f"loads, and with neither nothing can say which model wrote the seed files")
    pc, nc = prev.get("checkpoint"), new.get("checkpoint")
    if isinstance(pc, dict) and isinstance(nc, dict) and pc.get("sha256") and nc.get("sha256"):
        notes.append(f"the digest covers every file under the directory, download caches and training state "
                     f"included ({pc.get('n_files')} files, {pc.get('bytes')} bytes recorded; "
                     f"{nc.get('n_files')} files, {nc.get('bytes')} bytes now)")
    notes += [advice for advice in dict.fromkeys(_pin_advice(r) for r in (pc, nc)) if advice]
    return notes


def refuse_resume_change(out_dir: Path, manifest: dict, handshakes: list[dict] | None,
                         *, queue_started: bool | None = None) -> None:
    """THE resume-identity check, for a reference-server run and a --policy run alike: a real run of
    an arm that already has a run_manifest.json reuses that run's queue, and with it every seed
    file the earlier run finished, then writes its own manifest and summary over them. So before
    the queue is touched or the record replaced, refuse unless the earlier run recorded the same
    model, config, serving and benchmark (RESUME_IDENTITY), the same server handshake under
    --policy (refuse_handshake_mismatch), and the same execution (refuse_execution_change). A
    record that cannot be read refuses too; only the absence of a record means "nothing to resume".
    The default arm name (<checkpoint dir basename>_<serving>) is shared by every checkpoint step
    of that name, so a different model under the same arm is the expected mistake, not a corner.
    `queue_started` (the arm's queue has claimed a unit) only words the refusal: a run that never
    started a rollout left nothing to keep, and the refusal says the record can be removed.
    A resume of the same unpinned reference is not refused; it is logged, with how to pin one."""
    p = Path(out_dir) / "run_manifest.json"
    prev = recorded_run_manifest(p.parent)
    if prev is None:
        return
    if "error" in prev:
        raise SystemExit(f"cannot verify that this resume runs what the earlier run under this name ran: "
                         f"{prev['error']}; fix or remove the record, or use a new --run")
    external = handshakes is not None
    if external:
        refuse_handshake_mismatch(p.parent, handshakes, (manifest.get("checkpoint") or {}).get("sha256"))
    elif (prev.get("serving") or {}).get("external"):
        raise SystemExit(f"{p} records a --policy run under this name, not a reference-server run; resuming it "
                         f"here would silently mix the two runs' seed files. Use a new --run.")
    diffs = resume_differences(prev, manifest, external=external)
    if diffs:
        shown = "; ".join(f"{k}: {_present(o)} -> {_present(n)}" for k, (o, n) in diffs.items())
        notes = "".join(f" {n[0].upper()}{n[1:]}." for n in _resume_notes(prev, manifest, diffs))
        if queue_started is False:
            then = (f" That run started no rollout (its queue records no claimed unit), so nothing of it would "
                    f"be kept: remove {p} and run again, or use a new --run.")
        else:
            then = " Resuming would keep that run's seed files and file them under this one's record. Use a new --run."
        raise SystemExit(f"{p} recorded another run under this name ({shown}).{notes}{then}")
    advice = _pin_advice(manifest.get("checkpoint"))
    if advice:
        log(f"WARNING resuming {p}'s run by the checkpoint's name alone: {advice}")
    refuse_execution_change(p.parent, manifest.get("execution") or {})


# --- benchmark identity -----------------------------------------------------------------------

def load_task_ids(spec: str) -> list[int]:
    """`--tasks` is a comma list, a .json list, or a .json benchmark manifest with `task_ids`."""
    if not spec.endswith(".json"):
        return [int(t) for t in spec.split(",") if t.strip()]
    doc = json.loads(Path(spec).read_text(encoding="utf-8"))
    return [int(t) for t in (doc["task_ids"] if isinstance(doc, dict) else doc)]


def benchmark_identity(task_ids: list[int], benchmark: str | None = None) -> dict:
    """Name the benchmark a run scored, by content rather than by what the caller typed.

    A summary must say WHICH benchmark it measured. This is
    `axis_qvla.bench.spec`'s identity (three digests, one per tracked file) plus an explicit
    `matches_manifest`, so a custom --tasks list is visible in the record instead of silent.
    """
    from axis_qvla.bench.spec import DEFAULT_NAME, BenchmarkError, load_benchmark  # noqa: PLC0415

    try:
        b = load_benchmark(benchmark or DEFAULT_NAME)
    except BenchmarkError as exc:
        return {"name": benchmark, "error": str(exc), "matches_manifest": False,
                "task_ids": sorted(task_ids)}
    return {
        **b.identity(),
        "manifest": str(b.declaration_path.relative_to(REPO_ROOT)),
        "matches_manifest": sorted(task_ids) == sorted(b.task_ids),
        "task_ids": sorted(task_ids),
    }


def harness_layout_manifest(path: Path, out_dir: Path) -> Path:
    """The layout file the WORKERS read, written (`harness_layout`, then the bytes)."""
    target, data = harness_layout(path, out_dir)
    if data is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return target


def harness_layout(path: Path, out_dir: Path) -> tuple[Path, bytes | None]:
    """The layout file the WORKERS read, and its bytes when it must be written there (None: the
    declared file itself). Nothing is written here, so a run refused before it starts leaves an
    earlier run's copy as that run recorded it.

    The tracked benchmark layouts file (`conf/benchmarks/axis-heldout-v1/<name>.layouts.json`) keys
    `tasks` by id, the way the other two benchmark files do; the layout generator
    (`axis_qvla.bench.sim.layouts`) iterates `tasks` as a LIST of entries carrying `task_id`. Handing the
    dict-shaped file straight to the harness fails inside build_scene on every rollout -- 1,200
    `error:` seed files after eight servers have started. So a dict-shaped file is rewritten, once,
    as a list-shaped sibling under the arm's out dir, and a list-shaped file is passed through
    untouched. Same content either way; run_manifest.json records both digests.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = doc.get("tasks")
    if not isinstance(tasks, dict):
        return Path(path), None
    entries = [tasks[k] for k in sorted(tasks, key=int)]
    for k, e in zip(sorted(tasks, key=int), entries):
        e.setdefault("task_id", int(k))
    adapted = {**doc, "tasks": entries, "adapted_from": str(path)}
    return out_dir / "layout_manifest.harness.json", json.dumps(adapted, indent=1).encode("utf-8")


def with_asset_flags(worker_flags: str, asset_dir: Path | None, asset_lock: Path | None) -> str:
    """Prepend --asset-dir/--asset-checksums to the worker flags, as --layout-manifest is
    prepended: the rollout CLI refuses to run under assets.fetch 'refuse' without --asset-dir."""
    for flag, val in (("--asset-checksums", asset_lock), ("--asset-dir", asset_dir)):
        if val is not None:
            worker_flags = f"{flag} {val} {worker_flags}".strip()
    return worker_flags


def with_root_flags(worker_flags: str, render_root: Path | None, task_data_root: Path | None) -> str:
    """Prepend --render-root/--task-data-root to the worker flags when given (bench eval always gives
    --task-data-root)."""
    for flag, val in (("--task-data-root", task_data_root), ("--render-root", render_root)):
        if val is not None:
            worker_flags = f"{flag} {val} {worker_flags}".strip()
    return worker_flags


def asset_abort(queue: WorkQueue, log: Callable[[str], None]) -> int | None:
    """The driver's exit when a worker aborted the queue on a refused scene asset (exit 4 from the
    rollout CLI). The arm is not tallied and no arm_summary.json is written: seeds that
    were never run must not read as a scored arm, and a missing eval input is not a policy result."""
    reason = queue.aborted()
    if reason is None:
        return None
    log(f"ABORTED: a scene asset was refused, so this run is not scored and no arm_summary.json is "
        f"written. {reason}. Fix --asset-dir / the asset checksum manifest (ASSETS.lock.json) and re-run; the queue resumes.")
    return EXIT_ASSET_REFUSED


def declared_n_seeds(benchmark: str | None) -> int:
    """`--n-seeds` when not given: the benchmark declaration's, the same file the seeds come from."""
    from axis_qvla.bench.spec import BenchmarkError, load_benchmark  # noqa: PLC0415

    try:
        return load_benchmark(benchmark or DEFAULT_BENCHMARK).n_seeds
    except BenchmarkError as exc:
        raise SystemExit(f"--n-seeds not given and the benchmark declaration cannot be read: {exc}") from None


def _refuse_misfiled_arm(arm: str, seed_set: str) -> None:
    """The arm names the result directory, so it must say which seed set it holds
    (axis_qvla.bench.sim.arms.check_seed_set_arm): refused before anything is written."""
    from axis_qvla.bench.sim.arms import ArmError, check_seed_set_arm  # noqa: PLC0415

    try:
        check_seed_set_arm(arm, seed_set)
    except ArmError as exc:
        raise SystemExit(f"--run, --seed-set {seed_set}: {exc}") from None


def execution_guard(a, env_base: dict) -> dict | None:
    """The run's execution record, refused when the environment the workers inherit changes a
    constant the benchmark declares, unless --off-protocol. Fills in the declaration's two
    conventions first (what `bench eval` exports), so a direct call never executes the code
    default for an unset one. A tally executes nothing: no record, nothing to refuse."""
    from axis_qvla.bench import execution  # noqa: PLC0415
    from axis_qvla.bench.spec import BenchmarkError, load_benchmark  # noqa: PLC0415

    if a.tally_only:
        return None
    try:
        b = load_benchmark(a.benchmark or DEFAULT_BENCHMARK)
    except BenchmarkError:
        b = None        # benchmark_identity has already recorded why
    if b is not None and b.execution:
        for var, value in execution.convention_env(b).items():
            env_base.setdefault(var, value)
    try:
        record = execution.run_record(env_base, b, off_protocol=a.off_protocol)
    except ValueError as exc:      # a malformed number: nothing could say what would execute
        raise SystemExit(f"REFUSED before any server starts: {exc}") from None
    if record["deviations"] and not a.off_protocol:
        raise SystemExit("REFUSED before any server starts: " + execution.refusal(record["deviations"], env_base))
    if record["deviations"]:
        log("OFF-PROTOCOL run (--off-protocol), recorded in its run_manifest.json: "
            + "; ".join(f"{k} {v['executed']!r} (declared {v['declared']!r})"
                        for k, v in sorted(record["deviations"].items())))
    return record


# --- main -------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", dest="arm", required=True, metavar="RUN",
                    help="the name the results are filed under (out_<run>/; a development read ends in _dev)")
    ap.add_argument("--config", default=None, help="openpi train config name (required unless --policy)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint directory, digested into the run record, or a reference the server "
                         "loads by name (<org>/<name>, a URL), recorded as that exact string; a revision is "
                         "pinned by a local copy at it, since no reference server loads <org>/<name>@<revision> "
                         "(required unless --policy; with --policy an optional record of what the server serves)")
    ap.add_argument("--policy", action="append", default=[], metavar="HOST:PORT",
                    help="attach to a running policy server instead of starting one (repeatable; workers "
                         "are spread over them). --serving/--backend do not apply.")
    ap.add_argument("--ready-timeout-s", type=float, default=600.0,
                    help="--policy only: seconds to wait for each running server's handshake (default "
                         "600); a reference server this driver starts gets 600")
    ap.add_argument("--tasks", required=True, help="comma list, or a .json file of task ids")
    ap.add_argument(
        "--run-root", type=Path, default=None,
        help="campaign directory; --queue-root/--out-dir/--work-dir default to "
             "<run-root>/{q,out,work}_<run>. Lets a caller such as `bench eval` give one host "
             "root plus the run name instead of three composed paths.",
    )
    ap.add_argument("--queue-root", type=Path, default=None)
    ap.add_argument("--log-root", type=Path, required=True,
                    help="where the per-seed files are written AND read: handed to every worker "
                         "and on to every rollout (<log-root>/<served run>/task_<id>/), then tallied")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="the run's records: run_manifest.json (a real run, before the first server "
                         "starts), tally_manifest.json (--tally-only), dry_run_manifest.json "
                         "(--dry-run), arm_summary.json at the end, and the server and worker logs")
    ap.add_argument(
        "--layout-manifest", type=Path, default=None,
        help="prepended to the worker flags as '--layout-manifest <path>'; a separate flag so a "
             "spec can resolve it against a host root (param_paths) instead of splicing a "
             "path into a flag string.",
    )
    ap.add_argument("--asset-dir", type=Path, default=None,
                    help="prepended to the worker flags: the local qvla-v1-bench assets/ directory")
    ap.add_argument("--asset-checksums", dest="asset_lock", metavar="FILE", type=Path, default=None,
                    help="prepended to the worker flags: the asset checksum manifest (ASSETS.lock.json), "
                         "checked on every opened file")
    ap.add_argument("--render-root", type=Path, default=None,
                    help="prepended to the worker flags: the axis-render checkout (default: the rollout CLI's)")
    ap.add_argument("--task-data-root", type=Path, default=None,
                    help="prepended to the worker flags: the qvla-v1-bench tasks/ directory")
    ap.add_argument(
        "--work-dir", type=Path, default=None,
        help="ROOT for scene materialisation (default <run-root>/work_<run>; the rollout CLI requires "
        "one). The worker appends /<worker_id>/task_<id>, because workers sharing one work dir race "
        "on derived MJCF includes and a reader sees a half-written file.",
    )
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="first N declared seeds per task (default: the benchmark declaration's "
                         "n_seeds); handed to every worker and rollout")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--batch", type=int, default=10,
                    help="units a worker claims at once and runs serially; batch x the slowest "
                         "rollout must stay under --stale-after-s (SmolVLA: 3)")
    ap.add_argument("--stale-after-s", type=float, default=1800.0,
                    help="a claim older than this is re-issued to another worker (workqueue "
                         "staleness window); raise it instead of shrinking --batch if imports "
                         "dominate")
    ap.add_argument("--base-port", type=int, default=34000)
    ap.add_argument("--mem-fraction", default="0.5")
    ap.add_argument(
        "--serving", choices=SERVINGS, default=None,
        help="the protocol, ONE word: 'bare' serves the checkpoint as-is; 'conditioned' tags "
             "every prompt with quality tag 5 on the server and passes --cfg-conditioning to "
             "the client (results land under <run>_w<G>_dualsim).",
    )
    ap.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND,
                    help="which policy server --serving derives: " +
                         "; ".join(f"{k}: {s.label}" for k, s in REFERENCE_SERVERS.items()))
    ap.add_argument("--guidance-scale", type=float, default=0.0,
                    help="CFG guidance weight under --serving=conditioned (openpi only)")
    ap.add_argument("--benchmark", default=None,
                    help="tracked benchmark declaration, as <version>/<name> or <name> "
                         "(conf/benchmarks/<version>/<name>.json) that seeds and the identity "
                         "stamp come from; default franka_heldout24 of axis-heldout-v1")
    ap.add_argument("--seed-set", choices=SEED_SETS, default="test",
                    help="run the declared test seeds (default) or the development seeds; a dev run's "
                         "--run must end in _dev and a test run's must not")
    ap.add_argument("--off-protocol", action="store_true",
                    help="run although the environment changes an execution constant the benchmark "
                         "declares; the deviation is recorded in run_manifest.json")
    ap.add_argument("--xla-pin", default="--xla_gpu_autotune_level=0",
                    help="XLA_FLAGS for BOTH servers and workers; '-' to leave unset (NOT advised: "
                         "unpinned autotuning is the measured cause of eval irreproducibility)")
    ap.add_argument("--openpi-dir", default="openpi")
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--worker-python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true",
                    help="write the queue and dry_run_manifest.json (no version or GPU probes) and "
                         "start no server or worker")
    ap.add_argument("--tally-only", action="store_true",
                    help="start nothing; rebuild meta, fingerprint and arm_summary.json from the "
                         "result files already on disk, recorded in tally_manifest.json beside the run's own "
                         "run_manifest.json")
    a = ap.parse_args(argv)
    try:
        check_arm_name(a.arm)
    except ArmError as exc:
        ap.error(str(exc))
    if a.run_root is not None:
        a.queue_root = a.queue_root or a.run_root / f"q_{a.arm}"
        a.out_dir = a.out_dir or a.run_root / out_dir_name(a.arm)
        a.work_dir = a.work_dir or a.run_root / f"work_{a.arm}"
    if a.queue_root is None or a.out_dir is None:
        ap.error("--queue-root and --out-dir are required unless --run-root is given")
    # Absolute once, here: the workers and their rollouts receive this exact string.
    a.log_root = a.log_root.absolute()
    a.checkpoint = local_checkpoint(a.checkpoint)
    if a.policy:
        proto = external_protocol(a)
    else:
        if a.config is None or a.checkpoint is None:
            ap.error("--config and --checkpoint are required unless --policy names a running server")
        proto = resolve_protocol(a)
        if not a.tally_only:
            problem = config_file_problem(a.config)
            if problem:
                raise SystemExit(problem)
    a.serve_flags, a.worker_flags, a.server_cmd = proto.serve_flags, proto.worker_flags, proto.server_cmd
    a.backend, a.guidance_scale = proto.backend, proto.guidance_scale
    _refuse_misfiled_arm(a.arm, a.seed_set)
    # The environment every server and worker inherits, checked against the declaration before
    # anything is written: a refused run leaves no directory, queue or record behind.
    env_base = dict(os.environ)
    if a.xla_pin != "-":
        env_base["XLA_FLAGS"] = a.xla_pin
    env_base["OPENPI_DIR"] = a.openpi_dir
    if a.server_python:
        env_base["SERVER_PYTHON"] = a.server_python
    a.execution_record = execution_guard(a, env_base)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.layout_manifest_declared, a.layout_manifest_bytes = a.layout_manifest, None
    if a.layout_manifest is not None:
        # Written after the resume check below: a refused resume must leave the earlier run's copy.
        a.layout_manifest, a.layout_manifest_bytes = harness_layout(a.layout_manifest, a.out_dir)
        a.worker_flags = f"--layout-manifest {a.layout_manifest} {a.worker_flags}".strip()
    a.worker_flags = with_asset_flags(a.worker_flags, a.asset_dir, a.asset_lock)
    a.worker_flags = with_root_flags(a.worker_flags, a.render_root, a.task_data_root)
    if a.n_seeds is None:
        a.n_seeds = declared_n_seeds(a.benchmark)

    task_ids = load_task_ids(a.tasks)
    bench = benchmark_identity(task_ids, a.benchmark)
    a.benchmark_identity = bench
    if not bench["matches_manifest"]:
        log(f"WARNING task list is NOT the tracked benchmark {bench.get('name')}: "
            f"{len(task_ids)} tasks {task_ids}; the summary will say so")
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]

    units = build_units(task_ids, a.n_seeds,
                        lambda tid: default_seed_source(tid, a.benchmark, a.seed_set), a.seed_set)
    if a.policy:
        if a.tally_only:
            a.external_handshakes = recorded_handshakes(a.out_dir)
        elif not a.dry_run:
            a.external_handshakes = external_handshakes(a.endpoints, timeout_s=a.ready_timeout_s)
    manifest = run_manifest(a, env_base, units, probe=not a.dry_run)
    if not a.tally_only and not a.dry_run:
        # Before the queue is touched or run_manifest.json is overwritten: a resumed run that
        # would run another model, config, serving, server or execution must not reuse either.
        # Needs the fresh manifest (the checkpoint digest), so it runs here -- still before the
        # write below and before queue.initialize.
        refuse_resume_change(a.out_dir, manifest, a.external_handshakes if a.policy else None,
                             queue_started=WorkQueue(a.queue_root, stale_after_s=a.stale_after_s).started())
    if a.layout_manifest_bytes is not None:
        a.layout_manifest.write_bytes(a.layout_manifest_bytes)
    # run_manifest.json is the record of the run that writes the seed files, so only a real run
    # writes it. A tally describes files an EARLIER run wrote and a dry run starts nothing (no
    # probes); each records beside that run's manifest, never over it (the summary embeds both).
    record = ("tally_manifest.json" if a.tally_only else
              "dry_run_manifest.json" if a.dry_run else "run_manifest.json")
    (a.out_dir / record).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    advice = _pin_advice(manifest["checkpoint"])
    if advice:
        log(f"WARNING {advice}")
    ck = manifest["checkpoint"]
    named = (f"sha256 {ck['sha256'][:16]} ({ck.get('n_files', '?')} files, {manifest['checkpoint_digest_s']} s)"
             if ck.get("sha256") else f"reference {ck['ref']}" if ck.get("ref") else "not identified")
    log(f"{record}: checkpoint {named}; repo {manifest['sources']['repo'].get('head', '?')[:9]}"
        f"{' DIRTY' if manifest['sources']['repo'].get('dirty') else ''}")
    if a.tally_only:
        # The queue's ABORTED marker outranks the result files: an arm a refused asset stopped
        # part-way must not get arm_summary.json -- a run with a summary is complete and is not
        # re-run. Left in place.
        aborted = asset_abort(WorkQueue(a.queue_root, stale_after_s=a.stale_after_s), log)
        if aborted is not None:
            log("tally-only refused: the queue is ABORTED, so this partial run is not summarised")
            return aborted
        log("tally-only: no servers or workers; summarising the result files on disk")
    else:
        queue = WorkQueue(a.queue_root, stale_after_s=a.stale_after_s)
        queue.initialize(units)
        prev_abort = queue.clear_abort()
        if prev_abort is not None:
            log(f"previous run of this queue was aborted ({prev_abort}); cleared, resuming")
        log(f"run={a.arm} tasks={len(task_ids)} units={len(units)} gpus={len(gpus)} "
            f"workers/gpu={a.workers_per_gpu} XLA_FLAGS={env_base.get('XLA_FLAGS', '<unset>')}")
        if a.dry_run:
            log("dry run: queue written, no processes started")
            return 0

        servers: list[subprocess.Popen] = []
        workers: list[subprocess.Popen] = []
        try:
            if not a.policy:
                # openpi's server (no server_cmd) may need to download its tokenizer at start.
                asset = None if a.server_cmd else openpi_asset_problem(env_base)
                if asset:
                    log(f"NOTE {asset}")
                for i, gpu in enumerate(gpus):
                    servers.append(start_server(gpu, a.base_port + i, a.config, a.checkpoint,
                                                a.serve_flags.split(), a.out_dir, env_base,
                                                a.mem_fraction, a.server_cmd))
                for i, gpu in enumerate(gpus):
                    try:
                        md = wait_ready("127.0.0.1", a.base_port + i, 600.0, require_noise=True,
                                        proc=servers[i])
                    except SystemExit as exc:
                        if not asset:
                            raise
                        raise SystemExit(f"{exc}; {asset}") from None
                    log(f"gpu{gpu} ready: horizon={md.get('action_horizon')} dim={md.get('action_dim')}")
            eps = a.endpoints if a.policy else [("127.0.0.1", a.base_port + i) for i in range(len(gpus))]

            for i, gpu in enumerate(gpus):
                for w in range(a.workers_per_gpu):
                    # The SERVER index is part of the id: --gpus may name one GPU twice (two servers
                    # on a GPU whose single server is CPU-bound), and a worker id that were only
                    # (gpu, w) would then be shared by two processes -- one work dir (the ParseError
                    # race) and one queue owner (one could complete the other's units).
                    wid = f"gpu{gpu}s{i}w{w}"
                    host_w, port_w = eps[(i * a.workers_per_gpu + w) % len(eps)] if a.policy else eps[i]
                    env = {
                        **worker_env_base(env_base),
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "MUJOCO_EGL_DEVICE_ID": str(gpu),  # MUST equal CUDA_VISIBLE_DEVICES
                    }
                    cmd = [
                        a.worker_python, "-m", "axis_qvla.bench.sim.queue_worker",
                        "--queue-root", str(a.queue_root), "--worker-id", wid,
                        "--host", host_w, "--port", str(port_w), "--variant", a.arm,
                        # --checkpoint-dir is provenance only on the rollout side (axis_qvla.bench.sim.cli);
                        # under --policy with no --checkpoint, a.checkpoint is None, which Popen
                        # cannot take as an argv element -- hand on a provenance STRING instead.
                        "--checkpoint-dir", a.checkpoint or "external:" + ",".join(a.policy),
                        "--batch", str(a.batch),
                        "--stale-after-s", str(a.stale_after_s),
                        "--summary", str(a.out_dir / f"worker_{wid}.json"),
                        # What the tally below assumes, handed on rather than re-derived: the
                        # rollout writes under THIS root, resolves prompts and seeds from THIS
                        # declaration, and accepts THIS many of its seeds.
                        "--log-root", str(a.log_root), "--n-seeds", str(a.n_seeds),
                        "--benchmark", a.benchmark or DEFAULT_BENCHMARK,
                        # The worker hands it to every rollout when it is not the default.
                        "--seed-set", a.seed_set,
                    ]
                    if a.work_dir is not None:
                        cmd += ["--work-dir", str(a.work_dir)]
                    if a.worker_flags:
                        # As ONE argv element: argparse would read its option-looking tokens
                        # (--cfg-conditioning) as the worker's own.
                        cmd += ["--rollout-args", a.worker_flags]
                    lf = (a.out_dir / f"worker_{wid}.log").open("w")
                    workers.append(subprocess.Popen(cmd, env=env, stdout=lf,
                                                    stderr=subprocess.STDOUT))
            log(f"{len(workers)} workers started")

            # Wait on PIDs, never on a ps|grep pattern this driver's own argv could contain.
            rc = 0
            for p in workers:
                r = p.wait()
                if r != 0:
                    rc = r
                    log(f"worker pid {p.pid} exited {r}")
            done, in_flight, total = queue.progress()
            dups = queue.duplicate_completions()
            log(f"queue drained: {done}/{total} done, {in_flight} in flight, worker rc={rc}, "
                f"{dups} duplicate completions (re-issued by the {a.stale_after_s:.0f} s "
                f"staleness window; 0 is healthy)")
            aborted = asset_abort(queue, log)
            if aborted is not None:
                return aborted   # the finally below still terminates every server
        finally:
            for p in servers:
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)  # by pid, never by pattern
            for p in servers:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
            log("servers terminated")

    served = (external_served(a.arm, a.external_handshakes) if a.policy
              else result_dir_name(a.arm, a.worker_flags))
    meta = recompute_meta(a.log_root, served, task_ids, a.n_seeds)
    fp = fingerprint(a.log_root, served, task_ids)
    tot = sum(v["n_success"] for v in meta["per_task"].values())
    n = sum(v["n_seeds"] for v in meta["per_task"].values())
    rate = 100.0 * sum(
        v["n_success"] / v["n_seeds"] for v in meta["per_task"].values() if v["n_seeds"]
    ) / max(len(meta["per_task"]), 1)
    summary = {
        "arm": a.arm, "served_arm": served, "log_root": str(a.log_root),
        "checkpoint": a.checkpoint, "config": a.config,
        "xla_flags": env_base.get("XLA_FLAGS", "<unset>"),
        "n_tasks": len(task_ids), "n_seeds": a.n_seeds, "seed_set": a.seed_set,
        "successes": tot, "rollouts": n, "all24_pct": round(rate, 2),
        "incomplete_tasks": meta["incomplete"], "errored_rollouts": meta.get("errored", 0),
        "eval_fingerprint": fp,
        "benchmark": bench,
        # The serving protocol, so a bare and a conditioned read of one checkpoint differ here
        # and not only in the number.
        "serving": proto.serving, "backend": proto.backend,
        "guidance_scale": proto.guidance_scale,
        "serve_flags": a.serve_flags, "worker_flags": a.worker_flags,
        "server_cmd": a.server_cmd,
        "external": list(a.policy) or None,
        # The run that produced the files; under --tally-only that is the earlier run's record.
        "provenance": recorded_run_manifest(a.out_dir) if a.tally_only else manifest,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if a.tally_only:
        summary["tally_provenance"] = manifest
    if n == 0:
        # NEVER persist an empty arm under the real summary name. arm_summary.json marks a run
        # complete, so an empty one (sha256 of nothing, e3b0c442...) would make a failed arm look
        # done and be silently skipped on every later pass.
        (a.out_dir / FAILED_SUMMARY_FILE).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        log("REFUSING to write arm_summary.json: 0 rollouts -- recorded as arm_summary.FAILED.json")
        return 3
    (a.out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(json.dumps(summary))
    if meta.get("errored"):
        log(f"REFUSING: {meta['errored']}/{n} rollouts carry an error: reason -- this is a broken "
            f"harness, not a weak run. Do not report this number.")
        return 2
    if meta["incomplete"]:
        log(f"WARNING incomplete tasks: {meta['incomplete']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
