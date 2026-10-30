"""The `bench` command group: verify a benchmark declaration, evaluate ONE checkpoint, read scores.

Core tier: stdlib only at module scope; axis_qvla.bench.launch/spec/score and the campaign driver are
imported inside the command functions. Every message that names a command takes the program from
the parser it was registered under (`args.bench_prog`).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from axis_qvla.bench.backends import BACKENDS, DEFAULT_BACKEND, REFERENCE_SERVERS
from axis_qvla.core.hosts import HostProfileError, available_hosts

if TYPE_CHECKING:
    from axis_qvla.bench.launch import EvalRequest, Launch
    from axis_qvla.core.hosts import HostProfile


def _cmd_bench_check(args: argparse.Namespace) -> int:
    from axis_qvla.bench import versions
    from axis_qvla.bench.spec import DEFAULT_NAME, BenchmarkError, check, load_benchmark

    try:
        b = load_benchmark(args.benchmark or DEFAULT_NAME)
    except BenchmarkError as exc:
        print(f"FAIL  {exc}")
        return 1
    for k, v in b.identity().items():
        print(f"{k:16s} {v}")
    # "frozen" is check_version's result, not merely whether a VERSION.json once existed
    # (b.frozen_sha256): an EDITED version still has one, and must not print "yes".
    print(f"{'frozen':16s} {'yes' if not versions.check_version(b.conf_dir) else 'no'}")
    print(f"{'directory':16s} {b.conf_dir}")
    print(f"{'conventions':16s} action_space={b.action_space} image_geometry={b.image_geometry} "
          f"dual_sim={b.dual_sim} xla_pin={b.xla_pin or '-'}")
    if b.execution:
        print(f"{'execution':16s} " + " ".join(f"{k}={v}" for k, v in sorted(b.execution.items()) if k != "schema"))
    problems = check(b)
    for pr in problems:
        print(f"FAIL  {pr}")
    print(f"\n{'consistent' if not problems else f'{len(problems)} problem(s)'}: "
          f"{len(b.task_ids)} tasks x {b.n_seeds} seeds")
    if args.corpus_prompts:
        from axis_qvla.bench.tasks.leakage import instruction_overlap, load_corpus_prompts

        try:
            found = instruction_overlap(b, load_corpus_prompts(args.corpus_prompts))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"FAIL  --pretraining-prompts {args.corpus_prompts} cannot be read as a prompt list "
                  f"({type(exc).__name__}: {exc})")
            return 2
        for o in found:
            tasks = ", ".join(f"{t} ({n})" for t, n in o.corpus_tasks)
            print(f"{'instruction':16s} task {o.task_id} {o.instruction!r} is also pretraining task {tasks}")
        print(f"{'':16s} {len(found)} instructions, {sum(o.episodes for o in found)} pretraining episodes "
              f"(reported, not refused)")
    return 1 if problems else 0


def _policy_refusal(args: argparse.Namespace) -> str | None:
    """--policy (a server you started), a results-map checkpoint (--checkpoint-id) and the
    reference-server options are distinct ways to say what is evaluated; mixing them would record
    one and run another."""
    if args.from_run:
        from axis_qvla.bench import local_run

        try:
            local_run.check_run_id(args.from_run)
        except local_run.LocalRunError as exc:
            return str(exc)
        mixed = [flag for flag, v in (("--policy", args.policy), ("--checkpoint", args.checkpoint),
                                      ("--checkpoint-id", args.checkpoint_id), ("--config", args.config),
                                      ("--backend", args.backend)) if v not in (None, [])]
        if mixed:
            return (f"--from-run serves the checkpoint of a run trained here, on that run's backbone with its "
                    f"config; {', '.join(mixed)} would name another. Give one or the other.")
        if args.verify_checkpoint:
            return ("--verify-checkpoint compares a paper checkpoint with the pub_digest cells.json records for it; "
                    "a run trained here (--from-run) has no published digest")
        return None
    if args.pretrain_seed is not None and args.ft_seed is None:
        return "--pretrain-seed needs --ft-seed: a results file records the pretraining seed with the fine-tune seed"
    if args.policy:
        mixed = [flag for flag, v in (("--backend", args.backend), ("--serving", args.serving),
                                      ("--guidance-scale", args.guidance_scale), ("--config", args.config),
                                      ("--checkpoint-id", args.checkpoint_id))
                 if v is not None]
        if mixed:
            return (f"--policy evaluates a server you started yourself; {', '.join(mixed)} describe a "
                    f"reference server `eval` starts. Give one or the other.")
        if not args.arm:
            return "--policy needs --run: results are filed under it and no checkpoint names them"
        return None
    if args.checkpoint_id:
        from axis_qvla.bench import paper_checkpoint

        if args.config is not None:
            return ("--checkpoint-id reads the server's config from the paper run that trained the "
                    "checkpoint; do not also pass --config")
        try:
            want = paper_checkpoint.backend_of(args.checkpoint_id)
        except paper_checkpoint.PaperCheckpointError as exc:
            return str(exc)
        if args.backend is not None and args.backend != want:
            return (f"--checkpoint-id {args.checkpoint_id} is a {REFERENCE_SERVERS[want].short_label} checkpoint, "
                    f"served with --backend {want}; leave --backend out")
        return None
    if not (args.checkpoint and args.config):
        return ("--checkpoint and --config (or --checkpoint-id, or --from-run) are required unless --policy "
                "names a running server")
    return None


class _Refused(Exception):
    """Nothing can be served as asked; the message says why. `pending`: a --from-run run that may
    resolve later (not trained yet), for which --print still prints the command."""

    def __init__(self, message: str, *, pending: bool = False):
        super().__init__(message)
        self.pending = pending


@dataclasses.dataclass(frozen=True)
class _Target:
    """What `eval` serves, resolved once from the flags as given or a results-map checkpoint
    (--checkpoint-id). `serve_config` is the rendered openpi config's bytes; `fetch` is the `qvla
    data get` target of a local copy this command chose itself, checked to be there before anything
    starts. `serving` is set when the target fixes it (one mode of a --from-run run), else --serving
    applies."""
    checkpoint: str | None
    config: str | None
    backend: str
    arm: str | None
    checkpoint_id: str | None = None
    serve_config: bytes | None = None
    ft_seed: int | None = None
    pretrain_seed: int | None = None
    fetch: str | None = None
    serving: str | None = None


def _resolve_target(args: argparse.Namespace, host: HostProfile) -> _Target:
    """The one place the flags become what is served. A results-map checkpoint id gives the backend,
    the LeRobot policy config (pi0.5's is rendered from its paper run) and, without --checkpoint,
    the local copy `qvla data get checkpoints/<id>` made -- the verified path, the only one
    `eval` reads a defaulted checkpoint from."""
    from axis_qvla.bench import openpi_serve, paper_checkpoint
    from axis_qvla.core.paths import Paths

    seeds = {"ft_seed": args.ft_seed, "pretrain_seed": args.pretrain_seed}
    try:
        if args.checkpoint_id:
            cid = args.checkpoint_id
            backend = paper_checkpoint.backend_of(cid)
            # The results map's record is read whether or not --checkpoint names the directory: a
            # LeRobot checkpoint of the map is served under the policy config its eval_command records.
            # Only an id outside the map is let through, and only with --checkpoint.
            # A results map that cannot be read is reported, never read as "not in the map".
            try:
                found = paper_checkpoint.lookup(cid)
            except paper_checkpoint.NotInResultsMap:
                if args.checkpoint is None:
                    raise
                found = None
            checkpoint = args.checkpoint or str(paper_checkpoint.local_dir(found, Paths(host)))
            config = None if backend == DEFAULT_BACKEND else (found.config if found else backend)
            data = openpi_serve.render_serve_config(cid, Paths(host)) if backend == DEFAULT_BACKEND else None
            arm = args.arm or paper_checkpoint.default_run(cid, args.serving or "bare")
            return _Target(checkpoint, config, backend, arm, cid, data, **seeds,
                           fetch=f"checkpoints/{cid}" if args.checkpoint is None else None)
        backend = args.backend or DEFAULT_BACKEND
        # The openpi config a paper checkpoint was trained under, as a frozen cells.json eval_command
        # spells it, is served as that checkpoint's --checkpoint-id (an undocumented alias T0 pins).
        if not args.policy and backend == DEFAULT_BACKEND and args.config:
            cid = openpi_serve.checkpoint_for_config(args.config, args.arm)
            if cid:
                return _Target(args.checkpoint, args.config, backend, args.arm, cid,
                               openpi_serve.render_serve_config(cid, Paths(host)), **seeds)
        return _Target(args.checkpoint, args.config, backend, args.arm, **seeds)
    except (openpi_serve.ServeConfigError, paper_checkpoint.PaperCheckpointError) as exc:
        raise _Refused(str(exc)) from None


def _from_run_targets(args: argparse.Namespace, host: HostProfile) -> list[_Target]:
    """`--from-run`: one target per serving mode the run records (bare first), or the one --serving
    narrows to, each filed under its own run (axis_qvla.bench.local_run). The checkpoint, the backend,
    the server's config and the seeds it records are the run's own; --ft-seed/--pretrain-seed only
    fill a seed it does not record. Under --print nothing is written into the run's directory."""
    from axis_qvla.bench import local_run
    from axis_qvla.core.paths import Paths

    try:
        lr = local_run.resolve(args.from_run, Paths(host), write=not args.print_only)
        modes = local_run.select(lr, args.serving)
        ft_seed, pretrain_seed = local_run.seeds(lr, args.ft_seed, args.pretrain_seed)
    except local_run.LocalRunError as exc:
        raise _Refused(str(exc), pending=exc.pending) from None
    if args.guidance_scale and not any(m.serving == "conditioned" for m in modes):
        raise _Refused(f"--guidance-scale weighs conditioned serving; --from-run {lr.id} is evaluated "
                       f"{' and '.join(m.name for m in modes)} only")
    return [_Target(str(lr.checkpoint), lr.config, lr.backend, lr.arm(m, args.arm), None, lr.serve_config,
                    ft_seed=ft_seed, pretrain_seed=pretrain_seed, serving=m.serving) for m in modes]


def _resolve_targets(args: argparse.Namespace, host: HostProfile) -> list[_Target]:
    """Everything one `eval` evaluates, in order: one target, or one per mode of a --from-run run."""
    return _from_run_targets(args, host) if args.from_run else [_resolve_target(args, host)]


def _serve_target(req: EvalRequest, data: bytes, host: HostProfile) -> tuple[Path, bytes, Path]:
    """A config openpi serves: where it goes (beside the run manifest of the run `req` is filed
    under), its bytes, and the run's out directory (where that run manifest is written)."""
    from axis_qvla.bench import openpi_serve
    from axis_qvla.bench.launch import filed_arm, run_root
    from axis_qvla.bench.sim.arms import out_dir_name

    arm = filed_arm(req)
    return openpi_serve.config_path(run_root(host), arm), data, run_root(host) / out_dir_name(arm)


def _config_differences(old: bytes, new: bytes) -> list[str]:
    """The dotted keys whose values differ between two rendered configs."""
    import json

    from axis_qvla.train.parity import flatten

    try:
        a, b = flatten(json.loads(old)), flatten(json.loads(new))
    except (ValueError, AttributeError):
        return ["(the recorded file is not a JSON config)"]
    gone = object()
    return sorted(k for k in a.keys() | b.keys() if a.get(k, gone) != b.get(k, gone))


def _run_recorded(out_dir: Path) -> bool:
    """Whether a run of the arm started: the driver writes run_manifest.json before the first
    server. A directory that cannot be checked counts as recorded, the cautious answer."""
    try:
        return (out_dir / "run_manifest.json").exists()
    except OSError:
        return True


def _write_serve_config(path: Path, data: bytes, out_dir: Path) -> str | None:
    """Write the server's config, or say why not. It is written before the dry run lists it or the
    driver is handed it, so the argv a dry run lists names a file that is there. A file already there
    with other bytes is the config recorded for this arm (by a run, a dry run or a refused start):
    replacing it could file rollouts served with two configs under one arm, so it is refused,
    naming the keys that differ and, when no run of the arm started, the file to delete. The same
    bytes are a resume and are kept as they are."""
    from axis_qvla.core.canon import sha256_bytes

    try:
        old = path.read_bytes()
    except FileNotFoundError:
        old = None
    except OSError as exc:
        return f"{path} cannot be read ({type(exc).__name__}: {exc}); refused before any server starts"
    if old is not None:
        if old == data:
            return None
        keys = _config_differences(old, data)
        head = (f"{path} already records another config for this run (sha256 {sha256_bytes(old)[:12]}; this "
                f"command renders {sha256_bytes(data)[:12]}; they differ in {', '.join(keys[:8])}"
                f"{', ...' if len(keys) > 8 else ''}).")
        manifest = out_dir / "run_manifest.json"
        if not _run_recorded(out_dir):
            return (f"{head} No run under this name started ({manifest} does not exist), so a dry run or a refused "
                    f"start wrote it: delete {path} and run again.")
        return (f"{head} A run under this name recorded it ({manifest}), and its rollouts were served with the "
                f"recorded config. Use a new --run: the driver resumes a run only with the config bytes it "
                f"recorded, so a config rendered under a moved [roots] entry is another config too.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as exc:
        return (f"the server's config cannot be written to {path} ({type(exc).__name__}: {exc}); the "
                f"commands this run would print and start name it. Refused before any server starts")
    return None


def _execution_refusal(args: argparse.Namespace, b, run_env: dict) -> str | None:
    """Why the run must not start under `run_env` (the caller's environment with the composed one
    over it: what the driver hands every worker), or None. A benchmark with no execution.json
    declares nothing to hold a run to; a deviation from it needs --off-protocol. A tally executes
    no rollout, so no environment can be a deviation of it."""
    from axis_qvla.bench import execution

    if execution.declared(b) is None:
        return f"benchmark {b.name!r} declares no execution constants ({b.conf_dir}/execution.json)"
    if args.tally_only:
        return None
    try:
        dev = execution.run_record(run_env, b, off_protocol=False)["deviations"]
    except ValueError as exc:      # a malformed number: nothing could say what would execute
        return str(exc)
    if args.off_protocol:
        return None
    return execution.refusal(dev, run_env) if dev else None


def _digest_root(target: _Target) -> Path | None:
    """The directory cells.json's `pub_digest` is defined over for what `target` serves, or None when
    that cannot be said. pi0.5: the checkpoint directory served. SmolVLA/X-VLA: the checkpoint step
    directory (pub_digest leaves its training_state/ out), whose pretrained_model/ the server loads --
    the parent of a defaulted copy, or of a --checkpoint named pretrained_model."""
    from axis_qvla.bench.paper_checkpoint import LEROBOT_MODEL_DIR

    ckpt = Path(str(target.checkpoint))
    if target.backend == DEFAULT_BACKEND:
        return ckpt
    return ckpt.parent if target.fetch or ckpt.name == LEROBOT_MODEL_DIR else None


def _is_dir(path: Path) -> bool:
    """Whether `path` is a directory; absent is False, any other OSError (EACCES) is raised."""
    try:
        return path.stat().st_mode & 0o170000 == 0o040000
    except (FileNotFoundError, NotADirectoryError):
        return False


def _local_copy_refusal(prog: str, target: _Target) -> str | None:
    """A local copy `eval` chose itself (`target.fetch`) is served only from where `qvla data get`
    puts it, and only once it is all there: present is not complete (an interrupted fetch leaves a
    partial directory), so the count and summed size of its published files are compared with the
    pub_digest cells.json records. Stat-based -- re-hashing every byte is --verify-checkpoint's job --
    and an unreadable entry is reported, never read as absent."""
    from axis_qvla.bench import checkpoint_digest
    from axis_qvla.paper.cli import DEFAULT_CELLS

    get = f"`{prog} data get {target.fetch}`"
    served, root = Path(str(target.checkpoint)), _digest_root(target)
    try:
        there = _is_dir(served) and _is_dir(root)
    except OSError as exc:
        return f"the local copy {served} cannot be read ({type(exc).__name__}: {exc}). Refused before any server starts."
    if not there:
        return f"the local copy {served} is not there: {get} fetches it, verified. Refused before any server starts."
    cid = str(target.checkpoint_id)
    try:
        want = checkpoint_digest.recorded(cid, DEFAULT_CELLS.read_bytes())
        files = checkpoint_digest.published_files(root, checkpoint_digest.family(cid))
        size = sum((root / rel).stat().st_size for rel in files)
    except (checkpoint_digest.CheckpointDigestError, OSError) as exc:
        return (f"the local copy {root} cannot be checked complete ({type(exc).__name__}: {exc}). "
                f"Refused before any server starts.")
    if (len(files), size) != (want.get("n_files"), want.get("bytes")):
        return (f"the local copy {root} is incomplete: its published files are {len(files)} files, {size} bytes; "
                f"cells.json records {want.get('n_files')} files, {want.get('bytes')} bytes for {cid}. {get} "
                f"fetches the rest, verified (an interrupted fetch resumes). Refused before any server starts.")
    return None


def _checkpoint_refusal(prog: str, target: _Target) -> str | None:
    """`--verify-checkpoint`: why what is served is not the published checkpoint the id names, or
    None. Compared at `_digest_root`, the directory pub_digest is defined over, for every backbone; a
    defaulted copy is first checked to be there and complete, so an absent one names its `data get`."""
    from axis_qvla.bench import checkpoint_digest
    from axis_qvla.paper.cli import DEFAULT_CELLS

    if not target.checkpoint_id:
        return ("--verify-checkpoint compares a paper checkpoint with the pub_digest cells.json records for it: "
                "give --checkpoint-id <cells.json checkpoint.id> (or the paper --config of a pi0.5 eval_command)")
    if not target.checkpoint:
        return "--verify-checkpoint needs --checkpoint, the local copy of the checkpoint to compare"
    if target.fetch:
        refused = _local_copy_refusal(prog, target)
        if refused:
            return refused
    root = _digest_root(target)
    if root is None:
        return (f"--verify-checkpoint: --checkpoint {target.checkpoint} is not a pretrained_model/ directory, so the "
                f"{target.checkpoint_id} step directory that cells.json's pub_digest covers is unknown. Pass "
                f"--checkpoint <step directory>/pretrained_model, or leave --checkpoint out to serve the "
                f"`{prog} data get checkpoints/{target.checkpoint_id}` copy")
    # A defaulted copy is named as what it is, not as a --checkpoint the user did not give, and is
    # fetched again rather than served as another run.
    label = f"the local copy {root}" if target.fetch else None
    try:
        found = checkpoint_digest.mismatch(root, target.checkpoint_id, DEFAULT_CELLS.read_bytes(), label=label)
    except (checkpoint_digest.CheckpointDigestError, OSError) as exc:
        return f"--verify-checkpoint: {exc}"
    if found and target.fetch:
        found += f" `{prog} data get {target.fetch}` fetches it again, verified"
    return found


def _option_table(parser: argparse.ArgumentParser) -> tuple[tuple[str, str, str, object], ...]:
    """(dest, long flag, kind, default) of every option `parser` declares, kind one of flag,
    append, value: what `_eval_command` renders an invocation back from."""
    out = []
    for act in parser._actions:
        if not act.option_strings or act.dest == "help":
            continue
        kind = "flag" if act.nargs == 0 else "append" if isinstance(act, argparse._AppendAction) else "value"
        out.append((act.dest, max(act.option_strings, key=len), kind, act.default))
    return tuple(out)


def _eval_tokens(args: argparse.Namespace) -> list[str]:
    """This `bench eval`'s arguments as typed (less --dry-run and --print): what `_eval_command`
    prints and `_reexec` runs."""
    toks: list[str] = []
    for dest, flag, kind, default in args.eval_options:
        value = getattr(args, dest, default)
        if dest in ("dry_run", "print_only") or value == default:
            continue
        if kind == "flag":
            toks.append(flag)
        else:
            toks += [t for v in (value if kind == "append" else [value]) for t in (flag, str(v))]
    return toks


def _eval_command(args: argparse.Namespace, python: Path) -> str:
    """The command that runs this `bench eval` under `python`: `bench eval` itself with the same
    arguments, never the driver it composes. Every check this command makes -- the declaration, the
    asset checksum manifest and texture bank, the roots, --verify-checkpoint, the execution
    constants, the recorded server config -- then runs again on the command printed."""
    import shlex

    from axis_qvla.bench.launch import REPO_ROOT

    return (f"PYTHONPATH={shlex.quote(str(REPO_ROOT))} {shlex.quote(str(python))} -m axis_qvla.cli bench eval "
            + " ".join(shlex.quote(t) for t in _eval_tokens(args)))


def _root_prog(args: argparse.Namespace) -> str:
    """The program `bench` was registered under (`qvla`), for the commands a refusal names."""
    return args.bench_prog.rsplit(" ", 1)[0]


# Set in the env execve hands the child, so a child that is STILL not running under `launch.python`
# (its venv's own `bin/python` is a wrapper that execs some other interpreter) is told that, once,
# rather than re-exec forever: _cmd_bench_eval checks this before calling _reexec a second time.
REEXEC_MARKER = "QVLA_EVAL_REEXEC"


def _reexec(python: Path, args: argparse.Namespace, command: str = "eval") -> int:
    """Replace this process with this same `bench <command>` under `python`, the host's evaluation
    environment, so the command a user types runs where it must and makes every check again there.
    Nothing has been started yet, so nothing is left behind; os.execve does not return. Flushed before
    the call: once execve replaces this process, anything buffered in this interpreter is gone."""
    from axis_qvla.bench.launch import REPO_ROOT

    argv = [str(python), "-m", "axis_qvla.cli", "bench", command, *_eval_tokens(args)]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), REEXEC_MARKER: "1"}
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.execve(str(python), argv, env)
    except OSError as exc:
        print(f"FAIL  cannot re-execute under {python}: {exc}")
        return 2
    return 0   # reached only where os.execve is replaced (the tests)


@dataclasses.dataclass(frozen=True)
class _Plan:
    """One evaluation `eval` runs: what is served, the composed driver call, and the server config
    to write first (path, bytes, the run's out directory), when the backend reads one."""
    target: _Target
    req: EvalRequest
    launch: Launch
    serve_config: tuple[Path, bytes, Path] | None


def _print_plan(args: argparse.Namespace, host: HostProfile, plans: list[_Plan], asset_dir: Path | None,
                task_root: Path | None) -> None:
    """`--print`: what the run will fetch, read, serve and write, as `# ` comment lines, then the one
    runnable command (which runs every evaluation listed). Needs only the host profile -- no GPU, no
    network, no data -- so a batch script (and the docs' CI) can take it anywhere. The serve config
    is rendered by the run itself, not here."""
    from axis_qvla.bench import finish
    from axis_qvla.bench.launch import run_root

    for p in plans:
        if p.target.fetch:
            print(f"# fetch first: {_root_prog(args)} data get {p.target.fetch}")
    if asset_dir is not None:
        print(f"# inputs: {asset_dir} and {task_root} (`{_root_prog(args)} data get bench` fills them)")
    if args.from_run:
        t = plans[0].target
        print(f"# from run: {args.from_run} (fine-tune seed {t.ft_seed}, pretraining seed {t.pretrain_seed})")
    for p in plans:
        if args.policy:
            print(f"# evaluates: the running server(s) {', '.join(args.policy)}")
        else:
            config = p.serve_config[0] if p.serve_config else p.target.config
            how = f", served {p.target.serving} as run {p.launch.arm}" if args.from_run else ""
            print(f"# serves: --backend {p.target.backend} {p.target.checkpoint} with config {config}{how}")
        print(f"# results file: {finish.results_path(run_root(host), p.launch.arm)}")
    print(_eval_command(args, plans[0].launch.python))


def _print_unresolved(args: argparse.Namespace, host: HostProfile, why: str) -> int:
    """`--print` of a --from-run run that does not resolve yet (not trained, or not finished): it
    needs no data, so the command that will evaluate the run is printed all the same. The run names
    are the modes its eval.json records, or read from the run id while it has no directory
    (axis_qvla.bench.local_run.expected_arms)."""
    from axis_qvla.bench import finish, local_run
    from axis_qvla.bench.launch import LaunchError, eval_python, run_root
    from axis_qvla.bench.sim.arms import seed_set_arm

    try:
        python, root = eval_python(host), run_root(host)
    except LaunchError as exc:
        print(f"FAIL  {exc}")
        return 2
    runs = host.roots.get("runs")
    try:
        arms = local_run.expected_arms(args.from_run, args.arm, args.serving,
                                       Path(runs) / args.from_run if runs else None)
    except local_run.LocalRunError as exc:
        print(f"FAIL  {exc}")
        return 2
    print(f"# serves: run {args.from_run}, which does not resolve on this machine yet ({why})")
    for arm in arms:
        print(f"# results file: {finish.results_path(root, seed_set_arm(arm, args.seed_set))}")
    print(_eval_command(args, python))
    return 0


def _bench_dirs(args: argparse.Namespace, host: HostProfile) -> tuple[Path | None, Path | None]:
    """--asset-dir and --task-data-root as given, else the assets/ and tasks/ of the qvla-v1-bench
    copy `qvla data get bench` made under the host's corpus root. A tally reads neither and gets no
    default. The defaults are not written back into `args`, so the command `--dry-run` (and
    `--print`) prints stays the one the user typed."""
    if args.tally_only or "corpus" not in host.roots:
        return args.asset_dir, args.task_data_root
    from axis_qvla.bench import inputs
    from axis_qvla.core.paths import Paths

    assets, tasks = inputs.default_bench_dirs(Paths(host))
    return args.asset_dir or assets, args.task_data_root or tasks


def _on_disk_refusal(args: argparse.Namespace, b, host: HostProfile, req: EvalRequest, target: _Target,
                     asset_dir: Path | None, task_root: Path | None) -> str | None:
    """What is on disk, before the interpreter: the roots, then the Polyhaven bank, which is linked
    from the assets copy into the tree the workers render from (axis_qvla.bench.inputs.link_bank) and
    then compared with the checksum manifest. A run refused here prints no command at all, and a
    command printed later re-enters this one, so no printed command skips these checks."""
    from axis_qvla.bench import inputs
    from axis_qvla.bench.launch import missing_root, render_tree

    if target.fetch:
        refused = _local_copy_refusal(_root_prog(args), target)
        if refused:
            return refused
    if asset_dir is None:
        return ("the benchmark declares assets.fetch 'refuse': pass --asset-dir <qvla-v1-bench assets/> (or "
                "declare [roots] corpus, under which `qvla data get bench` puts it). Refused before any server starts.")
    gap = missing_root(args.render_root, task_root, dual_sim=b.dual_sim, asset_dir=asset_dir)
    if gap:
        return gap
    tree, lock = render_tree(host, req.render_root), inputs.asset_lock_for(b)
    if b.dual_sim:
        try:
            linked = inputs.link_bank(tree, asset_dir, lock)
        except OSError as exc:
            return (f"the Polyhaven bank cannot be linked into {tree / inputs.BANK_RENDER_REL} "
                    f"({type(exc).__name__}: {exc}). Refused before any server starts.")
        if linked:
            print(f"linked {len(linked)} Polyhaven texture files from {inputs.bank_source(asset_dir)} into "
                  f"{tree / inputs.BANK_RENDER_REL}")
    return inputs.refusal(tree, lock, dual_sim=b.dual_sim)


def _cmd_bench_eval(args: argparse.Namespace) -> int:
    """THE one command a benchmark user runs. Everything but what is served comes from the
    declaration and the host profile (axis_qvla.bench.launch.compose); what is served is resolved once
    (`_resolve_target`)."""
    from axis_qvla.bench import inputs
    from axis_qvla.bench.launch import EvalRequest, LaunchError, compose, interpreter_problem, missing_env, run_root
    from axis_qvla.bench.spec import DEFAULT_NAME, BenchmarkError, check, load_benchmark

    refusal = _policy_refusal(args)
    if refusal:
        print(f"FAIL  {refusal}")
        return 2
    # A relative path on this machine, absolute once, before anything reads it: the driver digests it
    # here and the server loads it from its own checkout; every command printed below names it so.
    from axis_qvla.bench.sim.campaign import local_checkpoint

    args.checkpoint = local_checkpoint(args.checkpoint)
    try:
        from axis_qvla.data.userconf import load_profile

        host = load_profile(args.host)
        b = load_benchmark(args.benchmark or DEFAULT_NAME)
    except (HostProfileError, BenchmarkError) as exc:
        print(f"FAIL  {exc}")
        return 2
    problems = check(b)
    if problems:
        # A declaration that disagrees with itself must not be evaluated: the number would
        # carry this benchmark's name and measure something else.
        for pr in problems:
            print(f"FAIL  {pr}")
        print(f"FAIL  benchmark {b.name!r} is inconsistent; fix {b.conf_dir} first "
              f"(`{args.bench_prog} check`)")
        return 2
    try:
        targets = _resolve_targets(args, host)
    except _Refused as exc:
        if args.print_only and exc.pending:
            # --print needs no data: a run not trained yet still prints the command that will evaluate it.
            return _print_unresolved(args, host, str(exc))
        print(f"FAIL  {exc}")
        return 2
    asset_dir, task_root = _bench_dirs(args, host)
    # Every evaluation is composed and checked before any starts: a --from-run run evaluated bare and
    # cond must not spend the bare evaluation and then be refused the cond one.
    plans: list[_Plan] = []
    for target in targets:
        serving = target.serving or args.serving or "bare"
        req = EvalRequest(
            checkpoint=target.checkpoint, config=target.config, arm=target.arm, serving=serving,
            # A --from-run mode served bare takes no guidance; the conditioned one takes --guidance-scale.
            guidance_scale=(args.guidance_scale or 0.0) if target.serving in (None, "conditioned") else 0.0,
            backend=target.backend, benchmark=args.benchmark or DEFAULT_NAME,
            gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, batch=args.batch,
            mem_fraction=args.mem_fraction, tally_only=args.tally_only,
            render_root=str(args.render_root.absolute()) if args.render_root else None,
            policy=tuple(args.policy), seed_set=args.seed_set, off_protocol=args.off_protocol,
        )
        serve_config = None
        try:
            if target.serve_config is not None:
                serve_config = _serve_target(req, target.serve_config, host)
                req = dataclasses.replace(req, config=str(serve_config[0]))
            launch = compose(req, host, b)
        except LaunchError as exc:
            print(f"FAIL  {exc}")
            return 2
        if args.verify_checkpoint:
            refused = _checkpoint_refusal(_root_prog(args), target)
            if refused:
                print(f"FAIL  {refused}")
                return 2
        # One argv for the dry run's listing and the real call.
        launch = dataclasses.replace(launch, argv=[*launch.argv, *inputs.path_argv(
            asset_dir, args.render_root, task_root, asset_lock=inputs.asset_lock_for(b))])
        refused = _execution_refusal(args, b, {**os.environ, **launch.env})
        if refused:
            print(f"FAIL  {refused}")
            return 2
        plans.append(_Plan(target, req, launch, serve_config))
    if len({p.launch.arm for p in plans}) != len(plans):
        print(f"FAIL  the evaluations of this command would share one run name ({plans[0].launch.arm}); "
              f"each needs its own")
        return 2
    note = _rollout_logs_note(host, plans)
    if note:
        print(note, file=sys.stderr)   # stderr: --print's stdout is a batch script
    if args.print_only:
        _print_plan(args, host, plans, asset_dir, task_root)
        return 0
    # Written here, before the dry run lists it or the driver is handed it. A tally starts no
    # server, so it neither writes nor compares the file.
    for p in plans:
        if p.serve_config and not args.tally_only:
            refused = _write_serve_config(*p.serve_config)
            if refused:
                print(f"FAIL  {refused}")
                return 2

    if args.dry_run:
        print("The driver invocation `eval` composes, listed for inspection, not to be run on its own: the "
              "driver does not check the asset checksum manifest or the texture bank; the command printed last does.")
        for p in plans:
            print(f"run: {p.launch.arm}")
            for k, v in sorted(p.launch.env.items()):
                print(f"env  {k}={v}")
            print(f"python: {p.launch.python}")
            print("argv: -m axis_qvla.bench.sim.campaign " + " ".join(p.launch.argv))
            if p.serve_config:
                done = "not written: a tally starts no server" if args.tally_only else "written"
                print(f"serve config: {p.serve_config[0]} ({len(p.serve_config[1])} bytes, {done}; "
                      f"rendered for {p.target.checkpoint_id or f'run {args.from_run}'})")
        print("\ndry run: nothing started. The command that runs it, making every check `eval` makes:\n  "
              + _eval_command(args, plans[0].launch.python))
        return 0

    # What is on disk, before the interpreter (`_on_disk_refusal`).
    for p in plans:
        gap = None if args.tally_only else _on_disk_refusal(args, b, host, p.req, p.target, asset_dir, task_root)
        if gap:
            print(f"FAIL  {gap}")
            return 2
    python = plans[0].launch.python
    # The driver imports openpi_client and its workers need the dual-sim client, so it runs under the
    # host's evaluation environment: this command re-executes itself there or, when it is not
    # built, names the one command that builds it -- rather than start eight servers and fail.
    if os.path.realpath(sys.executable) != os.path.realpath(str(python)):
        if os.environ.get(REEXEC_MARKER) == "1":
            # Already re-executed once under the eval environment, and sys.executable is STILL not it: its
            # venv's `bin/python` is a wrapper that execs some other interpreter. Re-exec again would
            # loop forever (or re-exec the same wrapper, which lands here again), so this is refused
            # rather than retried.
            print(f"FAIL  `{args.bench_prog} eval` re-executed under {python} but sys.executable is "
                  f"{sys.executable}: that venv's python is not a plain interpreter (it execs something "
                  f"else), so this command cannot run there. Point [venv] eval at a plain interpreter in "
                  f"{host.source()}.")
            return 2
        gap = interpreter_problem(python)
        if gap:
            print(f"FAIL  `{args.bench_prog} eval` must run under the evaluation environment ({python}), "
                  f"which {gap}: build it with `{_root_prog(args)} env build eval` (or set [venv] eval in "
                  f"{host.source()}), then run:\n  {_eval_command(args, python)}")
            return 2
        return _reexec(python, args)
    # Running under the evaluation environment now, whether this call started there or arrived via _reexec:
    # the marker is this process's own business, never the driver's, a server's or a worker's.
    os.environ.pop(REEXEC_MARKER, None)
    for p in plans:
        gap = missing_env(p.req, host)
        if gap:
            print(f"FAIL  {gap}")
            return 2
    from axis_qvla.bench import finish
    from axis_qvla.bench.sim import campaign

    for p in plans:
        os.environ.update(p.launch.env)
        rc = int(campaign.main(p.launch.argv))
        if rc != 0:
            return rc
        rc = finish.finish(run_root(host), p.launch.arm, b, ft_seed=p.target.ft_seed,
                           pretrain_seed=p.target.pretrain_seed)
        if rc != 0:
            return rc
    return 0


# `bench score` options that apply to results files only, and their defaults there. The parser
# leaves them None, so `--run-root`, whose table uses none of them, can refuse each one given.
_FILE_ONLY = ("view", "bootstrap", "per_task", "allow_off_protocol", "benchmark", "bench_conf_dir")


def _cmd_bench_score(args: argparse.Namespace) -> int:
    from axis_qvla.bench.results.cli import SCORE_DEFAULTS, score_files
    from axis_qvla.bench.spec import DEFAULT_NAME

    if bool(args.run_root) == bool(args.files):
        print("FAIL  give results files (pooled and per-task success with intervals) or --run-root "
              "(one line per run summary), not both or neither")
        return 2
    if args.files:
        if args.arm:
            print("FAIL  --run selects runs under --run-root; with results files, name the files")
            return 2
        for key, default in {**SCORE_DEFAULTS, "benchmark": DEFAULT_NAME}.items():
            if getattr(args, key) is None:
                setattr(args, key, default)
        return score_files(args)
    given = [f"--{k.replace('_', '-')}" for k in _FILE_ONLY if getattr(args, k) is not None]
    if given:
        print(f"FAIL  {', '.join(given)} apply to results files; --run-root lists the run summaries as "
              f"the driver wrote them and uses none of them")
        return 2
    from axis_qvla.bench.results.summaries import ScoreError, format_table, read_summaries

    try:
        rows = read_summaries(Path(args.run_root), args.arm or None)
    except ScoreError as exc:
        print(f"FAIL  {exc}")
        return 2
    if not rows:
        print(f"no out_<run>/arm_summary.json under {args.run_root}")
        return 1
    print(format_table(rows))
    return 0


def _cmd_bench_versions(args: argparse.Namespace) -> int:
    from axis_qvla.bench import versions

    for v in versions.available_versions():
        try:
            d = versions.version_dir(v)
            sha = versions.version_sha256(versions.content_files(d))
            problems = versions.check_version(d)
            frozen = versions.frozen_record(d)
        except versions.VersionError as exc:
            # One bad directory names its own problem and does not stop the rest from listing.
            print(f"{v}  ERROR  {exc}")
            continue
        state = "frozen" if not problems else ("EDITED" if frozen else "not frozen")
        print(f"{v}  {sha}  {state}")
    return 0


# What `check-server` imports beyond the standard library; the evaluation environment has all three.
CHECK_SERVER_MODULES = ("numpy", "websockets", "openpi_client")


def _cmd_bench_check_server(args: argparse.Namespace) -> int:
    """Run the check here when this interpreter has what it imports; otherwise re-execute under the
    evaluation environment, as `bench eval` does, or name the one command that builds it. A user who
    installed only the `eval` extra has none of the three in their own interpreter."""
    import importlib.util

    missing = [m for m in CHECK_SERVER_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        return _check_server_elsewhere(args, missing)
    os.environ.pop(REEXEC_MARKER, None)
    from axis_qvla.bench.sim import check_server   # numpy: imported only when the command runs

    return check_server.run(args)


def _rollout_logs_note(host, plans) -> str | None:
    """A profile with no `rollout_logs` root files the per-seed rollout files under the paper's layout
    in the backend's logs root (axis_qvla.bench.launch.rollout_log_root): say where, and the root that moves them."""
    from axis_qvla.bench.launch import ROLLOUT_LOGS_ROOT

    if ROLLOUT_LOGS_ROOT in host.roots:
        return None
    logs = sorted({a.split("=", 1)[1] for p in plans for a in p.launch.argv if a.startswith("--log-root=")})
    if not logs:
        return None
    return (f"NOTE  {host.source()} declares no [roots] {ROLLOUT_LOGS_ROOT}: the per-seed rollout files go to "
            f"{', '.join(logs)}; add `{ROLLOUT_LOGS_ROOT} = \"<dir>\"` to its [roots] to file them there instead.")


def _check_server_elsewhere(args: argparse.Namespace, missing: list[str]) -> int:
    from axis_qvla.bench.launch import interpreter_problem
    from axis_qvla.data.userconf import USER_CONFIG, load_profile
    from axis_qvla.train import envs

    prog = f"{args.bench_prog} check-server"
    try:
        profile = load_profile(args.host)
    except HostProfileError as exc:
        if args.host or Path(USER_CONFIG).is_file():
            print(f"FAIL  {exc}")   # a profile was named or is here, and it does not load: say why
            return 2
        profile = None              # no qvla.toml here: the environment where `env build eval` puts it
    python = envs.env_python("eval", profile)
    lacks = f"this interpreter ({sys.executable}) has no {', '.join(missing)}"
    if os.environ.get(REEXEC_MARKER) == "1":
        print(f"FAIL  `{prog}` re-executed under {python}, and {lacks}: rebuild it with "
              f"`{_root_prog(args)} env build eval --rebuild`.")
        return 2
    gap = interpreter_problem(python)
    if gap:
        print(f"FAIL  `{prog}` runs under the evaluation environment ({python}), which {gap}, and {lacks}: "
              f"build it with `{_root_prog(args)} env build eval`, then run this command again.")
        return 2
    return _reexec(python, args, "check-server")


def add_bench_parsers(sub: argparse._SubParsersAction) -> None:
    bench = sub.add_parser(
        "bench",
        help="the held-out benchmark: check the declaration, evaluate a checkpoint, read scores",
        description="The held-out Franka benchmark (conf/benchmarks/<version>/<name>.json): verify the "
                    "declaration, evaluate ONE checkpoint on it, score results files, or read run "
                    "summaries back.",
    )
    bs = bench.add_subparsers(dest="subcommand", required=True)
    # "<parent prog> bench", from whichever CLI registered the group (`qvla bench` in public).
    prog = bench.prog

    chk = bs.add_parser("check", help="print the benchmark identity; exit 1 on any inconsistency")
    chk.add_argument("--benchmark", default=None,
                     help="declaration, as <version>/<name> or <name> (default franka_heldout24 of "
                          "axis-heldout-v1)")
    chk.add_argument("--pretraining-prompts", dest="corpus_prompts", metavar="FILE", default=None,
                     help="report the instructions the benchmark shares with the pretraining data "
                          "(experiments/paper/pretraining_prompts.json for the paper's)")
    chk.set_defaults(func=_cmd_bench_check, bench_prog=prog)

    vs = bs.add_parser("versions", help="list the benchmark versions under conf/benchmarks/ and their hashes")
    vs.set_defaults(func=_cmd_bench_versions, bench_prog=prog)

    ev = bs.add_parser(
        "eval",
        help="evaluate ONE checkpoint on the benchmark (the campaign driver, fully composed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Evaluate one checkpoint on the held-out benchmark. You name the checkpoint, its\n"
            "config and how it is served; the task list, seeds, layouts, conventions and XLA\n"
            "pin come from conf/benchmarks/<version>/<name>.json, and every path and interpreter from\n"
            "the host profile, ./qvla.toml or --host (roots r1, bench, openpi, eval; venvs eval, online).\n"
            "\n"
            "Serving is ONE word: 'bare' serves the checkpoint as-is; 'conditioned' tags every\n"
            "prompt with quality tag 5 on the server and passes --cfg-conditioning to the client\n"
            "(a tagged finetune is a different policy untagged, so results land under\n"
            "<run>_w<G>_dualsim). The same checkpoint served both ways is two runs.\n"
            "\n"
            "Writes <r1>/out_<run>/run_manifest.json before the first server starts and\n"
            "arm_summary.json at the end (checkpoint sha256, benchmark identity, eval\n"
            "fingerprint); per-seed files land under the profile's rollout_logs root (qvla init:\n"
            "<r1>/rollout_logs/), the same for every backend.\n"
            "--tally-only records itself in tally_manifest.json instead, beside the original\n"
            "run's run_manifest.json; --dry-run starts nothing and writes nothing but a rendered\n"
            "server config (below).\n"
            f"Read them back with `{prog} score --run-root <r1>`; `results build` turns the run into\n"
            f"a results file, which `{prog} score <file>` scores with intervals.\n"
            "\n"
            "Run it from any interpreter: it re-executes itself under the evaluation environment, or\n"
            "names the `env build` command when that environment is not yet built, rather than\n"
            "start eight servers and fail. --print prints the commands it would run instead of\n"
            "running them (for a batch script). Example (a cells.json eval_command):\n"
            "  python -m axis_qvla.cli bench eval --host <host profile> \\\n"
            "      --checkpoint '<local copy of axisrobotics/qvla-v1-pi05@v1.0:ft/pi05_axis_heldout_qual_v2_bc__qual_v2_bc/13579>' \\\n"
            "      --config pi05_axis_heldout_qual_v2_bc --serving bare --run r1bc_r2 \\\n"
            "      --asset-dir '<local copy of axisrobotics/qvla-v1-bench@v1.0:assets>' \\\n"
            "      --render-root third_party/axis-render \\\n"
            "      --task-data-root '<local copy of axisrobotics/qvla-v1-bench@v1.0:tasks>'\n"
            "A paper checkpoint of any backbone is named by --checkpoint-id <cells.json checkpoint.id>:\n"
            "the backend and the server's config come from the results map (pi0.5's is rendered into\n"
            "<r1>/out_<run>/serve/openpi_config.json) and, without --checkpoint, the local copy\n"
            "`qvla data get checkpoints/<id>` made is served.\n"
            "A fine-tune `qvla train` ran on this machine is named by --from-run <run id>: its\n"
            "backbone, checkpoint, server config and seeds come from <runs>/<run id>, and a run\n"
            "fine-tuned with --cond is evaluated bare and cond (runs <run id>_b and <run id>_c).\n"
            "Any other policy: start its server, check it with `" + prog + " check-server --policy\n"
            "HOST:PORT`, then pass --policy HOST:PORT --run <name> instead of --checkpoint/--config.\n"
            "Add --dry-run first: it lists the composed driver argv, prints the command that runs it,\n"
            "and starts nothing."
            "\n\n"
            "At the end it writes <r1>/out_<run>/results.json and prints the all-24 and per-task table."
        ),
    )
    ev.add_argument("--checkpoint", default=None,
                    help="checkpoint DIRECTORY, digested by content; or a reference the server loads by "
                         "name (a Hub id <org>/<name>, a gs:// URL), identified by that exact string, which "
                         "can serve other weights later. To pin a revision, download it to a local directory "
                         "(hf download <org>/<name> --revision <commit> --local-dir <dir>) and pass that: no "
                         "reference server loads <org>/<name>@<revision> "
                         "(required unless --policy; with --policy, an optional record of what the server serves)")
    ev.add_argument("--checkpoint-id", default=None, metavar="ID",
                    help="a paper checkpoint, any backbone (experiments/paper/cells.json checkpoint.id): the "
                         "backend, the server's config and, without --checkpoint, the local copy `qvla data get "
                         "checkpoints/<id>` made are read from the results map (`qvla paper reeval` does this for you)")
    ev.add_argument("--from-run", default=None, metavar="RUN_ID",
                    help="evaluate a fine-tune `qvla train` ran on this machine (<runs>/<RUN_ID>): its backbone's "
                         "server, checkpoint, config and seeds are the run's own, and each serving mode the run "
                         "records (bare; bare and cond after `qvla train finetune --cond`) is evaluated, or the "
                         "one --serving names. With --print it prints the command before the run is trained")
    ev.add_argument("--verify-checkpoint", action="store_true",
                    help="with --checkpoint-id, any backbone: refuse unless the published files of what is "
                         "served (--checkpoint, else the `qvla data get checkpoints/<id>` copy; for SmolVLA/X-VLA "
                         "the step directory holding pretrained_model/) match the checkpoint.pub_digest cells.json "
                         "records; reads every byte of the checkpoint, so off by default")
    ev.add_argument("--config", default=None,
                    help="the server's config. openpi: a rendered config file (<file>.json); a paper checkpoint is "
                         "named with --checkpoint-id instead (the openpi config name a frozen cells.json "
                         "eval_command passes here is served the same way). SmolVLA/X-VLA: the lerobot policy "
                         "name. Required unless --policy, --checkpoint-id or --from-run")
    ev.add_argument("--policy", action="append", default=[], metavar="HOST:PORT",
                    help="evaluate a policy server that is already running (repeatable); see "
                         "docs/reference/policy-server.md and `check-server`. Nothing is started.")
    ev.add_argument("--run", dest="arm", default=None, metavar="RUN",
                    help="the run name results and summaries are filed under. Default "
                         "<checkpoint dir basename>_<serving>; with --checkpoint-id, the id (its / as .) and "
                         "_<serving>; with --from-run, the run id. A --from-run run evaluated bare and cond "
                         "is filed as <RUN>_b and <RUN>_c")
    ev.add_argument("--serving", choices=("bare", "conditioned"), default=None,
                    help="how a reference server serves the checkpoint (default bare; with --from-run, "
                         "every mode the run records)")
    ev.add_argument("--asset-dir", type=Path, default=None,
                    help="the qvla-v1-bench assets/ tree (default <corpus>/qvla-v1-bench/assets, where `qvla data "
                         "get bench` puts it); its Polyhaven bank is linked into the render root")
    ev.add_argument("--render-root", type=Path, default=None,
                    help="axis-render checkout for the render half (default third_party/axis-render)")
    ev.add_argument("--task-data-root", type=Path, default=None,
                    help="the qvla-v1-bench tasks/ directory (default <corpus>/qvla-v1-bench/tasks, where `qvla "
                         "data get bench` puts it)")
    ev.add_argument("--guidance-scale", type=float, default=None,
                    help="CFG guidance weight under --serving conditioned (openpi only; default 0)")
    ev.add_argument("--backend", choices=BACKENDS, default=None,
                    help="reference policy server to start (default openpi): " +
                         "; ".join(f"{k}: {s.label}" for k, s in REFERENCE_SERVERS.items()))
    ev.add_argument("--benchmark", default=None,
                    help="declaration, as <version>/<name> or <name> (default franka_heldout24 of "
                         "axis-heldout-v1)")
    ev.add_argument("--host", default=None,
                    help="a qvla.toml path or a host profile name; default ./qvla.toml, then $AXIS_HOST"
                         + (f" (profiles: {', '.join(available_hosts())})" if available_hosts() else ""))
    ev.add_argument("--gpus", default=None, help="comma list; default every GPU the host declares")
    ev.add_argument("--workers-per-gpu", type=int, default=4)
    ev.add_argument("--batch", type=int, default=10, help="seeds per rollout subprocess")
    ev.add_argument("--mem-fraction", default="0.5", help="XLA_PYTHON_CLIENT_MEM_FRACTION per server")
    ev.add_argument("--dry-run", action="store_true",
                    help="list the composed driver argv and environment (for inspection, not to be run on "
                         "their own), and print the `eval` command that runs them with every check; start "
                         "nothing. A paper checkpoint's rendered server config is written")
    ev.add_argument("--tally-only", action="store_true",
                    help="start nothing; rebuild arm_summary.json from the result files on disk "
                         "(recorded in tally_manifest.json)")
    from axis_qvla.bench.spec import SEED_SETS   # stdlib only: the parser stays numpy-free

    ev.add_argument("--seed-set", choices=SEED_SETS, default="test",
                    help="evaluate the declared test seeds (default) or the development seeds, which "
                         "tuning and checkpoint selection must use; a dev run is filed under <run>_dev")
    ev.add_argument("--off-protocol", action="store_true",
                    help="run although the environment changes a declared execution constant (replan, "
                         "step budget, ...); the deviation is recorded in run_manifest.json and every "
                         "task's meta.json")
    ev.add_argument("--ft-seed", type=int, default=None, metavar="SEED",
                    help="the fine-tune training seed, recorded in the results file (default: not recorded; "
                         "with --from-run, the run's own, which this may only repeat)")
    ev.add_argument("--pretrain-seed", type=int, default=None, metavar="SEED",
                    help="the pretraining seed, recorded with --ft-seed (default: no pretraining; with "
                         "--from-run, the run's own, or this when the run records none)")
    ev.add_argument("--print", dest="print_only", action="store_true",
                    help="print the one command that runs this evaluation (for a batch script) and start nothing")
    ev.set_defaults(func=_cmd_bench_eval, bench_prog=prog, eval_options=_option_table(ev))

    sc = bs.add_parser("score", help="score results files (all-24 and per-task, with seed-bootstrap "
                                     "intervals), or list the run summaries under a campaign root")
    sc.add_argument("files", nargs="*", help="qvla.results/v1 files (`results build` makes one from a run)")
    sc.add_argument("--run-root", default=None, help="instead: the campaign root holding out_<run>/ dirs")
    sc.add_argument("--run", dest="arm", action="append", default=[], metavar="RUN",
                    help="with --run-root: only these runs (repeatable); default every run found")
    sc.add_argument("--benchmark", default=None,
                    help="with results files: the declaration they are scored against (default franka_heldout24)")
    sc.add_argument("--bench-conf-dir", default=None, help=argparse.SUPPRESS)
    from axis_qvla.bench.results.cli import add_score_options   # stdlib only: the parser stays numpy-free

    add_score_options(sc)
    # Parser-level defaults override the shared options' own: None means "not given" (_FILE_ONLY).
    sc.set_defaults(func=_cmd_bench_score, bench_prog=prog, **{k: None for k in _FILE_ONLY})

    from axis_qvla.bench.protocol import add_check_server_arguments   # stdlib only: the parser stays numpy-free

    cs = bs.add_parser(
        "check-server",
        help="check that a running policy server speaks the protocol `eval` uses (docs/reference/policy-server.md)",
        description="Connect to a running policy server, read its handshake and send a few "
                    "observations shaped as the evaluation sends them; exit 0 if it conforms. It runs "
                    "under the evaluation environment ([venv] eval): from an interpreter without numpy, "
                    "websockets and openpi-client it re-executes itself there, or names `env build eval` "
                    "when that environment is not built.")
    add_check_server_arguments(cs)
    cs.add_argument("--host", default=None,
                    help="a qvla.toml path or a host profile name, for its [venv] eval (default ./qvla.toml)")
    cs.set_defaults(func=_cmd_bench_check_server, bench_prog=prog, eval_options=_option_table(cs))
