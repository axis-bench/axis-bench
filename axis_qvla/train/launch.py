"""Running a command in an environment this interpreter does not share: training, merges, builds.

Why: `qvla train`, `qvla env build` and the LeRobot merges each run a command in another
interpreter (axis_qvla.train.envs). One place composes it: its argv, its working directory and the
variables it adds. `--print` prints it as one line to run yourself, for example in a SLURM job.
Otherwise it runs with its output on this terminal and its exit code returned. The child is waited on
by its handle, never through a process listing.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping


class LaunchError(ValueError):
    """A command cannot be composed for an environment."""


@dataclasses.dataclass(frozen=True)
class Launch:
    argv: tuple[str, ...]
    cwd: pathlib.Path
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)   # added to the caller's environment

    def shell(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()))
        cmd = " ".join(shlex.quote(a) for a in self.argv)
        return f"cd {shlex.quote(str(self.cwd))} && " + (f"env {env} " if env else "") + cmd


def run_streaming(launch: Launch) -> int:
    """Run `launch` with its output on this terminal; its exit code (a signal's as 128 + n)."""
    proc = subprocess.Popen(list(launch.argv), cwd=launch.cwd, env={**os.environ, **launch.env})
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        rc = proc.wait()          # the child got the same SIGINT: let it stop, then report its code
    return 128 - rc if rc < 0 else rc


# What every command runs a Launch with; a test replaces it with a recorder and starts nothing.
RUNNER: Callable[[Launch], int] = run_streaming


def run(launch: Launch) -> int:
    return RUNNER(launch)


def from_rendered(argv, *, python: pathlib.Path, cwd: pathlib.Path,
                  extra_env: Mapping[str, str] | None = None) -> Launch:
    """The command axis_qvla.train.generate.render returns (`env K=V ... python|torchrun ...`), to run with
    `python`'s environment: `python` becomes that interpreter, `torchrun` the one beside it."""
    argv, env = list(argv), {}
    if argv and argv[0] == "env":
        argv.pop(0)
        while argv and "=" in argv[0] and not argv[0].startswith("-"):
            key, _, value = argv.pop(0).partition("=")
            env[key] = value
    if not argv:
        raise LaunchError("the rendered command is empty")
    prog = {"python": str(python), "torchrun": str(pathlib.Path(python).parent / "torchrun")}.get(argv[0])
    if prog is None:
        raise LaunchError(f"cannot run {argv[0]!r} in an environment (only python and torchrun)")
    return Launch((prog, *argv[1:]), pathlib.Path(cwd), {**env, **(extra_env or {})})


def task_dirs(roots_file, corpus_base) -> list[pathlib.Path]:
    """The task directories a roots index names (a bare mapping, or a v2 index's `roots`), in
    ascending task id: the order a merged copy holds them in."""
    try:
        doc = json.loads(pathlib.Path(roots_file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LaunchError(f"cannot read {roots_file} ({exc.__class__.__name__})") from None
    roots = doc["roots"] if isinstance(doc, dict) and isinstance(doc.get("roots"), dict) else doc
    if not isinstance(roots, dict) or not roots:
        raise LaunchError(f"{roots_file} names no task directories")
    try:
        ordered = sorted(roots.items(), key=lambda kv: int(kv[0]))
    except ValueError:
        raise LaunchError(f"{roots_file}: every key must be a task id") from None
    return [pathlib.Path(corpus_base) / str(rel) for _, rel in ordered]


def merge_launch(spec, paths, *, python: pathlib.Path, checkout: pathlib.Path) -> Launch | None:
    """The merge that builds a LeRobot run's merged copy on this machine while it is not there
    (conf/recipes/rowspaces.toml `lerobot_copy` "cut" or "plain"; docs: the merged finetune copy);
    None for pi0.5, a published copy or one that is built."""
    from axis_qvla.train import ids
    from axis_qvla.train.recipes import load_rowspaces

    if spec.backbone == "pi05-openpi":
        return None
    lay = load_rowspaces()[ids.content_id(spec.rowspace)[0]]
    if lay.lerobot_copy == "published":
        return None
    rs = ids.rowspace_dirs(paths, spec.rowspace)
    try:
        os.stat(rs["lerobot_dataset"])
        return None
    except (FileNotFoundError, NotADirectoryError):
        pass
    argv = [str(python), "-m", "lerobot.scripts.axis_merge_datasets", "--out", str(rs["lerobot_dataset"])]
    if lay.lerobot_copy == "cut":
        argv += ["--ranges", str(rs["ranges"]), "--action-source", str(lay.action_source)]
    return Launch((*argv, *map(str, task_dirs(rs["roots"], rs["corpus_base"]))), pathlib.Path(checkout))


def record_merge(spec, paths) -> None:
    """After `merge_launch`'s merge built a plain copy: record the roots index it was merged from
    (axis_qvla.train.plain_copy.PLAIN_RECORD), so binding refuses the copy once that index changes. A cut
    copy's own record is written by the merge (axis_qvla.train.cut_copy.CUT_RECORD)."""
    from axis_qvla.train import ids
    from axis_qvla.train.plain_copy import write_record
    from axis_qvla.train.recipes import load_rowspaces

    if load_rowspaces()[ids.content_id(spec.rowspace)[0]].lerobot_copy == "plain":
        write_record(ids.rowspace_dirs(paths, spec.rowspace))


def training_launch(spec, argv, paths, *, python: pathlib.Path, checkout: pathlib.Path, smoke: bool) -> Launch:
    """The rendered training command in the backbone's environment. A pi0.5 smoke keeps its whole
    train state on one GPU, so it raises JAX's preallocation and compiles into a new, empty cache."""
    extra = {}
    if spec.backbone == "pi05-openpi" and smoke:
        cache = paths.scratch / "jax"
        cache.mkdir(parents=True, exist_ok=True)
        extra = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
                 "JAX_COMPILATION_CACHE_DIR": tempfile.mkdtemp(dir=cache, prefix=f"{spec.id}.smoke.")}
    return from_rendered(argv, python=python, cwd=checkout, extra_env=extra)


def visible_gpus(profile, environ: Mapping[str, str]) -> int | None:
    """The GPUs a launch here sees: CUDA_VISIBLE_DEVICES when it is set, else qvla.toml's `gpus`.
    A CUDA_VISIBLE_DEVICES that is set but names no GPU hides every GPU from the launch, which would
    then train on the CPU or fail inside its first step; it is refused, naming the variable."""
    cvd = environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        n = len([x for x in cvd.split(",") if x.strip()])
        if not n:
            raise LaunchError(f"CUDA_VISIBLE_DEVICES is set to {cvd!r}, which names no GPU, so the training "
                              f"would see none; unset it or name the GPUs to train on")
        return n
    return int(profile.gpus) or None


def default_fsdp_devices(spec, recipe, paths, profile, environ: Mapping[str, str], *, resume: bool) -> int | None:
    """--fsdp-devices when none is given and the run trains here: a resumed run's recorded value, else
    the visible GPUs when the preset's sharding cannot run on them (None keeps the preset's). A
    launch's device count must be a multiple of its sharding, so a preset whose sharding the visible
    GPUs already are a multiple of -- the smoke presets' explicit 1 always is -- trains as the preset
    declares it. The openpi line also splits the global batch over every device it sees, so visible
    GPUs that do not divide it can run no sharding at all: that is refused here, before anything is
    rendered, naming the two choices that fix it."""
    from axis_qvla.train import ids
    from axis_qvla.train.recipes import resolved_fields

    if resume:
        try:
            return json.loads((ids.run_root(paths, spec.id) / "run.json").read_text(encoding="utf-8"))["fsdp_devices"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
    n = visible_gpus(profile, environ)
    if not n:
        return None
    preset, batch = int(recipe.structure["fsdp_devices"]), int(resolved_fields(recipe, spec)["batch_size"])
    if batch % n:
        raise LaunchError(f"{spec.id}: the {n} visible GPUs do not divide its global batch of {batch}, so neither "
                          f"its preset's sharding over {preset} devices nor one over {n} can launch here: choose "
                          f"a number of GPUs that divides {batch} with CUDA_VISIBLE_DEVICES, and pass "
                          f"--fsdp-devices N for a sharding over N of them")
    return None if n % preset == 0 else n
