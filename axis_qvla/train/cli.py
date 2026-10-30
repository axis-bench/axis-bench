"""`qvla train`: bind a run spec, render it into its run directory and train it in the backbone's environment.

    qvla train <run_spec.toml> [--host <qvla.toml>] [--smoke] [--no-bind] [--fsdp-devices N] [--resume]
               [--print | --render]
    qvla train pretrain --labels <name> --backbone <pi05|smolvla|xvla> [--run NAME] [--print]
    qvla train finetune --from <run id> [--cond] [--run NAME] [--print]
    qvla env build|check <environment> ...               the training environments (axis_qvla.train.env_cli)
    qvla train smoke-fixture --episodes <dir> ...        the two-task smoke fixture (axis_qvla.train.smoke)
    qvla train merge-check <task dir> ...                a plain LeRobot copy's inputs (axis_qvla.train.plain_copy)

Three modes. By default the run trains here, in the backbone's own environment (axis_qvla.train.envs;
`qvla env build <backbone>` builds it), from its checkout, with its output streamed and its exit
code returned (axis_qvla.train.launch). A run whose base model (its own `base:<name>`, or that of the run
it starts from) is missing from the base-weights copy fetches it first, verified against
conf/recipes/bases.lock.json (axis_qvla.train.runs.fetch_base; network; exit 3 when it cannot, naming
`qvla data get base-weights --include <name>`). A run starting from a published checkpoint fetches
the same way the base files its policy still loads (the pi0.5 tokenizer, the SmolVLM2 processor,
the X-VLA base; axis_qvla.train.runs.fetch_serving, naming `qvla data get base-weights --serving <backbone>`). A LeRobot run whose merged copy is not built yet runs
that merge first. `--print` reads only the run spec and its recipe (no data, no GPU, no network) and prints the
one `qvla train` line that trains the run on the machine that runs it, for example a SLURM job.
`--render` binds against this machine's data, writes the run directory and prints the backbone's
own command(s) without training; it fetches nothing, so a missing pi0.5 base is refused by binding,
naming the command that fetches it. When the run trains here and no --fsdp-devices is given, a pi0.5
run shards over the visible GPUs (CUDA_VISIBLE_DEVICES, else qvla.toml `gpus`) when the preset's
sharding cannot run on them; visible GPUs that do not divide its batch are refused, naming
--fsdp-devices; a resume keeps its recorded sharding.

A spec whose compiled labels are named by name (what `qvla train pretrain` writes) is completed from
them in every mode but `--print` (axis_qvla.train.method_runs.complete_from_labels): a real run's labels
must be on the paper's budget, and a SmolVLA / X-VLA run gains the overrides their exported files
require, unless it sets them itself.

Rendering (in every mode but `--print`) writes <runs>/<run id>/{run_spec.toml, run.json,
<backbone config>} (a smoke render writes the same files under <runs>/<run id>/rendered-smoke/ instead, so it
never overwrites a real run's recorded files), and names.json there when an input was named by name
(axis_qvla.train.names; a render naming nothing by name removes a stale one, and a resume rewrites
neither it nor binding.json). The backbone's command runs from that backbone's checkout with its venv. By
default it first binds the run's SignalBundle (axis_qvla.train.bind, which needs numpy): it refuses,
before any GPU time and listing every problem at once, a frame-index mismatch, a missing
capability, a reducer the backbone does not implement, a plan that does not fit the budget, a
per-rank batch that does not multiply out to the global batch, or an input the config would open
that is missing or has other bytes than the run names; it records the binding in binding.json
beside the mode's rendered files, and for SmolVLA / X-VLA it exports the bundle to the legacy files
their training reads. A pi0.5 finetune arm that names no bundle has its finetune plan compiled
here. `--no-bind` (pi0.5 only) skips that step; the openpi line repeats the checks when it loads
the bundle.

Both backbones refuse to train into a directory a run already wrote (LeRobot's output_dir; for
pi0.5 a real run's checkpoint directory <runs>/<arm id>/train/, since a smoke's starts over at
every launch). Left unchecked, that surfaces only once training launches, after GPU allocation, so
this command checks the same directory first and refuses here instead, naming it; it never deletes
a directory itself. Two pi0.5 launch options change how the run starts, never what it trains:
`--fsdp-devices N` shards the train state over N devices instead of the preset's (the global
batch is unchanged), and `--resume` continues the run in its checkpoint directory from the step
after its last saved checkpoint -- it is accepted only when that directory exists and the arm
renders exactly the config the run recorded; binding is checked again and binding.json is not
rewritten. A checkpoint directory that holds no saved step (a launch that failed inside its first
training step, which the openpi line does not clean up) is named as such in the refusal, with the
one way on: remove it and render again.

Exit codes: 0 trained (or printed, or rendered), 2 refused before training, 3 a missing base could
not be fetched, else the training's (or merge's) own.

Core tier: stdlib only at import.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import shlex

# `qvla train <name> ...` for the commands that are not a run-spec file.
_SUBCOMMANDS = {"env": "axis_qvla.train.envs", "smoke-fixture": "axis_qvla.train.smoke",
                "merge-check": "axis_qvla.train.plain_copy", "pretrain": "axis_qvla.train.pretrain",
                "finetune": "axis_qvla.train.finetune"}
# Words kept working and listed nowhere: `qvla train env plan|check`, what `qvla env build|check` were
# called before they had their own group.
HIDDEN_SUBCOMMANDS = frozenset({"env"})


class _TrainParser(argparse.ArgumentParser):
    """`qvla train` is two grammars sharing one subcommand slot: `train <run_spec> [options]` (every
    argument registered directly on this parser, like every other `_GROUPS` group) and
    `train <word> ...` for each word of _SUBCOMMANDS (`env`, `smoke-fixture`, `merge-check`,
    `pretrain`, `finetune`; forwarded whole to that module's main). Checking the first token before the normal grammar
    runs keeps both shapes true to their own command line: `train <run_spec> --host X` and
    `train --host X <run_spec>` both parse."""

    def parse_known_args(self, args=None, namespace=None):
        if args and args[0] in _SUBCOMMANDS:
            module, rest = _SUBCOMMANDS[args[0]], args[1:]
            ns = argparse.Namespace(func=lambda _ns, _m=module, _r=rest: importlib.import_module(_m).main(_r))
            return ns, []
        return super().parse_known_args(args, namespace)


def _check_output_dir(spec, paths, *, smoke: bool, resume: bool = False) -> None:
    """The backbone's own refusal of an existing output directory happens only once training
    launches. Check the same directory here, before that, so a re-run of a smoke -- or a real run
    retried after a previous attempt -- is refused before any GPU is allocated, not after. A pi0.5
    smoke's checkpoint directory is wiped at every launch, so it is not checked; a pi0.5 real run's
    is either new or, with `resume`, the one to continue."""
    from axis_qvla.train import ids
    from axis_qvla.train.generate import GenerateError

    out_dir = ids.lerobot_output_dir(paths, spec.id, smoke=smoke)
    if spec.backbone == "pi05-openpi":
        if smoke:
            return
        if resume and not out_dir.exists():
            raise GenerateError(f"{spec.id}: --resume continues the run in {out_dir}, which is not there; "
                                f"render without --resume to start the run")
        if not resume and out_dir.exists():
            if not _saved_steps(out_dir):
                raise GenerateError(
                    f"{spec.id}: {out_dir} already exists and holds no saved step: a launch stopped before its "
                    f"first checkpoint (one that fails inside a training step leaves the directory). The pinned "
                    f"openpi fork refuses to train into it, so remove {out_dir} yourself (qvla train never deletes "
                    f"a directory) and render again.")
            raise GenerateError(
                f"{spec.id}: {out_dir} already exists. The pinned openpi fork refuses to train into an existing "
                f"checkpoint directory unless it resumes, so this launch would fail after GPU allocation. "
                f"Pass --resume to continue that run from its last saved step, or remove {out_dir} yourself "
                f"(qvla train never deletes a directory) to start over.")
        return
    if out_dir.exists():
        raise GenerateError(
            f"{spec.id}: {out_dir} already exists. LeRobot refuses to train into an existing "
            f"output_dir when resume is false, so re-running this run would fail at launch, after "
            f"GPU allocation. Remove {out_dir} yourself first (qvla train never deletes a "
            f"directory) before rendering it again.")


def _saved_steps(checkpoint_dir: pathlib.Path) -> list[int]:
    """The steps saved in a pi0.5 checkpoint directory (its integer-named subdirectories). An
    unreadable directory raises OSError, never reads as one with no step."""
    import os

    with os.scandir(checkpoint_dir) as it:
        return sorted(int(e.name) for e in it if e.name.isdigit() and e.is_dir())


MODES = ("run", "render", "print")


def _check_launch_options(spec, *, fsdp_devices: int | None, resume: bool, smoke: bool, no_bind: bool) -> None:
    """What the launch options themselves refuse, before anything is read or built (`--print` too)."""
    from axis_qvla.train.generate import GenerateError

    if no_bind and spec.backbone != "pi05-openpi":
        raise GenerateError(f"{spec.id}: a {spec.backbone} run must be bound: its label files "
                            f"are exported from its compiled labels at bind time")
    if spec.backbone != "pi05-openpi" and (fsdp_devices is not None or resume):
        raise GenerateError(f"{spec.id}: --fsdp-devices and --resume are pi0.5 launch options; a "
                            f"{spec.backbone} run's GPUs are its recipe's world")
    if resume and smoke:
        raise GenerateError(f"{spec.id}: a smoke run starts over at every launch; it is not resumed")


def _check_resumed_config(spec, run: pathlib.Path, files: dict[str, bytes]) -> None:
    """A resumed run continues the config its first launch trained: the arm must render exactly
    the config recorded in the run directory (with the same --fsdp-devices)."""
    import json

    from axis_qvla.train.generate import GenerateError

    recorded = run / "openpi_config.json"
    try:
        old = recorded.read_bytes()
    except FileNotFoundError:
        raise GenerateError(f"{spec.id}: --resume needs the run's recorded {recorded}, which is not there") from None
    new = files["openpi_config.json"]
    if old == new:
        return
    try:
        a, b = json.loads(old), json.loads(new)
        keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    except ValueError:
        keys = ["(the recorded file is not JSON)"]
    raise GenerateError(f"{spec.id}: --resume would continue {run} with another config than it recorded "
                        f"(differs in {keys}); a resumed run trains the config it started with. To start over with "
                        f"this config instead, remove {run / 'train'} yourself and render without --resume")


def _print_command(path: pathlib.Path, *, host: str | None, smoke: bool, no_bind: bool,
                   fsdp_devices: int | None, resume: bool) -> int:
    """`--print`: the one command that trains the run on the machine it runs on. Reads only the run
    spec, its recipe and qvla.toml: no data, no GPU, no network."""
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data.userconf import load_profile
    from axis_qvla.train.armspec import ArmSpecError, load_armspec
    from axis_qvla.train.generate import GenerateError, check_renderable
    from axis_qvla.train.recipes import RecipeError, load_recipe

    path = pathlib.Path(path).resolve()
    try:
        spec = load_armspec(path)
        check_renderable(spec)
        load_recipe(spec.preset)
        _check_launch_options(spec, fsdp_devices=fsdp_devices, resume=resume, smoke=smoke, no_bind=no_bind)
        try:
            toml = load_profile(host).path
        except HostProfileError:
            if host is not None:
                raise
            toml = None                 # no ./qvla.toml here: the training machine's own is read there
    except (ArmSpecError, RecipeError, GenerateError, HostProfileError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    argv = ["qvla", "train", str(path), *(["--host", str(toml)] if toml is not None else []),
            *[flag for flag, on in (("--smoke", smoke), ("--no-bind", no_bind), ("--resume", resume)) if on],
            *(["--fsdp-devices", str(fsdp_devices)] if fsdp_devices is not None else [])]
    print(f"OK    {spec.id}: {path}")
    print(" ".join(shlex.quote(a) for a in argv))
    return 0


def train_spec(path: pathlib.Path, *, host: str | None = None, smoke: bool = False, no_bind: bool = False,
               fsdp_devices: int | None = None, resume: bool = False, mode: str = "run") -> int:
    """Train (mode "run"), render ("render") or plan ("print") the run spec at `path` (module docstring)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    if mode == "print":
        return _print_command(path, host=host, smoke=smoke, no_bind=no_bind, fsdp_devices=fsdp_devices,
                              resume=resume)
    import os

    from axis_qvla.core.canon import dumps_canonical
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.core.paths import Paths
    from axis_qvla.data.userconf import load_profile
    from axis_qvla.data.bases import BasesError
    from axis_qvla.data.hf import HfDigestMismatch, HfHttpError
    from axis_qvla.train import envs, ids, launch, names, runs
    from axis_qvla.train.armspec import ArmSpecError, load_armspec
    from axis_qvla.train.generate import GenerateError, check_renderable, render
    from axis_qvla.train.recipes import RecipeError, load_recipe

    render_only = mode == "render"
    try:
        profile = load_profile(host)
        paths = Paths(profile)
        spec, resolved = names.resolve_names(load_armspec(path), paths)
        if "bundle" in resolved and not names.is_published_labels(resolved["bundle"]["name"]):
            from axis_qvla.train.method_runs import complete_from_labels

            spec = complete_from_labels(spec, paths, smoke=smoke)
        check_renderable(spec)          # a declaration is refused as one, before any other refusal
        recipe = load_recipe(spec.preset)
        tier = envs.BACKBONE_TIER[spec.backbone]
        _check_launch_options(spec, fsdp_devices=fsdp_devices, resume=resume, smoke=smoke, no_bind=no_bind)
        if not render_only:
            missing = envs.missing_problem(tier, profile)
            if missing:
                raise GenerateError(f"{spec.id}: {missing}")
            launch.visible_gpus(profile, os.environ)        # an empty CUDA_VISIBLE_DEVICES is refused here
            if fsdp_devices is None and spec.backbone == "pi05-openpi":
                fsdp_devices = launch.default_fsdp_devices(spec, recipe, paths, profile, os.environ, resume=resume)
        _check_output_dir(spec, paths, smoke=smoke, resume=resume)
        base = None if render_only else runs.init_base(spec, paths)
        if base is not None and runs.missing_base_files(base, paths):
            print(f"GET   base weights {base} -> {ids.base_weights_root(paths)}", flush=True)
            try:
                runs.fetch_base(base, paths)
            except (HfHttpError, HfDigestMismatch, BasesError, OSError) as exc:
                print(f"FAIL  {spec.id}: fetching the base weights {base}: {exc}; fetch them where the "
                      f"publishers are reachable with `qvla data get base-weights --include {base}`")
                return 3
        serving = None if render_only or base is not None else runs.init_serving(spec, paths)
        if serving is not None and runs.missing_serving_files(serving, paths):
            print(f"GET   base weights --serving {serving} -> {ids.base_weights_root(paths)}", flush=True)
            try:
                runs.fetch_serving(serving, paths)
            except (HfHttpError, HfDigestMismatch, BasesError, OSError) as exc:
                print(f"FAIL  {spec.id}: fetching the {serving} base files its policy loads beside its "
                      f"checkpoint: {exc}; fetch them where the publishers are reachable with "
                      f"`qvla data get base-weights --serving {serving}`")
                return 3
        merge = None
        if spec.backbone != "pi05-openpi":
            merge = launch.merge_launch(spec, paths, python=envs.env_python(tier, profile),
                                        checkout=envs.checkout(tier, profile))
        if merge is not None and not render_only:
            print(f"RUN   {merge.shell()}", flush=True)
            rc = launch.run(merge)
            if rc:
                copy = ids.rowspace_dirs(paths, spec.rowspace)["lerobot_dataset"]
                left = (f"; remove the partial {copy} yourself (qvla train never deletes a directory) before "
                        f"training again") if os.path.lexists(copy) else ""
                print(f"FAIL  {spec.id}: the merged LeRobot copy was not built (exit {rc}){left}")
                return rc
            launch.record_merge(spec, paths)
            merge = None
        signal_files, bound, bound_dir = {}, None, None
        if not no_bind:
            from axis_qvla.train.bind import BindError, bind

            try:
                binding = bind(spec, recipe, paths, smoke=smoke, record=not resume, copy_pending=merge is not None)
                signal_files, bound, bound_dir = dict(binding.signal_files), binding.bundle, binding.bundle_dir
            except BindError as exc:
                raise GenerateError(str(exc)) from None
        files, argv_out = render(spec, recipe, paths, smoke=smoke, signal_files=signal_files, bundle=bound,
                                 bundle_dir=bound_dir, fsdp_devices=fsdp_devices, resume=resume)
        run = ids.run_root(paths, spec.id)
        if resume:
            _check_resumed_config(spec, run, files)
        # names.json is the record of what the names in the spec resolved to: written with binding.json
        # (never on a resume, which keeps both as the first launch recorded them), and removed when this
        # render named nothing by name, so a stale one never describes a run it did not render.
        names_file = ids.rendered_dir(paths, spec.id, smoke=smoke) / names.NAMES_FILE
        if resolved and not resume:
            files[names_file.relative_to(run).as_posix()] = dumps_canonical(resolved)
        elif not resolved and not resume and os.path.lexists(names_file):
            names_file.unlink()
        for name, data in sorted(files.items()):
            dest = run / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
        train = launch.training_launch(spec, argv_out, paths, python=envs.env_python(tier, profile),
                                       checkout=envs.checkout(tier, profile), smoke=smoke)
    except (ArmSpecError, RecipeError, GenerateError, envs.EnvError, launch.LaunchError,
            names.NameResolutionError, ids.IdError, HostProfileError, BasesError, ValueError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    print(f"OK    {spec.id} -> {run}")
    if render_only:
        if merge is not None:
            print(merge.shell())
        print(train.shell())
        return 0
    print(f"RUN   {train.shell()}", flush=True)
    rc = launch.run(train)
    print(f"{'OK    ' if rc == 0 else 'FAIL  '}{spec.id}: training exited {rc}")
    return rc


def _cmd_arm(args: argparse.Namespace) -> int:
    return train_spec(args.arm, host=args.host, smoke=args.smoke, no_bind=args.no_bind,
                      fsdp_devices=args.fsdp_devices, resume=args.resume,
                      mode="print" if args.print_only else "render" if args.render_only else "run")


def add_train_parsers(sub: argparse._SubParsersAction) -> None:
    # sub._parser_class picks what sub.add_parser() instantiates; swap it in just for this one
    # call so "train" gets _TrainParser's env-dispatch while every other _GROUPS group is
    # unaffected (restored in the `finally` whether add_parser raises or not).
    saved_parser_class, sub._parser_class = sub._parser_class, _TrainParser
    try:
        t = sub.add_parser("train", help="train a run spec in its backbone's environment (--print: the command only)")
    finally:
        sub._parser_class = saved_parser_class
    t.add_argument("arm", metavar="run_spec", type=pathlib.Path, help="the run spec (TOML) to train")
    t.add_argument("--host", default=None, help="this machine's qvla.toml (default: ./qvla.toml, then $AXIS_HOST)")
    t.add_argument("--smoke", action="store_true", help="a 20-step run (plan[:20]); final step 19")
    t.add_argument("--no-bind", dest="no_bind", action="store_true",
                   help="pi0.5 only: skip the pre-launch checks of the compiled labels")
    t.add_argument("--fsdp-devices", dest="fsdp_devices", type=int, default=None, metavar="N",
                   help="pi0.5 only: shard the train state over N devices (default: the visible GPUs when "
                        "training here and they divide the batch, else the preset's); the global batch is "
                        "unchanged, and the launch's device count must be a multiple of N")
    t.add_argument("--resume", action="store_true",
                   help="pi0.5 only: continue the run in <runs>/<run id>/train/ from its last saved step")
    how = t.add_mutually_exclusive_group()
    how.add_argument("--print", dest="print_only", action="store_true",
                     help="print the one command that trains the run where it runs (e.g. in a SLURM job); reads "
                          "only the run spec: no data, GPU or network")
    how.add_argument("--render", dest="render_only", action="store_true",
                     help="check the run against this machine's data, write its run directory and print the "
                          "backbone's own command, without training")
    t.set_defaults(func=_cmd_arm)
