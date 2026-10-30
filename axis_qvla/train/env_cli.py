"""`qvla env build|check <environment>`: build an environment from its pinned lock, then check it.

Why: training and evaluation each run in their own interpreter (axis_qvla.train.envs). These commands
used to be printed for the user to run in the right directory, and the build had to be repeated
whenever a `uv sync` removed packages. `build` runs each build line in the environment's checkout,
stops at the first failure (exit code returned), and checks the result exactly as `check` does:
the lock's digest, the Python version and gate imports, and the full installed set against the lock.
`--print` only prints the lines. An environment that is already built and passes `check` is left as
it is (`uv venv` refuses to re-create an existing one, and nothing would change); one that is built
but differs from its pins is repaired in place, every line but its `uv venv`; `--rebuild` removes a
`uv venv` environment and builds it from scratch.

    qvla env build <pi05|smolvla|xvla|eval|render|offline|droid_convert> [--print] [--rebuild]
    qvla env check <name> [--python <venv python>] [--checkout <dir>]

Exit codes: 0 built and checked, 1 built but differs from its pins, 2 refused, else the failing line's.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil


def _problems(spec, python, checkout) -> list[str]:
    from axis_qvla.train import envs

    return envs.check_lock(spec, checkout) + envs.check_built(spec, python) \
        + envs.check_closure(spec, python, checkout)


def _check(spec, python, checkout, *, hint: str = "") -> int:
    """`hint` follows a FAIL line: the one way on when the check fails."""
    problems = _problems(spec, python, checkout)
    for x in problems:
        print(f"DIFF  {x}")
    print(f"{'FAIL' if problems else 'OK'}    {spec.tier} environment at {python}" + (hint if problems else ""))
    return 1 if problems else 0


def _is_built(venv: pathlib.Path, python: pathlib.Path) -> bool:
    """A venv is built when its interpreter entry exists, even as a link that no longer resolves, or
    when its pyvenv.cfg does: such a venv is built and failing (repaired, or --rebuild), never one to
    create over. lstat, not stat, so a dangling link is seen. An unreadable path raises OSError."""
    for p in (python, venv / "pyvenv.cfg"):
        try:
            os.lstat(p)
            return True
        except (FileNotFoundError, NotADirectoryError):
            continue
    return False


def _cmd_build(args: argparse.Namespace) -> int:
    from axis_qvla.train import envs, launch

    try:
        tier = envs.tier_of(args.name)
        spec = envs.load_env(tier)
    except envs.EnvError as exc:
        print(f"FAIL  {exc}")
        return 2
    steps = [launch.Launch(("sh", "-c", line), envs.REPO / spec.checkout) for line in spec.build]
    if args.print_only:
        print(f"# {spec.description}")
        for s in steps:
            print(s.shell())
        return 0
    venv, python = envs.venv_dir(tier), envs.env_python(tier)
    try:
        built = _is_built(venv, python)
    except OSError as exc:
        print(f"FAIL  cannot read {python} ({exc.__class__.__name__})")
        return 2
    if built and not args.rebuild and not _problems(spec, python, None):
        print(f"OK    {args.name} already built at {python}")
        print("      it passes its check; --rebuild builds it again")
        return 0
    problems = envs.check_lock(spec)
    if problems:
        for x in problems:
            print(f"FAIL  {x}")
        print(f"      fetch the checkout first: git submodule update --init {spec.checkout}")
        return 2
    if shutil.which("uv") is None:
        print("FAIL  uv is not on PATH: install it (pip install uv), then run this again")
        return 2
    creates = [s for s in steps if s.argv[-1].startswith("uv venv")]
    hint = ""
    if built and creates and args.rebuild:
        if venv.is_symlink():
            # rmtree refuses a link, and removing what it points at is not this command's to do
            print(f"FAIL  {venv} is a symlink; remove it yourself")
            return 2
        print(f"RM    {venv}", flush=True)
        try:
            shutil.rmtree(venv)
        except OSError as exc:
            print(f"FAIL  cannot remove {venv} ({exc.__class__.__name__}: {exc}); remove it yourself")
            return 2
    elif built and creates:
        print(f"SKIP  {creates[0].shell()}: {venv} exists; repairing it in place "
              f"(--rebuild re-creates it)")
        steps = [s for s in steps if s not in creates]
        hint = f"; repairing it in place did not fix it: run `qvla env build {args.name} --rebuild`"
    for s in steps:
        print(f"RUN   {s.shell()}", flush=True)
        rc = launch.run(s)
        if rc:
            print(f"FAIL  exit {rc}: {s.shell()}" + hint)
            return rc
    return _check(spec, python, None, hint=hint)


def _cmd_check(args: argparse.Namespace) -> int:
    from axis_qvla.train import envs

    try:
        tier = envs.tier_of(args.name)
        spec = envs.load_env(tier)
    except envs.EnvError as exc:
        print(f"FAIL  {exc}")
        return 2
    return _check(spec, args.python or envs.env_python(tier), args.checkout)


def add_env_parsers(sub: argparse._SubParsersAction) -> None:
    from axis_qvla.train.envs import ENV_NAMES

    names = ", ".join(ENV_NAMES)
    e = sub.add_parser("env", help="build and check the training and evaluation environments")
    esub = e.add_subparsers(dest="env_command", required=True)
    b = esub.add_parser("build", help="build an environment from its pinned lock, then check it")
    b.add_argument("name", metavar="environment", help=f"one of {names}")
    b.add_argument("--print", dest="print_only", action="store_true",
                   help="print the build commands instead of running them")
    b.add_argument("--rebuild", action="store_true",
                   help="build it again even when it is already built and passes its check")
    b.set_defaults(func=_cmd_build)
    c = esub.add_parser("check", help="check a built environment against its pins")
    c.add_argument("name", metavar="environment", help=f"one of {names}")
    c.add_argument("--python", type=pathlib.Path, default=None,
                   help="the environment's interpreter (default: where `qvla env build` puts it)")
    c.add_argument("--checkout", type=pathlib.Path, default=None,
                   help="the environment's checkout when it is not at its submodule path")
    c.set_defaults(func=_cmd_check)
