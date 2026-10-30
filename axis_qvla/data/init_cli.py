"""`qvla init`: write ./qvla.toml, the one file that says where this machine keeps AXIS-Bench's data.

Why: every command builds its paths from qvla.toml (axis_qvla.data.userconf), and editing the example by
hand with placeholders was the first thing every tutorial asked for. `qvla init` writes the whole
file: the data roots under one directory (default ./srv) and the environments and checkouts under
this checkout.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_init(args: argparse.Namespace) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data.userconf import InitError, write_user_config

    try:
        dest = write_user_config(Path.cwd(), args.root, force=args.force, gpus=args.gpus)
    except (InitError, HostProfileError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    print(f"OK    wrote {dest} (data under {Path(args.root).resolve()})")
    return 0


def add_init_parsers(sub: argparse._SubParsersAction) -> None:
    from axis_qvla.data.userconf import DEFAULT_ROOT

    p = sub.add_parser("init", help="write ./qvla.toml: where this machine keeps the data, runs and environments")
    p.add_argument("--root", default=DEFAULT_ROOT, metavar="DIR",
                   help=f"directory every data root goes under (default ./{DEFAULT_ROOT})")
    p.add_argument("--force", action="store_true", help="replace an existing ./qvla.toml")
    p.add_argument("--gpus", type=int, default=1, metavar="N",
                   help="GPUs `qvla bench eval` and `qvla train` use by default (default 1)")
    p.set_defaults(func=_cmd_init)
