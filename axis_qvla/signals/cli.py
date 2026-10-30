"""`qvla labels`: compile the paper's labels, export compiled labels' legacy files, check a compiled-labels tree.

Core tier at import (stdlib only): building the `qvla` parser imports this module under a bare
interpreter, so each command imports the offline-tier modules it runs. Exit codes: 0 clean,
1 problems found (check), 2 configuration or I/O error, or an input the build refuses (one FAIL
line naming it, never a traceback).

    qvla labels build-paper --pretraining-data <qvla-v1-corpus> --episodes <qvla-v1-corpus>/episodes --out <dir>
    qvla labels export <compiled labels dir> --format lerobot-legacy --out <dir>
    qvla labels check <compiled labels root>
"""
from __future__ import annotations

import argparse
import pathlib

FORMATS = ("openpi-legacy", "lerobot-legacy")


def _cmd_build(a: argparse.Namespace) -> int:
    from axis_qvla.signals import paper
    from axis_qvla.signals.program import RefusedDtype, Unscorable

    try:
        man = paper.build(pack=a.pack, corpus=a.corpus, out=a.out, only=set(a.only) if a.only else None)
    except (ValueError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    # These two are KeyError/TypeError *subclasses* raised only at the declaration- and lock-input
    # sites `build` means to refuse by name (axis_qvla.signals.paper.DeclaredFieldMissing,
    # axis_qvla.signals.program.RefusedDtype): a bare KeyError or TypeError raised anywhere else (a bug,
    # not a refusal) is not one of these and is not caught here, so it keeps its traceback.
    except paper.DeclaredFieldMissing as exc:
        print(f"FAIL  KeyError: {exc} is missing (an edited declaration, or a checksum-manifest or input record without it)")
        return 2
    except RefusedDtype as exc:
        print(f"FAIL  TypeError: {exc}")
        return 2
    except Unscorable as exc:       # an episode a declared scorer cannot score
        print(f"FAIL  Unscorable: {exc}")
        return 2
    print(f"OK    {len(man['bundles'])} compiled labels, {len(man['rowspaces'])} frame indexes, "
          f"{len(man['arms'])} runs -> {a.out}")
    return 0


def _cmd_export(a: argparse.Namespace) -> int:
    from axis_qvla.signals.bundle import export_legacy, load_bundle

    try:
        paths = export_legacy(load_bundle(a.bundle), a.format, a.out)
    except (ValueError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    for p in paths:
        print(p)
    return 0


def _cmd_check(a: argparse.Namespace) -> int:
    from axis_qvla.signals import paper

    try:
        problems = paper.check(a.bundles)
    except (ValueError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    for p in problems[:50]:
        print(f"DIFF  {p}")
    print(f"{'FAIL' if problems else 'OK'}    {len(problems)} problem(s)")
    return 1 if problems else 0


def add_signals_parsers(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("labels", help="compile, export and check the paper's quality labels and training weights")
    ss = s.add_subparsers(dest="subcommand", required=True)
    b = ss.add_parser("build-paper", help="compile every paper run's labels from the published pretraining data")
    b.add_argument("--pretraining-data", dest="pack", metavar="DIR", type=pathlib.Path, required=True,
                   help="the pretraining data, a qvla-v1-corpus tree (SIGNALS.lock.json, index/, signals/v1/)")
    b.add_argument("--episodes", dest="corpus", metavar="DIR", type=pathlib.Path, required=True,
                   help="its episodes/ directory (qvla-v1-corpus/episodes)")
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.add_argument("--only", action="append", default=None,
                   help="compile only these labels (repeatable, a name of conf/signals/paper_bundles.toml); "
                        "no manifest is written")
    b.set_defaults(func=_cmd_build)
    e = ss.add_parser("export", help="write compiled labels' legacy files, byte-equal to what the paper's runs read")
    e.add_argument("bundle", metavar="COMPILED", type=pathlib.Path)
    e.add_argument("--format", choices=FORMATS, required=True)
    e.add_argument("--out", type=pathlib.Path, required=True)
    e.set_defaults(func=_cmd_export)
    c = ss.add_parser("check", help="verify a compiled-labels tree against the paper's declaration and re-export "
                                    "every file the checksum manifest lists from it, byte for byte")
    c.add_argument("bundles", metavar="COMPILED", type=pathlib.Path)
    c.set_defaults(func=_cmd_check)
