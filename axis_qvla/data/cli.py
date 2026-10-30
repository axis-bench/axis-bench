"""`qvla data get | verify`: fetch the published data through the release checksum manifest, or
compare a copy with it. `get base-weights` fetches the base models from their publishers, verified
against conf/recipes/bases.lock.json (axis_qvla.data.bases).

Core tier, stdlib only. The token and endpoint come from $HF_TOKEN / $HF_ENDPOINT and are never
printed. Exit codes: 0 clean, 1 mismatches (or, for `get`, bytes that disagree with the checksum
manifest), 2 configuration or I/O error, 3 (`get` only) the Hub (or a base's publisher) could not be
reached or answered a file the checksum manifest lists with an HTTP error (404 a commit the checksum manifest pins no
longer serves it, 401/403 a private repo without a valid token, ...).
"""
from __future__ import annotations

import argparse
import pathlib

from axis_qvla.core.lock import SOURCE_TREE, default_lock, load_lock
from axis_qvla.data.hf import HfHttpError, endpoint_from_env, token_from_env
from axis_qvla.data.verify import verify_local, verify_remote


def _cmd_verify(a: argparse.Namespace) -> int:
    try:
        lock = load_lock(a.lock)
        if a.local is not None:
            if not a.repo:
                print("FAIL  --local needs --repo")
                return 2
            problems = verify_local(lock, a.repo, a.local)
        else:
            problems = verify_remote(lock, token=token_from_env(), endpoint=endpoint_from_env())
    except (ValueError, OSError, HfHttpError) as exc:
        print(f"FAIL  {exc}")
        return 2
    for p in problems[:50]:
        print(f"DIFF  {p}")
    print(f"{'FAIL' if problems else 'OK'}    {len(problems)} problem(s) over {len(lock['files'])} files in the checksum manifest")
    return 1 if problems else 0


def _cmd_get(a: argparse.Namespace) -> int:
    import json

    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data import get, layout
    from axis_qvla.data.hf import HfDigestMismatch
    from axis_qvla.data.userconf import user_paths

    if a.serving and a.target != "base-weights":
        print("FAIL  --serving selects base weights: `qvla data get base-weights --serving <backbone>`")
        return 2
    if a.target == "base-weights":
        return _get_bases(a)
    if a.lock is None:
        a.lock = default_lock()
    if not pathlib.Path(a.lock).is_file():
        print("FAIL  " + get.LOCAL_ROUTE.format(lock=a.lock))
        return 2
    try:
        lock = load_lock(a.lock)
        cells = json.loads(pathlib.Path(a.cells).read_bytes()) if a.target.startswith("checkpoints/") else {}
        sels = get.resolve(a.target, cells, with_init=a.with_init)
        plans = [(s, get.rows_for(lock, s, a.include)) for s in sels]
        paths = None if a.dest else user_paths(a.host)
        token, endpoint = token_from_env(), endpoint_from_env()
        for s, rows in plans:
            dest = pathlib.Path(a.dest) if a.dest else layout.default_dest(s.repo, paths)
            get.fetch(lock, s, rows, dest, token=token, endpoint=endpoint)
    except HfDigestMismatch as exc:
        print(f"FAIL  differs from the release checksum manifest, refused and not kept: {exc}")
        return 1
    except HfHttpError as exc:
        print(f"FAIL  {exc}")
        return 3
    except (get.GetError, HostProfileError, ValueError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    return 0


def _get_bases(a: argparse.Namespace) -> int:
    """The base weights, from their publishers, verified against conf/recipes/bases.lock.json."""
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data import bases, layout
    from axis_qvla.data.hf import HfDigestMismatch
    from axis_qvla.data.userconf import user_paths

    try:
        lock = bases.load(a.lock or bases.LOCK)
        if a.serving:
            lock = bases.serving_lock(lock, a.serving)
        names = bases.select(lock, a.include)
        dest = pathlib.Path(a.dest) if a.dest else layout.base_weights_dest(user_paths(a.host))
        bases.fetch(lock, names, dest, hf_endpoint=endpoint_from_env(), gcs_endpoint=bases.GCS_ENDPOINT)
    except HfDigestMismatch as exc:
        print(f"FAIL  differs from the base weights' checksum list, refused and not kept: {exc}")
        return 1
    except HfHttpError as exc:
        print(f"FAIL  {exc}")
        return 3
    except (bases.BasesError, HostProfileError, ValueError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    return 0


def add_data_parsers(sub: argparse._SubParsersAction) -> None:
    d = sub.add_parser("data", help="get and verify the benchmark's data against the release checksum manifest")
    ds = d.add_subparsers(dest="subcommand", required=True)
    vf = ds.add_parser("verify", help="compare the checksum manifest with the Hub (no download) or a local tree")
    vf.add_argument("--checksums", dest="lock", metavar="FILE", type=pathlib.Path, default=default_lock(),
                    help="the release checksum manifest (default conf/release/v1.0.lock.json)")
    vf.add_argument("--local", type=pathlib.Path, default=None)
    vf.add_argument("--repo", default=None)
    vf.set_defaults(func=_cmd_verify)

    g = ds.add_parser("get", help="download a published data set, verified file by file against the release "
                                  "checksum manifest")
    g.add_argument("target", help="corpus | bench | results | realworld | traces-sample | base-weights | "
                                  "checkpoints/<result, run or checkpoint id>")
    g.add_argument("--include", action="append", default=[], metavar="GLOB",
                   help="only files whose path inside the target matches (repeatable), e.g. 'index/*'; "
                        "for base-weights, a base of conf/recipes/bases.toml, e.g. pi05_droid")
    g.add_argument("--with-init", action="store_true", help="checkpoints/...: also the checkpoint it was trained from")
    g.add_argument("--serving", choices=("pi05", "smolvla", "xvla"), default=None,
                   help="base-weights only: just what that backbone's reference policy server reads at start "
                        "(to evaluate a released checkpoint), not the training bases; a declared [roots] "
                        "hf_cache takes precedence over this copy for the HF-hosted bases")
    g.add_argument("--dest", default=None, help="directory to fill (default: from qvla.toml, docs/reference/data.md)")
    g.add_argument("--host", default=None, help="a qvla.toml path or a host profile name (default ./qvla.toml)")
    g.add_argument("--checksums", dest="lock", metavar="FILE", type=pathlib.Path, default=None,
                   help="the checksum manifest (default conf/release/v1.0.lock.json; for base-weights, "
                        "conf/recipes/bases.lock.json)")
    g.add_argument("--results-map", dest="cells", metavar="FILE", type=pathlib.Path,
                   default=SOURCE_TREE / "experiments" / "paper" / "cells.json",
                   help="the map of the paper's results to their checkpoints (default experiments/paper/cells.json)")
    g.set_defaults(func=_cmd_get)
