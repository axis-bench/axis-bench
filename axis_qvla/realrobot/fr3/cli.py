"""`python -m axis_qvla.realrobot.fr3 {check,norm,local-runs,serve-config}`: the FR3 line's declarations,
checked, its norm statistics, the run specs that train it, and the config a run is served with.

    check       [--line FILE] [--runs DIR] [--presets]
                [--run ID --handshake FILE --dataset DIR [--embodiment EMBODIMENT]]   a finetune run's server
                [--norm FILE --dataset DIR [--embodiment EMBODIMENT]]                 a norm-statistics file
    norm        --dataset DIR --embodiment EMBODIMENT [--out FILE] [--host qvla.toml]
    local-runs  [--host qvla.toml] [--line FILE] [--runs ID ...]
    serve-config --run FILE [--out FILE] [--host qvla.toml]

EMBODIMENT is fr3, realalign_sim or fr3_droid.

`check` always checks the line record, its ArmSpecs and every run's units; `--presets` adds the recipe
presets, `--handshake` a policy server's handshake (a JSON file, or a `qvla bench check-server
--report` file) against `--run`, served from the finetune set `--dataset` (the FR3 finetune set, or with
`--embodiment fr3_droid` your demonstrations in the training layout), and `--norm` a norm file against
the dataset it should describe. Each problem is printed as a FAIL line; exit 0 when there are
none, 1 when there are. A flag that only means something beside another (`--run`, `--dataset`,
`--embodiment`) is refused without it (exit 2), never ignored. `norm` writes openpi's norm_stats.json
layout over every frame of `--dataset`, and refuses (exit 2, naming it) a directory that is not a
training set of that embodiment. `norm --out` defaults to the file `local-runs` reads for a
realalign_sim or fr3_droid set, and `serve-config --out` to <runs>/<run id>/serve.json. `local-runs`
writes, for each run of the line record (or each of `--runs`), the run spec `qvla train` trains: the
shipped one with its frame index, compiled labels and norm statistics named by content, read from this
machine's local layout (axis_qvla.realrobot.fr3.local_runs); it prints `OK    <run> -> <run spec>` per run,
and refuses (exit 2, naming it) a run whose inputs are not built. `serve-config` writes the openpi
config a policy server is started with for a checkpoint of the run spec `--run` (one `local-runs`
wrote): the run's training config with nothing bound, so its image crop, action horizon and action
units are the ones it trained with; it refuses (exit 2, naming it) a run spec that names an input
`local:`.

Offline tier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EMBODIMENTS = ("fr3", "realalign_sim", "fr3_droid")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.realrobot.fr3", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the line record, its ArmSpecs, presets, a server's handshake, a norm file")
    c.add_argument("--line", type=Path, default=REPO / "experiments" / "realrobot" / "line.toml")
    c.add_argument("--runs", type=Path, default=REPO / "experiments" / "realrobot" / "runs")
    c.add_argument("--presets", action="store_true", help="also check the recipe presets")
    c.add_argument("--run", help="the finetune run a --handshake serves")
    c.add_argument("--handshake", type=Path, help="a handshake JSON, or a check-server --report file")
    c.add_argument("--dataset", type=Path, help="the dataset a --handshake or --norm is checked against")
    c.add_argument("--norm", type=Path, help="a norm_stats.json to check")
    c.add_argument("--embodiment", default=None, choices=EMBODIMENTS,
                   help="the --dataset's embodiment (default fr3)")
    n = sub.add_parser("norm", help="norm statistics over a training set, in the model's state order")
    n.add_argument("--dataset", type=Path, required=True)
    n.add_argument("--embodiment", required=True, choices=EMBODIMENTS)
    n.add_argument("--out", type=Path, default=None,
                   help="default: where `fr3 local-runs` reads this embodiment's statistics")
    n.add_argument("--host", default=None, help="a qvla.toml path or a host profile name (default ./qvla.toml)")
    lr = sub.add_parser("local-runs", help="run specs that name this machine's real-world inputs by content")
    lr.add_argument("--host", default=None, help="a qvla.toml path or a host profile name (default ./qvla.toml)")
    lr.add_argument("--line", type=Path, default=REPO / "experiments" / "realrobot" / "line.toml")
    lr.add_argument("--runs", nargs="+", help="these runs only (default: every run of the line record)")
    sc = sub.add_parser("serve-config", help="the openpi config a policy server of a real-world run is started with")
    sc.add_argument("--run", type=Path, required=True, help="a run spec `local-runs` wrote")
    sc.add_argument("--out", type=Path, default=None, help="default: <runs>/<run id>/serve.json")
    sc.add_argument("--host", default=None, help="a qvla.toml path or a host profile name (default ./qvla.toml)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    if a.cmd == "check" and a.handshake is not None and (a.run is None or a.dataset is None):
        ap.error("--handshake needs --run and --dataset")
    if a.cmd == "check" and a.norm is not None and a.dataset is None:
        ap.error("--norm needs --dataset")
    if a.cmd == "check" and a.run is not None and a.handshake is None:
        ap.error("--run needs --handshake (the run is the one a server's handshake is checked against)")
    if a.cmd == "check" and a.dataset is not None and a.handshake is None and a.norm is None:
        ap.error("--dataset needs --handshake or --norm (it is what they are checked against)")
    if a.cmd == "check" and a.embodiment is not None and a.norm is None and a.handshake is None:
        ap.error("--embodiment needs --norm or --handshake (it is the --dataset's embodiment)")
    if a.cmd == "check" and a.embodiment is None:
        a.embodiment = "fr3"

    if a.cmd == "norm" and a.out is None:
        from axis_qvla.core.hosts import HostProfileError
        from axis_qvla.realrobot.layout import STAGE_OF, layout
        if a.embodiment not in STAGE_OF:
            ap.error(f"norm --embodiment {a.embodiment}: give --out (only {', '.join(sorted(STAGE_OF))} "
                     f"have a place in the local layout)")
        try:
            a.out = layout(a.host).norm_file(STAGE_OF[a.embodiment])
        except (ValueError, OSError, HostProfileError) as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2

    from axis_qvla.realrobot.fr3 import checks, norm
    from axis_qvla.realrobot.fr3.line import LineError, load_line

    if a.cmd == "local-runs":
        from axis_qvla.core.hosts import HostProfileError
        from axis_qvla.data.userconf import user_paths
        from axis_qvla.realrobot.fr3.local_runs import write_local_runs
        try:
            written = write_local_runs(user_paths(a.host), load_line(a.line), runs=a.runs)
        except (ValueError, OSError, HostProfileError) as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2
        for run_id, path in written.items():
            print(f"OK    {run_id} -> {path}")
        return 0
    if a.cmd == "serve-config":
        from axis_qvla.core.hosts import HostProfileError
        from axis_qvla.data.userconf import user_paths
        from axis_qvla.realrobot.fr3.local_runs import write_serve_config
        try:
            run_id, out = write_serve_config(a.run, a.out, user_paths(a.host))
        except (ValueError, OSError, HostProfileError) as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2
        print(f"serve config of {run_id} -> {out}")
        return 0
    if a.cmd == "norm":
        try:
            stats = norm.compute_norm(a.dataset, a.embodiment)
        except ValueError as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 2
        norm.write_norm(a.out, stats)
        print(f"norm statistics of {a.dataset.name} ({a.embodiment}) -> {a.out}")
        return 0
    try:
        ln = load_line(a.line)
    except LineError as exc:
        print(f"FAIL  {exc}")
        return 1
    problems = checks.check_record(ln) + checks.check_runs(ln, a.runs)
    problems += [p for run in sorted(ln.runs) for p in checks.check_units(ln, run)]
    if a.presets:
        problems += checks.check_presets(ln)
    if a.handshake is not None:
        try:
            handshake = json.loads(a.handshake.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{a.handshake}: cannot read the handshake ({exc.__class__.__name__}: {exc})")
        else:
            problems += checks.check_serve(handshake, ln, a.run, a.dataset, a.embodiment)
    if a.norm is not None:
        problems += checks.check_norm(a.norm, a.dataset, a.embodiment)
    problems = list(dict.fromkeys(problems))     # one line per problem, however many runs share it
    for p in problems:
        print(f"FAIL  {p}")
    print("ALL PASS" if not problems else f"{len(problems)} problem{'s' * (len(problems) != 1)}")
    return 1 if problems else 0
