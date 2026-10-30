"""Steps 5 and 7 of the real-world tutorial: `qvla realrobot pretrain` and `qvla realrobot finetune`.

One command per stage. A stage's runs are fixed by the line record
(experiments/realrobot/line.toml): stage 1 has `bc` and `cond`, stage 2 has `bc`, `cond` and `only`
(by default). Each step does three things:
- it computes its stage's norm statistics where the runs read them, or keeps them when they still
  describe the data;
- it writes the run specs that name this machine's inputs by name (`fr3 local-runs`);
- it trains each run with `qvla train <run spec>`, which trains in the backbone's environment and
  names the `qvla env build` command when that environment is missing.
`--print` prints the training commands instead. `--runs` names a subset, short (`cond`) or in full.

`qvla train` runs as a child process of this interpreter (`_run_child`), not in-process: it is another
group's command, and a child keeps its argparse exits, its environment changes and its exit code its
own. Its exit code is passed through, and the first run that fails stops the stage.

Module scope is stdlib only.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from axis_qvla.realrobot.layout import LINE, REPO, layout, run_id
from axis_qvla.realrobot.steps.common import ensure_norm, fail

DEFAULTS = {"pretrain": ("bc", "cond"), "finetune": ("bc", "cond", "only")}
NEEDS = {"pretrain": ("d8", "qvla realrobot rerender"), "finetune": ("demos", "qvla realrobot demos <your DROID recordings>")}
QVLA = "axis_qvla.cli"           # `qvla`'s module


def _run_child(argv: list[str]) -> int:
    """Run `argv` with this checkout importable, in the current directory (so a relative --host
    resolves as the user typed it), and return its exit code."""
    path = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH", "")) if p)
    sys.stdout.flush()              # this step's lines first, then the child's
    return subprocess.run(argv, env={**os.environ, "PYTHONPATH": path}, check=False).returncode


def _train(spec: Path, *, host: str | None, print_only: bool) -> int:
    """`qvla train <spec> [--host H] [--print]`: trains in the backbone's environment; `--print` prints the command."""
    return _run_child([sys.executable, "-m", QVLA, "train", str(spec), *(["--host", host] if host else []),
                       *(["--print"] if print_only else [])])


def _stage(args, stage: str) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.realrobot.fr3.line import load_line
    from axis_qvla.realrobot.fr3.local_runs import write_local_runs

    runs = [run_id(stage, r) for r in (args.runs or DEFAULTS[stage])]
    try:
        line = load_line(LINE)
        other = [r for r in runs if line.runs.get(r, {}).get("stage") != stage]
        if other:
            return fail(f"{', '.join(other)}: not a {stage} run of the line record (the {stage} runs: "
                        f"{', '.join(DEFAULTS[stage])})")
        lay = layout(args.host)
        data_dir, writer = NEEDS[stage]
        if not getattr(lay, data_dir).is_dir():
            return fail(f"no {getattr(lay, data_dir)}: run `{writer}` first")
        ensure_norm(lay, stage)
        written = write_local_runs(lay.paths, line, runs=runs)
    except (ValueError, OSError, HostProfileError) as exc:
        return fail(str(exc))
    for rid, path in written.items():
        print(f"OK    {rid} -> {path}")
    for i, (rid, path) in enumerate(written.items(), 1):
        if not args.print:
            print(f"training {rid} ({i} of {len(written)})", flush=True)
        rc = _train(path, host=args.host, print_only=args.print)
        if rc != 0:
            return rc
    return 0


def pretrain(args) -> int:
    return _stage(args, "pretrain")


def finetune(args) -> int:
    return _stage(args, "finetune")
