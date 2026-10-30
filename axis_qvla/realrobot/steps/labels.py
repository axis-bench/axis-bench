"""Step 4 of the real-world tutorial: `qvla realrobot labels [RENDER]`, a re-render's quality labels in one command.

Why: one command for the five label tools: `realign phase`, `render labels`,
`render gates`, `realign build` and `realign compile`. They always run in that order on one render, and
each reads what the one before wrote, so the paths are the layout's (axis_qvla.realrobot.layout). RENDER
defaults to <cache>/derived/realworld/render. Its siblings source/, d8/, attempts/, curves/ and built/
are read and written, and the compiled labels go to <cache>/bundles, where `fr3 local-runs` reads them.

`realign phase` takes about 9 CPU-hours on the full data. So it runs task by task, and a task whose
phase curves are already in attempts/ is kept (each task's file is written whole or not at all); the
same command resumes. The other four stages take minutes and run every time. The command prints one
line per stage and names the compiled labels by run, never by digest.

Exit 0 done; 1 a gate failed (the render's or the labels'); 2 a refused input, naming the step that
writes it. Runs in the renderer's environment or the offline one (numpy, pandas, pyarrow, MuJoCo for
contacts, PyAV for the render's videos). Module scope is stdlib only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from axis_qvla.realrobot.steps.common import fail


def _phase(d8: Path, source: Path, attempts: Path) -> dict:
    """`realign phase`, task by task, keeping the tasks attempts/ already holds."""
    from axis_qvla.realrobot.realign.curves import read_attempt_store, write_attempt_store
    from axis_qvla.realrobot.realign.phase import phase_curves

    roots = d8 / "roots.json"
    tasks = sorted(int(t) for t in json.loads(roots.read_text(encoding="utf-8"))["roots"])
    attempts.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in attempts.iterdir()}
    todo = [t for t in tasks if f"task_{t}.npz" not in have]
    for t in todo:
        write_attempt_store(attempts, phase_curves(roots, source, [t]))
    wanted = set(tasks)
    got = [c for (tid, _aid), c in read_attempt_store(attempts).items() if tid in wanted]
    return {"episodes": len(got), "neutral": sum(c.t0 is None for c in got),
            "computed": len(todo), "kept": len(tasks) - len(todo)}


def _curves(d8: Path, attempts: Path, curves: Path) -> int:
    from axis_qvla.realrobot.realign.curves import read_attempt_store
    from axis_qvla.realrobot.render.labels import render_curve_store

    return render_curve_store(d8, read_attempt_store(attempts), curves)


def _gates(render: Path, d8: Path, curves: Path, attempts: Path) -> None:
    from axis_qvla.realrobot.render.gates import run_gates

    run_gates(render, droid8d_root=d8, render_store=curves, attempt_curves_dir=attempts)


def labels(args) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.realrobot.layout import layout
    from axis_qvla.realrobot.realign import gates as realign_gates
    from axis_qvla.realrobot.realign.bundles import write_bundles
    from axis_qvla.realrobot.realign.cli import OVERRIDE_REFUSAL, build
    from axis_qvla.realrobot.render import gates as render_gates
    from axis_qvla.signals.methods.awr_chunk import CORPUS_ROOT_ENV

    if os.environ.get(CORPUS_ROOT_ENV):            # refused before 9 CPU-hours of phase, not after
        return fail(OVERRIDE_REFUSAL)
    try:
        lay = layout(args.host)
    except HostProfileError as exc:
        return fail(str(exc))
    render = Path(args.render) if args.render is not None else lay.render
    d8, source, attempts, curves, built = (render.parent / n for n in ("d8", "source", "attempts", "curves", "built"))
    runs = tuple(args.runs)
    try:
        for need in (render, d8 / "roots.json", source / "manifest.json"):
            if not need.exists():
                return fail(f"no {need}: run `qvla realrobot rerender` first")
        ph = _phase(d8, source, attempts)
        print(f"phase curves: {ph['episodes']} episodes ({ph['neutral']} neutral); {ph['computed']} tasks computed, "
              f"{ph['kept']} kept -> {attempts}")
        print(f"curve store: {_curves(d8, attempts, curves)} episodes -> {curves}")
        _gates(render, d8, curves, attempts)
        print(f"all hard gates pass; {render / 'render_gates.json'}")
        s = build(built, roots=d8 / "roots.json", curves_store=curves, runs=runs)
        print(f"labels of {' and '.join(runs)}: {s['episodes']} episodes, {s['neutral']} neutral -> {built}")
        from axis_qvla.realrobot.fr3.local_runs import labels_name

        names = sorted(labels_name(k) for k in write_bundles(built, lay.bundles, runs=runs))
        print(f"compiled labels {', '.join(names)} -> {lay.bundles}")
    except (render_gates.GateError, realign_gates.GateError) as exc:
        return fail(str(exc), 1)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:     # CurveError and DataError are ValueErrors
        return fail(str(exc))
    return 0
