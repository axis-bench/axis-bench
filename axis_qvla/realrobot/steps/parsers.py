"""The real-world tutorial's steps as `qvla realrobot <step>`: one command per step, in the page's order.

Why: one command per tutorial step, with no path typed by hand.
Each step here reads and writes the fixed layout of axis_qvla.realrobot.layout, under the roots of
qvla.toml, and runs the existing tools in order:

    workspace     1  a workspace profile, from the reference one and your camera's calibration
    check-camera  2  the profile's outlines on a frame of your exterior camera
    rerender      3  the simulation demonstrations re-rendered for the profile, and their training view
    labels        4  phase curves, the curve store, the render's gates, the labels, the compiled labels
    pretrain      5  norm statistics, the stage-1 run specs, and `qvla train` of each
    demos         6  your DROID recordings converted, the training set and its norm statistics
    finetune      7  the stage-2 run specs and `qvla train` of each
    serve         8  the policy server of a fine-tuned run, checked against the run; the drive kit
    (the drive kit is an optional client for a DROID control stack: the page's "Connect your robot")

Stdlib only: the `qvla` parser imports this module, and each step's code is imported when it runs.

`workspace`'s subparser overrides argparse's private `_negative_number_matcher` so that a negative number
in scientific notation (e.g. -2.22e-16, how a real extrinsic's near-zero entries print) in
`--base-to-camera` parses as a value rather than an unrecognised option; it must not be removed.
"""
from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path


def _run(module: str, fn: str):
    """A `func` that imports `module` only when the step runs."""
    return lambda args: getattr(importlib.import_module(module), fn)(args)


def _host(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=None, help="this machine's qvla.toml (default ./qvla.toml)")


def _workspace(tools) -> None:
    p = tools.add_parser("workspace", help="step 1: a workspace profile from the reference one and your camera's calibration")
    p.add_argument("out", type=Path, help="the profile to write, e.g. my_workspace.toml")
    p.add_argument("--id", default=None, help="the workspace's name (default: the file's stem)")
    p.add_argument("--base-to-camera", type=float, nargs=12, metavar="V",
                   help="the exterior camera's extrinsic as OpenCV's solvePnP gives it (x_cam = R x_base + t): "
                        "R row by row, then t in metres")
    # argparse's own negative-number regex (^-\d+$|^-\d*\.\d+$) does not recognise scientific notation
    # (e.g. "-2.22e-16", which a real extrinsic's near-zero entries print as), so it would read such a
    # value as an unrecognised option and stop collecting the 12 values short. Extend it for this parser.
    p._negative_number_matcher = re.compile(r"^-\d*\.?\d+(?:[eE][+-]?\d+)?$")
    p.add_argument("--fy", type=float, default=None, help="the exterior camera's vertical focal length, in pixels")
    p.add_argument("--image-height", type=int, default=720, help="the height of the image --fy is of (default 720)")
    p.add_argument("--force", action="store_true", help="overwrite an existing profile")
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.render_steps", "workspace"))


def _check_camera(tools) -> None:
    p = tools.add_parser("check-camera", help="step 2: the profile's table and robot outlined on a frame of your exterior camera")
    p.add_argument("--profile", type=Path, default=None, help="your workspace profile (default: the reference one)")
    p.add_argument("--image", type=Path, required=True, help="a frame of your exterior camera")
    p.add_argument("--joints", type=float, nargs=7, default=None, metavar="Q",
                   help="the robot's joint positions when the frame was taken (DROID's joint_positions)")
    p.add_argument("--out", type=Path, default=Path("overlay.png"), help="the overlay (default overlay.png)")
    p.add_argument("--tasks", type=int, nargs="+", default=None, help="prepare only these tasks (default: every task)")
    _host(p)
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.render_steps", "check_camera"))


def _rerender(tools) -> None:
    p = tools.add_parser("rerender", help="step 3: re-render the simulation demonstrations for your workspace, and their training view")
    p.add_argument("--profile", type=Path, default=None, help="your workspace profile (default: the reference one)")
    p.add_argument("--tasks", type=int, nargs="+", default=None,
                   help="these tasks only; several processes with disjoint --tasks render into one layout")
    _host(p)
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.render_steps", "rerender"))


def _labels(tools) -> None:
    p = tools.add_parser("labels", help="step 4: phase curves, gates, labels and compiled labels of the re-render")
    p.add_argument("render", nargs="?", type=Path, default=None,
                   help="the re-render (default <cache>/derived/realworld/render); its siblings source/, d8/, "
                        "attempts/, curves/ and built/ are read and written")
    p.add_argument("--runs", nargs="+", default=["bc", "conditioning"], choices=("bc", "conditioning", "awr", "filtering"),
                   help="the stage-1 runs to label (default: bc and conditioning)")
    _host(p)
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.labels", "labels"))


def _demos(tools) -> None:
    p = tools.add_parser("demos", help="step 6: your DROID recordings converted to the fine-tuning set, with its norm statistics")
    p.add_argument("recordings", nargs="?", type=Path, default=None,
                   help="your DROID recordings (with aggregated-annotations-030724.json)")
    p.add_argument("--converted", type=Path, default=None, help="a dataset openpi's DROID converter already wrote")
    p.add_argument("--replace", action="store_true", help="write the training set again")
    p.add_argument("--print", action="store_true", help="print the converter's command; run nothing")
    _host(p)
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.demos", "demos"))


def _stage_parser(tools, name: str, help_: str, runs: str) -> None:
    p = tools.add_parser(name, help=help_)
    p.add_argument("--runs", nargs="+", default=None, metavar="RUN",
                   help=f"these runs only, short or in full (default: {runs})")
    p.add_argument("--print", action="store_true", help="write the run specs and print the training commands; train nothing")
    _host(p)
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.train_steps", name))


def _pretrain(tools) -> None:
    _stage_parser(tools, "pretrain", "step 5: pretrain on the re-render (norm statistics, run specs, qvla train)", "bc cond")


def _finetune(tools) -> None:
    _stage_parser(tools, "finetune", "step 7: fine-tune on your demonstrations (run specs, qvla train)", "bc cond only")


def _serve(tools) -> None:
    p = tools.add_parser("serve", help="step 8: serve a fine-tuned run, checked against the run")
    p.add_argument("run", help="the fine-tuning run: bc, cond or only (or its full id)")
    p.add_argument("--port", type=int, default=8000, help="the port the server listens on (default 8000)")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="seconds to wait for the server to load its checkpoint and listen (default 900); "
                        "once it listens, reading its handshake gets its own --timeout")
    p.add_argument("--check", metavar="HOST:PORT", default=None, help="check a server already running there; start nothing (writes no kit)")
    p.add_argument("--print", action="store_true", help="write the serve config and the drive kit, and print the server's command; start nothing")
    p.add_argument("--replace-kit", action="store_true",
                   help="overwrite a drive kit file already there, even where it differs from this checkout's (default: keep it, and say so)")
    _host(p)
    p.set_defaults(func=_run("axis_qvla.realrobot.steps.serve", "serve"))


STEPS = (_workspace, _check_camera, _rerender, _labels, _pretrain, _demos, _finetune, _serve)   # in the page's order


def add_step_parsers(tools) -> None:
    for add in STEPS:
        add(tools)
