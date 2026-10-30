"""`python -m axis_qvla.realrobot.render {source,prepare,render,droid8d,labels,gates,report,overlay}`: the kinematic
renderer.

    source   --trace T --scene-dir D --scene-xml REL --task-id N --attempt-id N --prompt S --out EP.json
                                                                 a source_episode.json from a recorded trace
    prepare  [--data DIR] [--out DIR] [--host qvla.toml] [--tasks N ...]   the render input tree of the simulation
                                                                 demonstrations in qvla-v1-realworld
    render   (--episodes EP.json ... | --data DIR [--tasks N ...]) [--robots DIR] --out DIR [--profile WS.toml]
             one LeRobot v3 task per task id, and roots.json indexing every task in --out
             (--robots DIR holds robotiq_2f85/: axis-render's assets/robots, or a Menagerie checkout at the locked commit)
             (--gripper-xml FILE --unlocked-gripper: any 2F-85-shaped model, recorded as unlocked)
    droid8d  --src DIR --out DIR                                 the DROID-8D training view + roots.json
    labels   --pretraining-data DIR --attempt-curves DIR --out DIR   the curve store `realign build` reads
             (--export-curves DIR --alignment-map FILE instead of --attempt-curves: the curves are first keyed by
              attempt into DIR/attempts/, inside --out, which the curve store's readers do not glob)
    gates    --out DIR [--droid8d DIR --curves DIR --attempt-curves DIR]   hard gates -> render_gates.json
    report   --task DIR --episode N --ref-task DIR --ref-episode N --out FILE   reported metrics
    overlay  --image FRAME --scene SCENE.xml [--robots DIR] --out PNG [--profile WS.toml] [--joints q0..q6]

`--robots` defaults to this checkout's third_party/axis-render/assets/robots; `prepare`'s paths default
to <corpus>/qvla-v1-realworld and <cache>/derived/realworld/source. `render` refuses with a `SystemExit`
naming `--robots` and that default path when `--robots` is omitted and the default is missing or is not
the locked gripper; pass `--robots` explicitly to use a different checkout.

Render tier; this module is stdlib only and imports the pipeline inside each command. Exit 0 on success;
a refused input is one `FAIL <message>` line on stderr and exit 2 (`source` when --out is not in
--scene-dir; `prepare` and `render --data` when the simulation demonstrations in qvla-v1-realworld lack
what they need; `render` on a profile, source episode, scene or gripper it cannot use, a --out that
keeps another workspace profile, or a half-written task directory; `droid8d` into a task that exists; `overlay` on a profile, scene or
image it cannot use; `gates` when --droid8d, --curves and --attempt-curves are not given together, or
on an input it refuses). `render` renders one task at a time into `task_<tid>.partial/`, renamed
`task_<tid>/` when whole, keeps the tasks --out already holds (so the same command resumes), and when
the IK refuses a task, renders the others, prints a FAIL line per failed task and exits 1. `gates`
exits 1 when a hard gate fails, after writing render_gates.json, and on a pass says on stderr what it
did not certify (an unlocked gripper; tags not built for a constant-quality corpus).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from axis_qvla.realrobot.render import profile, rig


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _source(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import source
    if Path(a.out).resolve().parent != Path(a.scene_dir).resolve():
        print(f"FAIL --out {a.out} is not in --scene-dir {a.scene_dir}: the episode file names its scene "
              f"relative to its own directory", file=sys.stderr)
        return 2
    doc = source.from_trace(a.trace, task_id=a.task_id, attempt_id=a.attempt_id, prompt=a.prompt,
                            scene_xml=a.scene_xml, scene_dir=a.scene_dir)
    source.write_source_episode(a.out, doc)
    print(f"{doc['trajectory_data']['n_steps']} source rows -> {a.out}")
    return 0


def _prepare(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import data
    if a.data is None or a.out is None:
        from axis_qvla.core.hosts import HostProfileError
        from axis_qvla.realrobot.layout import layout
        try:
            lay = layout(a.host)
        except HostProfileError as exc:
            print(f"FAIL {exc}", file=sys.stderr)
            return 2
        a.data, a.out = a.data or lay.data, a.out or lay.source
    try:
        done = data.prepare(a.data, a.out, a.tasks)
    except data.DataError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"{_count(done['tasks'], 'task')}, {_count(done['episodes'], 'episode')}, "
          f"{_count(done['layouts'], 'mesh layout')} -> {a.out}")
    return 0


def _render(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import assets, data, pipeline, source
    if a.robots is None and not (a.gripper_xml is not None and a.unlocked_gripper):
        raise SystemExit("give --robots (a directory holding robotiq_2f85/, checked against menagerie.lock.json), "
                         "or --gripper-xml with --unlocked-gripper")
    # Every refusal is a ValueError (DataError, ProfileError, SourceError, AssemblyError, AssetError, another
    # workspace profile in --out) or an OSError (a task directory that exists half-written); an IK refusal
    # costs only its task and is reported below.
    try:
        ws = profile.load_profile(a.profile)
        if a.robots is not None:
            try:
                gripper_xml, lock = assets.check_menagerie(a.robots), json.loads(assets.LOCK.read_text())["commit"]
            except (ValueError, OSError):
                if getattr(a, "robots_from_layout", False):
                    raise SystemExit(f"--robots was not given, and the layout default {a.robots} is not a "
                                     f"usable, locked Menagerie checkout; pass --robots explicitly") from None
                raise
        else:
            gripper_xml, lock = a.gripper_xml, None
        if a.data is not None:
            manifest = data.load_manifest(a.data)
            data.check_prepared(a.data, manifest, a.tasks)
            tasks = data.task_episode_files(a.data, manifest, a.tasks)
        else:
            tasks = {}
            for p in a.episodes:
                tasks.setdefault(source.episode_task_id(p), []).append(p)
        done = pipeline.render_tasks(tasks, gripper_xml, a.out, profile=ws, gripper_lock=lock)
    except (ValueError, OSError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    if done.kept:
        print(f"already rendered: tasks {done.kept} (kept)")
    print(f"rendered tasks {done.rendered} -> {a.out}")
    if done.failed:
        for msg in done.failed.values():
            print(f"FAIL {msg}", file=sys.stderr)
        print(f"{_count(len(done.failed), 'task')} not rendered ({', '.join(map(str, done.failed))}): "
              f"their recordings are beyond the IK's acceptance; the rendered tasks are kept", file=sys.stderr)
        return 1
    return 0


def _droid8d(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import droid8d, report
    try:
        roots = droid8d.build_corpus(a.src, a.out)
    except (OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"roots index: {roots}")
    v = report.droid8d_velocity(a.out)
    print(f"joint velocity at the +-1 bound (reported): {v['velocity_at_bound']} entries on {v['rows_at_bound']} "
          f"of {v['rows']} rows")
    return 0


def _labels(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import labels
    attempts = a.attempt_curves
    if a.export_curves is not None:
        attempts = a.out / "attempts"
        labels.attempt_store_from_export(a.export_curves, a.alignment_map, attempts)
    n = labels.render_curve_store(a.corpus, labels.read_attempt_store(attempts), a.out)
    print(f"{n} episodes -> {a.out}")
    return 0


def _gates(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import gates
    try:
        report = gates.run_gates(a.out, droid8d_root=a.droid8d, render_store=a.curves,
                                 attempt_curves_dir=a.attempt_curves)
    except gates.GateError as exc:
        print(f"FAIL {exc}; see {a.out / 'render_gates.json'}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"all hard gates pass; {a.out / 'render_gates.json'}")
    # What a pass did not certify, said where a reader of the exit status looks.
    if not report["gripper_lock"]["locked"]:
        print("note: the gripper is not the locked 2F-85 (rendered with --unlocked-gripper); the gripper gate "
              "certifies only the locked model", file=sys.stderr)
    why = report.get("label_tag", {}).get("tags_not_built")
    if why:
        print(f"note: label_tag compared ranges and weights only, no tags: {why}", file=sys.stderr)
    return 0


def _report(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import report
    print(json.dumps(report.write_report(a.task, a.episode, a.ref_task, a.ref_episode, a.out), indent=1))
    return 0


def _overlay(a: argparse.Namespace) -> int:
    from axis_qvla.realrobot.render import assets, overlay
    try:
        ws = profile.load_profile(a.profile)
        rgb, masks = overlay.render_view(a.scene, assets.check_menagerie(a.robots), ws, a.joints)
        overlay.write_png(a.out, overlay.compose(overlay.fit(overlay.read_image(a.image)), rgb, masks))
    except (ValueError, OSError) as exc:          # ProfileError, AssetError, AssemblyError are ValueErrors
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    print(f"overlay of {a.image.name} on workspace {ws.id} -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m axis_qvla.realrobot.render", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("source", help="a source_episode.json from a recorded trace")
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--scene-dir", type=Path, required=True, help="directory the source_episode.json will sit in")
    p.add_argument("--scene-xml", required=True, help="the scene MJCF, relative to --scene-dir")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--attempt-id", type=int, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_source)
    p = sub.add_parser("prepare", help="the render input tree of the simulation demonstrations in qvla-v1-realworld")
    p.add_argument("--data", type=Path, default=None,
                   help="the downloaded simulation demonstrations in qvla-v1-realworld (default: from qvla.toml)")
    p.add_argument("--out", type=Path, default=None, help="the render input tree, beside --data (default: from qvla.toml)")
    p.add_argument("--tasks", type=int, nargs="+", help="task ids (default: every task of --data)")
    p.add_argument("--host", default=None, help="this machine's qvla.toml (default ./qvla.toml)")
    p.set_defaults(func=_prepare)
    p = sub.add_parser("render", help="render source episodes to LeRobot v3 tasks")
    p.add_argument("--episodes", type=Path, nargs="+")
    p.add_argument("--data", type=Path, help="a prepared real-world tree (render prepare --out)")
    p.add_argument("--tasks", type=int, nargs="+", help="task ids of --data (default: every task)")
    p.add_argument("--robots", type=Path, help="a directory holding robotiq_2f85/, checked against the lock")
    p.add_argument("--gripper-xml", type=Path)
    p.add_argument("--unlocked-gripper", action="store_true")
    p.add_argument("--profile", type=Path, default=profile.REFERENCE,
                   help="the workspace profile (default: the reference workspace)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_render)
    p = sub.add_parser("droid8d", help="the DROID-8D training view of a render")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_droid8d)
    p = sub.add_parser("labels", help="attempt-keyed phase curves -> the pretraining data's curve store")
    p.add_argument("--pretraining-data", dest="corpus", type=Path, required=True)
    p.add_argument("--attempt-curves", type=Path)
    p.add_argument("--export-curves", type=Path, help="curves keyed by export index, keyed by attempt into <out>/attempts/ first")
    p.add_argument("--alignment-map", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_labels)
    p = sub.add_parser("gates", help="the hard gates -> render_gates.json")
    p.add_argument("--out", type=Path, required=True, help="the render's output root (task_*/ + roots.json)")
    p.add_argument("--droid8d", type=Path)
    p.add_argument("--curves", type=Path)
    p.add_argument("--attempt-curves", type=Path)
    p.set_defaults(func=_gates)
    p = sub.add_parser("report", help="reported metrics against a reference render of the same attempt")
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--ref-task", type=Path, required=True)
    p.add_argument("--ref-episode", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_report)
    p = sub.add_parser("overlay", help="the outlines of the rendered table and robot on a real camera frame")
    p.add_argument("--profile", type=Path, default=profile.REFERENCE,
                   help="the workspace profile (default: the reference workspace)")
    p.add_argument("--image", type=Path, required=True, help="a frame of the real exterior camera")
    p.add_argument("--scene", type=Path, required=True,
                   help="a recorded scene, e.g. a variant_<i>.xml of the prepared simulation demonstrations in qvla-v1-realworld")
    p.add_argument("--robots", type=Path, default=None, help="a directory holding robotiq_2f85/, checked against "
                   "the lock (default: third_party/axis-render/assets/robots)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--joints", type=float, nargs=7, default=rig.PIXELMATCH_HOME,
                   help="the robot's joint positions when the frame was taken")
    p.set_defaults(func=_overlay)
    a = ap.parse_args(argv)
    a.robots_from_layout = False
    if a.cmd in ("render", "overlay") and a.robots is None and getattr(a, "gripper_xml", None) is None:
        from axis_qvla.realrobot.layout import ROBOTS
        a.robots, a.robots_from_layout = ROBOTS, True
    if a.cmd == "render":
        if (a.episodes is None) == (a.data is None):
            ap.error("render needs --episodes, or --data (a prepared real-world tree)")
        if a.tasks is not None and a.data is None:
            ap.error("render: --tasks selects tasks of --data")
    if a.cmd == "labels":
        by_attempts = a.attempt_curves is not None and a.export_curves is None and a.alignment_map is None
        by_export = a.attempt_curves is None and a.export_curves is not None and a.alignment_map is not None
        if not (by_attempts or by_export):
            ap.error("labels needs --attempt-curves, or --export-curves with --alignment-map")
    return a.func(a)
