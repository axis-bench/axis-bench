"""Steps 1 to 3 of the real-world tutorial: the workspace profile, its check against your camera, the re-render.

Why: step 1 was a copy of the reference profile plus three formulas the user applied by hand. They
inverted an OpenCV extrinsic, turned the camera 180 degrees about x, and converted a focal length to
fovy. A mistake in any of them shows up only as a misplaced overlay. `workspace` writes the profile
from the calibration itself.

`check-camera` overlays the profile on the first recorded scene of the data. `rerender` renders every
task, or `--tasks`, and writes the training view task by task, so the same command resumes.

`workspace` is stdlib. `check-camera` and `rerender` run the renderer (MuJoCo, mink, PyAV) inside their
functions. Module scope is stdlib only.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from axis_qvla.realrobot.steps.common import fail


def quat_from_matrix(m) -> list[float]:
    """(w, x, y, z), w >= 0, of a 3 x 3 rotation matrix given as rows (Shepperd's method)."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    if q[0] < 0:
        q = [-v for v in q]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def camera_pose_from_opencv(r: list[float], t: list[float]) -> tuple[list[float], list[float]]:
    """The profile's camera `pos` and `quat` from an OpenCV extrinsic that maps base-frame points into the
    camera, x_cam = R x_base + t (`r` row by row). The camera's pose in the base frame is (Rᵀ, -Rᵀ t).
    MuJoCo's camera looks along -z with y up, OpenCV's along +z with y down, so MuJoCo's axes are OpenCV's
    turned 180° about x: the columns (c1, -c2, -c3)."""
    rows = [r[0:3], r[3:6], r[6:9]]
    cam_to_base = [[rows[j][i] for j in range(3)] for i in range(3)]
    pos = [-sum(cam_to_base[i][j] * t[j] for j in range(3)) for i in range(3)]
    mujoco_axes = [[cam_to_base[i][0], -cam_to_base[i][1], -cam_to_base[i][2]] for i in range(3)]
    return pos, quat_from_matrix(mujoco_axes)


def fovy_from_focal(fy: float, height: int) -> float:
    """The vertical field of view in degrees of a pinhole camera of focal length `fy` pixels on an image
    `height` pixels high."""
    return math.degrees(2 * math.atan(height / (2 * fy)))


def _set(text: str, pattern: str, line: str) -> str:
    """`text` with the first line matching `pattern` replaced by `line`. ValueError when none matches (the
    reference profile changed shape)."""
    new, n = re.subn(pattern, lambda _m: line, text, count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"the reference profile has no line matching {pattern!r}")
    return new


def _vec(xs) -> str:
    return "[" + ", ".join(repr(float(v)) for v in xs) + "]"


def workspace(args) -> int:
    from axis_qvla.realrobot.render import profile

    out = Path(args.out)
    if out.exists() and not args.force:
        return fail(f"{out} exists; pass --force to overwrite it")
    ws_id = args.id or out.stem
    text = profile.REFERENCE.read_text(encoding="utf-8")
    text = _set(text, r"^id = .*$", f'id = "{ws_id}"')
    text = _set(text, r"^about = .*$", f'about = "Workspace {ws_id}, written from the reference profile by '
                                       f'`qvla realrobot workspace`."')
    done = []
    if args.base_to_camera is not None:
        pos, quat = camera_pose_from_opencv(args.base_to_camera[:9], args.base_to_camera[9:])
        text = _set(text, r"^pos = \[[^\]]*\]", f"pos = {_vec(pos)}")       # the first `pos` is [camera]'s
        text = _set(text, r"^quat = \[[^\]]*\]", f"quat = {_vec(quat)}")
        done.append("camera pose")
    if args.fy is not None:
        text = _set(text, r"^fovy = [0-9.eE+-]+", f"fovy = {fovy_from_focal(args.fy, args.image_height)!r}")
        done.append("field of view")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    try:
        profile.load_profile(out)
    except profile.ProfileError as exc:
        out.unlink()
        return fail(str(exc))
    left = "table, background and fixtures" + ("" if done else ", camera pose and field of view")
    print(f"workspace {ws_id} -> {out} ({' and '.join(done) or 'the reference camera'}; set its {left} as measured)")
    return 0


def prepare_source(lay, tasks=None) -> Path:
    """The render input tree of `tasks` (default every task) at the layout's source/. `render prepare`
    keeps what is already there, so this is cheap when the tree is built."""
    from axis_qvla.realrobot.render import data

    done = data.prepare(lay.data, lay.source, tasks)
    print(f"{done['tasks']} tasks, {done['episodes']} episodes, {done['layouts']} mesh layouts -> {lay.source}")
    return lay.source


def check_camera(args) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.realrobot.layout import ROBOTS, layout
    from axis_qvla.realrobot.render import cli as render_cli
    from axis_qvla.realrobot.render import data

    try:
        source = prepare_source(layout(args.host), args.tasks)
        manifest = data.load_manifest(source)
        tid = data.select(manifest, args.tasks)[0]
        variant = manifest["tasks"][str(tid)]["variants"][0]
    except (ValueError, OSError, HostProfileError) as exc:      # DataError is a ValueError
        return fail(str(exc))
    argv = ["overlay", "--image", str(args.image), "--scene", str(source / data.scene_rel(tid, variant)),
            "--robots", str(ROBOTS), "--out", str(args.out)]
    if args.profile is not None:
        argv += ["--profile", str(args.profile)]
    if args.joints is not None:
        argv += ["--joints", *map(str, args.joints)]
    return render_cli.main(argv)


def training_view(render: Path, d8: Path) -> tuple[int, int]:
    """(written, kept): `render droid8d` for each rendered task d8/ does not hold yet. Each task is written
    to task_<tid>.partial/ and renamed when whole (the renderer's own convention), so an interrupted run
    resumes. Then d8/roots.json is written over every task."""
    import shutil

    from axis_qvla.realrobot.render import droid8d, writer

    d8.mkdir(parents=True, exist_ok=True)
    have = writer.task_dirs(d8)
    todo = {t: p for t, p in writer.task_dirs(render).items() if t not in have}
    for tid, src in todo.items():
        part = d8 / f"task_{tid}.partial"
        if part.exists():
            shutil.rmtree(part)
        droid8d.convert_task(src, part)
        part.rename(d8 / f"task_{tid}")
    writer.roots_index(d8)
    return len(todo), len(have)


def rerender(args) -> int:
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.realrobot.layout import ROBOTS, layout
    from axis_qvla.realrobot.render import cli as render_cli

    try:
        lay = layout(args.host)
        source = prepare_source(lay, args.tasks)
    except (ValueError, OSError, HostProfileError) as exc:
        return fail(str(exc))
    argv = ["render", "--data", str(source), "--robots", str(ROBOTS), "--out", str(lay.render)]
    if args.profile is not None:
        argv += ["--profile", str(args.profile)]
    if args.tasks:
        argv += ["--tasks", *map(str, args.tasks)]
    rc = render_cli.main(argv)         # 1: some tasks beyond the IK's acceptance; the others are rendered
    if rc == 2:
        return rc
    try:
        written, kept = training_view(lay.render, lay.d8)
    except (ValueError, OSError) as exc:
        return fail(str(exc))
    print(f"training view: {written} tasks written, {kept} kept -> {lay.d8}")
    return rc
