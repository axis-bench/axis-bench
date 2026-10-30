"""One recorded attempt -> one rendered episode; a set of attempts -> rendered tasks, a roots index and
the workspace profile they were rendered in (`workspace.toml`, with the images it names). Tasks render
one at a time, each whole or not at all, so a render of many tasks resumes where it stopped and an IK
refusal costs only its task (render_tasks).

Render tier. The sidecar records everything the gates recompute: the recorded row times, knot rows, the
source TCP at each knot's frame time, the pinch offset in the base frame, IK errors, finger openness, the
settled gripper extremes, the gripper lock commit and the workspace id.
"""
from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

from axis_qvla.realrobot.render import assemble, draw, episode, gripper, ik, rig, source, writer
from axis_qvla.realrobot.render.profile import FILE as PROFILE_FILE, WorkspaceProfile, load_profile


def render_episode(ep: source.SourceEpisode, gripper_xml: Path, task_writer: writer.TaskWriter,
                   profile: WorkspaceProfile, *, gripper_lock: str | None = None) -> dict:
    src_model = mujoco.MjModel.from_xml_path(str(ep.scene_xml))
    height = profile.table.height
    stream = source.source_stream(src_model, ep.packed, height)
    solved = ik.solve_knots(assemble.build_target_scene(ep.scene_xml, gripper_xml, objects=False, profile=profile),
                            stream)
    scene = assemble.build_target_scene(ep.scene_xml, gripper_xml, key=(ep.task_id, ep.attempt_id), profile=profile)
    ext = gripper.settle_extremes(scene, solved.arm_q[0])
    states = episode.compose(scene, ep.packed, solved.arm_q, ext, stream.openness, height)
    kt = ep.packed.row_times
    at = states.grid_t[states.knot_rows]
    tcp_at = np.stack([np.interp(at, kt, stream.tcp[:, i]) for i in range(3)], axis=1)
    c2w, ee = draw.poses(scene, states.qpos)
    sidecar = {
        "schema": "qvla.realrobot.render_episode/1", "profile": rig.PROFILE_ID, "task_id": ep.task_id,
        "attempt_id": ep.attempt_id, "n_knots": ep.packed.n_steps, "frames": len(states.grid_t),
        "lead_time": ep.packed.lead_time, "row_times": kt.tolist(), "knot_rows": states.knot_rows.tolist(),
        "source_tcp": tcp_at.tolist(), "openness": stream.openness.tolist(),
        "pinch_in_base": scene.model.site_pos[scene.pinch_site].tolist(),
        "perr_m": solved.perr.tolist(), "oerr_rad": solved.oerr.tolist(), "seeds": solved.seeds.tolist(),
        "convergence_knot": solved.convergence_knot, "reach_pct": solved.reach_pct,
        "gripper": {"open_separation": ext.open_separation, "closed_separation": ext.closed_separation,
                    "open_ctrl": ext.open_ctrl, "closed_ctrl": ext.closed_ctrl, "lock_commit": gripper_lock},
        "interpolation": states.methods,
        "workspace": profile.id,
    }
    data = writer.EpisodeData(ep.attempt_id, states.qpos[:, scene.arm_qpos].astype(np.float32),
                              states.action.astype(np.float32), ee, c2w, sidecar)
    task_writer.add_episode(data, draw.frames(scene, states.qpos))
    return sidecar


def _keep_profile(profile: WorkspaceProfile, out_root: Path) -> None:
    """Keep `profile` beside the render as `workspace.toml`, with the images it names at the same paths
    inside `out_root`, so the kept copy loads from the render directory alone. A directory that already
    keeps another profile -- other text or other image bytes -- is refused: one render, one workspace."""
    kept = out_root / PROFILE_FILE
    images = {out_root / rel: profile.directory / rel for rel in profile.images()}
    if kept.exists():
        if kept.read_text(encoding="utf-8") != profile.text or any(
                not dst.is_file() or dst.read_bytes() != src.read_bytes() for dst, src in images.items()):
            raise ValueError(f"{out_root} holds a render of another workspace profile ({kept}); "
                             f"render into a new directory")
        return
    for dst, src in images.items():
        if dst.resolve() != src.resolve():          # a render into the profile's own directory keeps it in place
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    kept.write_text(profile.text, encoding="utf-8")


@dataclasses.dataclass(frozen=True)
class Rendered:
    rendered: list[int]             # tasks rendered by this call
    kept: list[int]                 # tasks already rendered in the directory, left as they are
    failed: dict[int, str]          # task -> the IK's refusal; nothing of the task is written


def _read(tid: int, eps: Sequence[Path | source.SourceEpisode]) -> list[source.SourceEpisode]:
    out = [e if isinstance(e, source.SourceEpisode) else source.read_source_episode(e) for e in eps]
    other = [e for e in out if e.task_id != tid]
    if other:
        raise ValueError(f"task {tid}: attempt {other[0].attempt_id} is an episode of task {other[0].task_id}")
    if len({e.prompt for e in out}) != 1:
        raise ValueError(f"task {tid}: attempts disagree on the instruction")
    return out


def render_tasks(tasks: Mapping[int, Sequence[Path | source.SourceEpisode]], gripper_xml: Path, out_root: Path, *,
                 profile: WorkspaceProfile | None = None, gripper_lock: str | None = None) -> Rendered:
    """Render `tasks` ({task id: its source episodes, as files or read}) into `out_root`, one task at a
    time: a task's episodes are read when its turn comes, written into `task_<tid>.partial/` and renamed
    `task_<tid>/` when whole. A task already rendered there (its meta/info.json written) is kept and not
    read, so a re-run resumes. An IK refusal costs its task only: the task is removed and named in
    `failed`, and the others render. Any other error removes the partial task and propagates."""
    profile = profile or load_profile()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _keep_profile(profile, out_root)
    rendered, kept, failed = [], [], {}
    for tid, eps in sorted(tasks.items()):
        if (out_root / f"task_{tid}" / "meta" / "info.json").is_file():
            kept.append(tid)
            continue
        episodes = _read(tid, eps)
        w = writer.TaskWriter(out_root, tid, episodes[0].prompt, cameras=profile.cameras, workspace=profile.id)
        try:
            for ep in sorted(episodes, key=lambda e: e.attempt_id):
                try:
                    render_episode(ep, gripper_xml, w, profile, gripper_lock=gripper_lock)
                except ik.IkError as exc:         # which attempt: the solver's own message cannot say
                    raise ik.IkError(f"task {tid} attempt {ep.attempt_id}: {exc}") from exc
            w.close()
        except ik.IkError as exc:
            w.abort()
            failed[tid] = str(exc)
            continue
        except BaseException:
            w.abort()
            raise
        rendered.append(tid)
    writer.roots_index(out_root)
    return Rendered(rendered, kept, failed)
