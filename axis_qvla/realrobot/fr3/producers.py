"""Each embodiment declaration against what produces its data, so a declaration edited on one side only fails.

- fr3: the recording's declaration (axis_qvla.realrobot.realdata.state_order): its state and action layouts
  and ACTION_UNITS (DROID's normalised joint-velocity command in [-1, 1] at 15 Hz, absolute gripper);
- realalign_sim: the renderer's (axis_qvla.realrobot.render.rig, stdlib only, read inside the functions so
  this offline module never imports the render package at module scope): FPS, the +-1 rad/s
  VELOCITY_CLIP of the DROID-8D view, its layouts and the 640 x 360 frame size.

Offline tier (stdlib only).
"""
from __future__ import annotations

__all__ = ["unit_problems", "layout_problems"]


def _against(emb, want: dict, who: str) -> list[str]:
    got = {"state.stored": emb.state_stored, "state.model": emb.state_model, "action.stored": emb.action_stored,
           "action.units": emb.action_units, "action.range": emb.action_range, "fps": emb.fps,
           "action.gripper_units": emb.gripper_units, "images.stored": emb.image_stored}
    return [f"{emb.id}: embodiment {k} {got[k]!r}, {who} declares {want[k]!r}" for k in want if got[k] != want[k]]


def _fr3_unit_problems(emb) -> list[str]:
    """The FR3 embodiment's action units against the recording's declaration (state_order.ACTION_UNITS)."""
    from axis_qvla.realrobot.realdata.state_order import ACTION_UNITS as u

    return _against(emb, {"action.units": u["arm"], "action.range": tuple(u["arm_range"]),
                          "fps": u["fps"], "action.gripper_units": u["gripper"]}, "the recording")


def _fr3_layout_problems(emb) -> list[str]:
    """The FR3 embodiment's state and action layouts against the recording's declaration."""
    from axis_qvla.realrobot.realdata import state_order as so

    return _against(emb, {"state.stored": so.STORED_STATE, "state.model": so.MODEL_STATE,
                          "action.stored": so.STORED_ACTIONS}, "the recording")


def _sim_unit_problems(emb) -> list[str]:
    """The rendered view's fps and action range against its producer (axis_qvla.realrobot.render.rig, stdlib only)."""
    from axis_qvla.realrobot.render import rig

    return _against(emb, {"fps": rig.FPS, "action.range": (-rig.VELOCITY_CLIP, rig.VELOCITY_CLIP)}, "the renderer")


def _sim_layout_problems(emb) -> list[str]:
    """The rendered view's state and action layouts and frame size against its producer."""
    from axis_qvla.realrobot.render import rig

    return _against(emb, {"state.stored": rig.DROID8D_STATE, "state.model": rig.DROID8D_STATE,
                          "action.stored": rig.DROID8D_ACTION, "images.stored": (rig.WIDTH, rig.HEIGHT)},
                    "the renderer")


_PRODUCERS = {"fr3": (_fr3_unit_problems, _fr3_layout_problems),
              "realalign_sim": (_sim_unit_problems, _sim_layout_problems)}


def unit_problems(emb) -> list[str]:
    """The embodiment's fps, units and range against its producer's ([] for one without a producer)."""
    return _PRODUCERS[emb.id][0](emb) if emb.id in _PRODUCERS else []


def layout_problems(emb) -> list[str]:
    """The embodiment's state and action layouts (and, for the render, frame size) against its producer's."""
    return _PRODUCERS[emb.id][1](emb) if emb.id in _PRODUCERS else []
