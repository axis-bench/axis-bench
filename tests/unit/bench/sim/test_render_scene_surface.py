"""A pinned scene must keep its own table -- eval may not paint over it.

The held-out benchmark renders its training demos with `fixed_scene_index`, which pins the room and
the table. The eval harness independently forced the table to a herringbone-parquet texture, so the
policy trained on a white tabletop and was evaluated on a wooden one. That is invisible in any
success rate and was only caught by putting a training frame and an eval frame side by side at the
SAME timestep.

This guards the decision, not the rendering. It lives in axis_qvla/bench/sim/surface_appearance.py, which
is stdlib-only, so the rule is testable without a GPU or a MuJoCo build.
"""

from __future__ import annotations

from axis_qvla.bench.sim.surface_appearance import DEFAULT_TABLE_TEXTURE, surface_appearance_override


def test_pinned_scene_keeps_its_own_table() -> None:
    cfg = {"render_profile": {"scene": {"fixed_scene_index": 0}}}
    assert surface_appearance_override(cfg, DEFAULT_TABLE_TEXTURE) is None


def test_unpinned_scene_still_gets_a_fixed_table() -> None:
    # Without a pinned scene the table would otherwise be a lottery, so the override still applies.
    cfg = {"render_profile": {"scene": {"scene_seed": 12345}}}
    got = surface_appearance_override(cfg, DEFAULT_TABLE_TEXTURE)
    assert got is not None
    assert got["mode"] == "fixed_texture_material"
    assert got["table"]["texture_file"] == DEFAULT_TABLE_TEXTURE


def test_missing_render_profile_is_treated_as_unpinned() -> None:
    assert surface_appearance_override({}, DEFAULT_TABLE_TEXTURE) is not None
    assert surface_appearance_override({"render_profile": {}}, DEFAULT_TABLE_TEXTURE) is not None


def test_explicit_none_skips_the_override() -> None:
    assert surface_appearance_override({"render_profile": {"scene": {}}}, None) is None
