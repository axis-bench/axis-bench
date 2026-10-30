"""Whether eval may repaint the table, kept apart from the renderer that applies it.

Lives in its own module because it is a DECISION, not a rendering step: importing `render_scene`
drags in `axis_replay_native.robosuite_render`, which only exists in the eval environment, and a
rule this consequential should be testable without a GPU or a MuJoCo build. Stdlib only.
"""

from __future__ import annotations

# MuJoCo cannot load JPEG textures ("Non-PNG texture, assuming custom binary file format").
# AXIS-Render's curated bank ships .jpg because their randomizer *uploads* bitmaps through
# TextureModder (the mujoco==3.1.1 path). For a deterministic fixed texture we reference a PNG
# conversion of the same asset instead.
DEFAULT_TABLE_TEXTURE = "assets/table_textures/tabletop_1k/herringbone_parquet_1k.png"

# README caps: reflectance <= 0.20, specular <= 0.30 (apply in fixed and randomized runs).
TABLE_MATERIAL = {
    "rgba": [1.0, 1.0, 1.0, 1.0],
    "texrepeat": [1.0, 1.0],
    "reflectance": 0.20,
    "shininess": 0.25,
    "specular": 0.30,
}


def surface_appearance_override(cfg: dict, table_texture: str | None) -> dict | None:
    """The table appearance eval should force, or None to leave the scene's own table alone.

    A PINNED SCENE ALREADY DETERMINES THE TABLE -- do not paint over it. The override exists for
    corpora rendered with a randomized scene, where the table had to be forced to something fixed so
    eval was not a lottery. When the camera config carries `fixed_scene_index`, the training render
    took its table from that recipe and recorded `surface_appearance: null`; forcing herringbone
    parquet here instead shows the policy a table it never trained on.

    Measured on task 953 at MATCHED timesteps (train frame 0 vs eval step 0): with the override,
    training had a white tabletop and eval a wood parquet one; with it skipped, table, floor, walls
    and viewpoint all agree and only object placement differs -- which is the axis the benchmark is
    meant to vary. Passing `table_texture=None` skips it explicitly.
    """
    if table_texture is None:
        return None
    if "fixed_scene_index" in ((cfg.get("render_profile") or {}).get("scene") or {}):
        return None
    return {"mode": "fixed_texture_material",
            "table": {"texture_file": table_texture, **TABLE_MATERIAL}}
