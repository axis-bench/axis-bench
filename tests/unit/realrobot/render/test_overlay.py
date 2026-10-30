"""The overlay's image arithmetic: fitting a camera frame to the render's size, outlines, the composite."""
from __future__ import annotations

import numpy as np

from axis_qvla.realrobot.render import overlay, rig


def test_a_1280x720_frame_is_halved_to_the_render_size():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, 640:] = 200
    out = overlay.fit(img)
    assert out.shape == (rig.HEIGHT, rig.WIDTH, 3) and out[:, :320].max() == 0 and out[:, 320:].min() == 200


def test_a_4_by_3_frame_is_centre_cropped_to_16_by_9_first():
    img = np.zeros((960, 1280, 3), np.uint8)
    img[:120] = img[-120:] = 255                  # the bands a 16:9 crop removes
    assert overlay.fit(img).max() == 0


def test_the_outline_is_the_mask_boundary():
    m = np.zeros((10, 10), bool)
    m[3:7, 3:7] = True
    o = overlay.outline(m)
    assert o.sum() == 12 and o[3, 3] and not o[4, 4]


def test_the_composite_is_the_outlined_frame_beside_the_blend():
    real = np.full((rig.HEIGHT, rig.WIDTH, 3), 100, np.uint8)
    ren = np.full((rig.HEIGHT, rig.WIDTH, 3), 200, np.uint8)
    table = np.zeros((rig.HEIGHT, rig.WIDTH), bool)
    table[100:200, 100:300] = True
    out = overlay.compose(real, ren, {"table": table})
    assert out.shape == (rig.HEIGHT, 2 * rig.WIDTH, 3)
    assert tuple(out[100, 150]) == overlay.COLOURS["table"] and tuple(out[150, 150]) == (100, 100, 100)
    assert tuple(out[150, rig.WIDTH + 150]) == (150, 150, 150)
