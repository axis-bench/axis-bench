"""`square360`: the geometry the served config owns, declared instead of left to an env var.

Requested by the stage-1 pretraining agent, whose point stands -- a convention that lives only in
`AXIS_EVAL_IMAGE_GEOMETRY=raw` is one unexported variable away from vanishing. The profile is a
PASSTHROUGH rather than a `crop=(360, 360, 140, 0)`, because cropping here would move the 360->224
resize out of openpi and into PIL: same framing, different resampler, so the served array would no
longer be the array training produced.
"""
from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.core.camera_geometry import (
    GEOMETRY_PROFILES,
    OUT_H,
    OUT_W,
    SOURCE_H,
    SOURCE_W,
    geometry_for,
    on_disk_size,
)
from axis_qvla.bench.sim.obs_geometry import observation_from_render


def _frame():
    """A frame whose every pixel encodes its own (x, y), so any crop or resize is detectable."""
    ys, xs = np.mgrid[0:SOURCE_H, 0:SOURCE_W]
    return np.stack([(xs % 256), (ys % 256), np.full_like(xs, 7)], axis=-1).astype(np.uint8)


class TestProfileShape:
    def test_both_cameras_are_passthrough(self):
        for cam in ("third_person", "wrist"):
            assert geometry_for(cam, "square360")["passthrough"] is True

    def test_it_declares_the_source_size_because_that_is_what_is_served(self):
        for cam in ("third_person", "wrist"):
            assert on_disk_size(cam, "square360") == (SOURCE_W, SOURCE_H)

    def test_no_other_profile_is_passthrough(self):
        for name, prof in GEOMETRY_PROFILES.items():
            if name == "square360":
                continue
            for cam, g in prof.items():
                assert not g.get("passthrough"), f"{name}/{cam} must not be passthrough"


class TestPassthroughBehaviour:
    def test_the_frame_is_returned_untouched(self):
        f = _frame()
        out = observation_from_render(f, "third_person", "square360")
        assert out.shape == f.shape
        assert np.array_equal(out, f)

    def test_it_matches_the_env_var_route_exactly(self, monkeypatch):
        """The declared profile and AXIS_EVAL_IMAGE_GEOMETRY=raw must not be two conventions."""
        f = _frame()
        by_profile = observation_from_render(f, "wrist", "square360")
        monkeypatch.setenv("AXIS_EVAL_IMAGE_GEOMETRY", "raw")
        by_env = observation_from_render(f, "wrist", "droid")
        assert np.array_equal(by_profile, by_env)

    def test_droid_still_letterboxes_so_the_default_did_not_move(self):
        out = observation_from_render(_frame(), "third_person", "droid")
        assert out.shape == (OUT_H, OUT_W, 3)
        # A 640x360 source letterboxes to 224x126 of content with black top and bottom.
        assert out[0].max() == 0 and out[-1].max() == 0
        assert out[OUT_H // 2].max() > 0


class TestItReproducesTheTrainingCrop:
    def test_the_served_frame_squares_to_the_same_pixels_training_saw(self):
        """Passthrough is only correct if the DOWNSTREAM crop lands where training's did.

        Training feeds the stored 640x360 to `_center_crop_square`, which takes
        `side = min(h, w) = 360`, `left = (640-360)//2 = 140`, `top = 0`. Reproduced here against
        the passthrough output so the claim is checked rather than asserted in a comment.
        """
        f = _frame()
        served = observation_from_render(f, "third_person", "square360")

        h, w = served.shape[:2]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        assert (side, top, left) == (360, 0, 140)
        assert np.array_equal(served[top:top + side, left:left + side], f[0:360, 140:500])


def test_an_unknown_profile_still_raises_rather_than_defaulting():
    from axis_qvla.core.camera_geometry import GeometryError

    with pytest.raises(GeometryError):
        geometry_for("third_person", "square720")
