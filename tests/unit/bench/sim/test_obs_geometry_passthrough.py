"""The eval must not pre-square a frame whose served config squares it itself.

`AxisFrankaInputs(center_crop=True)` centre-crops its input to a square. In training that input is
the stored 640x360 frame (model sees the middle 56% of width, no padding); a 224x224 letterbox
produced first would make that crop a no-op and show the model the full width with black bands.

These tests pin the passthrough that keeps the two fields of view equal, and that the DEFAULT is
untouched, since every other eval in the repo depends on the profile path.
"""

from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.bench.sim import obs_geometry
from axis_qvla.core.camera_geometry import model_input_size


def _frame(h=360, w=640):
    """A frame with a horizontal gradient, so a width crop is detectable from the pixels."""
    col = np.linspace(0, 255, w, dtype=np.uint8)
    return np.repeat(np.tile(col, (h, 1))[:, :, None], 3, axis=2)


def test_default_is_unchanged_and_returns_model_input_size(monkeypatch):
    """With the env unset, behaviour must be exactly what every existing eval already relies on."""
    monkeypatch.delenv(obs_geometry._GEOMETRY_ENV, raising=False)
    out = obs_geometry.observation_for_obs_key(_frame(), "base_0_rgb")
    W, H = model_input_size()
    assert out.shape == (H, W, 3)
    assert out.dtype == np.uint8


def test_passthrough_returns_the_frame_untouched(monkeypatch):
    """`raw` must hand the render over unmodified, so the policy's own crop reproduces training."""
    monkeypatch.setenv(obs_geometry._GEOMETRY_ENV, "raw")
    src = _frame()
    out = obs_geometry.observation_for_obs_key(src, "base_0_rgb")
    assert out.shape == src.shape
    np.testing.assert_array_equal(out, src)


def test_passthrough_preserves_the_full_width_that_the_crop_needs(monkeypatch):
    """The point of the passthrough is that the FULL WIDTH survives to the policy's crop.

    Under the default path a 640x360 frame is letterboxed, so its horizontal extent is squeezed and
    a later centre-crop can no longer recover the middle-56% framing training used. Asserting on the
    pixels rather than the shape is what makes this a geometry test and not a plumbing test.
    """
    src = _frame()
    monkeypatch.setenv(obs_geometry._GEOMETRY_ENV, "raw")
    raw = obs_geometry.observation_for_obs_key(src, "base_0_rgb")
    h = raw.shape[0]
    crop = raw[:, (raw.shape[1] - h) // 2:(raw.shape[1] - h) // 2 + h]     # what AxisFrankaInputs does
    assert crop.shape[0] == crop.shape[1] == 360
    # the crop must be a strict horizontal subrange of the source, not the whole width rescaled
    assert crop[0, 0, 0] > src[0, 0, 0]
    assert crop[0, -1, 0] < src[0, -1, 0]


@pytest.mark.parametrize("value", ["profile", "PROFILE", "", "letterbox", "0"])
def test_only_the_literal_raw_enables_passthrough(monkeypatch, value):
    """Anything other than `raw` keeps the profile path -- a typo must not silently change geometry.

    This is the failure mode the repo has already paid for: an env var that quietly redirects a
    convention, with nothing raising and nothing logged.
    """
    monkeypatch.setenv(obs_geometry._GEOMETRY_ENV, value)
    out = obs_geometry.observation_for_obs_key(_frame(), "base_0_rgb")
    W, H = model_input_size()
    assert out.shape == (H, W, 3)


def test_raw_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv(obs_geometry._GEOMETRY_ENV, "  RaW ")
    out = obs_geometry.observation_for_obs_key(_frame(), "base_0_rgb")
    assert out.shape == (360, 640, 3)


def test_passthrough_applies_to_the_wrist_camera_too(monkeypatch):
    """Cropping only one camera would give the two views different effective fields of view."""
    monkeypatch.setenv(obs_geometry._GEOMETRY_ENV, "raw")
    src = _frame()
    out = obs_geometry.observation_for_obs_key(src, "left_wrist_0_rgb")
    np.testing.assert_array_equal(out, src)
