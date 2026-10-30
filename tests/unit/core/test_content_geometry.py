"""The geometry PROFILES are a declared table, and these tests keep each one honest.

How a 640x360 render becomes a 224x224 observation depends on what the model will be finetuned
onto, so there is no single right answer -- there are named profiles, and picking the wrong one is
a silent failure. These tests pin what each profile promises, so a future edit that "unifies" them
back into one geometry fails loudly instead of quietly retargeting a corpus.

Stdlib + the core-tier table only; no video is encoded here.
"""

from __future__ import annotations

import pytest

from axis_qvla.core.camera_geometry import (
    CAMERAS,
    DEFAULT_PROFILE,
    GEOMETRY_PROFILES,
    GRIPPER_SPAN_X,
    LIBERO_WRIST_BAND_FRAC,
    GeometryError,
    geometry_for,
    model_input_size,
    on_disk_size,
)

SRC_W, SRC_H = 640, 360


def test_the_default_profile_is_droid_because_that_is_what_we_are_building():
    """The current corpus comes from the pre-re-render wrist pose, which is DROID-aligned."""
    assert DEFAULT_PROFILE == "droid"


@pytest.mark.parametrize("prof", sorted(GEOMETRY_PROFILES))
def test_every_profile_declares_every_camera(prof):
    """A missing camera raises mid-build, after hours of decoding."""
    assert set(GEOMETRY_PROFILES[prof]) == set(CAMERAS)


@pytest.mark.parametrize("prof", sorted(GEOMETRY_PROFILES))
@pytest.mark.parametrize("cam", CAMERAS)
def test_stored_frame_never_exceeds_the_model_input(prof, cam):
    """Storing bigger than the encoder input would be thrown away by resize_with_pad.

    EXCEPT under a passthrough profile, where the premise is false by design: the served config
    (`AxisFrankaInputs(center_crop=True)`) crops the frame itself, so the extra pixels are the
    crop's INPUT rather than waste. Asserted the other way round for those.
    """
    from axis_qvla.core.camera_geometry import geometry_for

    W, H = model_input_size()
    w, h = on_disk_size(cam, prof)
    if geometry_for(cam, prof).get("passthrough"):
        assert (w, h) == (SRC_W, SRC_H), (
            f"{prof}/{cam} is passthrough, so it must hand over the SOURCE frame, not {w}x{h}"
        )
        return
    assert w <= W and h <= H, f"{prof}/{cam} stores {w}x{h}, larger than the {W}x{H} model input"


# ---- droid ------------------------------------------------------------------------------------


@pytest.mark.parametrize("cam", CAMERAS)
def test_droid_keeps_the_full_frame_and_lets_resize_with_pad_finish(cam):
    """DROID's convention IS openpi's resize_with_pad. Storing 224x126 and letting the dataloader
    pad is byte-identical to baking it, for fewer bytes -- and it is what the existing corpora
    already hold, so a DROID-aligned corpus needs no rebuild."""
    g = geometry_for(cam, "droid")
    assert g["crop"] is None, "cropping would discard view DROID-pretrained pi0 expects to see"
    assert g["pad_to"] is None, "droid leaves the padding to resize_with_pad"
    assert (g["width"], g["height"]) == (224, round(224 * SRC_H / SRC_W))


def test_droid_treats_both_cameras_identically():
    """There is no per-camera special-casing in DROID's preprocessing, and inventing one would
    make our corpus disagree with the pretraining it is meant to match."""
    assert geometry_for("third_person", "droid") == geometry_for("wrist", "droid")


# ---- libero -----------------------------------------------------------------------------------


def test_libero_external_is_cropped_square_and_pays_no_padding():
    g = geometry_for("third_person", "libero")
    assert g["crop"] is not None, "a square external view must come from a CROP, never a squash"
    assert g["pad_to"] is None
    assert g["width"] == g["height"] == 224


def test_libero_external_crop_is_centred_and_inside_the_source():
    cw, ch, cx, cy = geometry_for("third_person", "libero")["crop"]
    assert cw == ch, "the external crop must be square, so it pays no padding"
    assert cx == (SRC_W - cw) // 2 and cy == (SRC_H - ch) // 2, "centred on both axes"
    assert cx + cw <= SRC_W and cy + ch <= SRC_H


def test_libero_external_is_tighter_than_the_wrist_but_not_a_keyhole():
    """The external is zoomed IN past full height: 256 keeps 40% of the frame width. The bound
    keeps the field of view between a third and a half of the frame."""
    ext = geometry_for("third_person", "libero")["crop"][0]
    wrist = geometry_for("wrist", "libero")["crop"][0]
    assert ext < wrist, "the external is the tighter of the two"
    assert 0.33 <= ext / SRC_W <= 0.50, f"external keeps {ext / SRC_W:.0%} of frame width"


def test_libero_wrist_crop_contains_the_whole_measured_gripper():
    """The crop may be as narrow as we like EXCEPT that it must keep the open gripper whole.

    GRIPPER_SPAN_X is measured from real frames; this assertion ties the geometry to that
    measurement.
    """
    cw, ch, cx, cy = geometry_for("wrist", "libero")["crop"]
    lo, hi = GRIPPER_SPAN_X
    assert cx <= lo, f"crop starts at x={cx}, clipping the left fingertip which begins at {lo}"
    assert cx + cw >= hi, f"crop ends at x={cx + cw}, clipping the right fingertip which ends at {hi}"


def test_libero_wrist_keeps_full_height_so_the_band_is_the_only_dead_region():
    cw, ch, cx, cy = geometry_for("wrist", "libero")["crop"]
    assert ch == SRC_H and cy == 0
    assert cx + cw <= SRC_W


def test_libero_wrist_pads_at_the_bottom():
    g = geometry_for("wrist", "libero")
    assert g["pad_align"] == "top", "the band belongs at the BOTTOM of the canvas"
    assert g["pad_to"] == model_input_size()


def test_libero_wrist_band_matches_the_declared_band_fraction():
    """The band is LIBERO_WRIST_BAND_FRAC of the canvas (36 rows of 224), within two rows."""
    g = geometry_for("wrist", "libero")
    _, H = model_input_size()
    band = (H - g["height"]) / H
    assert abs(band - LIBERO_WRIST_BAND_FRAC) * H <= 2, (
        f"wrist band is {band:.1%} ({(H - g['height'])} rows); the declared band is "
        f"{LIBERO_WRIST_BAND_FRAC:.1%} ({round(LIBERO_WRIST_BAND_FRAC * H)} rows). Keep them within "
        f"two rows."
    )


def test_the_wrist_crop_cannot_narrow_further_without_clipping_the_gripper():
    """The band is thin because the crop is narrow, and the crop is floored by the gripper span."""
    lo, hi = GRIPPER_SPAN_X
    floor = 2 * max(SRC_W / 2 - lo, hi - SRC_W / 2)          # narrowest centred crop containing it
    cw = geometry_for("wrist", "libero")["crop"][0]
    assert cw >= floor, f"crop {cw} is narrower than the {floor:.0f} needed to keep the gripper whole"
    assert cw - floor < 20, (
        f"crop {cw} sits {cw - floor:.0f} px above the {floor:.0f} floor -- there is room to thin "
        f"the band further; if that is deliberate, say why here"
    )


def test_libero_external_pays_no_padding_even_though_the_wrist_does():
    """The two cameras are treated differently ON PURPOSE -- a square external crop costs nothing,
    while squaring the wrist would cost the gripper. An edit that "unifies" them should fail."""
    ext = geometry_for("third_person", "libero")
    assert ext["pad_to"] is None and (ext["width"], ext["height"]) == model_input_size()
    assert geometry_for("wrist", "libero")["pad_to"] is not None


# ---- refusals ---------------------------------------------------------------------------------


def test_unknown_profile_is_refused_not_defaulted():
    with pytest.raises(GeometryError, match="unknown geometry profile"):
        geometry_for("wrist", "libero_v2_final")


def test_unknown_camera_is_refused_not_defaulted():
    with pytest.raises(GeometryError):
        geometry_for("nose_cam", "droid")

# ---- never stretch --------------------------------------------------------------------------


@pytest.mark.parametrize("prof", sorted(GEOMETRY_PROFILES))
@pytest.mark.parametrize("cam", CAMERAS)
def test_every_camera_is_scaled_uniformly_never_stretched(prof, cam):
    """STANDING INSTRUCTION: the image is cropped and PADDED, never stretched.

    A non-uniform scale is the tempting way to remove a band -- squash 429x360 into 224x224 and the
    padding disappears -- and it is forbidden: it distorts every object in the frame by the aspect
    ratio, which no amount of finetuning undoes and which no shape check would ever notice. If a
    band cannot be made to match the target exactly, the band is wrong; the geometry is not.
    """
    g = geometry_for(cam, prof)
    cw, ch = g["crop"][:2] if g["crop"] else (SRC_W, SRC_H)
    sx, sy = g["width"] / cw, g["height"] / ch
    assert abs(sx / sy - 1) < 0.005, (
        f"{prof}/{cam} scales x by {sx:.5f} and y by {sy:.5f} (ratio {sx / sy:.5f}) -- that is a "
        f"stretch. Crop and pad instead; an imperfect band beats a distorted image."
    )
