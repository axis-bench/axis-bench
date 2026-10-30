"""Per-camera image geometry, as NAMED PROFILES chosen by the finetuning target.

CORE TIER: stdlib only. It has to be, because the two consumers cannot import each other -- the
build runs under the offline venv (PyAV, pyarrow) and the rollout eval under the online venv (jax,
openpi). A constant both tiers need lives here, not in either of them.

WHY PROFILES, AND NOT ONE GEOMETRY. How a 640x360 render should become a 224x224 observation is a
property of **what the model will be finetuned onto**, not of our renderer:

    droid   -- the DROID convention, which is exactly openpi's `resize_with_pad`: preserve aspect,
               resize to fit, pad the remainder BLACK and CENTRED. pi0/pi0.5 were pretrained on
               DROID this way, so a DROID-aligned corpus must look like it. The resulting 98 black
               rows on a 640x360 source are NOT waste here -- they are the convention.

    libero  -- a square-image convention: the external view is cropped square, so the model
               receives no padding; the wrist view keeps nearly the full width (the gripper spans
               it) on a 224x224 canvas with a black band along the bottom.

    square360 -- passthrough: the served config applies the transform itself (below).

Same renders, different target, different correct answer, so the profile is declared by name and
both tiers read the same declaration.

WHY BOTH TIERS READ THIS FILE. The corpus and the eval harness must agree pixel-for-pixel. They are
separate processes in separate environments, so one declaration plus checks on both sides of it
keeps them in agreement. `tests/unit/core/test_content_geometry.py` pins what each profile declares,
and `tests/unit/bench/sim/test_obs_geometry_passthrough.py` and
`tests/unit/bench/sim/test_square360_profile.py` pin that the eval transform applies it.

Per-camera fields:
    width/height : content size after the optional crop and the rescale.
    crop         : (w, h, x, y) in SOURCE pixels, applied first. None = no crop.
    pad_to       : (W, H) black canvas the content is placed on. None = no padding.
    pad_align    : "top" or "center" -- where the content sits on that canvas.
"""

from __future__ import annotations

SOURCE_W, SOURCE_H = 640, 360
OUT_W, OUT_H = 224, 224

# Full-width 640x360 scaled to 224 wide is 224x126 -- what `resize_with_pad` computes via
# ratio = max(640/224, 360/224). Spelled here so the DROID profile is derived, not guessed.
_FULL_W, _FULL_H = OUT_W, int(SOURCE_H / max(SOURCE_W / OUT_W, SOURCE_H / OUT_H))

# MEASURED, not eyeballed: the horizontal span of the open gripper in the wrist view, in source
# pixels. Taken over 12 tasks by finding columns that are persistently near-black in the bottom 90
# rows across frames; 7 of the 12 agree on exactly this span and the rest are tasks whose SCENE is
# dark at the frame edges, which is why the modal value is used rather than the union.
#
# It is recorded here because it is the only thing justifying the wrist crop width: any crop must
# contain it, and `tests/unit/core/test_content_geometry.py` asserts that it does. If the wrist
# camera is ever re-mounted or re-rendered, this number is measured again and the crop follows it.
GRIPPER_SPAN_X = (120, 532)

# The `libero` profile's wrist band, as a fraction of the 224-row canvas.
LIBERO_WRIST_BAND_FRAC = 36 / 224          # 16.1%

# The wrist crop width that gives that band: content height = 224 * 360 / W, so a narrower crop
# means a taller content block and a thinner band. 429 lands on 188 rows of content and 36 of band,
# and a centred crop must be at least 424 wide to keep GRIPPER_SPAN_X whole.
WRIST_CROP_W = 429

# STORED as 224x126, NOT as a padded 224x224. `resize_with_pad` completes it to 224x224 with
# centred black at train time, which is byte-identical to baking the pad -- so storing the smaller
# frame is the same model input for fewer bytes, AND it is what the existing corpora already hold.
# A DROID-aligned corpus therefore needs no rebuild.
_DROID_CAM = {"width": _FULL_W, "height": _FULL_H,
              "crop": None, "pad_to": None, "pad_align": "center"}

# `passthrough` means: this module performs NO geometry at all and yields the source frame, because
# the SERVED CONFIG applies the transform itself. Only the `square360` profile sets it. A consumer
# that reads width/height without honouring it gets the source size, which is the truth for a
# passthrough profile -- the stored/served frame really is SOURCE_W x SOURCE_H.
_PASSTHROUGH_CAM = {"width": SOURCE_W, "height": SOURCE_H,
                    "crop": None, "pad_to": None, "pad_align": "center",
                    "passthrough": True}

GEOMETRY_PROFILES: dict[str, dict[str, dict]] = {
    # The DROID convention for BOTH cameras: full frame, aspect-preserving resize, and the
    # remainder left to `resize_with_pad` (black, centred) exactly as pi0/pi0.5 saw DROID.
    "droid": {"third_person": dict(_DROID_CAM), "wrist": dict(_DROID_CAM)},

    # EXTERNAL -- cropped square (256 of 640 columns at full height), centred on the manipulation
    # region, then resized to 224 with no padding.
    #
    # WRIST -- kept wide: the gripper fingers run to the left and right frame edges, so a square
    # crop would cut them. The crop is WRIST_CROP_W wide (it contains GRIPPER_SPAN_X), resized to
    # 224x188 and placed at the top of a 224x224 canvas, leaving a 36-row band at the bottom
    # (LIBERO_WRIST_BAND_FRAC); stretching it to square would distort the aspect by 1.28x.
    "libero": {
        "third_person": {"width": OUT_W, "height": OUT_H,
                         "crop": (256, 256, (SOURCE_W - 256) // 2, (SOURCE_H - 256) // 2),
                         "pad_to": None, "pad_align": "top"},
        "wrist": {"width": OUT_W, "height": round(OUT_W * SOURCE_H / WRIST_CROP_W),
                  "crop": (WRIST_CROP_W, SOURCE_H, (SOURCE_W - WRIST_CROP_W) // 2, 0),
                  "pad_to": (OUT_W, OUT_H), "pad_align": "top"},
    },

    # THE SERVED CONFIG OWNS THE GEOMETRY. Hand the source frame over untouched.
    #
    # For `AxisFrankaInputs(center_crop=True)` -- what every `pi05_axis_heldout_multitask*` config
    # and the droid8 stage-1 use. That transform centre-crops its input to a square, and in
    # TRAINING its input is the stored 640x360 frame, so the model sees a centred 360x360 crop
    # (`side = min(h, w) = 360`, `left = (640-360)//2 = 140`) resized to 224 by openpi's own
    # pipeline. At SERVE time the harness must therefore hand over the SAME 640x360 frame.
    #
    # WHY PASSTHROUGH AND NOT `crop=(360, 360, 140, 0)`. A cropping profile gives the same FIELD
    # OF VIEW, but this module would then do the 360->224 resize itself, in PIL, while training
    # does it downstream inside openpi. Same framing, different resampler -- so the served array
    # would be close to the trained one but not equal to it. Passing the frame through keeps eval
    # on the EXACT code path training used, which is what makes a parity check a comparison rather
    # than an approximation.
    #
    # It is a NAMED PROFILE so the convention is declared and recorded, instead of living only in
    # the `AXIS_EVAL_IMAGE_GEOMETRY=raw` environment variable, which a run does not record.
    "square360": {
        "third_person": dict(_PASSTHROUGH_CAM),
        "wrist": dict(_PASSTHROUGH_CAM),
    },
}

# What a build uses unless it says otherwise. DROID, because that is the corpus being built from
# the current (pre-re-render) wrist pose, which is DROID-aligned.
DEFAULT_PROFILE = "droid"

CAMERAS = tuple(GEOMETRY_PROFILES[DEFAULT_PROFILE])


class GeometryError(KeyError):
    """An unknown profile or camera. Raised rather than defaulted: guessing here produces a corpus
    that trains fine and evaluates wrong, which is the failure this module exists to prevent."""


def profile(name: str | None = None) -> dict[str, dict]:
    name = name or DEFAULT_PROFILE
    try:
        return GEOMETRY_PROFILES[name]
    except KeyError:
        raise GeometryError(
            f"unknown geometry profile {name!r}; known: {', '.join(sorted(GEOMETRY_PROFILES))}"
        ) from None


def geometry_for(camera: str, profile_name: str | None = None) -> dict:
    prof = profile(profile_name)
    try:
        return prof[camera]
    except KeyError:
        raise GeometryError(
            f"profile {profile_name or DEFAULT_PROFILE!r} has no camera {camera!r}; "
            f"known: {', '.join(sorted(prof))}"
        ) from None


def on_disk_size(camera: str, profile_name: str | None = None) -> tuple[int, int]:
    """`(width, height)` of the frame as STORED -- the padded canvas when there is one.

    A consumer reading a feature shape must get the frame it will actually decode, not the pre-pad
    content size.
    """
    g = geometry_for(camera, profile_name)
    return tuple(g["pad_to"]) if g["pad_to"] else (g["width"], g["height"])


def content_box(camera: str, profile_name: str | None = None) -> tuple[int, int, int, int]:
    """`(x0, y0, w, h)` of the real pixels inside the stored frame; the rest is padding."""
    g = geometry_for(camera, profile_name)
    w, h = g["width"], g["height"]
    ow, oh = on_disk_size(camera, profile_name)
    y0 = 0 if g["pad_align"] == "top" else (oh - h) // 2
    return ((ow - w) // 2, y0, w, h)


def model_input_size() -> tuple[int, int]:
    """`(W, H)` the vision encoder actually receives, after the dataloader's `resize_with_pad`."""
    return (OUT_W, OUT_H)
