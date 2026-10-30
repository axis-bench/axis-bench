"""Turn a rendered frame into the observation the model was TRAINED on, per camera.

ONLINE TIER (numpy only; no jax/torch needed, but it is imported by the rollout harness).

This is the eval half of a two-process convention. The build tier writes the corpus with
the geometry profiles of `axis_qvla.core.camera_geometry`; this applies the identical transform to a live
render. Neither can import the other -- offline venv vs online venv -- so the geometry is declared
once in the core tier and both sides consume it.

WHY NOT `resize_with_pad`: `image_tools.resize_with_pad(frame, 224, 224)` for every camera preserves
aspect and CENTRES the content, so a 640x360 render becomes 224x126 of content with the black split
evenly top and bottom. The training corpus does something different and per-camera: the external
view is a centre CROP with no padding at all, and the wrist keeps full width and is padded at the
BOTTOM only. Feeding a centred letterbox to a model trained on either of those is a silent
train/eval mismatch -- no error, no log line, just a policy looking at pixels in the wrong places.

A convention mismatch between train and eval is invisible to a test on either side alone, since
the two processes only ever talk over a wire. The guard is a direct comparison: one frame down both
the build and the eval path, checked for exact array equality.
"""

from __future__ import annotations

from axis_qvla.core.camera_geometry import DEFAULT_PROFILE, geometry_for, on_disk_size


# The policy payload names an image by its OBS KEY, the corpus names it by camera. The rollout
# harnesses each had their own spelling of this ("frontview"/"wrist", "pi05/base_cam"), so the
# mapping is stated once, here, against the keys the SERVER actually receives -- which is the only
# spelling all three harnesses agree on.
OBS_KEY_TO_CAMERA = {
    "base_0_rgb": "third_person",
    "left_wrist_0_rgb": "wrist",
}


_GEOMETRY_ENV = "AXIS_EVAL_IMAGE_GEOMETRY"


def _passthrough_requested() -> bool:
    """True when the SERVED CONFIG squares the frame itself and this module must not pre-square it.

    WHY. `AxisFrankaInputs(center_crop=True)` -- set by every `pi05_axis_heldout_multitask*` config
    to match stage-1 -- centre-crops its input to a square. In TRAINING that input is the stored
    640x360 frame, so the model sees the middle 56% of the width with no padding. If this module
    first produced a 224x224 letterbox, the same crop would be a no-op and the model would see the
    full width with black bands: one transform object, two fields of view. Handing over the raw frame
    keeps the served field of view identical to the trained one.

    Set `AXIS_EVAL_IMAGE_GEOMETRY=raw` when serving a config with `image_center_crop=True`; the
    frame is then handed over untouched and the policy's own transform reproduces training exactly.
    Default is unchanged, so no existing eval changes meaning.
    """
    import os

    return os.environ.get(_GEOMETRY_ENV, "profile").strip().lower() == "raw"


def observation_for_obs_key(frame, obs_key: str, profile_name: str = DEFAULT_PROFILE):
    """`observation_from_render` addressed by the payload key the policy server expects."""
    try:
        camera = OBS_KEY_TO_CAMERA[obs_key]
    except KeyError:
        from axis_qvla.core.camera_geometry import GeometryError

        raise GeometryError(
            f"no camera mapped to obs key {obs_key!r}; known: {', '.join(OBS_KEY_TO_CAMERA)}"
        ) from None
    return observation_from_render(frame, camera, profile_name)


def observation_from_render(frame, camera: str, profile_name: str = DEFAULT_PROFILE):
    """A rendered `(H, W, 3)` uint8 frame as the stored corpus would hold it, for `camera` under `profile_name`.

    The profile MUST match the one the corpus was built with (recorded in its
    meta/info.json under `axis_pretrain.geometry_profile`). Serving a DROID-preprocessed
    observation to a policy trained under another profile is a silent mismatch, not an error.

    Applies crop -> resize -> pad exactly as the builder does. Returns uint8 `(224, 224, 3)`.
    """
    import numpy as np

    g = geometry_for(camera, profile_name)
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) frame for {camera!r}, got shape {arr.shape}")

    if g.get("passthrough") or _passthrough_requested():
        # The served config squares the frame itself; pre-squaring here would make that crop a
        # no-op and hand the model a different field of view than training. See
        # `_passthrough_requested` for why.
        #
        # Two routes reach this branch and they mean the same thing. `g["passthrough"]` is the
        # DECLARED one -- the `square360` profile -- and is preferred, because a profile name is
        # recorded with the run while an environment variable silently reverts on a box where it
        # was never exported. The env var stays supported so existing eval scripts keep working.
        return np.ascontiguousarray(arr, dtype=np.uint8)

    if g["crop"] is not None:
        cw, ch, cx, cy = g["crop"]
        if arr.shape[0] < cy + ch or arr.shape[1] < cx + cw:
            raise ValueError(
                f"{camera}: crop {g['crop']} does not fit a {arr.shape[1]}x{arr.shape[0]} render. "
                f"Refusing rather than silently observing a different framing than training used."
            )
        arr = arr[cy:cy + ch, cx:cx + cw]

    arr = _resize(arr, g["width"], g["height"])

    if g["pad_to"] is not None:
        ow, oh = on_disk_size(camera, profile_name)
        canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
        y0 = 0 if g["pad_align"] == "top" else (oh - g["height"]) // 2
        x0 = (ow - g["width"]) // 2
        canvas[y0:y0 + g["height"], x0:x0 + g["width"]] = arr
        arr = canvas

    # Finish with the dataloader's own step. A profile describes the STORED frame; the encoder
    # sees whatever `resize_with_pad` makes of it. For `droid` that means the centred black bars
    # get added here (the corpus stores 224x126); for `libero` the stored frame is already 224x224
    # and this is a no-op. Applying it unconditionally is what makes this function return the
    # MODEL INPUT for either profile, which is the only thing worth comparing against training.
    return _resize_with_pad_centered(arr)


def _resize_with_pad_centered(arr):
    """Replicates openpi's `resize_with_pad` to the model input size, in numpy.

    Reimplemented rather than imported because this module must stay usable without jax, and
    because a parity test that calls the same function on both sides would prove nothing. The
    arithmetic mirrors `openpi.shared.image_tools.resize_with_pad`: ratio = max(w/W, h/H), then
    `divmod(target - resized, 2)` with the remainder going to the far edge.
    """
    import numpy as np

    from axis_qvla.core.camera_geometry import model_input_size

    W, H = model_input_size()
    h, w = arr.shape[:2]
    if (w, h) == (W, H):
        return np.ascontiguousarray(arr, dtype=np.uint8)
    ratio = max(w / W, h / H)
    rw, rh = int(w / ratio), int(h / ratio)
    arr = _resize(arr, rw, rh)
    pad_h0, rem_h = divmod(H - rh, 2)
    pad_w0, rem_w = divmod(W - rw, 2)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[pad_h0:pad_h0 + rh, pad_w0:pad_w0 + rw] = arr
    return canvas


def _resize(arr, width: int, height: int):
    """Bilinear resize with no aspect correction -- the caller has already made it match.

    PIL rather than cv2: the online venv ships Pillow, and cv2 is not a dependency we want to add
    to the rollout path for one call.
    """
    import numpy as np
    from PIL import Image

    if arr.shape[0] == height and arr.shape[1] == width:
        return np.ascontiguousarray(arr, dtype=np.uint8)
    im = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).resize(
        (width, height), Image.BILINEAR
    )
    return np.asarray(im, dtype=np.uint8)
