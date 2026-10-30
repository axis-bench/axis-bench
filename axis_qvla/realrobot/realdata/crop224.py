"""The 224x224 centre crop of a 16:9 recording, and which recorded frame a dataset row shows.

WHY: two geometries reach a 224x224 policy input from a 16:9 camera. pi05-DROID letterboxes the
whole frame (black bars, 224x126 of content); the real-robot line's finetunes and their serve path
crop the full-height centre square (the central 720x720 of 1280x720) and resize it. A model trained
on one and served the other sees a different scene scale, and nothing errors. A letterboxed dataset's
crop version is built from the full-resolution recordings, since a letterboxed row cannot be
un-padded (its content is only 224x126).

    crop224:  frame -> centre square, full height -> cv2.resize to 224x224, INTER_LINEAR
    pad224:   frame -> centred on a black square  -> cv2.resize to 224x224, INTER_LINEAR

THE MATCH: a stored row is not recorded frame i at a fixed offset -- the collection loop pops frames
from a camera queue, so the lag is usually 2 frames and drifts on slow steps. Each row is matched to
the recorded frame within +-4 whose pad224 has the lowest mean squared error against the row's
stored letterboxed image (ties to the earlier frame); that frame's crop224 replaces the row's pixels.

OpenCV does the resize because it IS the serve-time resize (the robot env calls cv2.resize), and its
fixed-point INTER_LINEAR is what has to be matched bit for bit; OpenCV 4.11 and 5.0 agree on it.
OpenCV is imported inside resize224, so the rest of axis_qvla.realrobot.realdata runs without it: install
opencv-python-headless to crop.

Frames are (H, W, 3) RGB uint8. INTER_LINEAR treats channels independently, so the reference
builder's BGR-in / RGB-out gives the same pixels.

Rewritten from the dataset producers' FR3 data tools (credited in NOTICE).
Offline tier: numpy; OpenCV inside resize224.
"""
from __future__ import annotations

import dataclasses

import numpy as np

SIZE = 224
WINDOW = 4


def _rgb(frame: object) -> np.ndarray:
    x = np.asarray(frame)
    if x.ndim != 3 or x.shape[2] != 3 or x.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 frame, got {x.shape} {x.dtype}")
    return x


def center_square(frame: np.ndarray) -> np.ndarray:
    """The centred square of the shorter side: 1280x720 -> columns 280..999, every row."""
    x = _rgb(frame)
    h, w = x.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return x[y0:y0 + side, x0:x0 + side]


def letterbox_square(frame: np.ndarray) -> np.ndarray:
    """The frame centred on a black square of its longer side: 1280x720 -> rows 280..999 of 1280x1280."""
    x = _rgb(frame)
    h, w = x.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = x
    return out


def resize224(image: np.ndarray) -> np.ndarray:
    """cv2.resize to 224x224 with INTER_LINEAR: the serve-time resize."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("crop224 resizes with OpenCV, as the robot does at serve time: "
                          "pip install opencv-python-headless") from exc
    return cv2.resize(np.ascontiguousarray(_rgb(image)), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def crop224(frame: np.ndarray) -> np.ndarray:
    return resize224(center_square(frame))


def pad224(frame: np.ndarray) -> np.ndarray:
    return resize224(letterbox_square(frame))


@dataclasses.dataclass(frozen=True)
class RowMatch:
    """For each stored row: the recorded frame chosen, and its MSE against the row."""

    frame: np.ndarray
    mse: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        return self.frame - np.arange(len(self.frame))


def match_rows(stored: np.ndarray, padded: np.ndarray, window: int = WINDOW) -> RowMatch:
    """Match row i to the frame j in [i - window, i + window] whose `padded` image is closest (MSE).

    A row past the recording's end (i - window beyond the last frame) takes the last frame.
    """
    stored, padded = np.asarray(stored), np.asarray(padded)
    if stored.ndim != 4 or padded.ndim != 4 or stored.shape[1:] != padded.shape[1:]:
        raise ValueError(f"rows {stored.shape} and recorded frames {padded.shape} must be (N, H, W, 3) alike")
    m = len(padded)
    if m == 0:
        raise ValueError("an empty recording matches nothing")
    pads = padded.astype(np.float32)
    frames, mses = [], []
    for i in range(len(stored)):
        ref = stored[i].astype(np.float32)
        lo, hi = max(0, i - window), min(m - 1, i + window)
        if lo > hi:
            j, mse = m - 1, float(np.mean((ref - pads[m - 1]) ** 2))
        else:
            d = np.mean((pads[lo:hi + 1] - ref[None]) ** 2, axis=(1, 2, 3))
            k = int(np.argmin(d))
            j, mse = lo + k, float(d[k])
        frames.append(j)
        mses.append(mse)
    return RowMatch(np.asarray(frames, dtype=np.int64), np.asarray(mses, dtype=np.float64))


def match_report(match: RowMatch, n_recorded: int) -> dict:
    """The per-camera log of the reference builder: offsets, MSE quantiles, rows the recording does not cover."""
    offsets = match.offsets.tolist()
    if not offsets:
        raise ValueError("no rows to report")
    hist: dict[str, int] = {}
    for o in offsets:
        hist[str(o)] = hist.get(str(o), 0) + 1
    median = int(np.median(offsets))
    return {"recorded_frames": n_recorded, "offset_hist": hist, "median_offset": median,
            "mse_median": float(np.median(match.mse)), "mse_p95": float(np.percentile(match.mse, 95)),
            "mse_max": float(np.max(match.mse)),
            "rows_outside_recording": sum(1 for i in range(len(offsets)) if not 0 <= i + median < n_recorded)}
