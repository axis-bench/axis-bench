"""Read per-episode control signals out of a local LeRobot v3.0 parquet dataset.

WHY THIS EXISTS
    The idle filter (idle_filter.py) is deliberately signal-agnostic: it takes an array
    and returns ranges. Something has to decide WHICH array, and that decision is a
    property of how our rendered datasets are laid out, not of the filter. That mapping
    lives here so the filter stays reusable and the choice stays reviewable in one place.

WHICH SIGNAL, AND WHY IT IS NOT openpi's
    openpi feeds `action_dict/joint_velocity` -- DROID's COMMANDED joint velocity -- and
    calls a frame idle when that command is not CHANGING (|diff| < 1e-3).

    Our commanded stream is the raw dataset's `action` column: a joint-POSITION target.
    It cannot be used the same way, and this is measured, not assumed. On task 1644
    (390 328 frames / 1202 episodes) the command is a 5 Hz zero-order hold sampled at
    15 fps: 76.9% of frames have an EXACTLY zero command delta, and the surviving deltas
    are quantised steps (~0.044 / ~0.089 rad). Retention through the openpi algorithm is
    consequently flat at 0.591 / 0.591 / 0.587 for idle_eps 1e-4 / 1e-3 / 1e-2 -- three
    orders of magnitude of threshold with 0.5 pp of effect. There is nothing to calibrate.

    So we use `observation.state` instead, the ACHIEVED state, and rely on an identity
    that holds exactly in our DROID-8D conversion (verified to 4.8e-7 on task 1644):

        droid8d action[:, :7] == fps * diff(state[:, :7])

    Feeding state positions to `idle_mask` with `forward=True` (which pretrain_ranges uses)
    therefore tests |action label at row t| = |fps*(state[t+1]-state[t])| < idle_eps * fps per
    joint. That is literally "this frame teaches the policy to hold still" -- the exact failure
    openpi's filter exists to prevent ("policies that output fewer stationary actions"). The
    forward alignment matters: the default backward diff would test action[t-1], one frame late
    (it drops the motion-onset frame and keeps a trailing idle frame). We get openpi's intent
    through our own action space rather than copying a signal our data does not have.

    The gripper channel is included (openpi's 7-D joint velocity omits it). During a grasp
    the arm is stationary while the fingers close; excluding the gripper would discard the
    single most informative moment of a pick-and-place. Measured over all 1202 episodes of
    task 1644 at idle_eps 1e-3: retention 0.7174 arm-only -> 0.7567 with the gripper, so
    3.9% of all frames are grasp/release dwell that an arm-only signal would throw away.

Offline (numpy + pandas/pyarrow). Runs under the offline environment.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from axis_qvla.data.convert_droid_actions import gripper_calibration, gripper_closedness

RAW_STATE_DIM = 9  # 7 arm joints + 2 finger widths (metres)
DROID8D_STATE_DIM = 8  # 7 arm joints + gripper closedness in [0, 1]

_STATE_COL = "observation.state"
_EPISODE_COL = "episode_index"


def _read_columns(root: Path, columns: list[str]) -> pa.Table:
    files = sorted(Path(root).glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root/'data'}")
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def _fixed_width(column: pa.ChunkedArray, width: int) -> np.ndarray:
    """List-of-float parquet column -> (N, width) float64, without a per-row Python list."""
    flat = column.combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(flat, dtype=np.float64).reshape(-1, width)


def state_to_idle_signal(state: np.ndarray) -> np.ndarray:
    """Map a raw (9-D) or DROID-8D (8-D) state matrix to the 8-D idle signal.

    9-D input has its two finger widths collapsed to the same closedness scalar the
    DROID-8D converter writes, using the same whole-dataset 1%/99% width quantiles, so
    raw and converted datasets produce byte-comparable ranges. Calibration MUST be global
    rather than per-episode: an episode that never opens the gripper would otherwise get
    its measurement noise stretched across the full [0, 1] range and read as motion.

    A dataset whose gripper never moves cannot be calibrated; there we drop the gripper
    channel rather than fail, since an arm-only signal is still correct, just blind to
    grasp dwells.
    """
    x = np.asarray(state, dtype=np.float64)
    if x.shape[1] == DROID8D_STATE_DIM:
        return x
    if x.shape[1] != RAW_STATE_DIM:
        raise ValueError(f"expected {RAW_STATE_DIM}-D or {DROID8D_STATE_DIM}-D state, got {x.shape[1]}-D")
    open_w, closed_w = gripper_calibration(0.5 * (x[:, 7] + x[:, 8]))
    try:
        closedness = gripper_closedness(x[:, 7], x[:, 8], open_w, closed_w)
    except ValueError:
        return x[:, :7]
    return np.concatenate([x[:, :7], closedness[:, None]], axis=1)


@dataclasses.dataclass(frozen=True)
class DatasetSignals:
    """One task's idle signal, split by episode. Built once, queried per episode."""

    root: Path
    episode_index: np.ndarray  # (N,) int64, row-aligned with `signal`
    signal: np.ndarray  # (N, 8) float64

    @classmethod
    def load(cls, root: Path) -> DatasetSignals:
        table = _read_columns(Path(root), [_STATE_COL, _EPISODE_COL])
        width = len(table[_STATE_COL][0])
        state = _fixed_width(table[_STATE_COL], width)
        episodes = np.asarray(table[_EPISODE_COL].combine_chunks().to_numpy(), dtype=np.int64)
        return cls(Path(root), episodes, state_to_idle_signal(state))

    def episodes(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (episode_index, signal[T, 8]) in file order.

        LeRobot v3.0 writes rows grouped and ordered by episode, so runs of equal
        episode_index delimit episodes. We assert that rather than trust it: a shuffled
        file would silently produce garbage ranges that still look plausible.
        """
        ep = self.episode_index
        if len(ep) == 0:
            return
        if np.any(np.diff(ep) < 0):
            raise ValueError(f"{self.root}: episode_index is not non-decreasing")
        starts = np.flatnonzero(np.concatenate([[True], ep[1:] != ep[:-1]]))
        ends = np.concatenate([starts[1:], [len(ep)]])
        for s, e in zip(starts, ends):
            yield int(ep[s]), self.signal[s:e]
