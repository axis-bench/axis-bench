"""Every number of the label and signal pipeline the paper does not spell out, by name.

Each entry gives the value, where it belongs in the paper, what it means, and the module attribute
that holds it; tests/unit/labels/test_constants.py fails if a module's value moves without this
table.

Stdlib only.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Constant:
    name: str
    value: object
    equation: str
    meaning: str
    module: str
    attribute: str


CONSTANTS: tuple[Constant, ...] = (
    Constant("W_PROGRESS", 0.8, "Eq. 1", "weight of task progress in the reward", "axis_qvla.labels.reward", "W_PLACE_V2"),
    Constant("W_CONTACT", 0.2, "Eq. 1", "weight of the hand-object contact shaping term", "axis_qvla.labels.reward", "W_CONTACT_V2"),
    Constant("CONTACT_TANH_K", 5.0, "Eq. 1", "contact shaping is 1 - tanh(k * hand-object distance in m)",
             "axis_qvla.labels.reward", "CONTACT_TANH_K"),
    Constant("PROGRESS_EPS_M", 0.005, "Eq. 1", "floor (m) on the initial object-to-goal distance progress is normalised by",
             "axis_qvla.labels.reward", "PROGRESS_EPS_M"),
    Constant("CLIP_TAIL_FRAMES", 0, "Eq. 1", "trace steps kept after the success frame (the reward stops at success)",
             "axis_qvla.labels.reward", "CLIP_TAIL_FRAMES"),
    Constant("HOLD_STABILITY_BONUS", 0.0, "Eq. 1", "bonus for holding at the goal (none)", "axis_qvla.labels.reward",
             "HOLD_STABILITY_BONUS"),
    Constant("RENDER_FRAME_OFFSET", 2, "implementation (not in the paper)",
             "render row of the first trace step (5 Hz trace onto 15 fps rows)",
             "axis_qvla.labels.resample", "RENDER_FRAME_OFFSET"),
    Constant("RENDER_FPS", 15.0, "Sec. III.A", "render rows per second", "axis_qvla.labels.pipeline", "RENDER_FPS"),
    Constant("PHASE_LAMBDA", 0.5, "Eq. 3", "height of the approach band", "axis_qvla.labels.phase", "LAMBDA"),
    Constant("PHASE_BONUS", 0.10, "Eq. 3", "step at the start of the manipulation band", "axis_qvla.labels.phase", "BONUS"),
    Constant("CONTACT_MIN_RUN", 3, "Eq. 3", "frames of continuous target contact that make stable contact",
             "axis_qvla.labels.phase", "MIN_RUN"),
    Constant("EDGE_FRAMES", 5, "Eq. 3", "contact within this many frames of either end is not a phase boundary",
             "axis_qvla.labels.phase", "EDGE_FRAMES"),
    Constant("MIN_APPROACH_M", 0.02, "Eq. 3", "an approach shorter than this (m) is no approach", "axis_qvla.labels.phase",
             "MIN_APPROACH_M"),
    Constant("TARGET_CONTACT", ("contact_target_finger", "contact_target_eef"), "Eq. 3",
             "target contact by a finger OR the hand/palm", "axis_qvla.labels.phase", "TARGET_CONTACT"),
    Constant("AWR_TAU", 10.0, "Sec. IV.C", "w = min(exp(tau * Delta), cap)", "axis_qvla.signals.blocks.awr", "DEFAULT_TAU"),
    Constant("AWR_CAP", 2.0, "Sec. IV.C", "upper bound of w", "axis_qvla.signals.blocks.awr", "DEFAULT_CAP"),
    Constant("AWR_HORIZON", 10, "Eq. 4", "Delta_t = V(min(t + H - 1, T - 1)) - V(t)", "axis_qvla.signals.blocks.awr",
             "DEFAULT_HORIZON"),
    Constant("MIN_WINDOW_FRAMES", 3, "Sec. IV.B",
             "a phase-clamped window shorter than this has Delta = 0; longer ones are rescaled by (H - 1) / span",
             "axis_qvla.signals.methods.awr_chunk", "MIN_WINDOW_FRAMES"),
    Constant("PHASE_FIT_MIN_FRAMES", 10_000, "implementation (not in the paper)",
             "the per-phase quantile fit needs at least this many trusted frames per phase",
             "axis_qvla.signals.methods.awr_chunk", "PHASE_FIT_MIN_FRAMES"),
    Constant("MAX_CAPPED_RATE", 0.5, "implementation (not in the paper)", "the pooled capped rate the phases are matched to "
             "must lie in (0, this)", "axis_qvla.signals.methods.awr_chunk", "MAX_CAPPED_RATE"),
    Constant("N_BINS", 5, "Eq. 5", "quality tags are global quintiles", "axis_qvla.signals.blocks.schedule", "N_BINS"),
    Constant("NEUTRAL_Q", 3, "Eq. 5", "tag of an episode whose weights are all 1 (unscored)",
             "axis_qvla.signals.blocks.schedule", "NEUTRAL_Q"),
    Constant("DROP_WHOLE", 0.15, "App. A", "probability the whole condition is dropped from a row",
             "axis_qvla.signals.blocks.tags", "DROP_WHOLE"),
    Constant("DROP_COMPONENT", 0.05, "App. A", "probability the quality field is dropped, independently",
             "axis_qvla.signals.blocks.tags", "DROP_COMPONENT"),
    Constant("TAGGED_FRACTION", (1.0 - 0.15) * (1.0 - 0.05), "App. A", "share of trainable rows that carry a tag (80.75%)",
             "axis_qvla.signals.blocks.tags", "TAGGED_FRACTION"),
    Constant("PROMPT_MARKER", "\nQuality: ", "Sec. IV.C", "text appended to the instruction before the tag",
             "axis_qvla.signals.methods.cfg", "PROMPT_MARKER"),
    Constant("FILTER_KEEP_FRAC", 0.5, "Sec. IV.C",
             "share of trainable frames quality filtering keeps (ranked, ties in pretraining-data order)",
             "axis_qvla.signals.blocks.schedule", "DEFAULT_KEEP_TOP_FRAC"),
    Constant("SEED_CONTROL", 42, "App. A", "seed of the random-50 control runs", "axis_qvla.signals.methods.random50",
             "SEED_CONTROL"),
    Constant("IDLE_EPS", 1e-3, "implementation (not in the paper)",
             "a frame is idle when every state dimension changes less than this",
             "axis_qvla.labels.idle_filter", "DROID_IDLE_EPS"),
    Constant("MIN_IDLE_LEN", 7, "implementation (not in the paper)", "idle runs at least this long are removed",
             "axis_qvla.labels.idle_filter", "MIN_IDLE_LEN"),
    Constant("MIN_NON_IDLE_LEN", 16, "implementation (not in the paper)", "kept runs shorter than this are removed",
             "axis_qvla.labels.idle_filter", "MIN_NON_IDLE_LEN"),
    Constant("TRIM_LAST_N", 10, "implementation (not in the paper)", "frames trimmed from the end of every kept run",
             "axis_qvla.labels.idle_filter", "TRIM_LAST_N"),
)
