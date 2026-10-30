"""Every number the benchmark, its policy-server protocol and its scorer fix that a result depends on,
by name.

The registry pattern of axis_qvla.labels.constants (its `Constant`). Each entry gives the value, where the
paper states it (an equation, a section, a table, a figure or an appendix) or "implementation (not in
the paper)", what it means, and where it lives: a module attribute, or a key of a shipped file (`/`
between levels, `len:` for a count). A constants page is generated from it (tests/docs/constants_page.py), and
tests/docs/test_constants_page.py fails when a value moves without its entry.

Core tier: stdlib only.
"""
from __future__ import annotations

from axis_qvla.labels.constants import Constant

DECLARATION = "conf/benchmarks/axis-heldout-v1/franka_heldout24.json"
EXECUTION = "conf/benchmarks/axis-heldout-v1/execution.json"
NOT_IN_PAPER = "implementation (not in the paper)"

CONSTANTS: tuple[Constant, ...] = (
    Constant("N_TASKS", 24, "Sec. III.B", "held-out tasks; all-24 is the mean of their success rates",
             DECLARATION, "len:task_ids"),
    Constant("N_SEEDS", 50, "Sec. V.A", "test seeds (initial scene layouts) per task", DECLARATION, "n_seeds"),
    Constant("FPS", 15, "Sec. III.A", "control and render rate (Hz)", EXECUTION, "fps"),
    Constant("REPLAN", 8, "App. A", "action rows executed per policy call before the next observation",
             EXECUTION, "replan"),
    Constant("MAX_STEPS", 1500, NOT_IN_PAPER, "steps before an episode fails (100 s)", EXECUTION, "max_steps"),
    Constant("N_STABLE_FRAMES", 45, NOT_IN_PAPER, "consecutive frames the checker must hold for success (3 s)",
             EXECUTION, "n_stable_frames"),
    Constant("STATE_DIM", 8, "Sec. III.A", "state sent to a policy: seven joint angles (rad), then gripper closedness",
             "axis_qvla.bench.protocol", "STATE_DIM"),
    Constant("MIN_ACTION_DIM", 8, "Sec. III.A", "action columns a reply carries: seven joint velocities (rad/s), "
             "then gripper closedness", "axis_qvla.bench.protocol", "MIN_ACTION_DIM"),
    Constant("ACTION_FPS", 15, "Sec. III.A", "rate a reply's actions are executed at (Hz)", "axis_qvla.bench.protocol",
             "ACTION_FPS"),
    Constant("PROTOCOL_REPLAN", 8, "App. A", "the protocol's default rows executed open loop per request",
             "axis_qvla.bench.protocol", "DEFAULT_REPLAN"),
    Constant("RAW_HEIGHT", 360, "App. B", "height of the camera frames sent to a policy (px)", "axis_qvla.bench.protocol",
             "RAW_HEIGHT"),
    Constant("RAW_WIDTH", 640, "App. B", "width of the camera frames sent to a policy (px)", "axis_qvla.bench.protocol",
             "RAW_WIDTH"),
    Constant("IMAGE_KEYS", ("base_0_rgb", "left_wrist_0_rgb"), NOT_IN_PAPER, "the two camera keys of a request",
             "axis_qvla.bench.protocol", "IMAGE_KEYS"),
    Constant("N_BOOT", 10_000, NOT_IN_PAPER, "seed-bootstrap resamples of the all-24 mean (the paper's 10,000 "
             "resamples, Sec. IV.B, are of its human-study correlations)", "axis_qvla.bench.results.bootstrap", "N_BOOT"),
    Constant("BOOT_SEED", 7, NOT_IN_PAPER, "seed of the bootstrap's random.Random", "axis_qvla.bench.results.bootstrap", "SEED"),
    Constant("LEVEL", 0.95, NOT_IN_PAPER, "coverage of the bootstrap and Wilson intervals",
             "axis_qvla.bench.results.bootstrap", "LEVEL"),
    Constant("Z95", 1.959963984540054, NOT_IN_PAPER, "the two-sided 95 % normal quantile of the Wilson interval",
             "axis_qvla.bench.results.bootstrap", "Z95"),
    Constant("VIEWS", ("all24", "informative_20", "usable_12"), NOT_IN_PAPER,
             "the one aggregate and the two appendix views", "axis_qvla.bench.results.score", "VIEWS"),
    Constant("MIN_CLAIM_SEEDS", 3, "Sec. V.F", "paired training seeds a contrast needs before it states a claim",
             "axis_qvla.bench.results.score", "MIN_CLAIM_SEEDS"),
    Constant("BETA_TOLERANCE", 1e-9, NOT_IN_PAPER, "how far a recorded guidance beta may be from 1 + w",
             "axis_qvla.bench.results.recorded", "BETA_TOLERANCE"),
)
