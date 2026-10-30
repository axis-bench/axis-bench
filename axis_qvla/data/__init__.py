"""Get, verify and inspect the benchmark's data. Core tier, except convert_droid_actions.py and
lerobot_local.py, which read LeRobot parquet with numpy/pandas/pyarrow at module scope and are
offline tier (tests/ratchets/test_venv_tiers.py's CORE_OFFLINE)."""
from __future__ import annotations
