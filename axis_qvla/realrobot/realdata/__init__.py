"""Real-robot fine-tuning data tools: the FR3 idle filter, the tail-10 ranges, the 224 centre crop,
and your DROID-layout demonstrations in the training layout.

Three data steps sit between an FR3 recording and what training reads: which frames are cut as idle
(idle_filter, export), which frames may start an action chunk (tail), and what pixels a 224x224 training
image holds (crop224, crop224_dataset). The stored state layout is declared, not inferred
(state_order). A recording in the DROID layout becomes the per-task training layout in one step
(training_set).

`python -m axis_qvla.realrobot.realdata {idle-ranges,tail,crop224,training-set}`: see axis_qvla.realrobot.realdata.cli.

Offline tier: numpy and pyarrow; Pillow to decode stored PNG frames; OpenCV only for crop224's
resize and PyAV only for video recordings, each imported where it is used.
"""
from __future__ import annotations
