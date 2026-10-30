"""The real-world line's quality labels and training weights: the simulation benchmark's computations,
on phase curves.

OFFLINE TIER: numpy, pandas, pyarrow only. This package calls the pinned label and weight builders
(`axis_qvla.signals`) unchanged and adds what the real-world line needs: phase curves from the recordings
(phase.py), the curve store, a metadata-only view, the tail ranges, the pure-conditioning tags, gates,
the command line and the compiler seam that turns a build into compiled labels (bundles.py).
"""
