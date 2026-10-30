"""Results as data: seed files, eval fingerprints, the qvla.results/v1 format and the scorer.

Why this package exists: every released number is recomputed from raw `seed_<n>.json` bytes, and
every copy of those bytes is verified against the fingerprint stamped at evaluation time. Reading,
fingerprinting and scoring live together so there is exactly one implementation of each rule: how a
result is read, fingerprinted and scored.

Core tier: stdlib only at module level (plus first-party modules under the import root),
Python >= 3.11.
"""
from __future__ import annotations
