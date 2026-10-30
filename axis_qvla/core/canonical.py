"""Module names inside recorded identities are spelled under one canonical import root.

WHY. A row space, a bundle and every record over them is content-addressed: its id is the sha256
of its canonical JSON, so every string it records is part of the id. Some of those strings name
code -- the idle filter a row space was cut with, the builder that made a row space -- and the
published ids, the signals lock and the paper config hashes all spell those names under the
canonical root below. The package may import under another root, so a name derived from the live
package would give a rebuilt row space another id than the published one, and so every bundle over
it and every arm bound to one.

HOW. The root is a plain assignment, and every recorded module name is built from it by
`canonical_name`. No string literal here or at an identity site spells a root-prefixed module name,
so the recorded bytes are the same whatever the package imports under. The tests require every
recorded name, row-space id and bundle id to come out unchanged;
tests/unit/signals/test_recorded_names.py pins the ids.

A recorded name is a record, not an import target: it names a module under the canonical root, which
may import under another one. `live_name` gives the spelling the running package imports it by.

Core tier: stdlib only.
"""
from __future__ import annotations

# A plain assignment, never a path segment or an import_module argument, so a rename keeps it.
CANONICAL_ROOT = "axis"
# The root this package imports under.
LIVE_ROOT = __name__.split(".")[0]


def canonical_name(dotted: str) -> str:
    """`<canonical root>.<dotted>`: a module or function name as every recorded identity spells it."""
    return CANONICAL_ROOT + "." + dotted


def live_name(recorded: str) -> str:
    """A recorded name as the running tree imports it."""
    head, sep, rest = recorded.partition(".")
    if head != CANONICAL_ROOT or not sep or not rest:
        raise ValueError(f"{recorded!r} is not a module name under the canonical root {CANONICAL_ROOT!r}")
    return LIVE_ROOT + "." + rest
