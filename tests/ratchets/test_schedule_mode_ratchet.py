"""The live signal builder's MODES must not grow a mode the paper's bundles never compile.

`conf/signals/paper_bundles.toml` is the paper's own declaration of every training signal it
ships (`axis_qvla.signals.paper.load_conf`). Two of its methods -- `drop_top50` and `random50` --
compile through `axis_qvla.signals.planners.SchedulePlanner`, whose `mode` follows
`axis_qvla.signals.methods.filtering`'s own rule exactly: `DropTop50.run` always hands it "drop_top";
`Random50.run` hands it "drop_top" for an episode-granularity control and "drop_random"
otherwise. `_schedule_modes_from_paper_bundles` below re-derives that rule from the declared
bundles rather than hand-listing the two mode strings, so a third schedule-reachable method added
to `paper_bundles.toml` without also being wired through `SchedulePlanner` construction widens
the derived set automatically instead of silently passing.

`vanilla` and `drop` are the two modes no bundle ever needs to declare: `vanilla` is drawn by
`PermutationPlanner` (the `bc`/`awr_*`/`cfg` methods' "openpi-rowsampler" planner), never
`SchedulePlanner`, and `drop` (WVM's kappa=0 keep-mask, with no top-fraction parameter to declare)
has no method class of its own in `axis_qvla.signals.methods` at all. They are named explicitly, once,
as the two modes this ratchet grants without a paper-bundle citation; anything else must earn its
place by appearing in a bundle.

It holds the schedule builder to the modes a paper bundle compiles (plus vanilla and drop, which
need none).
"""
from __future__ import annotations

from axis_qvla.signals.blocks.schedule import MODES
from axis_qvla.signals.paper import load_conf

# The only two modes a bundle never needs to declare -- see the module docstring.
_NO_BUNDLE_NEEDED = frozenset({"vanilla", "drop"})


def _schedule_modes_from_paper_bundles() -> frozenset[str]:
    """Every mode `axis_qvla.signals.methods.filtering` would hand `SchedulePlanner`, for the bundles
    actually declared in conf/signals/paper_bundles.toml -- mirrors `DropTop50.run`/`Random50.run`
    exactly, so this drifts with the real compiler rather than against it."""
    modes = set()
    for entry in load_conf().values():
        if entry["method"] == "drop_top50":
            modes.add("drop_top")
        elif entry["method"] == "random50":
            modes.add("drop_top" if entry.get("granularity") == "episode" else "drop_random")
    return frozenset(modes)


def test_paper_bundles_actually_use_drop_top_and_drop_random():
    """Sanity on the derivation itself: if paper_bundles.toml stopped declaring either method, the
    ratchet below would silently stop checking anything for it."""
    modes = _schedule_modes_from_paper_bundles()
    assert modes == {"drop_top", "drop_random"}, modes


def test_live_modes_are_exactly_the_paper_bundles_plus_the_two_unbundled_modes():
    allowed = _schedule_modes_from_paper_bundles() | _NO_BUNDLE_NEEDED
    extra = set(MODES) - allowed
    assert not extra, (
        f"MODES has {extra} with no paper bundle behind it and no place on _NO_BUNDLE_NEEDED "
        f"(see this module's docstring) -- most likely a reintroduced `anneal`"
    )
    missing = allowed - set(MODES)
    assert not missing, f"a paper bundle needs mode(s) {missing} that MODES no longer offers"


def test_anneal_specifically_stays_gone():
    """`anneal` is not a mode: no released run used it."""
    assert "anneal" not in MODES
