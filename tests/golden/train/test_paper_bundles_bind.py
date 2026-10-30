"""Every paper arm binds against the staged bundles: its bundle loads and fits its run.

Slow. `--pretraining-data` names a qvla-v1-corpus tree holding the signal compiler's staged bundles/
(`qvla labels build-paper`), read through tests.conftest.require_option, which fails rather than
skips when it is missing. The 32 pi0.5 finetunes name no bundle (their plan is compiled at bind
time) and are not checked here.
"""
from __future__ import annotations

import collections
import pathlib

import pytest

from axis_qvla.train.armspec import load_armspec
from axis_qvla.train.bind import check_bundle, run_shape
from axis_qvla.train.paper import paper_arm_files
from axis_qvla.train.recipes import load_recipe
from tests.conftest import require_option

pytestmark = pytest.mark.slow


def bind_problems(arm_files, bundles_root: pathlib.Path) -> tuple[int, int, dict[str, list[str]]]:
    """(arms that name a bundle, distinct bundles, {arm id: problems}); each bundle is loaded once."""
    from axis_qvla.signals.bundle import load_bundle

    by_bundle = collections.defaultdict(list)
    for path in arm_files:
        spec = load_armspec(path)
        if spec.bundle is not None:
            by_bundle[spec.bundle].append(spec)
    problems: dict[str, list[str]] = {}
    for ref, specs in sorted(by_bundle.items()):
        try:
            bundle = load_bundle(bundles_root / ref[len("sha256:"):])
        except (OSError, ValueError) as exc:
            problems.update({s.id: [f"cannot load {ref}: {exc}"] for s in specs})
            continue
        for spec in specs:
            steps, gb, pr, w = run_shape(spec, load_recipe(spec.preset), smoke=False)
            found = check_bundle(spec, bundle, steps=steps, global_batch=gb, per_rank=pr, world=w, smoke=False)
            if found:
                problems[spec.id] = found
    return sum(map(len, by_bundle.values())), len(by_bundle), problems


def test_every_paper_arm_binds_against_the_staged_bundles(request):
    pack = pathlib.Path(require_option(request.config, "--pretraining-data"))
    n_arms, n_bundles, problems = bind_problems(paper_arm_files(), pack / "bundles")
    assert problems == {} and n_arms == 153 and n_bundles <= 53
