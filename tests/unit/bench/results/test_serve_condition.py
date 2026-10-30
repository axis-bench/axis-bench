"""One ServeCondition: the results format re-exports the signal compiler's, field for field."""
from __future__ import annotations

import dataclasses

from axis_qvla.bench.results import format as fmt
from axis_qvla.signals import schema


def test_results_and_bundles_share_one_serve_condition():
    assert fmt.ServeCondition is schema.ServeCondition
    assert [f.name for f in dataclasses.fields(fmt.ServeCondition)] == ["fields", "uncond_fields", "beta"]
    assert fmt.ServeCondition({"quality": "5"}, None, 1.0) == schema.ServeCondition(fields={"quality": "5"})
