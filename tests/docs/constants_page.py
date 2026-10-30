"""The constants registry's page, generated from each package's constants registry.

    python -m tests.docs.constants_page > constants.md
    python -m tests.docs.constants_page --label-table      # the table of docs/reference/data.md#quality-labels

A registry (`CONSTANTS` of axis_qvla.labels.constants, axis_qvla.bench.constants, axis_qvla.paper.constants and
axis_qvla.train.constants) names each number a result depends on that the paper does not spell out in
full, where it belongs in the paper, and where its value lives. That is a module attribute, or a key
of a shipped file: a path relative to the repository, a key path with `/` between levels, and
`len:<key>` to count a list. `resolve` reads the value; tests/docs/test_constants_page.py fails when a
registry and its source disagree.

Core tier: stdlib only.
"""
from __future__ import annotations

import importlib
import json
import tomllib

from tests.docs.runner import REPO

REGISTRIES = (("Labels and training weights", "axis_qvla.labels.constants"), ("Benchmark and scoring", "axis_qvla.bench.constants"),
              ("Reproducing the paper's tables", "axis_qvla.paper.constants"), ("Training", "axis_qvla.train.constants"))

HEADER = """# Constants

Look up a number here: every number a result depends on, with where the paper states it (an
equation, a section or an appendix of arXiv 2610.18869) or "implementation (not in the paper)", what
it means, and where the code keeps it. Generated from the packages' constants registries. The label
and training-weight constants are explained in context on the [labels page](labels.md).
"""


def constants():
    return [(title, importlib.import_module(module).CONSTANTS) for title, module in REGISTRIES]


def _file(source: str) -> bool:
    return source.endswith((".json", ".toml"))


def resolve(c) -> object:
    count = c.attribute.startswith("len:")
    key = c.attribute.removeprefix("len:")
    if _file(c.module):
        text = (REPO / c.module).read_text(encoding="utf-8")
        value = json.loads(text) if c.module.endswith(".json") else tomllib.loads(text)
        for part in key.split("/"):
            value = value[part]
    else:
        value = getattr(importlib.import_module(c.module), key)
    return len(value) if count else value


def fmt(v) -> str:
    if isinstance(v, str):
        return "`" + json.dumps(v) + "`"
    if isinstance(v, (tuple, list)):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return f"{v:,}"
    return repr(v)


def where(c) -> str:
    if not _file(c.module):
        return f"`{c.module}.{c.attribute}`"
    key = c.attribute.removeprefix("len:")
    count = "the count of " if c.attribute.startswith("len:") else ""
    return f"{count}`{key}` in `{c.module}`"


# The label constants a quality-method author needs, rendered the same way into the Quality labels
# section of docs/reference/data.md, between its `label-constants` markers
# (tests/docs/test_constants_page.py holds the page to `label_table()`).
DATA_PAGE = REPO / "docs" / "reference" / "data.md"
# The table names the part of the label pipeline each constant belongs to, by content rather than by
# equation, section or appendix number, which can move in a camera-ready.
DATA_LABELS = {
    "W_PROGRESS": "reward", "W_CONTACT": "reward", "CONTACT_TANH_K": "reward",
    "PHASE_LAMBDA": "phase curve", "PHASE_BONUS": "phase curve", "CONTACT_MIN_RUN": "phase curve",
    "EDGE_FRAMES": "phase curve", "MIN_APPROACH_M": "phase curve", "TARGET_CONTACT": "phase curve",
    "AWR_TAU": "training weights", "AWR_CAP": "training weights", "AWR_HORIZON": "training weights",
    "MIN_WINDOW_FRAMES": "training weights", "N_BINS": "quality tags", "NEUTRAL_Q": "quality tags",
    "DROP_WHOLE": "conditioning", "DROP_COMPONENT": "conditioning", "TAGGED_FRACTION": "conditioning",
    "PROMPT_MARKER": "conditioning",
}


def label_table() -> str:
    by_name = {c.name: c for c in importlib.import_module("axis_qvla.labels.constants").CONSTANTS}
    rows = "\n".join(f"| `{c.name}` | {fmt(c.value)} | {part} | {c.meaning} | {where(c)} |"
                     for c, part in ((by_name[n], part) for n, part in DATA_LABELS.items()))
    return f"| constant | value | part | meaning | where |\n|---|---|---|---|---|\n{rows}"


def render() -> str:
    parts = [HEADER]
    for title, cs in constants():
        rows = "\n".join(f"| `{c.name}` | {fmt(c.value)} | {c.equation} | {c.meaning} | {where(c)} |" for c in cs)
        parts.append(f"## {title}\n\n| constant | value | paper | meaning | where |\n|---|---|---|---|---|\n{rows}\n")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys

    print(label_table() if sys.argv[1:] == ["--label-table"] else render(), end="")
