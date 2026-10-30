"""What the benchmark shares with the pretraining corpus: task ids (refused) and instructions (reported).

OFFLINE TIER: stdlib only, so both the build side and a CI check can import it.

A scored task that pretraining saw measures memorisation, not transfer, and nothing about a run
shows it: the pretrained arm just looks better. `assert_no_leakage` therefore refuses any benchmark
task id present in the pretraining roots index; none of the 24 is among the corpus's 731
(tests/unit/bench/tasks/test_heldout_leakage.py). Instructions are a weaker overlap: a different
task (another scene and object layout) can carry the same instruction, which a language-conditioned
policy may still exploit. `instruction_overlap` reports them with the corpus episodes they touch,
using the rule of `normalise_instruction`; for the published corpus it finds three instructions under
four corpus task ids, 22 of 5,000 episodes.

The roots index is versioned: v2 nests the mapping under "roots" and also records the action space,
while v1 IS the mapping. Both are read, because a caller passing a v2 file to a v1 reader gets
five keys ("version", "base", "roots", ...) and a disjointness check that passes vacuously.
"""
from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from pathlib import Path

from axis_qvla.bench.spec import load_benchmark

_VARIANT = re.compile(r"\s+\d+$")


class LeakageError(AssertionError):
    """A scored task was present in pretraining. Raised, never warned: a warning in a log nobody
    reads is how a silent convention mismatch can survive training and scoring undetected."""


def read_roots(path: str | Path) -> dict[str, str]:
    """Task id -> relative root, from a v1 (bare mapping) or v2 (nested under "roots") index.

    Refuses an index whose shape it does not recognise rather than returning something empty: an
    empty mapping makes every leakage check pass.
    """
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(d).__name__}")
    if "roots" in d:                      # v2
        roots = d["roots"]
        if not isinstance(roots, dict):
            raise ValueError(f"{path}: 'roots' is {type(roots).__name__}, expected an object")
        return {str(k): str(v) for k, v in roots.items()}
    # v1: the mapping itself. Guard against a v2 file read by the v1 path, whose keys would be
    # metadata names and whose disjointness check would pass without comparing anything.
    if {"version", "base", "derived_from"} & set(d):
        raise ValueError(
            f"{path}: looks like a versioned index but has no 'roots' key; refusing to read its "
            f"metadata keys as task ids"
        )
    return {str(k): str(v) for k, v in d.items()}


def leaked_tasks(scored: list[str] | set[str], pretrain_roots: str | Path) -> list[str]:
    """The scored task ids that also appear in the pretraining index, numerically sorted."""
    pre = set(read_roots(pretrain_roots))
    hit = {str(t) for t in scored} & pre
    return sorted(hit, key=lambda s: (len(s), s))


def assert_no_leakage(scored: list[str] | set[str], pretrain_roots: str | Path) -> None:
    """Raise unless every scored task is absent from pretraining."""
    leaked = leaked_tasks(scored, pretrain_roots)
    if leaked:
        raise LeakageError(
            f"{len(leaked)} scored task(s) are in the pretraining set and would measure "
            f"memorisation, not transfer: {leaked}. Drop them from the scored list, or report the "
            f"benchmark as seen-task rather than held-out."
        )


def benchmark_task_ids(name: str = "franka_heldout24") -> list[str]:
    """Every task the benchmark SCORES, as strings, read from the tracked declaration.

    These are what must be absent from pretraining.
    """
    return [str(t) for t in load_benchmark(name).task_ids]


def normalise_instruction(text: str, *, variant: bool = False) -> str:
    """Case and whitespace folded; with `variant`, one trailing number dropped. The corpus names a
    second variant of a task 'Rearrange Three Items 2', and that is the same instruction; the
    benchmark's own prompts are canonical, so only the corpus side drops it."""
    folded = " ".join(str(text).lower().split())
    return _VARIANT.sub("", folded) if variant else folded


@dataclasses.dataclass(frozen=True)
class Overlap:
    task_id: int
    instruction: str
    corpus_tasks: tuple[tuple[str, int], ...]     # (corpus task id, its episodes), ascending by id

    @property
    def episodes(self) -> int:
        return sum(n for _, n in self.corpus_tasks)


def load_corpus_prompts(path: str | Path) -> dict[str, dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != "qvla.corpus_prompts/1":
        raise ValueError(f"{path}: not a qvla.corpus_prompts/1 file")
    return {str(k): v for k, v in doc["tasks"].items()}


def instruction_overlap(bench, corpus: Mapping[str, Mapping]) -> list[Overlap]:
    """Every benchmark task whose instruction a corpus task also carries, in benchmark order."""
    by_rule: dict[str, list[tuple[str, int]]] = {}
    for tid, t in corpus.items():
        by_rule.setdefault(normalise_instruction(t["prompt"], variant=True), []).append((str(tid), int(t["n_episodes"])))
    out = []
    for tid in bench.task_ids:
        prompt = bench.task(tid).prompt
        hits = by_rule.get(normalise_instruction(prompt))
        if hits:
            out.append(Overlap(int(tid), prompt.strip(), tuple(sorted(hits, key=lambda h: int(h[0])))))
    return out
