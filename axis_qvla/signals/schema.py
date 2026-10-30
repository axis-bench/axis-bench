"""The SignalBundle schema: what a compiled quality method hands a backbone's consumer.

A bundle fixes, for one training run, which corpus rows every optimizer step draws (`plan`), the
loss weight of every row (`weights`, NaN on rows no plan may draw), the prompt codes of every
drawn row (`cond`, rendered to text by `vocab`), how the trained policy is served (`serve`) and the
row space all of them index (`rowspace`). On disk it is content-addressed (axis_qvla.signals.bundle):
two bundles describe the same signal exactly when their ids are equal.

`program` is the method's declared identity (method, version, parameters) and `program_sha` its
canonical sha256; `inputs` names the digests of what it read; `legacy` and `legacy_arrays` describe
the files the paper's training runs consumed, which axis_qvla.signals.bundle.export_legacy writes back
byte for byte.

Stdlib only, so core-tier code (the results format, the scorer) can import these types. Arrays
are numpy arrays at run time and are typed `Any` here.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from axis_qvla.core.canon import dumps_canonical, sha256_bytes

SCHEMA = "qvla.signal_bundle/1"
ROWSPACE_SCHEMA = "qvla.rowspace/1"
REDUCERS = ("mean", "global_ratio", "per_rank_ratio")
FORMATS = ("openpi-legacy", "lerobot-legacy")
CODECS = ("npz", "json", "json-indent1", "json-indent2-nl")
CAPABILITIES = frozenset({"plan", "loss_weights", "cond_prompt", "guidance", "flow_time", "oracle"})
ROWSPACE_KINDS = ("corpus", "flat")
NO_CODE = 0        # the field is dropped on this row: the bare prompt, the unconditional branch
FORBIDDEN = 255    # never drawable; a consumer that reads one raises


class SchemaError(ValueError):
    """A bundle, row space or vocabulary violates the schema."""


def _positive_int(owner: str, name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError(f"{owner}.{name} must be a positive int, got {value!r}")


@dataclasses.dataclass(frozen=True)
class Budget:
    """Optimizer steps x global batch: how many row draws a plan holds."""

    steps: int
    global_batch: int

    def __post_init__(self) -> None:
        _positive_int("Budget", "steps", self.steps)
        _positive_int("Budget", "global_batch", self.global_batch)

    @property
    def samples(self) -> int:
        return self.steps * self.global_batch


@dataclasses.dataclass(frozen=True)
class ServeCondition:
    """How a trained policy is served. `fields` gives a code per conditioned field ({} is the bare
    prompt); `uncond_fields` is None for no guidance, else the fields of the unconditional pass;
    `beta` is v = v_u + beta * (v_c - v_u), so openpi's guidance scale w is beta - 1."""

    fields: Mapping[str, str]
    uncond_fields: Mapping[str, str] | None = None
    beta: float = 1.0

    @property
    def guided(self) -> bool:
        return self.uncond_fields is not None


BARE = ServeCondition({}, None, 1.0)


@dataclasses.dataclass(frozen=True)
class CondVocab:
    """Code -> exact prompt text per conditioned field, and the corpus instructions the texts are
    appended to (what a consumer tokenizes to check its token budget)."""

    fields: tuple[str, ...]
    text: Mapping[str, Mapping[int, str]]
    prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.fields)) != len(self.fields):
            raise SchemaError(f"a field is declared twice: {self.fields}")
        extra = set(self.text) - set(self.fields)
        if extra:
            raise SchemaError(f"code text for undeclared fields {sorted(extra)}")
        for f in self.fields:
            codes = self.text.get(f)
            if not codes:
                raise SchemaError(f"field {f!r} has no code text")
            bad = [c for c in codes
                   if isinstance(c, bool) or not isinstance(c, int) or not NO_CODE < c < FORBIDDEN]
            if bad:
                raise SchemaError(f"field {f!r}: codes must be ints in 1..254, got {bad}")

    def render(self, prompt: str, codes) -> str:
        """The exact text a consumer trains or serves on for one row's codes (one per field)."""
        codes = [int(c) for c in codes]
        if len(codes) != len(self.fields):
            raise SchemaError(f"{len(codes)} codes for {len(self.fields)} fields")
        out = prompt
        for f, c in zip(self.fields, codes):
            if c == NO_CODE:
                continue
            if c not in self.text[f]:
                raise SchemaError(f"field {f!r}: code {c} is not drawable")
            out += self.text[f][c]
        return out

    def worst_case_prompt(self) -> str:
        """The longest instruction followed by the longest text of every field."""
        base = max(self.prompts, key=lambda s: (len(s), s)) if self.prompts else ""
        return base + "".join(max(self.text[f].values(), key=lambda s: (len(s), s)) for f in self.fields)


@dataclasses.dataclass(frozen=True, eq=False)
class RowSpace:
    """The flat row index a bundle's plan, weights and codes address.

    `arrays` of a corpus row space: `episode_task`, `episode_index`, `episode_from` (global row of
    frame 0), `episode_length`, `episode_attempt` (one per episode, walk order), `ranges_entry`
    (the episode of each ranges-file key, in file order), `ranges_counts` and `ranges_spans`
    (the half-open spans, file order). A flat row space has none."""

    name: str
    kind: str
    n_rows: int
    n_trainable: int
    corpus_fingerprint: str | None
    source: str
    roots: Any = None
    roots_path: str | None = None
    roots_codec: str | None = None
    ranges_path: str | None = None
    ranges_codec: str | None = None
    prompts: tuple[str, ...] = ()
    idle_filter: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    heldout_excluded: tuple[int, ...] = ()
    arrays: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ROWSPACE_KINDS:
            raise SchemaError(f"RowSpace.kind must be one of {ROWSPACE_KINDS}, got {self.kind!r}")
        _positive_int("RowSpace", "n_rows", self.n_rows)
        if isinstance(self.n_trainable, bool) or not 0 < self.n_trainable <= self.n_rows:
            raise SchemaError(f"RowSpace.n_trainable must be in 1..n_rows, got {self.n_trainable!r}")
        if self.roots_codec is not None and self.roots_codec not in CODECS:
            raise SchemaError(f"RowSpace.roots_codec must be one of {CODECS}, got {self.roots_codec!r}")
        if self.ranges_codec is not None and self.ranges_codec not in CODECS:
            raise SchemaError(f"RowSpace.ranges_codec must be one of {CODECS}, got {self.ranges_codec!r}")


@dataclasses.dataclass(frozen=True)
class LegacyFile:
    """One file a paper training run consumed: where it goes, its encoding, and its content as an
    ordered tree whose array leaves name bundle arrays (see axis_qvla.signals.legacy)."""

    format: str
    lock_id: str | None
    path: str
    codec: str
    doc: Any

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise SchemaError(f"legacy format must be one of {FORMATS}, got {self.format!r}")
        if self.codec not in CODECS:
            raise SchemaError(f"legacy codec must be one of {CODECS}, got {self.codec!r}")
        parts = self.path.split("/")
        if self.path.startswith("/") or ".." in parts or "" in parts:
            raise SchemaError(f"legacy path {self.path!r} must be relative and stay inside the export root")


def _check_serve_fields(vocab: CondVocab | None, fields: Mapping[str, str], where: str) -> None:
    """A ServeCondition's `fields` (or `uncond_fields`) must name only the bundle's vocab fields,
    with codes the vocab can actually render (0, the dropped code, is always allowed)."""
    if not fields:
        return
    if vocab is None:
        raise SchemaError(f"{where} names fields {sorted(fields)} but the compiled labels have no vocab")
    extra = set(fields) - set(vocab.fields)
    if extra:
        raise SchemaError(f"{where}: fields {sorted(extra)} are not in the vocab")
    for f, code in fields.items():
        try:
            c = int(code)
        except (TypeError, ValueError):
            raise SchemaError(f"{where}: field {f!r} code {code!r} is not an int") from None
        if c != NO_CODE and c not in vocab.text[f]:
            raise SchemaError(f"{where}: field {f!r} code {c} is not drawable")


def check_serve(vocab: CondVocab | None, serve) -> None:
    """The serve rule, shared by the bundle and the compiler: every condition's `fields`, and its
    `uncond_fields` when it guides, name only vocab fields with integer codes the vocab renders
    (0, the dropped code, is always allowed)."""
    for s in serve:
        _check_serve_fields(vocab, s.fields, "serve.fields")
        if s.uncond_fields is not None:
            _check_serve_fields(vocab, s.uncond_fields, "serve.uncond_fields")


@dataclasses.dataclass(frozen=True, eq=False)
class SignalBundle:
    """A compiled quality method for one run. See the module docstring."""

    program: Mapping[str, Any]
    program_sha: str
    rowspace: RowSpace
    budget: Budget
    seed: int
    plan: Any
    weights: Any = None
    reducer: str = "mean"
    cond: Any = None
    vocab: CondVocab | None = None
    serve: tuple[ServeCondition, ...] = (BARE,)
    controls: tuple[str, ...] = ()
    requires: frozenset[str] = frozenset({"plan"})
    diagnostics: Mapping[str, float] = dataclasses.field(default_factory=dict)
    inputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    legacy: tuple[LegacyFile, ...] = ()
    legacy_arrays: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reducer not in REDUCERS:
            raise SchemaError(f"reducer must be one of {REDUCERS}, got {self.reducer!r}")
        if (self.weights is None) != (self.reducer == "mean"):
            raise SchemaError("weights and reducer disagree: weighted compiled labels declare global_ratio or "
                              "per_rank_ratio, unweighted ones mean")
        if (self.cond is None) != (self.vocab is None):
            raise SchemaError("cond and vocab come together: codes nobody can render are unusable")
        unknown = set(self.requires) - CAPABILITIES
        if unknown:
            raise SchemaError(f"unknown capabilities {sorted(unknown)}")
        if not self.serve:
            raise SchemaError("compiled labels declare at least one serve condition")
        if self.weights is not None and "loss_weights" not in self.requires:
            raise SchemaError("compiled labels with weights must declare 'loss_weights' in requires")
        if self.cond is not None and "cond_prompt" not in self.requires:
            raise SchemaError("compiled labels with cond must declare 'cond_prompt' in requires")
        check_serve(self.vocab, self.serve)
        try:
            want = sha256_bytes(dumps_canonical(self.program))
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"program is not canonical JSON: {exc}") from exc
        if self.program_sha != want:
            raise SchemaError(f"program_sha {self.program_sha!r} is not the canonical sha256 of program")
