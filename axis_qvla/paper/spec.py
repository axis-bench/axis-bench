"""Load and validate experiments/paper/paper.json, the layout of the paper's simulation results.

Why: T0 renders what the paper shows -- Figs. 1, 4(c), 5, 6 and 7, Table I and the in-text numbers
of Secs. IV.B and V.B-V.F -- and every cell, label and comparison of that layout is data, not code.
A figure is a list of bars (VLA, group, label, cell); Table I declares, per VLA and training seed,
which cell plays each role and which role differences it reports; a claim names a kind of
computation and its cells (axis_qvla.paper.claims). Everything is checked here, at load, so a typo is a
named SpecError rather than a KeyError while rendering.

Core tier: stdlib only.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Mapping

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "paper" / "paper.json"
SCHEMA = "qvla.paper_layout/2"
# claim kind -> required keys (axis_qvla.paper.claims implements each)
KINDS: dict[str, frozenset[str]] = {
    "diff": frozenset({"a", "b", "dp"}), "rel": frozenset({"a", "ref", "dp"}), "sr": frozenset({"a", "dp"}),
    "means": frozenset({"groups", "dp"}), "signs": frozenset({"pairs"}),
    "table1_mean": frozenset({"row", "dp"}), "table1_signs": frozenset({"rows"}),
    "table1_vla_mean_signs": frozenset({"rows"}), "replicate_signs": frozenset({"vla", "a", "b"}),
}
_BAD_TEXT = (",", '"', "\n", "\r")


class SpecError(ValueError):
    """The paper layout is malformed."""


@dataclasses.dataclass(frozen=True)
class Bar:
    vla: str
    group: str
    label: str
    cell: str
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Figure:
    key: str
    title: str
    bars: tuple[Bar, ...]


@dataclasses.dataclass(frozen=True)
class Vla:
    key: str
    label: str
    bc: str
    ft_only: str


@dataclasses.dataclass(frozen=True)
class PaperSpec:
    paper: str
    vlas: tuple[Vla, ...]
    figures: tuple[Figure, ...]
    fig1: Mapping[str, str]
    table1: Mapping[str, object]
    sections: tuple[Mapping[str, object], ...]
    claims: tuple[Mapping[str, object], ...]

    def vla(self, key: str) -> Vla:
        return next(v for v in self.vlas if v.key == key)

    def cells(self) -> frozenset[str]:
        """Every cell the layout reads."""
        out = {b.cell for f in self.figures for b in f.bars}
        for reps in self.table1["replicates"].values():
            out |= {c for r in reps for c in r["cells"].values()}
        for c in self.claims:
            out |= {c[k] for k in ("a", "b", "ref") if k in c and c["kind"] != "replicate_signs"}
            out |= {x for p in c.get("pairs", ()) for x in (p["a"], p["b"])}
            out |= {x for g in c.get("groups", {}).values() for x in g}
        return frozenset(out)


def _need(doc: Mapping, key: str, where: str) -> object:
    if key not in doc:
        raise SpecError(f"{where}: missing {key!r}")
    return doc[key]


def _text(s: object, where: str) -> str:
    if not isinstance(s, str) or any(ch in s for ch in _BAD_TEXT):
        raise SpecError(f"{where}: {s!r} must be a string without commas, quotes or newlines (it is a CSV field)")
    return s


def _table1(t: Mapping, vlas: set[str]) -> None:
    roles = set(_need(t, "roles", "table1"))
    for r in _need(t, "rows", "table1"):
        if {r.get("a"), r.get("b")} - roles or not r.get("id") or not r.get("label"):
            raise SpecError(f"table1 row {r!r}: needs id, label and roles a, b from {sorted(roles)}")
    reps = _need(t, "replicates", "table1")
    if set(reps) != vlas:
        raise SpecError(f"table1 replicates cover {sorted(reps)}, need {sorted(vlas)}")
    for v, rs in reps.items():
        if len({r["seed"] for r in rs}) != len(rs) or any(set(r["cells"]) != roles for r in rs):
            raise SpecError(f"table1 replicates of {v}: one entry per seed, each naming every role")


def _claims(claims: list, vlas: set[str], rows: set[str], roles: set[str]) -> None:
    ids = [c.get("id") for c in claims]
    if len(set(ids)) != len(ids):
        raise SpecError("duplicate claim ids")
    for c in claims:
        where = f"claim {c.get('id')!r}"
        need = KINDS.get(c.get("kind"))
        if need is None or not need <= set(c) or not c.get("label") or not c.get("section"):
            raise SpecError(f"{where}: kind must be one of {sorted(KINDS)}, with {sorted(need or ())}, label, section")
        named = [c["row"]] if "row" in c else list(c.get("rows", ()))
        if set(named) - rows or (c["kind"] == "replicate_signs" and ({c["a"], c["b"]} - roles or c["vla"] not in vlas)):
            raise SpecError(f"{where}: names an unknown Table I row, role or VLA")


def load_spec(path: pathlib.Path | None = None) -> PaperSpec:
    p = pathlib.Path(path) if path is not None else DEFAULT_PATH
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise SpecError(f"{p}: schema must be {SCHEMA}")
    vlas = tuple(Vla(v["key"], v["label"], v["bc"], v["ft_only"]) for v in _need(doc, "vlas", "layout"))
    keys = {v.key for v in vlas}
    figures = []
    for f in _need(doc, "figures", "layout"):
        bars = []
        for b in _need(f, "bars", f"figure {f.get('key')}"):
            if b.get("vla") not in keys:
                raise SpecError(f"figure {f['key']}: bar {b!r} names an unknown VLA")
            bars.append(Bar(b["vla"], _text(b["group"], f["key"]), _text(b["label"], f["key"]),
                            _text(b["cell"], f["key"]), _text(b.get("note", ""), f["key"])))
        figures.append(Figure(f["key"], f["title"], tuple(bars)))
    t1 = _need(doc, "table1", "layout")
    _table1(t1, keys)
    rows = {r["id"] for r in t1["rows"]}
    fig1 = _need(doc, "fig1", "layout")
    if fig1.get("row") not in rows:
        raise SpecError(f"fig1 names Table I row {fig1.get('row')!r}, which does not exist")
    claims = _need(doc, "claims", "layout")
    _claims(claims, keys, rows, set(t1["roles"]))
    known = {f.key for f in figures} | {"fig1", "table1"}
    sections = tuple(_need(doc, "sections", "layout"))
    stray = sorted({x for s in sections for x in s["figures"]} - known)
    if stray:
        raise SpecError(f"sections name unknown figures: {stray}")
    return PaperSpec(doc["paper"], vlas, tuple(figures), fig1, t1, sections, tuple(claims))
