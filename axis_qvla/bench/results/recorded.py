"""What an evaluation's records say about serving and execution, as results-header fields.

Why: the serving condition and the executed constants of a results file are facts the rollouts
recorded -- every task's meta.json carries the server's handshake (`server_metadata`), the four
constants it executed and the deviations it executed (`execution_deviations`), and the run record
carries the driver's own verdict (`execution`). These readers turn those records into the
header's `serve` and `execution`, never from an arm's name, and refuse records that disagree:
an arm served or executed two ways is not one result. `axis_qvla.bench.results.build` calls them.

Core tier: stdlib only.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from axis_qvla.bench import execution
from axis_qvla.bench.results.format import ServeCondition
from axis_qvla.bench.sim.arms import DUAL_SIM

CONSTANTS = ("fps", "replan", "max_steps", "n_stable_frames")
BETA_TOLERANCE = 1e-9
# Quality tag 0 is the trainer's "untagged" sentinel (openpi's quality_conditioning.NO_TAG; the
# trained bins are 1-5), never a condition. The openpi server refuses `--quality-tag 0` when it
# starts, and so do the LeRobot and X-VLA reference servers (parse_quality_tag), which would
# otherwise append "\nQuality: 0" to every prompt, a condition no training stage produced. So a conditioned server publishing tag 0 is neither the
# bare cell (its prompts carry a suffix) nor a conditioned cell (no model was trained on that
# suffix): it is refused here, when a results file is built, and by the scorer, when one is read.
NO_TAG = 0


class BuildError(ValueError):
    """The evaluation on disk cannot be turned into a results file."""


def check_quality_tag(tag: object) -> None:
    """A quality tag names a bin: a positive integer (an int, or its decimal string in a results
    file). Raises ValueError for the untagged sentinel (see NO_TAG) and for anything else."""
    text = str(tag).strip()
    if text == str(NO_TAG):
        raise ValueError(f"quality tag {NO_TAG} is the untagged sentinel, not a condition: a server that "
                         f"conditions on it is neither the bare result nor a conditioned one. Serve bare (no "
                         f"tag) or a trained bin")
    if isinstance(tag, bool) or not (text.isascii() and text.isdigit() and int(text) > 0):
        raise ValueError(f"quality tag {tag!r} is not a quality bin (a positive integer)")


def parse_quality_tag(text: str) -> int:
    """argparse `type=` for a reference server's --quality-tag: the tag as an int, refused at
    parse time (check_quality_tag) so a server never starts serving a tag that no result could be
    built from."""
    try:
        check_quality_tag(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return int(str(text).strip())


def quality_fields(md: Mapping) -> dict[str, str]:
    """The serve fields a server's handshake states: {} for an unconditioned server, {"quality": <tag>}
    for a quality-tagged one. A conditioned server without a tag, or with the sentinel 0, is refused:
    no serve fields describe what it did."""
    if not md.get("cfg_conditioning"):
        return {}
    if md.get("quality_tag") is None:
        raise BuildError("the server is conditioned but publishes no quality_tag: no serve fields describe it")
    try:
        check_quality_tag(md["quality_tag"])
    except ValueError as exc:
        raise BuildError(str(exc)) from None
    return {"quality": str(md["quality_tag"])}


def serve_condition(metas: Sequence[Mapping]) -> ServeCondition:
    """How the arm was served, from the server's own handshake as every task directory recorded it
    (`server_metadata`): a quality-tagged server serves {"quality": <tag>}; a guided one
    (guidance_scale w != 0) adds an unconditional pass with the field dropped, at beta = 1 + w."""
    sm = [m.get("server_metadata") or {} for m in metas]
    if any(x != sm[0] for x in sm):
        raise BuildError("server_metadata differs between task directories: the run was served two ways")
    md = sm[0]
    g = float(md.get("guidance_scale") or 0.0)
    fields = quality_fields(md)
    if not fields:
        if g:
            raise BuildError(f"guidance_scale {g} on a server that is not conditioned")
        return ServeCondition(fields={}, uncond_fields=None, beta=1.0)
    if md.get("beta") is not None and abs(float(md["beta"]) - (1.0 + g)) > BETA_TOLERANCE:
        raise BuildError(f"the server publishes beta {md['beta']} beside guidance_scale {g}; beta is 1 + w")
    return ServeCondition(fields=fields, uncond_fields={} if g else None, beta=1.0 + g)


def _number(key: str, v: object) -> int | float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BuildError(f"meta.json records {key} = {v!r}, not a number")
    return int(v) if float(v).is_integer() else float(v)


def executed_constants(metas: Sequence[Mapping], manifest: Mapping | None, bench) -> dict:
    """{fps, replan, max_steps, n_stable_frames, off_protocol[, deviations]}. The constants are
    what every task's meta.json recorded (identical in all, or refused); the deviations join the
    run record's, the rollouts' and any recorded constant that is not the declared one, plus a
    rollout domain other than the dual-sim one a dual_sim benchmark declares. Two records that
    disagree on what executed are refused, and so are rollouts that recorded different deviations
    in different tasks. `off_protocol` is true for any deviation, and for a run the driver was
    told to let deviate (--off-protocol), deviation or not."""
    ex = [{k: m[k] for k in CONSTANTS if m.get(k) is not None} for m in metas]
    if any(e != ex[0] for e in ex):
        raise BuildError("execution constants differ between task directories: the run was executed two ways")
    out: dict = {k: _number(k, v) for k, v in ex[0].items()}
    rolled = [m["execution_deviations"] for m in metas if m.get("execution_deviations") is not None]
    if any(r != rolled[0] for r in rolled):
        raise BuildError("the rollouts recorded different execution deviations in different task directories")
    domains = {m.get("domain") for m in metas} - {None}
    if len(domains) > 1:
        raise BuildError(f"the rollouts ran in different domains: {sorted(domains)}")
    record = (manifest or {}).get("execution") or {}
    claims: dict[str, list] = {}
    for source in (record.get("deviations") or {}, rolled[0] if rolled else {}):
        for k, v in source.items():
            claims.setdefault(k, []).append(dict(v))
    decl = execution.declared(bench)
    for k, v in out.items():
        if decl is not None and v != decl[k]:
            claims.setdefault(k, []).append({"executed": v, "declared": decl[k]})
    if bench.dual_sim and domains and domains != {DUAL_SIM}:
        claims["domain"] = [{"executed": domains.pop(), "declared": DUAL_SIM}]
    dev = {}
    for k, vs in sorted(claims.items()):
        if any(x.get("executed") != vs[0].get("executed") for x in vs):
            raise BuildError(f"{k}: the run record and the rollouts disagree on what executed "
                             f"({sorted({repr(x.get('executed')) for x in vs})})")
        dev[k] = vs[0]
    out["off_protocol"] = bool(dev) or bool(record.get("off_protocol"))
    if dev:
        out["deviations"] = dev
    return out
