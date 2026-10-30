"""Where actions come from: a local torch checkpoint, or a served pi0.5 over a websocket.

They differ in more than transport -- a local checkpoint owns its sampler (and for `cfg`, its
guidance weight), while a served policy owns its sampler on the SERVER, so guidance there is
verified rather than set.

THE ACTION BUFFER IS PER SEED, NOT PER SWEEP. `policy_local.build_local_policy` returns a *factory*; each
call yields a fresh `policy_fn` with an empty chunk buffer. Reusing one across seeds leaks the
tail of the previous rollout's action chunk into the first steps of the next, which shows up
as an unexplained few-percent difference between two runs of the same checkpoint.

torch and openpi_client are imported inside functions: this module is read by tooling under
the offline venv, where neither is installed, and the venv-isolation ratchet pins that.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path


class PolicyError(ValueError):
    """A policy could not be built, or was described inconsistently with its server."""


def connect_served_policy(host: str, port: int) -> tuple[object, dict]:
    """Open the websocket client and return `(client, server_metadata)`."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=host, port=port)
    return client, client.get_server_metadata()


@dataclasses.dataclass(frozen=True)
class ServedConditioning:
    """How the server samples, as the client will NAME it in the arm directory.

    `source` records whether the values were read off the handshake ("server") or restated
    by the caller and verified against it ("asserted"). Both spellings land in meta.json so
    a reader can tell a run that trusted the server from one that pinned it.
    """

    cfg_conditioning: bool
    guidance_scale: float
    source: str


def served_conditioning(metadata: dict | None) -> tuple[bool, float]:
    """`(cfg_conditioning, guidance_scale)` as the server publishes them in its handshake.

    A server that publishes neither key is an unconditioned one: openpi's
    `create_trained_policy` sets `cfg_conditioning` exactly when a quality tag is served, and
    `serve_lerobot.build_metadata` mirrors that, so absence means the plain policy.
    """
    meta = metadata or {}
    return bool(meta.get("cfg_conditioning", False)), float(meta.get("guidance_scale", 0.0))


def check_served_guidance(
    metadata: dict | None, *, cfg_conditioning: bool, guidance_scale: float
) -> None:
    """Refuse to mislabel a run: the server owns the sampling config, we only name it.

    The websocket protocol carries observations, not sampling kwargs, so guidance is set when
    `serve_policy.py` is launched. Without this check a mislaunched server quietly writes w=2
    rollouts into the w=0 directory, and nothing downstream can tell.
    """
    served_cfg, served_w = served_conditioning(metadata)
    if served_cfg != cfg_conditioning or served_w != float(guidance_scale):
        raise PolicyError(
            f"server/client guidance mismatch: server has cfg_conditioning={served_cfg} "
            f"guidance_scale={served_w}, this run was invoked with "
            f"cfg_conditioning={cfg_conditioning} guidance_scale={guidance_scale}. "
            f"Relaunch serve_policy.py with the matching "
            f"--cfg-conditioning/--guidance-scale."
        )


def resolve_conditioning(
    metadata: dict | None,
    *,
    cfg_conditioning: bool | None = None,
    guidance_scale: float | None = None,
) -> ServedConditioning:
    """The conditioning a run is named after: read from the server, optionally asserted.

    The handshake is the source of record and the flags, when given, are an assertion:
    `--no-cfg-conditioning` against a tagged server is refused, and a caller who says nothing
    gets the server's values.

    An asserted `guidance_scale` alone (no conditioning flag) is checked against the served
    scale with the served conditioning; an asserted conditioning alone is checked with the
    served scale.
    """
    served_cfg, served_w = served_conditioning(metadata)
    if cfg_conditioning is None and guidance_scale is None:
        return ServedConditioning(served_cfg, served_w, source="server")
    want_cfg = served_cfg if cfg_conditioning is None else bool(cfg_conditioning)
    want_w = served_w if guidance_scale is None else float(guidance_scale)
    check_served_guidance(metadata, cfg_conditioning=want_cfg, guidance_scale=want_w)
    return ServedConditioning(want_cfg, want_w, source="asserted")
