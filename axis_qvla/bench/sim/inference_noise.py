"""Per-rollout inference noise: makes a rollout's sampler noise a pure function of (task, seed).

WHY THIS EXISTS. pi0.5 is a flow-matching sampler: every `infer()` call draws one noise tensor
(`openpi/src/openpi/models/pi0.py`, `jax.random.normal(rng, ...)`), and a 1500-step rollout makes
~188 such calls. The server, left to itself, draws them from a SINGLE per-process key created at
`Policy.__init__` (`jax.random.key(0)`) and advanced once per request. A rollout's actions therefore
depend on **how many requests every preceding rollout happened to make** -- 34 calls for a short
episode, 188 for a timeout. That is invisible while each task gets its own short-lived server, and
becomes a correctness bug the moment a server is reused across tasks or serves several workers.

So noise is not derived from server state at all. `noise_for` is a **pure function** of
(salt, task_id, seed, call_index): no generator, no cursor, no ordering assumption. Call 137 of a
rollout is the same array whether it is reached in sequence, after a resume, on another box, or
from a worker that has served a thousand unrelated requests in between. That is what makes the
persistent-server work queue safe: request order does not change the noise a rollout is given.
It does not make a rollout bitwise reproducible. The policy also sees rendered frames, and the
renderer's multisample colour resolve differs by +-1 LSB between renders, so a rollout repeated
in isolation agrees with the first statistically, not bit for bit.

Statistically each (task, seed) draws its OWN stream, so the 50 seeds of a task sample 50 distinct
(layout, noise) pairs. Noise variance is already integrated into the 50-sample estimate; evaluating
one layout under several noise draws would re-sample a layout instead of visiting a new one, and is
strictly less efficient (intra-layout correlation rho ~ 0.43 => design effect 1 + 2*rho = 1.86).

Tier: online (eval). numpy only -- no jax, no mujoco, so both the harness and tests can import it.
"""

from __future__ import annotations

import numpy as np

# Bump ONLY to deliberately re-roll an entire campaign's inference noise. Changing it invalidates
# every stored result, so it is a campaign-level decision, not a knob. "AXIS" in ASCII.
NOISE_SALT = 0x41584953

__all__ = ["NOISE_SALT", "noise_for", "noise_fingerprint"]


def noise_for(
    task_id: int,
    seed: int,
    call_index: int,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """The noise tensor for one `infer()` call, as a pure function of its coordinates.

    `action_horizon` and `action_dim` must come from the SERVER's advertised metadata, never from a
    local constant: a shape mismatch would be silently broadcast or padded by the model rather than
    raising, which is the same class of defect as a config whose convention is assumed instead of
    declared.
    """
    if min(task_id, seed, call_index, action_horizon, action_dim) < 0:
        raise ValueError(
            f"noise coordinates must be non-negative: task_id={task_id} seed={seed} "
            f"call_index={call_index} horizon={action_horizon} dim={action_dim}"
        )
    # SeedSequence spawns a well-distributed state from the tuple; unlike hashing it is a documented,
    # version-stable mapping, so this reproduces across numpy builds and platforms.
    ss = np.random.SeedSequence([NOISE_SALT, int(task_id), int(seed), int(call_index)])
    rng = np.random.Generator(np.random.Philox(ss))
    return rng.standard_normal((int(action_horizon), int(action_dim)), dtype=np.float32)


def noise_fingerprint(task_id: int, seed: int, action_horizon: int, action_dim: int, n: int = 8) -> str:
    """A short digest over the first `n` draws of a rollout's stream, for provenance in meta.json.

    Recording this lets a later run prove it used the same noise stream without storing the arrays.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(f"{NOISE_SALT}:{task_id}:{seed}:{action_horizon}:{action_dim}".encode())
    for i in range(n):
        h.update(noise_for(task_id, seed, i, action_horizon, action_dim).tobytes())
    return h.hexdigest()[:16]
