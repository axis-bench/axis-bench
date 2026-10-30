"""Gate A of the reproducible-eval plan: the noise stream is a pure function of its coordinates.

If any of these fail, the persistent-server work queue is unsafe: results would depend on request
order again, which is precisely the defect the queue is designed around.
"""

from __future__ import annotations

import numpy as np
import pytest

from axis_qvla.bench.sim.inference_noise import NOISE_SALT, noise_for, noise_fingerprint

H, D = 10, 32  # pi0.5 action_horizon / padded action_dim, as advertised by the server


def test_same_coordinates_give_identical_bytes():
    """CATCHES a stateful generator sneaking back in: two independent calls must agree bitwise."""
    a = noise_for(806, 806_000, 137, H, D)
    b = noise_for(806, 806_000, 137, H, D)
    assert a.tobytes() == b.tobytes()
    assert a.shape == (H, D) and a.dtype == np.float32


def test_out_of_order_access_matches_sequential_access():
    """CATCHES an implementation whose call_index is really a cursor. A resumed or re-queued
    rollout reaches call 5 without having drawn 0..4; it must still get the same array."""
    sequential = [noise_for(1889, 1_889_003, i, H, D) for i in range(6)]
    assert noise_for(1889, 1_889_003, 5, H, D).tobytes() == sequential[5].tobytes()
    assert noise_for(1889, 1_889_003, 0, H, D).tobytes() == sequential[0].tobytes()


@pytest.mark.parametrize(
    "a,b",
    [
        ((806, 806_000, 0), (806, 806_001, 0)),  # different seed
        ((806, 806_000, 0), (807, 806_000, 0)),  # different task
        ((806, 806_000, 0), (806, 806_000, 1)),  # different call
    ],
)
def test_distinct_coordinates_give_distinct_noise(a, b):
    """CATCHES a key collapse -- e.g. XOR-ing the coordinates, which maps distinct rollouts onto one
    stream and would make 50 seeds silently share noise."""
    assert noise_for(*a, H, D).tobytes() != noise_for(*b, H, D).tobytes()


def test_stream_within_one_rollout_does_not_repeat():
    """CATCHES re-seeding per call with a rollout-level key, which would send the SAME noise on
    every one of the ~188 inference calls -- a degenerate sampler that still 'reproduces'."""
    seen = {noise_for(953, 953_007, i, H, D).tobytes() for i in range(64)}
    assert len(seen) == 64


def test_shape_comes_from_arguments_not_a_constant():
    """The harness must take horizon/dim from server metadata; a local constant would be a silent
    convention assumption (CLAUDE.md: declare the build, never assume it)."""
    assert noise_for(806, 806_000, 0, 4, 7).shape == (4, 7)


def test_negative_coordinates_are_refused():
    """Refuse rather than fold: a -1 sentinel leaking in must not quietly alias another stream."""
    with pytest.raises(ValueError):
        noise_for(806, -1, 0, H, D)


def test_distribution_is_standard_normal():
    """Sanity: the model expects unit Gaussian noise. A wrong scale trains nothing but degrades
    sampling silently."""
    x = np.concatenate([noise_for(806, 806_000, i, H, D).ravel() for i in range(200)])
    assert abs(float(x.mean())) < 0.02
    assert abs(float(x.std()) - 1.0) < 0.02


def test_fingerprint_is_stable_and_coordinate_sensitive():
    assert noise_fingerprint(806, 806_000, H, D) == noise_fingerprint(806, 806_000, H, D)
    assert noise_fingerprint(806, 806_000, H, D) != noise_fingerprint(806, 806_001, H, D)


def test_salt_is_the_documented_campaign_knob():
    """Pins the salt so an accidental edit is a failing test, not a silently re-rolled campaign."""
    assert NOISE_SALT == 0x41584953
