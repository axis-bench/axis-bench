"""A GPU job running this suite sets CUDA_VISIBLE_DEVICES, which `qvla train` reads for its default
--fsdp-devices (axis_qvla.train.launch.default_fsdp_devices); no test may depend on the machine's GPUs."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_visible_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture(autouse=True)
def _no_base_weights_fetch(monkeypatch):
    """`qvla train` fetches the base files a run from a published checkpoint loads (axis_qvla.train.runs.
    fetch_serving); no test reaches a publisher. A test of the fetch replaces this with its own."""
    from axis_qvla.data import bases

    monkeypatch.setattr(bases, "fetch", lambda lock, names, dest, **kw: (0, 0))
