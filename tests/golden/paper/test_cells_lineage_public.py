"""What the published cell map says about lineage: every cell names how it was trained, every
cell names its published checkpoint, and the seeds pair as documented."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CELLS = REPO / "experiments" / "paper" / "cells.json"
ARMS = REPO / "experiments" / "paper" / "runs"     # index.json's `file` is relative to this directory
PATH_OPTION = re.compile(r"^--(?:data\.[\w-]+|weight-loader\.params-path)$")   # options whose value is a path


def _cells() -> list[dict]:
    return json.loads(CELLS.read_bytes())["cells"]


def test_each_cell_names_its_published_checkpoint():
    """`checkpoint` is the published checkpoint: its id, where it is published, and its published
    file set {bytes, n_files, sha256}."""
    for c in _cells():
        ck = c["checkpoint"]
        assert set(ck) == {"id", "ref", "pub_digest"}, c["cell"]
        assert ck["ref"].startswith(f"axisrobotics/qvla-v1-{c['backbone']}@v1.0:"), c["cell"]
        assert set(ck["pub_digest"]) == {"bytes", "n_files", "sha256"}, c["cell"]


def test_lineage_names_the_initial_checkpoint_and_no_machine():
    for c in _cells():
        lin, seed = c["lineage"], c["training_seed"]
        assert (lin["init"] is None) == (seed["pretrain"] is None), c["cell"]
        if lin["init"]:
            assert lin["init"]["ref"].startswith(f"axisrobotics/qvla-v1-{c['backbone']}@v1.0:pretrain/"), c["cell"]
            assert lin["init"]["training_seed"] == seed["pretrain"], c["cell"]
        for o in lin["train_overrides"] + ((lin["init"] or {}).get("train_overrides") or []):
            assert not re.search(r"(?<![\w/])s[23]:", o), (c["cell"], o)
            key, _, value = o.partition("=")
            if PATH_OPTION.match(key) or "/" in value:      # a path: under the training data root
                assert value.startswith("libero_5k_v2/"), (c["cell"], o)


def test_seed_pairing_is_as_documented():
    by = {c["cell"]: c["training_seed"] for c in _cells()}
    pi = {k: v for k, v in by.items() if k.endswith(("43|bare", "43|cond", "44|bare", "44|cond"))
          and not k.startswith(("svla", "xvla"))}
    assert by["bc43|bare"] == {"pretrain": 43, "ft": 43}
    assert all(v["ft"] == 42 for k, v in pi.items() if k != "bc43|bare")
    lerobot = [c for c in _cells() if c["backbone"] in ("smolvla", "xvla") and c["training_seed"]["pretrain"] is not None]
    assert all(c["training_seed"]["pretrain"] == c["training_seed"]["ft"] for c in lerobot)
    assert sum(c["training_seed"]["pretrain"] is None for c in _cells()) == 3


def test_every_cell_names_the_arm_that_trained_it():
    index = json.loads((ARMS / "index.json").read_bytes())["arms"]
    for c in _cells():
        arm = c["lineage"]["arm"]
        assert index[arm]["checkpoint"] == c["checkpoint"]["id"], c["cell"]
        assert (ARMS / index[arm]["file"]).is_file(), arm


def test_the_seeds_are_the_ones_the_arm_index_pairs():
    """One pairing, recorded once: each cell's {pretrain, ft} is its arm's (pretrain_seed, seed), and
    every replicate group of the arm index pairs its cells on that pretrain seed."""
    doc = json.loads((ARMS / "index.json").read_bytes())
    by = {c["cell"]: c for c in _cells()}
    for c in by.values():
        a = doc["arms"][c["lineage"]["arm"]]
        assert c["cell"] in a["cells"], c["cell"]
        assert c["training_seed"] == {"pretrain": a["pretrain_seed"], "ft": a["seed"]}, c["cell"]
    for group in doc["replicates"]:
        for m in group["members"]:
            for cell in m["cells"]:
                assert by[cell]["lineage"]["arm"] == m["arm"], cell
                assert by[cell]["training_seed"] == {"pretrain": m["pretrain_seed"], "ft": m["ft_seed"]}, cell


def test_each_init_is_the_checkpoint_its_arm_starts_from():
    """The init a cell's lineage names is, by ref and by pub_digest, the `init` its paper ArmSpec
    starts from (`<repo>:<path>#sha256:<pub_digest>` in the arm index); a cell with no init is
    an arm that starts from base weights. A pub_digest changed in one file and not the other
    fails here."""
    index = json.loads((ARMS / "index.json").read_bytes())["arms"]
    for c in _cells():
        init, want = c["lineage"]["init"], index[c["lineage"]["arm"]]["init"]
        if init is None:
            assert want.startswith("base:"), (c["cell"], want)
            continue
        ref, sep, sha = want.partition("#sha256:")
        assert sep and ref == init["ref"].replace("@v1.0:", ":", 1), (c["cell"], want, init["ref"])
        assert sha == init["pub_digest"], c["cell"]
