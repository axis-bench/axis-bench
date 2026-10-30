"""The committed cell map (published as qvla-v1-results:cells.json) is complete and clean: the
123 cells the paper reports, one read each."""
from __future__ import annotations

import json
import re
from pathlib import Path

from axis_qvla.bench.spec import load_benchmark
from axis_qvla.bench.sim.arms import served_arm_name
from axis_qvla.paper.cells import validate_cells
from axis_qvla.core.canon import dumps_canonical
from axis_qvla.bench.results.fingerprint import units_sha256

REPO = Path(__file__).resolve().parents[3]
CELLS = REPO / "experiments" / "paper" / "cells.json"
R1_DECLARATION_SHA256 = "fcff87cbf738fb15849f15adf91e06e8d3e88434c6aa0b0f8d19d6fe209eb037"
R1_UNITS_SHA256 = "b3f14d19894fb3e50e932329cfe98d991049c05387476313af9a9b24215388a9"
_ABS_PATH = re.compile(r"(?<![\w.:-])/(?:disk|home|localdisk|bigdata|rhome|mnt|data|tmp|root|opt)/")
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_DOMAIN = re.compile(r"\b[a-z0-9-]+\.(?:com|net|org|cn|io|edu)\b", re.I)
_USER_HOST = re.compile(r"[\w.-]+@(?!v1\.0:)[\w-]+\.[\w.-]+")


def _doc() -> dict:
    return json.loads(CELLS.read_bytes())


def test_valid_complete_and_in_pooled_order():
    doc = _doc()
    assert validate_cells(doc) == []
    runs = json.loads((REPO / "experiments" / "paper" / "cells_runs.json").read_text())["cells"]
    assert [(c["cell"], c["run"], c["box"]) for c in doc["cells"]] == [(r["cell"], r["run"], r["box"]) for r in runs]
    assert len(doc["cells"]) == 123 and len({c["run"] for c in doc["cells"]}) == 123
    assert (sum(c["box"] == "s2" for c in doc["cells"]), sum(c["box"] == "s3" for c in doc["cells"])) == (62, 61)


def test_bench_block_pins_the_r1_benchmark():
    b = load_benchmark()
    assert _doc()["bench"] == {"name": "franka_heldout24", "r1_declaration_sha256": R1_DECLARATION_SHA256,
                               "task_ids": list(b.task_ids), "n_seeds": 50, "units_sha256": R1_UNITS_SHA256}
    assert units_sha256(b.task_ids, 50) == R1_UNITS_SHA256


def test_served_names_follow_the_serving_protocol():
    for c in _doc()["cells"]:
        cond = bool(c["serve"]["fields"])
        assert cond == c["cell"].endswith("|cond"), c["cell"]
        assert c["served"] == served_arm_name(c["run"], cfg_conditioning=cond, guidance_scale=0.0, domain="dualsim")
        assert c["serve"]["beta"] == 1.0 and c["serve"]["uncond_fields"] is None
        assert f"--run {c['run']}" in c["eval_command"]
        assert f"--serving {'conditioned' if cond else 'bare'}" in c["eval_command"]


def test_one_checkpoint_one_digest():
    ids: dict[str, str] = {}
    for c in _doc()["cells"]:
        d = c["checkpoint"]["pub_digest"]["sha256"]
        assert ids.setdefault(d, c["checkpoint"]["id"]) == c["checkpoint"]["id"], c["cell"]


def test_no_machine_path_or_host_name():
    text = CELLS.read_text(encoding="utf-8")
    assert not _ABS_PATH.search(text)
    assert not _IPV4.search(text)
    assert not _DOMAIN.search(text)
    assert not _USER_HOST.search(text)


def test_committed_bytes_are_canonical():
    assert dumps_canonical(_doc()) == CELLS.read_bytes()


def test_every_eval_command_parses_with_the_qvla_parser_and_names_every_root():
    """Each cell's eval_command is the full public `qvla bench eval` argv -- it
    parses with the real parser, and names the host profile, the assets, the render root and the
    task-data root that a public checkout must pass (none has a usable default there)."""
    import shlex

    from axis_qvla.cli import build_parser

    backend = {"pi05": "openpi", "smolvla": "smolvla", "xvla": "xvla"}
    bench = "axisrobotics/qvla-v1-bench@v1.0:"
    for c in _doc()["cells"]:
        argv = shlex.split(c["eval_command"])
        assert argv[:3] == ["qvla", "bench", "eval"], c["cell"]
        ns = build_parser().parse_args(argv[1:])
        assert ns.func.__name__ == "_cmd_bench_eval" and ns.host, c["cell"]
        assert (ns.arm, ns.backend) == (c["run"], backend[c["backbone"]]), c["cell"]
        assert ns.serving == ("conditioned" if c["serve"]["fields"] else "bare"), c["cell"]
        assert c["checkpoint"]["ref"] in ns.checkpoint, c["cell"]
        assert str(ns.render_root) == "third_party/axis-render", c["cell"]
        assert f"{bench}assets" in str(ns.asset_dir) and f"{bench}tasks" in str(ns.task_data_root), c["cell"]
