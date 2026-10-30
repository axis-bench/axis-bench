"""Driver logic that is testable without a GPU: unit construction, tally rebuild, fingerprint,
the serving protocol, and the benchmark identity stamp.

The GPU path is gate C/D's job. What is checked here are the places where a scheduling or
protocol change can silently corrupt a REPORTED NUMBER rather than crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_qvla.bench import spec as bench_spec
from axis_qvla.bench.sim import campaign as drv
from tests.unit.bench.sim.driver_stubs import (FakeProc, SubprocessShim, driver_stack, make_checkpoint,
                                               stub_probes)

BENCHMARK_MANIFEST = bench_spec.CONF_DIR / "franka_heldout24.json"


def _seeds(tid: int):
    return [tid * 1000 + i for i in range(50)]


def test_units_are_canonical_and_complete():
    units = drv.build_units([806, 1889], 50, _seeds)
    assert len(units) == 100
    assert units[0].task_id == 806 and units[0].seed == 806_000
    assert units[50].task_id == 1889
    assert len({(u.task_id, u.seed) for u in units}) == 100


def test_a_task_with_too_few_seeds_is_refused():
    """CATCHES an arm silently scored over fewer rollouts than its peers -- the denominator
    shrinks for one task and the aggregate moves without any error."""
    with pytest.raises(SystemExit, match="declares 10 test seeds, need 50"):
        drv.build_units([806], 50, lambda tid: _seeds(tid)[:10])


def test_a_task_missing_from_the_manifest_names_itself():
    """A bare StopIteration from _task_meta is unreadable and arrives half-way through building
    the queue; the driver must say which task."""
    def boom(tid):
        raise StopIteration

    with pytest.raises(SystemExit, match="task 813 is not in the layout/seed manifest"):
        drv.build_units([813], 50, boom)


def _write_seed_files(root: Path, arm: str, tid: int, n: int, n_success: int,
                      reason: str = "success") -> None:
    # The REAL layout, as on disk: one dir per arm (`<arm>_dualsim`), task subdirs inside.
    d = root / f"{arm}_dualsim" / f"task_{tid}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"seed_{tid * 1000 + i}.json").write_text(
            # Mirror SeedResult EXACTLY (seed/success/n_steps/sim_time/reason); a helper that
            # invents field names tests the helper, not the reader.
            json.dumps({"seed": tid * 1000 + i, "success": i < n_success,
                        "n_steps": 100 + i, "sim_time": 1.0 * i, "reason": reason}),
            encoding="utf-8",
        )


def test_fingerprint_is_stable_and_content_sensitive(tmp_path):
    _write_seed_files(tmp_path, "arm", 806, 5, 3)
    a = drv.fingerprint(tmp_path, "arm_dualsim", [806])
    assert a == drv.fingerprint(tmp_path, "arm_dualsim", [806])
    _write_seed_files(tmp_path, "arm", 806, 5, 4)  # one outcome flips
    assert drv.fingerprint(tmp_path, "arm_dualsim", [806]) != a


def test_fingerprint_does_not_depend_on_task_argument_order(tmp_path):
    """Two runs that list tasks differently must still fingerprint identically, or the hash
    would report a difference that does not exist in the results."""
    _write_seed_files(tmp_path, "arm", 806, 3, 1)
    _write_seed_files(tmp_path, "arm", 1889, 3, 2)
    assert drv.fingerprint(tmp_path, "arm_dualsim", [806, 1889]) == drv.fingerprint(
        tmp_path, "arm_dualsim", [1889, 806]
    )


def test_empty_results_do_not_silently_fingerprint_as_equal(tmp_path):
    """CATCHES a run that produced nothing hashing the same as another empty run and being
    mistaken for a reproducible match."""
    empty = drv.fingerprint(tmp_path, "arm_dualsim", [806])
    _write_seed_files(tmp_path, "arm", 806, 1, 1)
    assert drv.fingerprint(tmp_path, "arm_dualsim", [806]) != empty


def test_an_all_error_arm_is_refused_not_reported_as_zero_percent(tmp_path):
    """A scene-build crash writes a seed file with reason='error:...' and the process still exits
    0; an arm whose rollouts all errored is refused, never reported as a 0% arm."""
    _write_seed_files(tmp_path, "arm", 806, 5, 0, reason="error:build_render:FileNotFoundError")
    meta = drv.recompute_meta(tmp_path, "arm_dualsim", [806], 5)
    assert meta["errored"] == 5
    assert meta["incomplete"] == []          # complete, and still must not be reported


def test_a_healthy_arm_reports_no_errors(tmp_path):
    _write_seed_files(tmp_path, "arm", 806, 5, 3)
    meta = drv.recompute_meta(tmp_path, "arm_dualsim", [806], 5)
    assert meta["errored"] == 0
    assert meta["per_task"][806] == {"n_success": 3, "n_seeds": 5}


class _DeadProc:
    returncode = 1

    def poll(self):
        return 1


class _LiveProc:
    returncode = None

    def poll(self):
        return None


def _closed_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # released on close; nothing listens on it


def test_wait_ready_fails_fast_when_the_server_has_exited():
    """CATCHES the 57-minute hang: openpi's client constructor retries forever on a closed port,
    so the fail-fast check must run BEFORE it is entered, every iteration."""
    import time
    import openpi_client.websocket_client_policy  # noqa: F401 - warm the import; it is not what is timed
    t0 = time.time()
    with pytest.raises(SystemExit, match="exited rc=1"):
        drv.wait_ready("127.0.0.1", _closed_port(), timeout_s=30.0, require_noise=True,
                       proc=_DeadProc())
    assert time.time() - t0 < 5.0


def test_wait_ready_honours_its_timeout_on_a_closed_port():
    """A live-but-silent server must not block past `timeout_s` either."""
    import time
    t0 = time.time()
    with pytest.raises(SystemExit, match="not ready after"):
        drv.wait_ready("127.0.0.1", _closed_port(), timeout_s=1.0, require_noise=True,
                       proc=_LiveProc())
    assert time.time() - t0 < 8.0


ALL24 = [806, 807, 808, 809, 855, 878, 882, 888, 890, 945, 953, 966, 1050, 1223, 1231, 1426,
         1435, 1436, 1438, 1459, 1525, 1527, 1632, 1889]


def test_tracked_benchmark_manifest_is_the_24_task_set():
    """The tracked scored list: 24 unique ids, exactly the set every paper table scores, 50 seeds."""
    doc = json.loads(BENCHMARK_MANIFEST.read_text())
    assert doc["name"] == "franka_heldout24"
    assert sorted(doc["task_ids"]) == ALL24 and len(set(doc["task_ids"])) == 24
    assert doc["n_seeds"] == 50
    assert set(doc["subsets"]["usable_12"]) < set(ALL24)
    assert not set(int(k) for k in doc["excluded"]) & set(ALL24)


def test_benchmark_identity_is_the_declarations_identity_plus_a_match_flag():
    """The stamp IS axis_qvla.bench.spec's identity (three per-file digests), so an arm_summary and
    the benchmark check agree on what 'this benchmark' means; the match flag makes a custom
    --tasks list visible instead of silent."""
    ident = drv.benchmark_identity(list(reversed(ALL24)))
    assert ident["matches_manifest"] and ident["name"] == "franka_heldout24"
    expect = bench_spec.load_benchmark().identity()
    assert {k: ident[k] for k in expect} == expect
    assert len(ident["sha256"]) == len(ident["tasks_sha256"]) == len(ident["layouts_sha256"]) == 64
    assert ident["manifest"] == "conf/benchmarks/axis-heldout-v1/franka_heldout24.json"
    custom = drv.benchmark_identity(ALL24[:20])
    assert not custom["matches_manifest"] and custom["task_ids"] == ALL24[:20]
    missing = drv.benchmark_identity(ALL24, "no_such_benchmark")
    assert not missing["matches_manifest"] and "error" in missing


def test_tasks_accepts_a_list_a_json_list_and_the_manifest(tmp_path):
    assert drv.load_task_ids("806, 1889,") == [806, 1889]
    p = tmp_path / "l.json"
    p.write_text("[806, 1889]")
    assert drv.load_task_ids(str(p)) == [806, 1889]
    assert drv.load_task_ids(str(BENCHMARK_MANIFEST)) == ALL24


# --- serving protocol ---------------------------------------------------------------------------

LEGACY_SERVE = "--quality-tag 5 --guidance-scale 0"
LEGACY_WORKER = "--cfg-conditioning --guidance-scale 0"
SMOLVLA_CMD = ("{server_python} axis_qvla/bench/servers/lerobot_server.py --policy-type smolvla "
               "--ckpt {ckpt} --host 127.0.0.1 --port {port}")
XVLA_CMD = ("{server_python} axis_qvla/bench/servers/xvla.py --policy-type xvla "
            "--ckpt {ckpt} --host 127.0.0.1 --port {port} --domain-id 20")


def test_serving_derives_exactly_the_flags_the_campaign_hand_typed():
    """The 30 per-arm flag strings in the two paper specs collapse to one word; the derived
    strings must be byte-equal to what those specs carried, or the served protocol changes
    under the same arm name."""
    bare = drv.serving_protocol("bare")
    assert (bare.serve_flags, bare.worker_flags, bare.server_cmd) == ("", "", None)
    cond = drv.serving_protocol("conditioned")
    assert (cond.serve_flags, cond.worker_flags, cond.server_cmd) == (LEGACY_SERVE, LEGACY_WORKER, None)
    sb = drv.serving_protocol("bare", "smolvla")
    assert (sb.serve_flags, sb.worker_flags, sb.server_cmd) == ("", "", SMOLVLA_CMD)
    sc = drv.serving_protocol("conditioned", "smolvla")
    assert sc.server_cmd == SMOLVLA_CMD + " --quality-tag 5" and sc.worker_flags == LEGACY_WORKER
    g = drv.serving_protocol("conditioned", "openpi", 1.5)
    assert g.serve_flags == "--quality-tag 5 --guidance-scale 1.5"
    assert drv.result_dir_name("x", g.worker_flags) == "x_w1.5_dualsim"
    # X-VLA: the launchers' server command verbatim (domain id 20 is part of the protocol).
    xb = drv.serving_protocol("bare", "xvla")
    assert (xb.serve_flags, xb.worker_flags, xb.server_cmd) == ("", "", XVLA_CMD)
    xc = drv.serving_protocol("conditioned", "xvla")
    assert xc.server_cmd == XVLA_CMD + " --quality-tag 5" and xc.worker_flags == LEGACY_WORKER


def test_serving_refuses_a_guidance_scale_that_cannot_take_effect():
    with pytest.raises(ValueError, match="requires serving=conditioned"):
        drv.serving_protocol("bare", guidance_scale=1.5)
    with pytest.raises(ValueError, match="SmolVLA.*cannot apply guidance"):
        drv.serving_protocol("conditioned", "smolvla", 1.5)
    with pytest.raises(ValueError, match="serving must be one of"):
        drv.serving_protocol("tagged")


def _ns(**kw):
    import types
    return types.SimpleNamespace(**{**dict(serving=None, backend="openpi", guidance_scale=0.0), **kw})


def test_the_protocol_is_serving_alone_and_defaults_to_bare():
    assert drv.resolve_protocol(_ns()) == drv.serving_protocol("bare")
    assert drv.resolve_protocol(_ns(backend="smolvla")) == drv.serving_protocol("bare", "smolvla")
    assert drv.resolve_protocol(_ns(serving="conditioned", backend="xvla")) == drv.serving_protocol(
        "conditioned", "xvla")
    with pytest.raises(SystemExit, match="requires serving=conditioned"):
        drv.resolve_protocol(_ns(guidance_scale=1.5))


# --- the layout file the workers read -----------------------------------------------------------

def test_tracked_layouts_file_is_reshaped_for_the_layout_generator(tmp_path):
    """The tracked <name>.layouts.json keys `tasks` by id; axis_qvla.bench.sim.layouts iterates a
    LIST with `task_id`. Handing the dict straight to the harness dies in build_scene on every
    rollout. The adapted copy must be what the generator's lookup accepts, and the same content."""
    from axis_qvla.bench.sim.layout_plan import _is_single_variant

    b = bench_spec.load_benchmark()
    out = drv.harness_layout_manifest(b.layouts_path, tmp_path)
    assert out == tmp_path / "layout_manifest.harness.json"
    doc = json.loads(out.read_text())
    assert isinstance(doc["tasks"], list) and doc["adapted_from"] == str(b.layouts_path)
    src = json.loads(b.layouts_path.read_text())["tasks"]
    assert [int(t["task_id"]) for t in doc["tasks"]] == sorted(int(k) for k in src)
    entry = next(t for t in doc["tasks"] if int(t["task_id"]) == 1459)
    assert entry["train"] == src["1459"]["train"] and entry["layout_box_m"] == src["1459"]["layout_box_m"]
    # The generator's own manifest reader accepts it (a dict-shaped file raises here).
    assert isinstance(_is_single_variant(1459, out), bool)
    with pytest.raises(AttributeError):
        _is_single_variant(1459, b.layouts_path)


def test_the_single_task_cli_applies_the_same_adaptation_before_planning(tmp_path):
    """axis_qvla.bench.sim.cli's own module docstring presents a direct invocation --
    `--task-id 1889 --variant vanilla ... --dual-sim`, no `--layout-manifest` -- as "the one
    entrypoint for served-policy rollout evaluation". Against the default benchmark that
    resolves to the SAME dict-keyed tracked file as the queue worker's own default (the test
    above), so without the queue worker's adaptation it hits the exact AttributeError pinned
    there, for a direct caller instead of the worker. Proves the docstring's own argument
    parsing and the resulting layout resolution end to end -- no render, no server, no
    `--layout-manifest` override, nothing stubbed -- using axis_qvla.bench.sim.layout_plan.
    adapt_layout_manifest (what axis_qvla.bench.sim.cli.main calls) and a synthetic task_data_root
    (task 1459's one training variant is index 0, so a single scene_variants entry suffices)."""
    from axis_qvla.bench.sim import cli
    from axis_qvla.bench.sim.layout_plan import adapt_layout_manifest, plan_layouts

    args = cli.build_parser().parse_args([
        "--task-id", "1459", "--variant", "vanilla", "--host", "127.0.0.1", "--port", "8000",
        "--checkpoint-dir", "unused", "--log-root", "logs", "--work-dir", "work",
        "--task-data-root", str(tmp_path / "bench"),
    ])
    assert args.layout_manifest is None   # the docstring's own invocation never passes it

    source = cli.resolve_task_source(args)
    assert source.layouts_default == bench_spec.load_benchmark().layouts_path

    task_dir = tmp_path / "bench" / "task_1459"
    task_dir.mkdir(parents=True)
    task_dir.joinpath("task.json").write_text(json.dumps({"scene_variants": [
        {"initial_state": {"objects": {
            "obj_a": {"pos": [0.40, -0.20, 0.80]}, "obj_b": {"pos": [0.60, 0.10, 0.80]}}}},
    ]}))

    layout_manifest = adapt_layout_manifest(args.layout_manifest or source.layouts_default, tmp_path / "harness")
    layouts = plan_layouts(args.task_id, [1459000], manifest=layout_manifest,
                           task_data_root=tmp_path / "bench")
    assert set(layouts) == {1459000}
    assert layouts[1459000].base_variant_index == 0
    assert set(layouts[1459000].object_xy) == {"obj_a", "obj_b"}


def test_a_list_shaped_layout_manifest_passes_through_untouched(tmp_path):
    """A list-shaped layout manifest must reach the workers at ITS path, so run_manifest's digest
    names the file the campaign actually used."""
    p = tmp_path / "heldout_eval_manifest.json"
    p.write_text(json.dumps({"tasks": [{"task_id": 806, "train": []}]}))
    assert drv.harness_layout_manifest(p, tmp_path / "out") == p
    assert not (tmp_path / "out").exists()


def test_dry_run_records_serving_and_benchmark_identity(tmp_path):
    """End to end through main() with no GPU: --dry-run writes dry_run_manifest.json (the same
    builder as run_manifest.json) carrying the derived protocol and the declaration's identity,
    and adapts the tracked layouts file."""
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x")
    b = bench_spec.load_benchmark()
    rc = drv.main([
        "--run", "probe", "--config", "cfg", "--checkpoint", str(ck), "--tasks", "806,1889",
        "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
        "--layout-manifest", str(b.layouts_path), "--serving", "conditioned",
        "--openpi-dir", str(tmp_path), "--n-seeds", "3", "--dry-run",
    ])
    assert rc == 0
    assert not (tmp_path / "r1" / "out_probe" / "run_manifest.json").exists()
    m = json.loads((tmp_path / "r1" / "out_probe" / "dry_run_manifest.json").read_text())
    assert m["serving"]["serving"] == "conditioned" and m["serving"]["backend"] == "openpi"
    assert m["serving"]["serve_flags"] == LEGACY_SERVE
    assert m["serving"]["worker_flags"].endswith(LEGACY_WORKER)
    assert m["benchmark"]["name"] == "franka_heldout24" and not m["benchmark"]["matches_manifest"]
    assert m["benchmark"]["sha256"] == b.sha256
    assert m["layout_manifest"]["declared"]["sha256"] == b.layouts_sha256
    assert m["layout_manifest"]["path"].endswith("layout_manifest.harness.json")
    assert m["n_units"] == 6
    assert (tmp_path / "r1" / "q_probe").exists()


def test_checkpoint_digest_is_by_content(tmp_path):
    ck = tmp_path / "13579"
    (ck / "params").mkdir(parents=True)
    (ck / "params" / "a.bin").write_bytes(b"weights-v1")
    (ck / "config.json").write_text("{}")
    d1 = drv.digest_tree(ck)
    assert d1["n_files"] == 2 and d1["bytes"] == 12 and len(d1["sha256"]) == 64
    assert drv.digest_tree(ck)["sha256"] == d1["sha256"]
    (ck / "params" / "a.bin").write_bytes(b"weights-v2")  # same size, same names
    assert drv.digest_tree(ck)["sha256"] != d1["sha256"]


def test_run_manifest_names_model_code_protocol_and_seeds(tmp_path, monkeypatch):
    """CATCHES the audit's identity-by-label finding: the manifest must carry a content digest
    of the checkpoint, the serving protocol, the noise salt, the exact unit list and the source
    state -- not just a --run string and a path."""
    import types
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "w.bin").write_bytes(b"x")
    lm = tmp_path / "layouts.json"
    lm.write_text("{}")
    a = types.SimpleNamespace(
        arm="r1x", config="cfg", checkpoint=str(ck), serve_flags="--quality-tag 5",
        worker_flags=f"--layout-manifest {lm} --cfg-conditioning --guidance-scale 0",
        server_cmd=None, mem_fraction="0.5", workers_per_gpu=4, batch=10, stale_after_s=1800.0,
        openpi_dir=str(tmp_path), worker_python="python3",
    )
    asked: list = []

    def fake_git_state(d):
        asked.append(Path(d))
        return ({"path": str(d), "head": "0" * 40, "dirty": False} if Path(d) == drv.REPO_ROOT
                else {"path": str(d), "error": "not a git repository"})

    monkeypatch.setattr(drv, "git_state", fake_git_state)   # a source archive has no .git
    units = drv.build_units([806], 3, _seeds)
    m = drv.run_manifest(a, {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}, units, probe=False)
    assert m["checkpoint"]["sha256"] == drv.digest_tree(ck)["sha256"]
    assert m["serving"]["worker_flags"].endswith("--cfg-conditioning --guidance-scale 0")
    assert m["noise_salt"] == 0x41584953
    assert m["layout_manifest"]["sha256"] == drv.sha256_file(lm)
    assert m["n_units"] == 3 and len(m["units_sha256"]) == 64
    assert m["env_pins"]["XLA_FLAGS"] == "--xla_gpu_autotune_level=0"
    assert m["sources"]["repo"]["head"] == "0" * 40 and "error" in m["sources"]["openpi"]
    assert asked == [drv.REPO_ROOT, tmp_path]               # the driver's checkout, then the server's
    assert m["sources"]["files"]["axis_qvla/bench/sim/rollout_dualsim.py"] is not None


def test_result_dir_follows_the_rollout_scripts_naming():
    """CATCHES the mis-tally: a conditioned arm writes <arm>_w<g>_dualsim, not <arm>_dualsim."""
    assert drv.result_dir_name("r1bc_r2", "--layout-manifest /m.json") == "r1bc_r2_dualsim"
    assert drv.result_dir_name("r1scfgnp_c", "--layout-manifest /m.json --cfg-conditioning --guidance-scale 0") == "r1scfgnp_c_w0_dualsim"
    assert drv.result_dir_name("x", "--cfg-conditioning --guidance-scale 1.5") == "x_w1.5_dualsim"


def test_tally_only_is_a_declared_mode():
    ap = drv.build_parser() if hasattr(drv, "build_parser") else None
    import inspect
    assert "--tally-only" in inspect.getsource(drv.main)


# --- one log root, the benchmark and n_seeds, through all three argv layers ----------------------
#
# The shared stubs (driver_stubs.py) run all three processes for real except what needs a GPU.

def _flag(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, t in enumerate(argv) if t == flag]


def test_a_full_run_tallies_exactly_the_files_its_rollouts_wrote(tmp_path, monkeypatch):
    """The host's log root is not the checkout on the workers' PYTHONPATH (the public layout,
    and a worker box with two checkouts). The driver must hand its ONE log root to every worker and on to every
    rollout, so the tally reads what was written."""
    seen = driver_stack(monkeypatch, tmp_path)
    host_logs = tmp_path / "host_bench" / "eval_results" / "slb_pi05" / "rollout_logs"
    rc = drv.main([
        "--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)),
        "--tasks", "806,1889", "--run-root", str(tmp_path / "r1"), "--log-root", str(host_logs),
        "--openpi-dir", str(tmp_path), "--n-seeds", "3", "--gpus", "0,1", "--workers-per-gpu", "1",
        "--task-data-root", str(tmp_path / "tasks"),
    ])
    assert seen["workers"] and all(_flag(w, "--log-root") == [str(host_logs)] for w in seen["workers"])
    assert seen["rollouts"] and {ns.log_root for ns in seen["rollouts"]} == {host_logs}
    summary = json.loads((tmp_path / "r1" / "out_probe" / "arm_summary.json").read_text())
    assert rc == 0 and (summary["rollouts"], summary["successes"]) == (6, 4)
    assert summary["log_root"] == str(host_logs)
    assert summary["eval_fingerprint"] == drv.fingerprint(host_logs, "probe_dualsim", [806, 1889])


def test_a_relative_log_root_is_made_absolute_once_for_writer_and_reader(tmp_path, monkeypatch):
    seen = driver_stack(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = drv.main([
        "--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)), "--tasks", "806",
        "--run-root", "r1", "--log-root", "logs", "--openpi-dir", str(tmp_path), "--n-seeds", "2",
        "--gpus", "0", "--workers-per-gpu", "1", "--task-data-root", "tasks",
    ])
    assert rc == 0
    assert {ns.log_root for ns in seen["rollouts"]} == {tmp_path / "logs"}


def _pilot_benchmark(tmp_path: Path, monkeypatch, n_seeds: int) -> list[int]:
    """A declaration that is NOT the default one: other seeds, and more of them than 50."""
    conf = tmp_path / "conf_benchmarks"
    conf.mkdir()
    tids = [806, 1889]
    tasks = {str(t): {"task_name_canonical": f"task {t}",
                      "test_seeds": [t * 1000 + 500 + i for i in range(n_seeds)],
                      "gripper": {"open_w": 0.08, "closed_w": 0.002}} for t in tids}
    (conf / "pilot.json").write_text(json.dumps({
        "name": "pilot", "version": "0", "task_ids": tids, "n_seeds": n_seeds,
        "tasks_file": "pilot.tasks.json", "layouts_file": "pilot.layouts.json",
        "action_space": "droid8", "image_geometry": "raw"}))
    (conf / "pilot.tasks.json").write_text(json.dumps({"tasks": tasks}))
    (conf / "pilot.layouts.json").write_text(json.dumps({"tasks": {str(t): {} for t in tids}}))
    monkeypatch.setattr(bench_spec, "CONF_DIR", conf)
    # The identity stamp names the declaration relative to the repo; this one lives in tmp_path.
    monkeypatch.setattr(drv, "benchmark_identity",
                        lambda ids, name=None: {"name": name, "matches_manifest": True})
    return tids


def test_the_benchmark_and_its_n_seeds_reach_every_worker_and_rollout(tmp_path, monkeypatch):
    """The driver resolved seeds from --benchmark but never forwarded it, and the worker
    hardcoded --n-seeds 50, so every rollout re-resolved prompt and seeds from the DEFAULT
    declaration and refused seeds past the 50th. n_seeds now comes from the declaration."""
    seen = driver_stack(monkeypatch, tmp_path)
    tids = _pilot_benchmark(tmp_path, monkeypatch, n_seeds=60)
    logs = tmp_path / "logs"
    rc = drv.main([
        "--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)),
        "--tasks", ",".join(map(str, tids)), "--benchmark", "pilot",
        "--run-root", str(tmp_path / "r1"), "--log-root", str(logs),
        "--openpi-dir", str(tmp_path), "--gpus", "0", "--workers-per-gpu", "1",
        "--task-data-root", str(tmp_path / "tasks"),
    ])
    assert all(_flag(w, "--benchmark") == ["pilot"] and _flag(w, "--n-seeds") == ["60"]
               for w in seen["workers"])
    assert {(ns.benchmark, ns.n_seeds) for ns in seen["rollouts"]} == {("pilot", 60)}
    summary = json.loads((tmp_path / "r1" / "out_probe" / "arm_summary.json").read_text())
    assert rc == 0 and summary["n_seeds"] == 60 and summary["rollouts"] == 120
    assert not summary["incomplete_tasks"]


# --- provenance records ---------------------------------------------------------------------

def _manifest_args(tmp_path: Path, **kw):
    import types

    base = dict(arm="r1x", config="cfg", checkpoint=str(make_checkpoint(tmp_path)), serve_flags="",
                worker_flags="", server_cmd=None, mem_fraction="0.5", workers_per_gpu=4, batch=10,
                stale_after_s=1800.0, openpi_dir=str(tmp_path), worker_python="/venvs/render/bin/python",
                backend="openpi")
    return types.SimpleNamespace(**{**base, **kw})


def _started_server(tmp_path: Path, monkeypatch, a, env: dict) -> dict:
    """argv and working directory of the server start_server would launch, captured, nothing started."""
    got: dict = {}

    def popen(cmd, **kw):
        got.update(cmd=list(cmd), cwd=kw.get("cwd"))
        return FakeProc()

    with monkeypatch.context() as mp:
        mp.setattr(drv, "subprocess", SubprocessShim(Popen=popen))
        drv.start_server(0, 34000, a.config, a.checkpoint, a.serve_flags.split(), tmp_path / "srv",
                         env, a.mem_fraction, a.server_cmd)
    return got


@pytest.mark.parametrize("backend", ["smolvla", "xvla"])
def test_the_server_probe_runs_the_interpreter_the_lerobot_server_starts_with(tmp_path, monkeypatch, backend):
    """The probe took `server_cmd.split()[0]`, which for the LeRobot backends is the
    unformatted '{server_python}', so versions.server was an error record on every such run."""
    seen: dict = {}
    stub_probes(monkeypatch, seen)
    a = _manifest_args(tmp_path, server_cmd=drv.serving_protocol("bare", backend).server_cmd, backend=backend)
    env = {"SERVER_PYTHON": "/venvs/lerobot/bin/python"}
    m = drv.run_manifest(a, env, drv.build_units([806], 1, _seeds), probe=True)
    assert "versions" in m
    assert seen["probed"] == ["/venvs/render/bin/python", "/venvs/lerobot/bin/python"]
    assert seen["probed"][1] == _started_server(tmp_path, monkeypatch, a, env)["cmd"][0]


@pytest.mark.parametrize("server_python", [None, "/venvs/openpi/bin/python"])
def test_the_openpi_server_probe_matches_the_interpreter_start_server_uses(tmp_path, monkeypatch, server_python):
    """Without --server-python the server runs under the driver's own interpreter, not the
    worker's; the probe must name the one that actually runs."""
    seen: dict = {}
    stub_probes(monkeypatch, seen)
    a = _manifest_args(tmp_path)
    env = {} if server_python is None else {"SERVER_PYTHON": server_python}
    drv.run_manifest(a, env, drv.build_units([806], 1, _seeds), probe=True)
    assert seen["probed"][1] == _started_server(tmp_path, monkeypatch, a, env)["cmd"][0]


# Every module on the rollout path. The list is checked, not used: the manifest computes its own.
ROLLOUT_PATH = ("campaign", "queue_worker", "workqueue", "inference_noise", "cli", "rollout_dualsim",
                "rollout_render", "render_scene", "layouts", "layout_plan", "domain", "policy",
                "runner", "record", "replan", "arms", "assets", "asset_guard", "checker",
                "task_data", "obs_cameras", "obs_geometry", "action_mapping")


@pytest.mark.parametrize("backend", ["openpi", "smolvla", "xvla"])
def test_run_manifest_digests_every_module_on_the_rollout_path(tmp_path, monkeypatch, backend):
    """KEY_SOURCES digested 7 files and none of the render half, so two runs on different
    render code had identical manifests. `sources.files` is the driver checkout's client closure,
    the same for every backend; no server script is in it, because no server runs from here."""
    stub_probes(monkeypatch)
    m = drv.run_manifest(_manifest_args(tmp_path, backend=backend), {"OPENPI_DIR": str(tmp_path)},
                         drv.build_units([806], 1, _seeds), probe=False)
    files = m["sources"]["files"]
    pkg = drv.__name__.split(".")[0]
    for mod in ROLLOUT_PATH:
        rel = f"{pkg}/bench/sim/{mod}.py"
        assert files.get(rel) == drv.sha256_file(drv.REPO_ROOT / rel), rel
    assert files.get(f"{pkg}/bench/spec.py") and files.get(f"{pkg}/bench/sim/__init__.py")
    assert not any(k.startswith(("scripts/", f"{pkg}/bench/servers/")) for k in files)


@pytest.mark.parametrize("backend", ["openpi", "smolvla", "xvla"])
def test_the_server_script_is_digested_where_the_server_runs_it(tmp_path, monkeypatch, backend):
    """start_server launches the template's RELATIVE script with
    cwd=OPENPI_DIR, and `bench eval` points OPENPI_DIR at the LeRobot checkout (a different,
    stale checkout from the driver's). The record digested the DRIVER checkout's copy under the
    launched script's name -- a plausible digest of a file that did not run. It must digest the
    file the launched argv names, resolved against the cwd Popen is given, and name that root."""
    stub_probes(monkeypatch)
    server_root = tmp_path / "lerobot_checkout"
    a = _manifest_args(tmp_path, server_cmd=drv.serving_protocol("bare", backend).server_cmd,
                       backend=backend, openpi_dir=str(server_root))
    env = {"OPENPI_DIR": str(server_root), "SERVER_PYTHON": "/venvs/lerobot/bin/python"}
    launched = _started_server(tmp_path, monkeypatch, a, env)
    script = launched["cmd"][1]
    ran = Path(launched["cwd"]) / script
    ran.parent.mkdir(parents=True)
    ran.write_text("# the copy the server actually runs: not the driver checkout's\n")
    m = drv.run_manifest(a, env, drv.build_units([806], 1, _seeds), probe=False)
    server = m["sources"]["server"]
    assert server == {"root": str(server_root), "script": script, "sha256": drv.sha256_file(ran)}
    assert script not in m["sources"]["files"]
    if (drv.REPO_ROOT / script).exists():
        assert server["sha256"] != drv.sha256_file(drv.REPO_ROOT / script)
    # A server checkout without the script records no digest rather than borrowing one.
    ran.unlink()
    m = drv.run_manifest(a, env, drv.build_units([806], 1, _seeds), probe=False)
    assert m["sources"]["server"] == {"root": str(server_root), "script": script, "sha256": None}


def test_the_dry_run_of_an_xvla_arm_records_the_server_checkouts_script(tmp_path, monkeypatch):
    """End to end through main(): `--openpi-dir` names the checkout the server runs from, and the
    dry run's manifest digests the X-VLA server script THERE."""
    stub_probes(monkeypatch)
    server_root = tmp_path / "smolvla_bench"
    (server_root / "axis_qvla" / "bench" / "servers").mkdir(parents=True)
    ran = server_root / "axis_qvla" / "bench" / "servers" / "xvla.py"
    ran.write_text("# box copy\n")
    assert drv.main(["--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)),
                     "--tasks", "806", "--run-root", str(tmp_path / "r1"),
                     "--log-root", str(tmp_path / "logs"), "--serving", "bare", "--backend", "xvla",
                     "--openpi-dir", str(server_root), "--n-seeds", "1", "--gpus", "0",
                     "--dry-run"]) == 0
    m = json.loads((tmp_path / "r1" / "out_probe" / "dry_run_manifest.json").read_text())
    assert m["sources"]["server"]["sha256"] == drv.sha256_file(ran)
    assert m["sources"]["server"]["script"] == "axis_qvla/bench/servers/xvla.py"


def test_tally_only_never_overwrites_the_run_manifest(tmp_path, monkeypatch):
    """--tally-only wrote its own (tally-time) provenance over run_manifest.json, so the
    record of what produced the results was lost the first time anyone re-tallied them."""
    stub_probes(monkeypatch)
    out = tmp_path / "r1" / "out_probe"
    out.mkdir(parents=True)
    recorded = b'{"recorded": "by the run that wrote the seed files"}'
    (out / "run_manifest.json").write_bytes(recorded)
    _write_seed_files(tmp_path / "logs", "probe", 806, 2, 1)
    argv = ["--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)), "--tasks", "806",
            "--run-root", str(tmp_path / "r1"), "--log-root", str(tmp_path / "logs"),
            "--openpi-dir", str(tmp_path), "--n-seeds", "2", "--tally-only"]
    assert drv.main(argv) == 0
    assert (out / "run_manifest.json").read_bytes() == recorded
    tally = json.loads((out / "tally_manifest.json").read_text())
    summary = json.loads((out / "arm_summary.json").read_text())
    assert summary["provenance"] == json.loads(recorded)
    assert summary["tally_provenance"] == tally and tally["checkpoint"]["n_files"] == 1
    # An arm whose run left no manifest is summarised with none, never with the tally's.
    (out / "run_manifest.json").unlink()
    assert drv.main(argv) == 0
    assert not (out / "run_manifest.json").exists()
    assert json.loads((out / "arm_summary.json").read_text())["provenance"] is None


def test_a_dry_run_never_overwrites_the_run_manifest(tmp_path, monkeypatch):
    """A dry run starts no server, so it probes nothing; writing its record over run_manifest.json
    would replace the record of the run that wrote the seed files with a no-probe one, the same
    rule as --tally-only's."""
    stub_probes(monkeypatch)
    out = tmp_path / "r1" / "out_probe"
    out.mkdir(parents=True)
    recorded = b'{"recorded": "by the run that wrote the seed files"}'
    (out / "run_manifest.json").write_bytes(recorded)
    assert drv.main(["--run", "probe", "--config", "cfg", "--checkpoint", str(make_checkpoint(tmp_path)),
                     "--tasks", "806", "--run-root", str(tmp_path / "r1"),
                     "--log-root", str(tmp_path / "logs"), "--openpi-dir", str(tmp_path),
                     "--n-seeds", "1", "--gpus", "0", "--dry-run"]) == 0
    assert (out / "run_manifest.json").read_bytes() == recorded
    dry = json.loads((out / "dry_run_manifest.json").read_text())
    assert dry["n_units"] == 1 and "versions" not in dry


# A normal arm (two tasks, no diagnostics), fingerprinted by the driver at 6fe75e2, before the
# seed-file selection became exact. The eval fingerprint is an identity: it may not move.
NORMAL_ARM_FINGERPRINT = "d28733bce70fa98fa926e52232dbc6ca7e6a9e51e2ba56741620ac17f466de99"


def test_exact_seed_selection_keeps_a_normal_fingerprint_and_ignores_diagnostics(tmp_path):
    """Under --diagnostics the dual-sim backend writes seed_<n>.diag_objz.json, which the
    `seed_*.json` minus `*.diag.json` filter took for a result: it moved the fingerprint and added
    a phantom seed to the tally. Only seed_<int>.json is a result now; a normal arm's fingerprint
    is byte-for-byte what it was."""
    _write_seed_files(tmp_path, "arm", 806, 5, 3)
    _write_seed_files(tmp_path, "arm", 1889, 4, 1)
    assert drv.fingerprint(tmp_path, "arm_dualsim", [1889, 806]) == NORMAL_ARM_FINGERPRINT
    d = tmp_path / "arm_dualsim" / "task_806"
    for name in ("seed_806000.diag.json", "seed_806000.diag_objz.json", "seed_806000.mp4",
                 "seed_806000.diag_frame0.png", "seed_806000.diag_frame0_base.png", "meta.json",
                 ".seed_806001.json.tmp99"):
        (d / name).write_text('{"seed": 806000, "success": true}')
    assert drv.fingerprint(tmp_path, "arm_dualsim", [1889, 806]) == NORMAL_ARM_FINGERPRINT
    meta = drv.recompute_meta(tmp_path, "arm_dualsim", [806, 1889], 5)
    assert meta["per_task"][806] == {"n_success": 3, "n_seeds": 5}


def test_the_driver_help_and_run_manifest_name_every_record(capsys):
    """A real run writes run_manifest.json, --tally-only tally_manifest.json and
    --dry-run dry_run_manifest.json, but the run_manifest docstring said
    every invocation wrote run_manifest.json before the first server starts, and the help named none."""
    with pytest.raises(SystemExit):
        drv.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    doc = " ".join(drv.run_manifest.__doc__.split())
    for name in ("run_manifest.json", "tally_manifest.json", "dry_run_manifest.json"):
        assert name in out and name in doc, name
    assert "arm_summary.json" in out
