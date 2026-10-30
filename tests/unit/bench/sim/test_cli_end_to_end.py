"""Drive `axis_qvla.bench.sim.cli.main` all the way to written results, with stubbed backends.

A success count is not a gate: rollouts are nondeterministic (the policy RNG is one running
stream per server process, with no per-episode reset), and an equality gate against a number that
moves +-2 on repeat would either fail on a correct runner or pass on a broken one.

What is pinned here is that the CLI writes the artifacts the published runs carry: the arm
directory, the per-seed schema, and a meta.json carrying every provenance field those runs
recorded. That is the part a later reader depends on, and it is checkable without a GPU, a
simulator or a policy server.
"""
from __future__ import annotations

import json
import shutil
import sys
import types

import pytest

from axis_qvla.bench import spec as bench_spec
from axis_qvla.bench.sim import cli

# Exactly what the published runs' meta.json carries. Losing any one of these silently
# un-attributes a result set, which is the failure the runner's two-phase write exists to stop.
LEGACY_META_FIELDS = {
    "task_id", "variant", "arm", "cfg_conditioning", "guidance_scale", "server_metadata",
    "checkpoint_dir", "prompt", "open_w", "closed_w", "fps", "replan", "n_stable_frames",
    "max_steps", "checker", "eval_layouts", "domain", "layout_manifest", "layouts", "server",
    "git_sha",
}

# Exactly what it wrote per seed.
LEGACY_SEED_FIELDS = {
    "seed", "success", "n_steps", "sim_time", "reason", "guidance_scale", "cfg_conditioning",
    "eval_layouts", "base_variant_index",
}

SEEDS = [11, 12, 13]
PROMPT = "Rotate the Computer Mouse"


@pytest.fixture
def fixture_paths(tmp_path, monkeypatch):
    """A benchmark declaring one task (`pilot866`, beside a copy of the tracked declaration), plus
    somewhere to write."""
    conf = tmp_path / "conf_benchmarks"
    conf.mkdir()
    for name in ("franka_heldout24.json", "franka_heldout24.tasks.json", "franka_heldout24.layouts.json",
                 "execution.json"):
        shutil.copy(bench_spec.CONF_DIR / name, conf / name)
    (conf / "pilot866.json").write_text(json.dumps({
        "name": "pilot866", "version": "0", "task_ids": [866], "n_seeds": len(SEEDS),
        "tasks_file": "pilot866.tasks.json", "layouts_file": "pilot866.layouts.json",
        "action_space": "droid8", "image_geometry": "raw"}))
    (conf / "pilot866.tasks.json").write_text(json.dumps({"tasks": {"866": {
        "task_name_canonical": PROMPT, "test_seeds": SEEDS, "gripper": {"open_w": 0.08, "closed_w": 0.002}}}}))
    (conf / "pilot866.layouts.json").write_text(json.dumps({"tasks": {"866": {}}}))
    monkeypatch.setattr(bench_spec, "CONF_DIR", conf)
    # Also the --asset-dir of every run below, and the CLI refuses an asset dir that does not exist.
    (tmp_path / "data").mkdir()
    return {"logs": tmp_path / "logs", "work": tmp_path / "work", "data": tmp_path / "data"}


@pytest.fixture
def stub_stack(monkeypatch):
    """Stand in for the policy server, the checker and the physics backend.

    Only these three touch a GPU, a socket or MuJoCo. Everything between them is the code
    under test, so stubbing exactly here maximises what the run actually exercises.
    """
    # `metadata` is what the stub server publishes in its handshake; a test that wants a
    # conditioned server mutates it before calling main.
    calls = {"rollouts": [], "metadata": {
        "cfg_conditioning": False, "guidance_scale": 0.0, "ckpt": "stub",
        "supports_client_noise": True, "action_horizon": 10, "action_dim": 8,
    }}

    client_mod = types.ModuleType("openpi_client.websocket_client_policy")

    class _Client:
        def __init__(self, host, port):
            calls["server"] = f"{host}:{port}"

        def get_server_metadata(self):
            return dict(calls["metadata"])

    client_mod.WebsocketClientPolicy = _Client
    monkeypatch.setitem(sys.modules, "openpi_client", types.ModuleType("openpi_client"))
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", client_mod)

    checker_mod = types.ModuleType("axis_qvla.bench.sim.checker")
    checker_mod.make_stable_checker = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.checker", checker_mod)

    class _Res:
        def __init__(self, seed):
            # Seed 12 succeeds; the others time out. A run where every seed agrees cannot
            # distinguish "the tally works" from "the tally is a constant".
            self.success = seed == 12
            self.n_steps = 300 if self.success else 1500
            self.sim_time = self.n_steps / 15.0
            self.reason = "success" if self.success else "timeout"

    phys_mod = types.ModuleType("axis_qvla.bench.sim.rollout_physics")

    def _rollout(task_dir, client, *, init_state_seed, video_path=None, **kw):
        calls["rollouts"].append(
            {"seed": init_state_seed, "task_dir": task_dir, "video_path": video_path, **kw}
        )
        return _Res(init_state_seed)

    phys_mod.rollout_pi05_policy = _rollout
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.rollout_physics", phys_mod)
    return calls


@pytest.fixture
def dualsim_stubs(stub_stack, monkeypatch):
    """Stand in for the layout planner and the dual-sim backend, on top of `stub_stack`.

    The planner is stubbed because a real plan needs the task's layout manifest and scene
    directory; what is under test here is everything the CLI decides BEFORE the backend --
    the arm directory, the seed set, the provenance -- not the layout sampler.
    """
    class _Layout:
        base_variant_index = 4

        def to_json(self):
            return {"base_variant_index": 4}

    plan_mod = types.ModuleType("axis_qvla.bench.sim.layout_plan")
    plan_mod.EXIT_UNEVALUABLE = 3
    # Identity: the real one reshapes a dict-keyed benchmark layouts file (a filesystem read
    # this fixture exists to avoid); callers pass its result straight to plan_layouts, which is
    # itself stubbed below, so a pass-through keeps "no real file is ever touched" true.
    plan_mod.adapt_layout_manifest = lambda manifest, out_dir: manifest
    plan_mod.plan_layouts = (
        lambda task_id, seeds, manifest, task_data_root=None: {int(s): _Layout() for s in seeds})
    plan_mod.write_unevaluable = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.layout_plan", plan_mod)

    layouts_mod = types.ModuleType("axis_qvla.bench.sim.layouts")
    layouts_mod.UnsupportedTaskError = type("UnsupportedTaskError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.layouts", layouts_mod)

    class _Res:
        success, n_steps, sim_time, reason = False, 1500, 100.0, "timeout"

    ds_mod = types.ModuleType("axis_qvla.bench.sim.rollout_dualsim")

    def _rollout(task_dir, client, *, init_state_seed, **kw):
        stub_stack["rollouts"].append({"seed": init_state_seed, "task_dir": task_dir, **kw})
        return _Res()

    ds_mod.rollout_pi05_dualsim = _rollout
    monkeypatch.setitem(sys.modules, "axis_qvla.bench.sim.rollout_dualsim", ds_mod)
    return stub_stack


def _run(fixture_paths, *extra):
    return cli.main([
        "--task-id", "866", "--variant", "vanilla",
        "--checkpoint-dir", "/ckpt/pi05_axis_slb_866_vanilla/9999",
        "--no-dual-sim", "--no-eval-layouts", "--benchmark", "pilot866",
        "--log-root", str(fixture_paths["logs"]),
        "--work-dir", str(fixture_paths["work"]),
        "--task-data-root", str(fixture_paths["data"]),
        "--asset-dir", str(fixture_paths["data"]),
        *extra,
    ])


def _log_dir(fixture_paths, arm="vanilla"):
    return fixture_paths["logs"] / arm / "task_866"


def test_a_physics_sweep_runs_to_completion(fixture_paths, stub_stack):
    assert _run(fixture_paths) == 0
    assert [c["seed"] for c in stub_stack["rollouts"]] == SEEDS


def test_meta_carries_every_field_the_legacy_script_wrote(fixture_paths, stub_stack):
    """Provenance loss is silent: the results still look fine, they just no longer say what
    produced them, so every field is checked."""
    _run(fixture_paths)
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert LEGACY_META_FIELDS <= set(meta), LEGACY_META_FIELDS - set(meta)


def test_seed_records_carry_every_field_the_legacy_script_wrote(fixture_paths, stub_stack):
    _run(fixture_paths)
    rec = json.loads((_log_dir(fixture_paths) / "seed_12.json").read_text())
    assert LEGACY_SEED_FIELDS <= set(rec), LEGACY_SEED_FIELDS - set(rec)


def test_the_tally_counts_successes_not_seeds(fixture_paths, stub_stack):
    _run(fixture_paths)
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert (meta["n_success"], meta["n_seeds"]) == (1, 3)


def test_a_finished_run_is_marked_complete(fixture_paths, stub_stack):
    """`complete` is what tells a reader the number is final rather than a crashed partial."""
    _run(fixture_paths)
    assert json.loads((_log_dir(fixture_paths) / "meta.json").read_text())["complete"] is True


def test_a_crashed_sweep_leaves_attributable_partials(fixture_paths, stub_stack, monkeypatch):
    """The whole point of writing meta.json before the loop. Legacy wrote it only at the end,
    so a sweep killed at seed 14 of 20 left fourteen results naming no checkpoint at all."""
    class _Timeout:
        success, n_steps, sim_time, reason = False, 5, 0.3, "timeout"

    def _boom(task_dir, client, *, init_state_seed, **kw):
        if init_state_seed == 13:
            raise RuntimeError("simulator died")
        return _Timeout()

    monkeypatch.setattr(
        sys.modules["axis_qvla.bench.sim.rollout_physics"], "rollout_pi05_policy", _boom
    )
    with pytest.raises(RuntimeError, match="simulator died"):
        _run(fixture_paths)

    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert meta["complete"] is False
    assert meta["checkpoint_dir"] == "/ckpt/pi05_axis_slb_866_vanilla/9999"
    assert (_log_dir(fixture_paths) / "seed_11.json").is_file()


def test_domain_is_recorded_so_two_domains_cannot_be_averaged(fixture_paths, stub_stack):
    _run(fixture_paths)
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert meta["domain"] == "physics"
    assert meta["eval_layouts"] is False


def test_arm_name_override_redirects_the_whole_run(fixture_paths, stub_stack):
    """The held-out sweeps write hx_<task>_<arm>, which no variant name derives."""
    _run(fixture_paths, "--run-name", "hx_866_base")
    assert (_log_dir(fixture_paths, "hx_866_base") / "meta.json").is_file()


def test_diagnostics_off_asks_the_backend_for_no_video(fixture_paths, stub_stack):
    _run(fixture_paths)
    assert all(c["video_path"] is None for c in stub_stack["rollouts"])


def test_diagnostics_on_names_an_mp4_in_the_physics_domain(fixture_paths, stub_stack):
    """imageio picks the container from the suffix; `.diag.json` is not a video container."""
    _run(fixture_paths, "--diagnostics")
    paths = [c["video_path"] for c in stub_stack["rollouts"]]
    assert [p.name for p in paths] == [f"seed_{s}.mp4" for s in SEEDS]


def test_the_prompt_reaches_the_backend_verbatim(fixture_paths, stub_stack):
    """task_name_canonical matches meta/tasks.parquet; a display name is a string the policy
    never saw in training and its failure looks like a capability gap."""
    _run(fixture_paths)
    assert stub_stack["rollouts"][0]["prompt"] == PROMPT


def test_the_gripper_widths_reach_the_backend(fixture_paths, stub_stack):
    _run(fixture_paths)
    assert stub_stack["rollouts"][0]["open_w"] == 0.08
    assert stub_stack["rollouts"][0]["closed_w"] == 0.002


def test_n_seeds_truncates_in_declared_order(fixture_paths, stub_stack):
    _run(fixture_paths, "--n-seeds", "2")
    assert [c["seed"] for c in stub_stack["rollouts"]] == SEEDS[:2]


def test_a_guidance_mismatch_stops_before_any_rollout(fixture_paths, stub_stack):
    """The server owns the sampling config. A mislaunched server would otherwise write w=2
    rollouts into the w=0 directory and nothing downstream could tell."""
    from axis_qvla.bench.sim.policy import PolicyError

    with pytest.raises(PolicyError, match="mismatch"):
        _run(fixture_paths, "--cfg-conditioning", "--guidance-scale", "2.0")
    assert stub_stack["rollouts"] == []


# --- --seeds: the subset a queue worker hands one invocation ------------------------------

def test_seed_subset_is_honoured_in_declared_order(fixture_paths, stub_stack):
    """Order is the declaration's, not the caller's: a subset run must agree seed-for-seed with
    the full run, or two workers splitting one task would evaluate it differently."""
    _run(fixture_paths, "--seeds", "13,11")
    assert [c["seed"] for c in stub_stack["rollouts"]] == [11, 13]
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert meta["seeds"] == [11, 13]


def test_an_undeclared_seed_is_refused_before_any_rollout(fixture_paths, stub_stack):
    """A seed with no screened initial state would add an off-benchmark condition to the
    denominator and still be counted."""
    with pytest.raises(SystemExit, match="not among"):
        _run(fixture_paths, "--seeds", "11,99")
    assert stub_stack["rollouts"] == []


def test_an_empty_seed_list_is_refused(fixture_paths, stub_stack):
    with pytest.raises(SystemExit, match="empty"):
        _run(fixture_paths, "--seeds", ",")
    assert stub_stack["rollouts"] == []


# --- --benchmark: prompt, seeds and gripper from the tracked declaration -------------------

def _run_benchmark(fixture_paths, *extra):
    """The default benchmark supplies the task."""
    return cli.main([
        "--task-id", "806", "--variant", "vanilla",
        "--checkpoint-dir", "/ckpt/base/9999",
        "--no-dual-sim", "--no-eval-layouts",
        "--log-root", str(fixture_paths["logs"]),
        "--work-dir", str(fixture_paths["work"]),
        "--task-data-root", str(fixture_paths["data"]),
        "--asset-dir", str(fixture_paths["data"]),
        *extra,
    ])


def test_benchmark_supplies_prompt_seeds_and_gripper(fixture_paths, stub_stack):
    from axis_qvla.bench.spec import load_benchmark

    entry = load_benchmark().task(806)
    assert _run_benchmark(fixture_paths, "--seeds", "806003,806001") == 0
    first = stub_stack["rollouts"][0]
    assert first["prompt"] == entry.prompt
    assert (first["open_w"], first["closed_w"]) == (entry.open_w, entry.closed_w)
    assert [c["seed"] for c in stub_stack["rollouts"]] == [806001, 806003]


def test_benchmark_identity_is_recorded_by_content(fixture_paths, stub_stack):
    """A run record that names the benchmark by label alone cannot tell two editions of the
    task list apart; the digest can."""
    _run_benchmark(fixture_paths, "--seeds", "806000")
    meta = json.loads((fixture_paths["logs"] / "vanilla" / "task_806" / "meta.json").read_text())
    assert meta["benchmark"]["name"] == "franka_heldout24"
    assert len(meta["benchmark"]["sha256"]) == 64
    assert meta["task_manifest"] is None


def test_benchmark_n_seeds_is_the_default_seed_count(fixture_paths, stub_stack):
    from axis_qvla.bench.spec import load_benchmark

    _run_benchmark(fixture_paths)
    assert len(stub_stack["rollouts"]) == load_benchmark().n_seeds


def test_a_task_outside_the_benchmark_is_refused(fixture_paths, stub_stack):
    with pytest.raises(SystemExit, match="not in benchmark"):
        cli.main([
            "--task-id", "866", "--variant", "vanilla", "--checkpoint-dir", "/c",
            "--no-dual-sim", "--no-eval-layouts", "--log-root", str(fixture_paths["logs"]),
            "--work-dir", str(fixture_paths["work"]), "--task-data-root", str(fixture_paths["data"]),
        ])
    assert stub_stack["rollouts"] == []


# --- conditioning is read off the server, and the flags are an assertion -------------------

def _run_dualsim(fixture_paths, *extra):
    return cli.main([
        "--task-id", "866", "--variant", "vanilla",
        "--checkpoint-dir", "/ckpt/pi05_axis_slb_866_vanilla/9999",
        "--dual-sim", "--benchmark", "pilot866",
        "--log-root", str(fixture_paths["logs"]),
        "--work-dir", str(fixture_paths["work"]),
        "--task-data-root", str(fixture_paths["data"]),
        "--asset-dir", str(fixture_paths["data"]),
        "--render-root", str(fixture_paths["data"]),
        *extra,
    ])


def test_a_conditioned_server_names_the_w0_dualsim_arm_without_any_flag(
    fixture_paths, dualsim_stubs,
):
    """A worker need not restate the server's conditioning: the handshake is read."""
    dualsim_stubs["metadata"].update({"cfg_conditioning": True, "guidance_scale": 0.0})
    assert _run_dualsim(fixture_paths) == 0
    d = _log_dir(fixture_paths, "vanilla_w0_dualsim")
    meta = json.loads((d / "meta.json").read_text())
    assert meta["cfg_conditioning"] is True and meta["guidance_scale"] == 0.0
    assert meta["conditioning_source"] == "server"
    assert meta["arm"] == "vanilla_w0_dualsim" and meta["domain"] == "dualsim"
    rec = json.loads((d / "seed_12.json").read_text())
    assert rec["cfg_conditioning"] is True and rec["base_variant_index"] == 4
    assert not _log_dir(fixture_paths, "vanilla_dualsim").exists()


def test_an_unconditioned_server_names_the_bare_dualsim_arm(fixture_paths, dualsim_stubs):
    assert _run_dualsim(fixture_paths) == 0
    assert (_log_dir(fixture_paths, "vanilla_dualsim") / "meta.json").is_file()


def test_a_matching_assertion_is_accepted_and_recorded_as_such(fixture_paths, dualsim_stubs):
    dualsim_stubs["metadata"].update({"cfg_conditioning": True, "guidance_scale": 0.0})
    assert _run_dualsim(fixture_paths, "--cfg-conditioning", "--guidance-scale", "0") == 0
    meta = json.loads((_log_dir(fixture_paths, "vanilla_w0_dualsim") / "meta.json").read_text())
    assert meta["conditioning_source"] == "asserted"


def test_a_mismatching_assertion_is_refused_before_any_rollout(fixture_paths, dualsim_stubs):
    """--no-cfg-conditioning against a tagged server: refused, exactly as the restatement
    was. Deriving from the server must not make the assertion toothless."""
    from axis_qvla.bench.sim.policy import PolicyError

    dualsim_stubs["metadata"].update({"cfg_conditioning": True, "guidance_scale": 0.0})
    with pytest.raises(PolicyError, match="mismatch"):
        _run_dualsim(fixture_paths, "--no-cfg-conditioning")
    assert dualsim_stubs["rollouts"] == []
    assert not _log_dir(fixture_paths, "vanilla_w0_dualsim").exists()
    assert not _log_dir(fixture_paths, "vanilla_dualsim").exists()


def test_a_guided_server_carries_its_scale_in_the_arm_name(fixture_paths, dualsim_stubs):
    """Four scales of a sweep share one checkpoint; the name is the only thing keeping them
    out of one directory."""
    dualsim_stubs["metadata"].update({"cfg_conditioning": True, "guidance_scale": 1.5})
    assert _run_dualsim(fixture_paths) == 0
    assert (_log_dir(fixture_paths, "vanilla_w1.5_dualsim") / "meta.json").is_file()


# --- --task-data-root reaches the planner, the checker and the run record ------------------

def test_the_task_data_root_reaches_the_planner_and_the_checker(fixture_paths, dualsim_stubs, monkeypatch):
    """cli.py hands --task-data-root to plan_layouts and to make_stable_checker:
    a root that reached only the rollout backend would score against the default task files."""
    seen = {}
    plan_mod = sys.modules["axis_qvla.bench.sim.layout_plan"]
    inner = plan_mod.plan_layouts

    def _plan(task_id, seeds, manifest, task_data_root=None):
        seen["plan"] = task_data_root
        return inner(task_id, seeds, manifest)

    monkeypatch.setattr(plan_mod, "plan_layouts", _plan)
    monkeypatch.setattr(sys.modules["axis_qvla.bench.sim.checker"], "make_stable_checker",
                        lambda *a, **k: seen.setdefault("checker", k.get("task_data_root")))
    assert _run_dualsim(fixture_paths) == 0
    dualsim_stubs["rollouts"][0]["checker_factory"]({})
    assert seen == {"plan": fixture_paths["data"], "checker": fixture_paths["data"]}
    assert all(c["task_dir"] == fixture_paths["data"] / "task_866" for c in dualsim_stubs["rollouts"])


def test_the_run_record_hashes_the_task_files_it_read(fixture_paths, stub_stack, monkeypatch):
    """meta.json ties the result to the task files by content."""
    import hashlib

    monkeypatch.delenv("AXIS_HOST", raising=False)
    d = fixture_paths["data"] / "task_866"
    d.mkdir(parents=True)
    (d / "task.json").write_text('{"scene_variants": []}')
    (d / "checker_config.json").write_text('{"checker": {}}')
    _run(fixture_paths)
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert meta["task_data"] == {"root": None, "files": {
        "task.json": hashlib.sha256(b'{"scene_variants": []}').hexdigest(),
        "checker_config.json": hashlib.sha256(b'{"checker": {}}').hexdigest()}}


def test_refuse_without_an_asset_dir_stops_before_the_server(fixture_paths, stub_stack):
    """A scene asset that is not local must fail before any GPU or server time is spent."""
    with pytest.raises(SystemExit, match="--asset-dir"):
        cli.main([
            "--task-id", "866", "--variant", "vanilla", "--checkpoint-dir", "/ckpt/x/9999",
            "--no-dual-sim", "--no-eval-layouts", "--benchmark", "pilot866",
            "--log-root", str(fixture_paths["logs"]), "--work-dir", str(fixture_paths["work"]),
            "--task-data-root", str(fixture_paths["data"]),
        ])
    assert "server" not in stub_stack and stub_stack["rollouts"] == []


def test_the_asset_policy_is_recorded(fixture_paths, stub_stack):
    _run(fixture_paths)
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert meta["assets"] == {"fetch": "refuse", "asset_dir": str(fixture_paths["data"]),
                              "base_url": None, "lock": None}
    assert stub_stack["rollouts"][0]["asset_fetch"] == "refuse"


def test_a_refused_asset_aborts_the_cli_with_exit_4_and_no_seed_file(fixture_paths, stub_stack,
                                                                     monkeypatch, capsys):
    """A refused or lock-mismatched asset stops the run by name; it is never recorded as an error
    seed that scoring would tally as a task failure."""
    import axis_qvla.bench.sim.rollout_physics as phys
    from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED, AssetFetchRefused

    inner = phys.rollout_pi05_policy

    def refuse_on_12(task_dir, client, *, init_state_seed, **kw):
        if init_state_seed == 12:
            raise AssetFetchRefused("asset 'tabletopgen/a.obj' is not under /assets")
        return inner(task_dir, client, init_state_seed=init_state_seed, **kw)

    monkeypatch.setattr(phys, "rollout_pi05_policy", refuse_on_12)
    assert _run(fixture_paths) == EXIT_ASSET_REFUSED
    err = capsys.readouterr().err
    assert "ABORT" in err and "tabletopgen/a.obj" in err
    d = _log_dir(fixture_paths)
    assert (d / "seed_11.json").is_file() and not (d / "seed_12.json").exists()
    assert not (d / "seed_13.json").exists()
    assert json.loads((d / "meta.json").read_text())["complete"] is False


def test_an_ordinary_rollout_error_is_still_recorded_as_a_seed(fixture_paths, stub_stack, monkeypatch):
    import axis_qvla.bench.sim.rollout_physics as phys

    class _Err:
        success, n_steps, sim_time, reason = False, 0, 0.0, "error:build_physics:no mjcf_xml"

    monkeypatch.setattr(phys, "rollout_pi05_policy", lambda *a, **k: _Err())
    assert _run(fixture_paths) == 0
    rec = json.loads((_log_dir(fixture_paths) / "seed_12.json").read_text())
    assert rec["reason"] == "error:build_physics:no mjcf_xml"


# --- a broken --asset-checksums / --asset-dir aborts BEFORE the server ---
# The checksum manifest is read once, before the first scene build: a missing file, bad JSON or a
# bad row is one refusal (exit 4, which the queue worker turns into an aborted queue and the driver
# into an unscored run), not an error record per seed.

def _lock_file(fixture_paths, kind: str):
    p = fixture_paths["work"].parent / "ASSETS.lock.json"
    row = {"path": "assets/tabletopgen/a.obj", "size": 1, "sha256": "0" * 64}
    if kind == "valid":
        p.write_text(json.dumps({"schema": "qvla.assets_lock/1", "files": [row]}))
    elif kind == "not json":
        p.write_text("{")
    elif kind == "row without sha256":
        p.write_text(json.dumps({"files": [{k: v for k, v in row.items() if k != "sha256"}]}))
    elif kind == "directory":
        p.mkdir()
    return p


@pytest.mark.parametrize("kind", ["missing", "not json", "row without sha256", "directory"])
def test_a_broken_asset_lock_aborts_before_the_server(fixture_paths, stub_stack, capsys, kind):
    from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED

    assert _run(fixture_paths, "--asset-checksums", str(_lock_file(fixture_paths, kind))) == EXIT_ASSET_REFUSED
    err = capsys.readouterr().err
    assert "ABORT" in err and "lock" in err
    assert "server" not in stub_stack and stub_stack["rollouts"] == []
    assert not _log_dir(fixture_paths).exists()


@pytest.mark.parametrize("kind", ["missing", "a file"])
def test_an_asset_dir_that_is_not_a_directory_aborts_before_the_server(fixture_paths, stub_stack,
                                                                      capsys, kind):
    from axis_qvla.bench.sim.assets import EXIT_ASSET_REFUSED

    p = fixture_paths["work"].parent / "no_assets"
    if kind == "a file":
        p.write_text("x")
    assert _run(fixture_paths, "--asset-dir", str(p)) == EXIT_ASSET_REFUSED
    err = capsys.readouterr().err
    assert "ABORT" in err and "asset dir" in err
    assert "server" not in stub_stack and stub_stack["rollouts"] == []


def test_a_valid_lock_is_accepted_recorded_and_forwarded(fixture_paths, stub_stack):
    p = _lock_file(fixture_paths, "valid")
    assert _run(fixture_paths, "--asset-checksums", str(p)) == 0
    meta = json.loads((_log_dir(fixture_paths) / "meta.json").read_text())
    assert meta["assets"]["lock"] == str(p)
    assert {c["asset_lock"] for c in stub_stack["rollouts"]} == {str(p)}


# --- a missing root is refused up front, by flag name ---------

def test_a_missing_task_data_root_is_refused_by_flag_name(fixture_paths, stub_stack):
    with pytest.raises(SystemExit, match="--task-data-root"):
        _run(fixture_paths, "--task-data-root", str(fixture_paths["work"].parent / "no_tasks"))
    assert "server" not in stub_stack and stub_stack["rollouts"] == []


def test_a_missing_render_root_is_refused_by_flag_name(fixture_paths, dualsim_stubs):
    with pytest.raises(SystemExit, match="--render-root"):
        _run_dualsim(fixture_paths, "--render-root", str(fixture_paths["work"].parent / "no_render"))
    assert "server" not in dualsim_stubs and dualsim_stubs["rollouts"] == []


def test_the_physics_domain_needs_no_render_root(fixture_paths, stub_stack):
    assert _run(fixture_paths, "--render-root", str(fixture_paths["work"].parent / "no_render")) == 0


def test_an_unreadable_root_is_reported_as_unreadable_not_missing(fixture_paths, stub_stack):
    """EACCES is not 'missing'."""
    locked = fixture_paths["work"].parent / "locked"
    (locked / "tasks").mkdir(parents=True)
    locked.chmod(0)
    try:
        with pytest.raises(SystemExit) as exc:
            _run(fixture_paths, "--task-data-root", str(locked / "tasks"))
    finally:
        locked.chmod(0o755)
    msg = str(exc.value)
    assert "--task-data-root" in msg and "cannot be read (PermissionError)" in msg
    assert "not a directory" not in msg
    assert "server" not in stub_stack and stub_stack["rollouts"] == []


# --- meta.json records the replan the backend executes ------------------------

def test_meta_records_the_replan_the_dual_sim_rollout_executes(fixture_paths, dualsim_stubs, monkeypatch):
    """The dual-sim backend consumes `effective_replan(replan)`, which honours AXIS_EVAL_REPLAN,
    while meta.json recorded PI05_REPLAN: a replan=4 run was filed as a replan=8 one."""
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "4")
    assert _run_dualsim(fixture_paths) == 0
    meta = json.loads((_log_dir(fixture_paths, "vanilla_dualsim") / "meta.json").read_text())
    assert meta["replan"] == 4
    assert {c["replan"] for c in dualsim_stubs["rollouts"]} == {4}


def test_the_physics_loop_ignores_the_replan_override_and_meta_says_so(fixture_paths, stub_stack, monkeypatch):
    """Only the dual-sim loop honours AXIS_EVAL_REPLAN; the physics loop consumes `replan` as
    passed, so its record keeps the default."""
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "4")
    assert _run(fixture_paths) == 0
    assert json.loads((_log_dir(fixture_paths) / "meta.json").read_text())["replan"] == 8
    assert {c["replan"] for c in stub_stack["rollouts"]} == {8}


def test_a_malformed_replan_override_is_refused_before_any_rollout(fixture_paths, dualsim_stubs, monkeypatch):
    monkeypatch.setenv("AXIS_EVAL_REPLAN", "0")
    with pytest.raises(SystemExit, match="AXIS_EVAL_REPLAN"):
        _run_dualsim(fixture_paths)
    assert dualsim_stubs["rollouts"] == []
