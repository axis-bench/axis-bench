"""`bench eval --from-run <id>`: a fine-tune `qvla train` ran here is served on its backbone, from its
own checkpoint, with the config the training package rendered for serving, its seeds recorded, and
every serving mode its eval.json records evaluated.

The training package's serve interface (axis_qvla.train.serve.serve_spec) is replaced by a stub
with its declared contract (`write=False` writes nothing and carries the config's bytes; RunNotReady
for a run missing or unfinished), so this runs before that package is in this checkout."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from axis_qvla.bench import local_run
from axis_qvla.cli import main
from tests.unit.bench.eval_stubs import bench_inputs, lerobot_host
from tests.unit.bench.test_openpi_serve import host  # noqa: F401  (a fixture)

BARE = SimpleNamespace(name="bare", serving="bare", fields=MappingProxyType({}))
COND = SimpleNamespace(name="cond", serving="conditioned", fields=MappingProxyType({"quality": "5"}))


class ServeSpecError(ValueError):
    pass


class RunNotReady(ServeSpecError):
    pass


def _spec(paths, run_id="pi05.finetune.mine", *, backend="openpi", stage="finetune", modes=(BARE,),
          ft_seed=7, pretrain_seed=42, write=True):
    """B's ServeSpec. A pi0.5 run's config is written to serve_config.json, or with write=False only
    carried as bytes."""
    run = paths.runs / run_id
    data = None
    if backend == "openpi":
        ckpt, config = run / "train" / "final", run / "serve_config.json"
        if write:
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_bytes(b'{"serve": 1}')
        else:
            data = b'{"serve": 1}'
    else:
        ckpt, config = run / "train" / "checkpoints" / "last" / "pretrained_model", None
    return SimpleNamespace(run_id=run_id, backbone={"openpi": "pi05"}.get(backend, backend), backend=backend,
                           stage=stage, checkpoint=ckpt, config=config, config_bytes=data, modes=tuple(modes),
                           ft_seed=ft_seed, pretrain_seed=pretrain_seed, pretrain="pi05.pretrain.mine")


def _run(*args, **kw):
    """serve_spec's answer for a run, built when it is asked (with its `write`)."""
    return lambda paths, write=True: _spec(paths, *args, write=write, **kw)


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []


@pytest.fixture
def serve(monkeypatch):
    """The training package's serve interface: `runs[run id]` is what serve_spec returns for it (a
    callable is called with the paths and `write`; RunNotReady when absent, as for a run not trained
    yet; `errors[run id]` raised as given); `calls` the run dirs it was asked about, `writes` each
    call's `write`."""
    state = SimpleNamespace(runs={}, errors={}, calls=[], writes=[])

    def serve_spec(run_dir, paths, *, write=True):
        state.calls.append(Path(run_dir))
        state.writes.append(write)
        name = Path(run_dir).name
        if name in state.errors:
            raise state.errors[name]
        if name not in state.runs:
            raise RunNotReady(f"{run_dir} is not a run `qvla train` wrote: no run_spec.toml")
        return state.runs[name](paths, write=write) if callable(state.runs[name]) else state.runs[name]

    monkeypatch.setitem(sys.modules, "axis_qvla.train.serve", SimpleNamespace(
        serve_spec=serve_spec, ServeSpecError=ServeSpecError, RunNotReady=RunNotReady))
    return state


def _paths(tmp_path):
    return SimpleNamespace(runs=tmp_path / "runs")


# --- what a run resolves to ------------------------------------------------------------------------

def test_a_pi05_finetune_serves_its_checkpoint_with_the_config_the_training_package_rendered(tmp_path, serve):
    paths = _paths(tmp_path)
    serve.runs["pi05.finetune.mine"] = _spec(paths)
    lr = local_run.resolve("pi05.finetune.mine", paths)
    assert serve.calls == [paths.runs / "pi05.finetune.mine"]
    assert (lr.backend, lr.config, lr.serve_config) == ("openpi", None, b'{"serve": 1}')
    assert lr.checkpoint == paths.runs / "pi05.finetune.mine" / "train" / "final"
    assert (lr.ft_seed, lr.pretrain_seed, [m.name for m in lr.modes]) == (7, 42, ["bare"])
    assert lr.arm(lr.modes[0]) == "pi05.finetune.mine" and lr.arm(lr.modes[0], "mine") == "mine"


def test_a_lerobot_finetune_serves_its_pretrained_model_under_its_policy_name(tmp_path, serve):
    paths = _paths(tmp_path)
    serve.runs["smolvla.finetune.mine"] = _spec(paths, "smolvla.finetune.mine", backend="smolvla", pretrain_seed=None)
    lr = local_run.resolve("smolvla.finetune.mine", paths)
    assert (lr.backend, lr.config, lr.serve_config, lr.pretrain_seed) == ("smolvla", "smolvla", None, None)
    assert lr.checkpoint.parts[-2:] == ("last", "pretrained_model")


def test_a_cond_run_is_evaluated_bare_then_cond_each_under_its_own_run(tmp_path, serve):
    paths = _paths(tmp_path)
    serve.runs["x.cond"] = _spec(paths, "x.cond", modes=(BARE, COND))
    lr = local_run.resolve("x.cond", paths)
    assert [(m.name, m.serving, lr.arm(m)) for m in lr.modes] == [("bare", "bare", "x.cond_b"),
                                                                    ("cond", "conditioned", "x.cond_c")]
    # --serving narrows to one mode, which keeps its two-mode name, so a later full run finds it.
    (only,) = local_run.select(lr, "conditioned")
    assert lr.arm(only, "r") == "r_c"
    with pytest.raises(local_run.LocalRunError, match="--serving conditioned is not one of them"):
        local_run.select(dataclasses.replace(lr, modes=lr.modes[:1]), "conditioned")


def test_a_pretraining_run_is_refused_naming_the_finetune(tmp_path, serve):
    paths = _paths(tmp_path)
    serve.runs["pi05.pretrain.mine"] = _spec(paths, "pi05.pretrain.mine", stage="pretrain")
    with pytest.raises(local_run.LocalRunError, match="fine-tune it first .`qvla train finetune --from pi05.pretrain.mine`") as e:
        local_run.resolve("pi05.pretrain.mine", paths)
    assert not e.value.pending


def test_an_unfinished_run_is_refused_with_the_training_packages_reason(tmp_path, serve):
    with pytest.raises(local_run.LocalRunError, match="no run_spec.toml") as e:
        local_run.resolve("pi05.finetune.mine", _paths(tmp_path))
    assert e.value.pending


def test_a_checkout_without_the_serve_interface_says_to_update(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "axis_qvla.train.serve", None)     # import of it fails, as before B merges
    with pytest.raises(local_run.LocalRunError, match=r"update this checkout \(axis_qvla.train.serve is missing\)") as e:
        local_run.resolve("pi05.finetune.mine", _paths(tmp_path))
    assert e.value.pending


def test_a_mode_served_otherwise_than_this_benchmark_serves_it_is_refused(tmp_path, serve):
    paths = _paths(tmp_path)
    odd = SimpleNamespace(name="cond", serving="conditioned", fields={"quality": "4"})
    serve.runs["x"] = _spec(paths, "x", modes=(BARE, odd))
    with pytest.raises(local_run.LocalRunError, match="'quality': '4'"):
        local_run.resolve("x", paths)


@pytest.mark.parametrize("bad", ["../x", "a/b", ""])
def test_a_run_id_that_is_not_one_directory_is_refused(tmp_path, serve, bad):
    with pytest.raises(local_run.LocalRunError, match="is not a run id"):
        local_run.resolve(bad, _paths(tmp_path))
    assert serve.calls == []


# --- bench eval --from-run -------------------------------------------------------------------------

def _record(monkeypatch, d: Path) -> list[dict]:
    """The campaign driver and finish replaced by recorders of every call, in order."""
    from axis_qvla.bench import finish
    from axis_qvla.bench.sim import campaign
    from tests.unit.bench.eval_stubs import make_envs

    calls: list[dict] = []

    def fake_main(argv):
        cfg = next((x[len("--config="):] for x in argv if x.startswith("--config=")), None)
        calls.append({"argv": list(argv), "config_at_start": Path(cfg).read_bytes() if cfg and cfg.endswith(".json") else None})
        return 0
    monkeypatch.setattr(campaign, "main", fake_main)
    monkeypatch.setattr(finish, "finish", lambda root, arm, b, **kw: calls.append({"finish": (arm, kw)}) or 0)
    monkeypatch.setattr(sys, "executable", str(d / "venvs" / "render" / "bin" / "python"))
    make_envs(d)
    return calls


def test_from_run_serves_the_run_with_its_rendered_config_and_seeds(host, serve, monkeypatch):
    calls = _record(monkeypatch, host)
    serve.runs["pi05.finetune.mine"] = _spec
    assert main(["bench", "eval", "--host", "demo", "--from-run", "pi05.finetune.mine", *bench_inputs(host, monkeypatch)]) == 0
    run, done = calls
    cfg = host / "r1" / "out_pi05.finetune.mine" / "serve" / "openpi_config.json"
    ckpt = host / "runs" / "pi05.finetune.mine" / "train" / "final"
    assert f"--config={cfg}" in run["argv"] and f"--checkpoint={ckpt}" in run["argv"]
    assert "--run=pi05.finetune.mine" in run["argv"] and "--serving=bare" in run["argv"]
    assert run["config_at_start"] == b'{"serve": 1}'
    assert done == {"finish": ("pi05.finetune.mine", {"ft_seed": 7, "pretrain_seed": 42})}


def test_from_run_of_a_cond_run_evaluates_bare_then_cond_and_finishes_each(host, serve, monkeypatch):
    calls = _record(monkeypatch, host)
    serve.runs["pi05.finetune.mine.cond"] = _run("pi05.finetune.mine.cond", modes=(BARE, COND))
    assert main(["bench", "eval", "--host", "demo", "--from-run", "pi05.finetune.mine.cond", "--guidance-scale", "1.5",
                 *bench_inputs(host, monkeypatch)]) == 0
    assert [c.get("finish", (None,))[0] for c in calls] == [None, "pi05.finetune.mine.cond_b",
                                                            None, "pi05.finetune.mine.cond_c"]
    bare, cond = calls[0]["argv"], calls[2]["argv"]
    assert "--serving=bare" in bare and "--guidance-scale=0" in bare and "--run=pi05.finetune.mine.cond_b" in bare
    assert "--serving=conditioned" in cond and "--guidance-scale=1.5" in cond and "--run=pi05.finetune.mine.cond_c" in cond
    # Each mode's server config sits beside its own run.
    assert all(c["config_at_start"] == b'{"serve": 1}' for c in (calls[0], calls[2]))


def test_serving_narrows_a_cond_run_to_one_mode(host, serve, monkeypatch):
    calls = _record(monkeypatch, host)
    serve.runs["x.cond"] = _run("x.cond", modes=(BARE, COND))
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x.cond", "--serving", "conditioned", "--run", "r",
                 *bench_inputs(host, monkeypatch)]) == 0
    assert [c["finish"][0] for c in calls if "finish" in c] == ["r_c"]


def test_guidance_for_a_run_evaluated_bare_only_is_refused(host, serve, monkeypatch, capsys):
    _record(monkeypatch, host)
    serve.runs["x"] = _run("x")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", "--guidance-scale", "1", "--dry-run"]) == 2
    assert "--guidance-scale weighs conditioned serving" in capsys.readouterr().out


def test_from_run_of_a_lerobot_run_passes_its_policy_name_and_pretrained_model(host, serve, capsys):
    lerobot_host(host)
    serve.runs["smolvla.finetune.mine"] = _run("smolvla.finetune.mine", backend="smolvla")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "smolvla.finetune.mine", "--dry-run"]) == 0
    out = capsys.readouterr().out
    ckpt = host / "runs" / "smolvla.finetune.mine" / "train" / "checkpoints" / "last" / "pretrained_model"
    assert "--backend=smolvla" in out and "--config=smolvla" in out and f"--checkpoint={ckpt}" in out
    assert "serve config:" not in out and " bench eval --from-run smolvla.finetune.mine" in out.splitlines()[-1]


@pytest.mark.parametrize("extra", [["--checkpoint", "c"], ["--checkpoint-id", "pi05/ft/x"], ["--config", "c"],
                                   ["--backend", "xvla"], ["--policy", "127.0.0.1:9"]])
def test_from_run_names_everything_it_serves(extra, capsys):
    assert main(["bench", "eval", "--from-run", "my.ft", *extra]) == 2
    assert "--from-run serves the checkpoint of a run trained here" in capsys.readouterr().out


def test_from_run_has_no_published_digest_to_verify(capsys):
    assert main(["bench", "eval", "--from-run", "my.ft", "--verify-checkpoint"]) == 2
    assert "has no published digest" in capsys.readouterr().out


def test_from_run_print_needs_no_run_yet(host, serve, capsys):
    assert main(["bench", "eval", "--host", "demo", "--from-run", "my.ft", "--print"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("# serves: run my.ft, which does not resolve on this machine yet")
    assert f"# results file: {host / 'r1' / 'out_my.ft' / 'results.json'}" in lines
    assert " bench eval --from-run my.ft" in lines[-1] and "--print" not in lines[-1]


def test_from_run_print_before_the_serve_interface_is_there(host, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "axis_qvla.train.serve", None)
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x.cond", "--print"]) == 0
    out = capsys.readouterr().out
    assert "update this checkout (axis_qvla.train.serve is missing)" in out
    assert f"# results file: {host / 'r1' / 'out_x.cond_b' / 'results.json'}" in out
    assert f"# results file: {host / 'r1' / 'out_x.cond_c' / 'results.json'}" in out


def test_from_run_without_print_refuses_a_run_not_trained_yet(host, serve, capsys):
    assert main(["bench", "eval", "--host", "demo", "--from-run", "my.ft", "--dry-run"]) == 2
    assert "FAIL  --from-run my.ft: " in capsys.readouterr().out


def test_from_run_print_of_a_pretraining_run_is_refused(host, serve, capsys):
    serve.runs["p"] = _run("p", stage="pretrain")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "p", "--print"]) == 2
    assert "fine-tune it first" in capsys.readouterr().out


def test_from_run_print_of_a_trained_run_names_each_mode(host, serve, capsys):
    serve.runs["x.cond"] = _run("x.cond", modes=(BARE, COND))
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x.cond", "--print"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert "# from run: x.cond (fine-tune seed 7, pretraining seed 42)" in lines
    assert [ln.rsplit(" as run ", 1)[1] for ln in lines if ln.startswith("# serves:")] == ["x.cond_b", "x.cond_c"]
    assert sum(ln.startswith("# results file:") for ln in lines) == 2
    assert lines[-1].endswith("bench eval --from-run x.cond --host demo")


# --- --print writes nothing; refusals are FAIL lines -------------------------------------------------

def _eval_json(runs: Path, run_id: str, serving: list[str]) -> Path:
    import json

    path = runs / run_id / "eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "qvla.train_eval/1", "serving": serving}))
    return path


def test_print_asks_the_training_package_to_write_nothing(host, serve, capsys):
    serve.runs["x.cond"] = _run("x.cond", modes=(BARE, COND))
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x.cond", "--print"]) == 0
    assert serve.writes == [False] and _files(host / "runs") == []
    assert sum(ln.startswith("# serves:") for ln in capsys.readouterr().out.splitlines()) == 2


def test_a_run_serves_the_config_bytes_the_training_package_carries(host, serve, monkeypatch):
    calls = _record(monkeypatch, host)
    serve.runs["x"] = lambda paths, write=True: _spec(paths, "x", write=False)   # bytes, no file
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", *bench_inputs(host, monkeypatch)]) == 0
    assert serve.writes == [True] and calls[0]["config_at_start"] == b'{"serve": 1}'


def test_a_server_config_that_cannot_be_read_is_a_fail_line(host, serve, capsys):
    def no_bytes(paths, write=True):     # neither bytes carried nor a file written
        return SimpleNamespace(**{**vars(_spec(paths, "x", write=False)), "config_bytes": None})
    serve.runs["x"] = no_bytes
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", "--dry-run"]) == 2
    assert "FAIL  --from-run x: its server config " in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--print", "--dry-run"])
def test_a_run_that_cannot_be_read_is_a_fail_line_not_a_traceback(host, serve, capsys, flag):
    serve.errors["x"] = PermissionError(13, "Permission denied", str(host / "runs" / "x"))
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", flag]) == 2
    assert "FAIL  --from-run x: the run cannot be read (PermissionError: " in capsys.readouterr().out


def test_a_serve_interface_without_write_is_not_asked_under_print(host, monkeypatch, capsys):
    """Before B's write=False: --print does not call serve_spec (it would write the run's serve
    config) and prints from the run id and eval.json; a real evaluation still calls it."""
    calls = []

    def serve_spec(run_dir, paths):
        calls.append(run_dir)
        (Path(run_dir) / "serve_config.json").write_bytes(b"{}")
        raise ServeSpecError("not served in this test")

    monkeypatch.setitem(sys.modules, "axis_qvla.train.serve", SimpleNamespace(serve_spec=serve_spec,
                                                                         ServeSpecError=ServeSpecError))
    eval_json = _eval_json(host / "runs", "x.cond", ["bare"])
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x.cond", "--print"]) == 0
    out = capsys.readouterr().out
    assert calls == [] and _files(host / "runs") == [eval_json]
    assert "takes no write=False" in out
    # eval.json records bare only: one run, named by the id, whatever the id's suffix suggests.
    assert [ln for ln in out.splitlines() if ln.startswith("# results file:")] == [
        f"# results file: {host / 'r1' / 'out_x.cond' / 'results.json'}"]
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x.cond", "--dry-run"]) == 2
    assert len(calls) == 1


def test_only_a_run_not_ready_is_pending_under_print(host, serve, capsys):
    serve.errors["x"] = ServeSpecError("x holds run y, whose directory is z")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", "--print"]) == 2
    assert "FAIL  --from-run x: x holds run y, whose directory is z" in capsys.readouterr().out
    serve.errors["x"] = RunNotReady("x cannot be served: it has not finished")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", "--print"]) == 0
    assert "does not resolve on this machine yet (--from-run x: x cannot be served" in capsys.readouterr().out


def test_without_run_not_ready_every_serve_refusal_is_pending(tmp_path, monkeypatch):
    def serve_spec(run_dir, paths, *, write=True):
        raise ServeSpecError("no run_spec.toml")

    monkeypatch.setitem(sys.modules, "axis_qvla.train.serve", SimpleNamespace(serve_spec=serve_spec,
                                                                         ServeSpecError=ServeSpecError))
    with pytest.raises(local_run.LocalRunError, match="no run_spec.toml") as e:
        local_run.resolve("x", _paths(tmp_path), write=False)
    assert e.value.pending


def test_print_of_a_run_not_ready_names_the_modes_its_eval_json_records(host, serve, capsys):
    _eval_json(host / "runs", "plain", ["bare", "cond"])
    assert main(["bench", "eval", "--host", "demo", "--from-run", "plain", "--print"]) == 0
    files = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("# results file:")]
    assert files == [f"# results file: {host / 'r1' / f'out_plain{s}' / 'results.json'}" for s in ("_b", "_c")]


def test_expected_arms_read_eval_json_then_the_directory_then_the_id(tmp_path):
    runs = tmp_path / "runs"
    assert local_run.expected_arms("a.cond", run_dir=runs / "a.cond") == ["a.cond_b", "a.cond_c"]   # no dir yet
    (runs / "a.cond").mkdir(parents=True)
    assert local_run.expected_arms("a.cond", run_dir=runs / "a.cond") == ["a.cond"]                 # dir, no eval.json
    _eval_json(runs, "a.cond", ["bare", "cond"])
    assert local_run.expected_arms("a.cond", "r", "conditioned", runs / "a.cond") == ["r_c"]
    _eval_json(runs, "a.cond", ["bare"])
    with pytest.raises(local_run.LocalRunError, match="--serving conditioned is not one of them"):
        local_run.expected_arms("a.cond", None, "conditioned", runs / "a.cond")
    _eval_json(runs, "a.cond", ["cond"])
    with pytest.raises(local_run.LocalRunError, match="bare first, then cond"):
        local_run.expected_arms("a.cond", run_dir=runs / "a.cond")


# --- seeds -----------------------------------------------------------------------------------------

def test_seed_flags_fill_a_seed_the_run_does_not_record(host, serve, monkeypatch):
    calls = _record(monkeypatch, host)
    serve.runs["x"] = _run("x", pretrain_seed=None)
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", "--ft-seed", "7", "--pretrain-seed", "42",
                 *bench_inputs(host, monkeypatch)]) == 0
    assert calls[-1] == {"finish": ("x", {"ft_seed": 7, "pretrain_seed": 42})}


@pytest.mark.parametrize("flag,value,want", [("--pretrain-seed", "43", "the run records pretrain seed 42; drop "
                                                                     "--pretrain-seed or pass 42"),
                                             ("--ft-seed", "8", "the run records fine-tune seed 7; drop --ft-seed "
                                                                "or pass 7")])
def test_a_seed_flag_that_disagrees_with_the_run_is_refused(host, serve, capsys, flag, value, want):
    serve.runs["x"] = _run("x")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", flag, value, "--print"]) == 2
    assert f"FAIL  --from-run x: {want}" in capsys.readouterr().out


def test_a_seed_flag_may_repeat_the_runs_own(host, serve, capsys):
    serve.runs["x"] = _run("x")
    assert main(["bench", "eval", "--host", "demo", "--from-run", "x", "--pretrain-seed", "42", "--print"]) == 0
    assert "# from run: x (fine-tune seed 7, pretraining seed 42)" in capsys.readouterr().out


# --- the default run name and an unreadable results map -------------------------------------------

def test_a_checkpoint_id_without_run_is_filed_under_the_id(host, capsys):
    """M4: the default run is the id, never `pretrained_model_bare` shared by every LeRobot copy."""
    lerobot_host(host)
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", "smolvla/ft/mvft_smolvla_bc", "--print"]) == 0
    out = capsys.readouterr().out
    assert f"# results file: {host / 'r1' / 'out_smolvla.ft.mvft_smolvla_bc_bare' / 'results.json'}" in out


def test_an_unreadable_results_map_is_reported_even_with_checkpoint(host, monkeypatch, capsys):
    """N2: --checkpoint lets an id outside the map through; a map that cannot be read is not that."""
    from axis_qvla.paper import cli as paper_cli

    (host / "cells.json").write_bytes(b"not json")
    monkeypatch.setattr(paper_cli, "DEFAULT_CELLS", host / "cells.json")
    lerobot_host(host)
    assert main(["bench", "eval", "--host", "demo", "--checkpoint-id", "smolvla/ft/y", "--checkpoint", str(host / "ck"),
                 "--run", "s", "--dry-run"]) == 2
    assert "the results map cannot be read" in capsys.readouterr().out
