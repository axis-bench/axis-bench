"""A run `qvla train` trained on this machine, as what `bench eval --from-run <run id>` serves.

Why: a method author's path ends in a benchmark table -- `qvla train finetune --from <pretrain run>`
then `qvla bench eval --from-run <finetune run id>` -- and serving a pi0.5 checkpoint used to take a
Python call to render its serve config. The training package owns the run
directory's layout, so it says what to serve: `axis_qvla.train.serve.serve_spec` returns the backbone's
reference server, the checkpoint, the server's config, the serving modes the run was trained for
(bare; plus cond for a `--cond` fine-tune, its eval.json `serving`) and the seeds. This module only
turns that into the targets `bench eval` evaluates, one per mode, bare first, and refuses what the
benchmark does not evaluate: a pretraining run (the protocol scores a checkpoint fine-tuned on its
24 held-out tasks) and a mode the benchmark's servers would serve differently from how the run was
trained.

Each mode is its own run: two modes of one checkpoint are filed as `<run>_b` and `<run>_c`, the
paper's spelling (cells.json r1bct_b / r1bct_c), whichever of them `--serving` narrows to, so a
later full evaluation finds the same directories. A run evaluated one way only is filed as `<run>`.

`--print` writes nothing: it asks `serve_spec(..., write=False)`, and a serve interface without that
keyword is not asked at all -- the command is printed from the run id and the run's eval.json.

Core tier: stdlib only; axis_qvla.train.serve is imported inside `resolve`, so this checkout works (and
says what is missing) before the training package that provides it is there.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

from axis_qvla.bench.backends import BACKENDS, DEFAULT_BACKEND, QUALITY_TAG

SERVE_MODULE = "axis_qvla.train.serve"
# How each table row is served by `bench eval --serving`, and the server fields that serving
# implies (cells.json `serve.fields`): a mode the run records otherwise is refused, not re-served.
MODE_SERVING = {"bare": ("bare", {}), "cond": ("conditioned", {"quality": str(QUALITY_TAG)})}
MODE_SUFFIX = {"bare": "_b", "cond": "_c"}
EVAL_FILE = "eval.json"         # the run's serving modes, as `qvla train finetune` records them


class LocalRunError(ValueError):
    """The run cannot be evaluated here as asked; the message names the fix. `pending` is True when
    that may change without another command line: the run is not there or not finished yet (the
    training package's RunNotReady), or this checkout lacks the training package's serve interface,
    or that interface cannot resolve a run without writing into it (`--print` still prints the
    command)."""

    def __init__(self, message: str, *, pending: bool = False):
        super().__init__(message)
        self.pending = pending


@dataclasses.dataclass(frozen=True)
class LocalMode:
    name: str          # "bare" | "cond": the results table row
    serving: str       # `bench eval --serving`: "bare" | "conditioned"


@dataclasses.dataclass(frozen=True)
class LocalRun:
    id: str
    backend: str
    checkpoint: Path
    config: str | None            # LeRobot: the policy name the server is recorded under; pi0.5: None
    serve_config: bytes | None    # pi0.5: the server's config, rendered by the training package
    modes: tuple[LocalMode, ...]  # bare first
    ft_seed: int | None
    pretrain_seed: int | None
    pretrain: str | None = None   # the pretraining run it was fine-tuned from, when trained here

    def arm(self, mode: LocalMode, run: str | None = None) -> str:
        """The run name `mode` is filed under: `run` (default the run id), suffixed _b/_c when the
        run is evaluated more than one way."""
        base = run or self.id
        return base + MODE_SUFFIX[mode.name] if len(self.modes) > 1 else base


def _serve_module(run_id: str):
    import importlib

    try:
        return importlib.import_module(SERVE_MODULE)
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("axis_qvla.train"):
            raise
        detail = "" if exc.name == SERVE_MODULE else f"; {exc.name} cannot be imported"
        raise LocalRunError(f"--from-run {run_id} needs the training package's serve interface: update this "
                            f"checkout ({SERVE_MODULE} is missing{detail})", pending=True) from None


def _modes(run_id: str, spec) -> tuple[LocalMode, ...]:
    out = []
    for m in spec.modes:
        want = MODE_SERVING.get(m.name)
        if want is None or (m.serving, dict(m.fields)) != want:
            raise LocalRunError(f"--from-run {run_id}: the run is to be evaluated {m.name!r} served {m.serving!r} "
                                f"with fields {dict(m.fields)}; this benchmark serves "
                                + "; ".join(f"{k} as --serving {s} with fields {f}" for k, (s, f) in MODE_SERVING.items())
                                + ". Update this checkout")
        out.append(LocalMode(m.name, m.serving))
    if not out:
        raise LocalRunError(f"--from-run {run_id}: the run records no serving mode to evaluate")
    return tuple(out)


def check_run_id(run_id: str) -> None:
    """Raise LocalRunError unless `run_id` names one directory under the runs root: it is joined to
    that root and names the results directories, so a `/` or `..` would read and write elsewhere."""
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise LocalRunError(f"--from-run {run_id!r} is not a run id (one path component under the runs root, "
                            f"as `qvla train` names it: <backbone>.finetune.<name>[.cond])")


def _takes_write(fn) -> bool:
    """Whether the installed serve_spec accepts `write=` (`write=False` writes nothing)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is p.VAR_KEYWORD or (p.name == "write" and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD))
               for p in params)


def _not_ready(serve, exc: Exception) -> bool:
    """A refusal that may lift once the run is trained: the module's RunNotReady, or, for a module
    that does not tell a missing or unfinished run from any other refusal, every ServeSpecError."""
    cls = getattr(serve, "RunNotReady", None)
    return isinstance(exc, cls) if isinstance(cls, type) else True


def resolve(run_id: str, paths, *, write: bool = True) -> LocalRun:
    """What `bench eval --from-run run_id` serves. `paths` is an axis_qvla.core.paths.Paths. `write=False`
    (`--print`) writes nothing under the run's directory."""
    check_run_id(run_id)
    serve = _serve_module(run_id)
    from axis_qvla.train import ids

    kwargs = {}
    if not write:
        if not _takes_write(serve.serve_spec):
            raise LocalRunError(f"this checkout's {SERVE_MODULE}.serve_spec cannot resolve a run without writing its "
                                f"server config into the run's directory (it takes no write=False), so --print does "
                                f"not call it; update this checkout", pending=True)
        kwargs["write"] = False
    try:
        spec = serve.serve_spec(ids.run_root(paths, run_id), paths, **kwargs)
    except serve.ServeSpecError as exc:
        raise LocalRunError(f"--from-run {run_id}: {exc}", pending=_not_ready(serve, exc)) from None
    except OSError as exc:
        raise LocalRunError(f"--from-run {run_id}: the run cannot be read ({type(exc).__name__}: {exc})") from None
    if spec.stage != "finetune":
        raise LocalRunError(f"--from-run {run_id} is a {spec.stage} run; the benchmark evaluates a checkpoint "
                            f"fine-tuned on its 24 held-out tasks: fine-tune it first "
                            f"(`qvla train finetune --from {run_id}`), then evaluate that run")
    if spec.backend not in BACKENDS:
        raise LocalRunError(f"--from-run {run_id} is served by {spec.backend!r}, which is not a reference server "
                            f"of this checkout ({', '.join(BACKENDS)}): update this checkout")
    data = None
    if spec.backend == DEFAULT_BACKEND:
        data = getattr(spec, "config_bytes", None)
        if data is None:
            if spec.config is None:
                raise LocalRunError(f"--from-run {run_id}: the training package rendered no server config for this "
                                    f"pi0.5 run; update this checkout")
            try:
                data = Path(spec.config).read_bytes()
            except OSError as exc:
                raise LocalRunError(f"--from-run {run_id}: its server config {spec.config} cannot be read "
                                    f"({type(exc).__name__}: {exc})") from None
    # A LeRobot server loads the checkpoint alone; the policy name stands for --config in the run's
    # record (the paper's eval_commands keep their own strings, cells.json).
    config = None if spec.backend == DEFAULT_BACKEND else spec.backend
    return LocalRun(spec.run_id, spec.backend, Path(spec.checkpoint), config, data, _modes(run_id, spec),
                    spec.ft_seed, spec.pretrain_seed, spec.pretrain)


def seeds(lr: LocalRun, ft_seed: int | None, pretrain_seed: int | None) -> tuple[int | None, int | None]:
    """The seeds the results file records: the run's own. `--ft-seed`/`--pretrain-seed` fill a seed
    the run does not record (a fine-tune of published pretrain weights has no pretraining seed) and
    are refused when they disagree with one it does."""
    out = []
    for label, flag, recorded, given in (("fine-tune", "--ft-seed", lr.ft_seed, ft_seed),
                                         ("pretrain", "--pretrain-seed", lr.pretrain_seed, pretrain_seed)):
        if given is not None and recorded is not None and given != recorded:
            raise LocalRunError(f"--from-run {lr.id}: the run records {label} seed {recorded}; drop {flag} "
                                f"or pass {recorded}")
        out.append(recorded if recorded is not None else given)
    if out[1] is not None and out[0] is None:
        raise LocalRunError(f"--from-run {lr.id} records no fine-tune seed: pass --ft-seed with --pretrain-seed "
                            f"(a results file records the pretraining seed with the fine-tune seed)")
    return out[0], out[1]


def select(lr: LocalRun, serving: str | None) -> tuple[LocalMode, ...]:
    """The modes to evaluate: every mode the run records, or the one `--serving` narrows to."""
    if serving is None:
        return lr.modes
    hit = tuple(m for m in lr.modes if m.serving == serving)
    if not hit:
        raise LocalRunError(f"--from-run {lr.id} is evaluated {' and '.join(m.name for m in lr.modes)} (its eval.json "
                            f"`serving`); --serving {serving} is not one of them")
    return hit


def _recorded_modes(run_id: str, run_dir: Path) -> list[str] | None:
    """The serving modes the run's eval.json records; ["bare"] for a run directory without one (the
    training package's default); None when there is no run directory yet."""
    path = run_dir / EVAL_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        try:
            run_dir.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as exc:
            raise LocalRunError(f"--from-run {run_id}: {run_dir} cannot be read ({type(exc).__name__}: {exc})") from None
        return ["bare"]
    except (OSError, ValueError) as exc:
        raise LocalRunError(f"--from-run {run_id}: {path} cannot be read ({type(exc).__name__}: {exc})") from None
    serving = doc.get("serving") if isinstance(doc, dict) else None
    if not (isinstance(serving, list) and serving and serving[0] == "bare" and len(set(serving)) == len(serving)
            and all(isinstance(m, str) and m in MODE_SERVING for m in serving)):
        raise LocalRunError(f"--from-run {run_id}: {path} records serving {serving!r}; this benchmark evaluates "
                            f"bare first, then cond")
    return serving


def expected_arms(run_id: str, run: str | None = None, serving: str | None = None,
                  run_dir: Path | None = None) -> list[str]:
    """The run names `--from-run run_id` files under, for `--print` before the run resolves: the
    modes the run's eval.json records (bare only for a run directory without one); for a run with no
    directory under `run_dir` yet, read from the id (`qvla train finetune --cond` names a run
    evaluated bare and cond `<...>.cond`)."""
    base = run or run_id
    modes = _recorded_modes(run_id, run_dir) if run_dir is not None else None
    if modes is None:
        if not run_id.endswith(".cond"):
            return [base]
        modes = list(MODE_SERVING)
    hit = [m for m in modes if serving in (None, MODE_SERVING[m][0])]
    if not hit:
        raise LocalRunError(f"--from-run {run_id} is evaluated {' and '.join(modes)} (its eval.json `serving`); "
                            f"--serving {serving} is not one of them")
    return [base + MODE_SUFFIX[m] if len(modes) > 1 else base for m in hit]
