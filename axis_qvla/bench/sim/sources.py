"""Which files of this checkout a served-policy rollout runs, so a run manifest can digest every one.

A hand-kept tuple of "key sources" digested seven files and missed the whole render half
(rollout_render, render_scene, layouts, layout_plan, ...), so two runs on different render code
recorded identical source digests. A list drifts the moment someone adds an import; the import
graph cannot. So the set is COMPUTED: the static import closure of the three entry points the
driver runs (the driver, the queue worker and the rollout CLI), function-local imports included,
plus every package `__init__` those imports execute.

No server script is in it. Every backend's server runs from the server checkout (`--openpi-dir`,
the cwd `start_server` gives it), which for SmolVLA and X-VLA `bench eval` points at
`smolvla_bench` -- on a worker box, can be a different checkout from the driver's. A server script digested HERE
would be a plausible digest of a file that did not run, so the run manifest digests the launched
script where it runs (axis_qvla.bench.sim.campaign.launched_server_script).

Static on purpose: the closure is read from the AST, never by importing, because the driver
runs on any interpreter and must not pull in the render stack just to describe it. Only imports
under this package's root are followed; the root is read from `__name__`, not spelled, so the
closure is computed whatever the root is called.

Core tier: stdlib only.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT_PKG = __name__.split(".")[0]
ENTRY_MODULES = tuple(f"{__package__}.{m}" for m in ("campaign", "queue_worker", "cli"))


def _module_file(root: Path, module: str) -> Path | None:
    rel = Path(*module.split("."))
    for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _imported(path: Path, module: str) -> set[str]:
    """Dotted names one module imports, anywhere in its body, resolved to absolute names.

    `from pkg import name` yields both `pkg` and `pkg.name`, since `name` may be a submodule;
    the caller keeps only names that are files. A file that does not parse contributes no
    edges but is still digested: a broken source on the rollout path must show in the record,
    not stop the driver from writing one.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError):
        return set()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                stem = f"{base}.{node.module}" if node.module else base
            else:
                stem = node.module or ""
            out.add(stem)
            out.update(f"{stem}.{a.name}" for a in node.names if a.name != "*")
    return {m for m in out if m == ROOT_PKG or m.startswith(ROOT_PKG + ".")}


def _with_parents(module: str) -> list[str]:
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def rollout_sources(root: Path) -> tuple[str, ...]:
    """Repo-relative POSIX paths of every file of `root` on the rollout path, sorted.

    The same for every backend: the backends differ only in the server, which is not run from here.
    """
    root = Path(root)
    seen: set[str] = set()
    files: set[Path] = set()
    todo = list(ENTRY_MODULES)
    while todo:
        mod = todo.pop()
        for name in _with_parents(mod):
            if name in seen:
                continue
            seen.add(name)
            path = _module_file(root, name)
            if path is None:
                continue
            files.add(path)
            todo.extend(_imported(path, name))
    return tuple(sorted(p.relative_to(root).as_posix() for p in files))
