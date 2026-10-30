"""Run a page's shell blocks as a reader would, and find the files a public checkout has.

`run_page` runs the blocks of the given tags in page order, in a scratch directory that shows the
public layout: links to the top-level entries a page may name (VIEW) and to the public root files
it reads (PUBLIC_FILES). A block writes only new paths; a write under a link would reach the checkout.
On PATH, `python`, `python3` and `python3.<minor>` are this interpreter and `qvla` is the `qvla` script
of its environment (`pip install -e .`) or, without one, this checkout's CLI module; the rest of that
environment's scripts follow. These shims log every command a block starts into the run's RAN log,
not what those start in turn (tests/docs/coverage.py). Nothing else of this shell's environment
reaches a block but the GL backend (PASS_ENV), and its stdin is empty. A block runs under BASH
(`-e -u -o pipefail`, `inherit_errexit`), after tests/docs/shell.py has refused every form those
leave unchecked, in a process group of its own, killed when it ends; a `background=1` block (a
server, tests/docs/servers.py) is killed when the page's run ends, and fails the page once it has
exited, checked after every later block. A block's output goes to a file, never to a pipe a server
could hold open. The first failing block ends the run.

`public_file` finds a file at the checkout root (README, LICENSE, the CI workflow), or among the
release templates where a checkout keeps them there.

Core tier: stdlib only.
"""
from __future__ import annotations

import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys

import axis_qvla.cli as qvla_cli
from tests.docs import servers as server_rules
from tests.docs.blocks import blocks, commands, output_problems, shell_blocks
from tests.docs.coverage import RAN, needed, ran
from tests.docs.shell import refused

REPO = pathlib.Path(__file__).resolve().parents[2]
VIEW = ("axis", "axis_qvla", "conf", "docs", "experiments", "tests", "third_party")
PASS_ENV = ("MUJOCO_GL", "PYOPENGL_PLATFORM")
PUBLIC_FILES = ("qvla.example.toml",)
_PUBLIC_DIRS = (("conf", "release", "public", "overrides"), ("conf", "release", "public"), ())
BASH = ("bash", "-O", "inherit_errexit", "-euo", "pipefail", "-c")


def public_file(rel: str) -> pathlib.Path | None:
    """`rel` (a path of the public checkout) in this checkout, or None."""
    return next((REPO.joinpath(*d, rel) for d in _PUBLIC_DIRS if REPO.joinpath(*d, rel).is_file()), None)


def ci_workflow() -> str:
    return public_file(".github/workflows/ci.yml").read_text(encoding="utf-8")


def ci_job(name: str, text: str | None = None) -> str:
    """The text of job `name` of the CI workflow (or of the workflow `text`), up to the next job."""
    m = re.search(rf"^  {re.escape(name)}:[ \t]*\n(.*?)(?=^  [\w-]+:[ \t]*$|\Z)",
                  ci_workflow() if text is None else text, re.M | re.S)
    return m.group(1) if m else ""


def pages() -> list[pathlib.Path]:
    """The pages whose shell blocks run: Install, then every tutorial."""
    top = [REPO / "docs" / "install.md"] if (REPO / "docs" / "install.md").is_file() else []
    return top + sorted((REPO / "docs" / "tutorials").glob("*.md"))


def pages_with(tag: str) -> list[pathlib.Path]:
    return [p for p in pages() if any(b.shell and b.tag == tag for b in blocks(p.read_text(encoding="utf-8")))]


TOP_PAGES = ("index.md", "install.md")    # the top-level docs/*.md pages


def shipped_pages() -> list[tuple[str, pathlib.Path]]:
    """(published path, file) of every documentation page."""
    out = [("README.md", public_file("README.md"))]
    out += [(f"docs/{n}", REPO / "docs" / n) for n in TOP_PAGES if (REPO / "docs" / n).is_file()]
    for d in ("tutorials", "reference"):
        out += [(f"docs/{d}/{p.name}", p) for p in sorted((REPO / "docs" / d).glob("*.md"))]
    return out


def _qvla_line() -> str:
    installed = shutil.which("qvla")
    if installed and pathlib.Path(installed).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()):
        return f'exec "{installed}" "$@"'
    return f'exec "{sys.executable}" -m {qvla_cli.__name__} "$@"'


def scratch(root: pathlib.Path) -> tuple[pathlib.Path, dict[str, str]]:
    """(the directory the blocks run in, their environment), both made under `root`."""
    work, bin_, home = root / "work", root / "bin", root / "home"
    for d in (work, bin_, home):
        d.mkdir(parents=True)
    for name in VIEW:
        if (REPO / name).exists():
            (work / name).symlink_to(REPO / name)
    for name in PUBLIC_FILES:
        if (found := public_file(name)) is not None:
            (work / name).symlink_to(found)
    log = shlex.quote(str(root / RAN))
    python = f'exec "{sys.executable}" "$@"'
    for name, line in (("qvla", _qvla_line()), ("python", python), ("python3", python),
                       (f"python3.{sys.version_info.minor}", python)):
        (bin_ / name).write_text(f'#!/bin/sh\n[ -n "${{QVLA_DOCS_NESTED:-}}" ] || printf "%s\\n" "{name} $*" >> {log}\n'
                                 f"QVLA_DOCS_NESTED=1\nexport QVLA_DOCS_NESTED\n{line}\n")
        (bin_ / name).chmod(0o755)
    path = (str(bin_), str(pathlib.Path(sys.executable).parent), "/usr/bin", "/bin")
    env = {"PATH": os.pathsep.join(dict.fromkeys(path)), "HOME": str(home),
           "PYTHONPATH": str(REPO), "LANG": "C.UTF-8"}
    env.update({k: os.environ[k] for k in PASS_ENV if k in os.environ})
    return work, env


def _start(text: str, work: pathlib.Path, env: dict[str, str], fh) -> subprocess.Popen:
    return subprocess.Popen([*BASH, text], cwd=work, env=env, stdin=subprocess.DEVNULL,
                            stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)       # whatever the block left running in its group
    except ProcessLookupError:
        pass
    proc.wait()


def run_block(text: str, work: pathlib.Path, env: dict[str, str], log: pathlib.Path,
              timeout: int) -> tuple[int | None, str]:
    """(exit status, None when killed at `timeout`; the block's combined output)."""
    with log.open("w") as fh:
        proc = _start(text, work, env, fh)
        try:
            code: int | None = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
        _stop(proc)
    return code, log.read_text(encoding="utf-8", errors="replace")


def run_page(page: pathlib.Path, tags: tuple[str, ...], root: pathlib.Path) -> list[str]:
    """Run `page`'s shell blocks of `tags` in order; every problem, [] when all ran as the page says
    and ran every at-scale command they name."""
    work, env = scratch(root)
    bs = blocks(page.read_text(encoding="utf-8"))
    bad = server_rules.problems(shell_blocks(bs))
    if bad:
        return [f"{page.name}: {p}" for p in bad]
    servers: list[tuple[int, subprocess.Popen, pathlib.Path]] = []
    try:
        found = _run(page, bs, tags, root, work, env, servers)
    finally:
        found_after = _exited(page, servers)
        for _, proc, _ in servers:
            _stop(proc)
    found += found_after
    if not found:
        named = {k for b in bs if b.shell and b.tag in tags and not b.background for k in commands(b.text)}
        scale = needed([b.text for b in bs if b.shell and b.tag in ("gpu", "setup")]) & named
        log = root / RAN
        got = ran(log.read_text(encoding="utf-8") if log.is_file() else "")
        found = [f"{page.name}: `{k}` is named by a `{'/'.join(tags)}` block but never ran"
                 for k in sorted(scale - got)]
    return found


def _exited(page: pathlib.Path, servers: list) -> list[str]:
    """A problem for each server that has exited, which is then no longer one of `servers`."""
    out = []
    for entry in list(servers):
        line, proc, log = entry
        code = proc.poll()
        if code is not None:
            servers.remove(entry)
            _stop(proc)
            tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
            out.append(f"{page.name}:{line}: the background block exited ({code}) while the page ran\n{tail}")
    return out


def _run(page, bs, tags, root, work, env, servers) -> list[str]:
    """Run the blocks of `tags` in order, starting the background ones into `servers`; the problems
    up to the first failing block."""
    found: list[str] = []
    for k, b in enumerate(bs):
        if not (b.shell and b.tag in tags):
            continue
        bad = refused(b.text)
        if bad:
            return found + [f"{page.name}:{b.line + n}: `{cur}`: {why}" for n, cur, why in bad]
        log = root / f"block-{b.line}.log"
        if b.background:                         # its shims log nothing: a server is never coverage
            with log.open("w") as fh:
                servers.append((b.line, _start(b.text, work, {**env, "QVLA_DOCS_NESTED": "1"}, fh), log))
            continue
        code, out = run_block(b.text, work, env, log, b.timeout)
        dead = _exited(page, servers)
        if code != 0:
            why = f"timed out after {b.timeout} s" if code is None else f"exit {code}"
            return found + [f"{page.name}:{b.line}: {why}\n{out[-4000:]}"] + dead
        if dead:
            return found + dead
        after = bs[k + 1] if k + 1 < len(bs) else None
        if after is not None and after.tag == "output":
            found += [f"{page.name}:{after.line}: not printed by the block at line {b.line}: {w!r}\n{out[-2000:]}"
                      for w in output_problems(after.text, out)]
    return found
