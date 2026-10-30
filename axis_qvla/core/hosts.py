"""A machine's shape, declared in one reviewable file per host.

Core tier -- stdlib only, because both venvs import this. Every machine-specific value (roots,
venvs, queues, ports, proxy, secrets) is declared in the profile, and the cache env vars are
derived from `roots.cache`, never written out by hand.
"""
from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOSTS_DIR = REPO_ROOT / "conf" / "hosts"

LAUNCHERS = ("slurm", "local")
REQUIRED_ROOTS = ("corpus", "cache", "scratch", "runs")
# The closed venv vocabulary. A spec names one of these; a host maps each it has to a path. Kept
# here so a host profile is validated against the same set a spec is: a venv key no spec could
# name is refused when the profile loads, not when a spec asks for it.
VENV_KEYS = ("offline", "online", "eval", "render", "grm", "libero_plus")
# `eval` is the evaluation environment (MuJoCo, the simulator's renderer, the policy-server client):
# `bench eval`, its driver and workers, and `bench check-server` run under it. A profile may also
# call it `render`; the two keys are one environment, so a profile that declares only one of them
# has it read under both. `qvla env build render` is a different environment (the real-robot renderer's).
EVAL_VENV, LEGACY_EVAL_VENV = "eval", "render"

# The secrets a profile that declares none is assumed to need. The `qvla` CLI reads no secrets
# file; only a host check reads this list.
DEFAULT_REQUIRED_SECRETS = ("hf.token",)


class HostProfileError(RuntimeError):
    """A profile is missing, malformed, or names an unknown launcher."""


@dataclasses.dataclass(frozen=True)
class Queue:
    """A named (account, partition[, max_time]) on ONE machine.

    A machine with several queues is one profile with several queues, not several profiles: its
    queues share its roots, venvs and modules, so a profile per queue would copy everything that
    describes the machine, and the copies would drift apart.
    """

    name: str
    partition: str | None = None
    account: str | None = None
    # SLURM's wall-clock cap for this queue, "HH:MM:SS" or "D-HH:MM:SS". Optional; when set,
    # an experiment asking for longer is refused at submit time instead of being killed
    # mid-run hours later.
    max_time: str | None = None


@dataclasses.dataclass(frozen=True)
class HostProfile:
    name: str
    launcher: str
    roots: dict[str, Path]
    venvs: dict[str, Path]
    partition: str | None = None
    account: str | None = None
    # Wall-clock cap of the DEFAULT queue: the default queue can have one too, and an experiment
    # asking for longer is refused at submit time.
    max_time: str | None = None
    gpus: int = 0
    modules: tuple[str, ...] = ()
    ports: tuple[int, ...] = ()
    # A local proxy (Clash/mihomo-style API) this machine must switch to another routing mode
    # before it can reach the Hugging Face Hub: `{"api": "http://127.0.0.1:9090",
    # "transfer_mode": "global"}`. Empty on a machine that reaches the Hub directly.
    #
    # A machine fact, so it is declared with the machine. A proxy whose rules route the Hub's
    # host directly can open the CONNECT tunnel and then have TLS reset, which reads as no network
    # at all on a machine whose network works; nothing diagnoses that automatically.
    proxy: dict[str, str] = dataclasses.field(default_factory=dict)
    # "section.key" names of the secrets this machine needs; see DEFAULT_REQUIRED_SECRETS.
    requires_secrets: tuple[str, ...] = DEFAULT_REQUIRED_SECRETS
    # Named alternatives to the top-level partition/account. Empty on a host with one queue.
    queues: dict[str, Queue] = dataclasses.field(default_factory=dict)
    # The file this profile was actually read from (set by load_host; None on a profile built by
    # hand, e.g. in a test). A profile comes from ./qvla.toml, an explicit --host path or a named
    # profile directory, so the name alone does not say which file: an error naming the file to
    # edit reads this (`source`).
    path: Path | None = None

    def source(self) -> str:
        """The file to tell someone to edit, or, for a profile built by hand, its name."""
        return str(self.path) if self.path is not None else f"the {self.name!r} host profile"

    def queue(self, name: str | None) -> Queue:
        """Resolve a queue by name; `None` means the profile's top-level partition/account.

        An unknown name is refused rather than falling back to the default. Silently
        defaulting would put a job on the wrong partition -- where it may sit for hours behind
        someone else's work, or be charged to the wrong account -- and nothing in the job's
        output would say so.
        """
        if name is None:
            return Queue(
                name="default", partition=self.partition, account=self.account,
                max_time=self.max_time,
            )
        if name not in self.queues:
            raise HostProfileError(
                f"host {self.name!r} has no queue {name!r} "
                f"(has: {', '.join(sorted(self.queues)) or 'none'}). Add it under [queues] "
                f"in {self.source()} rather than editing the experiment to match a machine."
            )
        q = self.queues[name]
        # UNSET FIELDS INHERIT THE HOST DEFAULT. A queue that overrides only the partition
        # (`long = { partition = "gpu" }`) must keep the host's account -- emitting no
        # --account at all sends the job to the submitter's default account, which on a
        # shared cluster is usually not the one with the GPU allocation.
        return Queue(
            name=q.name,
            partition=q.partition if q.partition is not None else self.partition,
            account=q.account if q.account is not None else self.account,
            max_time=q.max_time,
        )

    @property
    def env(self) -> dict[str, str]:
        """Cache env vars derived from `roots.cache`, never hand-written."""
        cache = self.roots["cache"]
        return {
            "HF_HOME": str(cache / "hf"),
            "HF_XET_CACHE": str(cache / "hf" / "xet"),
            "OPENPI_DATA_HOME": str(cache / "openpi"),
            "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
            "AXIS_DATALOADER_ROOT": str(cache / "dataloader"),
        }


def _hosts_dir(explicit: Path | None) -> Path:
    """Explicit argument wins; otherwise $AXIS_HOSTS_DIR; otherwise the repo default.

    The env override lets a test (or someone trying a config change) point at a temp
    directory without editing the repo, without disturbing callers that pass `hosts_dir=`.
    """
    if explicit is not None:
        return explicit
    override = os.environ.get("AXIS_HOSTS_DIR")
    return Path(override) if override else HOSTS_DIR


def available_hosts(hosts_dir: Path | None = None) -> list[str]:
    """Names of every host profile, sorted.

    `os.scandir`, not `Path.glob`: `glob`
    swallows `PermissionError` and would report an unreadable conf directory as ZERO profiles. This
    function's output is quoted in the error message every other failure path prints
    ("Available: (none)"), so the swallow would actively misdirect -- it reads as a typo in the
    host name when the real problem is the directory. Only `FileNotFoundError` means "none yet".
    """
    d = _hosts_dir(hosts_dir)
    names: list[str] = []
    try:
        with os.scandir(d) as entries:
            for entry in entries:
                if entry.name.endswith(".toml") and entry.is_file():
                    names.append(entry.name[: -len(".toml")])
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise HostProfileError(
            f"cannot list {d}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable (e.g. not mode 000)."
        ) from None
    return sorted(names)


def load_host(name: str | None = None, *, hosts_dir: Path | None = None) -> HostProfile:
    d = _hosts_dir(hosts_dir)
    name = name or os.environ.get("AXIS_HOST")
    if not name:
        raise HostProfileError(
            "AXIS_HOST is not set and no host name was given. "
            f"Available profiles: {', '.join(available_hosts(d)) or '(none)'}"
        )
    path = d / f"{name}.toml"
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise HostProfileError(
            f"cannot access {path}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable (e.g. not mode 000)."
        ) from None
    if not is_file:
        raise HostProfileError(
            f"no host profile {name!r} in {d}. "
            f"Available: {', '.join(available_hosts(d)) or '(none)'}"
        )
    return _parse(name, _read_toml(path), path)


def _read_toml(path: Path) -> dict:
    """Read and parse a profile; an unreadable file or invalid TOML raises HostProfileError.

    HostProfileError's contract covers a "malformed" profile, and its callers catch only that type,
    so an unclosed [section] must not escape as a raw tomllib.TOMLDecodeError -- those callers exist
    to give actionable guidance on a fresh machine. The file's own text never enters the message:
    these errors land in world-readable job logs.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HostProfileError(
            f"cannot read {path}: {exc.__class__.__name__}. Check that every parent "
            f"directory is traversable (e.g. not mode 000)."
        ) from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        raise HostProfileError(
            f"{path} is not valid TOML. It is hand-edited; check for a typo (e.g. an "
            f"unclosed [section] header)."
        ) from None


def _parse(name: str, raw: dict, path: Path) -> HostProfile:
    launcher = raw.get("launcher", "local")
    if launcher not in LAUNCHERS:
        raise HostProfileError(
            f"{path}: launcher must be one of {LAUNCHERS}, got {launcher!r}"
        )

    roots_raw = raw.get("roots") or {}
    missing = [k for k in REQUIRED_ROOTS if k not in roots_raw]
    if missing:
        raise HostProfileError(f"{path}: [roots] is missing {', '.join(missing)}")

    roots: dict[str, Path] = {}
    for key, value in roots_raw.items():
        p = Path(str(value))
        if not p.is_absolute():
            raise HostProfileError(
                f"{path}: roots.{key} must be absolute, got {value!r}"
            )
        roots[key] = p

    venvs: dict[str, Path] = {}
    for key, value in (raw.get("venv") or {}).items():
        if key not in VENV_KEYS:
            raise HostProfileError(
                f"{path}: [venv] key {key!r} is not one of {VENV_KEYS}; a spec could never name it"
            )
        p = Path(str(value))
        venvs[key] = p if p.is_absolute() else REPO_ROOT / p
    for have, alias in ((LEGACY_EVAL_VENV, EVAL_VENV), (EVAL_VENV, LEGACY_EVAL_VENV)):
        if have in venvs and alias not in venvs:
            venvs[alias] = venvs[have]

    return HostProfile(
        name=name,
        launcher=launcher,
        roots=roots,
        venvs=venvs,
        partition=raw.get("partition"),
        account=raw.get("account"),
        max_time=raw.get("max_time"),
        gpus=int(raw.get("gpus", 0)),
        modules=tuple(raw.get("modules", ())),
        ports=tuple(int(x) for x in ((raw.get("ports") or {}).get("allowed", ()))),
        requires_secrets=_parse_required_secrets(raw, path),
        queues=_parse_queues(raw, path),
        proxy=_parse_proxy(raw, path),
        path=path,
    )


def _parse_proxy(raw: dict, path: Path) -> dict[str, str]:
    """`[proxy]` with `api` and `transfer_mode`, or nothing.

    Both keys are required together: an api with no mode says nothing about what to switch to,
    and a mode with no api cannot be applied. Half a declaration would fail at transfer time on
    the remote machine, the least convenient place to discover it.
    """
    entry = raw.get("proxy")
    if not entry:
        return {}
    if not isinstance(entry, dict):
        raise HostProfileError(f"{path}: [proxy] must be a table with api and transfer_mode")
    missing = [k for k in ("api", "transfer_mode") if not entry.get(k)]
    if missing:
        raise HostProfileError(
            f"{path}: [proxy] is missing {', '.join(missing)}. Declare both, or omit [proxy] "
            f"entirely on a machine that reaches HuggingFace directly."
        )
    return {"api": str(entry["api"]), "transfer_mode": str(entry["transfer_mode"])}


def _parse_queues(raw: dict, path: Path) -> dict[str, Queue]:
    """`[queues.<name>]` tables, each carrying a partition and/or an account.

    A queue naming neither is refused: it would resolve to the host default while claiming to
    be a distinct queue, so a job asking for one account would quietly run on another and be
    charged to the wrong account.
    """
    out: dict[str, Queue] = {}
    for qname, entry in (raw.get("queues") or {}).items():
        if not isinstance(entry, dict):
            raise HostProfileError(
                f"{path}: queues.{qname} must be a table with partition and/or account"
            )
        partition = entry.get("partition")
        account = entry.get("account")
        if partition is None and account is None:
            raise HostProfileError(
                f"{path}: queues.{qname} sets neither partition nor account, so it is "
                f"indistinguishable from the host default while claiming to be a queue"
            )
        out[str(qname)] = Queue(
            name=str(qname),
            partition=str(partition) if partition is not None else None,
            account=str(account) if account is not None else None,
            max_time=str(entry["max_time"]) if entry.get("max_time") is not None else None,
        )
    return out


def slurm_seconds(spec: str) -> int:
    """Parse a SLURM wall-clock string to seconds.

    Accepts the forms this repo's specs actually use: "HH:MM:SS", "MM:SS", "D-HH:MM:SS".
    Raises ValueError on anything else rather than guessing -- a misparsed limit that comes
    out too large defeats the check it exists to power.
    """
    text = spec.strip()
    days = 0
    if "-" in text:
        d, _, text = text.partition("-")
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, (m, s) = 0, parts
    elif len(parts) == 1:
        h, m, s = 0, parts[0], 0
    else:
        raise ValueError(f"unrecognised SLURM time {spec!r}")
    return days * 86400 + h * 3600 + m * 60 + s


def _parse_required_secrets(raw: dict, path: Path) -> tuple[str, ...]:
    declared = raw.get("requires_secrets")
    if declared is None:
        return DEFAULT_REQUIRED_SECRETS
    entries = tuple(str(x) for x in declared)
    # "section.key" is the only shape run_checks can act on; a bare "token" would silently
    # check secrets["" ]["token"] and always report absent, which reads as a broken machine.
    bad = [e for e in entries if e.count(".") != 1 or e.startswith(".") or e.endswith(".")]
    if bad:
        raise HostProfileError(
            f"{path}: requires_secrets entries must be 'section.key', got {bad}"
        )
    return entries
