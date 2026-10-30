"""The only place an absolute path is constructed.

Every artifact is `root / relative`, and ONLY the relative half is ever written into an index
row or a manifest. That is what makes two machines interchangeable: the same stored string
resolves to each host's own disk.

A record of what ran (a run manifest's command line) is the exception: it names the machine's
own paths, because it answers "what ran where".
Core tier -- stdlib only.
"""
from __future__ import annotations

import dataclasses
import posixpath
from pathlib import Path

from axis_qvla.core.hosts import HostProfile


class PathOutsideRoots(ValueError):
    """A path belongs to no declared root, or names a root this host does not have."""


@dataclasses.dataclass(frozen=True)
class Paths:
    host: HostProfile

    @property
    def corpus(self) -> Path:
        return self.host.roots["corpus"]

    @property
    def cache(self) -> Path:
        return self.host.roots["cache"]

    @property
    def scratch(self) -> Path:
        return self.host.roots["scratch"]

    @property
    def runs(self) -> Path:
        return self.host.roots["runs"]

    @property
    def catalog(self) -> Path:
        return self.cache / "catalog"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def rel(self, path: Path | str) -> str:
        """Absolute path -> '<root>/<relative>'. Raises if it is under no root."""
        p = Path(path)
        # Longest root first: roots may nest, and the most specific one is the right answer.
        for name, root in sorted(
            self.host.roots.items(), key=lambda kv: len(str(kv[1])), reverse=True
        ):
            try:
                relative = p.relative_to(root)
            except ValueError:
                continue
            # relative_to(root) on the root ITSELF returns Path('.'), which would store as
            # "corpus/." -- round-trips fine, but it is a stored key, and two spellings of
            # one location is exactly the ambiguity this function exists to remove.
            return name if relative == Path(".") else f"{name}/{relative}"
        raise PathOutsideRoots(
            f"{p} is under none of this host's roots "
            f"({', '.join(sorted(self.host.roots))}). Storing it would pin the record "
            f"to one machine."
        )

    def resolve(self, rel: str) -> Path:
        """'<root>/<relative>' -> an absolute path on THIS host."""
        name, _, tail = rel.partition("/")
        if name not in self.host.roots:
            raise PathOutsideRoots(
                f"{name!r} is not a root on host {self.host.name!r} "
                f"({', '.join(sorted(self.host.roots))})"
            )
        # `root / "/etc/passwd"` silently DISCARDS root and yields the absolute path, so a
        # tail that is itself absolute -- what an empty component like "corpus//etc/passwd"
        # partitions to -- would resolve outside every root while looking relative. Callers
        # guard their own stored strings, but the join is here, so the guard belongs here.
        if posixpath.isabs(tail):
            raise PathOutsideRoots(
                f"{rel!r} has an absolute remainder {tail!r}; joining it would discard the "
                f"{name!r} root entirely and resolve outside every declared root"
            )
        return self.host.roots[name] / tail
