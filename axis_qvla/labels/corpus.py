"""Per-frame labels read from the published corpus, for scorers that need more than a reward.

`tasks(rows)` yields, for every task of a row space in task-id order, its frames' label columns and
its episode metadata, exactly as the AWR builder reads them (axis_qvla/signals/methods/awr_chunk.py). The
approach-aware scorer binds each episode's object through a names table
(conf/signals/episode_body_names.json): the corpus fingerprint does not cover `body_names` /
`object_body`, so `names(rows)` returns that table restricted to the row space's episodes and
refuses it unless it digests to the pinned value. Without a table the binding reads the corpus
metadata as it is.

Offline tier: numpy, pandas.
"""
from __future__ import annotations

import dataclasses
import glob
import pathlib
from collections.abc import Iterator
from typing import ClassVar

from axis_qvla.signals import otree
from axis_qvla.signals.rowspace import episode_keys
from axis_qvla.signals.schema import RowSpace


@dataclasses.dataclass(frozen=True)
class CorpusLabels:
    corpus: pathlib.Path
    names_table: pathlib.Path | None = None
    names_sha256: str | None = None

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "reward", "reward_valid", "contact_target_finger", "contact_target_eef",
        "observation.body_pose", "observation.eef_pose", "episode_index", "frame_index")

    def tasks(self, rows: RowSpace) -> Iterator[tuple]:
        import pandas as pd

        from axis_qvla.signals.methods.awr_chunk import episode_meta

        roots = otree.decode(rows.roots)["roots"]
        for tid, rel in sorted(roots.items(), key=lambda kv: int(kv[0])):
            task_dir = pathlib.Path(self.corpus) / rel
            files = sorted(glob.glob(str(task_dir / "data" / "**" / "*.parquet"), recursive=True))
            if not files:
                raise FileNotFoundError(f"task {tid}: no data parquet under {task_dir}")
            frames = pd.concat([pd.read_parquet(f, columns=list(self.COLUMNS)) for f in files], ignore_index=True)
            yield int(tid), frames, episode_meta(task_dir).set_index("episode_index")

    def names(self, rows: RowSpace) -> dict | None:
        if self.names_table is None:
            return None
        from axis_qvla.data import episode_names as en

        shipped = en.load_table(self.names_table)
        keys = episode_keys(rows)
        missing = [k for k in keys if k not in shipped]
        if missing:
            raise ValueError(f"{len(missing)} episodes of {rows.name} are not in {self.names_table} (first: {missing[0]})")
        table = {k: shipped[k] for k in keys}
        if self.names_sha256 is not None and en.table_digest(table) != self.names_sha256:
            raise ValueError(f"{self.names_table} restricted to {rows.name} does not digest to {self.names_sha256}")
        return table

    def digests(self, rows: RowSpace) -> dict[str, str]:
        out = {"corpus_fingerprint": str(rows.corpus_fingerprint)}
        table = self.names(rows)
        if table is not None:
            from axis_qvla.data import episode_names as en

            out["episode_body_names"] = en.table_digest(table)
        return out
