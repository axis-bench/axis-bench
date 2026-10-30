"""What several real-world steps share. Stdlib at module scope."""
from __future__ import annotations

import sys


def fail(message: str, code: int = 2) -> int:
    """Print one refusal line on stderr, as the real-robot tools do, and return the exit code."""
    print(f"FAIL  {message}", file=sys.stderr)
    return code


def ensure_norm(lay, stage: str):
    """The norm statistics of `stage`'s data at lay.norm_file(stage). They are kept when they describe
    that data (axis_qvla.realrobot.fr3.norm_check) and computed over every frame otherwise. compute_norm
    raises ValueError for a directory that is not a training set of the stage's embodiment."""
    from axis_qvla.realrobot.fr3 import norm
    from axis_qvla.realrobot.fr3.norm_check import check_norm
    from axis_qvla.realrobot.layout import NORMS

    _name, data_dir, embodiment = NORMS[stage]
    data, path = getattr(lay, data_dir), lay.norm_file(stage)
    if path.is_file() and not check_norm(path, data, embodiment):
        print(f"norm statistics of {data.name} ({embodiment}): kept {path}")
        return path
    norm.write_norm(path, norm.compute_norm(data, embodiment))
    print(f"norm statistics of {data.name} ({embodiment}) -> {path}")
    return path
