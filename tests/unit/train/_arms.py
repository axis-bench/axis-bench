"""Shared fixtures for the training tests: ArmSpec texts and a host profile in tmp_path."""
from __future__ import annotations

import pathlib

from axis_qvla.core.hosts import load_host
from axis_qvla.core.paths import Paths

H = "c" * 64
PUB = f"axisrobotics/qvla-v1-pi05:pretrain/example/20604#sha256:{H}"


def arm_toml(*, id="pi05.finetune.example", backbone="pi05-openpi", stage="finetune", track="paper",
             preset="pi05.finetune.lora.paper", rowspace="heldout24_ft", init=PUB, bundle=True, norm=True,
             seed=42, overrides="") -> str:
    out = [f'[arm]\nid = "{id}"\nbackbone = "{backbone}"\nstage = "{stage}"\ntrack = "{track}"\n']
    if init:
        out.append(f'[init]\nckpt = "{init}"\n')
    rs = rowspace if ("#" in rowspace or rowspace.startswith("local:")) else f"{rowspace}#sha256:{H}"
    out.append(f'[data]\nrowspace = "{rs}"\nasset_id = "Devon018/Franka-Datasets-v2"\n')
    if bundle:
        out.append(f'[signal]\nbundle = "{bundle if isinstance(bundle, str) else "sha256:" + H}"\n')
    if norm:
        out.append(f'[norm]\nartifact = "heldout24_qual_v2#sha256:{H}"\n')
    out.append(f'[recipe]\npreset = "{preset}"\nseed = {seed}\n')
    if overrides:
        out.append(f"[recipe.overrides]\n{overrides}\n")
    return "\n".join(out)


def host(tmp_path: pathlib.Path) -> tuple[pathlib.Path, Paths]:
    """A user `qvla.toml` in tmp_path (what `--host` takes) and the Paths it gives."""
    d = tmp_path / "hosts"
    d.mkdir()
    roots = {k: tmp_path / "roots" / k for k in ("corpus", "cache", "scratch", "runs")}
    body = "\n".join(f'{k} = "{v}"' for k, v in roots.items())
    (d / "qvla.toml").write_text(f'launcher = "local"\n\n[roots]\n{body}\n')
    return d / "qvla.toml", Paths(load_host("qvla", hosts_dir=d))
