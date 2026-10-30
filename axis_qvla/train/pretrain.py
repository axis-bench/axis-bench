"""`qvla train pretrain`: pretrain a backbone on your compiled labels, on the paper's budget.

Why: the method is the only thing its author writes. This writes the paper's pretraining run spec for
the backbone, with the labels named by name (axis_qvla.train.method_runs), to <runs>/<run id>/spec.toml,
then trains it as `qvla train <spec>` does (`--print`: plans it, reading no data). Its run id is
<backbone>.pretrain.<--run, default the labels' name>[.s<seed>], which `qvla train finetune --from` takes.

`--seed N` sets the pretraining's training seed (default the paper's first replicate's: 42 for pi0.5,
0 for SmolVLA and X-VLA); a seed other than that default suffixes the run id with `.s<N>`
(`pi05.pretrain.<name>.s43`), and the fine-tune of it inherits the suffix. The fine-tune trains at
the seed the paper's convention gives this one (42 for pi0.5; the pretraining's own seed for SmolVLA
and X-VLA; `qvla train finetune` prints it) and records this pretraining seed, which is what `qvla
results contrast` pairs the run on: with `--b paper:'bc|bare'`, a seed-43 run is contrasted with the
paper's seed-43 replicate (the paper's pi0.5 replicates are 42, 43 and 44; SmolVLA's and X-VLA's 0, 1
and 2).

Core tier at import.
"""
from __future__ import annotations

import argparse

from axis_qvla.train.cli import train_spec


def main(argv: list[str]) -> int:
    from axis_qvla.train.method_runs import FAMILIES

    p = argparse.ArgumentParser(prog="qvla train pretrain",
                                description="pretrain a backbone on your compiled labels with the paper's budget")
    p.add_argument("--labels", required=True, metavar="NAME", help="your compiled labels, by the name the compile printed")
    p.add_argument("--backbone", required=True, choices=sorted(FAMILIES))
    p.add_argument("--run", default=None, metavar="NAME",
                   help="the run's name (default: the labels'); its id is <backbone>.pretrain.<name>[.s<seed>]")
    p.add_argument("--seed", type=int, default=None, metavar="N",
                   help="the pretraining seed (default the paper's first replicate's: pi05 42, smolvla/xvla 0); "
                        "another seed adds .s<N> to the run id. The paper's replicates: pi05 42/43/44, "
                        "smolvla/xvla 0/1/2")
    p.add_argument("--host", default=None,
                   help="this machine's qvla.toml, a path or a profile name (default: ./qvla.toml)")
    p.add_argument("--fsdp-devices", dest="fsdp_devices", type=int, default=None, metavar="N",
                   help="pi0.5 only: shard the train state over N devices")
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="write the run spec and print the command that trains it; reads no data")
    a = p.parse_args(argv)
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data.userconf import user_paths
    from axis_qvla.train.armspec import ArmSpecError
    from axis_qvla.train.method_runs import MethodRunError, pretrain_spec, write_spec

    try:
        spec = pretrain_spec(a.backbone, a.labels, run=a.run, seed=a.seed)
        path = write_spec(user_paths(a.host), spec)
    except (MethodRunError, ArmSpecError, HostProfileError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    print(f"OK    wrote {path}")
    rc = train_spec(path, host=a.host, fsdp_devices=a.fsdp_devices, mode="print" if a.print_only else "run")
    if rc == 0:
        print(f"NEXT  qvla train finetune --from {spec.id}" + (f" --host {a.host}" if a.host else ""))
    return rc
