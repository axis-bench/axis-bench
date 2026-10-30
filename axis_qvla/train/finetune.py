"""`qvla train finetune`: the paper's fine-tune of a pretraining run on the 24 held-out tasks.

Why: the benchmark protocol fixes everything after pretraining. By default the fine-tune is plain
and is evaluated bare. `--cond` is for a method that produces conditioning labels: it fine-tunes with
the quality tag, as the paper fine-tunes its conditioned pretrains, and records that the run is
evaluated bare and conditioned (eval.json `serving`, read by axis_qvla.train.serve.serve_spec). Its run id
is <backbone>.finetune.<name>[.s<seed>][.cond], which `qvla bench eval --from-run` takes; this prints that
command once the run is planned (`--print`) or trained. Training needs the pretraining run finished;
`--print` plans from its spec alone. The fine-tune trains at the seed the paper's convention gives
the pretraining's seed, read from the paper's own fine-tune run specs
(axis_qvla.train.method_runs.paper_finetune_seed): 42 for pi0.5, the pretraining's own seed for SmolVLA
and X-VLA; the SEEDS line prints the paper's table. eval.json records the pretraining seed, which the
results file carries and `qvla results contrast` pairs on (replicates pair on the PRETRAIN seed). A
`--run` name keeps the pretraining's `.s<seed>` suffix.

Core tier at import.
"""
from __future__ import annotations

import argparse

from axis_qvla.train.cli import train_spec


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="qvla train finetune",
                                description="fine-tune a pretraining run with the paper's recipe on the 24 held-out tasks")
    p.add_argument("--from", dest="source", required=True, metavar="RUN_ID",
                   help="a pretraining run under <runs> (`qvla train pretrain` printed its id)")
    p.add_argument("--cond", action="store_true",
                   help="fine-tune with the quality tag and evaluate bare and conditioned (methods that condition)")
    p.add_argument("--run", default=None, metavar="NAME",
                   help="the run's name (default: the pretraining run's); its id is "
                        "<backbone>.finetune.<name>[.s<seed>][.cond], keeping the pretraining's seed suffix")
    p.add_argument("--host", default=None,
                   help="this machine's qvla.toml, a path or a profile name (default: ./qvla.toml)")
    p.add_argument("--fsdp-devices", dest="fsdp_devices", type=int, default=None, metavar="N",
                   help="pi0.5 only: shard the train state over N devices")
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="write the run spec and print the command that trains it; reads no data")
    a = p.parse_args(argv)
    from axis_qvla.core.hosts import HostProfileError
    from axis_qvla.data.userconf import user_paths
    from axis_qvla.train import ids, runs
    from axis_qvla.train.armspec import ArmSpecError
    from axis_qvla.train.method_runs import MethodRunError, finetune_spec, paper_finetune_seed, write_spec
    from axis_qvla.train.serve import write_eval_record

    try:
        paths = user_paths(a.host)
        pretrain = runs.load_planned_or_run_spec(ids.run_root(paths, a.source))
        if pretrain.id != a.source:
            raise MethodRunError(f"{ids.run_root(paths, a.source)} holds run {pretrain.id}, not {a.source}")
        if not a.print_only:
            why = runs.unfinished(pretrain.id, pretrain.backbone, paths)
            if why is not None:
                raise MethodRunError(why.message(f"{a.source} has not finished", ids.run_root(paths, a.source),
                                                 pretrain.backbone))
        spec = finetune_spec(pretrain, cond=a.cond, run=a.run)
        _, why = paper_finetune_seed(pretrain.backbone, pretrain.seed)
        path = write_spec(paths, spec)
        write_eval_record(path.parent, pretrain=pretrain.id, pretrain_seed=pretrain.seed, cond=a.cond)
    except (MethodRunError, ArmSpecError, HostProfileError, OSError) as exc:
        print(f"FAIL  {exc}")
        return 2
    print(f"OK    wrote {path}")
    print(f"SEEDS pretrain {pretrain.seed}, fine-tune {spec.seed} ({why}); "
          f"`qvla results contrast` pairs this run on pretrain seed {pretrain.seed}")
    rc = train_spec(path, host=a.host, fsdp_devices=a.fsdp_devices, mode="print" if a.print_only else "run")
    if rc == 0:
        print(f"NEXT  qvla bench eval --from-run {spec.id}" + (f" --host {a.host}" if a.host else ""))
    return rc
