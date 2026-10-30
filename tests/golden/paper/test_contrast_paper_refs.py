"""`qvla results contrast` with `paper:<result>` sides: a method author's own results file against the
paper's published plain BC, read from a results pack through the results map, paired on the pretrain
seed. Against the fixture pack (2 tasks x 3 seeds), no network.

The author's file stands in for what `qvla bench eval --from-run` writes: the fixture pack's BC file
with the author's own run name, rows and training seeds ({pretrain, ft} as the fine-tune's eval.json
and run spec record them)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from axis_qvla.cli import main
from axis_qvla.train.armspec import load_armspec
from tests.golden.paper.fixtures import BC_FILE, FIX, user_results
from tests.unit.train._arms import host

COMMON = ["--benchmark", "fixture_bench", "--bench-conf-dir", str(FIX / "conf")]


def _pack_with_replicate_43(tmp_path: Path) -> tuple[Path, Path]:
    """The fixture pack plus the paper's seed-43 BC replicate (Table I pools bc43|bare with bc|bare);
    -> (pack, its results map)."""
    pack = tmp_path / "pack"
    shutil.copytree(FIX / "pack", pack)
    data = user_results(pack / "results/s2/r1fbc43_dualsim.json", arm="r1fbc43", pretrain=43, ft=43,
                         outcomes={806: [True, False, False], 1050: [False, False, True]}).read_bytes()   # SR 1/3
    doc = json.loads((FIX / "paper_fixture_cells.json").read_bytes())
    bc = next(c for c in doc["cells"] if c["cell"] == "bc|bare")
    bc = {k: v for k, v in bc.items() if k != "manifest"}
    doc["cells"].append({**bc, "cell": "bc43|bare", "run": "r1fbc43", "served": "r1fbc43_dualsim",
                         "training_seed": {"pretrain": 43, "ft": 43},
                         "results": {"path": "results/s2/r1fbc43_dualsim.json",
                                     "sha256": hashlib.sha256(data).hexdigest()}})
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(doc))
    return pack, cells


def _contrast(*sides: str, pack: Path = FIX / "pack", cells: Path = FIX / "paper_fixture_cells.json") -> int:
    return main(["results", "contrast", *sides, "--results", str(pack), "--results-map", str(cells), *COMMON])


def test_paper_refs_resolve_against_the_fixture_pack(capsys):
    """Only replicates the map holds: Table I pools cfg_ph_ep|cond with cfgpt43|cond and cfgpt44|cond,
    which the fixture's map lacks, so each reference is its seed-42 file."""
    assert _contrast("--a", "paper:cfg_ph_ep|cond", "--b", "paper:bc|bare") == 0
    out = capsys.readouterr().out
    assert f"published results: {FIX / 'pack'}" in out
    assert "paper:bc|bare = r1fbc (seed 42)" in out and "paper:cfg_ph_ep|cond = r1fcfg (seed 42)" in out
    assert "r1fcfg - r1fbc: mean" in out and "sha256" not in out


def test_an_unknown_paper_ref_is_refused_naming_the_list_and_the_close_ids(capsys):
    assert _contrast("--a", "paper:bc|bar", "--b", "paper:bc|bare") == 2
    out = capsys.readouterr().out
    assert out.startswith("FAIL  paper:'bc|bar' is not a published result (did you mean paper:'bc|bare'")
    assert "`qvla paper reeval --list` lists them" in out


def test_a_tampered_published_file_is_refused(tmp_path, capsys):
    pack = tmp_path / "pack"
    shutil.copytree(FIX / "pack", pack)
    (pack / BC_FILE).write_bytes((pack / BC_FILE).read_bytes() + b" ")
    assert _contrast("--a", "paper:cfg_ph_ep|cond", "--b", "paper:bc|bare", pack=pack) == 2
    out = capsys.readouterr().out
    assert out == (f"FAIL  {pack / BC_FILE} is not the file the paper published; re-fetch it with "
                   f"`qvla data get results`\n")
    assert "sha256" not in out and not re.search(r"[0-9a-f]{16}", out)       # no digest, whole or cut


def test_my_from_run_results_against_the_papers_bc_print_the_table_i_statistic(tmp_path, capsys):
    mine = user_results(tmp_path / "r1" / "out_pi05.finetune.mine" / "results.json", arm="pi05.finetune.mine",
                         pretrain=42, ft=42)
    assert _contrast("--a", str(mine), "--b", "paper:bc|bare", "--bc", "paper:bc|bare", "--relative") == 0
    out = capsys.readouterr().out
    assert "replicate 42: +66.67 % of plain BC" in out
    assert "pi05.finetune.mine - r1fbc: mean +66.67 % of plain BC" in out and "no claim" in out


def test_a_seed_43_pretrain_pairs_with_the_papers_replicate_43(tmp_path, capsys):
    """`qvla train pretrain --seed 43` -> `finetune --from` -> its results pair with replicate 43 only."""
    hosts, paths = host(tmp_path)
    assert main(["train", "pretrain", "--labels", "mine", "--backbone", "pi05", "--seed", "43",
                 "--host", str(hosts), "--print"]) == 0
    assert load_armspec(paths.runs / "pi05.pretrain.mine.s43" / "spec.toml").seed == 43
    assert main(["train", "finetune", "--from", "pi05.pretrain.mine.s43", "--host", str(hosts), "--print"]) == 0
    ft = paths.runs / "pi05.finetune.mine.s43"
    out = capsys.readouterr().out
    assert ("SEEDS pretrain 43, fine-tune 42 (the paper's pi0.5 fine-tunes, pretrain seed -> fine-tune seed: "
            "42 -> 42; 43 -> 42 (bc43: 43); 44 -> 42); `qvla results contrast` pairs this run on pretrain seed 43") in out
    assert "NEXT  qvla bench eval --from-run pi05.finetune.mine.s43" in out
    # What axis_qvla.train.serve.serve_spec hands `bench eval --from-run` as the results file's seeds: the
    # fine-tune's own seed (the paper's, 42) and the pretraining seed its eval.json records.
    ft_seed, pre_seed = load_armspec(ft / "spec.toml").seed, json.loads((ft / "eval.json").read_text())["pretrain_seed"]
    assert (ft_seed, pre_seed) == (42, 43)
    mine = user_results(tmp_path / "mine43.json", arm=ft.name, pretrain=pre_seed, ft=ft_seed)
    pack, cells = _pack_with_replicate_43(tmp_path)
    assert _contrast("--a", str(mine), "--b", "paper:bc|bare", "--bc", "paper:bc|bare", "--relative",
                     pack=pack, cells=cells) == 0
    out = capsys.readouterr().out
    assert "paper:bc|bare = r1fbc43 (seed 43)" in out and "r1fbc (seed 42)" not in out
    assert "replicate 43: +150.00 % of plain BC" in out                   # 100 (5/6 - 1/3) / (1/3)
    assert "ft seeds differ in replicate(s) 43" in out                    # 42 here, 43 in the paper's bc43


def test_next_to_files_a_reference_reads_the_replicates_at_their_seeds(tmp_path, capsys):
    pack, cells = _pack_with_replicate_43(tmp_path)
    assert _contrast("--a", str(user_results(tmp_path / "m42.json", arm="m", pretrain=42, ft=42)),
                     str(user_results(tmp_path / "m43.json", arm="m43", pretrain=43, ft=42)),
                     "--b", "paper:bc|bare", pack=pack, cells=cells) == 0
    out = capsys.readouterr().out
    assert "paper:bc|bare = r1fbc (seed 42), r1fbc43 (seed 43)" in out and "pairs 2" in out


def test_a_seed_the_paper_has_no_replicate_at_is_refused_naming_the_seeds_and_the_flag(tmp_path, capsys):
    mine = user_results(tmp_path / "mine7.json", arm="pi05.finetune.mine.s7", pretrain=7, ft=42)
    assert _contrast("--a", str(mine), "--b", "paper:bc|bare") == 2
    out = capsys.readouterr().out
    assert "has replicates at training seed(s) 42, none at 7" in out and "--seed 42" in out


@pytest.mark.parametrize("with_copy", [True, False])
def test_the_default_pack_is_the_local_copy_qvla_data_get_results_writes(tmp_path, capsys, monkeypatch, with_copy):
    hosts, paths = host(tmp_path)
    local = paths.corpus / "qvla-v1-results"
    if with_copy:
        shutil.copytree(FIX / "pack", local)
    # Without a local copy the default is the Hub's; a checksum manifest that is not there stops it
    # before any network access (exit 3), naming it.
    rc = main(["results", "contrast", "--a", "paper:cfg_ph_ep|cond", "--b", "paper:bc|bare", "--host", str(hosts),
               "--results-map", str(FIX / "paper_fixture_cells.json"), "--checksums", str(tmp_path / "no.lock.json"),
               "--cache", str(tmp_path / "cache"), *COMMON])
    out = capsys.readouterr().out
    if with_copy:
        assert rc == 0 and f"published results: {local}" in out
    else:
        assert rc == 3 and "no.lock.json" in out


def test_a_reference_to_another_backbones_result_is_refused_naming_the_matching_one(tmp_path, capsys):
    """Refused before anything is read (the pack named does not exist), against the paper's own map."""
    mine = user_results(tmp_path / "mine.json", arm="smolvla.finetune.mine", pretrain=0, ft=0)
    assert main(["results", "contrast", "--a", str(mine), "--b", "paper:bc|bare", "--results",
                 str(tmp_path / "nowhere"), *COMMON]) == 2
    assert capsys.readouterr().out == ("FAIL  paper:'bc|bare' is a pi05 result and your results are smolvla: "
                                       "use paper:'svla_bc|bare', the main results' same role for that backbone\n")


def test_references_spanning_two_vlas_are_refused_before_anything_is_read(tmp_path, capsys):
    """SmolVLA and X-VLA both train at seeds 0/1/2, so their replicates would pair seed for seed; they
    are unrelated runs, and a contrast made only of references must refuse them, not call it a claim."""
    rc = main(["results", "contrast", "--a", "paper:svla_bc|bare", "--b", "paper:xvla_bc|bare",
               "--bc", "paper:xvla_bc|bare", "--relative", "--results", str(tmp_path / "nowhere"), *COMMON])
    assert rc == 2
    assert capsys.readouterr().out == (
        "FAIL  paper:'svla_bc|bare' is smolvla and paper:'xvla_bc|bare' is xvla: a contrast compares one "
        "VLA's replicates; name results of one backbone (`qvla paper reeval --list` prints each one's "
        "backbone)\n")


def test_resolve_refuses_a_two_backbone_map_and_opens_nothing():
    """`resolve` itself, over a results map holding two backbones (the fixture's is pi0.5-only)."""
    import dataclasses

    from axis_qvla.bench.results.paper_refs import PaperRefError, resolve
    from axis_qvla.paper.cells import load_cells
    from axis_qvla.paper.spec import load_spec

    cells = load_cells((FIX / "paper_fixture_cells.json").read_bytes())
    two = dataclasses.replace(cells, cells=tuple(dataclasses.replace(c, backbone="xvla")
                                                 if c.cell == "cfg_ph_ep|cond" else c for c in cells.cells))
    table1 = load_spec().table1

    def opener():
        raise AssertionError("a refused contrast must open no source")

    with pytest.raises(PaperRefError, match=r"^paper:'bc\|bare' is pi05 and paper:'cfg_ph_ep\|cond' is xvla: "
                                            r"a contrast compares one VLA's replicates"):
        resolve(["paper:bc|bare", "paper:cfg_ph_ep|cond"], opener, two, table1, None)
    # One backbone resolves as before (the opener is reached).
    with pytest.raises(AssertionError, match="open no source"):
        resolve(["paper:bc|bare", "paper:cfg_ph_ep|cond"], opener, cells, table1, None)


def test_without_the_release_lock_both_commands_name_what_to_run(tmp_path, capsys):
    """No checksum manifest and the Hub default: `paper tables` and `results contrast` name
    `qvla data get results` and --results, never a raw `[Errno 2]`, before any network access."""
    lock = str(tmp_path / "no.lock.json")
    assert main(["paper", "tables", "--check", "--checksums", lock, "--cache", str(tmp_path / "c")]) == 3
    tables = capsys.readouterr().out
    assert main(["results", "contrast", "--a", "paper:cfg_ph_ep|cond", "--b", "paper:bc|bare",
                 "--results-map", str(FIX / "paper_fixture_cells.json"), "--checksums", lock,
                 "--cache", str(tmp_path / "c"), *COMMON]) == 3
    contrast = capsys.readouterr().out
    for out in (tables, contrast):
        assert f"the release checksum manifest {lock} is not in this checkout" in out, out
        assert "`qvla data get results`" in out and "--results <" in out and "Errno" not in out
    assert tables.startswith("FETCH  ") and contrast.startswith("FAIL  cannot read: ")
