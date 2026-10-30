from __future__ import annotations

import hashlib

from axis_qvla.realrobot.realdata import validate_fr3 as vf
from axis_qvla.realrobot.realdata.export import dataset_segments, export
from axis_qvla.realrobot.realdata.tail import chunk_starts, dumps, episode_lengths, tail_ranges
from tests.unit.realrobot.realdata.test_export import _world


def _published(tmp_path):
    rec = _world(tmp_path)
    segs = dataset_segments(rec)
    export(rec, segs, tmp_path / "pub")
    ranges = tail_ranges(episode_lengths(tmp_path / "pub" / "meta" / "episodes.jsonl"))
    text = dumps(ranges)
    (tmp_path / "tail.json").write_text(text)
    facts = vf.Facts(recording=(3, 160), published=(len(segs), sum(s.length for s in segs)),
                     chunk_starts=chunk_starts(ranges), tail10_md5=hashlib.md5(text.encode()).hexdigest())
    return rec, facts


def test_every_check_passes_on_a_dataset_the_tools_exported(tmp_path, capsys):
    rec, facts = _published(tmp_path)
    assert vf.run(rec.root, tmp_path / "pub", tmp_path / "tail.json", work=tmp_path / "work", stats=True, facts=facts) == 0
    assert "ALL PASS" in capsys.readouterr().out


def test_a_wrong_fact_fails_and_is_named(tmp_path, capsys):
    rec, facts = _published(tmp_path)
    bad = vf.Facts(facts.recording, (99, 1), facts.chunk_starts, facts.tail10_md5)
    assert vf.run(rec.root, tmp_path / "pub", tmp_path / "tail.json", facts=bad) == 1
    assert "FAIL  published size" in capsys.readouterr().out


def test_the_paper_facts_are_the_published_fr3_numbers():
    assert vf.FR3 == vf.Facts((250, 66_463), (392, 62_680), 58_760, "9e47e3c8f0fe9207a7f34f6d84147fb9")


def test_the_flags_check_holds_done_to_each_last_frame_and_the_other_two_flags_constant(tmp_path, capsys):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from axis_qvla.realrobot.realdata.lerobot_v21 import write_table

    rec = _world(tmp_path)
    check = vf.Checks()
    vf.check_flags(check, rec)
    assert check.failed == [] and "PASS  flags: 3/3 episodes" in capsys.readouterr().out
    path = rec.episode_path(1)
    t = pq.read_table(path)
    done = np.asarray(t.column("done").to_pylist())
    done[0] = True
    write_table(t.set_column(t.schema.get_field_index("done"), t.schema.field("done"), pa.array(done)), path)
    vf.check_flags(check, rec)
    assert check.failed == ["flags"] and "first bad [1]" in capsys.readouterr().out


def test_main_gives_a_fail_line_and_exit_2_on_a_mistyped_recording_path(tmp_path, capsys):
    """`run` raises `RealdataError` (via `Dataset.open`) for a path that is not a LeRobot v2.1 dataset
    directory; `main` must catch it the same way `realdata/cli.py` does -- one FAIL line, exit 2 --
    never an uncaught traceback."""
    _published(tmp_path)
    code = vf.main(["--recording", str(tmp_path / "no-such-recording"), "--published", str(tmp_path / "pub"),
                    "--tail-reference", str(tmp_path / "tail.json")])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL  ") and "no-such-recording" in err


def test_the_fr3_data_record_states_what_validate_fr3_checks():
    """experiments/realrobot/fr3_data.toml's counts are the ones validate_fr3 reproduces from the public
    copies, and its `derive` command runs on the copy it cites."""
    import tomllib
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    d = tomllib.loads((repo / "experiments" / "realrobot" / "fr3_data.toml").read_text(encoding="utf-8"))
    assert (d["recordings"], d["frames"]) == vf.FR3.recording
    assert (d["derived_episodes"], d["derived_frames"]) == vf.FR3.published
    assert "--export" not in d["derive"]
