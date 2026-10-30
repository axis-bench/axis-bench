"""A merged LeRobot copy cut on this machine binds only when every field of its cut record is what
the row space names: the ranges file, the action column, and the roots index's task directories
with their pinned meta/info.json."""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from axis_qvla.train.cut_copy import CUT_RECORD, REPO_FILES, check_cut_copy
from axis_qvla.train.recipes import RowSpaceLayout

TASKS = (806, 1050)   # task-id order; "task_1050" < "task_806" as strings


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _layout(tmp_path: pathlib.Path, *, pin=True):
    """A bench repository (roots index, ranges file, two task directories, its file list pinning
    all of them, as the published one does) and an empty copy directory beside it; returns
    (layout, rowspace_dirs mapping, repo root, record). `pin=False` writes a file list that pins none
    of them, `pin=None` no file list at all."""
    repo = tmp_path / "corpus" / "qvla-v1-bench"
    roots = {str(t): f"ft_demos/out_eef/task_{t}" for t in reversed(TASKS)}
    (repo / "ft_demos" / "splits").mkdir(parents=True)
    (repo / "ft_demos/splits/roots.json").write_text(json.dumps(roots))
    (repo / "ft_demos/splits/ranges.json").write_text(json.dumps({f"task_{t}--0": [[0, 5]] for t in TASKS}))
    files = [{"path": f"ft_demos/splits/{f}", "sha256": _sha((repo / "ft_demos/splits" / f).read_bytes())}
             for f in ("roots.json", "ranges.json")]
    for t in TASKS:
        info = repo / f"ft_demos/out_eef/task_{t}/meta/info.json"
        info.parent.mkdir(parents=True)
        info.write_text(json.dumps({"total_frames": t}))
        files.append({"path": f"ft_demos/out_eef/task_{t}/meta/info.json", "sha256": _sha(info.read_bytes())})
    if pin is not None:
        (repo / REPO_FILES).write_text(json.dumps({"schema": "qvla.bench_files/1", "files": files if pin else []}))
    copy = tmp_path / "cache" / "derived" / "ft_lerobot_merged"
    copy.mkdir(parents=True)
    rs = {"roots": repo / "ft_demos/splits/roots.json", "ranges": repo / "ft_demos/splits/ranges.json",
          "corpus_base": repo, "lerobot_dataset": copy}
    lay = RowSpaceLayout(name="heldout24_ft", repo="qvla-v1-bench", roots="ft_demos/splits/roots.json",
                         ranges="ft_demos/splits/ranges.json", corpus_base=".", lerobot_dataset="ft_lerobot_merged",
                         lerobot_copy="cut", n_rows=10, action_source="action")
    record = {  # the shape axis_merge_datasets --ranges writes
        "ranges_sha256": _sha(rs["ranges"].read_bytes()),
        "action_source": "action",
        "inputs": [{"root": f"task_{t}", "info_sha256": f["sha256"]} for t, f in zip(TASKS, files[2:], strict=True)],
        "episodes": [{"root": f"task_{t}", "episode_index": i, "src_episode_index": 0, "src_frame_start": 0,
                      "src_frame_stop": 5} for i, t in enumerate(TASKS)],
    }
    return lay, rs, repo, record


def _check(tmp_path, change=None, *, pin=True):
    lay, rs, repo, record = _layout(tmp_path, pin=pin)
    if change is not None:
        change(record, rs, repo)
    (rs["lerobot_dataset"] / CUT_RECORD).write_text(json.dumps(record))
    return check_cut_copy(lay, rs, repo)


def test_the_record_the_command_writes_binds(tmp_path):
    assert _check(tmp_path) == []


def test_another_ranges_file_is_refused(tmp_path):
    def stale(record, rs, repo):
        record["ranges_sha256"] = _sha(b"other spans")

    (problem,) = _check(tmp_path, stale)
    assert "was cut from a ranges file with sha256" in problem and "rebuild the copy from it" in problem


OTHER_SPANS = json.dumps({f"task_{t}--0": [[1, 6]] for t in TASKS})   # the same total, other frames


def test_the_ranges_pin_is_the_reference_when_the_repository_lists_it(tmp_path):
    """A ranges file edited in the local mirror does not move the reference: a copy cut from the
    edited spans is refused, though the copy and the file on disk agree, and a copy cut from the
    pinned spans still binds. With no pin, the file on disk is the reference."""
    def cut_from_edited(record, rs, repo):
        rs["ranges"].write_text(OTHER_SPANS)
        record["ranges_sha256"] = _sha(OTHER_SPANS.encode())

    def edited_after_the_cut(record, rs, repo):
        rs["ranges"].write_text(OTHER_SPANS)

    pinned = _sha(json.dumps({f"task_{t}--0": [[0, 5]] for t in TASKS}).encode())
    repo = tmp_path / "a" / "corpus" / "qvla-v1-bench"
    assert _check(tmp_path / "a", cut_from_edited) == [
        f"rowspace: {tmp_path / 'a/cache/derived/ft_lerobot_merged'} was cut from a ranges file with sha256 "
        f"{_sha(OTHER_SPANS.encode())[:12]}, but {repo / REPO_FILES} pins ft_demos/splits/ranges.json at "
        f"{pinned[:12]}; rebuild the copy from it"]
    assert _check(tmp_path / "b", edited_after_the_cut) == []
    assert _check(tmp_path / "c", cut_from_edited, pin=False) == []


def test_a_roots_index_that_is_not_the_pinned_one_is_refused(tmp_path):
    """The task directories are read from the roots index on disk, so when the repository pins
    the index, the file on disk must be the pinned one: an index edited to drop a task, with a copy
    merged to match it, is refused. With no pin, the file on disk is the reference."""
    def drop_a_task(record, rs, repo):
        rs["roots"].write_text(json.dumps({"806": "ft_demos/out_eef/task_806"}))
        record["inputs"] = record["inputs"][:1]

    pinned = _sha(json.dumps({str(t): f"ft_demos/out_eef/task_{t}" for t in reversed(TASKS)}).encode())
    edited = _sha(json.dumps({"806": "ft_demos/out_eef/task_806"}).encode())
    roots = tmp_path / "a/corpus/qvla-v1-bench/ft_demos/splits/roots.json"
    assert _check(tmp_path / "a", drop_a_task) == [
        f"rowspace: {roots} has sha256 {edited[:12]}, but {roots.parents[2] / REPO_FILES} pins it at "
        f"{pinned[:12]}, so the task directories it lists are not the repository's; fetch it again"]
    assert _check(tmp_path / "b", drop_a_task, pin=False) == []


def test_another_action_column_is_refused(tmp_path):
    """A copy whose action was read from action_eef has the same frames: only the record says so."""
    def eef(record, rs, repo):
        record["action_source"] = "action_eef"

    assert _check(tmp_path, eef) == [
        f"rowspace: {tmp_path / 'cache/derived/ft_lerobot_merged'} reads its action from 'action_eef', but frame "
        "index heldout24_ft trains on 'action'; rebuild the copy with --action-source action"]


@pytest.mark.parametrize("inputs, detail", [
    (lambda i: i[:1], "missing ['task_1050']"),
    (lambda i: [*i, {"root": "task_9999", "info_sha256": "0" * 64}], "not in it ['task_9999']"),
    (lambda i: [i[0], {"root": "task_9999", "info_sha256": "0" * 64}], "missing ['task_1050']; not in it ['task_9999']"),
    (lambda i: list(reversed(i)), "the same ones, in another order or twice"),
])
def test_other_task_directories_are_refused(tmp_path, inputs, detail):
    def change(record, rs, repo):
        record["inputs"] = inputs(record["inputs"])

    (problem,) = _check(tmp_path, change)
    assert "was merged from other task directories than" in problem and f"({detail})" in problem


def test_an_input_whose_info_json_is_not_the_pinned_one_is_refused(tmp_path):
    def other(record, rs, repo):
        record["inputs"][1]["info_sha256"] = "0" * 64

    (problem,) = _check(tmp_path, other)
    assert "was merged from a task_1050 whose meta/info.json has sha256 000000000000" in problem


def test_the_pin_is_the_reference_when_the_repository_lists_the_file(tmp_path):
    """A local info.json edited after the copy was cut does not move the reference: the record must
    match the repository's own pin."""
    def edit_disk(record, rs, repo):
        (repo / "ft_demos/out_eef/task_806/meta/info.json").write_text("edited")

    assert _check(tmp_path, edit_disk) == []


def test_the_file_on_disk_is_the_reference_when_the_repository_lists_none(tmp_path):
    assert _check(tmp_path, pin=False) == []

    def edit_disk(record, rs, repo):
        (repo / "ft_demos/out_eef/task_806/meta/info.json").write_text("edited")

    (problem,) = _check(tmp_path / "b", edit_disk, pin=False)
    assert "was merged from a task_806 whose meta/info.json" in problem


@pytest.mark.parametrize("record, problem", [
    (None, "is not there: "),
    ("not json", "is not JSON"),
    ([], "is not a cut record"),
])
def test_a_missing_or_malformed_record_is_refused(tmp_path, record, problem):
    lay, rs, repo, _ = _layout(tmp_path)
    if record is not None:
        (rs["lerobot_dataset"] / CUT_RECORD).write_text(record if isinstance(record, str) else json.dumps(record))
    (got,) = check_cut_copy(lay, rs, repo)
    assert problem in got


def test_a_published_mirror_without_its_file_list_is_refused(tmp_path):
    """A partial fetch that left BENCH_FILES.json out is never read as "nothing pinned": the
    file on disk would then be the reference for every file, edited or not."""
    repo = tmp_path / "corpus" / "qvla-v1-bench"
    assert _check(tmp_path, pin=None) == [
        f"rowspace: {repo / REPO_FILES} is not there, so the files this run reads cannot be checked against the "
        f"repository's pins (a partial fetch leaves it out); fetch it with them "
        f"(`qvla data get bench --include {REPO_FILES}`)"]


def test_a_record_missing_fields_lists_every_problem(tmp_path):
    def strip(record, rs, repo):
        record.clear()

    problems = _check(tmp_path, strip)
    assert len(problems) == 3
    assert any("records no ranges_sha256" in p for p in problems)
    assert any("reads its action from None" in p for p in problems)
    assert any("records no inputs" in p for p in problems)
