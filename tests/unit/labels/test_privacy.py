"""The privacy check of a staged traces sample: closed over every key, column, metadata entry and string
of each file kind, with value patterns over every string, parquet's dictionary, binary and metadata
values included."""
from __future__ import annotations

import io
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.labels import privacy as ps
from axis_qvla.labels import traces_sample as ts
from tests.unit.labels.test_traces_sample import MAIL, ROOTED, STORE_URL, _toy_survey

TRACE = "traces/task_800_attempt_11.jsonl"
GOALS = "goals_per_attempt/task_800_attempt_11.json"
CONFIG = "data/task_800/checker_config.json"
INFO = "corpus/task_800/meta/info.json"
DATA = "corpus/task_800/data/chunk-000/file-000.parquet"
EPISODES = "corpus/task_800/meta/episodes/chunk-000/file-000.parquet"
TASKS = "corpus/task_800/meta/tasks.parquet"
UUID = "-".join(["3f2b8c1e", "9d4a", "4e7b", "8c2d", "1a5f6e7d8c9b"])
WIN_HOME = "\\".join(["C:", "Users", "x"])             # a Windows user directory, assembled like the others
# Identifiers a person or a machine leaves in a record: each must be refused wherever it is planted.
PLANTED = {
    "session_id": ("session_id", "a91f3c"), "uid": ("uid", 1001), "device_id": ("device_id", "d-17"),
    "client_id": ("client_id", "c-3"), "pid": ("pid", 4242), "owner": ("owner", "alice"),
    "annotator": ("annotator", "bob"), "worker": ("worker", "w7"), "participant": ("participant", "p12"),
    "created_at, an ISO date": ("created_at", "2026-10-20T15:42:57Z"),
    "submitted_at, an epoch": ("submitted_at", 1760975000), "ts, an epoch": ("ts", 1760975000.5),
    "ip": ("ip", "10.1.2.3"), "ip_address": ("ip_address", "192.168.0.7"), "a UUID": ("run", UUID),
    "a phone number": ("contact", "+1 951-555-0100"), "a person's name": ("name", "Alice Smith"),
    "free-text notes": ("notes", "recorded by Alice in lab 2"), "a host name": ("machine", "gpu-node-17"),
    "a home-relative path": ("path", "home/alice/data/x.json"), "user": ("user", "alice"),
    "an e-mail": ("contact_email", MAIL),
}
# The values the pattern layer refuses on its own, under any key.
VALUES = {"an ISO-8601 date": "2026-10-20T15:42:57Z", "an IPv4 address": "10.1.2.3", "a UUID": UUID,
          "a phone number": "+1 951-555-0100", "a home path": "home/alice/data/x.json",
          "an e-mail address": MAIL, "an absolute path": ROOTED, "a URL": STORE_URL}


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> dict[str, bytes]:
    root = tmp_path_factory.mktemp("sample")
    out = root / "out"
    ts.stage(out=out, **_toy_survey(root))
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_the_toy_sample_passes(staged):
    assert ps.check_tree(staged) == []


def _content(found: list[str]) -> list[str]:
    """Findings other than the manifest's digest of the file a test edited."""
    return [f for f in found if "the sha256 manifest.json lists" not in f]


def _naming(found: list[str], key: str) -> list[str]:
    """Content findings that name `key` as a path segment or a quoted column."""
    return [f for f in _content(found) if f"/{key}:" in f or f"/{key} " in f or f"'{key}'" in f]


def _edit_json(files, rel, edit):
    files = dict(files)
    if rel.endswith(".jsonl"):
        lines = [json.loads(line) for line in files[rel].decode().splitlines()]
        edit(lines[0])
        files[rel] = "".join(json.dumps(e) + "\n" for e in lines).encode()
    else:
        doc = json.loads(files[rel])
        edit(doc)
        files[rel] = json.dumps(doc).encode()
    return files


def _parquet_bytes(table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


PLACES = {
    "trace event": lambda f, k, v: _edit_json(f, TRACE, lambda e: e.update({k: v})),
    "goal record": lambda f, k, v: _edit_json(f, GOALS, lambda g: g.update({k: v})),
    "goal leaf": lambda f, k, v: _edit_json(f, GOALS, lambda g: g["leaves"][0].update({k: v})),
    "checker configuration": lambda f, k, v: _edit_json(f, CONFIG, lambda c: c["checker"].update({k: v})),
    "info.json": lambda f, k, v: _edit_json(f, INFO, lambda i: i.update({k: v})),
    "manifest task": lambda f, k, v: _edit_json(f, "manifest.json", lambda m: m["tasks"][0].update({k: v})),
    "data parquet column": lambda f, k, v: {**f, DATA: _parquet_bytes(
        pq.read_table(pa.BufferReader(f[DATA])).append_column(k, pa.array([v, v])))},
}


@pytest.mark.parametrize("place", sorted(PLACES))
@pytest.mark.parametrize("case", sorted(PLANTED))
def test_every_planted_identifier_is_refused_wherever_it_is_planted(staged, case, place):
    key, value = PLANTED[case]
    found = ps.check_tree(PLACES[place](staged, key, value))
    assert _naming(found, key), found


SLOTS = {   # a text field each kind declares, with the domain that holds it
    "trace source": lambda f, v: _edit_json(f, TRACE, lambda e: e.update(source=v)),
    "goal leaf tracked body": lambda f, v: _edit_json(f, GOALS, lambda g: g["leaves"][0].update(tracked_body_name=v)),
    "checker object": lambda f, v: _edit_json(f, CONFIG, lambda c: c["checker"].update(sampleBodyName=v)),
    "info robot type": lambda f, v: _edit_json(f, INFO, lambda i: i.update(robot_type=v)),
    "manifest prompt": lambda f, v: _edit_json(f, "manifest.json", lambda m: m["tasks"][0].update(prompt=v)),
}


@pytest.mark.parametrize("slot", sorted(SLOTS))
@pytest.mark.parametrize("case", sorted(VALUES))
def test_an_identifying_value_in_a_declared_text_field_is_refused(staged, case, slot):
    assert _content(ps.check_tree(SLOTS[slot](staged, VALUES[case]))) != []


LEGAL = {"trace source": "packed_state_v1_direct", "goal leaf tracked body": "cup_0", "checker object": "cup_0",
         "info robot type": "axis_robosuite", "manifest prompt": "Put the cup on the plate"}


@pytest.mark.parametrize("slot", sorted(SLOTS))
def test_each_declared_text_field_takes_its_legal_value(staged, slot):
    """The control for the test above: its findings come from the planted value, not from the rewrite."""
    assert _content(ps.check_tree(SLOTS[slot](staged, LEGAL[slot]))) == []


@pytest.mark.parametrize("case", sorted(VALUES))
def test_the_pattern_layer_alone_refuses_identifying_values(case):
    assert [f for f in ps.scan({"v": VALUES[case]}) if case in f]


def test_names_are_the_episodes_own(staged):
    other_body = _edit_json(staged, TRACE, lambda e: e["body_pose"].update(bowl_9={"pos": [0], "quat": [1]}))
    assert [f for f in ps.check_tree(other_body) if "bowl_9" in f and "a body of the episode" in f]
    other_joint = _edit_json(staged, TRACE, lambda e: e["joint_qpos"].update(bowl_9_free=[0.0]))
    assert [f for f in ps.check_tree(other_joint) if "bowl_9_free" in f]
    other_object = _edit_json(staged, CONFIG, lambda c: c["checker"].update(sampleBodyName="towel_2"))
    assert [f for f in ps.check_tree(other_object) if "towel_2" in f]
    sibling = _edit_json(staged, CONFIG, lambda c: c["checker"].update(sampleBodyName="cup_3"))
    assert _content(ps.check_tree(sibling)) == []        # another instance of a body the task holds


def test_a_file_the_sample_never_holds_and_a_manifest_mismatch_are_refused(staged):
    found = ps.check_tree({**staged, "notes.txt": b"x", "traces/README.json": b"{}"})
    assert [f for f in found if f.startswith("notes.txt: not a file")]
    assert [f for f in found if f.startswith("traces/README.json: not a file")]
    edited = _edit_json(staged, CONFIG, lambda c: c["checker"].update(minDeltaX=0.2))
    assert [f for f in ps.check_tree(edited) if "not the sha256 manifest.json lists" in f]
    missing = {k: v for k, v in staged.items() if k != CONFIG}
    assert [f for f in ps.check_tree(missing) if "listed in manifest.json but not in the tree" in f]


def test_the_parquet_scan_reads_dictionary_binary_view_and_field_metadata(tmp_path):
    """Every string a parquet file can carry, nested or encoded, and every metadata entry."""
    cases = {
        "categorical": pd.DataFrame({"c": pd.Categorical([MAIL])}),
        "binary": pa.table({"b": pa.array([MAIL.encode()], pa.binary())}),
        "large binary in a list": pa.table({"b": pa.array([[MAIL.encode()]], pa.list_(pa.large_binary()))}),
        "string view": pa.table({"s": pa.array([MAIL], pa.string_view())}),
        "dictionary in a struct": pa.table({"s": pa.StructArray.from_arrays(
            [pa.array([MAIL]).dictionary_encode()], names=["d"])}),
        "field metadata": pa.table({"x": [1]}, schema=pa.schema([pa.field("x", pa.int64(), metadata={"src": ROOTED})])),
        "nested field metadata": pa.table({"x": [[1]]}, schema=pa.schema([pa.field(
            "x", pa.list_(pa.field("item", pa.int64(), metadata={"src": ROOTED})))])),
        "schema metadata": pa.table({"x": [1]}).replace_schema_metadata({"origin": STORE_URL}),
    }
    for name, table in cases.items():
        p = tmp_path / f"{name.replace(' ', '_')}.parquet"
        table.to_parquet(p) if isinstance(table, pd.DataFrame) else pq.write_table(table, p)
        assert ps.scan_file(p) != [], name
    dated = tmp_path / "dated.parquet"
    pq.write_table(pa.table({"t": pa.array([0], pa.timestamp("s"))}), dated)
    assert [f for f in ps.scan_file(dated) if "date or time column" in f]
    with pytest.raises(ValueError, match="cannot scan"):
        ps.scan_file(tmp_path / "x.mp4")


def test_field_metadata_and_undeclared_schema_metadata_are_refused_by_the_closed_layer(staged):
    table = pq.read_table(pa.BufferReader(staged[DATA]))
    with_field_md = table.cast(pa.schema([f.with_metadata({"k": "v"}) for f in table.schema]))
    assert [f for f in ps.check_tree({**staged, DATA: _parquet_bytes(with_field_md)}) if "carries metadata" in f]
    with_schema_md = table.replace_schema_metadata({"comment": "fine"})
    assert [f for f in ps.check_tree({**staged, DATA: _parquet_bytes(with_schema_md)}) if "'comment'" in f]


def test_a_symbolic_link_anywhere_in_a_tree_is_refused(tmp_path):
    kw = _toy_survey(tmp_path)
    out = tmp_path / "out"
    ts.stage(out=out, **kw)
    assert ps.scan_tree(out) == []
    os.symlink(out / "traces", out / "corpus" / "linked_dir")
    os.symlink(out / "manifest.json", out / "traces" / "linked.jsonl")
    found = ps.scan_tree(out)
    assert "corpus/linked_dir: a symbolic link" in found and "traces/linked.jsonl: a symbolic link" in found


def test_scan_finds_identifying_keys_and_machine_paths():
    assert ps.scan({"leaves": [{"goal_xyz": [0, 0, 0], "tracked_body_name": "glass3"}]}) == []
    found = ps.scan({"user_id": 7, "x": {"trajectory_s3_key": "k", "p": ROOTED}, "submit_nonce": 1,
                     "user_stage": "pro"})
    assert len(found) == 5


@pytest.mark.parametrize("planted", [
    {"userId": 7}, {"username": "a"}, {"e_mail": "a"}, {"s3Key": "k"}, {"bucket_name": "b"}, {"api_token": "t"},
    {"submitted_by": "a"}, {"uploader_id": 3}, {"hostname": "h"},
    {"note": f"written by {MAIL}"}, {"uri": STORE_URL}, {"msg": f"failed to open {ROOTED}/y.json"},
    {"p": "~/x/y"}, {"p": WIN_HOME}, {ROOTED: 1},
])
def test_scan_finds_a_planted_identifier_anywhere(planted):
    assert ps.scan({"leaves": [{"path": "0/1", "extra": [planted]}]}) != []


def test_scan_passes_what_the_records_hold():
    """Body names with a namespace or an object word, checker paths, checker operators, parquet
    metadata, LeRobot path templates, versions, fingerprints and digests."""
    assert ps.scan({"body_pose": {"franka/panda_hand": {}, "sample/soap_0": {}, "ice_bucket_0": {},
                                  "coffee_machine": {}, "glass3": {}}, "path": "0/1/2",
                    "operator": "and", "creator": {"library": "pyarrow", "version": "25.0.1"},
                    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    "fingerprint": "31734d2fccef920c", "splits": {"train": "0:24"},
                    "sha256": "b6c93486539f5ab060f34a9dd8129efac24b2cceff1d110d0c0d05342194bbf9"}) == []


def test_null_is_declared_only_where_the_extractor_writes_it(staged):
    """A leaf that never passed has no goal source, and one whose checker names no body no tracked body."""
    unpassed = _edit_json(staged, GOALS, lambda g: g["leaves"][0].update(goal_source=None, tracked_body_name=None))
    assert _content(ps.check_tree(unpassed)) == []
    for edit in (lambda e: e.update(source=None), lambda e: e["body_pose"].update(cup_0=None)):
        assert _content(ps.check_tree(_edit_json(staged, TRACE, edit))) != []


def _parquet_edit(files, rel, edit):
    table = pq.read_table(pa.BufferReader(files[rel]))
    return {**files, rel: _parquet_bytes(edit(table))}


def _pandas_edit(files, rel, edit):
    frame = pq.read_table(pa.BufferReader(files[rel])).to_pandas()
    edit(frame)
    return {**files, rel: _parquet_bytes(pa.Table.from_pandas(frame, preserve_index=rel == TASKS))}


@pytest.mark.parametrize("name", ["jdoe/test_tube_3", "jsmith42/test_tube_3", "alicesmith/", "jdoe/_7"])
@pytest.mark.parametrize("slot", ["sampleBodyName", "objName", "refName", "bodyName", "tracked_body_name"])
def test_a_namespace_no_task_body_uses_or_an_empty_stem_is_no_object_of_the_task(staged, name, slot):
    """Even when the task has a body of that stem (test_tube_3 here) and a body whose stem is empty
    (the robot's root, franka/)."""
    bodies = ["franka/", "franka/panda_hand", "cup_0", "test_tube_3"]
    files = _pandas_edit(staged, EPISODES, lambda f: f.__setitem__("body_names", [bodies]))
    if slot == "tracked_body_name":
        planted = _edit_json(files, GOALS, lambda g: g["leaves"][0].update(tracked_body_name=name))
    else:
        planted = _edit_json(files, CONFIG, lambda c: c["checker"].update({slot: name}))
    assert [f for f in _content(ps.check_tree(planted)) if repr(name) in f and "an object of the task" in f]
    for legal in ("franka/", "test_tube_4", "cup_0"):
        if slot == "tracked_body_name":
            ok = _edit_json(files, GOALS, lambda g: g["leaves"][0].update(tracked_body_name=legal))
        else:
            ok = _edit_json(files, CONFIG, lambda c: c["checker"].update({slot: legal}))
        assert _content(ps.check_tree(ok)) == [], legal


def test_a_duplicate_json_key_cannot_hide_a_value(staged):
    lines = staged[TRACE].decode().splitlines()
    lines[0] = '{"source": "' + MAIL + '", ' + lines[0][1:]           # the clean "source" comes later
    trace = {**staged, TRACE: ("\n".join(lines) + "\n").encode()}
    assert MAIL.encode() in trace[TRACE]
    assert [f for f in ps.check_tree(trace) if f.startswith(f"{TRACE}: cannot be read") and "'source'" in f]
    text = staged[CONFIG].decode()
    config = {**staged, CONFIG: text.replace('"checker"', '"checker": {"objName": "' + MAIL + '"}, "checker"', 1).encode()}
    assert MAIL.encode() in config[CONFIG]
    assert [f for f in ps.check_tree(config) if f.startswith(f"{CONFIG}: cannot be read") and "'checker'" in f]


def test_a_nested_parquet_field_name_is_checked(staged):
    struct = _parquet_edit(staged, EPISODES, lambda t: t.set_column(
        t.schema.get_field_index("object_body"), "object_body",
        pa.StructArray.from_arrays([pa.array(["cup_0"])], names=["jane_doe"])))
    assert [f for f in _content(ps.check_tree(struct)) if "object_body.jane_doe" in f]
    table = pq.read_table(pa.BufferReader(staged[EPISODES]))
    renamed = table.set_column(table.schema.get_field_index("body_names"), "body_names", pa.array(
        table.column("body_names").to_pylist(), pa.list_(pa.field("jane_doe", pa.string()))))
    buf = io.BytesIO()
    pq.write_table(renamed, buf, use_compliant_nested_type=False)       # keeps the child's name
    listed = {**staged, EPISODES: buf.getvalue()}
    assert pq.read_table(pa.BufferReader(listed[EPISODES])).schema.field("body_names").type.value_field.name == "jane_doe"
    assert [f for f in _content(ps.check_tree(listed)) if "body_names.jane_doe" in f]
    user = pa.table({"x": pa.StructArray.from_arrays([pa.array([1])], names=["user_id"])})
    assert [f for f in ps._scan_table(user, "t") if "user_id" in f]        # the value layer reads nested names too


def test_the_manifests_generated_fields_are_pinned(staged):
    def manifest(edit):
        return _content(ps.check_tree(_edit_json(staged, "manifest.json", edit)))
    assert [f for f in manifest(lambda m: m["families"].append("jdoe")) if "'jdoe'" in f]
    assert [f for f in manifest(lambda m: m.update(rule="Selected by Jane Doe")) if "the selection rule" in f]
    assert [f for f in manifest(lambda m: m["tasks"][0].update(family="jdoe")) if "'jdoe'" in f]
    assert [f for f in manifest(lambda m: m.update(schema="qvla.traces_sample/2")) if "this schema" in f]
    tasks = pq.read_table(pa.BufferReader(staged[TASKS]))
    md = json.loads(tasks.schema.metadata[b"pandas"])
    md["columns"][0]["pandas_type"] = "janedoe"
    typed = {**staged, TASKS: _parquet_bytes(tasks.replace_schema_metadata(
        {**tasks.schema.metadata, b"pandas": json.dumps(md).encode()}))}
    assert [f for f in _content(ps.check_tree(typed)) if "'janedoe'" in f]


def test_the_instruction_is_one_text_in_the_manifest_the_task_table_and_the_episode_table(staged):
    other = "Hand the cup to the plate"
    in_manifest = _edit_json(staged, "manifest.json", lambda m: m["tasks"][0].update(prompt=other))
    assert [f for f in ps.check_tree(in_manifest) if f.startswith(TASKS) and "not the manifest's instruction" in f]
    in_tasks = _pandas_edit(staged, TASKS, lambda f: setattr(f, "index", pd.Index([other])))
    assert [f for f in ps.check_tree(in_tasks) if f.startswith(TASKS) and repr(other) in f]
    in_episodes = _pandas_edit(staged, EPISODES, lambda f: f.__setitem__("tasks", [[other]]))
    assert [f for f in ps.check_tree(in_episodes) if f.startswith(EPISODES) and repr([other]) in f]


def test_simulation_numbers_and_body_names_are_bounded(staged):
    def trace(edit):
        return _content(ps.check_tree(_edit_json(staged, TRACE, edit)))
    assert [f for f in trace(lambda e: e.update(sim_time=1760975000.5)) if "/sim_time:" in f]
    assert [f for f in trace(lambda e: e.update(step_index=99)) if "/step_index:" in f]          # beyond 2 rows
    assert [f for f in trace(lambda e: e["body_pose"]["cup_0"].update(pos=[1760975000.0, 0, 0])) if "/pos[0]:" in f]
    assert [f for f in trace(lambda e: e["body_pose"]["cup_0"].update(quat=[float("nan"), 0, 0, 1])) if "/quat[0]:" in f]
    named = _pandas_edit(staged, EPISODES, lambda f: f.__setitem__("body_names", [["franka/panda_hand", "cup_0", "janedoe"]]))
    named = _edit_json(named, TRACE, lambda e: e["body_pose"].update(janedoe={"pos": [0.0], "quat": [1.0]}))
    assert [f for f in _content(ps.check_tree(named)) if "'janedoe'" in f and "body name" in f]


@pytest.mark.parametrize("plant", [
    lambda f: _edit_json(f, GOALS, lambda g: g["leaves"][0].update(checker_type="AliceSmithChecker")),
    lambda f: _edit_json(f, CONFIG, lambda c: c["checker"].update(type="AliceSmithChecker")),
], ids=["goal leaf checker_type", "checker configuration type"])
def test_a_checker_class_is_one_the_checkers_define(staged, plant):
    assert [f for f in _content(ps.check_tree(plant(staged))) if "'AliceSmithChecker' is not a checker class" in f]


def test_the_manifests_episodes_and_required_tasks_are_the_staged_ones(staged):
    def manifest(edit):
        return _content(ps.check_tree(_edit_json(staged, "manifest.json", edit)))
    assert manifest(lambda m: None) == []
    for edit in (lambda m: m["tasks"][0]["episodes"][0].update(length=3),
                 lambda m: m["tasks"][0]["episodes"][0].update(attempt_id=12),
                 lambda m: m["tasks"][0]["episodes"].append({"episode_index": 1, "attempt_id": 13, "length": 2})):
        assert [f for f in manifest(edit) if "/episodes: not the" in f]
    assert [f for f in manifest(lambda m: m["required"].append(999)) if "required: 999 is not a task" in f]
