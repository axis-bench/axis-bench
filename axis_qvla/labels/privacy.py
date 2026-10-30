"""What a staged traces sample may hold: a closed check, so that nothing identifying is published.

Two layers, both over every file of the tree:

1. The closed shape (`check_tree`). Every file must be one of the sample's kinds, by its path, and
   every key, parquet column, metadata entry and string in it must be one its kind declares: trace
   events, goal records, checker configurations, LeRobot `info.json` and parquet tables, the
   manifest. Names are checked against the sample's own data instead of being listed: a trace's
   bodies are its episode's `body_names` (and `world`), its joints the robot's and its bodies' free
   joints, and an object a checker or a goal leaf names is a body of the task or shares its
   namespace and non-empty stem with one. The manifest's generated fields are pinned (its schema,
   its rule, each family the verb family of its task's instruction), the instruction is one text
   in the manifest, the task table and the episode table, and simulation times, step counts and
   poses are bounded. A JSON object with a duplicate key cannot be read, and a nested parquet field
   is only a list's element. Anything else is refused by name, with the file.
2. Value patterns (`scan`, `scan_file`), over every string -- keys too, and in parquet every string,
   dictionary, binary and string-view cell, every field name and every field's metadata, nested
   ones included, and the schema metadata: absolute,
   home or drive paths, URLs, e-mail addresses, UUIDs, IPv4 addresses, ISO-8601 dates and phone
   numbers; keys naming a user, an account, an object store, a credential, a nonce or a host; and
   date or time parquet columns.

`scan_tree` refuses a symbolic link or any other non-regular file anywhere in a tree, then applies
both layers.

Offline tier: pyarrow.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import stat
from collections.abc import Mapping

SCHEMA = "qvla.traces_sample/1"
RULE = ("one task per instruction verb family (first word, lower-cased): the task with the fewest corpus "
        "episodes, ties to the smaller id; then every required task not chosen yet")


def verb_family(prompt: str) -> str:
    return prompt.split()[0].lower()


def _no_duplicate_keys(pairs) -> dict:
    """json's object hook: a key twice would let the last value hide the first from the check."""
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError(f"duplicate key {k!r}")
        obj[k] = v
    return obj


def _loads(text):
    return json.loads(text, object_pairs_hook=_no_duplicate_keys)


# --- the value layer --------------------------------------------------------------------------
# Keys are matched whole. Body names are keys too (a trace's body_pose), so an object word such as
# "bucket" or "operator" is identifying only in the spellings a record of a person or a store uses.
_FORBIDDEN_KEY = re.compile(r"""(?ix) ^ (?:
      (?:\w+_)? user (?:_?\w+)?                         # user, user_id, userId, user_stage, end_user
    | (?:\w+_)? e_?mail (?:_?\w+)?                      # email, e_mail, user_email
    | (?:\w+_)? account (?:_?id|_?name)?                # account, account_id
    | \w* nonce \w*                                     # nonce, submit_nonce
    | (?:s3_?)? bucket (?:_?name)?                      # bucket, bucket_name, s3_bucket
    | (?:\w+_)? (?:secret|password|passwd|credentials?|api_?key|access_?key) \w*
    | (?:\w+_)? token
    | host (?:_?name)?
    | (?:owner|author|creator|uploader|submitter|annotator|worker|operator) _? (?:id|name|e_?mail)
    | \w+ _by                                           # submitted_by, created_by
    ) $""")
_S3_KEY = re.compile(r"(?:^|_)[sS]3(?:_|[A-Z]|$)|[a-z]S3(?:_|[A-Z]|$)")     # s3_key, trajectory_s3_key, s3Key
_AT = r"(?:^|(?<=[\s'\"=(,\[:]))"                                            # start, or after a delimiter
_FORBIDDEN_VALUE = (
    (re.compile(r"^/|(?<=[\s'\"=(,\[:])/[\w.~-]+/"), "an absolute path"),
    (re.compile(_AT + r"~[\w.-]*/|(?i:(?:^|(?<=[^\w.-]))(?:home|users|rhome)/[\w.-]+/)"), "a home path"),
    (re.compile(_AT + r"[A-Za-z]:[\\/]"), "a drive path"),
    (re.compile(r"\b[A-Za-z][\w+.-]*://"), "a URL"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "an e-mail address"),
    (re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "a UUID"),
    (re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"), "an IPv4 address"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?"), "an ISO-8601 date"),
    (re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?![\w.])"), "a phone number"),
)


def _string_findings(s: str, where: str) -> list[str]:
    return [f"{where}: {what}" for pattern, what in _FORBIDDEN_VALUE if pattern.search(s)]


def scan(obj, where: str = "") -> list[str]:
    """The value layer over a JSON value: identifying keys, and identifying strings (keys included)."""
    found = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k)
            if _FORBIDDEN_KEY.match(key) or _S3_KEY.search(key):
                found.append(f"{where}/{key}: an identifying field")
            found += _string_findings(key, f"{where}/{key} (the key)")
            found += scan(v, f"{where}/{key}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += scan(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        found += _string_findings(obj, where)
    return found


def _type_test(name: str):
    import pyarrow as pa

    return getattr(pa.types, name, lambda _t: False)


def _children(t) -> list:
    """The child fields of a nested arrow type (none for a dictionary: its values have no field)."""
    import pyarrow as pa

    if any(_type_test(n)(t) for n in ("is_list", "is_large_list", "is_fixed_size_list", "is_list_view",
                                       "is_large_list_view")):
        return [t.value_field]
    if pa.types.is_struct(t):
        return [t.field(i) for i in range(t.num_fields)]
    if pa.types.is_map(t):
        return [t.key_field, t.item_field]
    return []


def _type_holds(t, leaf) -> bool:
    import pyarrow as pa

    if pa.types.is_dictionary(t):
        return _type_holds(t.value_type, leaf)
    return leaf(t) or any(_type_holds(f.type, leaf) for f in _children(t))


def _textual(t) -> bool:
    return any(_type_test(n)(t) for n in ("is_string", "is_large_string", "is_string_view", "is_binary",
                                           "is_large_binary", "is_binary_view", "is_fixed_size_binary"))


def _temporal(t) -> bool:
    return any(_type_test(n)(t) for n in ("is_timestamp", "is_date", "is_time", "is_duration", "is_interval"))


def _fields(schema):
    """(path, parent type, field) of every field, nested ones included; a path is a tuple of names
    (column names hold dots) and a top-level field's parent type is None."""
    def walk(field, path, parent):
        yield path, parent, field
        for child in _children(field.type):
            yield from walk(child, (*path, child.name), field.type)
    for field in schema:
        yield from walk(field, (field.name,), None)


def _texts(v):
    """A to_pylist value with bytes decoded as UTF-8 (so binary cells are scanned as text)."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (list, tuple)):
        return [_texts(x) for x in v]
    if isinstance(v, dict):
        return {k: _texts(x) for k, x in v.items()}
    return v


def _metadata(md) -> dict:
    """Arrow key-value metadata, each value parsed as JSON when it is JSON."""
    out = {}
    for k, v in (md or {}).items():
        text = v.decode("utf-8", "replace")
        try:
            out[k.decode("utf-8", "replace")] = _loads(text)
        except ValueError:
            out[k.decode("utf-8", "replace")] = text
    return out


def _scan_table(table, where: str) -> list[str]:
    found = []
    for path, _parent, field in _fields(table.schema):
        found += scan({field.name: None}, f"{where} field {'.'.join(path[:-1]) or '(top)'}")
        if field.metadata:
            found += scan(_metadata(field.metadata), f"{where} field {'.'.join(path)} metadata")
    for field in table.schema:
        if _type_holds(field.type, _temporal):
            found.append(f"{where} column {field.name}: a date or time column")
        if _type_holds(field.type, _textual):
            found += scan(_texts(table.column(field.name).to_pylist()), f"{where} column {field.name}")
    return found + scan(_metadata(table.schema.metadata), f"{where} metadata")


def _parse(rel: str, data: bytes):
    """A JSON value, a list of JSON lines, or an arrow table; any other kind of file is refused."""
    suffix = pathlib.PurePosixPath(rel).suffix
    if suffix == ".json":
        return _loads(data)
    if suffix == ".jsonl":
        return [_loads(line) for line in data.decode().splitlines() if line.strip()]
    if suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pq.read_table(pa.BufferReader(data), use_threads=False)
    raise ValueError(f"{rel}: cannot scan a {suffix or 'suffix-less'} file for identifying content")


def _scan_parsed(rel: str, obj) -> list[str]:
    if isinstance(obj, (dict, str, int, float)) or obj is None:
        return scan(obj, rel)
    if isinstance(obj, list) and rel.endswith(".jsonl"):
        return [f for i, line in enumerate(obj) for f in scan(line, f"{rel}:{i + 1}")]
    if isinstance(obj, list):
        return scan(obj, rel)
    return _scan_table(obj, rel)


def scan_file(path, data: bytes | None = None) -> list[str]:
    """The value layer over one file (its bytes `data` when given): .json, .jsonl or .parquet."""
    if pathlib.PurePath(path).suffix not in (".json", ".jsonl", ".parquet"):
        return _parse(str(path), b"")                       # raises: a kind it cannot read
    data = pathlib.Path(path).read_bytes() if data is None else data
    return _scan_parsed(str(path), _parse(str(path), data))


# --- the closed shape -------------------------------------------------------------------------
# A spec: NUM (a number, a bool or null); ("num", lo, hi) (bounded; hi may name a context value);
# NULL; ("text", check, what); ("list", spec); ("empty",) (an
# empty list or object); ("obj", {key: spec}) (only these keys); ("map", key check, what, spec);
# ("any", spec, ...); ("ref", name).
NUM = ("num",)
NULL = ("null",)
MAX_SIM_S = 3600.0                          # recordings run for minutes (198.6 s at most in the sample)
MAX_POSE = 1000.0                           # metres and radians
TIME = ("num", 0.0, MAX_SIM_S)
STEPS = ("num", 0, "length")                # bounded by the episode's render rows (3 per trace step)
POSE = ("num", -MAX_POSE, MAX_POSE)
POSES = ("any", POSE, ("list", POSE))
ROBOT_NAMESPACE = "franka"
_NAME = re.compile(r"[a-z][a-z0-9_]*(?:/[a-z0-9_]*)?")           # body, joint and object names
_BODY = re.compile(r"franka/[a-z0-9_]*|[a-z][a-z0-9_]*_\d+")  # a robot body or <object>_<n>
_PROSE = re.compile(r"[A-Za-z0-9 ,;:()'./&-]*")                  # task instructions, the selection rule
ROBOT_JOINTS = frozenset({*(f"franka/panda_joint{i}" for i in range(1, 8)),
                          "franka/panda_finger_joint1", "franka/panda_finger_joint2"})
DATA_COLUMNS = frozenset({
    "action", "action_eef", "contact_eef", "contact_finger", "contact_target_eef", "contact_target_finger",
    "episode_index", "frame_index", "index", "is_anchor", "observation.body_pose",
    "observation.cameras.third_person.camera_to_world", "observation.cameras.third_person.intrinsics",
    "observation.cameras.wrist.camera_to_world", "observation.cameras.wrist.intrinsics", "observation.ee_state",
    "observation.eef_pose", "observation.joint_qpos", "observation.state", "reward", "reward_valid", "state_eef",
    "success", "task_index", "timestamp"})
IMAGE_FEATURES = frozenset({"observation.images.third_person", "observation.images.wrist"})
FEATURES = DATA_COLUMNS | IMAGE_FEATURES
INFO_NAMES = frozenset({"action", "channel", "ee_state", "height", "row_major", "state", "width"})
EPISODE_COLUMNS = frozenset({
    "episode_index", "tasks", "length", "data/chunk_index", "data/file_index", "dataset_from_index",
    "dataset_to_index", "meta/episodes/chunk_index", "meta/episodes/file_index", "attempt_id",
    "source_episode_index", "body_names", "object_body", "target_body"})
_EPISODE_STATS = re.compile(r"stats/(?P<f>.+)/(?:min|max|mean|std|count|q01|q10|q50|q90|q99)")
_EPISODE_VIDEO = re.compile(r"videos/(?P<f>.+)/(?:chunk_index|file_index|from_timestamp|to_timestamp)")


def _text(check, what: str):
    return ("text", check, what)


def _enum(*values):
    return lambda v, _c: v in values


def _rx(pattern: str):
    rx = re.compile(pattern)
    return lambda v, _c: rx.fullmatch(v) is not None


def _ns_stem(name: str) -> tuple[str, str]:
    """(namespace, stem): `franka/panda_link0` -> ("franka", "panda_link0"), `cup_3` -> ("", "cup")."""
    ns, _sep, base = name.rpartition("/")
    return ns, re.sub(r"_\d+$", "", base)


def _episode_body(v, c) -> bool:
    return v in c["bodies"]


def _trace_body(v, c) -> bool:
    return v == "world" or v in c["bodies"]


def _joint(v, c) -> bool:
    return v in ROBOT_JOINTS or (v.endswith("_free") and v[:-len("_free")] in c["bodies"])


def _task_object(v, c) -> bool:
    """A body of the task, or another instance of one: the same namespace and the same non-empty stem."""
    if _NAME.fullmatch(v) is None:
        return False
    return v in c["task_bodies"] or (_ns_stem(v)[1] != "" and _ns_stem(v) in c["task_stems"])


def _info_name(v, c) -> bool:
    return v in c["task_bodies"] or v in ROBOT_JOINTS or v in INFO_NAMES


PANDAS_TYPES = ("float64", "int64", "list[float64]", "list[int64]", "list[list[list[float64]]]", "list[unicode]",
                "object", "unicode")                        # the pandas and numpy types the corpus tables use
NUMPY_TYPES = ("float64", "int64", "object", "str")
AXIS = _text(_enum("x", "y", "z"), "an axis")
# The checker classes the sample's configurations and goal records use, and the extractor's stand-in
# for an unimplemented one.
CHECKER_CLASSES = ("AlwaysFalseChecker", "CompositeChecker", "DirectedRotationChecker", "GripperOpenChecker",
                   "PositionDeltaChecker", "RelativeCylinderChecker", "RelativePositionBoundsChecker")
CHECKER_CLASS = _text(_enum(*CHECKER_CLASSES), "a checker class")
TASK_OBJECT = _text(_task_object, "an object of the task")
PROSE = _text(lambda v, _c: _PROSE.fullmatch(v) is not None, "instruction text")
FAMILY = _text(_rx(r"[a-z]+(?:/[a-z]+)?"), "a verb family")      # and its task's (check_tree)
NUMS = ("any", NUM, ("list", NUM))
SPECS: dict[str, tuple] = {
    "trace": ("obj", {
        "event": _text(_enum("state", "final"), "a trace event"),
        "source": _text(_enum("packed_state_v1_direct"), "a trace source"),
        "state_time_mode": _text(_enum("steps_per_ctrl_post_step"), "a state time mode"),
        "step_index": STEPS, "sim_time": TIME, "final_sim_time": TIME, "total_state_frames": STEPS,
        "joint_qpos": ("map", _joint, "a joint of the episode's robot or bodies", POSES),
        "body_pose": ("map", _trace_body, "a body of the episode",
                      ("obj", {"pos": ("list", POSE), "quat": ("list", ("num", -1.0, 1.0))})),
        # Empty in every recording; a non-empty one is not declared.
        "position_errors": ("empty",), "rotation_errors_deg": ("empty",), "error_report": ("empty",),
    }),
    "goals": ("obj", {
        "attempt_id": _text(lambda v, c: v == str(c["aid"]), "the attempt id of the file name"),
        "n_steps": STEPS, "top_first_pass_frame": STEPS,
        "leaves": ("list", ("obj", {
            "path": _text(_rx(r"\d+(?:/\d+)*"), "a checker-tree path"), "checker_type": CHECKER_CLASS,
            # null on a leaf whose checker names no body (tracked) or that never passed (goal_source)
            "tracked_body_name": ("any", TASK_OBJECT, NULL),
            "resolved_body_name": _text(_episode_body, "a body of the episode"),
            "pass_frame": STEPS, "n_pass_frames": STEPS, "stable_window": ("any", STEPS, ("list", STEPS)),
            "goal_xyz": NUMS,
            "goal_quat_xyzw": NUMS, "initial_object_position": NUMS, "initial_object_quat_wxyz": NUMS,
            "goal_source": ("any", _text(_enum("first_pass", "first_pass_fallback", "stable_window",
                                               "trivial_placement"), "a goal source"), NULL),
            "goal_type": _text(_enum("pnp", "position_delta", "directed_rotation"), "a goal family"),
            "d_init_goal": NUM, "delta_thresh": NUM, "delta_axes": ("list", AXIS),
            "axis_bounds": ("list", ("obj", {"axis": AXIS, "min": NUM, "max": NUM})),
            "target_angle_deg": NUM, "angle_tol_deg": NUM, "max_tilt_deg": NUM})),
    }),
    "checker_config": ("obj", {"checker": ("ref", "checker_node")}),
    "checker_node": ("obj", {
        "type": CHECKER_CLASS, "operator": _text(_enum("AND", "OR"), "a composite operator"),
        "checkers": ("list", ("ref", "checker_node")), "axes": ("any", AXIS, ("list", AXIS)),
        "objName": TASK_OBJECT, "refName": TASK_OBJECT, "sampleBodyName": TASK_OBJECT, "bodyName": TASK_OBJECT,
        "refType": _text(_enum("object"), "a reference type"), "captureRuntimeInitial": NUM,
        **{f"{side}Delta{ax}": NUM for side in ("min", "max") for ax in "XYZ"},
        **{k: NUM for k in ("threshold", "angleToleranceDeg", "maxTiltDeg", "targetAngleDeg", "heightMin",
                            "heightMax", "xyRadius")},
        "xRange": NUMS, "yRange": NUMS, "zRange": NUMS}),
    "info": ("obj", {
        "codebase_version": _text(_enum("v3.0"), "a LeRobot version"),
        "robot_type": _text(_enum("axis_robosuite"), "a robot type"),
        **{k: NUM for k in ("total_episodes", "total_frames", "total_tasks", "chunks_size", "data_files_size_in_mb",
                            "video_files_size_in_mb", "fps")},
        "splits": ("map", _enum("train"), "a split", _text(_rx(r"\d+:\d+"), "an episode range")),
        "data_path": _text(_enum("data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"), "the data path"),
        "video_path": _text(_enum("videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"),
                            "the video path"),
        "features": ("map", lambda v, _c: v in FEATURES, "a dataset feature", ("obj", {
            "dtype": _text(_enum("bool", "float32", "int64", "video"), "a feature dtype"),
            "shape": ("list", NUM), "names": ("any", NUM, ("list", _text(_info_name, "a feature name"))),
            "info": ("obj", {"has_audio": NUM, "video.channels": NUM, "video.fps": NUM, "video.height": NUM,
                             "video.width": NUM, "video.is_depth_map": NUM,
                             "video.codec": _text(_enum("h264"), "a codec"),
                             "video.pix_fmt": _text(_enum("yuv420p"), "a pixel format")})}))}),
    "manifest": ("obj", {
        "schema": _text(_enum(SCHEMA), "this schema"), "rule": _text(_enum(RULE), "the selection rule"),
        "families": ("list", FAMILY), "required": ("list", NUM),
        "tasks": ("list", ("obj", {"task_id": NUM, "family": FAMILY, "prompt": PROSE, "episodes": ("list", ("obj", {
            "episode_index": NUM, "attempt_id": NUM, "length": NUM}))})),
        "files": ("map", lambda v, _c: _kind(v) is not None and v != "manifest.json", "a staged path",
                  _text(_rx(r"[0-9a-f]{64}"), "a sha256")),
    }),
    "pandas": ("obj", {
        "index_columns": ("list", _text(lambda v, c: v in c["columns"], "a column")),
        "column_indexes": ("list", ("ref", "pandas_column")), "columns": ("list", ("ref", "pandas_column")),
        "creator": ("obj", {"library": _text(_enum("pyarrow"), "a writer"),
                            "version": _text(_rx(r"\d+(?:\.\d+)*"), "a version")}),
        "attributes": ("empty",), "pandas_version": _text(_rx(r"\d+(?:\.\d+)*"), "a version")}),
    "pandas_column": ("obj", {
        "name": ("any", NUM, _text(lambda v, c: v in c["columns"], "a column")),
        "field_name": ("any", NUM, _text(lambda v, c: v in c["columns"], "a column")),
        "pandas_type": _text(_enum(*PANDAS_TYPES), "a pandas type of the dataset tables"),
        "numpy_type": _text(_enum(*NUMPY_TYPES), "a numpy type of the dataset tables"),
        "metadata": ("any", NUM, ("obj", {"encoding": _text(_enum("UTF-8"), "an encoding")}))}),
    "huggingface": ("obj", {"fingerprint": _text(_rx(r"[0-9a-f]{16}"), "a dataset fingerprint"),
                            "info": ("obj", {"features": ("map", lambda v, _c: v in DATA_COLUMNS, "a data column",
                                                          ("ref", "hf_feature"))})}),
    "hf_feature": ("obj", {"_type": _text(_enum("Value", "List", "Sequence"), "a feature type"),
                           "dtype": _text(_enum("float32", "int64", "bool"), "a dtype"),
                           "feature": ("ref", "hf_feature"), "length": NUM}),
}
# Parquet kinds: which columns exist, the domain of each text column, and which metadata keys.
COLUMNS = {
    "data": lambda n: n in DATA_COLUMNS,
    "episodes": lambda n: n in EPISODE_COLUMNS or any(
        (m := rx.fullmatch(n)) is not None and m["f"] in allowed
        for rx, allowed in ((_EPISODE_STATS, FEATURES), (_EPISODE_VIDEO, IMAGE_FEATURES))),
    "tasks": lambda n: n in ("task_index", "__index_level_0__"),
}
NAME = _text(lambda v, _c: _BODY.fullmatch(v) is not None, "a robot body or an <object>_<n> body name")
TEXT_COLUMNS = {"data": {}, "tasks": {"__index_level_0__": PROSE},
                "episodes": {"tasks": ("list", PROSE), "body_names": ("list", NAME), "object_body": NAME,
                             "target_body": NAME}}
METADATA = {"data": {"huggingface"}, "episodes": {"pandas"}, "tasks": {"pandas"}}
_LAYOUT = (
    (re.compile(r"manifest\.json"), "manifest"),
    (re.compile(r"traces/task_(?P<t>\d+)_attempt_(?P<a>\d+)\.jsonl"), "trace"),
    (re.compile(r"goals_per_attempt/task_(?P<t>\d+)_attempt_(?P<a>\d+)\.json"), "goals"),
    (re.compile(r"data/task_(?P<t>\d+)/checker_config\.json"), "checker_config"),
    (re.compile(r"corpus/task_(?P<t>\d+)/meta/info\.json"), "info"),
    (re.compile(r"corpus/task_(?P<t>\d+)/meta/tasks\.parquet"), "tasks"),
    (re.compile(r"corpus/task_(?P<t>\d+)/meta/episodes/chunk-\d{3}/file-\d{3}\.parquet"), "episodes"),
    (re.compile(r"corpus/task_(?P<t>\d+)/(?:data)/chunk-\d{3}/file-\d{3}\.parquet"), "data"),
)


def _kind(rel: str) -> tuple[str, int | None, int | None] | None:
    """(kind, task id, attempt id) of a path in a staged sample, or None for a path it never holds."""
    for rx, kind in _LAYOUT:
        m = rx.fullmatch(rel)
        if m:
            g = m.groupdict()
            return kind, (int(g["t"]) if "t" in g else None), (int(g["a"]) if "a" in g else None)
    return None


def _check(spec, v, ctx, where: str, out: list[str]) -> None:
    tag = spec[0]
    if tag == "ref":
        _check(SPECS[spec[1]], v, ctx, where, out)
    elif tag == "null":
        if v is not None:
            out.append(f"{where}: a {type(v).__name__} where null is declared")
    elif tag == "num":
        if not (v is None or isinstance(v, (bool, int, float))):
            out.append(f"{where}: a {type(v).__name__} where a number is declared")
        elif len(spec) == 3 and v is not None:
            lo, hi = spec[1], (ctx.get(spec[2], 0) if isinstance(spec[2], str) else spec[2])
            if not lo <= v <= hi:                           # NaN is outside every range
                out.append(f"{where}: {v!r} is outside the declared range [{lo}, {hi}]")
    elif tag == "text":
        if not isinstance(v, str):
            out.append(f"{where}: a {type(v).__name__} where {spec[2]} is declared")
        elif not spec[1](v, ctx):
            out.append(f"{where}: {v!r} is not {spec[2]}")
    elif tag == "empty":
        if v not in ([], {}):
            out.append(f"{where}: holds a value where only an empty one is declared")
    elif tag == "list":
        if not isinstance(v, list):
            out.append(f"{where}: a {type(v).__name__} where a list is declared")
        else:
            for i, x in enumerate(v):
                _check(spec[1], x, ctx, f"{where}[{i}]", out)
    elif tag == "any":
        trials = []
        for alt in spec[1:]:
            trial: list[str] = []
            _check(alt, v, ctx, where, trial)
            if not trial:
                return
            trials.append(trial)
        out.extend(min(trials, key=len))               # ties report the first alternative
    elif not isinstance(v, dict):
        out.append(f"{where}: a {type(v).__name__} where an object is declared")
    elif tag == "obj":
        for k, x in v.items():
            if k not in spec[1]:
                out.append(f"{where}/{k}: a key not declared here")
            else:
                _check(spec[1][k], x, ctx, f"{where}/{k}", out)
    else:                                                   # "map"
        _tag, key_ok, what, value_spec = spec
        for k, x in v.items():
            if not key_ok(k, ctx):
                out.append(f"{where}/{k}: a key that is not {what}")
            else:
                _check(value_spec, x, ctx, f"{where}/{k}", out)


def _check_table(kind: str, table, ctx, where: str, out: list[str]) -> None:
    names = table.schema.names
    ctx = {**ctx, "columns": set(names)}
    for name in names:
        if not COLUMNS[kind](name):
            out.append(f"{where}: column {name!r} is not declared")
    for path, parent, field in _fields(table.schema):
        if field.metadata:
            out.append(f"{where}: field {'.'.join(path)} carries metadata, which is not declared")
        if parent is not None and not (_children(parent) == [field] and field.name in ("element", "item")
                                       and not _type_test("is_struct")(parent)):
            out.append(f"{where}: nested field {'.'.join(path)!r} is not declared (only a list's element is)")
    for field in table.schema:
        if COLUMNS[kind](field.name) and _type_holds(field.type, _textual):
            spec = TEXT_COLUMNS[kind].get(field.name)
            if spec is None:
                out.append(f"{where}: column {field.name!r} holds text, which is not declared")
                continue
            for i, v in enumerate(_texts(table.column(field.name).to_pylist())):
                _check(spec, v, ctx, f"{where} column {field.name}[{i}]", out)
    for key, value in _metadata(table.schema.metadata).items():
        if key not in METADATA[kind]:
            out.append(f"{where}: metadata key {key!r} is not declared")
        else:
            _check(SPECS[key], value, ctx, f"{where} metadata {key}", out)
    if kind == "episodes" and {"body_names", "object_body", "target_body"} <= set(names):
        cols = table.select(["body_names", "object_body", "target_body"]).to_pylist()
        for i, row in enumerate(cols):
            for c in ("object_body", "target_body"):
                if row[c] is not None and row[c] not in (row["body_names"] or []):
                    out.append(f"{where} column {c}[{i}]: {row[c]!r} is not a body of the episode")


def _context(parsed: Mapping, kinds: Mapping) -> dict:
    """Each staged episode's body names and length, by (task, attempt), and each task's bodies."""
    episodes: dict[tuple[int, int], set[str]] = {}
    lengths: dict[tuple[int, int], int] = {}
    for rel, (kind, tid, _a) in kinds.items():
        if kind == "episodes" and {"attempt_id", "body_names", "length"} <= set(parsed[rel].schema.names):
            for row in parsed[rel].select(["attempt_id", "body_names", "length"]).to_pylist():
                names = row["body_names"] or []
                episodes[(tid, int(row["attempt_id"]))] = {n for n in names if isinstance(n, str)}
                lengths[(tid, int(row["attempt_id"]))] = int(row["length"] or 0)
    tasks: dict[int, set[str]] = {}
    for (tid, _a), names in episodes.items():
        tasks.setdefault(tid, set()).update(names)
    return {"episodes": episodes, "lengths": lengths, "tasks": tasks}


def _check_manifest(man: dict, parsed: Mapping, kinds: Mapping, out: list[str]) -> None:
    """The manifest's families are its tasks' verb families, its required tasks are its tasks, each
    task's episodes are its staged episode table's rows, and each task's instruction is one text in
    the manifest, its task table and its episode table."""
    staged_rows: dict[int, list] = {}
    for rel, (kind, tid, _a) in kinds.items():
        names = set(parsed[rel].schema.names) if kind == "episodes" else set()
        if {"episode_index", "attempt_id", "length"} <= names:
            staged_rows.setdefault(tid, []).extend(
                (r["episode_index"], r["attempt_id"], r["length"])
                for r in parsed[rel].select(["episode_index", "attempt_id", "length"]).to_pylist())
    tasks = [t for t in man.get("tasks") or [] if isinstance(t, dict)]
    for t in tasks:
        listed = sorted((e.get("episode_index"), e.get("attempt_id"), e.get("length"))
                        for e in t.get("episodes") or [] if isinstance(e, dict))
        if listed != sorted(staged_rows.get(t.get("task_id"), [])):
            out.append(f"manifest.json/tasks/{t.get('task_id')}/episodes: not the (episode_index, attempt_id, "
                       f"length) rows of the task's staged episode table")
    required = man.get("required")
    if isinstance(required, list):
        for r in required:
            if r not in {t.get("task_id") for t in tasks}:
                out.append(f"manifest.json/required: {r!r} is not a task of the manifest")
    prompts = {}
    for t in man.get("tasks") or []:
        if isinstance(t, dict) and isinstance(t.get("prompt"), str) and t["prompt"].split():
            prompts[t.get("task_id")] = t["prompt"]
            if t.get("family") != verb_family(t["prompt"]):
                out.append(f"manifest.json/tasks/{t.get('task_id')}/family: {t.get('family')!r} is not the verb "
                           f"family of its instruction ({verb_family(t['prompt'])!r})")
    families = man.get("families")
    if isinstance(families, list):
        task_families = {verb_family(p) for p in prompts.values()}
        for f in families:
            if f not in task_families:
                out.append(f"manifest.json/families: {f!r} is the verb family of no task's instruction")
        if all(isinstance(f, str) for f in families) and families != sorted(set(families)):
            out.append("manifest.json/families: not sorted and unique")
    for rel, (kind, tid, _a) in kinds.items():
        if kind not in ("tasks", "episodes"):
            continue
        if tid not in prompts:
            out.append(f"{rel}: task {tid} is not a task of manifest.json")
            continue
        col, want = ("__index_level_0__", prompts[tid]) if kind == "tasks" else ("tasks", [prompts[tid]])
        if col not in parsed[rel].schema.names:
            out.append(f"{rel}: holds no instruction column {col!r}")
            continue
        for i, v in enumerate(_texts(parsed[rel].column(col).to_pylist())):
            if v != want:
                out.append(f"{rel} column {col}[{i}]: {v!r} is not the manifest's instruction for task {tid}")


def check_tree(files: Mapping[str, bytes]) -> list[str]:
    """Both layers over a staged sample given as {path in the tree: bytes}; every finding, by file."""
    found: list[str] = []
    kinds, parsed = {}, {}
    for rel in sorted(files):
        k = _kind(rel)
        if k is None:
            found.append(f"{rel}: not a file a traces sample holds")
            continue
        try:
            parsed[rel] = _parse(rel, files[rel])
        except ValueError as exc:
            found.append(f"{rel}: cannot be read ({exc})")
            continue
        kinds[rel] = k
    ctx_all = _context(parsed, kinds)
    for rel, (kind, tid, aid) in kinds.items():
        obj = parsed[rel]
        found += _scan_parsed(rel, obj)
        task_bodies = ctx_all["tasks"].get(tid, set())
        ctx = {"aid": aid, "task_bodies": task_bodies,
               "task_stems": {_ns_stem(n) for n in task_bodies if _ns_stem(n)[1]},
               "bodies": ctx_all["episodes"].get((tid, aid), set()),
               "length": ctx_all["lengths"].get((tid, aid), 0)}
        if kind in COLUMNS:
            _check_table(kind, obj, ctx, rel, found)
            continue
        if kind in ("trace", "goals") and (tid, aid) not in ctx_all["episodes"]:
            found.append(f"{rel}: no staged episode of task {tid} is attempt {aid}")
        for i, item in enumerate(obj if kind == "trace" else [obj]):
            _check(SPECS[kind], item, ctx, f"{rel}:{i + 1}" if kind == "trace" else rel, found)
    if "manifest.json" not in files:
        found.append("manifest.json: the sample's manifest is missing")
    if "manifest.json" in parsed and isinstance(parsed["manifest.json"], dict):
        _check_manifest(parsed["manifest.json"], parsed, kinds, found)
        listed = parsed["manifest.json"].get("files") or {}
        for rel in sorted(set(files) - {"manifest.json"} - set(listed)):
            found.append(f"{rel}: not listed in manifest.json")
        for rel, sha in sorted(listed.items()):
            if rel not in files:
                found.append(f"{rel}: listed in manifest.json but not in the tree")
            elif hashlib.sha256(files[rel]).hexdigest() != sha:
                found.append(f"{rel}: its bytes are not the sha256 manifest.json lists")
    return found


def scan_tree(root) -> list[str]:
    """Every finding in a staged sample on disk: a symbolic link or other non-regular file anywhere,
    then both layers over its files."""
    root = pathlib.Path(root)
    found, files = [], {}

    def refuse(exc: OSError) -> None:
        raise exc

    for dirpath, dirnames, filenames in os.walk(root, onerror=refuse):
        dirnames.sort()
        for name in sorted(dirnames + filenames):
            p = pathlib.Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            mode = os.lstat(p).st_mode
            if stat.S_ISLNK(mode):
                found.append(f"{rel}: a symbolic link")
            elif name in filenames:
                if stat.S_ISREG(mode):
                    files[rel] = p.read_bytes()
                else:
                    found.append(f"{rel}: not a regular file")
    return found + check_tree(files)
