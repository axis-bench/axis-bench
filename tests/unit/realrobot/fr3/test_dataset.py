"""A finetune set is held to its embodiment before it is trusted, and its frame geometry is measured from
its pixels, never read from a marker file or assumed."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from axis_qvla.realrobot.fr3 import dataset
from axis_qvla.realrobot.fr3.embodiment import load_embodiment
from tests.unit.realrobot.fr3.fr3_sets import BAR, SIZE, crop_frame, edit_info, fr3_set, letterbox_frame

FR3 = load_embodiment("fr3")


def test_centre_cropped_frames_measure_as_center_crop224(tmp_path):
    assert dataset.training_geometry(fr3_set(tmp_path / "crop")) == "center_crop224"


def test_letterboxed_frames_measure_as_letterbox224(tmp_path):
    assert dataset.training_geometry(fr3_set(tmp_path / "lb", letterbox_frame)) == "letterbox224"


def test_a_marker_file_does_not_decide_the_geometry(tmp_path):
    """Only the pixels do: a letterboxed set with a crop224 report is still letterboxed."""
    root = fr3_set(tmp_path / "lb", letterbox_frame)
    (root / "meta" / "crop224_report.json").write_text("{}")
    assert dataset.training_geometry(root) == "letterbox224"


def _dim_bars(e, r, c):
    x = letterbox_frame(e, r, c)
    x[:BAR] = x[SIZE - BAR:] = 30          # neither black bars nor content
    return x


def _black(e, r, c):
    return np.zeros((SIZE, SIZE, 3), np.uint8)          # dark bands, but no content between them either


def _grey_padded(e, r, c):
    x = letterbox_frame(e, r, c)
    x[:BAR] = x[SIZE - BAR:] = 128                    # a letterbox padded with grey: lit, but flat bands
    return x


def _mixed(e, r, c):
    return (letterbox_frame if e == 1 else crop_frame)(e, r, c)


def _cameras_differ(e, r, c):
    return (letterbox_frame if c == 1 else crop_frame)(e, r, c)


@pytest.mark.parametrize("frame, said", [
    (_dim_bars, "9 neither"),
    (_black, "9 neither"),
    (_grey_padded, "9 neither"),
    (_mixed, "3 letterboxed"),
])
def test_frames_that_are_neither_are_a_named_problem_that_says_what_was_measured(tmp_path, frame, said):
    root = fr3_set(tmp_path / "amb", frame)
    with pytest.raises(dataset.DatasetError) as exc:
        dataset.training_geometry(root)
    msg = str(exc.value)
    assert msg.startswith(f"{root}: ") and "image: of 9 sampled frames" in msg and said in msg
    assert "brightest 1%" in msg


def test_cameras_that_disagree_are_a_named_problem(tmp_path):
    root = fr3_set(tmp_path / "two", _cameras_differ)
    with pytest.raises(dataset.DatasetError, match="image center_crop224, extra_view_image letterbox224"):
        dataset.training_geometry(root)


def _widen(info, col, width):
    info["features"][col]["shape"] = [width]
    return info


@pytest.mark.parametrize("edit, named", [
    (lambda i: _widen(i, "actions", 7), "actions has shape [7], the fr3 embodiment's is [8]"),
    (lambda i: _widen(i, "state", 9), "state has shape [9], the fr3 embodiment's is [8]"),
    (lambda i: {**i, "fps": 10}, "fps 10, the fr3 embodiment's is 15"),
    (lambda i: {**i, "fps": 15.9}, "fps 15.9, the fr3 embodiment's is 15"),
    (lambda i: {**i, "features": {("observation.images.exterior" if k == "image" else
                                   "observation.images.wrist" if k == "extra_view_image" else k):
                                  ({**f, "dtype": "video"} if f["dtype"] == "image" else f)
                                  for k, f in i["features"].items()}},
     "camera columns {'observation.images.exterior': 'video', 'observation.images.wrist': 'video'}, the fr3 "
     "embodiment's are {'image': 'image', 'extra_view_image': 'image'}"),
])
def test_a_set_that_is_not_the_embodiments_is_refused_by_name(tmp_path, edit, named):
    root = edit_info(fr3_set(tmp_path / "x"), edit)
    with pytest.raises(dataset.DatasetError) as exc:
        dataset.open_set(root, FR3)
    assert str(exc.value).startswith(f"{root}: not an fr3 training set: ") and named in str(exc.value)


def test_frames_of_another_size_are_refused(tmp_path):
    from tests.unit.realrobot.realdata.test_export import _world

    root = _world(tmp_path).root                      # 16 x 16 frames
    with pytest.raises(dataset.DatasetError, match=r"image has shape \[16, 16, 3\], the fr3 embodiment's is \[224, 224, 3\]"):
        dataset.open_set(root, FR3)


@pytest.mark.parametrize("text", ["", "{}"])
def test_an_empty_info_json_is_refused(tmp_path, text):
    (tmp_path / "e" / "meta").mkdir(parents=True)
    (tmp_path / "e" / "meta" / "info.json").write_text(text)
    with pytest.raises(dataset.DatasetError, match=f"^{tmp_path / 'e'}: cannot read the fr3 training set"):
        dataset.open_set(tmp_path / "e", FR3)


def test_a_conforming_set_opens_and_its_data_must_be_as_wide_as_its_metadata(tmp_path):
    root = fr3_set(tmp_path / "ok")
    assert len(dataset.open_set(root, FR3).episodes) == 3
    from axis_qvla.realrobot.fr3 import norm

    p = root / "data" / "chunk-000" / "episode_000001.parquet"
    t = pq.read_table(p)
    narrow = pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(len(t) * 7, np.float32)), 7)
    pq.write_table(t.set_column(t.schema.get_field_index("actions"), "actions", narrow), p)
    with pytest.raises(ValueError, match="actions are 7 wide, the fr3 embodiment's are 8"):
        norm.columns(root, "fr3")
