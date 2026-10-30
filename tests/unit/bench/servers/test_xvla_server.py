"""--config-name defaults to the published config name for --policy-type (the name cells.json's
eval_command passes as --config), and an explicit --config-name overrides it verbatim; --quality-tag
takes a quality bin and refuses the untagged sentinel 0 before anything is served."""
from __future__ import annotations

import pytest

from axis_qvla.bench.servers.xvla import _resolved_config_name, build_parser


def test_config_name_defaults_to_the_published_name():
    args = build_parser().parse_args(["--policy-type", "xvla", "--ckpt", "x"])
    assert _resolved_config_name(args) == "axis_xvla"


def test_config_name_override_is_published_verbatim():
    args = build_parser().parse_args(
        ["--policy-type", "smolvla", "--ckpt", "x", "--config-name", "my_smolvla_config"])
    assert _resolved_config_name(args) == "my_smolvla_config"


@pytest.mark.parametrize("tag, match", [("0", "untagged sentinel"), ("-1", "not a quality bin"), ("x", "not a quality bin")])
def test_a_tag_that_is_not_a_bin_stops_the_server_at_parse_time(tag, match, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--policy-type", "xvla", "--ckpt", "x", "--quality-tag", tag])
    assert match in capsys.readouterr().err
    assert build_parser().parse_args(["--policy-type", "xvla", "--ckpt", "x", "--quality-tag", "5"]).quality_tag == 5
