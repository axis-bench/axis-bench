"""Training reads where `qvla data get` writes: every path comes from axis_qvla.data.layout."""
from __future__ import annotations

from axis_qvla.data.layout import BENCH, CORPUS, checkpoint_dest, default_dest, local_dest, norm_stats_dest
from axis_qvla.data.userconf import user_paths
from axis_qvla.train import ids
from tests.unit.train._arms import H, PUB, host


def test_the_qvla_toml_gives_the_roots(tmp_path):
    toml, paths = host(tmp_path)
    assert user_paths(str(toml)).corpus == paths.corpus


def test_published_inputs_are_where_the_layout_puts_them(tmp_path):
    _, paths = host(tmp_path)
    assert ids.checkpoint_dir(paths, PUB) == checkpoint_dest("axisrobotics/qvla-v1-pi05:pretrain/example/20604", paths)
    assert ids.bundle_dirs(paths, f"sha256:{H}")[0] == default_dest(CORPUS, paths, f"bundles/{H}")
    assert ids.rowspace_record_dirs(paths, f"x#sha256:{H}")[0] == default_dest(CORPUS, paths, f"bundles/rowspaces/{H}")
    for name in ("heldout24_qual_v2", "libero5k_v2_d8"):
        assert ids.norm_dir(paths, f"{name}#sha256:{H}") == norm_stats_dest(name, paths)
    ft = ids.rowspace_dirs(paths, f"heldout24_ft#sha256:{H}")
    assert ft["roots"] == default_dest(BENCH, paths, "ft_demos/splits/qual_v2.roots.json")
    assert ft["corpus_base"] == default_dest(BENCH, paths)
    pt = ids.rowspace_dirs(paths, f"libero5k_v2_d8#sha256:{H}")
    assert pt["corpus_base"] == default_dest(CORPUS, paths, "episodes")
    assert pt["lerobot_dataset"] == default_dest(CORPUS, paths, "lerobot_merged")
    assert ids.base_weights_root(paths) == paths.cache / "checkpoints" / "qvla-v1-base-weights"


def test_what_this_machine_makes_never_lands_in_a_mirrored_repository(tmp_path):
    _, paths = host(tmp_path)
    mirrored = [default_dest(r, paths) for r in ("qvla-v1-corpus", "qvla-v1-bench", "qvla-v1-pi05")]
    # `norm_dir` and `bundle_dirs`/`rowspace_record_dirs` never read conf/recipes/rowspaces.toml,
    # so a local id exercises "local id -> local_dest, never a mirrored repo" without a row space
    # declared there. `heldout24_ft`'s `lerobot_dataset` covers a *published* row space whose
    # LeRobot copy is still built locally (`lerobot_local`), the other half of that claim.
    local = [ids.bundle_dirs(paths, f"sha256:{H}")[1], ids.rowspace_record_dirs(paths, f"x#sha256:{H}")[1],
             ids.norm_dir(paths, f"smoke_2x3#sha256:{H}"),
             ids.rowspace_dirs(paths, f"heldout24_ft#sha256:{H}")["lerobot_dataset"]]
    assert local == [local_dest("bundles", H, paths), local_dest("bundles", f"rowspaces/{H}", paths),
                     local_dest("derived", "norm/smoke_2x3", paths), local_dest("derived", "ft_lerobot_merged", paths)]
    assert not any(p.is_relative_to(m) for p in local for m in mirrored)
