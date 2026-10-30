"""Where each published v1.0 file, and each file made from them on this machine, lives locally.

Why: `qvla data get` writes the published repositories and training and evaluation read them. A
path that is right for the writer and wrong for the reader fails only at the end of a long run, so
this module is the one answer: `qvla data get` and `qvla train` call it. `qvla bench eval` takes
its directories as explicit flags, which name the ones `qvla data get` fills here
(docs/reference/data.md).

- A dataset repository (corpus, bench, results, realworld) is mirrored at `<corpus root>/<repo name>/`, each
  file at its repository path, e.g. `<corpus>/qvla-v1-corpus/episodes/task_1000/...`.
- A model repository (the checkpoints of the three backbones) is mirrored at
  `<cache root>/checkpoints/<repo name>/`, e.g. `<cache>/checkpoints/qvla-v1-pi05/ft/<name>/13579/`.
- The base weights, fetched from their publishers (axis_qvla.data.bases), are one local copy at
  `<cache root>/checkpoints/qvla-v1-base-weights/`: openpi's download cache under `openpi/`, a
  Hugging Face cache under `hf/`.
- A file computed on this machine from published data (a signal bundle compiled here, a merged
  LeRobot copy) goes to `<cache root>/<kind>/<name>`, never inside a mirrored repository, whose
  bytes the release lock pins and `qvla data verify` compares.

Core tier: stdlib only.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

ORG = "axisrobotics"
DATASET_REPOS = ("qvla-v1-corpus", "qvla-v1-bench", "qvla-v1-results", "qvla-v1-realworld")
MODEL_REPOS = ("qvla-v1-pi05", "qvla-v1-smolvla", "qvla-v1-xvla")
BASE_WEIGHTS = "qvla-v1-base-weights"   # the local copy's directory under <cache>/checkpoints/
CORPUS = f"{ORG}/qvla-v1-corpus"
BENCH = f"{ORG}/qvla-v1-bench"
# Directories inside the mirrored repositories that code names.
CORPUS_EPISODES = "episodes"         # the per-task corpus: what a roots index's task directories are relative to
CORPUS_LEROBOT = "lerobot_merged"    # the merged LeRobot copy of the corpus
CORPUS_BUNDLES = "bundles"           # the paper's compiled signal bundles
# Published normalisation statistics, by name -> (repository, directory holding <asset id>/norm_stats.json).
NORM_STATS = {
    "heldout24_qual_v2": (BENCH, "norm_stats/heldout24_qual_v2"),   # the fine-tuning set's
    "libero5k_v2_d8": (CORPUS, "norm_stats/libero5k_v2_d8"),        # the pretraining corpus's (every pi0.5 pretrain)
}
LOCAL_KINDS = ("bundles", "derived")


class LayoutError(ValueError):
    """A repository, reference or path this layout does not place."""


def _name(repo: str) -> str:
    org, _, name = repo.rpartition("/")
    if org not in ("", ORG) or name not in DATASET_REPOS + MODEL_REPOS:
        raise LayoutError(f"{repo!r} is not a published v1.0 repository "
                          f"({', '.join(f'{ORG}/{n}' for n in DATASET_REPOS + MODEL_REPOS)})")
    return name


def relative_path(path: str) -> PurePosixPath:
    """`path` as a path inside a repository. Absolute, or climbing out with `..`, is refused:
    joined to a local directory, the first discards the directory and the second leaves it."""
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        raise LayoutError(f"{path!r} is not a path inside a repository")
    return p


def parse_ref(ref: str) -> tuple[str, str]:
    """`<org>/<repo>@<tag>:<path>` (a cells.json `checkpoint.ref`) or `<org>/<repo>:<path>` (a
    registry `dest`) -> (`<org>/<repo>`, `<path>` without trailing slashes): the one ref grammar,
    which `checkpoint_dest` and `qvla data get` both read. The tag is optional. A path that is
    empty or all slashes ("/", "//", ...) is refused, never read as the whole repository -- a
    checkpoint ref always names a checkpoint -- and so is one that leaves the repository."""
    head, sep, path = ref.partition(":")
    repo = head.partition("@")[0]
    rel = path.rstrip("/")
    if not sep or not rel or not repo:
        raise LayoutError(f"{ref!r} is not <org>/<repo>[@<tag>]:<path>")
    relative_path(rel)
    return repo, rel


def default_dest(repo: str, paths, path: str = "") -> Path:
    """The local location of `path` inside `repo` (the repository itself when `path` is empty): the
    directory `qvla data get` fills and every reader opens. `paths` is an axis_qvla.core.paths.Paths."""
    name = _name(repo)
    root = paths.corpus / name if name in DATASET_REPOS else paths.cache / "checkpoints" / name
    return root / relative_path(path) if path else root


def base_weights_dest(paths) -> Path:
    """The local copy of the base weights: what `qvla data get base-weights` fills and every run reads."""
    return paths.cache / "checkpoints" / BASE_WEIGHTS


def checkpoint_dest(ref: str, paths) -> Path:
    """A checkpoint ref (`parse_ref`) -> the local directory of that checkpoint."""
    repo, rel = parse_ref(ref)
    return default_dest(repo, paths, rel)


def norm_stats_dest(name: str, paths) -> Path:
    """The published normalisation statistics `name` (the directory holds `<asset id>/norm_stats.json`)."""
    if name not in NORM_STATS:
        raise LayoutError(f"no published normalisation statistics {name!r} (published: {', '.join(NORM_STATS)})")
    repo, path = NORM_STATS[name]
    return default_dest(repo, paths, path)


def local_dest(kind: str, name: str, paths) -> Path:
    """Where this machine keeps what it computed from published data: `<cache>/<kind>/<name>`
    (kind `bundles`: a signal bundle compiled here, by content id; `derived`: anything else)."""
    if kind not in LOCAL_KINDS:
        raise LayoutError(f"local kind {kind!r} is not one of {LOCAL_KINDS}")
    return paths.cache / kind / relative_path(name)
