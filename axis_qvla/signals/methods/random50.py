"""Random-50% controls for the two top-50% filtering granularities (paper Fig. 4(c)).

OFFLINE TIER: numpy + pandas only.

Each control keeps the same number of training frames as its quality-filtered twin and differs
from it ONLY in the selection rule; both then take the same schedule draw (uniform permutation
batches over the kept frames, about two epochs at the backbone's budget). Never re-drawn per epoch.

  episode control  `lerobot_schedule_drop_top_rand50_ep_<suffix>.npz`
      mode `drop_top` over SYNTHETIC per-episode constant weights drawn uniformly from (1, 2)
      with `--control-seed`: rank episodes at random, keep whole episodes until half the
      trainable frames. The boundary episode is cut by `top_fraction_mask`'s stable order exactly
      as the quality-filtered episode runs cut theirs.
  chunk control    `lerobot_schedule_drop_random_rand50_row_<suffix>.npz`
      mode `drop_random` (`random_fraction_mask`, seeded by `--control-seed`): half the trainable
      frames uniformly at random. Frames index the untouched pretraining data, so a kept frame's
      action chunk reads its original following frames.

The synthetic weights carry no reward: their keys and lengths are the template's, and every
value is > 1, so no episode is "unlabelled". Their provenance block names the template by
`--template-name` and its sha256; the released controls were drawn from the object-centric chunk
weights under the file name `awr_weights_v2.json`, and both strings are hashed into the
schedule's `config_hash`, so the name must be given exactly to reproduce the released bytes.

The budget flags default to the SmolVLA run's budget (20,621 x 64, keep 0.50, schedule seed 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from axis_qvla.signals.methods import schedule as build_index_schedule

SEED_CONTROL = 42
PRETRAIN_STEPS, PRETRAIN_GLOBAL_BATCH, DROP_KEEP_TOP_FRAC, SCHEDULE_SEED = 20621, 64, 0.5, 0


def synthetic_weights_doc(src: dict, *, template_sha256: str, template_name: str, reward_id: str,
                          per_episode_random: bool, seed: int = SEED_CONTROL) -> dict:
    """A weights document with the template's keys and lengths but values that carry NO reward:
    per-episode constant, either uniform-random in (1, 2) (episode control: only the rank matters)
    or 1.5 everywhere (chunk control: values are ignored by drop_random)."""
    rng = np.random.default_rng(seed)
    keys = sorted(src["weights"])  # sorted -> the draw is a function of the key set, not dict order
    w = {}
    for k in keys:
        n = len(src["weights"][k]); v = (1.0 + rng.random()) if per_episode_random else 1.5
        w[k] = [float(v)] * n
    return {"version": 1, "reward_id": reward_id, "tau": src["tau"], "cap": src["cap"], "horizon": src["horizon"],
            "n_episodes": src["n_episodes"], "n_rows": src["n_rows"], "n_neutral_rows": 0, "granularity": "episode",
            "provenance": {"control": "random-50% (owner 2026-09-15)", "seed": seed,
                           "keys_from": template_name,
                           "template_sha256": template_sha256, "carries_reward": False},
            "weights": w}


def synthetic_weights(template: Path, out: Path, reward_id: str, *, per_episode_random: bool,
                      template_name: str | None = None, seed: int = SEED_CONTROL) -> Path:
    """Write `synthetic_weights_doc` of `template` to `out` (its bytes are what the schedule hashes)."""
    template = Path(template)
    doc = synthetic_weights_doc(json.loads(template.read_text()),
                                template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
                                template_name=template_name or template.name, reward_id=reward_id,
                                per_episode_random=per_episode_random, seed=seed)
    Path(out).write_text(json.dumps(doc)); return Path(out)


def build(mode: str, weights: Path, reward: str, seed: int, *, roots, ranges, out_dir: Path,
          suffix: str, total_steps: int, batch: int, keep: float) -> Path:
    out = Path(out_dir) / f"lerobot_schedule_{mode}_{reward}_{suffix}.npz"
    if out.exists():
        out.unlink()
    build_index_schedule.main(["--mode", mode, "--weights", str(weights), "--roots", str(roots),
                               "--ranges", str(ranges), "--out", str(out), "--total-steps", str(total_steps),
                               "--batch", str(batch), "--seed", str(seed), "--keep-top-frac", str(keep)])
    with np.load(out, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])); rows = z["rows"]
    assert rows.shape == (total_steps, batch), rows.shape
    assert meta["mode"] == mode and meta["reward_id"] == reward, meta
    assert np.unique(rows.ravel()).size == meta["unique_frames"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="axis_qvla.signals.methods.random50", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", required=True, help="weights JSON whose keys/lengths the controls copy")
    ap.add_argument("--template-name", default=None,
                    help="name recorded as the template (default: the file's own name)")
    ap.add_argument("--roots", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", default="smolvla", help="backbone the budget belongs to")
    ap.add_argument("--total-steps", type=int, default=PRETRAIN_STEPS)
    ap.add_argument("--batch", type=int, default=PRETRAIN_GLOBAL_BATCH)
    ap.add_argument("--keep-top-frac", type=float, default=DROP_KEEP_TOP_FRAC)
    ap.add_argument("--schedule-seed", type=int, default=SCHEDULE_SEED,
                    help="draw seed of the episode control (the quality-filtering runs' own seed)")
    ap.add_argument("--control-seed", type=int, default=SEED_CONTROL,
                    help="seed of the synthetic episode ranking and of the random row subset")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(roots=a.roots, ranges=a.ranges, out_dir=out_dir, suffix=a.suffix,
                  total_steps=a.total_steps, batch=a.batch, keep=a.keep_top_frac)
    ep_w = synthetic_weights(Path(a.template), out_dir / "rand50_ep_weights.json", "rand50_ep",
                             per_episode_random=True, template_name=a.template_name, seed=a.control_seed)
    row_w = synthetic_weights(Path(a.template), out_dir / "rand50_row_weights.json", "rand50_row",
                              per_episode_random=False, template_name=a.template_name, seed=a.control_seed)
    outs = [build("drop_top", ep_w, "rand50_ep", a.schedule_seed, **common),
            build("drop_random", row_w, "rand50_row", a.control_seed, **common)]
    for o in outs:
        print(f"{o.name}: sha256={hashlib.sha256(o.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
