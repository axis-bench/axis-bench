"""Websocket serving shim: SmolVLA / GR00T N1.7 / X-VLA checkpoints behind openpi's wire protocol.

WHY: the AXIS rollout harness (the rollout CLI `python -m axis_qvla.bench.sim.cli`, driven by the
campaign driver axis_qvla.bench.sim.campaign and its queue workers) speaks openpi's websocket
protocol — a msgpack `server_metadata` handshake followed by obs-dict -> action-chunk
request/reply pairs. Serving a lerobot backbone through
the same protocol means Table-1 runs are evaluated by the EXACT harness that scored pi0.5,
with zero client changes.

ENVIRONMENT: lerobot (conf/envs/lerobot.toml), on a GPU host. The wire layer itself needs only
`websockets` + `openpi_client` (pure python: msgpack + numpy); lerobot/torch are imported
lazily inside the real-predictor constructors, so the protocol test runs GPU-free in any
venv that has the client package, driving the built-in `--policy-type mock` predictor.

METADATA PARITY: the rollout CLI reads `cfg_conditioning`/`guidance_scale` off this handshake
(axis_qvla.bench.sim.policy.resolve_conditioning) and names the run directory after them; its
`--cfg-conditioning`/`--guidance-scale` flags, when given, are an assertion and the run is
refused if the server disagrees. Mirroring openpi's `create_trained_policy` (policy_config.py:
a `quality_tag` turns metadata cfg_conditioning ON), a `--quality-tag` serve publishes
cfg_conditioning=True, guidance_scale=0.0 — so a tag-conditioned rollout lands its logs in
`<variant>_w0_dualsim/`, exactly like openpi's conditioned w=0 results, and a client asserting
`--no-cfg-conditioning` against a tagged server is refused rather than silently mislabelled.
An untagged serve publishes False/0.0, so plain rollouts land in `<variant>_dualsim/` and any
guidance-asserting client is refused. `--guidance-scale` exists only to be recorded and
checked: these backbones have no CFG sampling math, so any value other than 0 is refused at
launch.
`checkpoint_dir`/`config_name` are published so the run record, not stdout, says what
answered (ten evaluations sharing port 8000 is how numbers get attributed to the wrong run).

Usage (the campaign driver's X-VLA command, axis_qvla.bench.sim.campaign.XVLA_SERVER_CMD):
    python axis_qvla/bench/servers/xvla.py --policy-type xvla \
        --ckpt <xvla checkpoint>/pretrained_model --host 127.0.0.1 --port 8000 --domain-id 20
"""
from __future__ import annotations

# serve_lerobot.py plus the xvla backbone, kept as a SEPARATE file so a concurrently running
# SmolVLA campaign's serve_lerobot.py keeps its own digest; may be folded back together later.

import argparse
import asyncio
import http
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Protocol

import numpy as np

try:
    from openpi_client import msgpack_numpy
except ImportError as e:  # pragma: no cover - environment guidance, not logic
    raise ImportError(
        "serve_lerobot needs the openpi client package for the msgpack wire encoding "
        "(pure python; deps: msgpack, numpy). Add <repo>/openpi/packages/openpi-client/src "
        "to PYTHONPATH or `pip install -e` it into this venv."
    ) from e
import websockets.asyncio.server as _ws_server
import websockets.frames

# The quality-tag rule lives in one place (stdlib, core tier). `bench eval` puts the checkout on
# the server's PYTHONPATH; a server started by hand as a script from a checkout adds that checkout.
try:
    from axis_qvla.bench.results.recorded import parse_quality_tag
except ImportError:  # pragma: no cover - `python axis_qvla/bench/servers/<server>.py` with no PYTHONPATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from axis_qvla.bench.results.recorded import parse_quality_tag

logger = logging.getLogger(__name__)

# The action-chunk contract of every Table-1 arm: droid8 = 7 joint velocities + gripper,
# H=10. The CLIENT owns chunking/replanning; the server just returns the whole chunk.
# Backbones whose native horizon is longer (SmolVLA chunk_size=50, GR00T N1.7 native 40)
# are TRUNCATED to the first 10 steps -- no interpolation. Both backbones train on the
# same 15 fps FT data, so step k of their chunk and step k of the droid8 contract are the
# same wall-clock instant; the first 10 steps ARE the aligned prefix.
HORIZON = 10
ACTION_DIM = 8

# ---------------------------------------------------------------------------------------
# Quality-tag prompt suffix.
#
# `--quality-tag 5` must condition the backbone the same way openpi's slb_cfg conditions
# pi0.5: by appending "\nQuality: 5" to the instruction. The key below is a DUPLICATE of
# `openpi.training.slb_cfg.QUALITY_KEY` rather than an import (this module runs in the
# lerobot venv, which has no openpi package); byte-equality of the full suffix against
# openpi's `apply_metadata` is pinned by a tag-suffix test, which runs in the openpi venv where
# both are importable.
QUALITY_TAG_KEY = "Quality"
# Mirrors quality_conditioning.PROMPT_TAG_MARKER: the only trace a tag leaves in a prompt.
QUALITY_TAG_MARKER = f"\n{QUALITY_TAG_KEY}: "


def quality_tag_suffix(q: int) -> str:
    """The exact bytes appended to a prompt for quality bin `q`."""
    return f"\n{QUALITY_TAG_KEY}: {int(q)}"


def apply_quality_tag(prompt: str, q: int) -> str:
    """Append the quality tag, refusing a double tag.

    A client that already tagged its prompt (e.g. a cfg-aware harness) plus a server-side
    `--quality-tag` would condition on "...\\nQuality: 5\\nQuality: 5" -- in range,
    tokenizable, and matched by no training prompt. Same guard as openpi's
    AxisQualityConditioning."""
    prompt = str(prompt)
    if QUALITY_TAG_MARKER in prompt:
        raise ValueError(
            f"prompt {prompt[:120]!r} already carries a quality tag ({QUALITY_TAG_MARKER!r}); "
            "either the client tagged it or --quality-tag is applied twice."
        )
    return prompt + quality_tag_suffix(q)

# Wire keys (what the rollout drivers transmit -- see rollout_pi05_dualsim._build_obs and
# CAMERA_TO_OBS_KEY in rollout_pi05_render.py) -> backbone-agnostic Predictor keys.
WIRE_TO_PREDICTOR_KEYS = {
    "base_0_rgb": "image",          # third-person 224x224 HWC uint8
    "left_wrist_0_rgb": "wrist_image",
    "state": "state",               # 8-D float32
    "prompt": "prompt",             # task instruction string
}


class Predictor(Protocol):
    """One backbone = one Predictor. `reset()` is called at every new client connection
    (the openpi protocol has no reset message; a connection IS an episode session)."""

    def reset(self) -> None: ...

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        """obs: {image, wrist_image, state, prompt} -> float32 action chunk (HORIZON, 8)."""
        ...


def map_obs(wire_obs: dict[str, Any], quality_tag: int | None = None) -> dict[str, Any]:
    """Rename openpi wire keys to backbone-agnostic ones, validating the contract loudly.

    The 8-D state check mirrors AxisFrankaInputs: a 9-D state means an unconverted corpus
    and must fail here, at the server, not as a silently wrong rollout.

    `quality_tag` appends the openpi-equal "\\nQuality: <q>" suffix to the prompt, which is
    how training conditions a tagged arm; applying it HERE (before the predictor) keeps the
    tagged prompt on the exact path parity_check.py certifies."""
    missing = [k for k in WIRE_TO_PREDICTOR_KEYS if k not in wire_obs]
    if missing:
        raise KeyError(f"obs is missing wire keys {missing}; got {sorted(wire_obs)}")
    state = np.asarray(wire_obs["state"], dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(
            f"state has shape {state.shape}; the droid8 contract is (8,) "
            "[7 joint positions + gripper]. A 9-D state is the width of unconverted pretraining data."
        )
    prompt = str(wire_obs["prompt"])
    if quality_tag is not None:
        prompt = apply_quality_tag(prompt, quality_tag)
    return {
        "image": np.asarray(wire_obs["base_0_rgb"], dtype=np.uint8),
        "wrist_image": np.asarray(wire_obs["left_wrist_0_rgb"], dtype=np.uint8),
        "state": state,
        "prompt": prompt,
    }


def build_metadata(
    *, checkpoint_dir: str, config_name: str, quality_tag: int | None = None,
    guidance_scale: float = 0.0, noise_horizon: int = 50, noise_dim: int = 32,
) -> dict[str, Any]:
    """The handshake dict the rollout CLI reads its conditioning from and checks assertions against.

    Mirrors openpi's `create_trained_policy` metadata semantics: a quality tag IS the
    conditioning, so a tagged serve publishes cfg_conditioning=True (and the rollout client
    must say `--cfg-conditioning --guidance-scale 0`), while an untagged serve publishes
    False/0.0 so plain rollouts pass and guidance-expecting clients are refused. There is
    no CFG sampling math in these backbones, so a nonzero guidance_scale is refused here
    rather than silently published as a value the predictor does not implement."""
    if float(guidance_scale) != 0.0:
        raise ValueError(
            f"guidance_scale={guidance_scale} but the lerobot backbones implement no CFG "
            "sampling math; only w=0 (plain conditional sampling) can be served honestly."
        )
    md: dict[str, Any] = {
        "cfg_conditioning": quality_tag is not None,
        "guidance_scale": 0.0,
        # The client refuses to run without these: a server that ignored __noise__ would
        # silently fall back to its unseeded global generator while the run claimed to be
        # pinned. Shape is the SAMPLER's native chunk, not the truncated AXIS contract.
        "supports_client_noise": True,
        "action_horizon": int(noise_horizon),
        "action_dim": int(noise_dim),
        "checkpoint_dir": str(checkpoint_dir),
        "config_name": str(config_name),
    }
    if quality_tag is not None:
        md["quality_tag"] = int(quality_tag)
    return md


class MockPredictor:
    """Protocol-test predictor: zeros, with the predict-count-since-reset stamped into
    `actions[0, 0]` so per-connection reset semantics are observable over the wire."""

    def __init__(self) -> None:
        self._n = 0

    def reset(self) -> None:
        self._n = 0

    def predict(self, obs: dict[str, Any], noise: np.ndarray | None = None) -> np.ndarray:
        chunk = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
        chunk[0, 0] = float(self._n)
        self._n += 1
        return chunk


class LerobotPredictor:
    """Wraps a lerobot PreTrainedPolicy + its saved pre/post processor pipelines.

    Constructed only by `build_predictor` (lazy torch/lerobot imports live there); this
    class itself only holds references, so the module stays importable without lerobot."""

    def __init__(self, policy, preprocessor, postprocessor, torch_mod) -> None:
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._torch = torch_mod

    def reset(self) -> None:
        self._policy.reset()

    def build_policy_batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Raw obs dict -> the pre-preprocessor batch, exactly as `predict` builds it.

        Split out so parity_check.py can run THIS code (not a copy) against the training
        loader's batch construction and compare the preprocessor outputs key by key."""
        torch = self._torch
        device = next(self._policy.parameters()).device
        # Corpus feature names as the LeRobot corpus build keeps them (KEPT_FEATURES); images go
        # torch CHW float[0,1] (lerobot convention), state batched (1, 8), task = prompt.
        return {
            "observation.images.third_person": _to_chw01(obs["image"], torch, device),
            "observation.images.wrist": _to_chw01(obs["wrist_image"], torch, device),
            "observation.state": torch.from_numpy(obs["state"]).unsqueeze(0).to(device),
            "task": [obs["prompt"]],
        }

    def preprocess(self, obs: dict[str, Any]) -> dict[str, Any]:
        """The serve-side model inputs for `obs` (parity_check's unit of comparison)."""
        torch = self._torch
        with torch.no_grad():
            return self._pre(self.build_policy_batch(obs))

    def predict(self, obs: dict[str, Any], noise: np.ndarray | None = None) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            batch = self._pre(self.build_policy_batch(obs))
            # Caller-owned sampler noise. Without it the flow-matching sampler draws from
            # torch's GLOBAL generator, which nothing seeds -- so every server process
            # starts from OS entropy and no rollout can be reproduced.
            nz = None
            if noise is not None:
                nz = torch.from_numpy(np.ascontiguousarray(noise, dtype=np.float32))
                if nz.ndim == 2:
                    nz = nz.unsqueeze(0)
                nz = nz.to(next(self._policy.parameters()).device)
            actions = self._policy.predict_action_chunk(batch, noise=nz)  # (1, chunk_size, dim)
            actions = self._post(actions)
        chunk = np.asarray(actions.squeeze(0).to("cpu", torch.float32).numpy())
        if chunk.shape[0] < HORIZON or chunk.shape[1] < ACTION_DIM:
            raise ValueError(f"backbone returned chunk {chunk.shape}, need >= ({HORIZON}, {ACTION_DIM})")
        return np.ascontiguousarray(chunk[:HORIZON, :ACTION_DIM], dtype=np.float32)


def _to_chw01(img_hwc_u8: np.ndarray, torch, device):
    """uint8 HWC -> float32 (1, C, H, W) in [0, 1], bit-identical to the training loader.

    The /255 MUST happen on CPU: lerobot's video decode does `.type(torch.float32) / 255`
    on CPU (correctly-rounded IEEE division), while CUDA's scalar division lowers to a
    reciprocal multiply that is 1 ULP off for some byte values -- parity_check measured
    max|diff| = 5.96e-08 (= 2^-24) on every frame when this division ran on GPU. One ULP
    never moves a rollout, but the parity gate is exact by design and the CPU division is free."""
    t = torch.from_numpy(np.ascontiguousarray(img_hwc_u8))
    t = t.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    return t.to(device)


def build_predictor(policy_type: str, ckpt: str, stats_dataset: str | None = None, domain_id: int | None = None) -> Predictor:
    """Lazy-import constructor: ONLY this function touches torch/lerobot, so the mock
    path (and therefore the protocol test) runs in any venv with websockets.

    `stats_dataset` (a LeRobot dataset root) is REQUIRED when `ckpt` is a raw GR00T N1.7
    base checkpoint (e.g. nvidia/GR00T-N1.7-3B): a raw base carries no lerobot processor
    pipelines and no per-embodiment statistics, so the GR00T pack/decode steps need the
    dataset's min/max stats and the policy config needs the dataset's feature shapes --
    without them `make_pre_post_processors` raises (no stats) or decodes 0-dim actions
    (no output_features). A FINETUNED lerobot checkpoint saves both, so trained arms
    (the Task-8 eval path) need no flag; passing one anyway just forwards stats that the
    saved-pipeline path ignores."""
    if policy_type == "mock":
        return MockPredictor()

    import torch  # noqa: PLC0415 - deliberate lazy import, see module docstring

    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415

    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as cls  # noqa: PLC0415
    elif policy_type == "xvla":
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy as cls  # noqa: PLC0415
    elif policy_type == "groot":
        from lerobot.policies.groot.modeling_groot import GrootPolicy as cls  # noqa: PLC0415
    else:
        raise ValueError(f"unknown policy type {policy_type!r}")

    # Kernel selection must not depend on GPU contention at load time. cudnn.benchmark is
    # torch's analogue of XLA autotuning; it already defaults to False, and it is pinned off
    # so a future torch cannot flip it under a campaign. allow_tf32 stays as-is (it is
    # deterministic within a GPU model) and is logged with the run.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info(
        "numerics: cudnn.benchmark=%s cudnn.deterministic=%s cudnn.allow_tf32=%s matmul.allow_tf32=%s",
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats_meta = None
    if stats_dataset is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: PLC0415

        stats_meta = LeRobotDatasetMetadata(repo_id="local/stats", root=stats_dataset)

    config = None
    if policy_type == "groot":
        from lerobot.policies.groot.configuration_groot import (  # noqa: PLC0415
            GrootConfig,
            is_raw_groot_n1_7_checkpoint,
        )

        if is_raw_groot_n1_7_checkpoint(ckpt):
            if stats_meta is None:
                raise ValueError(
                    f"{ckpt!r} is a raw GR00T N1.7 base checkpoint: it has no processor "
                    "pipelines, no embodiment stats and no feature shapes. Pass "
                    "--stats-dataset <ft dataset root> so both can come from the dataset."
                )
            from lerobot.configs import FeatureType  # noqa: PLC0415
            from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: PLC0415

            # Mirror lerobot.policies.factory.make_policy's feature wiring so the config
            # matches what a training run over the same dataset would have produced.
            features = dataset_to_policy_features(stats_meta.features)
            config = GrootConfig(device=device)
            config.output_features = {
                k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
            }
            config.input_features = {
                k: ft for k, ft in features.items() if k not in config.output_features
            }

    policy = cls.from_pretrained(ckpt) if config is None else cls.from_pretrained(ckpt, config=config)
    policy.to(device)
    policy.eval()
    processor_kwargs: dict[str, Any] = {
        "preprocessor_overrides": {"device_processor": {"device": device}},
    }
    if policy_type == "xvla":
        # The wire observation lands as observation.images.{third_person,wrist}; X-VLA checkpoints
        # declare observation.images.{image,image2}. A checkpoint finetuned by our fork already
        # saves this rename map; the raw lerobot/xvla-base does not, so it is set here either way.
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {"rename_map": {
            "observation.images.third_person": "observation.images.image",
            "observation.images.wrist": "observation.images.image2"}}
        if domain_id is not None:
            processor_kwargs["preprocessor_overrides"]["xvla_add_domain_id"] = {"domain_id": int(domain_id)}
    if stats_meta is not None:
        processor_kwargs["dataset_stats"] = stats_meta.stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt,
        **processor_kwargs,
    )
    return LerobotPredictor(policy, preprocessor, postprocessor, torch)


class PredictorPolicyServer:
    """Openpi-compatible websocket loop (mirrors openpi.serving.websocket_policy_server:
    metadata frame on connect, then obs -> {"actions", "server_timing"} replies; errors are
    sent as a traceback TEXT frame then close(1011), which the openpi client surfaces as
    RuntimeError). Re-implemented rather than imported so the serving venv does not need
    the openpi package — only openpi_client."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        host: str,
        port: int,
        metadata: dict,
        quality_tag: int | None = None,
    ) -> None:
        self._predictor = predictor
        self._host = host
        self._port = port
        self._metadata = metadata
        self._quality_tag = quality_tag

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("serving on %s:%s | metadata=%s", self._host, self._port, self._metadata)
            await server.serve_forever()

    async def _handler(self, websocket: _ws_server.ServerConnection) -> None:
        logger.info("connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        self._predictor.reset()  # a connection is an episode session
        try:
            await websocket.send(packer.pack(self._metadata))
        except websockets.ConnectionClosed:
            # A readiness probe opens and closes at once; its close may land before the handshake.
            logger.info("connection from %s closed before the handshake", websocket.remote_address)
            return
        first_request = True
        while True:
            try:
                start = time.monotonic()
                wire_obs = msgpack_numpy.unpackb(await websocket.recv())
                noise = wire_obs.pop('__noise__', None)
                obs = map_obs(wire_obs, quality_tag=self._quality_tag)
                if first_request:
                    # Train/eval geometry parity is only observable HERE: log what the wire
                    # actually delivered, so a pre-squared 224x224 harness (the center_crop
                    # FOV mismatch) is visible in the server log of the first rollout.
                    first_request = False
                    logger.info(
                        "first obs: image=%s %s wrist=%s %s state=%s prompt=%r",
                        obs["image"].shape, obs["image"].dtype,
                        obs["wrist_image"].shape, obs["wrist_image"].dtype,
                        obs["state"].shape, obs["prompt"],
                    )
                chunk = np.asarray(self._predictor.predict(obs, noise=noise))
                if chunk.shape != (HORIZON, ACTION_DIM) or chunk.dtype != np.float32:
                    raise ValueError(
                        f"predictor returned {chunk.dtype} {chunk.shape}; the contract is "
                        f"float32 ({HORIZON}, {ACTION_DIM})"
                    )
                reply = {
                    "actions": chunk,
                    "server_timing": {"infer_ms": (time.monotonic() - start) * 1000},
                }
                await websocket.send(packer.pack(reply))
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                # TEXT frame + 1011 close is the openpi convention the client detects
                # (`isinstance(response, str)` -> RuntimeError with this traceback).
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                break


def _health_check(connection: _ws_server.ServerConnection, request: _ws_server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-type", required=True, choices=["smolvla", "xvla", "groot", "mock"])
    ap.add_argument("--domain-id", type=int, default=None,
                    help="xvla only: override the soft-prompt domain row saved in the checkpoint's preprocessor")
    ap.add_argument(
        "--config-name", default=None,
        help="the openpi/LeRobot train config name published in server_metadata and the run "
        "record; defaults to the published config name for --policy-type when not given",
    )
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (any string for mock)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--quality-tag", type=parse_quality_tag, default=None,
        help="append openpi's exact '\\nQuality: <q>' suffix to every prompt (tag conditioning) "
        "and publish the tag in server_metadata so the run record names it",
    )
    ap.add_argument(
        "--guidance-scale", type=float, default=0.0,
        help="recorded + published only (no CFG math exists in these backbones); any value "
        "other than 0 is refused. Present so a tag-conditioned serve line mirrors openpi's "
        "and the run record names w explicitly.",
    )
    ap.add_argument(
        "--stats-dataset", default=None,
        help="LeRobot dataset root supplying normalization stats + feature shapes; REQUIRED "
        "when --ckpt is a raw GR00T N1.7 base, ignored for finetuned lerobot checkpoints "
        "(their saved processor pipelines carry the training stats)",
    )
    return ap


def _resolved_config_name(args: argparse.Namespace) -> str:
    """The metadata config_name: an explicit --config-name, or the published default for this
    policy type (the name cells.json's eval_command passes as --config)."""
    return args.config_name or f"axis_{args.policy_type}"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    predictor = build_predictor(args.policy_type, args.ckpt, stats_dataset=args.stats_dataset, domain_id=args.domain_id)
    # The noise shape is the SAMPLER's native chunk, read off the loaded model -- never a
    # constant. GR00T N1.7's native horizon is 40 where SmolVLA's is 50, and a mismatched
    # shape reaching sample_noise would be broadcast or truncated rather than raising.
    _pol = getattr(predictor, "_policy", None)
    _pcfg = getattr(_pol, "config", None)
    _nh = getattr(_pcfg, "chunk_size", None)
    _nd = getattr(_pcfg, "max_action_dim", None)
    if _pol is not None and (_nh is None or _nd is None):
        raise SystemExit(
            f"cannot read the sampler's noise shape off {type(_pol).__name__}.config; "
            "refusing to advertise a guessed shape"
        )
    metadata = build_metadata(
        checkpoint_dir=args.ckpt,
        config_name=_resolved_config_name(args),
        quality_tag=args.quality_tag,
        guidance_scale=args.guidance_scale,
        noise_horizon=50 if _nh is None else int(_nh),
        noise_dim=32 if _nd is None else int(_nd),
    )
    PredictorPolicyServer(
        predictor, host=args.host, port=args.port, metadata=metadata,
        quality_tag=args.quality_tag,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
