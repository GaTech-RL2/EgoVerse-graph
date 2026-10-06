"""Dense routed Action Flow simulator boundary; no replacement sampler/model."""

from collections.abc import Mapping
from contextlib import nullcontext

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes_sim import _env_to_zarr_pushshapes_oriented

_PREFIX = "egomimic.pipeline.stages_action_flow."
_BOUNDARIES = {
    "pushshapes_sim_u_socket": (4, 3, "USocketModelStateObservationAdapter", "USocketRotVecNativeDecoder", "u_socket"),
    "pushshapes_sim_chain_gripper": (6, 4, "ChainGripperModelStateObservationAdapter", "ChainGripperPointsNativeDecoder", "chain_gripper"),
}


def is_routed_action_flow(cfg):
    return any(str(stage.get("_target_", "")) == _PREFIX + "ConditionalVelocityStage"
               for stage in OmegaConf.select(cfg, "model.pipeline.stages", default=[])) and bool(
                   OmegaConf.select(cfg, "deployment.action_decoders"))


def action_flow_contract(cfg, *, selected_embodiment_name, selected_embodiment_id,
                         expected_native_action_dim, action_chunk_start_index=0,
                         replan_every=None):
    """Fail closed on the reviewed dense H16, pre-step, routed Euler contract."""
    if selected_embodiment_name not in _BOUNDARIES:
        raise ValueError("unsupported routed Action Flow embodiment")
    width, native_width, adapter_name, decoder_name, _ = _BOUNDARIES[selected_embodiment_name]
    if selected_embodiment_id != int(get_embodiment_id(selected_embodiment_name)) or expected_native_action_dim != native_width:
        raise ValueError("Action Flow embodiment/native width mismatch")
    for field, expected in {"action_horizon": 16, "raw_action_horizon": 16,
                            "observation_horizon": 1, "action_target_offset": 0}.items():
        if OmegaConf.select(cfg, "planar." + field) != expected:
            raise ValueError(f"Action Flow requires planar.{field}={expected}")
    if OmegaConf.select(cfg, "planar.action_dims." + selected_embodiment_name) != width:
        raise ValueError("Action Flow model action width mismatch")
    if action_chunk_start_index != 0 or replan_every not in (None, 8):
        raise ValueError("Action Flow requires chunk start 0 and canonical replan 8")
    stages = OmegaConf.select(cfg, "model.pipeline.stages", default=[])
    velocity = [s for s in stages if s.get("_target_") == _PREFIX + "ConditionalVelocityStage"]
    if len(velocity) != 1:
        raise ValueError("Action Flow requires exactly one conditional velocity stage")
    stage = velocity[0]
    if stage.get("inference_method") != "euler" or int(stage.get("num_inference_steps", 0)) != 50:
        raise ValueError("reviewed routed Action Flow requires checkpoint Euler-50")
    for kind, name in (("observation_adapters", adapter_name), ("action_decoders", decoder_name)):
        item = OmegaConf.select(cfg, f"deployment.{kind}.{selected_embodiment_name}")
        if item is None or item.get("_target_") != "egomimic.pipeline.pushshapes." + name:
            raise ValueError(f"Action Flow {kind} boundary mismatch")
    decoder = instantiate(cfg.deployment.action_decoders[selected_embodiment_name])
    # Constructor and stochastic-axis checks are cheap and run before loading
    # the full checkpoint. Native decoders must preserve all leading axes.
    probe = torch.zeros(1, 2, 16, width)
    if width == 4:
        probe[..., 2] = 1.0
    decoded = decoder.decode(probe, context={"state_agent_obj": torch.zeros(1, 2, 1, 6)})
    if tuple(decoded.shape) != (1, 2, 16, native_width) or not torch.isfinite(decoded).all():
        raise ValueError("Action Flow decoder failed stochastic shape/finite probe")
    return decoder


def action_flow_metadata(cfg):
    stage = next(s for s in cfg.model.pipeline.stages if s.get("_target_") == _PREFIX + "ConditionalVelocityStage")
    return {"bridge": "pipeline_routed_action_flow_h16_v1", "sampler": "euler",
            "sampler_inference_steps": int(stage.num_inference_steps),
            "cfg_scale": float(stage.cfg_scale), "cfg_interval": list(stage.cfg_interval),
            "euler_time_grid": "uniform_reverse_1_to_0",
            "timestep_shift_alpha": float(stage.timestep_shift_alpha),
            "timestep_shift_active": False, "timing_semantics": "dense_fixed_rate",
            "replan_semantics": "dense_chunk_prefix_before_next_inference"}


class RoutedActionFlowPolicy:
    def __init__(self, *, algo, normalizer, decoder, embodiment_id, device, cfg,
                 embodiment_name, model_autocast_precision="fp32"):
        self.algo, self.normalizer, self.decoder = algo, normalizer, decoder
        self.embodiment_id, self.device = embodiment_id, device
        self.embodiment_name = embodiment_name
        if model_autocast_precision not in ("fp32", "bf16"):
            raise ValueError("model autocast precision must be fp32 or bf16")
        if model_autocast_precision == "bf16" and (
                device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 model autocast requires a BF16-capable CUDA device")
        self.model_autocast_precision = model_autocast_precision
        self.model_width, self.native_width, _, _, _ = _BOUNDARIES[embodiment_name]
        self.adapter = instantiate(cfg.deployment.observation_adapters[embodiment_name])
        self.token_horizon = self.decoded_horizon = 16

    def reset(self):
        pass

    @torch.inference_mode()
    def predict_native_actions(self, obs_env):
        # The checkpoint's FusedObsEncoder(n_obs_steps=1) consumes packed
        # [B,...] rows, not [B,1,...]. An extra temporal axis survives feature
        # concatenation and produces an invalid [B,1,C] condition.
        raw = _env_to_zarr_pushshapes_oriented(dict(obs_env), self.device)
        adapted = self.adapter.encode(raw)
        normalized = self.normalizer.normalize(adapted, self.embodiment_id)
        normalized["embodiment"] = torch.tensor([self.embodiment_id], device=self.device)
        context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if self.model_autocast_precision == "bf16" else nullcontext())
        with context:
            prediction = self.algo.forward_eval({self.embodiment_name: normalized})
        result = prediction.get(self.embodiment_name)
        tokens = result.get("pred_action") if isinstance(result, Mapping) else None
        if not torch.is_tensor(tokens) or tuple(tokens.shape) != (1, 16, self.model_width):
            raise RuntimeError("Action Flow returned invalid model action shape")
        if not torch.isfinite(tokens).all():
            raise RuntimeError("Action Flow returned non-finite model actions")
        # Native decoding and simulator arithmetic remain FP32; never cast
        # checkpoint parameters or the Euler integration state to BF16.
        actions = self.normalizer.unnormalize({"actions": tokens.float()}, self.embodiment_id)["actions"]
        # IK orientation context must remain in raw native coordinates; never
        # feed normalized proprio or an unexecuted future prediction back here.
        native = self.decoder.decode(actions, context=raw)
        if not torch.is_tensor(native):
            native = torch.as_tensor(native, device=self.device)
        if tuple(native.shape) != (1, 16, self.native_width) or not torch.isfinite(native).all():
            raise RuntimeError("Action Flow returned invalid/non-finite native actions")
        return native[0].float().cpu().numpy().astype(np.float32, copy=False)
