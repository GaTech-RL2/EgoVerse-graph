"""Shared scalar speed conditioning, identical in train and inference."""
import math
import torch
from torch import nn
from egomimic.pipeline.core import Stage


def build_speed_conditioned_pipeline(stages, speed_reference=None, encoding="scalar",
                                     condition_dim=128, device=None,
                                     compatibility_mode="current",
                                     conditioning_input="retiming_multiplier"):
    """Typed Action Flow graph adapter; leave the generic runner unchanged.

    Configure Hydra with _recursive_: false so the two consumers are wired
    before their read contracts are constructed. No stage is silently skipped.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from egomimic.pipeline.algo import PipelineAlgo
    if conditioning_input != "retiming_multiplier" or speed_reference is not None:
        raise ValueError("Precomputed-speed conditioning is retired; use the raw multiplier")
    configs = OmegaConf.to_container(stages, resolve=True)
    bridges = [c for c in configs if c.get("_target_", "").endswith(".LatentBridgeStage")]
    fields = [c for c in configs if c.get("_target_", "").endswith(".ConditionalVelocityStage")]
    encoders = [c for c in configs if c.get("_target_", "").endswith(
        (".RoutedContentEncoderStage", ".ContentEncoderStage"))]
    if len(bridges) != 1 or len(fields) != 1 or len(encoders) != 1:
        raise ValueError("Speed adapter requires one codec stage, bridge and shared field")
    if bridges[0].get("condition_key", "condition") != "condition":
        raise ValueError("Unexpected existing bridge condition")
    if fields[0].get("inference_condition_key", "condition") != "condition":
        raise ValueError("Unexpected existing inference condition")
    bridges[0]["condition_key"] = "speed_condition"
    fields[0]["inference_condition_key"] = "speed_condition"
    # Instantiate the unchanged stages first, preserving all old RNG draws.
    # Diagnostic preprocessing stops at the content encoder: condition must
    # already exist there, even though the encoder itself does not consume it.
    modules = [instantiate(c) for c in configs]
    modules.insert(configs.index(encoders[0]),
                   SharedSpeedCondition(speed_reference, encoding, condition_dim,
                                        conditioning_input=conditioning_input))
    return PipelineAlgo(modules, device=device, compatibility_mode=compatibility_mode)


class SharedSpeedCondition(Stage):
    def __init__(self, speed_reference=None, encoding="scalar", condition_dim=128,
                 hidden_dim=32, condition_key="condition", speed_key="retiming_rate",
                 output_key="speed_condition", initialization_seed=42,
                 conditioning_input="retiming_multiplier"):
        super().__init__()
        if encoding not in {"scalar", "fourier"}:
            raise ValueError("encoding must be scalar or fourier")
        if conditioning_input != "retiming_multiplier" or speed_key != "retiming_rate":
            raise ValueError("Precomputed-speed conditioning is retired; use retiming_multiplier")
        if speed_reference is not None:
            raise ValueError("Precomputed speed_reference is retired; use the raw multiplier")
        self.conditioning_input = conditioning_input
        # Strict reload must reject retired physical-speed checkpoints.
        self.register_buffer("retiming_multiplier_contract", torch.tensor(1))
        self.encoding = encoding
        self.condition_key, self.speed_key, self.output_key = condition_key, speed_key, output_key
        self.reads, self.writes = (condition_key, speed_key), (output_key,)
        # Added layers must not shift initialization of the unchanged model.
        with torch.random.fork_rng(devices=[]):
            # These layers initialize on CPU. torch.manual_seed also reseeds
            # CUDA generators, which fork_rng(devices=[]) does not restore.
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(
                initialization_seed).get_state())
            self.mlp = nn.Sequential(nn.Linear(1 if encoding == "scalar" else 5, hidden_dim),
                                     nn.SiLU(), nn.Linear(hidden_dim, condition_dim))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, batch):
        condition = batch[self.condition_key]
        multiplier = batch[self.speed_key].to(device=condition.device, dtype=torch.float32)
        if multiplier.ndim != 2 or multiplier.shape != (condition.shape[0], 1):
            raise ValueError(f"{self.speed_key} must have shape [B,1]")
        if not torch.isfinite(multiplier).all() or (multiplier <= 0).any():
            raise ValueError("retiming_rate must be finite and strictly positive")
        u = multiplier
        features = u if self.encoding == "scalar" else torch.cat(
            (u, u.sin(), u.cos(), (2*u).sin(), (2*u).cos()), dim=-1)
        delta = self.mlp(features)
        if condition.ndim == 3:
            delta = delta.unsqueeze(1)
        elif condition.ndim != 2:
            raise ValueError("condition must be [B,D] or [B,T,D]")
        batch[self.output_key] = condition + delta.to(condition.dtype)
        return batch


def requested_rollout_condition(cfg):
    """Resolve the checkpoint-selected conditioning contract without fallback."""
    from omegaconf import OmegaConf
    if OmegaConf.select(cfg, "model.pipeline._target_") != (
            "egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline"):
        return None, None
    kind = OmegaConf.select(cfg, "model.pipeline.conditioning_input")
    if kind == "retiming_multiplier":
        key, path = "retiming_rate", "deployment.requested_multiplier"
        if OmegaConf.select(cfg, "deployment.requested_speed") is not None:
            raise ValueError("Multiplier rollout must not supply deployment.requested_speed")
    else:
        raise ValueError("Precomputed requested_speed conditioning is retired; explicit retiming_multiplier required")
    value = OmegaConf.select(cfg, path)
    if value is None:
        raise ValueError(f"Conditioned rollout requires explicit {path}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {path}")
    return key, value
