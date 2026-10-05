"""Shared scalar speed conditioning, identical in train and inference."""
import math
import torch
from torch import nn
from egomimic.pipeline.core import Stage


def build_speed_conditioned_pipeline(stages, speed_reference, encoding="scalar",
                                     condition_dim=128, device=None,
                                     compatibility_mode="current"):
    """Typed Action Flow graph adapter; leave the generic runner unchanged.

    Configure Hydra with _recursive_: false so the two consumers are wired
    before their read contracts are constructed. No stage is silently skipped.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from egomimic.pipeline.algo import PipelineAlgo
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
                   SharedSpeedCondition(speed_reference, encoding, condition_dim))
    return PipelineAlgo(modules, device=device, compatibility_mode=compatibility_mode)


class SharedSpeedCondition(Stage):
    def __init__(self, speed_reference, encoding="scalar", condition_dim=128,
                 hidden_dim=32, condition_key="condition", speed_key="requested_speed",
                 output_key="speed_condition", initialization_seed=42):
        super().__init__()
        if encoding not in {"scalar", "fourier"}:
            raise ValueError("encoding must be scalar or fourier")
        if not math.isfinite(speed_reference) or speed_reference <= 0:
            raise ValueError("speed_reference must be positive and train-derived")
        self.encoding = encoding
        self.condition_key, self.speed_key, self.output_key = condition_key, speed_key, output_key
        self.reads, self.writes = (condition_key, speed_key), (output_key,)
        self.register_buffer("speed_reference", torch.tensor(float(speed_reference)))
        # Added layers must not shift initialization of the unchanged model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.mlp = nn.Sequential(nn.Linear(1 if encoding == "scalar" else 5, hidden_dim),
                                     nn.SiLU(), nn.Linear(hidden_dim, condition_dim))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, batch):
        condition = batch[self.condition_key]
        speed = batch[self.speed_key].to(device=condition.device, dtype=torch.float32)
        if speed.ndim != 2 or speed.shape != (condition.shape[0], 1):
            raise ValueError("requested_speed must have shape [B,1]")
        if not torch.isfinite(speed).all() or (speed < 0).any():
            raise ValueError("requested_speed must be finite and nonnegative")
        u = torch.log1p(speed / self.speed_reference)
        features = u if self.encoding == "scalar" else torch.cat(
            (u, u.sin(), u.cos(), (2*u).sin(), (2*u).cos()), dim=-1)
        delta = self.mlp(features)
        if condition.ndim == 3:
            delta = delta.unsqueeze(1)
        elif condition.ndim != 2:
            raise ValueError("condition must be [B,D] or [B,T,D]")
        batch[self.output_key] = condition + delta.to(condition.dtype)
        return batch
