"""Shared scalar speed conditioning, identical in train and inference."""

import math

import torch
from torch import nn

from egomimic.pipeline.core import Stage


def build_speed_conditioned_pipeline(
    stages,
    speed_reference=None,
    encoding="scalar",
    condition_dim=128,
    device=None,
    compatibility_mode="current",
    conditioning_input=None,
    flow_inference_method=None,
    dit_checkpoint_policy=None,
):
    """Typed Action Flow graph adapter; leave the generic runner unchanged.

    Configure Hydra with _recursive_: false so the two consumers are wired
    before their read contracts are constructed. No stage is silently skipped.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from egomimic.pipeline.algo import PipelineAlgo

    if conditioning_input is None:
        conditioning_input = (
            "native_speed" if speed_reference is not None else "retiming_multiplier"
        )
    configs = OmegaConf.to_container(stages, resolve=True)
    bridges = [
        c for c in configs if c.get("_target_", "").endswith(".LatentBridgeStage")
    ]
    fields = [
        c
        for c in configs
        if c.get("_target_", "").endswith(".ConditionalVelocityStage")
    ]
    encoders = [
        c
        for c in configs
        if c.get("_target_", "").endswith(
            (".RoutedContentEncoderStage", ".ContentEncoderStage")
        )
    ]
    if len(bridges) != 1 or len(fields) != 1 or len(encoders) != 1:
        raise ValueError(
            "Speed adapter requires one codec stage, bridge and shared field"
        )
    if bridges[0].get("condition_key", "condition") != "condition":
        raise ValueError("Unexpected existing bridge condition")
    if fields[0].get("inference_condition_key", "condition") != "condition":
        raise ValueError("Unexpected existing inference condition")
    bridges[0]["condition_key"] = "speed_condition"
    fields[0]["inference_condition_key"] = "speed_condition"
    if flow_inference_method is not None:
        if flow_inference_method != "euler":
            raise ValueError("Matched flow uses Euler with fixed evaluation count")
        fields[0]["inference_method"] = flow_inference_method
    if dit_checkpoint_policy is not None:
        configure_dit_checkpoint_policy(encoders[0], fields[0], dit_checkpoint_policy)
    # Instantiate the unchanged stages first, preserving all old RNG draws.
    # Diagnostic preprocessing stops at the content encoder: condition must
    # already exist there, even though the encoder itself does not consume it.
    modules = [instantiate(c) for c in configs]
    modules.insert(
        configs.index(encoders[0]),
        SharedSpeedCondition(
            speed_reference,
            encoding,
            condition_dim,
            conditioning_input=conditioning_input,
        ),
    )
    return PipelineAlgo(modules, device=device, compatibility_mode=compatibility_mode)


def build_multiplier_conditioned_pipeline(
    stages,
    encoding="scalar",
    condition_dim=128,
    device=None,
    compatibility_mode="current",
    conditioning_input="retiming_multiplier",
    flow_inference_method=None,
    dit_checkpoint_policy=None,
):
    """PushT raw-rate adapter; physical velocity/reference inputs are unsupported."""
    if conditioning_input != "retiming_multiplier":
        raise ValueError("PushT requires raw retiming_multiplier conditioning")
    return build_speed_conditioned_pipeline(
        stages,
        encoding=encoding,
        condition_dim=condition_dim,
        device=device,
        compatibility_mode=compatibility_mode,
        conditioning_input="retiming_multiplier",
        flow_inference_method=flow_inference_method,
        dit_checkpoint_policy=dit_checkpoint_policy,
    )


def configure_dit_checkpoint_policy(encoder, field, policy):
    """Use native per-backbone checkpointing, without intercepting image execution."""
    if policy not in {"all", "dit_half"}:
        raise ValueError("unsupported native DiT checkpoint policy")
    owners = list(encoder.get("encoders", {}).values()) or [encoder.get("encoder", {})]
    owners.append(field.get("field", {}))
    backbones = [owner.get("backbone", {}) for owner in owners]
    # Validate every owner before changing any configuration.
    for backbone in backbones:
        if backbone.get("_target_") != "egomimic.models.unite_dit.UniteDiTBackbone":
            raise ValueError("native DiT backbone required for checkpoint policy")
        if policy == "dit_half" and (
            not backbone.get("gradient_checkpointing")
            or int(backbone.get("depth", 0)) <= 0
            or int(backbone["depth"]) % 2
        ):
            raise ValueError("dit_half requires checkpointing and positive even depth")
    for backbone in backbones:
        backbone["checkpoint_policy"] = policy


class SharedSpeedCondition(Stage):
    def __init__(
        self,
        speed_reference=None,
        encoding="scalar",
        condition_dim=128,
        hidden_dim=32,
        condition_key="condition",
        speed_key="requested_speed",
        output_key="speed_condition",
        initialization_seed=42,
        conditioning_input=None,
    ):
        super().__init__()
        if conditioning_input is None:
            # Legacy positional references retain native-speed semantics;
            # reference-free new callers consume the raw multiplier.
            conditioning_input = (
                "native_speed" if speed_reference is not None else "retiming_multiplier"
            )
        if encoding not in {"scalar", "fourier"}:
            raise ValueError("encoding must be scalar or fourier")
        if conditioning_input not in {"native_speed", "retiming_multiplier"}:
            raise ValueError(
                "conditioning_input must be native_speed or retiming_multiplier"
            )
        if conditioning_input == "native_speed" and (
            speed_reference is None
            or not math.isfinite(speed_reference)
            or speed_reference <= 0
        ):
            raise ValueError("speed_reference must be positive and train-derived")
        self.conditioning_input = conditioning_input
        if conditioning_input == "retiming_multiplier":
            if speed_reference is not None:
                raise ValueError(
                    "Multiplier conditioning requires speed_reference=null"
                )
            speed_key = "retiming_rate"
            # Make strict reload reject historical physical-speed weights even
            # though the scalar MLP has the same tensor shapes in both modes.
            self.register_buffer("retiming_multiplier_contract", torch.tensor(1))
        self.encoding = encoding
        self.condition_key, self.speed_key, self.output_key = (
            condition_key,
            speed_key,
            output_key,
        )
        self.reads, self.writes = (condition_key, speed_key), (output_key,)
        self.register_buffer(
            "speed_reference",
            torch.tensor(
                float(speed_reference) if speed_reference is not None else 1.0
            ),
        )
        # Added layers must not shift initialization of the unchanged model.
        with torch.random.fork_rng(devices=[]):
            # These layers initialize on CPU. torch.manual_seed also reseeds
            # CUDA generators, which fork_rng(devices=[]) does not restore.
            torch.set_rng_state(
                torch.Generator(device="cpu")
                .manual_seed(initialization_seed)
                .get_state()
            )
            self.mlp = nn.Sequential(
                nn.Linear(1 if encoding == "scalar" else 5, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, condition_dim),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, batch):
        condition = batch[self.condition_key]
        speed = batch[self.speed_key].to(device=condition.device, dtype=torch.float32)
        if speed.ndim != 2 or speed.shape != (condition.shape[0], 1):
            raise ValueError(f"{self.speed_key} must have shape [B,1]")
        if not torch.isfinite(speed).all() or (speed < 0).any():
            raise ValueError(f"{self.speed_key} must be finite and nonnegative")
        if self.conditioning_input == "retiming_multiplier":
            if (speed <= 0).any():
                raise ValueError("retiming_rate must be strictly positive")
            u = speed  # Direct dimensionless multiplier; never measured XY speed.
        else:
            u = torch.log1p(speed / self.speed_reference)
        features = (
            u
            if self.encoding == "scalar"
            else torch.cat((u, u.sin(), u.cos(), (2 * u).sin(), (2 * u).cos()), dim=-1)
        )
        # Preserve native-speed dtype behavior; the new multiplier path owns
        # its explicit input-to-parameter dtype boundary.
        if (
            self.conditioning_input == "retiming_multiplier"
            or self.mlp[0].weight.dtype == torch.float64
        ):
            features = features.to(dtype=self.mlp[0].weight.dtype)
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

    if OmegaConf.select(cfg, "model.pipeline._target_") not in {
        "egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline",
        "egomimic.pipeline.stages_speed.build_multiplier_conditioned_pipeline",
    }:
        return None, None
    kind = OmegaConf.select(
        cfg, "model.pipeline.conditioning_input", default="native_speed"
    )
    if kind == "retiming_multiplier":
        key, path = "retiming_rate", "deployment.requested_multiplier"
        if OmegaConf.select(cfg, "deployment.requested_speed") is not None:
            raise ValueError(
                "Multiplier rollout must not supply deployment.requested_speed"
            )
    elif kind == "native_speed":
        key, path = "requested_speed", "deployment.requested_speed"
        if OmegaConf.select(cfg, "deployment.requested_multiplier") is not None:
            raise ValueError(
                "Native-speed checkpoint cannot use a multiplier condition"
            )
    else:
        raise ValueError("Unsupported conditioning_input")
    value = OmegaConf.select(cfg, path)
    if value is None:
        raise ValueError(f"Conditioned rollout requires explicit {path}")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (key == "retiming_rate" and value == 0):
        raise ValueError(f"Invalid {path}")
    return key, value
