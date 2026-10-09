"""Matched observation/data DP candidate reusing native Paper U-Net and DP stages."""

from copy import copy

import torch
from torch import nn

from egomimic.models.ddim_scheduler import DDIMScheduler
from egomimic.models.denoising_nets import (
    ConditionalResidualBlock1D,
    PaperConditionalUnet1D,
)
from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Pipeline, Stage, resolve_homogeneous_scalar
from egomimic.pipeline.stages_diffusion import (
    DiffusionDenoiserStage,
    DiffusionEpsilonLossStage,
    DiffusionNoisingStage,
)


def shared_native_unets(
    condition_dim=256,
    down_dims=(512, 1024, 2048),
    kernel_size=5,
    step_embed_dim=128,
    n_groups=8,
):
    """Private native first/last layers; all intermediate U-Net parameters shared.

    No second full U-Net allocation, action padding, latent codec or copied sampler.
    The two branches remain the maintained PaperConditionalUnet1D forward path.
    """
    robot = PaperConditionalUnet1D(
        input_dim=14,
        global_cond_dim=condition_dim,
        diffusion_step_embed_dim=step_embed_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=True,
    )
    human = copy(robot)
    human._modules = robot._modules.copy()
    human._parameters = robot._parameters.copy()
    human._buffers = robot._buffers.copy()
    human.input_dim = 138
    first = robot.down_modules[0]
    human.down_modules = nn.ModuleList(
        [
            nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        138,
                        down_dims[0],
                        cond_dim=step_embed_dim + condition_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=True,
                    ),
                    first[1],
                    first[2],
                ]
            ),
            *list(robot.down_modules)[1:],
        ]
    )
    human.final_conv = nn.Sequential(
        robot.final_conv[0], nn.Conv1d(down_dims[0], 138, 1)
    )
    return {"yam_bimanual": robot, "human_bimanual": human}


class SharedNativeDPStage(Stage):
    reads = ("target", "condition", "embodiment")
    writes = ("loss/diffusion_noise", "log/*")
    reads_by_mode = {"inference": ("condition", "embodiment")}
    writes_by_mode = {"inference": ("pred_action", "log/*")}
    objective = "epsilon"

    def __init__(
        self,
        condition_dim=256,
        action_horizon=100,
        inference_steps=50,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        step_embed_dim=128,
        n_groups=8,
        condition_dropout_probability=0.0,
    ):
        super().__init__()
        if action_horizon != 100:
            raise ValueError("Matched comparison requires100 action steps")
        self.condition_dropout_probability = float(condition_dropout_probability)
        if not 0 <= self.condition_dropout_probability <= 1:
            raise ValueError("Invalid condition dropout")
        self.null_condition = nn.Parameter(torch.randn(condition_dim) * 0.02)
        self.aliases = {"3": "human_bimanual", "7": "yam_bimanual"}
        models = shared_native_unets(
            condition_dim, down_dims, kernel_size, step_embed_dim, n_groups
        )
        graphs = {}
        for domain, width in (("yam_bimanual", 14), ("human_bimanual", 138)):
            scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
            )
            policy = DiffusionPolicy(
                models[domain], scheduler, action_horizon, inference_steps
            )
            graphs[domain] = Pipeline(
                [
                    DiffusionNoisingStage(scheduler, action_horizon, width),
                    DiffusionDenoiserStage(
                        policy, action_horizon, width, condition_dim
                    ),
                    DiffusionEpsilonLossStage(),
                ]
            )
        self.branches = nn.ModuleDict(graphs)

    def execute(self, batch, *, mode):
        raw = resolve_homogeneous_scalar(batch["embodiment"], label="DP embodiment")
        domain = self.aliases.get(str(raw), str(raw))
        if domain not in self.branches:
            raise ValueError("Unsupported DP embodiment")
        if mode == "train" and self.condition_dropout_probability > 0:
            batch = dict(batch)
            condition = batch["condition"]
            mask = (
                torch.rand(len(condition), device=condition.device)
                < self.condition_dropout_probability
            )
            batch["condition"] = torch.where(
                mask[:, None], self.null_condition.to(condition)[None], condition
            )
        return self.branches[domain].execute(batch, mode=mode)

    def forward(self, batch):
        return self.execute(batch, mode="train")


def build_matched_dp_pipeline(
    stages,
    speed_reference=None,
    encoding="scalar",
    condition_dim=256,
    device=None,
    compatibility_mode="current",
    conditioning_input="none",
    dp_inference_steps=100,
    dp_down_dims=(512, 1024, 2048),
    flow_inference_method=None,
    dp_condition_dropout_probability=0.0,
):
    from hydra.utils import instantiate

    from egomimic.pipeline.algo import PipelineAlgo

    if (
        speed_reference is not None
        or encoding != "scalar"
        or conditioning_input != "none"
    ):
        raise ValueError("Unaugmented DP requires observation-only conditioning")
    if compatibility_mode != "current":
        raise ValueError("Unsupported compatibility mode")
    prefix = list(stages)[:3]
    if [x["_target_"].split(".")[-1] for x in prefix] != [
        "HPTStemStage",
        "HPTTrunkStage",
        "EmbodimentActionTargetBuilder",
    ]:
        raise ValueError("Observation and target prefix must match Action Flow")
    modules = [instantiate(x) for x in prefix]
    modules.append(
        SharedNativeDPStage(
            condition_dim=condition_dim,
            inference_steps=dp_inference_steps,
            down_dims=dp_down_dims,
            condition_dropout_probability=dp_condition_dropout_probability,
        )
    )
    return PipelineAlgo(modules, device=device)


def matched_adamw(
    params,
    lr,
    betas,
    eps,
    weight_decay,
    adamw_weight_decay=None,
    muon_weight_decay=None,
    muon_momentum=None,
    muon_adjust_lr_fn=None,
):
    # Hydra merges inherited mappings; reject active stale composite options.
    if any(
        v is not None
        for v in (
            adamw_weight_decay,
            muon_weight_decay,
            muon_momentum,
            muon_adjust_lr_fn,
        )
    ):
        raise ValueError("Inherited composite optimizer options must be disabled")
    return torch.optim.AdamW(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )


def native_dp_scheduler(
    optimizer,
    max_steps,
    warmup_steps,
    warmup_start_factor,
    eta_min,
    **inherited_af_options,
):
    """Native DP cosine schedule; reject active inherited AF schedule options."""
    allowed = {
        "decay_start_1_steps",
        "decay_end_1_steps",
        "decay_start_2_steps",
        "decay_end_2_steps",
        "base_lr_1",
        "base_lr_2",
        "final_lr",
    }
    if set(inherited_af_options) - allowed or any(
        v is not None for v in inherited_af_options.values()
    ):
        raise ValueError("AF schedule options must be disabled for native DP")
    from egomimic.utils.schedulers import warmup_cosine_scheduler

    return warmup_cosine_scheduler(
        optimizer, max_steps, warmup_steps, warmup_start_factor, eta_min
    )
