"""Native epsilon Transformer with the existing EgoVerse attention blocks."""

import torch

from egomimic.models.denoising_nets import CrossTransformer


class NativeDiffusionTransformer(CrossTransformer):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        hidden_dim=752,
        depth=10,
        num_heads=8,
        action_horizon=100,
        dropout=0.1,
        num_train_timesteps=100,
    ):
        if hidden_dim % num_heads or hidden_dim < input_dim:
            raise ValueError(
                "Transformer width must divide heads and cover native action dimensions"
            )
        if num_train_timesteps < 2:
            raise ValueError("Diffusion requires at least two training timesteps")
        super().__init__(
            nblocks=depth,
            cond_dim=global_cond_dim,
            hidden_dim=hidden_dim,
            act_dim=input_dim,
            act_seq=action_horizon,
            n_heads=num_heads,
            dropout=dropout,
            mlp_layers=1,
            mlp_ratio=4,
            time_conditioning="additive",
        )
        self.input_dim = int(input_dim)
        self.global_cond_dim = int(global_cond_dim)
        self.action_horizon = int(action_horizon)
        self.num_train_timesteps = int(num_train_timesteps)

    def forward(self, x, timesteps, cond, *args, **kwargs):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.action_horizon, self.input_dim):
            raise ValueError("Diffusion Transformer requires [B,H,native_action_dim]")
        if cond.shape != (len(x), self.global_cond_dim):
            raise ValueError("Diffusion Transformer requires [B,condition_dim]")
        t = torch.as_tensor(timesteps, device=x.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(len(x))
        if t.numel() != len(x):
            raise ValueError("Diffusion timestep must be scalar or one per sequence")
        # Continuous sinusoidal labels represent the discrete DDPM step range.
        t = t.float() / (self.num_train_timesteps - 1)
        return super().forward(x, t, cond.unsqueeze(1), *args, **kwargs)
