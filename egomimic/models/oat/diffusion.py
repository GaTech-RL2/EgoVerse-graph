"""Released OAT diffusion backbone adapted to the graph's flat condition port."""

from egomimic.models.oat.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)


class GraphDiffusionTransformer(TransformerForDiffusion):
    """Restore the two observation tokens without changing the released network."""

    def __init__(self, *, n_obs_steps, cond_dim, **kwargs):
        super().__init__(n_obs_steps=n_obs_steps, cond_dim=cond_dim, **kwargs)
        self.observation_steps = int(n_obs_steps)
        self.observation_dim = int(cond_dim)
        self.global_cond_dim = self.observation_steps * self.observation_dim

    def forward(self, sample, timestep, global_cond):
        expected = (sample.shape[0], self.global_cond_dim)
        if tuple(global_cond.shape) != expected:
            raise ValueError(f"Expected observation condition {expected}")
        condition = global_cond.reshape(
            sample.shape[0], self.observation_steps, self.observation_dim
        )
        return super().forward(sample, timestep, cond=condition)
