"""Chunked goal-conditioned RL in the shared graph runtime."""
from egomimic.pipeline.core import Stage
from egomimic.models.q_chunking import DecoupledQChunking


class QChunkingStage(Stage):
    reads_by_mode = {
        "train": ("observations", "high_value_goals", "high_value_action_chunks",
                  "high_value_next_observations", "high_value_rewards",
                  "high_value_masks", "high_value_backup_horizon"),
        "inference": ("observations", "goals"),
    }
    writes_by_mode = {"train": ("loss/*", "log/*"), "inference": ("pred_action", "action_lengths")}

    def __init__(self, codec, **network):
        super().__init__()
        self.codec = codec
        if codec.native_horizon > network["backup_horizon"]:
            raise ValueError("policy window cannot exceed the critic's native backup horizon")
        self.agent = DecoupledQChunking(policy_dim=codec.encoded_dim, **network)

    def execute(self, batch, *, mode):
        if mode == "train":
            actions, _ = self.codec.encode(batch["high_value_action_chunks"])
            metrics = {} if batch.get("log_predictions", True) else None
            losses = self.agent.losses(batch, actions, metrics=metrics)
            batch.update({"loss/" + name: value for name, value in losses.items()})
            if metrics is not None:
                batch.update({"log/DQC/" + name: value for name, value in metrics.items()})
        else:
            latent = self.agent.sample(batch["observations"], batch["goals"],
                                       generator=batch.get("generator"))
            batch["pred_action"], batch["action_lengths"] = self.codec.decode(latent)
        return batch
