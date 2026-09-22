"""Chunked goal-conditioned RL in the shared graph runtime."""
from egomimic.pipeline.core import Stage
from egomimic.models.q_chunking import DecoupledQChunking
from egomimic.rldb.goal_replay import goal_window_targets


class QChunkingStage(Stage):
    reads_by_mode = {
        "train": ("observations", "high_value_goals", "high_value_action_chunks",
                  "high_value_next_observations", "high_value_rewards",
                  "high_value_masks", "high_value_backup_horizon"),
        "inference": ("observations", "goals"),
    }
    writes_by_mode = {"train": ("loss/*", "log/*"), "inference": ("pred_action", "action_lengths")}

    def __init__(self, codec, backup_mode="fixed", **network):
        super().__init__()
        self.codec = codec
        if codec.native_horizon > network["backup_horizon"]:
            raise ValueError("policy window cannot exceed the critic's native backup horizon")
        if backup_mode not in {"fixed", "policy_window"}:
            raise ValueError("backup_mode must be fixed or policy_window")
        self.backup_mode = backup_mode
        if backup_mode == "policy_window":
            if network.get("use_chunk_critic", True):
                raise ValueError("policy_window requires direct TD without the long chunk critic")
            if network.get("kappa_d", 0.5) != 0.5:
                raise ValueError("policy_window requires symmetric direct TD (kappa_d=0.5)")
            self.reads_by_mode = {**self.reads_by_mode, "train": (
                "observations", "high_value_goals", "high_value_action_chunks",
                "high_value_observation_trajectory", "high_value_goal_offset")}
        self.agent = DecoupledQChunking(policy_dim=codec.encoded_dim, **network)

    def execute(self, batch, *, mode):
        if mode == "train":
            actions, lengths = self.codec.encode(batch["high_value_action_chunks"])
            metrics = {} if batch.get("log_predictions", True) else None
            targets = batch
            if self.backup_mode == "policy_window":
                targets = {**batch, **goal_window_targets(batch, lengths, self.agent.discount)}
            losses = self.agent.losses(targets, actions, metrics=metrics,
                                       policy_weights=getattr(self.codec, "actor_loss_weights", None))
            batch.update({"loss/" + name: value for name, value in losses.items()})
            if metrics is not None:
                batch.update({"log/DQC/" + name: value for name, value in metrics.items()})
                duration = lengths.float()
                batch.update({"log/Chunk/native_duration/mean": duration.mean(),
                              "log/Chunk/native_duration/min": duration.min(),
                              "log/Chunk/native_duration/max": duration.max()})
                for label, quantile in (("p10", .1), ("p50", .5), ("p90", .9)):
                    batch[f"log/Chunk/native_duration/{label}"] = duration.quantile(quantile)
                batch["log/TD/native_backup_horizon/mean"] = targets["high_value_backup_horizon"].mean()
                active = targets["high_value_masks"] > 0
                if active.any():
                    batch["log/TD/nonterminal_backup_horizon/mean"] = targets["high_value_backup_horizon"][active].mean()
                if self.backup_mode == "policy_window":
                    for name, values in {
                        "duration": lengths.float(),
                        "backup_horizon": targets["high_value_backup_horizon"],
                        "bootstrap_discount": self.agent.discount ** targets["high_value_backup_horizon"],
                    }.items():
                        batch.update({f"log/SMDP/{name}/mean": values.mean().detach(),
                                      f"log/SMDP/{name}/min": values.min().detach(),
                                      f"log/SMDP/{name}/max": values.max().detach()})
                    batch["log/SMDP/goal_terminal_fraction"] = (1 - targets["high_value_masks"]).mean()
        else:
            latent = self.agent.sample(batch["observations"], batch["goals"],
                                       generator=batch.get("generator"),
                                       action_projection=getattr(self.codec, "project_latent", None))
            batch["pred_action"], batch["action_lengths"] = self.codec.decode(latent)
        return batch
