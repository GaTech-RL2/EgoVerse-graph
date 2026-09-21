"""Record paired native-time TD evidence on real replay before an RL launch."""
import argparse
import hashlib

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from egomimic.rldb.goal_replay import ShardedGoalReplay, goal_window_targets
from egomimic.utils.experiment_artifacts import ArtifactWriter


def audit(cfg):
    replay = ShardedGoalReplay(cfg.env_name, cfg.shards, cfg.cache_dir, cfg.seed,
        cfg.audit_samples, cfg.steps, backup_horizon=cfg.backup_horizon,
        discount=cfg.discount, include_future_observations=True,
        data_registry=cfg.get("data_registry"))
    batch = replay.load_shard(0).sample(cfg.audit_samples, np.random.RandomState(cfg.audit_seed))
    cfg.action_dim = batch["high_value_action_chunks"].shape[-1]
    reports, paired = {}, []
    for kind in ["native_window", "arc"]:
        codec_cfg = OmegaConf.create(OmegaConf.to_container(cfg.codec, resolve=True))
        codec_cfg.kind = kind
        codec = hydra.utils.instantiate(codec_cfg)
        encoded, lengths = codec.encode(batch["high_value_action_chunks"])
        _, decoded_lengths = codec.decode(encoded)
        torch.testing.assert_close(lengths, decoded_lengths, rtol=0, atol=0)
        targets = goal_window_targets(batch, lengths, cfg.discount)
        paired.append(targets)
        offsets = batch["high_value_goal_offset"].numpy()
        duration = lengths.numpy()
        hit = (offsets >= 0) & (offsets < duration)
        horizon = np.where(hit, offsets, duration)
        expected_obs = batch["high_value_observation_trajectory"].numpy()[np.arange(len(duration)), horizon]
        np.testing.assert_array_equal(targets["high_value_next_observations"].numpy(), expected_obs)
        np.testing.assert_allclose(targets["high_value_rewards"].numpy(), cfg.discount ** horizon * hit, rtol=2e-6)
        np.testing.assert_array_equal(targets["high_value_masks"].numpy(), ~hit)
        np.testing.assert_array_equal(targets["high_value_backup_horizon"].numpy(), horizon)
        unique, counts = np.unique(duration, return_counts=True)
        reports[kind] = {
            "encoded_scalars": codec.encoded_dim,
            "native_duration_histogram": dict(zip(map(str, unique), map(int, counts))),
            "mean_native_duration": float(duration.mean()),
            "goal_terminal_fraction": float(hit.mean()),
            "target_hashes": {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                              for key, value in targets.items()},
        }
    for key in paired[0]:
        torch.testing.assert_close(paired[0][key], paired[1][key], rtol=0, atol=0)
    writer = ArtifactWriter(cfg.output_dir, **OmegaConf.to_container(cfg.artifacts))
    return writer.json("native-time-audit.json", {
        "status": "PASS", "samples": cfg.audit_samples, "seed": cfg.audit_seed,
        "same_replay_targets": True, "decoded_durations_match": True,
        "discount_per_native_step": cfg.discount,
        "goal_reward_interval": "half_open_state_based", "arms": reports,
        "data_receipts": replay.receipts,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    audit(OmegaConf.load(args.config))
