"""All three action representations retain the current task's pinned pool."""

import pytest
from omegaconf import OmegaConf

from scripts.audit_hydra_configs import CONFIGS, compose_for_audit


@pytest.mark.parametrize(
    "task,train,val",
    [("stationary", 417, 24), ("earbuds", 339, 15), ("towels", 462, 17)],
)
def test_modes_preserve_pinned_episode_membership(task, train, val):
    signatures = []
    for mode in ("time", "duration", "velocity"):
        with compose_for_audit(
            CONFIGS / f"experiment/arc_real/{task}_{mode}.yaml"
        ) as cfg:
            data = OmegaConf.to_container(cfg.data, resolve=True)
        signature = []
        for split, count in (("train_datasets", train), ("valid_datasets", val)):
            dataset = data[split]["yam_bimanual"]
            assert dataset["expected_train_episode_count"] == count
            digest = dataset["expected_train_episode_names_sha256"]
            assert len(digest) == 64
            signature.append((dataset["filters"], count, digest))
        signatures.append(signature)
    assert signatures[0] == signatures[1] == signatures[2]
