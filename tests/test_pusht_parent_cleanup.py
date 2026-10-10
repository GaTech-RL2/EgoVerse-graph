"""Keep every PushT recipe's science identical when deduplicating its parents."""

import hashlib
import json
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scripts.audit_hydra_configs import CONFIGS, compose_for_audit

BASELINE = json.loads(
    (
        Path(__file__).parent / "fixtures/pusht_config_before_parent_cleanup.json"
    ).read_text()
)


@pytest.mark.parametrize(
    "parent",
    [
        "episode_split",
        "action_flow_uc_h16",
        "action_flow_points6_uc_h16",
        "standard_dp_uc_h16",
    ],
)
def test_shared_parents_construct_their_data_boundary(parent):
    with compose_for_audit(CONFIGS / "data/pusht/parents" / f"{parent}.yaml") as cfg:
        data = instantiate(cfg.data, _recursive_=False)
        assert data.preflight_configuration()


@pytest.mark.parametrize("recipe,expected", sorted(BASELINE["recipes"].items()))
def test_existing_pusht_recipe_keeps_exact_configuration(recipe, expected):
    with compose_for_audit(CONFIGS / "experiment/pusht" / f"{recipe}.yaml") as cfg:
        actual = OmegaConf.to_container(cfg, resolve=False)
    actual.pop("hydra", None)  # A parent's selection path intentionally changed.
    encoded = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected


def test_retired_parent_paths_have_no_selectable_config_references():
    for group, retired in (
        ("data", BASELINE["retired_data"]),
        ("experiment", BASELINE["retired_recipes"]),
    ):
        for old in retired:
            assert not (CONFIGS / group / "pusht" / f"{old}.yaml").exists()
            for path in CONFIGS.rglob("*.yaml"):
                assert f"pusht/{old}" not in path.read_text()
    assert not (
        CONFIGS / "data/pusht/planar_usocket_chain_manual4919_standard_retimed.yaml"
    ).exists()


@pytest.mark.parametrize(
    "recipe",
    [
        "planar_uc_manual4919_dp_261m_af_obs_multiplier",
        "action_flow_cotrain_uc_multiplier_interpolation",
    ],
)
def test_selected_pair_still_has_2999_and_4919_with_two_full_batches(recipe):
    with compose_for_audit(CONFIGS / "experiment/pusht" / f"{recipe}.yaml") as cfg:
        for group in ("train_datasets", "valid_datasets"):
            datasets = cfg.data[group]
            assert (
                datasets.pushshapes_sim_u_socket.resolver.expected_episode_count == 2999
            )
            assert (
                datasets.pushshapes_sim_chain_gripper.resolver.expected_episode_count
                == 4919
            )
            assert all(
                d.valid_ratio == 0.01 and d.split_seed == 42 for d in datasets.values()
            )
        assert all(
            d.batch_size == 32 for d in cfg.data.train_dataloader_params.values()
        )
