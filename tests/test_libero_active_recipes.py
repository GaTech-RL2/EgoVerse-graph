"""Keep active LIBERO choices unambiguous without loading data or checkpoints."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "egomimic/hydra_configs"
TARGETS = {
    "libero10": 605121,
    "libero_spatial": 270054,
    "libero_object": 325065,
    "libero_goal": 280056,
}


def test_active_directory_contains_only_selected_eight_recipes():
    expected = {
        f"{family}_{suite}_{suffix}_s42.yaml"
        for suite in TARGETS
        for family, suffix in (
            ("action_flow", "oat_dp_matched"),
            ("dp", "oat_batch16_keep_steps"),
        )
    }
    assert {p.name for p in (CONFIG / "experiment/libero").glob("*.yaml")} == expected


@pytest.mark.parametrize("suite,steps", TARGETS.items())
def test_active_pair_preserves_selected_scientific_contract(suite, steps):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        af = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=libero/action_flow_{suite}_oat_dp_matched_s42"],
        )
        dp = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=libero/dp_{suite}_oat_batch16_keep_steps_s42"],
        )
    assert (af.model.num_latent_tokens, af.model.latent_dim) == (16, 16)
    assert af.model.pipeline.stages[-1].action_velocity_weight == 1
    assert (af.trainer.max_steps, af.trainer.val_check_interval) == (120000, 20000)
    assert af.trainer.check_val_every_n_epoch is None
    assert af.callbacks.ema.decay == 0.9978
    assert af.trainer.gradient_clip_val == 3
    assert af.benchmark.batch_size == 32
    assert (
        dp.benchmark.batch_size,
        dp.trainer.accumulate_grad_batches,
        dp.trainer.max_steps,
    ) == (16, 1, steps)
    assert dp.callbacks.batch_budget.global_batch_size == 16
    if suite == "libero_spatial":
        assert dp.callbacks.model_checkpoint.every_n_train_steps == 15000
        assert dp.callbacks.model_checkpoint.every_n_epochs is None
    assert af.model.pipeline.stages[0] == dp.model.pipeline.stages[0]
    for cfg in (af, dp):
        assert cfg.data.train_datasets.libero_panda.valid_ratio == 0.1
        assert cfg.data.valid_datasets.libero_panda.valid_ratio == 0.1


@pytest.mark.parametrize(
    "recipe", sorted((CONFIG / "experiment/libero_historical").glob("*.yaml"))
)
def test_historical_profiles_remain_composable(recipe):
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.3"):
        compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=libero_historical/{recipe.stem}"],
        )
