"""Typed multiplier route preserves native per-source execution and full batches."""

import runpy
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from egomimic.pipeline.stages_speed import configure_dit_checkpoint_policy


def test_multiplier_training_contract():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "egomimic/hydra_configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "hydra/launcher=basic",
                "+experiment=pusht/action_flow_cotrain_uc_multiplier_interpolation",
                "trainer.precision=bf16",
                "trainer.devices=1",
                "trainer.num_nodes=1",
                "model.pipeline.stages.6.inference_method=euler",
                "model.pipeline.stages.6.num_inference_steps=50",
            ],
        )
    validator = runpy.run_path(
        str(root / "scripts/train/validate_multiplier_config.py")
    )
    validator["validate_speed_contract"](cfg, "scalar")
    assert cfg.callbacks.ema.decay == 0.9978
    assert cfg.callbacks.ema.use_warmup is False
    assert "homogeneous_dithalf" not in cfg.callbacks
    assert cfg.model.pipeline.dit_checkpoint_policy == "dit_half"
    assert not (root / "egomimic/pl_utils/homogeneous_training/callback.py").exists()
    assert cfg.norm_stats.sample_frac == 1.0
    assert cfg.norm_stats.norm_mode == "minmax"
    # The typed validator above requires one full batch32 per source.
    assert len(cfg.data.train_datasets) == 2


def _owners():
    def owner():
        return {
            "backbone": {
                "_target_": "egomimic.models.unite_dit.UniteDiTBackbone",
                "gradient_checkpointing": True,
                "depth": 12,
            }
        }

    return {"encoders": {"u": owner(), "chain": owner()}}, {"field": owner()}


def test_native_dit_half_does_not_change_image_or_decoder_execution():
    encoder, field = _owners()
    configure_dit_checkpoint_policy(encoder, field, "dit_half")
    for owner in [*encoder["encoders"].values(), field["field"]]:
        assert owner["backbone"]["checkpoint_policy"] == "dit_half"
    assert set(encoder) == {"encoders"}
    assert set(field) == {"field"}


@pytest.mark.parametrize("invalid", ["disabled", "odd_depth", "wrong_owner", "unknown"])
def test_native_dit_half_rejects_mismatch_atomically(invalid):
    encoder, field = _owners()
    backbone = field["field"]["backbone"]
    policy = "dit_half"
    if invalid == "disabled":
        backbone["gradient_checkpointing"] = False
    elif invalid == "odd_depth":
        backbone["depth"] = 11
    elif invalid == "wrong_owner":
        backbone["_target_"] = "unreviewed.Backbone"
    else:
        policy = "guess"
    with pytest.raises(ValueError):
        configure_dit_checkpoint_policy(encoder, field, policy)
    assert all(
        "checkpoint_policy" not in owner["backbone"]
        for owner in [*encoder["encoders"].values(), field["field"]]
    )


def test_launcher_child_helper_exists_and_parses():
    root = Path(__file__).resolve().parents[1]
    helper = root / "scripts/train/action_flow_run_child.sh"
    assert helper.is_file()
    subprocess.run(["bash", "-n", str(helper)], check=True)
    assert "--cpu-bind=none" in helper.read_text()
