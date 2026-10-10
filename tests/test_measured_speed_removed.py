"""Retired measured-speed paths must never become selectable again."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egomimic.pipeline.stages_speed import (
    SharedSpeedCondition,
    build_speed_conditioned_pipeline,
    requested_rollout_condition,
)
from egomimic.rldb.zarr.physical_retiming import PhysicalWindowRetiming

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("mode", ["native_speed", "physical_speed", "unknown"])
def test_conditioner_and_factory_reject_retired_modes(mode):
    with pytest.raises(ValueError, match="removed"):
        SharedSpeedCondition(conditioning_input=mode)
    with pytest.raises(ValueError, match="removed"):
        build_speed_conditioned_pipeline([], conditioning_input=mode)


def test_default_is_direct_multiplier_and_reference_cannot_select_old_mode():
    model = SharedSpeedCondition()
    assert model.speed_key == "retiming_rate"
    assert model.conditioning_input == "retiming_multiplier"
    assert "speed_reference" not in model.state_dict()
    with pytest.raises(ValueError, match="removed"):
        SharedSpeedCondition(1.0)
    with pytest.raises(ValueError, match="retiming_rate"):
        SharedSpeedCondition(speed_key="requested_speed")


def test_retimer_rejects_retired_mode_before_loading_data():
    with pytest.raises(ValueError, match="removed"):
        PhysicalWindowRetiming(
            [1.0],
            {"left": "pose_wxyz", "right": "pose_wxyz"},
            ["left", "right"],
            30,
            timestamp_key="clock",
            conditioning_input="native_speed",
        )


@pytest.mark.parametrize("value", [None, "native_speed"])
def test_rollout_never_falls_back_to_measured_speed(value):
    pipeline = {
        "_target_": "egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline"
    }
    if value is not None:
        pipeline["conditioning_input"] = value
    cfg = OmegaConf.create(
        {"model": {"pipeline": pipeline}, "deployment": {"requested_speed": 1.0}}
    )
    with pytest.raises(ValueError):
        requested_rollout_condition(cfg)


def test_configs_and_launchers_have_no_retired_routes():
    retired = (
        "human_speed_v1",
        "action_flow_cotrain_uc_speed_interpolation",
        "yam_human_keypoints_speed_h816_private512_s42",
        "AF_STATIONARY_SPEED_REFERENCE",
        "AF_SPEED_REFERENCE",
    )
    for root in (ROOT / "egomimic/hydra_configs", ROOT / "scripts/train"):
        for file in root.rglob("*"):
            if file.suffix not in {".yaml", ".sh", ".sbatch", ".py"}:
                continue
            body = file.read_text()
            assert not any(name in body for name in retired), file
