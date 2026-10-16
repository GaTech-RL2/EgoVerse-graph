import pytest
import torch

from egomimic.pipeline.stages_oat import OATObservationStage


class Recorder(torch.nn.Module):
    def forward(self, obs):
        self.obs = obs
        return torch.cat([v.flatten(2) for v in obs.values()], -1)


def test_historical_default_unchanged():
    encoder = Recorder()
    stage = OATObservationStage(encoder, ["state"])
    x = torch.arange(12.0).reshape(2, 2, 3)
    batch = {"state": x}
    stage(batch)
    assert encoder.obs["state"] is x
    assert torch.equal(batch["condition"], x.flatten(1))


def test_native_yam_alias_layout_range_and_three_views():
    encoder = Recorder()
    keys = ["front", "left", "right"]
    bindings = {k: k + "_rgb" for k in keys}
    bindings["pose"] = "ee_pose"
    stage = OATObservationStage(
        encoder,
        keys + ["pose"],
        key_bindings=bindings,
        chw_image_keys=keys,
        image_scale=2,
        image_offset=-1,
        native_data_context=True,
    )
    batch = {k: torch.linspace(0, 1, 48).reshape(1, 2, 3, 2, 4) for k in keys}
    batch["pose"] = torch.randn(1, 2, 14)
    original = {k: v.clone() for k, v in batch.items()}
    stage(batch)
    for k in keys:
        assert encoder.obs[k + "_rgb"].shape == (1, 2, 2, 4, 3)
        assert torch.equal(
            encoder.obs[k + "_rgb"], original[k].permute(0, 1, 3, 4, 2) * 2 - 1
        )
        assert torch.equal(batch[k], original[k])
    assert encoder.obs["ee_pose"] is batch["pose"]
    assert batch["condition"].shape == (1, 172)


def test_bad_layout_fails_closed():
    stage = OATObservationStage(Recorder(), ["front"], chw_image_keys=["front"])
    with pytest.raises(ValueError):
        stage({"front": torch.zeros(1, 2, 4, 4, 3)})


def test_native_jpeg_float64_becomes_float32_without_mutating_input():
    encoder = Recorder()
    stage = OATObservationStage(
        encoder, ["front"], chw_image_keys=["front"], image_scale=2, image_offset=-1
    )
    image = torch.ones(1, 2, 3, 2, 4, dtype=torch.float64)
    stage({"front": image})
    assert encoder.obs["front"].dtype == torch.float32
    assert image.dtype == torch.float64


def test_yam_energy_score_declared_horizon32_and_old100_guard():
    from egomimic.eval.yam_action_flow_eval import YamCartesianActionFlowEval

    class Norm:
        def unnormalize(self, data, embodiment_id):
            return data

    value = YamCartesianActionFlowEval(
        yam_action_horizon=32, energy_score_enabled=False
    )
    value.normalizer = Norm()
    samples = torch.zeros(32, 2, 32, 14)
    target = torch.zeros(2, 32, 14)
    answer = value._energy_values(samples, target, 7)
    assert all(torch.isfinite(x).all() and not x.any() for x in answer.values())
    assert value.energy_score_distance_metadata["complete_normalized_chunk_shape"] == [
        32,
        14,
    ]
    old = YamCartesianActionFlowEval(energy_score_enabled=False)
    old.normalizer = Norm()
    with pytest.raises(ValueError, match="100,14"):
        old._energy_values(samples, target, 7)


def test_yam_openloop_horizon32_preserves_native_backend():
    from egomimic.eval.yam_human_open_loop_eval import YamHumanOpenLoopEval

    value = YamHumanOpenLoopEval(
        robot_prediction_horizon=32, limit_val_batches=1.0, energy_score_enabled=False
    )
    assert value.robot_prediction_horizon == 32 and value.yam_action_horizon == 32
    assert value.energy_score_distance_metadata["yam_bimanual"][
        "complete_normalized_chunk_shape"
    ] == [32, 14]


def test_single_source_checkpoint_callback_yam_and_legacy_identity():
    from types import SimpleNamespace

    from egomimic.utils.libero_dit_half import LiberoDiTHalf

    old = LiberoDiTHalf()
    assert old.state_dict() == {
        "policy": "dit-half",
        "homogeneous": "not_applicable_single_source",
    }
    yam = LiberoDiTHalf(source_key="yam_bimanual")
    yam.scope = SimpleNamespace(counts={"encoder/direct": 0})
    yam.on_train_batch_start(None, None, {"yam_bimanual": {}}, 0)
    assert yam.before == {"encoder/direct": 0}
    yam.load_state_dict(yam.state_dict())
    with pytest.raises(ValueError, match="single source"):
        yam.on_train_batch_start(None, None, {"libero_panda": {}}, 0)
    with pytest.raises(ValueError, match="identity mismatch"):
        yam.load_state_dict(old.state_dict())
