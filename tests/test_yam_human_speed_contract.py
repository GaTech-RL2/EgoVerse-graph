"""Physical-time retiming and actual native leaf/condition boundaries."""

import numpy as np
import pytest
import torch

from egomimic.pipeline.stages_speed import SharedSpeedCondition
from egomimic.rldb.zarr.episode_split import complete_window_count
from egomimic.rldb.zarr.physical_retiming import PhysicalWindowRetiming
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

RATES = [0.2, 0.4, 0.6, 0.8, 1.0]


def fixture(human=True):
    n = 30 if human else 100
    dt = np.resize(np.array([0.02, 0.04, 0.03, 0.05]), n - 1)
    clock = np.concatenate(([0.0], np.cumsum(dt))) if human else np.arange(n) / 30
    pose = np.zeros((n, 7))
    pose[:, 0] = 0.2 * clock
    pose[:, 3] = 1
    keys = ("left.pose", "right.pose")
    obj = PhysicalWindowRetiming(
        RATES if human else [1.0],
        dict.fromkeys(keys, "pose_wxyz"),
        keys,
        n,
        3 if human else 1,
        "human" if human else "robot",
        5,
        "clock" if human else None,
    )
    keymap = {k: {"zarr_key": k, "horizon": n} for k in keys}
    if human:
        keymap["clock"] = {"zarr_key": "clock", "horizon": n}
    obj.bind_episode({"fps": 30}, keymap)
    raw = {k: pose.copy() for k in keys}
    if human:
        raw["clock"] = (clock * 1e9).round().astype(np.int64)
    return obj, raw, keymap


@pytest.mark.parametrize("view", range(5))
def test_all_selected_rates_follow_irregular_physical_time(view):
    obj, raw, _ = fixture()
    old = raw["left.pose"].copy()
    raw["_retiming_view"] = view
    got = obj.transform(raw)
    assert got["retiming_rate"] == np.float32(RATES[view])
    np.testing.assert_allclose(got["retiming_rate"], [RATES[view]])
    assert "requested_speed" not in got
    if view == 4:
        assert np.array_equal(got["left.pose"], old)
    np.testing.assert_allclose(np.linalg.norm(got["left.pose"][:, 3:], axis=1), 1)


@pytest.mark.parametrize("view", range(5))
def test_robot_five_identity_views_preserve_values(view):
    obj, raw, _ = fixture(False)
    old = raw["left.pose"].copy()
    raw["_retiming_view"] = view
    got = obj.transform(raw)
    assert np.array_equal(got["left.pose"], old)
    np.testing.assert_allclose(got["retiming_rate"], [1.0])
    assert "requested_speed" not in got


@pytest.mark.parametrize(
    "bad", ["short", "duplicate_clock", "float_clock", "zero_quaternion"]
)
def test_invalid_physical_futures_rejected(bad):
    obj, raw, _ = fixture()
    raw["_retiming_view"] = 0
    if bad == "short":
        raw["left.pose"] = raw["left.pose"][:-1]
    elif bad == "duplicate_clock":
        raw["clock"][1] = raw["clock"][0]
    elif bad == "float_clock":
        raw["clock"] = raw["clock"].astype(float)
    else:
        raw["left.pose"][:, 3:] = 0
    with pytest.raises(ValueError):
        obj.transform(raw)


def test_real_leaf_maps_views_anchors_and_never_reads_padded_tail(
    monkeypatch, tmp_path
):
    obj, _, keymap = fixture()
    total = 35
    clock = np.arange(total, dtype=np.int64) * 33333333
    pose = np.zeros((total, 7))
    pose[:, 0] = clock * 1e-9 * 0.2
    pose[:, 3] = 1
    arrays = {"clock": clock, "left.pose": pose, "right.pose": pose}

    class Reader:
        metadata = {
            "total_frames": total,
            "embodiment": "human_bimanual",
            "fps": 30,
            "features": {},
        }
        intrinsics = None

        def read(self, requests):
            return {
                key: arrays[key][a:z].copy() if z is not None else arrays[key][a].copy()
                for key, (a, z) in requests.items()
            }

    def init(self):
        self.episode_reader = Reader()
        self.metadata = self.episode_reader.metadata
        self.total_frames = total
        self.embodiment = "human_bimanual"
        self.keys_dict = {}
        self._image_keys = set()
        self._json_keys = set()

    monkeypatch.setattr(ZarrDataset, "init_episode", init)
    leaf = ZarrDataset(tmp_path / "episode.zarr", keymap, [obj])
    assert len(leaf) == complete_window_count(total, 30, 5) == 30
    values = [leaf[j] for j in range(5)]
    assert [int(x["frame_index"]) for x in values] == [0] * 5
    assert [int(x["retiming_view"]) for x in values] == list(range(5))
    assert int(leaf[len(leaf) - 1]["frame_index"]) == 5
    with pytest.raises(IndexError):
        leaf[len(leaf)]


def test_scalar_condition_rng_identity_gradient_and_explicit_inference_input():
    torch.manual_seed(5)
    before = torch.get_rng_state().clone()
    stage = SharedSpeedCondition(None, condition_dim=8)
    assert torch.equal(before, torch.get_rng_state())
    condition = torch.randn(3, 8, requires_grad=True)
    speed = torch.tensor([[0.01], [0.05], [0.1]])
    got = stage({"condition": condition, "retiming_rate": speed})["speed_condition"]
    torch.testing.assert_close(got, condition, rtol=0, atol=0)
    got.square().sum().backward()
    assert stage.mlp[-1].weight.grad.norm() > 0 and condition.grad.norm() > 0
    with pytest.raises(KeyError):
        stage({"condition": condition})
    with pytest.raises(ValueError):
        stage({"condition": condition, "retiming_rate": torch.full((3, 1), -1.0)})


@pytest.mark.parametrize("domain,dim", [(7, 14), (3, 138)])
def test_speed_reaches_actual_action_flow_objective(domain, dim):
    from egomimic.pipeline.stages_action_flow import (
        ActionFlowObjectiveStage,
        ConditionalVelocityStage,
        ContentDecoderStage,
        ContentEncoderStage,
        LatentBridgeStage,
    )

    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 8).double()

        def forward(self, state, time, condition, *, condition_drop_mask):
            visible = condition.masked_fill(condition_drop_mask[:, None], 0)
            return state.tanh() + self.linear(visible)[:, None] + time[:, None, None]

    torch.manual_seed(19)
    aliases = {"7": "robot", "3": "human"}
    encoders = {
        name: torch.nn.Sequential(torch.nn.Linear(d, 8), torch.nn.Tanh()).double()
        for name, d in (("robot", 14), ("human", 138))
    }
    decoders = {
        name: torch.nn.Linear(8, d).double()
        for name, d in (("robot", 14), ("human", 138))
    }
    speed = SharedSpeedCondition(condition_dim=3, speed_reference=None).double()
    field = Field()
    batch = {
        "target": torch.randn(2, 4, dim, dtype=torch.float64),
        "sampler/noise": torch.randn(2, 4, 8, dtype=torch.float64),
        "condition": torch.randn(2, 3, dtype=torch.float64),
        "retiming_rate": torch.tensor([[0.1], [0.3]], dtype=torch.float64),
        "embodiment": torch.full((2,), domain),
    }
    stages = [
        speed,
        ContentEncoderStage(encoders=encoders, selector_aliases=aliases),
        LatentBridgeStage(
            samples_per_content=3,
            condition_dropout_probability=0.0,
            condition_key="speed_condition",
        ),
        ConditionalVelocityStage(
            field,
            flow_clean_gradient_mode="all_stopgrad",
            inference_condition_key="speed_condition",
        ),
        ContentDecoderStage(decoders=decoders, selector_aliases=aliases),
        ActionFlowObjectiveStage(residual_key="action_flow/fm_velocity_residual"),
    ]
    for stage in stages:
        batch = stage(batch)
    loss = batch["loss/action_flow"]
    assert torch.isfinite(loss)
    loss.backward()
    assert speed.mlp[-1].weight.grad.norm() > 0
    assert field.linear.weight.grad.norm() > 0
    assert encoders[aliases[str(domain)]][0].weight.grad.norm() > 0
    assert decoders[aliases[str(domain)]].weight.grad.norm() > 0
