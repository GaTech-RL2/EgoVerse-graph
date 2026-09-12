"""Graph checkpoint, normalization and robot frame-boundary regressions."""

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.robot.arc_decoder import BimanualArcDecoder
from egomimic.robot.graph_policy import (
    CartesianGraphAdapter,
    GraphRobotPolicy,
    load_graph_policy,
    load_normalizer,
)
from egomimic.robot.interface import pose_matrix
from egomimic.robot.yam.kinematics import MujocoArmKinematics
from tests.test_robot_runtime import FakeRobot

PROPRIO, ACTION = "observations.state.ee_pose", "actions_cartesian"


def adapter(**kwargs):
    options = dict(
        base_T_model={arm: np.eye(4) for arm in ("left", "right")},
        camera_keys={"front_img_1": "front"},
        embodiment_id=7,
        rotation_mode="euler",
        action_frame="eef_frame",
        image_hw=(2, 3),
    )
    options.update(kwargs)
    return CartesianGraphAdapter(**options)


def normalizer(mode="zscore"):
    stats = {
        PROPRIO: {"mean": np.full(14, 2.0), "std": np.full(14, 2.0)},
        ACTION: {
            "mean": np.tile([0.1, 0, 0, 0, 0, 0, 0.5], 2),
            "std": np.full(14, 2.0),
        },
    }
    return MultiDataset(
        state={
            "norm_mode": mode,
            "embodiments": [7],
            "key_types": {7: {PROPRIO: "proprio_keys", ACTION: "action_keys"}},
            "shapes": {7: {PROPRIO: [14], ACTION: [2, 14]}},
            "zarr_keys": {7: {PROPRIO: PROPRIO, ACTION: ACTION}},
            "norm_stats": {7: stats},
        }
    )


class EchoStage(Stage):
    reads = (PROPRIO,)
    writes = ("pred_action",)

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(14))
        self.seen = None

    def forward(self, batch):
        self.seen = batch[PROPRIO].detach().clone()
        batch["pred_action"] = self.bias[None, None].expand(1, 2, 14)
        return batch


def test_graph_normalizes_proprio_and_unnormalizes_actions_once():
    stage = EchoStage()
    policy = GraphRobotPolicy(
        PipelineAlgo([stage], device="cpu"), normalizer(), adapter()
    )
    obs = FakeRobot().get_obs()
    result = policy.predict(obs)
    expected = (torch.tensor(obs["ee_poses"]) - 2) / (2 + 1e-6)
    torch.testing.assert_close(stage.seen[0].double(), expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(result[:, [0, 7]], 0.1, atol=1e-6)
    np.testing.assert_allclose(result[:, [6, 13]], 0.5, atol=1e-6)


def test_observation_color_and_camera_keys_follow_training_contract():
    result = adapter().observation(FakeRobot().get_obs())
    assert result["front"].shape == (1, 3, 2, 3)
    np.testing.assert_allclose(
        result["front"][0, :, 0, 0], np.array([30, 20, 10]) / 255.0
    )
    with pytest.raises(ValueError, match="observation"):
        adapter().observation({**FakeRobot().get_obs(), "front_img_1": None})


@pytest.mark.parametrize("frame", ["eef_frame", "model_frame"])
def test_action_frame_is_explicit_and_does_not_accumulate_within_chunk(frame):
    obs = FakeRobot().get_obs()
    obs["ee_poses"][:6] = [1, 2, 3, np.pi / 2, 0, 0]
    calibration = np.eye(4)
    calibration[0, 3] = 10
    boundary = adapter(
        action_frame=frame, base_T_model={"left": calibration, "right": np.eye(4)}
    )
    native = np.zeros((1, 2, 14))
    native[:, :, 0] = 0.2
    native[:, :, [6, 13]] = 0.5
    out = boundary.actions(native, obs)
    expected = [1, 2.2, 3] if frame == "eef_frame" else [10.2, 0, 0]
    np.testing.assert_allclose(out[0, :3], expected, atol=1e-6)
    np.testing.assert_allclose(out[0], out[1])


def test_6d_rotation_and_invalid_output():
    obs = FakeRobot().get_obs()
    native = np.zeros((1, 2, 20))
    for offset in (0, 10):
        native[:, :, offset + 3] = 1
        native[:, :, offset + 7] = 1
        native[:, :, offset + 9] = 0.5
    out = adapter(rotation_mode="6D").actions(native, obs)
    assert out.shape == (2, 14)
    np.testing.assert_allclose(pose_matrix(out[0, :6]), np.eye(4))
    native[0, 0, 3:6] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        adapter(rotation_mode="6D").actions(native, obs)


@pytest.mark.parametrize("layout", ["lab", "e1_dur", "e1_logdur", "e1_profile"])
def test_graph_adapter_decodes_all_arc_layouts_before_frame_conversion(layout):
    decoder = BimanualArcDecoder(
        token_layout=layout, resampled_vector_length=4, action_horizon=7
    )
    tokens = np.zeros((1, *decoder.shape))
    tokens[:, :, 6] = 0.5
    tokens[:, :, 13] = 0.5
    if layout == "lab":
        tokens[:, 0, :] = 0.01
    else:
        tokens[:, :, -2:] = 0.1 if layout != "e1_logdur" else np.log(0.1)
    poses = adapter(decoder=decoder).actions(tokens, FakeRobot().get_obs())
    assert poses.shape == (7, 14)
    assert np.isfinite(poses).all()


def test_normalizer_export_roundtrip_and_incomplete_cache_rejection(tmp_path):
    norm = normalizer()
    norm.cache_stats(str(tmp_path))
    path = tmp_path / "norm_stats/norm_stats.json"
    restored = load_normalizer(path)
    value = {PROPRIO: torch.ones(1, 14), ACTION: torch.ones(1, 2, 14)}
    for key, expected in norm.normalize(value, 7).items():
        torch.testing.assert_close(restored.normalize(value, 7)[key], expected)
    old = tmp_path / "stats-only.json"
    old.write_text(json.dumps({"stats": {}}))
    with pytest.raises(ValueError, match="normalizer_state"):
        load_normalizer(old)


def test_checkpoint_loading_is_strict_and_never_opens_training_datasets(tmp_path):
    normalizer().cache_stats(str(tmp_path))
    graph = PipelineAlgo([EchoStage()], device="cpu")
    ckpt = tmp_path / "model.ckpt"
    state = {f"nets.{key}": value for key, value in graph.nets.state_dict().items()}
    torch.save({"state_dict": state}, ckpt)
    training = tmp_path / "training.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {
                    "pipeline": {
                        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                        "stages": [
                            {"_target_": "tests.test_robot_graph_policy.EchoStage"}
                        ],
                    }
                },
                "data": {"_target_": "must.never.be.instantiated"},
            }
        ),
        training,
    )
    boundary = dict(
        _target_="egomimic.robot.graph_policy.CartesianGraphAdapter",
        base_T_model={a: np.eye(4).tolist() for a in ("left", "right")},
        camera_keys={"front_img_1": "front"},
        embodiment_id=7,
        rotation_mode="euler",
        action_frame="eef_frame",
        image_hw=[2, 3],
    )
    config = dict(
        normalizer_path=str(tmp_path / "norm_stats/norm_stats.json"),
        training_config=str(training),
        checkpoint=str(ckpt),
        device="cpu",
        adapter=boundary,
    )
    assert load_graph_policy(config).predict(FakeRobot().get_obs()).shape == (2, 14)
    torch.save({"state_dict": {**state, "nets.unexpected": torch.zeros(1)}}, ckpt)
    with pytest.raises(ValueError, match="key mismatch"):
        load_graph_policy(config)


def test_mujoco_fk_ik_roundtrip_on_actual_numeric_model(tmp_path):
    # A six-hinge chain exercises the real MuJoCo Jacobian, without hardware/models.
    body = '<site name="tcp_site" pos=".1 0 0"/>'
    for index in reversed(range(6)):
        axis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][index % 3]
        body = f'<body pos=".1 0 0"><joint name="joint{index + 1}" axis="{axis[0]} {axis[1]} {axis[2]}" range="-2 2"/><geom type="sphere" size=".01" mass=".1"/>{body}</body>'
    path = tmp_path / "arm.xml"
    path.write_text(
        f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody></mujoco>'
    )
    solver = MujocoArmKinematics(path, [f"joint{i + 1}" for i in range(6)])
    expected = solver.fk(np.array([0.1, 0.2, -0.1, 0.1, 0.1, -0.1]))
    actual = solver.fk(solver.ik(expected, np.zeros(6)))
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=0.002)
    assert Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude() < 0.02
