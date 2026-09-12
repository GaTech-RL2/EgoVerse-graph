import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from egomimic.robot.yam.adapter import YamCartesianAdapter
from egomimic.robot.yam.policy import GraphRobotPolicy, load_normalizer
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.pipeline.core import Stage


class ConstantStage(Stage):
    writes = ("pred_action",)

    def __init__(self):
        super().__init__()
        self.actions = torch.nn.Parameter(torch.zeros(1, 100, 14))

    def forward(self, batch):
        batch["pred_action"] = self.actions
        return batch


class Kinematics:
    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_euler("ZYX", q[3:]).as_matrix()
        return pose

    def ik(self, target, seed):
        return np.r_[
            target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_euler("ZYX")
        ]


def samples():
    return [
        dict(
            left=np.array([0.2, 0, 0, 0, 0, 0, 0.5]),
            right=np.array([-0.2, 0, 0, 0, 0, 0, 0.4]),
            images={
                k: np.full((8, 12, 3), [255, 0, 0], dtype=np.uint8)
                for k in ("top", "left_wrist", "right_wrist")
            },
        )
    ] * 2


def adapter(**kwargs):
    return YamCartesianAdapter(
        kinematics=Kinematics(),
        base_T_model={"left": np.eye(4), "right": np.eye(4)},
        camera_keys={"top": "image"},
        embodiment_id=7,
        image_hw=(4, 6),
        **kwargs,
    )


@pytest.mark.parametrize("rotation_mode,width", [("euler", 14), ("6D", 20)])
def test_observation_frames_images_and_wrist_action_anchors(rotation_mode, width):
    transform = np.eye(4)
    transform[:3, 3] = [0, 0.5, 0]
    value = adapter(rotation_mode=rotation_mode)
    value.base_T_model["left"] = transform
    obs = value.observation(samples())
    assert obs["image"].shape == (1, 3, 4, 6)
    torch.testing.assert_close(obs["image"][0, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(obs[value.proprio_key][0, :3], [0.2, -0.5, 0])
    chunk = np.zeros((1, 100, width))
    half = width // 2
    for offset in (0, half):
        chunk[0, :, offset] = 0.1
        chunk[0, :, offset + half - 1] = 0.5
        if width == 20:
            chunk[0, :, offset + 3 : offset + 9] = [1, 0, 0, 0, 1, 0]
    result = value.actions(chunk, samples())
    assert result.shape == (24, 14)
    np.testing.assert_allclose(result[:, 0], 0.3)
    np.testing.assert_allclose(result[:, 7], -0.1)
    # Repeated identical poses keep the observed wrist anchor; no cumulative drift.
    np.testing.assert_allclose(result[-1], result[0])


def test_absolute_actions_use_base_from_model_calibration():
    value = adapter(action_frame="model_frame")
    value.base_T_model["left"][:3, 3] = [0, 0.5, 0]
    chunk = np.zeros((1, 24, 14))
    chunk[..., [6, 13]] = 0.5
    actions = value.actions(chunk, samples())
    np.testing.assert_allclose(actions[:, 1], 0.5)
    np.testing.assert_allclose(actions[:, 0], 0)


def test_unreachable_and_malformed_actions_fail_before_returning_chunk():
    value = adapter()
    chunk = np.zeros((1, 24, 14))
    chunk[0, 23, 6] = 1.1
    with pytest.raises(ValueError, match="gripper"):
        value.actions(chunk, samples())
    with pytest.raises(ValueError, match="at least 24"):
        value.actions(chunk[:, :20], samples())
    chunk[:] = 0

    def fail(target, seed):
        raise ValueError("unreachable")

    value.kinematics.ik = fail
    with pytest.raises(ValueError, match="unreachable"):
        value.actions(chunk, samples())


def test_arc_decoder_is_applied_before_ik():
    from egomimic.robot.arc_decoder import BimanualArcDecoder

    value = adapter(decoder=BimanualArcDecoder(token_layout="e1_dur"))
    tokens = np.zeros((1, 100, 16))
    tokens[..., [6, 13]] = 0.5
    tokens[..., [14, 15]] = 1 / 30
    result = value.actions(tokens, samples())
    assert result.shape == (24, 14)
    np.testing.assert_allclose(result[:, [0, 7]], np.tile([0.2, -0.2], (24, 1)))


def normalizer():
    norm = MultiDataset(state={}, norm_mode="minmax")
    norm.embodiments = {7}
    norm.shapes = {
        7: {"actions_cartesian": (100, 14), "observations.state.ee_pose": (14,)}
    }
    norm.key_types = {
        7: {
            "actions_cartesian": "action_keys",
            "observations.state.ee_pose": "proprio_keys",
            "image": "camera_keys",
        }
    }
    norm.zarr_keys = {7: {key: key for key in norm.key_types[7]}}
    norm.norm_stats = {
        7: {
            key: {"min": np.zeros(shape), "max": np.ones(shape)}
            for key, shape in norm.shapes[7].items()
        }
    }
    return norm


def test_cache_preserves_schema_and_live_policy_normalizes_once(tmp_path):
    norm = normalizer()
    norm.cache_stats(str(tmp_path))
    loaded = load_normalizer(tmp_path / "norm_stats/norm_stats.json")
    assert loaded.key_types == norm.key_types

    class Graph:
        def process_batch_for_training(self, batch):
            return batch

        def forward_eval(self, batch):
            self.proprio = batch["robot"]["observations.state.ee_pose"]
            return {"robot": {"pred_action": torch.zeros(1, 100, 14)}}

    graph = Graph()
    value = adapter()
    value.actions = lambda native, samples: native
    policy = GraphRobotPolicy(graph, loaded, value)
    native = policy.predict(samples())
    np.testing.assert_allclose(graph.proprio[0, 0], -0.6, atol=1e-6)
    np.testing.assert_allclose(native, 0.5, atol=1e-6)
    bad = tmp_path / "old.json"
    bad.write_text(json.dumps({"stats": {}}))
    with pytest.raises(ValueError, match="normalizer_state"):
        load_normalizer(bad)


def test_deployment_loads_saved_graph_and_requires_complete_checkpoint(tmp_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from egomimic.robot.yam.policy import load_policy

    norm = normalizer()
    norm.cache_stats(str(tmp_path))
    graph_config = {
        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
        "stages": [{"_target_": f"{__name__}.ConstantStage"}],
    }
    training = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create({"model": {"pipeline": graph_config}}), training)
    graph = instantiate(graph_config, device="cpu")
    checkpoint = tmp_path / "model.ckpt"
    state = {"nets." + key: value for key, value in graph.nets.state_dict().items()}
    torch.save({"state_dict": state}, checkpoint)
    cfg = OmegaConf.create(
        {
            "normalizer_path": str(tmp_path / "norm_stats/norm_stats.json"),
            "training_config": str(training),
            "checkpoint": str(checkpoint),
            "device": "cpu",
            "adapter": {
                "_target_": "egomimic.robot.yam.adapter.YamCartesianAdapter",
                "kinematics": {"_target_": f"{__name__}.Kinematics"},
                "base_T_model": {
                    side: np.eye(4).tolist() for side in ("left", "right")
                },
                "camera_keys": {"top": "image"},
                "embodiment_id": 7,
            },
        }
    )
    policy = load_policy(cfg)
    actions = policy.predict(samples())
    assert actions.shape == (24, 14)
    np.testing.assert_allclose(actions[0, [0, 7]], [0.7, 0.3], atol=1e-6)
    torch.save({"state_dict": {"nets.wrong": torch.zeros(1)}}, checkpoint)
    with pytest.raises(ValueError, match="key mismatch"):
        load_policy(cfg)


def request():
    from egomimic.robot.yam import policy_inference_pb2 as pb

    observations = []
    for i, sample in enumerate(samples()):
        images = []
        for name, rgb in sample["images"].items():
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            assert ok
            images.append(
                pb.Image(
                    name=name, encoding="jpeg", width=12, height=8, data=jpeg.tobytes()
                )
            )
        observations.append(
            pb.Observation(
                sequence=i + 1,
                capture_monotonic_ns=(i + 1) * 33333333,
                left_joint_positions=sample["left"],
                right_joint_positions=sample["right"],
                images=images,
            )
        )
    return pb.PredictRequest(
        session_id="test-session",
        request_id=1,
        next_action_step=24,
        observations=observations,
    )


def test_rpc_round_trip_and_invalid_observations():
    grpc = pytest.importorskip("grpc")
    from egomimic.robot.yam import policy_inference_pb2 as pb
    from egomimic.robot.yam.server import create_server, decode_observations

    class Policy:
        def predict(self, samples):
            return np.zeros((24, 14))

    server, port = create_server(
        Policy(), model_id="test", task="from-yaml", address="127.0.0.1:0"
    )
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            call = channel.unary_unary(
                "/lerobot.inference.v1.PolicyInference/Predict",
                request_serializer=pb.PredictRequest.SerializeToString,
                response_deserializer=pb.ActionChunk.FromString,
            )
            result = call(request(), timeout=2)
            assert (result.session_id, result.request_id, result.start_action_step) == (
                "test-session",
                1,
                24,
            )
            assert (result.rows, result.cols, len(result.actions)) == (24, 14, 336)
            assert result.source_observation_sequence == 2
            invalid = request()
            invalid.observations[0].left_joint_positions[0] = float("nan")
            with pytest.raises(grpc.RpcError) as exc:
                call(invalid, timeout=2)
            assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            invalid = request()
            invalid.observations[1].sequence = 1
            with pytest.raises(ValueError, match="capture order"):
                decode_observations(invalid)
    finally:
        server.stop(0).wait()


def test_runtime_launcher_preserves_argv_and_rejects_wrong_revision(
    tmp_path, monkeypatch
):
    from egomimic.robot.yam import UPSTREAM_REVISION
    from egomimic.robot.yam import runtime

    (tmp_path / "rl2_yam/runtime").mkdir(parents=True)
    monkeypatch.setattr(
        runtime.subprocess,
        "check_output",
        lambda *args, **kwargs: UPSTREAM_REVISION + "\n",
    )
    directory, command = runtime.runtime_command(
        tmp_path, "teleop", ["--config", "a path.yaml"]
    )
    assert directory == tmp_path
    assert command == [
        "uv",
        "run",
        "--no-sync",
        "quest-teleop",
        "--config",
        "a path.yaml",
    ]
    monkeypatch.setattr(
        runtime.subprocess, "check_output", lambda *args, **kwargs: "different"
    )
    with pytest.raises(ValueError, match="separate checkout"):
        runtime.runtime_command(tmp_path, "teleop", [])


def test_mujoco_fk_ik_roundtrip_without_hardware(tmp_path):
    pytest.importorskip("mujoco")
    from egomimic.robot.yam.kinematics import MujocoArmKinematics

    nested = '<site name="tcp_site" pos=".1 0 0"/>'
    axes = ["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"]
    for i in reversed(range(6)):
        nested = (
            f'<body pos=".1 0 .05"><joint name="q{i}" axis="{axes[i]}" range="-2 2"/>'
            f'<geom type="sphere" size=".03" mass="1"/>{nested}</body>'
        )
    path = tmp_path / "arm.xml"
    path.write_text(
        '<mujoco><compiler angle="radian"/><worldbody>'
        + nested
        + "</worldbody></mujoco>"
    )
    kin = MujocoArmKinematics(path, [f"q{i}" for i in range(6)])
    seed = np.array([0, 0.2, -0.3, 0.1, 0.2, 0.1])
    target = kin.fk(seed + 0.01)
    solved = kin.ik(target, seed)
    actual = kin.fk(solved)
    assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= kin.position_tolerance
    assert (
        np.linalg.norm(
            Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).as_rotvec()
        )
        <= kin.rotation_tolerance
    )
    target[:3, 3] = [100, 100, 100]
    with pytest.raises(ValueError, match="unreachable"):
        kin.ik(target, seed)
