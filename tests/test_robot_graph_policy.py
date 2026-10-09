"""Graph checkpoint, normalization and robot frame-boundary regressions."""

import json
from types import SimpleNamespace

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
    configure_adapter_for_training,
    configure_flow_inference_steps,
    configure_profile_controls,
    load_graph_policy,
    load_normalizer,
    validate_graph_device,
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


class RetryStage(Stage):
    reads = (PROPRIO,)
    writes = ("pred_action",)

    def __init__(self, valid_after):
        super().__init__()
        self.valid_after, self.calls = valid_after, 0

    def forward(self, batch):
        self.calls += 1
        prediction = torch.zeros(1, 2, 14)
        if self.calls < self.valid_after:
            # The test normalizer maps 0.5 to a raw opening of 1.5.
            prediction[..., [6, 13]] = 0.5
        batch["pred_action"] = prediction
        return batch


class ConditionStage(Stage):
    reads = (PROPRIO,)
    writes = ("condition",)

    def forward(self, batch):
        batch["condition"] = batch[PROPRIO][..., :8]
        return batch


class ProfileSamplerStage(Stage):
    reads = (PROPRIO,)
    writes = ("pred_action",)

    def __init__(self):
        super().__init__()
        self.policy = SimpleNamespace(num_inference_steps=100)

    def forward(self, batch):
        batch["pred_action"] = torch.zeros(1, 100, 14)
        return batch


def declare_test_model(training, horizon=2, width=14, control_path=None):
    from tests.test_inference_config import training_config

    declaration = training_config(horizon=horizon, action_dim=width).model.inference
    declaration.input.keys = [PROPRIO, "front", "embodiment"]
    declaration.input.history_keys = [PROPRIO, "front"]
    declaration.profiles.default.overrides.replan_every.default = 1
    if control_path:
        declaration.profiles.default.overrides.inference_steps.target.attribute_path = (
            control_path
        )
    else:
        del declaration.profiles.default.overrides["inference_steps"]
    training.model.pipeline.stage_ids = {"sampler": 0}
    training.model.inference = declaration
    training.data_context_loader = {
        "_target_": "egomimic.rldb.zarr.data_module.load_data_context"
    }
    declaration.compatibility.normalizer_schema = {
        "action_key": ACTION,
        "native_shape": [horizon, width],
        "embodiment": 7,
    }
    training.model.inference = declaration
    return training


def save_bound_checkpoint(path, training, state):
    from egomimic.pipeline.checkpoint_binding import checkpoint_binding
    from egomimic.pipeline.inference_config import write_inference_config
    from egomimic.pl_utils.data_context import DataContext, serializable_state
    from egomimic.rldb.zarr.data_module import _digest

    norm = normalizer()
    snapshot = {
        "kind": "zarr-normalizer-v1",
        "normalizer_state": norm.to_state(),
        "sha256": _digest(norm.to_state()),
    }
    context = DataContext(norm, norm.shapes, (), snapshot)
    checkpoint = {
        "state_dict": state,
        "data_context": snapshot,
        "inference_binding": checkpoint_binding(training, context),
    }
    torch.save(checkpoint, path)
    (path.parent / "data-context.json").write_text(
        json.dumps(serializable_state(snapshot))
    )
    write_inference_config(
        training, path.parent / "inference-config.yaml", data_context=context
    )
    return checkpoint


def test_flow_rollout_can_override_only_its_euler_solver_budget():
    from egomimic.pipeline.stages_flow import FlowDenoiserStage

    flow = FlowDenoiserStage(
        torch.nn.Linear(1, 1),
        action_horizon=2,
        action_dim=14,
        condition_input_dim=8,
        num_inference_steps=50,
    )
    graph = PipelineAlgo([flow], device="cpu", stage_ids={"sampler": 0})

    configure_flow_inference_steps(graph, 10)

    assert flow.num_inference_steps == 10
    with pytest.raises(ValueError, match="positive integer"):
        configure_flow_inference_steps(graph, 0)
    with pytest.raises((KeyError, ValueError), match="sampler"):
        configure_flow_inference_steps(PipelineAlgo([EchoStage()], device="cpu"), 10)


def test_checkpoint_load_applies_flow_euler_override(tmp_path):
    from egomimic.pipeline.stages_flow import FlowDenoiserStage

    normalizer().cache_stats(str(tmp_path))
    # The strict loader rejects an empty checkpoint, so use a stateful model
    # even though this regression needs only construction, never a forward.
    flow = FlowDenoiserStage(
        torch.nn.Linear(1, 1),
        action_horizon=2,
        action_dim=14,
        condition_input_dim=8,
        num_inference_steps=50,
    )
    graph = PipelineAlgo(
        [ConditionStage(), flow], device="cpu", stage_ids={"sampler": 1}
    )
    ckpt = tmp_path / "flow.ckpt"
    state = {f"nets.{key}": value for key, value in graph.nets.state_dict().items()}
    torch.save({"state_dict": state}, ckpt)
    training = tmp_path / "flow.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {
                    "pipeline": {
                        "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                        "stages": [
                            {
                                "_target_": "tests.test_robot_graph_policy.ConditionStage"
                            },
                            {
                                "_target_": "egomimic.pipeline.stages_flow.FlowDenoiserStage",
                                "model": {
                                    "_target_": "torch.nn.Linear",
                                    "in_features": 1,
                                    "out_features": 1,
                                },
                                "action_horizon": 2,
                                "action_dim": 14,
                                "condition_input_dim": 8,
                                "num_inference_steps": 50,
                            },
                        ],
                    }
                }
            }
        ),
        training,
    )
    OmegaConf.save(
        declare_test_model(
            OmegaConf.load(training), control_path="num_inference_steps"
        ),
        training,
    )
    config = OmegaConf.load(training)
    config.model.pipeline.stage_ids.sampler = 1
    OmegaConf.save(config, training)
    save_bound_checkpoint(ckpt, config, state)
    boundary = dict(
        _target_="egomimic.robot.graph_policy.CartesianGraphAdapter",
        base_T_model={a: np.eye(4).tolist() for a in ("left", "right")},
        camera_keys={"front_img_1": "front"},
        embodiment_id=7,
        rotation_mode="euler",
        action_frame="eef_frame",
        image_hw=[2, 3],
    )

    policy = load_graph_policy(
        dict(
            normalizer_path=str(tmp_path / "data-context.json"),
            training_config=str(training),
            checkpoint=str(ckpt),
            device="cpu",
            adapter=boundary,
        )
    )

    assert policy.graph.pipeline.stages[1].num_inference_steps == 10
    assert policy.inference_controls()["inference_steps"]["value"] == 10
    assert policy.execution_plan(np.zeros((2, 14))).shape == (1, 14)


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


def test_graph_policy_resamples_whole_invalid_plan_without_clamping_grippers():
    stage = RetryStage(valid_after=2)
    policy = GraphRobotPolicy(
        PipelineAlgo([stage], device="cpu"),
        normalizer(),
        adapter(),
        max_valid_samples=2,
    )

    result = policy.predict(FakeRobot().get_obs())

    assert stage.calls == 2
    np.testing.assert_allclose(result[:, [6, 13]], 0.5, atol=1e-6)


def test_graph_adapter_clips_only_small_gripper_overshoots():
    native = np.zeros((1, 1, 14))
    native[0, 0, [6, 13]] = [-0.04, 1.04]

    result = adapter(gripper_clip_tolerance=0.05).actions(native, FakeRobot().get_obs())

    np.testing.assert_allclose(result[:, [6, 13]], [[0, 1]])
    native[0, 0, 6] = -0.06
    with pytest.raises(ValueError, match="gripper opening"):
        adapter(gripper_clip_tolerance=0.05).actions(native, FakeRobot().get_obs())


def test_graph_policy_never_commands_when_all_stochastic_samples_are_invalid():
    stage = RetryStage(valid_after=3)
    policy = GraphRobotPolicy(
        PipelineAlgo([stage], device="cpu"),
        normalizer(),
        adapter(),
        max_valid_samples=2,
    )

    with pytest.raises(ValueError, match="rejected all 2 sampled"):
        policy.predict(FakeRobot().get_obs())
    assert stage.calls == 2


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


@pytest.mark.parametrize(
    ("variant", "action_dim", "expected_layout"),
    [
        ("time", 14, None),
        ("arcvel", 16, "e1_profile"),
        ("arcdur", 16, "e1_dur"),
    ],
)
def test_selected_e1_model_derives_its_rollout_decoder(
    variant, action_dim, expected_layout
):
    training = OmegaConf.create(
        {
            "e1": {
                "variant": variant,
                "D": 0.4,
                "M": 100,
                "chunk_length": 100,
            },
            "hpt": {"action_dim": action_dim, "action_horizon": 100},
            "evaluator": {"dt": 1 / 30},
            "model": {
                "pipeline": {
                    "stages": [
                        {
                            "_target_": "egomimic.pipeline.stages_flow.FlowDenoiserStage",
                            "action_horizon": 100,
                            "action_dim": action_dim,
                        }
                    ]
                }
            },
        }
    )
    adapter_config = {
        "_target_": "egomimic.robot.graph_policy.CartesianGraphAdapter",
        "decoder": {"_target_": "old.decoder"},
    }

    training.model.pipeline.stage_ids = {"sampler": 0}
    training.model.inference = {"native_output": {"shape": [100, action_dim]}}
    profiles = {
        "default": {
            "stage_id": "sampler",
            "native_shape": [100, action_dim],
            "adapter": {
                "decoder": None
                if expected_layout is None
                else {
                    "_target_": "egomimic.robot.arc_decoder.BimanualArcDecoder",
                    "token_layout": expected_layout,
                }
            },
        }
    }
    selected = configure_adapter_for_training(adapter_config, training, profiles)

    if expected_layout is None:
        assert "decoder" not in selected
    else:
        assert selected["decoder"] == {
            "_target_": "egomimic.robot.arc_decoder.BimanualArcDecoder",
            "token_layout": expected_layout,
        }


def test_selected_e1_model_rejects_an_incompatible_token_width():
    training = OmegaConf.create(
        {
            "e1": {"variant": "arcvel"},
            "hpt": {"action_dim": 14, "action_horizon": 100},
            "model": {
                "pipeline": {
                    "stages": [
                        {
                            "_target_": "egomimic.pipeline.stages_flow.FlowDenoiserStage",
                            "action_horizon": 100,
                            "action_dim": 14,
                        }
                    ]
                }
            },
        }
    )

    training.model.pipeline.stage_ids = {"sampler": 0}
    training.model.inference = {"native_output": {"shape": [100, 14]}}
    with pytest.raises(ValueError, match="expects"):
        configure_adapter_for_training(
            {},
            training,
            {
                "flow_arcvel": {
                    "stage_id": "sampler",
                    "native_shape": [100, 16],
                    "adapter": {},
                }
            },
        )


def test_diffusion_profile_exposes_typed_controls_and_needs_no_arc_variant():
    target = f"{ProfileSamplerStage.__module__}.{ProfileSamplerStage.__qualname__}"
    training = OmegaConf.create(
        {
            "model": {
                "pipeline": {
                    "stages": [
                        {
                            "_target_": target,
                            "action_horizon": 100,
                            "action_dim": 14,
                        }
                    ]
                }
            }
        }
    )
    declare_test_model(training, horizon=100, control_path="policy.num_inference_steps")
    profiles = {
        "diffusion_time": {
            "stage_id": "sampler",
            "native_shape": [100, 14],
            "overrides": {
                "inference_steps": {
                    "label": "Diffusion denoising steps",
                    "description": "Denoising budget.",
                    "type": "integer",
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "default": 12,
                    "target": {
                        "kind": "stage_attribute",
                        "stage_id": "sampler",
                        "attribute_path": "policy.num_inference_steps",
                    },
                },
                "replan_every": {
                    "label": "Repredict every",
                    "description": "Execution prefix.",
                    "type": "integer",
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "default": 30,
                    "target": {
                        "kind": "policy_attribute",
                        "attribute_path": "replan_every",
                    },
                },
            },
            "adapter": {"decoder": None},
        }
    }
    stage = ProfileSamplerStage()
    graph = PipelineAlgo([stage], device="cpu", stage_ids={"sampler": 0})

    controls = configure_profile_controls(graph, training, profiles)
    selected = configure_adapter_for_training(
        {"decoder": {"_target_": "old.decoder"}}, training, profiles
    )
    policy = GraphRobotPolicy(
        graph,
        normalizer(),
        adapter(),
        inference_controls=controls,
    )

    assert stage.policy.num_inference_steps == 12
    assert "decoder" not in selected
    assert policy.inference_controls()["replan_every"]["value"] == 30
    assert policy.execution_plan(np.zeros((100, 14))).shape == (30, 14)

    controls = policy.apply_inference_overrides(
        {"inference_steps": 7, "replan_every": 4}
    )

    assert stage.policy.num_inference_steps == 7
    assert controls["inference_steps"]["value"] == 7
    assert policy.execution_plan(np.zeros((100, 14))).shape == (4, 14)
    with pytest.raises(ValueError, match="not exposed"):
        policy.apply_inference_overrides({"unknown": 1})


def test_inference_graph_owns_history_and_enforces_canonical_output():
    stage = EchoStage()
    contract = {
        "input": {"history_length": 2},
        "output": {"representation": "cartesian", "shape": [2, 14]},
    }
    policy = GraphRobotPolicy(
        PipelineAlgo([stage], device="cpu"),
        normalizer(),
        adapter(),
        inference_graph=contract,
    )

    result = policy.predict(FakeRobot().get_obs())

    assert stage.seen.shape == (1, 2, 14)
    assert result.shape == (2, 14)
    policy.reset()
    assert not policy._observation_history

    incompatible = GraphRobotPolicy(
        PipelineAlgo([EchoStage()], device="cpu"),
        normalizer(),
        adapter(),
        inference_graph={
            "input": {"history_length": 1},
            "output": {"representation": "cartesian", "shape": [3, 14]},
        },
    )
    with pytest.raises(ValueError, match="canonical output contract"):
        incompatible.predict(FakeRobot().get_obs())


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


def test_graph_policy_rejects_unsupported_cuda_before_model_or_robot_loading(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])

    with pytest.raises(RuntimeError, match="sm_120"):
        validate_graph_device("cuda:0")


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
    OmegaConf.save(
        declare_test_model(OmegaConf.load(training), control_path=None), training
    )
    checkpoint = save_bound_checkpoint(ckpt, OmegaConf.load(training), state)
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
        normalizer_path=str(tmp_path / "data-context.json"),
        training_config=str(training),
        checkpoint=str(ckpt),
        device="cpu",
        adapter=boundary,
    )
    assert load_graph_policy(config).predict(FakeRobot().get_obs()).shape == (2, 14)
    checkpoint["state_dict"] = {**state, "nets.unexpected": torch.zeros(1)}
    torch.save(checkpoint, ckpt)
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
