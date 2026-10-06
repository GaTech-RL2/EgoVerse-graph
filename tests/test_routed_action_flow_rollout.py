import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.eval.core.action_flow_rollout import (
    RoutedActionFlowPolicy, action_flow_contract, action_flow_metadata,
)
from egomimic.pipeline.stages_action_flow import ConditionalVelocityStage
from egomimic.pipeline.stages_sampler import DPStyleObsEncoder, FusedObsEncoder
from egomimic.rldb.zarr.chain_gripper_points import pose_control_to_points


def config():
    prefix = "egomimic.pipeline.pushshapes."
    names = ["pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"]
    return OmegaConf.create({
        "model": {"pipeline": {"stages": [{
            "_target_": "egomimic.pipeline.stages_action_flow.ConditionalVelocityStage",
            "inference_method": "euler", "num_inference_steps": 50,
            "cfg_scale": 4.0, "cfg_interval": [0., 1.], "timestep_shift_alpha": .5,
        }]}},
        "planar": {"action_horizon": 16, "raw_action_horizon": 16,
                   "observation_horizon": 1, "action_target_offset": 0,
                   "action_dims": dict(zip(names, [4, 6]))},
        "deployment": {
            "observation_adapters": dict(zip(names, [
                {"_target_": prefix + "USocketModelStateObservationAdapter"},
                {"_target_": prefix + "ChainGripperModelStateObservationAdapter"}])),
            "action_decoders": dict(zip(names, [
                {"_target_": prefix + "USocketRotVecNativeDecoder"},
                {"_target_": prefix + "ChainGripperPointsNativeDecoder"}]))}})


@pytest.mark.parametrize("name,emb,width", [("pushshapes_sim_u_socket", 19, 3),
                                          ("pushshapes_sim_chain_gripper", 20, 4)])
def test_boundary_routes_rotvec_state_and_decodes_native(name, emb, width):
    cfg = config()
    decoder = action_flow_contract(cfg, selected_embodiment_name=name,
        selected_embodiment_id=emb, expected_native_action_dim=width, replan_every=8)
    control = np.tile([256., 128., .3, .5], (16, 1))
    tokens = (np.column_stack((control[:, :2], np.cos(control[:, 2]), np.sin(control[:, 2])))
              if emb == 19 else pose_control_to_points(control))
    class ImageEncoder(torch.nn.Module):
        def forward(self, image):
            assert image.shape == (1, 3, 96, 96)
            return torch.zeros(image.shape[0], 64)
    fused = FusedObsEncoder(
        DPStyleObsEncoder(obs_specs={"state_agent_model": {"input_dim": 4}},
                          img_encoders={"front_img_1": ImageEncoder()}),
        inputs={"state_agent_model": "state_agent_model", "front_img_1": "front_img_1"},
        n_obs_steps=1)
    class Normalizer:
        def normalize(self, data, embodiment_id):
            assert embodiment_id == emb
            assert data["state_agent_model"].shape == (1, 4)
            assert data["front_img_1"].shape == (1, 3, 96, 96)
            # Exercise the original checkpoint's packed single-observation
            # encoder contract, rather than accepting a shape-only fake Algo.
            assert fused.forward(dict(data))["condition"].shape == (1, 68)
            return dict(data, state_agent_model=data["state_agent_model"] / 512)
        def unnormalize(self, data, embodiment_id):
            assert embodiment_id == emb
            return data
    class Algo:
        def forward_eval(self, batch):
            assert list(batch) == [name]
            assert batch[name]["embodiment"].tolist() == [emb]
            return {name: {"pred_action": torch.tensor(tokens, dtype=torch.float32)[None]}}
    policy = RoutedActionFlowPolicy(algo=Algo(), normalizer=Normalizer(), decoder=decoder,
        embodiment_id=emb, device=torch.device("cpu"), cfg=cfg, embodiment_name=name)
    output = policy.predict_native_actions({"agent_pos": np.array([256., 128.]),
        "agent_angle": np.array([.3]), "object_pose": np.array([300., 300., 0.]),
        "image": np.zeros((96, 96, 3), dtype=np.uint8)})
    assert output.shape == (16, width) and output.dtype == np.float32
    np.testing.assert_allclose(output, control[:, :width], atol=1e-4)


@pytest.mark.parametrize("key,value", [("inference_method", "dopri5"),
                                      ("num_inference_steps", 8)])
def test_contract_rejects_sampler_drift(key, value):
    cfg = config()
    cfg.model.pipeline.stages[0][key] = value
    with pytest.raises(ValueError, match="Euler-50"):
        action_flow_contract(cfg, selected_embodiment_name="pushshapes_sim_u_socket",
            selected_embodiment_id=19, expected_native_action_dim=3, replan_every=8)


def test_euler50_uses_original_stage_and_unshifted_grid():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.times = []
        def forward(self, state, time, condition, condition_drop_mask=None):
            self.times.append(time.clone())
            return torch.ones_like(state)
    field = Field()
    stage = ConditionalVelocityStage(field, inference_method="euler",
        num_inference_steps=50, cfg_scale=4., cfg_interval=(0., 1.), timestep_shift_alpha=.5)
    result = stage.execute({"sampler/noise": torch.ones(1, 8, 16),
                            "condition": torch.ones(1, 128)}, mode="inference")
    assert len(field.times) == 100  # conditioned + unconditioned at each update
    torch.testing.assert_close(torch.cat(field.times[::2]), torch.arange(50) * -.02 + 1.)
    torch.testing.assert_close(result["action_flow/generated_latent"], torch.zeros(1, 8, 16), atol=1e-6, rtol=0)
    assert action_flow_metadata(config())["timestep_shift_active"] is False


def test_bf16_refuses_cpu_instead_of_silently_using_fp32():
    with pytest.raises(ValueError, match="BF16-capable CUDA"):
        RoutedActionFlowPolicy(algo=None, normalizer=None, decoder=None,
            embodiment_id=19, device=torch.device("cpu"), cfg=config(),
            embodiment_name="pushshapes_sim_u_socket", model_autocast_precision="bf16")


def test_canonical_precision_cli_is_explicit():
    from egomimic.eval.core.ckpt_loading import build_parser
    action = next(a for a in build_parser()._actions if a.dest == "model_autocast_precision")
    assert action.default == "fp32"
    assert action.choices == ("fp32", "bf16")


def test_bf16_neural_outputs_keep_original_noise_and_euler_state_fp32():
    from types import SimpleNamespace
    from egomimic.pipeline.stages_sampler import GaussianLatentNoise
    from egomimic.eval.core.action_flow_rollout import install_fp32_sampler_boundaries
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)
            self.states, self.outputs = [], []
        def forward(self, state, time, condition, condition_drop_mask=None):
            self.states.append(state.dtype)
            result = self.linear(state)
            self.outputs.append(result.dtype)
            return result
    field = Field()
    noise = GaussianLatentNoise(8, 16)
    stage = ConditionalVelocityStage(field, inference_method="euler",
        num_inference_steps=50, cfg_scale=4., cfg_interval=(0., 1.))
    algo = SimpleNamespace(nets=torch.nn.ModuleList([noise, stage]))
    install_fp32_sampler_boundaries(algo)
    torch.manual_seed(42)
    expected = torch.randn(1, 8, 16)
    torch.manual_seed(42)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        batch = noise.forward({"condition": torch.ones(1, 128, dtype=torch.bfloat16)})
        torch.testing.assert_close(batch["sampler/noise"], expected, rtol=0, atol=0)
        result = stage.execute(batch, mode="inference")
    assert set(field.states) == {torch.float32}
    assert set(field.outputs) == {torch.bfloat16}
    assert result["action_flow/generated_latent"].dtype == torch.float32


def test_cpu_checkpoint_path_does_not_query_cuda_health(monkeypatch):
    from egomimic.eval.core.action_flow_rollout import validate_cuda_health
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU must not query CUDA")
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    validate_cuda_health(torch.device("cpu"))


@pytest.mark.parametrize("uuid", ["01234567-89ab-cdef-0123-456789abcdef",
                                 "GPU-01234567-89ab-cdef-0123-456789abcdef"])
def test_cuda_health_uses_nvml_uuid_prefix(monkeypatch, uuid):
    from types import SimpleNamespace
    import egomimic.eval.core.action_flow_rollout as bridge
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid=uuid))
    calls = []
    def query(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="healthy")
    monkeypatch.setattr(bridge.subprocess, "run", query)
    monkeypatch.setattr(bridge, "parse_ecc_health", lambda report: None)
    bridge.validate_cuda_health(torch.device("cuda:0"))
    assert calls == [["nvidia-smi", "-q", "-i", "GPU-01234567-89ab-cdef-0123-456789abcdef"]]
