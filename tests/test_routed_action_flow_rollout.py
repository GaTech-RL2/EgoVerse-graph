import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.eval.core.action_flow_rollout import (
    RoutedActionFlowPolicy, action_flow_contract, action_flow_metadata,
)
from egomimic.pipeline.stages_action_flow import ConditionalVelocityStage
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
    class Normalizer:
        def normalize(self, data, embodiment_id):
            assert embodiment_id == emb
            assert data["state_agent_model"].shape == (1, 1, 4)
            assert data["front_img_1"].shape == (1, 1, 3, 96, 96)
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
