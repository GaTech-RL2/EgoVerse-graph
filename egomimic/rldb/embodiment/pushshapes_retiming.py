from egomimic.rldb.embodiment.pushshapes import (
    get_planar_dense_transform_list,
    get_planar_keymap,
    get_planar_keymap_per_source_proprio,
)
from egomimic.rldb.zarr.action_chunk_transforms import PlanarAgentStateToRotVec4
from egomimic.rldb.zarr.planar_retiming import PlanarCommandRetiming


def get_standard_dp_retimed_keymap(
    rates,
    action_horizon=16,
    observation_horizon=1,
    action_target_offset=0,
    norm_mode=False,
    fps=30.0,
    model_proprio=False,
):
    if action_horizon != 16 or observation_horizon != 1 or action_target_offset != 0:
        raise ValueError("Standard-DP H16 obs1 pre-step contract required")
    transform = PlanarCommandRetiming(rates=rates, horizon=action_horizon, fps=fps)
    keymap_factory = (
        get_planar_keymap_per_source_proprio if model_proprio else get_planar_keymap
    )
    return keymap_factory(
        action_horizon=transform.required_frames,
        observation_horizon=observation_horizon,
        action_target_offset=action_target_offset,
        norm_mode=norm_mode,
    )


def get_standard_dp_retimed_transform_list(
    rates, action_horizon=16, fps=30.0, model_proprio=False, **_kwargs
):
    if action_horizon != 16:
        raise ValueError("Standard-DP H16 contract required")
    return [
        PlanarCommandRetiming(rates=rates, horizon=action_horizon, fps=fps),
        *get_planar_dense_transform_list(),
        *(
            [PlanarAgentStateToRotVec4(keys=["state_agent_model"])]
            if model_proprio
            else []
        ),
    ]
