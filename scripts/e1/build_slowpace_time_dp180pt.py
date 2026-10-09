"""Regenerate the DP-180 time leaf through the maintained DP graph."""

from recipe_builders import compose_recipe, model_leaf, require_compute_node

if __name__ == "__main__":
    require_compute_node()
    model_leaf(
        "dp180pt_wrists_time",
        "dp300pt_wrists_arcdur",
        action_token_dim=14,
        unet_down_dims=[488, 976, 1952],
    )
    cfg = compose_recipe("scratch_rl2_stattempo_slowpace_time_dp180pt")
    assert cfg.model.action_token_dim == 14
