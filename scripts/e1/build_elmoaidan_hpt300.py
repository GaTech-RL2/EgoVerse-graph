"""Regenerate the shared time data and inherited HPT-300 time model leaf."""

from recipe_builders import build_elmoaidan, compose_recipe, model_leaf

if __name__ == "__main__":
    build_elmoaidan(("time", "arcdur"))
    model_leaf(
        "hpt300_flow_wrists_time", "hpt300_flow_wrists_arcdur", action_token_dim=14
    )
    for variant in ("time", "arcdur"):
        cfg = compose_recipe(f"scratch_rl2_stattempo_elmoaidan_{variant}_hpt300")
        assert cfg.model.embed_dim == 840 and cfg.model.trunk_blocks == 19
