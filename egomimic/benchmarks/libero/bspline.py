"""Validation of the matched-backbone, fixed-replanning LIBERO BSP protocol."""

from egomimic.models.bspline import BSplineCodec

METHODS = {"unet": "bspline_unet", "oat_dp": "bspline_oat"}


def validate_bspline_config(config, method):
    if method not in METHODS.values():
        raise ValueError("Unknown BSP policy method")
    backbone = {v: k for k, v in METHODS.items()}[method]
    model = config["model"]
    protocol = model["benchmark_protocol"]
    codec = BSplineCodec(**dict(protocol["bspline"]))
    stages = model["pipeline"]["stages"]
    expected = (
        "egomimic.pipeline.stages_oat.OATObservationStage",
        "egomimic.pipeline.stages_io.ActionTargetBuilder",
        "egomimic.pipeline.stages_diffusion.DiffusionNoisingStage",
        "egomimic.pipeline.stages_diffusion.DiffusionDenoiserStage",
        "egomimic.pipeline.stages_diffusion.DiffusionEpsilonLossStage",
        "egomimic.pipeline.stages_bspline.BSplineDecodeStage",
    )
    if (
        tuple(s["_target_"] for s in stages) != expected
        or protocol.get("action_representation") != "bspline_policy_v1"
        or protocol.get("dp_backbone") != backbone
        or tuple(protocol.get(k) for k in ("horizon", "n_obs_steps", "n_action_steps"))
        != (32, 2, 16)
        or protocol.get("segment_alignment") is not False
        or protocol.get("execution_speed") != 1.0
        or protocol.get("segment_end_behavior") != "hold_until_fixed_replan"
        or protocol.get("upstream_commit") != codec.context()["upstream_commit"]
        or stages[1].get("action_key") != "bspline"
        or any(
            s.get("action_dim") != 8 or s.get("action_horizon") != codec.num_knots
            for s in stages[2:4]
        )
        or dict(stages[5]["codec"]) != dict(protocol["bspline"])
        or stages[5]["horizon"] != 32
    ):
        raise ValueError("BSP graph or fixed LIBERO execution protocol differs")
    policy = stages[3]["policy"]
    network = policy["model"]
    target = (
        "egomimic.models.denoising_nets.ConditionalUnet1D"
        if backbone == "unet"
        else "egomimic.models.oat.diffusion.GraphDiffusionTransformer"
    )
    if (
        network["_target_"] != target
        or network["input_dim"] != 8
        or policy["action_horizon"] != codec.num_knots
        or policy["num_inference_steps"] != (100 if backbone == "unet" else 10)
        or (
            backbone == "oat_dp"
            and (network["output_dim"] != 8 or network["horizon"] != codec.num_knots)
        )
    ):
        raise ValueError("BSP diffusion backbone or sampling configuration differs")
