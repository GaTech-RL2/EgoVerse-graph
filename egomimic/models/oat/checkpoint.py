"""Architecture restoration without importing an OAT workspace or old paths."""

from copy import deepcopy

from omegaconf import OmegaConf, open_dict


def validate_input_representation(stages, payload=None):
    """Require tokenizer, policy and ARC decoder to share the same units/codec.

    Legacy raw-action checkpoints have no representation field. ARC-tokenizer
    checkpoints must explicitly record their representation; tensor shapes
    alone cannot distinguish STK, DUR or different R/D settings.
    """
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage
    from egomimic.pipeline.stages_oat import OATPolicyStage, OATTokenizerStage

    stages = list(stages)
    learned = [s for s in stages if isinstance(s, (OATPolicyStage, OATTokenizerStage))]
    if not learned:
        return None
    if len(learned) != 1:
        raise ValueError("Expected one OAT tokenizer or policy stage")
    stage = learned[0]
    tokenizer = (
        stage.policy.action_tokenizer
        if isinstance(stage, OATPolicyStage)
        else stage.tokenizer
    )
    codecs = [s for s in stages if isinstance(s, LiberoArcStage)]
    representation = None
    if codecs:
        if (
            len(codecs) != 2
            or [s.operation for s in codecs] != ["encode", "decode"]
            or any(s.reconstruction for s in codecs)
            or not stages.index(codecs[0])
            < stages.index(stage)
            < stages.index(codecs[1])
            or (stage.action_key, stage.prediction_key) != ("target", "pred_arc")
            or codecs[0].encode_inference != isinstance(stage, OATTokenizerStage)
        ):
            raise ValueError("ARC+OAT requires encode, learned ARC tokens, then decode")
        representation = codecs[0].representation_context()
        if codecs[1].representation_context() != representation:
            raise ValueError("ARC+OAT encoder and decoder representations differ")
        if tokenizer.decoder.sample_horizon != representation[
            "num_waypoints"
        ] or tokenizer.decoder.sample_dim != len(representation["token_scale"]):
            raise ValueError("OAT tokenizer shape differs from ARC supports")
    elif (stage.action_key, stage.prediction_key) != ("actions", "pred_action"):
        raise ValueError("OAT representation ports lack their ARC codec")
    if (
        payload is not None
        and payload.get("oat_input_representation") != representation
    ):
        raise ValueError("Checkpoint and graph OAT input representations differ")
    if isinstance(stage, OATPolicyStage):
        if payload is not None:
            tokenizer._training_input_representation = representation
        elif (
            getattr(tokenizer, "_training_input_representation", None) != representation
        ):
            raise ValueError("Tokenizer and policy ARC representations differ")
    return representation


def policy_config_without_external_tokenizer(config, payload):
    config = deepcopy(config)
    for stage in config.model.pipeline.stages:
        if stage._target_ != "egomimic.pipeline.stages_oat.OATPolicyStage":
            continue
        if "oat_tokenizer_config" not in payload:
            raise ValueError("Policy checkpoint lacks its tokenizer architecture")
        with open_dict(stage.policy):
            stage.policy._target_ = (
                "egomimic.models.oat.factory.make_policy_from_config"
            )
            stage.policy._recursive_ = False
            stage.policy.pop("tokenizer_checkpoint", None)
            stage.policy.pop("use_ema", None)
            stage.policy.tokenizer_config = OmegaConf.create(
                payload["oat_tokenizer_config"]
            )
    return config
