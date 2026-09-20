"""Architecture restoration without importing an OAT workspace or old paths."""

from copy import deepcopy

from omegaconf import OmegaConf, open_dict


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
