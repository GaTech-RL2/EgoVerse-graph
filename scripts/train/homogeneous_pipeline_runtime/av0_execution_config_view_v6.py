"""Resolve execution-relevant config only; standalone Hydra resolver may be absent."""
def execution_config_view(config):
    from omegaconf import OmegaConf
    return {"model":{"pipeline":OmegaConf.to_container(config.model.pipeline,resolve=True)},
            "data":{"train_datasets":{str(k):None for k in config.data.train_datasets},
                    "train_dataloader_params":{str(k):{"batch_size":int(v.batch_size)} for k,v in config.data.train_dataloader_params.items()}}}
