import copy
import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from egomimic.benchmarks.libero.training_checks import (
    StepTiming,
    check_weights_only_payload,
)
from egomimic.trainHydra import _build_model_config_tree


def config():
    return OmegaConf.create(
        {
            "model": {"pipeline": {"stages": [{"codec": {"basis": "uniform"}}]}},
            "run_provenance": {"basis": "uniform", "source_commit": "exact"},
            "logger": {"wandb": {"id": "exact-run"}},
            "paths": {"output_dir": "/not-the-callback-path"},
            "trainer": {"devices": 4, "max_steps": 16},
            "callbacks": {"batch_budget": {"global_batch_size": 1024}},
        }
    )


def test_timing_accepts_actual_minimal_checkpoint_config(tmp_path):
    saved = _build_model_config_tree(config())
    assert set(saved) == {"model", "run_provenance"} and "paths" not in saved
    model = SimpleNamespace(hparams=SimpleNamespace(config_tree=saved))
    trainer = SimpleNamespace(
        global_step=1, global_rank=0, world_size=4, sanity_checking=False
    )
    callback = StepTiming(tmp_path)
    callback.on_train_batch_start(trainer, model, None, 0)
    callback.on_train_batch_end(trainer, model, None, None, 0)
    callback.on_validation_start(trainer, model)
    callback.on_validation_end(trainer, model)
    callback.on_train_end(trainer, model)
    receipt = json.loads((tmp_path / "timing-rank-0.json").read_text())
    assert receipt["global_step"] == 1 and receipt["validations"][0]["step"] == 1
    trainer.global_step = 0
    with pytest.raises(AssertionError):
        callback.on_validation_start(trainer, model)


@pytest.mark.parametrize(
    "bad", [None, "optimizer", "step", "ema", "basis", "run", "norm", "budget"]
)
def test_weights_only_identity_uses_real_schema(bad):
    cfg = config()
    norm = {"benchmark_context": {"data": "exact"}, "stats": {"scale": np.ones(7)}}
    payload = {
        "global_step": 16,
        "ema_num_updates": 16,
        "ema_state_dict": {},
        "training_budget": {
            "world_size": 4,
            "global_batch_size": 1024,
            "total_optimizer_steps": 16,
        },
        "hyper_parameters": {"config_tree": _build_model_config_tree(cfg)},
        "normalizer_state": copy.deepcopy(norm),
    }
    if bad == "optimizer":
        payload["optimizer_states"] = []
    elif bad in ("step", "ema"):
        payload["global_step" if bad == "step" else "ema_num_updates"] = 8
    elif bad == "basis":
        payload["hyper_parameters"]["config_tree"].model.pipeline.stages[
            0
        ].codec.basis = "fourier"
    elif bad == "run":
        payload["hyper_parameters"]["config_tree"].run_provenance.run_id = "wrong"
    elif bad == "norm":
        payload["normalizer_state"]["stats"]["scale"][0] = 2
    elif bad == "budget":
        payload["training_budget"]["global_batch_size"] = 512
    if bad is None:
        check_weights_only_payload(payload, cfg, 16, norm)
    else:
        with pytest.raises(AssertionError):
            check_weights_only_payload(payload, cfg, 16, norm)
