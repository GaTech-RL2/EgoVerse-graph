"""Data/evaluator boundaries work with opaque sources and immutable state."""

import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from egomimic.eval.eval import EvaluationDataRequirements, validation_trainer_overrides
from egomimic.pl_utils.pl_data_utils import MultiDataModuleWrapper, annotation_collate
from egomimic.rldb.zarr.data_module import ZarrDataModule, _digest
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class TinyDataset(Dataset):
    def __init__(self):
        self.norm_stats = None

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "embodiment": 7,
            "action": torch.zeros(2, 3),
            "record": "a" if index < 2 else "b",
            "frame": index % 2,
        }

    def episode_id_at(self, index):
        return "a" if index < 2 else "b"

    def frame_index_at(self, index):
        return index % 2

    def episode_length_at(self, index):
        return 2

    def set_norm_stats_from(self, owner):
        self.norm_stats = owner.norm_stats


def must_not_open_training():
    raise AssertionError("Standalone evaluation reopened the original training corpus")


def _state():
    return {
        "norm_mode": "quantile",
        "embodiments": [7],
        "key_types": {7: {"action": "action_keys"}},
        "zarr_keys": {7: {"action": "action"}},
        "shapes": {7: {"action": [2, 3]}},
        "norm_stats": {
            7: {
                "action": {
                    "quantile_1": np.zeros((2, 3)),
                    "quantile_99": np.ones((2, 3)),
                }
            }
        },
    }


def _module():
    return ZarrDataModule(
        train_datasets=OmegaConf.create(
            {"source": {"_target_": __name__ + ".must_not_open_training"}}
        ),
        valid_datasets=OmegaConf.create(
            {"unregistered_source": {"_target_": __name__ + ".TinyDataset"}}
        ),
        train_dataloader_params={},
        valid_dataloader_params={"unregistered_source": {"batch_size": 2}},
    )


def test_eval_restores_context_without_constructing_training_dataset():
    state = _state()
    module = _module()
    context = module.prepare_context(
        mode="eval",
        normalization={"norm_mode": "quantile"},
        restored_state={
            "kind": "zarr-normalizer-v1",
            "normalizer_state": state,
            "sha256": _digest(state),
        },
    )
    assert not module.train_datasets
    assert (
        module.valid_datasets["unregistered_source"].norm_stats
        is context.normalizer.norm_stats
    )
    assert context.validation_groups == ("valid",)
    assert context.snapshot()["sha256"] == _digest(state)


def test_eval_requires_context_and_rejects_corruption_or_mode_drift():
    with pytest.raises(ValueError, match="requires checkpoint data_context"):
        _module().prepare_context(mode="eval", normalization={})
    state = _state()
    saved = {
        "kind": "zarr-normalizer-v1",
        "normalizer_state": state,
        "sha256": _digest(state),
    }
    corrupted = deepcopy(saved)
    corrupted["normalizer_state"]["norm_stats"][7]["action"]["quantile_1"][0, 0] = 3
    with pytest.raises(ValueError, match="hash mismatch"):
        _module().prepare_context(
            mode="eval", normalization={}, restored_state=corrupted
        )
    with pytest.raises(ValueError, match="normalization mode differs"):
        _module().prepare_context(
            mode="eval", normalization={"norm_mode": "zscore"}, restored_state=saved
        )


def test_evaluator_capabilities_force_order_and_limit_complete_episodes():
    module = MultiDataModuleWrapper(
        {},
        {"opaque": TinyDataset()},
        {},
        {"opaque": {"batch_size": 3, "shuffle": True}},
    )
    module.configure_evaluation(
        EvaluationDataRequirements(
            ordered=True,
            complete_episodes=True,
            max_episodes=1,
            sample_id_key="record",
            frame_index_key="frame",
        )
    )
    assert len(module.valid_datasets["opaque"]) == 2
    loader = module.val_dataloader().iterables["opaque"]
    assert next(iter(loader))["frame"].tolist() == [0, 1]


def test_evaluator_preflight_rejects_missing_metadata_and_dropped_episode_tails():
    module = MultiDataModuleWrapper(
        {},
        {"opaque": TinyDataset()},
        {},
        {"opaque": {"batch_size": 3, "drop_last": True}},
    )
    with pytest.raises(ValueError, match="lacks metadata"):
        module.configure_evaluation(EvaluationDataRequirements(sample_id_key="missing"))
    with pytest.raises(ValueError, match="drop_last=False"):
        module.configure_evaluation(EvaluationDataRequirements(complete_episodes=True))


def test_evaluator_cannot_change_cluster_policy():
    class Bad:
        def trainer_overrides(self):
            return {"devices": 8}

    with pytest.raises(ValueError, match="validation-loop settings"):
        validation_trainer_overrides(Bad())


def test_annotation_collation_does_not_mutate_dataset_cached_samples():
    sample = {"text": ["pick", "place"], "action": torch.zeros(2)}
    for _ in range(2):
        batch = annotation_collate([sample])
        assert batch["text"] == [["pick", "place"]]
    assert sample["text"] == ["pick", "place"]


def test_local_training_imports_do_not_require_cloud_or_pi_backends():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'boto3', 'botocore', 'cloudpathlib', 'sqlalchemy', 'psycopg', 'openpi'}:
            raise AssertionError('Unexpected optional backend import: ' + fullname)
sys.meta_path.insert(0, Block())
import egomimic.trainHydra
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], []])
def test_normalization_rejects_a_source_with_no_finite_rows(values):
    with pytest.raises(ValueError, match="finite"):
        MultiDataset._compute_stats_for_array(np.asarray(values).reshape(-1, 1))


def test_normalization_excludes_bad_rows_without_poisoning_other_samples():
    stats = MultiDataset._compute_stats_for_array(
        np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, 9.0]])
    )
    assert np.array_equal(stats["mean"], [2.0, 3.0])
    assert all(np.isfinite(value).all() for value in stats.values())


@pytest.mark.parametrize(
    "options,mode",
    [
        ({"limit_val_batches": 1}, "eval"),
        ({"limit_val_batches": 0.5}, "eval"),
        ({"fast_dev_run": True}, "eval"),
        ({"overfit_batches": 1}, "eval"),
        ({"num_sanity_val_steps": 2}, "train"),
    ],
)
def test_complete_episode_preflight_rejects_truncated_trainer(options, mode):
    from egomimic.eval.eval import validate_validation_loop

    with pytest.raises(ValueError, match="Complete-episode"):
        validate_validation_loop(
            EvaluationDataRequirements(complete_episodes=True), options, mode=mode
        )


def test_complete_episode_preflight_accepts_full_loop_and_disabled_sanity():
    from egomimic.eval.eval import validate_validation_loop

    validate_validation_loop(
        EvaluationDataRequirements(complete_episodes=True),
        {"limit_val_batches": 1.0, "num_sanity_val_steps": 0, "devices": 8},
        mode="train",
    )


def test_complete_episode_preflight_rejects_missing_tail():
    class MissingTail(TinyDataset):
        def __len__(self):
            return 3

    module = MultiDataModuleWrapper(
        {}, {"source": MissingTail()}, {}, {"source": {"batch_size": 1}}
    )
    with pytest.raises(ValueError, match="missing its tail"):
        module.configure_evaluation(EvaluationDataRequirements(complete_episodes=True))
