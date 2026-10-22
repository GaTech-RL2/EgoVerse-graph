"""CPU tensor/checkpoint rejection tests; no production checkpoint is read."""

import copy
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.train.joint import crash_inspect as target


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CheckpointInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=.01)
        self.model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        for state in optimizer.state.values():
            state["step"] = torch.tensor(20000.)
        self.parent = dict(source_revision=target.FROZEN_SOURCE, run_id=target.RUN_ID,
                           source_manifest_sha256="a"*64, data_manifest_sha256="b"*64,
                           world_size=4, global_batch=32, per_rank_batch=8,
                           lr_policy=dict(schedule="constant", peak_lr=1e-4, warmup_optimizer_steps=0),
                           target_steps=300000, output=str(self.root/"run"))
        self.resolved = dict(model=dict(optimizer=dict(_target_="torch.optim.AdamW", lr=1e-4,
                                                     weight_decay=.01), scheduler=None))
        context = {"schema": "synthetic-context", "normalizer": {"mean": [0., 0., 0.]}}
        self.context_hash = fingerprint(context)
        provenance = {key: self.parent[key] for key in ("source_revision", "source_manifest_sha256",
                                                        "data_manifest_sha256", "run_id")}
        provenance["data_context_sha256"] = self.context_hash
        self.checkpoint = dict(
            global_step=20000, epoch=11, state_dict=self.model.state_dict(),
            hyper_parameters=dict(config_tree=dict(model=copy.deepcopy(self.resolved["model"]),
                                                  run_provenance=provenance)),
            optimizer_states=[optimizer.state_dict()], lr_schedulers=[], data_context=context,
            inference_binding=dict(data_context_sha256=self.context_hash),
            PixelProprioDataModule=dict(schema="mot-sampling-state-v1", consumed_batches=20000,
                distributed_sampling=dict(schema="pixel-distributed-sampling-v1", world_size=4,
                    per_rank_batch_size=8, global_batch_size=32,
                    training_sharding="global_draw_rank_slots",
                    validation_sharding="contiguous_global_batches_no_padding")),
            mot_rng_state=dict(schema="mot_rng_state.distributed.v1", world_size=4,
                ranks=[dict(torch_cpu=torch.get_rng_state(), torch_cuda=[torch.get_rng_state()],
                            numpy=np.random.RandomState(rank).get_state(), python=random.Random(rank).getstate(),
                            cuda_scope="current_device") for rank in range(4)]),
            loops=dict(fit_loop={
                "epoch_loop.automatic_optimization.optim_progress": {
                    "optimizer": {"step": {"total": {"completed": 20000, "ready": 20000}}}},
                "epoch_loop.batch_progress": {"total": {
                    "completed": 19999, "processed": 20000, "ready": 20000, "started": 20000}},
                "epoch_loop.state_dict": {"_batches_that_stepped": 19999},
                "epoch_loop.val_loop.batch_progress": {"total": {"completed": 0}},
            }))

    def inspect(self, checkpoint=None):
        return target.validate_payload(self.checkpoint if checkpoint is None else checkpoint,
            self.parent, self.resolved, expected_context_sha256=self.context_hash,
            context_fingerprint=fingerprint)

    def reject(self, checkpoint=None, match=None):
        with self.assertRaisesRegex(ValueError, match or ".+"):
            self.inspect(checkpoint)

    def first_optimizer_state(self):
        return next(iter(self.checkpoint["optimizer_states"][0]["state"].values()))

    def test_actual_torch_save_mmap_reload_has_expected_state_digests(self):
        path = self.root/"synthetic.ckpt"
        torch.save(self.checkpoint, path)
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        result = self.inspect(loaded)
        self.assertEqual(result["global_step"], 20000)
        self.assertEqual(result["optimizer_steps"], [20000])
        self.assertEqual(result["optimizer_state_sha256"], target.state_digest(self.checkpoint["optimizer_states"][0]))
        self.assertEqual(len(result["rank_rng_sha256"]), 4)
        self.assertFalse(result["loop_state"]["loop_training_resume_supported"])

    def test_actual_saved_dictconfig_shape_is_supported(self):
        self.checkpoint["hyper_parameters"]["config_tree"] = OmegaConf.create(
            self.checkpoint["hyper_parameters"]["config_tree"])
        self.assertEqual(self.inspect()["global_step"], 20000)

    def test_no_global_rng_draw_or_mutation_during_validation(self):
        before = target.state_digest(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                                          python=random.getstate()))
        self.inspect()
        after = target.state_digest(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                                         python=random.getstate()))
        self.assertEqual(before, after)

    def test_state_digest_preserves_tensor_dtype_shape_and_container_type(self):
        self.assertNotEqual(target.state_digest(torch.ones(2)), target.state_digest(torch.ones(2, dtype=torch.float64)))
        self.assertNotEqual(target.state_digest(torch.ones(2)), target.state_digest(torch.ones(1, 2)))
        self.assertNotEqual(target.state_digest([1, 2]), target.state_digest((1, 2)))

    def test_nonfinite_or_nonfp32_wrapper_weights_rejected(self):
        for value in [torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.ones(3, dtype=torch.float16)]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["state_dict"]["weight"] = value
            self.reject(checkpoint, "stored wrapper weights")

    def test_missing_full_optimizer_rejected(self):
        self.checkpoint["optimizer_states"] = []
        self.reject(match="full AdamW")

    def test_optimizer_parameter_identity_must_be_complete_and_unique(self):
        for params in [[0], [0, 0], [0, 1, 9]]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["optimizer_states"][0]["param_groups"][0]["params"] = params
            self.reject(checkpoint, "optimizer parameter state")

    def test_nonfinite_nonfp32_negative_or_mismatched_moments_rejected(self):
        for key, value in [("exp_avg", torch.tensor([float("nan")])),
                           ("exp_avg", torch.ones(2, 3, dtype=torch.float16)),
                           ("exp_avg_sq", -torch.ones(2, 3)),
                           ("exp_avg_sq", torch.ones(1))]:
            checkpoint = copy.deepcopy(self.checkpoint)
            next(iter(checkpoint["optimizer_states"][0]["state"].values()))[key] = value
            self.reject(checkpoint, "AdamW moments")

    def test_optimizer_step_must_equal_global_step(self):
        self.first_optimizer_state()["step"] = torch.tensor(0.)
        self.reject(match="step disagrees")

    def test_lr_weight_decay_and_scheduler_are_exact(self):
        for key, value in [("lr", 5e-5), ("weight_decay", 0.)]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["optimizer_states"][0]["param_groups"][0][key] = value
            self.reject(checkpoint, "groups/LR")
        self.checkpoint["lr_schedulers"] = [{"last_epoch": 20000}]
        self.reject(match="LR schedule")

    def test_saved_model_config_and_provenance_match(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["hyper_parameters"]["config_tree"]["model"]["optimizer"]["lr"] = 5e-5
        self.reject(checkpoint, "model configuration")
        for key in ["run_id", "source_revision", "data_context_sha256"]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["hyper_parameters"]["config_tree"]["run_provenance"][key] = "wrong"
            self.reject(checkpoint, "provenance")

    def test_context_content_and_binding_both_match(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["data_context"]["normalizer"]["mean"][0] = .1
        self.reject(checkpoint, "data context")
        self.checkpoint["inference_binding"]["data_context_sha256"] = "c"*64
        self.reject(match="data context")

    def test_sampler_cursor_and_all_topology_fields_checked(self):
        self.checkpoint["PixelProprioDataModule"]["consumed_batches"] = 0
        self.reject(match="sampler cursor")
        self.checkpoint["PixelProprioDataModule"]["consumed_batches"] = 20000
        for key, value in [("world_size", 2), ("global_batch_size", 16),
                           ("per_rank_batch_size", 4), ("training_sharding", "wrong")]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["PixelProprioDataModule"]["distributed_sampling"][key] = value
            self.reject(checkpoint, "sampler cursor")

    def test_missing_rank_or_old_rng_schema_rejected(self):
        self.checkpoint["mot_rng_state"]["ranks"].pop()
        self.reject(match="distributed RNG")

    def test_cpu_cuda_numpy_python_rng_payloads_checked(self):
        for key, value in [("torch_cpu", torch.ones(3, dtype=torch.float32)),
                           ("torch_cpu", torch.ones(3, dtype=torch.uint8)),
                           ("torch_cuda", []), ("numpy", ("invalid",)),
                           ("python", (0, (), None)), ("cuda_scope", "all_devices")]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["mot_rng_state"]["ranks"][0][key] = value
            self.reject(checkpoint, "RNG")

    def test_no_smoke_or_invalid_production_step(self):
        for step in [True, 0, 50, 38863, 300000]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["global_step"] = step
            self.reject(checkpoint, "scheduled production")

    def test_missing_malformed_or_nonfinite_loops_rejected(self):
        for loops in [None, {}, {"fit_loop": {}}, {"fit_loop": {"bad": float("nan")}}]:
            checkpoint = copy.deepcopy(self.checkpoint)
            checkpoint["loops"] = loops
            self.reject(checkpoint)

    def test_optimizer_loop_counter_mismatch_rejected(self):
        self.checkpoint["loops"]["fit_loop"]["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]["total"]["completed"] = 0
        self.reject(match="loop counter")

    def test_unknown_batch_loop_structure_rejected(self):
        self.checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["completed"] = 1
        self.reject(match="batch counters")

    def test_boundary_state_is_recorded_without_inventing_replay_safety(self):
        result = self.inspect()["loop_state"]
        self.assertEqual(result["batch_completed"], 19999)
        self.assertFalse(result["loop_training_resume_supported"])
        self.assertEqual(result["resume_loop_policy"], "reject_replayed_boundary_validation")

    def add_checkpoint(self, epoch, step, content=b"checkpoint"):
        directory = Path(self.parent["output"])/"checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/f"epoch_epoch={epoch:04d}-step_step={step:09d}.ckpt"
        path.write_bytes(content)
        latest = directory/"last.ckpt"
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
        return directory, path

    def test_inventory_selects_numeric_highest_before_loading(self):
        directory, _ = self.add_checkpoint(9999, 20000)
        _, highest = self.add_checkpoint(1, 40000)
        selected, rows = target.inventory_highest(directory, self.parent)
        self.assertEqual(selected, highest)
        self.assertEqual([row["global_step"] for row in rows], [20000, 40000])

    def test_empty_newest_prevents_silent_fallback(self):
        directory, _ = self.add_checkpoint(1, 20000)
        self.add_checkpoint(2, 40000, b"")
        with self.assertRaisesRegex(ValueError, "Newest checkpoint is empty"):
            target.inventory_highest(directory, self.parent)

    def test_new_milestone_during_restore_invalidates_inspection(self):
        directory, _ = self.add_checkpoint(1, 20000)
        selected, inventory = target.inventory_highest(directory, self.parent)
        target.verify_inventory_stable(directory, self.parent, selected, inventory)
        self.add_checkpoint(2, 40000)
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            target.verify_inventory_stable(directory, self.parent, selected, inventory)

    def test_historical_file_changed_during_restore_invalidates_inspection(self):
        directory, first = self.add_checkpoint(1, 20000)
        self.add_checkpoint(2, 40000)
        selected, inventory = target.inventory_highest(directory, self.parent)
        first.write_bytes(b"changed old milestone")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            target.verify_inventory_stable(directory, self.parent, selected, inventory)

    def test_wrong_last_symlink_or_duplicate_step_rejected(self):
        directory, first = self.add_checkpoint(1, 20000)
        self.add_checkpoint(2, 40000)
        (directory/"last.ckpt").unlink()
        (directory/"last.ckpt").symlink_to(first.name)
        with self.assertRaisesRegex(ValueError, "last.ckpt"):
            target.inventory_highest(directory, self.parent)
        self.add_checkpoint(9, 20000)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            target.inventory_highest(directory, self.parent)

    def test_no_symlink_checkpoint_or_unbounded_inventory(self):
        directory, first = self.add_checkpoint(1, 20000)
        other = directory/"epoch_epoch=0002-step_step=000040000.ckpt"
        other.symlink_to(first.name)
        with self.assertRaisesRegex(ValueError, "immutable checkpoint"):
            target.inventory_highest(directory, self.parent)
        with self.assertRaisesRegex(ValueError, "bounded directory"):
            target.inventory_highest(directory.parent, self.parent)

    def save_json_binding(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value, sort_keys=True))
        return dict(path=str(path), sha256=target.sha(path))

    def crash_child(self, parent, checkpoint, number):
        binding = self.save_json_binding(f"request-{number}.json", parent)
        receipt = dict(schema=target.SCHEMA, status="PASS", parent_request_sha256=binding["sha256"],
                       run_id=parent["run_id"], source_revision=parent["source_revision"],
                       global_step=int(target.MILESTONE.fullmatch(checkpoint.name)[2]),
                       checkpoint=dict(path=str(checkpoint), sha256=target.sha(checkpoint)))
        child = copy.deepcopy(parent)
        child["output"] = str(self.root/f"recovery-{number}")
        child["crash_recovery"] = dict(schema="joint-crash-recovery-v1", parent_request=binding,
            inspection=self.save_json_binding(f"inspection-{number}.json", receipt))
        return child

    def test_failed_recovery_before40k_uses_retained_ancestor20k(self):
        _, original = self.add_checkpoint(1, 20000)
        child = self.crash_child(self.parent, original, 1)
        directory = Path(child["output"])/"checkpoints"
        self.assertFalse(directory.exists())
        selected, rows = target.inventory_highest(directory, child)
        self.assertEqual(selected, original)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expected_sha256"], target.sha(original))
        target.verify_inventory_stable(directory, child, selected, rows)

    def test_two_repeated_early_failures_still_use_original20k(self):
        _, original = self.add_checkpoint(1, 20000)
        first = self.crash_child(self.parent, original, 1)
        second = self.crash_child(first, original, 2)
        # One empty checkpoint directory and one absent directory are both valid
        # because neither failed recovery owns any20k milestone yet.
        (Path(first["output"])/"checkpoints").mkdir(parents=True)
        selected, rows = target.inventory_highest(Path(second["output"])/"checkpoints", second)
        self.assertEqual(selected, original)
        self.assertEqual([row["global_step"] for row in rows], [20000])

    def test_foreign_or_repeated_output_ancestor_rejected(self):
        _, original = self.add_checkpoint(1, 20000)
        child = self.crash_child(self.parent, original, 1)
        child["run_id"] = "foreign-run"
        with self.assertRaisesRegex(ValueError, "inspection belongs|foreign scientific"):
            target.inventory_highest(Path(child["output"])/"checkpoints", child)
        child = self.crash_child(self.parent, original, 2)
        child["output"] = self.parent["output"]
        with self.assertRaisesRegex(ValueError, "output/cycle"):
            target.inventory_highest(Path(child["output"])/"checkpoints", child)

    def test_ancestor_with_newer_unreconciled_checkpoint_rejected(self):
        _, original = self.add_checkpoint(1, 20000)
        child = self.crash_child(self.parent, original, 1)
        self.add_checkpoint(2, 40000)
        with self.assertRaisesRegex(ValueError, "newer-than-cutoff"):
            target.inventory_highest(Path(child["output"])/"checkpoints", child)

    def test_resume_binding_cannot_select_checkpoint_outside_lineage(self):
        _, original = self.add_checkpoint(1, 20000)
        child = self.crash_child(self.parent, original, 1)
        receipt = target.bound(child["crash_recovery"]["inspection"])
        receipt["checkpoint"]["path"] = str(self.root/"foreign"/original.name)
        child["crash_recovery"]["inspection"] = self.save_json_binding("foreign-inspection.json", receipt)
        with self.assertRaisesRegex(ValueError, "outside prior retained history"):
            target.inventory_highest(Path(child["output"])/"checkpoints", child)

    def test_missing_retained_earlier_milestone_cannot_be_hidden(self):
        directory, _ = self.add_checkpoint(2, 40000)
        with self.assertRaisesRegex(ValueError, "Missing retained production milestone"):
            target.inventory_highest(directory, self.parent)

    def test_native_allocation_exact_owner_node_job_and_restart(self):
        env = dict(SLURM_JOB_ID="123", SLURMD_NODENAME="gpu19", SLURM_RESTART_COUNT="0")
        native = "JobId=123 JobState=RUNNING UserId=paphiwetsa3(41003) NodeList=gpu19 NumNodes=1 Restarts=0"
        result = target.verify_allocation("123", "gpu19", env=env, uid=41003, native_reader=lambda: native)
        self.assertEqual(result["job_id"], "123")
        for bad in [native.replace("RUNNING", "COMPLETING"), native.replace("gpu19", "gpu03"),
                    native.replace("41003", "41004"), native.replace("Restarts=0", "Restarts=1")]:
            with self.assertRaisesRegex(ValueError, "Native inspector"):
                target.verify_allocation("123", "gpu19", env=env, uid=41003, native_reader=lambda: bad)
        with self.assertRaisesRegex(ValueError, "allocated compute"):
            target.verify_allocation("123", "gpu19", env={}, uid=41003, native_reader=lambda: native)

    def toy_actual_restore(self, checkpoint):
        """Exercise real torch strict load/AdamW; stub source integration only."""
        captured = self

        class Context:
            def fingerprint(self):
                return captured.context_hash

            def bind(self, model):
                pass

        class Data:
            def prepare_context(self, **kwargs):
                return Context()

            def load_state_dict(self, state):
                self.state = copy.deepcopy(state)

            def state_dict(self):
                return self.state

        class Wrapper(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.layer = torch.nn.Linear(3, 2)
                self.layer.expert_parameter_counts = lambda: {"image": 8}
                self.model = types.SimpleNamespace(pipeline=types.SimpleNamespace(
                    stage_by_id=lambda stage: types.SimpleNamespace(model=self.layer)))

            def state_dict(self, *args, **kwargs):
                return self.layer.state_dict(*args, **kwargs)

            def load_state_dict(self, state, strict=True):
                return self.layer.load_state_dict(state, strict=strict)

        def instantiate(cfg, **kwargs):
            if cfg is None:
                return None
            if cfg.get("_target_") == "test.Data":
                return Data()
            return torch.optim.AdamW(kwargs["params"], lr=cfg.lr, weight_decay=cfg.weight_decay)

        modules = {
            "egomimic.pipeline.checkpoint_binding": types.SimpleNamespace(
                validate_checkpoint_binding=lambda checkpoint, config, context: checkpoint["inference_binding"]),
            "egomimic.pipeline.construction": types.SimpleNamespace(checkpoint_construction=contextlib.nullcontext),
            "egomimic.pl_utils.pl_model": types.SimpleNamespace(ModelWrapper=Wrapper),
        }
        cfg = OmegaConf.create({**self.resolved, "data": {"_target_": "test.Data"}})
        parent = {**self.parent, "expert_parameters": {"image": 8}, "parameters": 8}
        with patch.dict(sys.modules, modules), patch("hydra.utils.instantiate", side_effect=instantiate):
            return target.strict_cpu_restore(checkpoint, cfg, parent, self.context_hash)

    def test_real_cpu_wrapper_and_adamw_restore_executes(self):
        before = target.state_digest(self.checkpoint)
        result = self.toy_actual_restore(self.checkpoint)
        self.assertTrue(result["strict_full_wrapper_optimizer_reload"])
        self.assertTrue(result["actual_datamodule_restore"])
        self.assertEqual(result["actual_expert_parameters"], {"image": 8})
        self.assertEqual(target.state_digest(self.checkpoint), before)

    def test_actual_strict_wrapper_key_or_adamw_parameter_shape_failure(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint["state_dict"].pop("bias")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.toy_actual_restore(checkpoint)
        checkpoint = copy.deepcopy(self.checkpoint)
        state = next(iter(checkpoint["optimizer_states"][0]["state"].values()))
        state["exp_avg"] = state["exp_avg_sq"] = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "moment shape"):
            self.toy_actual_restore(checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
