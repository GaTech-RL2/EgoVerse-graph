"""Real CPU Torch object/callback guard tests, not CUDA/DDP certification."""

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch


from scripts.train.joint.crash_restore import (
    CrashRestoreProof, assert_actual_restore, equal_state, validate_rank_proofs,
    verify_restored_state, weights_digest,
)


def capture():
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=[], numpy=np.random.get_state(),
                python=random.getstate(), cuda_scope="current_device")


def fixture():
    network = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=.01)
    network(torch.ones(1, 2)).square().mean().backward(); optimizer.step(); optimizer.zero_grad()
    for value in optimizer.state.values():
        value["step"] = torch.tensor(20000.)
    sampling = dict(schema="mot-sampling-state-v1", consumed_batches=20000,
                    distributed_sampling=dict(world_size=4, global_batch_size=32))
    rng = dict(schema="mot_rng_state.distributed.v1", world_size=4,
               ranks=[capture() for _ in range(4)])
    module = SimpleNamespace(state_dict=network.state_dict,
                             training_behavior=SimpleNamespace(_pending_rng=rng))
    data = SimpleNamespace(state_dict=lambda: sampling, consumed_batches=20000)
    trainer = SimpleNamespace(world_size=4, precision="bf16-mixed", global_step=20000,
                              optimizers=[optimizer], lr_scheduler_configs=[], datamodule=data)
    saved = copy.deepcopy(dict(global_step=20000, state_dict=network.state_dict(),
                               optimizer_states=[optimizer.state_dict()], lr_schedulers=[],
                               PixelProprioDataModule=sampling, mot_rng_state=rng))
    return trainer, module, saved, network


class LoadedStateTests(unittest.TestCase):
    def test_exact_real_torch_model_and_adamw_state(self):
        trainer, module, saved, _ = fixture()
        self.assertIs(verify_restored_state(trainer, module, saved, start_step=20000,
                                           world_size=4), trainer.optimizers[0])

    def test_rejects_unrestored_or_wrong_dtype_weights(self):
        trainer, module, saved, network = fixture()
        with torch.no_grad():
            network.weight.add_(1.)
        with self.assertRaisesRegex(ValueError, "Weights"):
            verify_restored_state(trainer, module, saved, start_step=20000, world_size=4)
        trainer, module, saved, network = fixture(); network.double()
        with self.assertRaisesRegex(ValueError, "Weights"):
            verify_restored_state(trainer, module, saved, start_step=20000, world_size=4)

    def test_rejects_changed_optimizer_moments_lr_and_step(self):
        for field in ("exp_avg", "step", "lr"):
            with self.subTest(field=field):
                trainer, module, saved, _ = fixture()
                if field == "lr": trainer.optimizers[0].param_groups[0]["lr"] = 5e-5
                else: next(iter(trainer.optimizers[0].state.values()))[field].add_(1.)
                with self.assertRaisesRegex(ValueError, "AdamW"):
                    verify_restored_state(trainer, module, saved, start_step=20000, world_size=4)

    def test_rejects_matching_but_nonfinite_adamw_moments(self):
        trainer, module, saved, _ = fixture()
        next(iter(trainer.optimizers[0].state.values()))["exp_avg"].fill_(float("inf"))
        saved["optimizer_states"][0] = copy.deepcopy(trainer.optimizers[0].state_dict())
        with self.assertRaisesRegex(ValueError, "finiteFP32"):
            verify_restored_state(trainer, module, saved, start_step=20000, world_size=4)

    def test_rejects_scheduler_cursor_rng_or_world_reset(self):
        for field in ("scheduler", "cursor", "rng", "world"):
            with self.subTest(field=field):
                trainer, module, saved, _ = fixture()
                if field == "scheduler": trainer.lr_scheduler_configs = [object()]
                elif field == "cursor": trainer.datamodule.consumed_batches = 0
                elif field == "rng": module.training_behavior._pending_rng = None
                else: trainer.world_size = 2
                with self.assertRaises(ValueError):
                    verify_restored_state(trainer, module, saved, start_step=20000, world_size=4)

    def test_cpu_cannot_claim_four_rank_callback_start(self):
        callback = CrashRestoreProof(checkpoint="none", inspection_path="none",
                                     inspection_sha256="x", output="none", world_size=4,
                                     start_step=20000)
        with self.assertRaisesRegex(ValueError, "Actual four-rank CUDA"):
            callback.on_train_start(None, None)

    def test_no_implicit_diagnostic_validation_cadence(self):
        with self.assertRaisesRegex(ValueError, "explicit two-update"):
            CrashRestoreProof(checkpoint="none", inspection_path="none", inspection_sha256="x",
                               output="none", world_size=4, start_step=20000,
                               validation_interval=2)
        callback = CrashRestoreProof(checkpoint="none", inspection_path="none", inspection_sha256="x",
                                     output="none", world_size=4, start_step=20000,
                                     diagnostic=True, validation_interval=2)
        self.assertTrue(callback.diagnostic)

    def test_rejects_replayed_validation_at_the_checkpoint_step(self):
        callback = CrashRestoreProof(checkpoint="none", inspection_path="none", inspection_sha256="x",
                                     output="none", world_size=4, start_step=20000)
        with self.assertRaisesRegex(ValueError, "replayed"):
            callback.on_validation_start(SimpleNamespace(sanity_checking=False, global_step=20000), None)

    def test_weights_digest_handles_scalar_buffers_and_dtypes(self):
        first = weights_digest(dict(value=torch.tensor(1., dtype=torch.float32)))
        self.assertNotEqual(first, weights_digest(dict(value=torch.tensor(1., dtype=torch.float64))))

    def test_actual_save_hook_rejects_boundary_offcadence_and_duplicate_save(self):
        callback = CrashRestoreProof(checkpoint="none", inspection_path="none", inspection_sha256="x",
                                     output="none", world_size=4, start_step=20000)
        for step in (20000, 20001, 20002, 39999):
            with self.subTest(step=step), self.assertRaises(ValueError):
                callback.on_save_checkpoint(SimpleNamespace(global_step=step), None, dict(global_step=step))
        callback.on_save_checkpoint(SimpleNamespace(global_step=40000), None, dict(global_step=40000))
        self.assertEqual(callback.observed_saves, [40000])
        with self.assertRaisesRegex(ValueError, "repeated a checkpoint"):
            callback.on_save_checkpoint(SimpleNamespace(global_step=40000), None, dict(global_step=40000))

    def test_actual_save_hook_checks_serialized_step_and_diagnostic_terminal(self):
        callback = CrashRestoreProof(checkpoint="none", inspection_path="none", inspection_sha256="x",
                                     output="none", world_size=4, start_step=20000,
                                     diagnostic=True, validation_interval=2)
        with self.assertRaisesRegex(ValueError, "boundary checkpoint"):
            callback.on_save_checkpoint(SimpleNamespace(global_step=20002), None, dict(global_step=20000))
        with self.assertRaisesRegex(ValueError, "off-cadence"):
            callback.on_save_checkpoint(SimpleNamespace(global_step=20004), None, dict(global_step=20004))
        callback.on_save_checkpoint(SimpleNamespace(global_step=20002), None, dict(global_step=20002))
        self.assertEqual(callback.observed_saves, [20002])


class RNGObserverTests(unittest.TestCase):
    def setUp(self):
        self.previous = capture()

    def tearDown(self):
        self.restore(self.previous)

    @staticmethod
    def restore(state):
        torch.set_rng_state(state["torch_cpu"]); np.random.set_state(state["numpy"])
        random.setstate(state["python"])

    def callback(self):
        return CrashRestoreProof(checkpoint="none", inspection_path="none", inspection_sha256="x",
                                 output="none", world_size=4, start_step=20000)

    def test_observes_actual_cpu_numpy_python_rng_restore_without_consuming_rng(self):
        torch.manual_seed(42); np.random.seed(42); random.seed(42); expected = capture()
        torch.rand(2); np.random.random(); random.random()
        implementation = SimpleNamespace(__name__="egomimic.pl_utils.world_model_behavior",
                                          _restore_rng_state=self.restore,
                                          _capture_rng_state=lambda **kw: capture())
        original = implementation._restore_rng_state
        callback = self.callback()
        with patch("scripts.train.joint.crash_restore.importlib.import_module", return_value=implementation):
            callback._install_rng_observer(SimpleNamespace(training_behavior=object()), expected)
            implementation._restore_rng_state(expected)
        self.assertTrue(callback.rng_restore_observed)
        self.assertTrue(equal_state(capture(), expected))
        self.assertIs(implementation._restore_rng_state, original)

    def test_rejects_broken_restore_and_restores_observer_on_exception(self):
        expected = capture(); torch.rand(1)
        implementation = SimpleNamespace(__name__="egomimic.pl_utils.world_model_behavior",
                                          _restore_rng_state=lambda state: None,
                                          _capture_rng_state=lambda **kw: capture())
        original = implementation._restore_rng_state; callback = self.callback()
        with patch("scripts.train.joint.crash_restore.importlib.import_module", return_value=implementation):
            callback._install_rng_observer(SimpleNamespace(training_behavior=object()), expected)
            with self.assertRaisesRegex(ValueError, "not exactly restored"):
                implementation._restore_rng_state(expected)
        callback.on_exception(None, None, ValueError())
        self.assertIs(implementation._restore_rng_state, original)

    def test_rejects_another_saved_rank_rng(self):
        expected = capture(); torch.rand(1); wrong = capture()
        implementation = SimpleNamespace(__name__="egomimic.pl_utils.world_model_behavior",
                                          _restore_rng_state=self.restore,
                                          _capture_rng_state=lambda **kw: capture())
        callback = self.callback()
        with patch("scripts.train.joint.crash_restore.importlib.import_module", return_value=implementation):
            callback._install_rng_observer(SimpleNamespace(training_behavior=object()), expected)
            with self.assertRaisesRegex(ValueError, "another saved rank"):
                implementation._restore_rng_state(wrong)
        callback.on_exception(None, None, ValueError())


class RankProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.inspection = dict(schema="joint-crash-checkpoint-inspection-v1", status="PASS",
                               global_step=20000, world_size=4, run_id="same-run",
                               source_revision="source", checkpoint=dict(path="/actual-checkpoint", sha256="f"*64))
        self.inspection_sha = "e"*64
        self.paths = []
        for rank in range(4):
            row = dict(schema="joint-crash-restore-rank-v1", status="PASS", rank=rank,
                       world_size=4, start_step=20000, global_step=20002, actual_updates=2,
                       inspection_sha256=self.inspection_sha, checkpoint="/actual-checkpoint",
                       checkpoint_sha256="f"*64,
                       run_id="same-run", source_revision="source", diagnostic=False,
                       job_id="123", gpu_uuid="GPU-"+str(uuid.UUID(int=rank+1)),
                       strict_weights_optimizer_sampler_rng=True, actual_native_rank_rng_restored=True,
                       fp32_losses=2, finite_bf16_outputs=2, finite_fp32_gradients=2,
                       optimizer_update_lrs=[1e-4]*2, no_extra_save=True, no_replayed_validation=True,
                       save_cadence_guard_installed=True, observed_saves_through_proof=[],
                       initial_weights_sha256="a"*64, initial_all_rank_weights=["a"*64]*4,
                       weights_sha256="b"*64, all_rank_weights=["b"*64]*4)
            path = self.root / f"restore-step000020000-rank{rank}.json"
            path.write_text(json.dumps(row)); self.paths.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, **kwargs):
        return validate_rank_proofs(self.root, inspection=self.inspection,
                                    inspection_sha256=self.inspection_sha,
                                    start_step=20000, expected_job_id="123", **kwargs)

    def mutate(self, rank, **values):
        path = self.paths[rank]; row = json.loads(path.read_text()); row.update(values)
        path.write_text(json.dumps(row))

    def test_complete_counterfactual_shape_is_accepted_only_as_unit_fixture(self):
        self.assertEqual(len(self.check()), 4)

    def test_missing_rank_wrong_job_uuid_or_rng_proof_fails(self):
        mutations = [dict(job_id="124"), dict(gpu_uuid="GPU-"+str(uuid.UUID(int=1))),
                     dict(actual_native_rank_rng_restored=False), dict(optimizer_update_lrs=[5e-5]*2),
                     dict(weights_sha256="c"*64), dict(actual_updates=3, global_step=20003)]
        original = self.paths[1].read_text()
        for values in mutations:
            with self.subTest(values=values):
                self.paths[1].write_text(original); self.mutate(1, **values)
                with self.assertRaises(ValueError): self.check()
        self.paths[1].unlink()
        with self.assertRaisesRegex(ValueError, "Missing immutable"):
            self.check()

    def test_diagnostic_flag_cannot_substitute_for_production_restore(self):
        self.mutate(0, diagnostic=True)
        with self.assertRaisesRegex(ValueError, "current inspected recovery"):
            self.check()

    def test_diagnostic_requires_actual_postfit_validation_proof(self):
        for rank in range(4): self.mutate(rank, diagnostic=True)
        with self.assertRaises(FileNotFoundError): self.check(diagnostic=True)
        for rank in range(4):
            (self.root / f"restore-validation-rank{rank}.json").write_text(json.dumps(dict(
                schema="joint-crash-restore-validation-rank-v1", status="PASS", rank=rank,
                world_size=4, job_id="123", start_step=20000, global_step=20002,
                validation_steps=[20002], global_validation_batches=100,
                local_validation_batches=25, inspection_sha256=self.inspection_sha,
                observed_saves=[20002],
                actual_scheduled_validation_after_optimizer=True, metrics={"image_mse": .01})))
        self.assertEqual(len(self.check(diagnostic=True)), 4)
        path = self.root / "restore-validation-rank3.json"
        row = json.loads(path.read_text()); row["global_validation_batches"] = 99
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "scheduled recovery validation"):
            self.check(diagnostic=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
