"""Real config/receipt rejection tests; no allocation or runtime PASS is claimed."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from omegaconf import OmegaConf

smoke = importlib.import_module("scripts.train.joint.crash_smoke")


def binding(path, value, text=False):
    path.write_text(value if text else json.dumps(value))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class CrashSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.parent = dict(
            schema="joint-training-v2", phase="full", source_revision=smoke.SOURCE,
            run_id="mot-shared-libero-image-discrete-lr1e-4-constant-s42-300k-20261021-v1",
            active_streams=["image"], arm="discrete", parameters=88178688,
            expert_parameters=dict(image=88178688), expert_widths=[768],
            world_size=4, global_batch=32, per_rank_batch=8, target_steps=300000,
            lr_policy=dict(schedule="constant", peak_lr=1e-4, warmup_optimizer_steps=0),
            source=str(self.base/"source"), source_manifest=str(self.base/"manifest"),
            source_manifest_sha256="a"*64, source_archive=str(self.base/"source.tar"),
            source_archive_sha256="b"*64, data_manifest=str(self.base/"data"),
            data_manifest_sha256="c"*64, data_cache=dict(root="/tmp/data"),
            lambda_runtime=dict(node="gpu19"), runtime_localization=None, allocation=None,
            output=str(self.base/"production"), local_logs=str(self.base/"oldlogs"))
        self.parent["job_name"] = "joint-image-lr1e4-full-4gpu"
        self.q = copy.deepcopy(self.parent)
        self.q.update(schema=smoke.SCHEMA, output=str(self.base/"diagnostic"),
                      local_logs=str(self.base/"newlogs"), reserved_checkpoint_bytes=4_000_000_000,
                      job_name="joint-image-lr1e4-crash-diagnostic")
        self.q["parent_request"] = binding(self.base/"parent.json", self.parent)
        self.q["resolved"] = binding(self.base/"resolved.yaml", "model: {}\n", text=True)
        self.checkpoint = self.base/"input.ckpt"
        self.checkpoint.write_bytes(b"retained production bytes")
        self.inspection = dict(schema="joint-crash-checkpoint-inspection-v1", status="PASS",
            parent_request_sha256=self.q["parent_request"]["sha256"],
            parent_resolved_sha256=self.q["resolved"]["sha256"], source_revision=smoke.SOURCE,
            run_id=self.parent["run_id"], world_size=4, global_batch=32, global_step=20000,
            strict_full_wrapper_optimizer_reload=True, sampler_all_rank_rng_verified=True,
            checkpoint=dict(path=str(self.checkpoint), sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()))
        self.q["inspection"] = binding(self.base/"inspection.json", self.inspection)
        manifest = binding(self.base/"FILES.sha256", "test content", text=True)
        self.q["operational_service"] = dict(root=str(self.base), manifest=manifest["path"],
                                             manifest_sha256=manifest["sha256"])
        self.allocation = patch("scripts.train.joint.crash_inspect.verify_allocation", return_value={})
        self.allocation.start()
        self.restore = patch("scripts.train.joint.crash_restore.assert_actual_restore", return_value=[])
        self.restore.start()

    def tearDown(self):
        self.allocation.stop()
        self.restore.stop()
        self.temp.cleanup()

    def test_request_exact_bindings_and_no_input_write(self):
        before = self.checkpoint.read_bytes()
        parent, inspected = smoke.validate_request(self.q)
        self.assertEqual(parent, self.parent)
        self.assertEqual(inspected["global_step"], 20000)
        self.assertEqual(self.checkpoint.read_bytes(), before)
        self.assertFalse(Path(self.q["output"]).exists())

    def test_reject_changed_identity_and_runtime(self):
        for key, bad in (("run_id", "other"), ("active_streams", ["image", "proprio"]),
                         ("world_size", 2), ("lambda_runtime", {"node": "gpu21"})):
            q = copy.deepcopy(self.q);q[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_request(q)

    def test_reject_checkpoint_or_receipt_tamper(self):
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaises(ValueError): smoke.validate_request(self.q)

    def test_reject_production_path_overlap(self):
        for path in (self.parent["output"], self.parent["output"]+"/child", str(self.base)):
            q = copy.deepcopy(self.q);q["output"] = path
            with self.subTest(path=path), self.assertRaises(ValueError): smoke.validate_request(q)

    def test_config_unchanged_model_data_and_local_only_logging(self):
        original = OmegaConf.create(dict(
            model=dict(optimizer=dict(lr=1e-4, weight_decay=.01), scheduler=None,
                       pipeline=dict(science="unchanged")),
            data=dict(batch_size=8, distributed_world_size=4, science="same draws"),
            seed=42, runtime=dict(disable_tf32=True), joint=dict(max_tokens=3216),
            world_model_attention=dict(backend="fa4"), paths=dict(output_dir=self.parent["output"]),
            val_at_start=False, val_at_end=False,
            trainer=dict(devices=4,precision="bf16-mixed",max_steps=300000,val_check_interval=20000),
            logger=dict(wandb=dict(id=self.parent["run_id"], resume="never"),csv=dict()),
            evaluator=dict(pixel_diagnostics=dict(log_wandb=True,artifact_dir="old")),
            callbacks=dict(model_checkpoint=dict(), allocation_boundary=dict(), joint_smoke=dict())))
        before = OmegaConf.to_container(original, resolve=True)
        cfg = smoke.phase_config(original, self.q, self.parent, self.inspection,
                                 request_sha256="d"*64)
        self.assertEqual(before, OmegaConf.to_container(original, resolve=True))
        self.assertEqual(cfg.model, original.model)
        self.assertEqual(cfg.data, original.data)
        self.assertEqual(set(cfg.logger), {"csv"})
        self.assertEqual(cfg.trainer.max_steps, 20002)
        self.assertEqual(cfg.trainer.val_check_interval, 2)
        self.assertFalse(cfg.evaluator.pixel_diagnostics.log_wandb)
        self.assertTrue(cfg.callbacks.crash_restore_proof.diagnostic)
        self.assertEqual(cfg.callbacks.model_checkpoint.save_top_k, -1)
        self.assertFalse(Path(self.q["output"]).exists())

    def _finalization_fixture(self, state="COMPLETED", exit_code="0:0", host="0", job="123"):
        qpath = self.base/"request.json";qpath.write_text(json.dumps(self.q))
        output = Path(self.q["output"]);output.mkdir(exist_ok=True)
        candidate = dict(schema="joint-crash-recovery-smoke-candidate-v1",
            status="ACTUAL_DIAGNOSTIC_PASS_NATIVE_PENDING", request_sha256=smoke.sha(qpath),
            job_id="123", finite_optimizer_updates=2, all_rank_full_state_restored=True,
            source_revision=self.parent["source_revision"],parent_request_sha256=self.q["parent_request"]["sha256"],
            inspection_sha256=self.q["inspection"]["sha256"],checkpoint_sha256=self.inspection["checkpoint"]["sha256"],
            service_manifest_sha256=self.q["operational_service"]["manifest_sha256"],world_size=4,global_batch=32,
            scheduled_validation_anchors=100,no_replayed_boundary_validation_or_save=True,
            production_science_unchanged=True,parent_checkpoint_modified=False,production_wandb_written=False,
            sample_finite=True,hidden_query_labels_nan=True,sample_steps=20)
        proofs=[]
        for rank in range(4):
            for relative in (f"restore-proof/restore-step000020000-rank{rank}.json",
                             f"restore-proof/restore-validation-rank{rank}.json",
                             f"validation-proof/rank{rank}.json"):
                path=output/relative;path.parent.mkdir(exist_ok=True)
                row=dict(status="PASS",job_id="123",world_size=4,rank=rank,
                         start_step=20000,global_step=20002)
                if relative.startswith("validation-proof"):
                    row.update(schema="joint-crash-diagnostic-validation-rank-v1",
                        scheduled_validation_steps=[20002],scheduled_validation_anchors=100,
                        local_validation_anchors=25,no_replayed_boundary_validation=True)
                proofs.append(binding(path,row))
        logs=Path(self.q["local_logs"])/"csv/step000020000";logs.mkdir(parents=True,exist_ok=True)
        (logs/"metrics.csv").write_text("step,val/libero_gripper/forecast/image_normalized_mse,val/libero_gripper/completion/image_normalized_mse\n20001,0.01,0.02\n")
        candidate["rank_proofs"]=proofs
        final=output/"checkpoints/diagnostic-step=000020002.ckpt";final.parent.mkdir(exist_ok=True)
        final.write_bytes(b"synthetic full state")
        candidate["final_diagnostic_checkpoint"]={"path":str(final),"sha256":smoke.sha(final),
                                                  "strict_full_wrapper_optimizer_reload":True}
        (output/"crash-recovery-smoke.candidate.json").write_text(json.dumps(candidate))
        native = binding(self.base/"native.txt", f"JobId={job} JobState={state} ExitCode={exit_code} "
            f"UserId=paphiwetsa3(41003) NodeList=gpu19 JobName={self.q['job_name']} Requeue=0 Restarts=0", text=True)
        hostbind = binding(self.base/"host.txt", host, text=True)
        return qpath, native, hostbind

    def test_native_running_is_not_diagnostic_pass(self):
        path, native, host = self._finalization_fixture(state="RUNNING")
        with self.assertRaises(ValueError):
            smoke.finalize(path, self.q, native_binding=native,host_binding=host,job_id="123")
        self.assertFalse((Path(self.q["output"])/"crash-recovery-smoke.json").exists())

    def test_host_nonzero_and_wrong_job_cannot_finalize(self):
        for opts in (dict(host="1"),dict(job="124"),dict(exit_code="1:0")):
            with self.subTest(opts=opts):
                path,native,host=self._finalization_fixture(**opts)
                with self.assertRaises(ValueError):
                    smoke.finalize(path,self.q,native_binding=native,host_binding=host,job_id="123")

    def test_finalization_is_exclusive_and_binds_native(self):
        path,native,host=self._finalization_fixture()
        record=smoke.finalize(path,self.q,native_binding=native,host_binding=host,job_id="123")
        self.assertEqual(record["status"],"PASS")
        self.assertEqual(record["native"],native)
        with self.assertRaises(FileExistsError):
            smoke.finalize(path,self.q,native_binding=native,host_binding=host,job_id="123")

    def test_parent_current_state_and_duplicate_writer_guards(self):
        native=binding(self.base/"parent-native.txt",f"JobId=122 JobState=CANCELLED "
            f"UserId=paphiwetsa3(41003) JobName={self.parent['job_name']}",text=True)
        self.q["failure"]=binding(self.base/"failure.json",dict(schema="joint-crash-failure-v1",
            parent_request_sha256=self.q["parent_request"]["sha256"],run_id=self.parent["run_id"],
            recovery_authorized=True,human_retired=False,job_id=122,native=native))
        with patch.dict(smoke.os.environ,SLURM_JOB_ID="123"):
            with patch.object(smoke.subprocess,"run",return_value=SimpleNamespace(
                    returncode=0,stdout="JobState=COMPLETING",stderr="")):
                with self.assertRaises(ValueError):smoke.assert_parent_terminal(self.q,self.parent)
            results=[SimpleNamespace(returncode=0,stdout="JobState=CANCELLED",stderr=""),
                     SimpleNamespace(returncode=0,stdout="124|joint-image-lr1e4-other|PENDING\n",stderr="")]
            with patch.object(smoke.subprocess,"run",side_effect=results):
                with self.assertRaises(ValueError):smoke.assert_parent_terminal(self.q,self.parent)
            results=[SimpleNamespace(returncode=0,stdout="JobState=CANCELLED",stderr=""),
                     SimpleNamespace(returncode=0,stdout="123|joint-image-lr1e4-this|RUNNING\n",stderr="")]
            with patch.object(smoke.subprocess,"run",side_effect=results):
                smoke.assert_parent_terminal(self.q,self.parent)

    def test_finalization_rejects_changed_rank_proof(self):
        path,native,host=self._finalization_fixture()
        proof=Path(self.q["output"])/"restore-proof/restore-step000020000-rank0.json"
        proof.write_text("{}")
        with self.assertRaises(ValueError):
            smoke.finalize(path,self.q,native_binding=native,host_binding=host,job_id="123")


if __name__ == "__main__":
    unittest.main()
