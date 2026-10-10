"""Execute the complete real Bash launcher through safe native tool handoff.

No model, data, Python tool, GPU, scheduler, credential or training execution.
Every preceding guard/hash/path/clean-Git check and actual array remains real.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

R = Path(__file__).parents[1]
PROFILE = "libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42"
PLANAR = "pusht/action_flow_bc_usocket_recon1_s42"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class WholeLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        files = [
            "scripts/train/launch_action_flow_usocket.sbatch",
            "scripts/train/libero_native_launch_phases.sh",
            "scripts/train/libero_native_schema_contract.py",
            "scripts/train/verify_action_flow_training_smoke.py",
            "scripts/train/validate_slurm_job_contract.py",
            "scripts/train/check_checkpoint_storage.py",
            "scripts/ice/ice_gpu_probe.py",
            "scripts/ice/ice_requeue_runner.py",
            "scripts/ice/relay_batch_signals.sh",
            "scripts/ice/validate_lightning_checkpoint.py",
            "scripts/ice/capture_runtime_lock.py",
            "tools/libero_maintained_dispatch_v1.py",
            "tools/validate_action_flow_config.py",
            "scripts/ice/validate_planar_dataset.py",
            "egomimic/hydra_configs/experiment/" + PROFILE + ".yaml",
            "egomimic/hydra_configs/experiment/libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_s42.yaml",
            "egomimic/hydra_configs/experiment/libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_s42.yaml",
            "egomimic/hydra_configs/experiment/libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_s42.yaml",
        ]
        from egomimic.benchmarks.libero.native_launch_profiles import AV0_PROFILES

        files += [
            "egomimic/hydra_configs/experiment/" + p.experiment + ".yaml"
            for p in AV0_PROFILES.values()
        ]
        for rel in files:
            p = self.repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(R / rel, p)
        self.launcher = self.repo / files[0]
        for name in ["data", "output-parent"]:
            (self.root / name).mkdir()
        self.split = self.root / "split.json"
        self.split.write_text("{}\n")
        self.receipt = self.root / "receipt.json"
        self.receipt.write_text("{}\n")
        self.norm = self.root / "norm.json"
        self.norm.write_text("{}\n")
        self.preflight = self.root / "preflight.json"
        self.preflight.write_text("{}\n")
        self.override = self.root / "binding.override"
        self.override.write_text(
            '++norm_stats.native_saved_state_binding={test:"fixture-only-no-execution"}\n'
        )
        self.mock = self.repo / "safe-tools/mock-python"
        self.mock.parent.mkdir()
        self.mock.write_text("""#!/bin/bash
set -Eeuo pipefail
case "$1" in
 */capture_runtime_lock.py)
  while test "$#" -gt 0; do
   if test "$1" = --output; then shift; printf '{"scope":"mocked-runtime-no-python-execution"}\\n' > "$1"; break; fi
   shift
  done ;;
 */libero_native_launch_contract.py)
  printf 'NATIVE_DISPATCH_PREPARE_STOP\\n'
  printf '%s\\n' "$*" > "$SAFE_HANDOFF_LOG"
  replay_variable=${SAFE_REPLAY_VARIABLE:-LIBERO_SPATIAL_REPLAY_ROOT}
  printf '%s\\n' "${!replay_variable:-}" > "$SAFE_HANDOFF_LOG.replay"
  exit 91 ;;
 *) printf 'UNEXPECTED_EXTERNAL_TOOL\\n' >&2; exit 92 ;;
esac
""")
        self.mock.chmod(0o755)
        cp = self.repo / "scripts/ice/validate_lightning_checkpoint.py"
        cp.chmod(0o755)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "safe launcher fixture",
            ],
            check=True,
        )
        head = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("AF_", "ICE_", "SLURM_", "WANDB_", "LIBERO_"))
        }
        env.update(
            AF_REPO=str(self.repo),
            AF_EXPECTED_HEAD=head,
            AF_PYTHON=str(self.mock),
            AF_DATASET_DIR=str(self.root / "data"),
            AF_OUTPUT_DIR=str(self.root / "output-parent/run"),
            AF_EXPERIMENT=PROFILE,
            AF_WANDB_ENTITY="rl2-group",
            AF_WANDB_PROJECT="pushshapes-action-flow",
            AF_WANDB_RUN_ID="lib10-noaug-smoke-ce2c-20261007",
            AF_WANDB_GROUP="cotrain-velocity-first-wave-20261007",
            AF_WANDB_TAGS="[libero10,noaug,action-flow,h240,dithalf,bf16,euler50,smoke]",
            AF_EXPECTED_SPLIT_MANIFEST_SHA256=sha(self.split),
            AF_EXPECTED_LAUNCHER_SHA256=sha(self.launcher),
            AF_CONTENT_MANIFEST=str(self.receipt),
            AF_EXPECTED_CONTENT_MANIFEST_SHA256=sha(self.receipt),
            AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256="3ed2b86cf97eaa27045af81e6d6c4834c8659d15eda69a2394d835e602014e1d",
            AF_NATIVE_DATASET_SHA256="3ed2b86cf97eaa27045af81e6d6c4834c8659d15eda69a2394d835e602014e1d",
            AF_NATIVE_HISTORICAL_SPLIT_SHA256="c588c831ef0deab61a59a21840ea9e11dafe92a170d9023a87e0acee831df236",
            AF_CLUSTER_LABEL="lambda",
            AF_EXPECTED_ACCOUNT="loaner",
            AF_EXPECTED_PARTITION="rl2",
            AF_EXPECTED_QOS="loaner-normal",
            AF_SPLIT_MANIFEST=str(self.split),
            AF_NORM_STATS_PATH=str(self.norm),
            AF_EXPECTED_NORM_SHA256=sha(self.norm),
            AF_NATIVE_SCHEMA_DRIVER=str(
                self.repo / "scripts/train/libero_native_schema_contract.py"
            ),
            AF_NATIVE_SCHEMA_DRIVER_SHA256=sha(
                self.repo / "scripts/train/libero_native_schema_contract.py"
            ),
            AF_NATIVE_BINDING_OVERRIDE=str(self.override),
            AF_NATIVE_BINDING_OVERRIDE_SHA256=sha(self.override),
            AF_NATIVE_DATASET_VALIDATOR=str(
                self.repo / "tools/libero_maintained_dispatch_v1.py"
            ),
            AF_NATIVE_CONFIG_VALIDATOR=str(
                self.repo / "tools/libero_maintained_dispatch_v1.py"
            ),
            AF_NATIVE_SMOKE_VERIFIER=str(
                self.repo / "scripts/train/verify_action_flow_training_smoke.py"
            ),
            AF_NATIVE_PHASE="smoke",
            AF_PRECISION="bf16-mixed",
            AF_LAUNCH_MODE="run",
            AF_RUN_KIND="smoke",
            AF_MAX_RESTARTS="0",
            AF_EXPECTED_GPU_CONSTRAINT="(null)",
            AF_EXPECTED_MEMORY="128G",
            AF_EXPECTED_TIME_LIMIT="02:00:00",
            AF_PREFLIGHT_RESULT=str(self.preflight),
            AF_EXPECTED_PREFLIGHT_SHA256=sha(self.preflight),
            AF_NATIVE_SCHEMA_ALLOW_PENDING_PROOFS="false",
            SLURM_STEP_ID="fixture-step",
            SLURM_JOB_ID="123456",
            SLURM_JOB_ACCOUNT="loaner",
            SLURM_JOB_PARTITION="rl2",
            SLURM_JOB_QOS="loaner-normal",
            SLURM_JOB_NUM_NODES="1",
            SLURM_NTASKS="1",
            SLURM_CPUS_PER_TASK="8",
            SLURM_RESTART_COUNT="0",
            SAFE_HANDOFF_LOG=str(self.root / "handoff.log"),
        )
        self.env = env

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, **changes):
        env = dict(self.env)
        env.update(changes)
        return subprocess.run(
            ["bash", str(self.launcher)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def assertRejected(self, result, text):
        self.assertEqual(result.returncode, 64, result.stdout + result.stderr)
        self.assertIn(text, result.stderr)
        self.assertFalse((self.root / "handoff.log").exists())

    def test_exact_v7_native_smoke_envelope_reaches_real_native_dispatch(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
        self.assertIn("NATIVE_DISPATCH_PREPARE_STOP", result.stdout)
        self.assertIn("--action prepare", (self.root / "handoff.log").read_text())
        argv = (
            self.root / "output-parent/run/provenance/restart-0/exact-phase.argv0"
        ).read_bytes()
        self.assertIn(b"trainer.max_steps=2\0", argv)
        self.assertIn(b"trainer.precision=bf16-mixed\0", argv)
        self.assertIn(b"++run_provenance.preflight_result_sha256=", argv)

    def suite_values(self, suite, seed):
        record = json.loads(
            (R / "tests/fixtures/libero_native_suite_corpus_authority.json").read_text()
        )["suites"][suite]
        return dict(
            AF_EXPERIMENT="libero_historical/action_flow_"
            + suite
            + "_h240_euler50_dithalf_80k_s42",
            AF_SEED=str(seed),
            AF_NATIVE_DATASET_SHA256=record["dataset_logical_sha256"],
            AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256=record[
                "dataset_logical_sha256"
            ],
            AF_NATIVE_HISTORICAL_SPLIT_SHA256=record["historical_split_sha256"],
            AF_EXPECTED_GPU_CONSTRAINT="H100|H200"
            if suite == "libero_spatial"
            else "(null)",
            AF_CLUSTER_LABEL="ice" if suite == "libero_spatial" else "lambda",
        )

    def test_actual_goal_object_corpus_and_both_training_seeds_reach_dispatch(self):
        for suite in ("libero_goal", "libero_object"):
            for seed in (42, 43):
                with self.subTest(suite=suite, seed=seed):
                    values = self.suite_values(suite, seed)
                    values["AF_OUTPUT_DIR"] = str(
                        self.root / "output-parent" / (suite + "-" + str(seed))
                    )
                    result = self.invoke(**values)
                    self.assertEqual(
                        result.returncode, 91, result.stdout + result.stderr
                    )
                    argv = (
                        Path(values["AF_OUTPUT_DIR"])
                        / "provenance/restart-0/exact-phase.argv0"
                    ).read_bytes()
                    self.assertIn(("seed=" + str(seed) + "\0").encode(), argv)
                    self.assertIn(
                        (
                            "++run_provenance.dataset_content_aggregate_sha256="
                            + values["AF_NATIVE_DATASET_SHA256"]
                            + "\0"
                        ).encode(),
                        argv,
                    )
                    self.assertIn(
                        (
                            "++run_provenance.historical_split_manifest_sha256="
                            + values["AF_NATIVE_HISTORICAL_SPLIT_SHA256"]
                            + "\0"
                        ).encode(),
                        argv,
                    )

    def test_cross_suite_corpus_and_historical_split_rejected(self):
        for suite in ("libero_spatial", "libero_goal", "libero_object"):
            for field, message in [
                ("AF_NATIVE_DATASET_SHA256", "native dataset identity mismatch"),
                (
                    "AF_NATIVE_HISTORICAL_SPLIT_SHA256",
                    "historical split identity mismatch",
                ),
                (
                    "AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256",
                    "native dataset aggregate identity mismatch",
                ),
            ]:
                with self.subTest(suite=suite, field=field):
                    values = self.suite_values(suite, 42)
                    values[field] = self.env[field]
                    self.assertRejected(self.invoke(**values), message)

    def test_spatial_ice_actual_whole_path_constrained_dispatch(self):
        result = self.invoke(**self.suite_values("libero_spatial", 42))
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
        argv = (
            self.root / "output-parent/run/provenance/restart-0/exact-phase.argv0"
        ).read_bytes()
        self.assertIn(
            b"+experiment=libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_s42\0",
            argv,
        )
        self.assertEqual(
            (self.root / "handoff.log.replay").read_text().strip(),
            str(self.root / "data"),
        )

    def test_spatial_native_actual_whole_path_null_reaches_hardware_guard(self):
        result = self.invoke(
            **{
                **self.suite_values("libero_spatial", 42),
                "AF_EXPECTED_GPU_CONSTRAINT": "(null)",
            }
        )
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)

    def test_native_full_positive_restart_cap_and_null_constraint_reach_dispatch(self):
        result = self.invoke(
            AF_RUN_KIND="full",
            AF_MAX_RESTARTS="4",
            AF_SMOKE_RESULT=str(self.preflight),
            AF_EXPECTED_SMOKE_SHA256=sha(self.preflight),
        )
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
        self.assertIn("--phase full", (self.root / "handoff.log").read_text())
        argv = (
            self.root / "output-parent/run/provenance/restart-0/exact-phase.argv0"
        ).read_bytes()
        self.assertIn(b"trainer.max_steps=80000\0", argv)
        self.assertIn(b"trainer.val_check_interval=15000\0", argv)
        self.assertIn(b"++callbacks.model_checkpoint.every_n_train_steps=5000\0", argv)

    def test_native_unsupported_gpu_constraint_rejected(self):
        self.assertRejected(
            self.invoke(AF_EXPECTED_GPU_CONSTRAINT="RTX4090"),
            "AF_EXPECTED_GPU_CONSTRAINT must be H100",
        )

    def test_native_invalid_restart_counter_rejected(self):
        self.assertRejected(
            self.invoke(SLURM_RESTART_COUNT="bad"), "invalid SLURM_RESTART_COUNT"
        )

    def test_native_smoke_requeue_cap_must_be_zero(self):
        self.assertRejected(
            self.invoke(AF_MAX_RESTARTS="1"), "native smoke requires AF_MAX_RESTARTS=0"
        )

    def test_negative_restart_cap_rejected(self):
        self.assertRejected(
            self.invoke(AF_MAX_RESTARTS="-1"), "native smoke requires AF_MAX_RESTARTS=0"
        )

    def test_native_restart_attempt_rejected(self):
        self.assertRejected(
            self.invoke(SLURM_RESTART_COUNT="1"), "restart limit exceeded"
        )

    def test_native_h100_constraint_reaches_hardware_guard(self):
        result = self.invoke(AF_EXPECTED_GPU_CONSTRAINT="H100")
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)

    def test_native_wrong_precision_rejected(self):
        self.assertRejected(
            self.invoke(AF_PRECISION="bf16"), "native BF16-mixed required"
        )

    def test_native_wrong_cpu_envelope_rejected_before_tool(self):
        self.assertRejected(
            self.invoke(SLURM_CPUS_PER_TASK="16"), "exact eight CPUs required"
        )

    def planar(self, **changes):
        params = dict(
            AF_EXPERIMENT=PLANAR,
            AF_PRECISION="bf16",
            AF_MAX_RESTARTS="4",
            AF_EXPECTED_CONTENT_MANIFEST_SHA256="a1c81fb0ce8967aba795383a293180f9ba08a0ecfdd6f4a878afb20b39733761",
            AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256="80f835ad37c3d5c5b7b2d5c3e1656c307ee567a1f63f51081165bf404b8ceb52",
        )
        params.update(changes)
        return self.invoke(**params)

    def test_av0_suites_reach_same_canonical_phase_with_distinct_identity(self):
        from egomimic.benchmarks.libero.native_launch_profiles import AV0_PROFILES

        for suite, profile in AV0_PROFILES.items():
            with self.subTest(suite=suite):
                values = self.suite_values(suite, 42)
                values["AF_EXPERIMENT"] = profile.experiment
                values["SAFE_REPLAY_VARIABLE"] = profile.replay_environment
                output = self.root / "output-parent" / ("run-" + suite)
                values["AF_OUTPUT_DIR"] = str(output)
                result = self.invoke(**values)
                self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
                argv = (output / "provenance/restart-0/exact-phase.argv0").read_bytes()
                self.assertIn(
                    ("+experiment=" + profile.experiment + "\0").encode(), argv
                )
                self.assertEqual(
                    (self.root / "handoff.log.replay").read_text().strip(),
                    str(self.root / "data"),
                )

    def test_general_zero_restart_gate_preserved(self):
        self.assertRejected(self.planar(AF_MAX_RESTARTS="0"), "invalid AF_MAX_RESTARTS")

    def test_general_null_gpu_constraint_gate_preserved(self):
        self.assertRejected(self.planar(), "AF_EXPECTED_GPU_CONSTRAINT must be H100")

    def test_native_full_zero_restart_gate_unchanged(self):
        self.assertRejected(self.invoke(AF_RUN_KIND="full"), "invalid AF_MAX_RESTARTS")


if __name__ == "__main__":
    unittest.main()
