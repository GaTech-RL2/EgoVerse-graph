"""Native phase proof adapter around canonical exact argv and shared validators."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROFILE = "libero/action_flow_libero10_h240_euler50_dithalf_80k_s42"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def model_contract(cfg):
    import copy

    model = copy.deepcopy(cfg["model"])
    model.pop("gradient_telemetry_cadence", None)
    evaluator = copy.deepcopy(cfg["evaluator"])
    evaluator.pop("artifact_root", None)
    for name in ("artifact_identity", "native_diagnostic_config"):
        if isinstance(evaluator.get(name), dict):
            evaluator[name].pop("config_path", None)
            if name == "native_diagnostic_config":
                evaluator[name].pop("artifact_root", None)
    return dict(
        evaluator=evaluator,
        schema="libero-native-scientific-contract/v1",
        model=model,
        normalizer=cfg["normalizer"],
        binding=cfg["norm_stats"]["native_saved_state_binding"],
        train_datasets=cfg["data"]["train_datasets"],
        valid_datasets=cfg["data"]["valid_datasets"],
        train_batch=cfg["data"]["train_dataloader_params"]["libero_panda"][
            "batch_size"
        ],
        valid_batch=cfg["data"]["valid_dataloader_params"]["libero_panda"][
            "batch_size"
        ],
        precision=cfg["trainer"]["precision"],
        accumulate_grad_batches=cfg["trainer"]["accumulate_grad_batches"],
        seed=cfg["seed"],
        ema=cfg["callbacks"]["ema"],
        dit_half=cfg["callbacks"]["dit_half"],
    )


def contract_sha(cfg):
    return hashlib.sha256(
        json.dumps(
            model_contract(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def checked(path, expected):
    assert sha(path) == expected, "native evidence SHA mismatch"
    return json.loads(Path(path).read_text())


def runtime_equivalent(a, b):
    for key in (
        "python",
        "installed_distributions",
        "core_distribution_versions",
        "dependency_inputs",
        "source",
    ):
        assert a[key] == b[key], ("native runtime differs", key)
    ta = dict(a["torch_runtime"])
    tb = dict(b["torch_runtime"])
    ta.pop("cuda_available_on_capture_node", None)
    tb.pop("cuda_available_on_capture_node", None)
    assert ta == tb, "native torch runtime differs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--action", choices=["preflight", "prepare", "finalize"], required=True
    )
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--argv", type=Path)
    p.add_argument("--phase", choices=["preflight", "smoke", "full"])
    a = p.parse_args()
    assert os.environ.get("SLURM_STEP_ID"), "scheduled srun only"
    sys.path[:0] = [str(a.repo), str(a.repo / "tools")]
    from omegaconf import OmegaConf

    if a.action in ("preflight", "prepare"):
        assert a.argv and a.phase
        env = dict(os.environ)
        env["AF_NATIVE_CONSTRUCT"] = "true" if a.action == "preflight" else "false"
        env["AF_NATIVE_OPERATIONAL_VERIFIER_PENDING"] = "false"
        env["AF_NATIVE_PENDING_PROOFS"] = ""
        subprocess.run(
            [
                sys.executable,
                str(a.repo / "scripts/train/libero_native_schema_contract.py"),
                "--repo",
                str(a.repo),
                "--phase",
                a.phase,
                "--output",
                str(a.output),
                "--argv",
                str(a.argv),
                "--expected-argv-sha256",
                sha(a.argv),
            ],
            env=env,
            check=True,
            cwd=a.repo,
        )
        from egomimic.benchmarks.libero.native_launch_profiles import profile_for_config

        cfg = OmegaConf.to_container(
            OmegaConf.load(a.output / "resolved.yaml"), resolve=True
        )
        PROFILE = profile_for_config(cfg).experiment
        binding = cfg["norm_stats"]["native_saved_state_binding"]
        fingerprint = contract_sha(cfg)
        runtime = json.loads((a.output / "runtime-lock.json").read_text())
        if a.action == "preflight":
            result = json.loads((a.output / "RESULT.json").read_text())
            constructor = checked(
                result["CPU_constructor"]["path"], result["CPU_constructor"]["sha256"]
            )
            constructor["model_contract_sha256"] = fingerprint
            # A new immutable receipt references genuine constructor output; never relabel its config hash.
            constructor["constructor_original"] = result["CPU_constructor"]
            cp = a.output / "CPU_CONSTRUCTOR_CONTRACT.json"
            write(cp, constructor)
            preflight = dict(
                schema="libero-native-launch-preflight/v1",
                status="PASS",
                profile=PROFILE,
                source_commit=binding["source_commit"],
                resolved_config_sha256=sha(a.output / "resolved.yaml"),
                resolved_config_path=str((a.output / "resolved.yaml").resolve()),
                model_contract_sha256=fingerprint,
                normalization_sha256=binding["file_sha256"],
                split_manifest_sha256=binding["split_receipt_sha256"],
                dataset_sha256=cfg["run_provenance"]["dataset_sha256"],
                CPU_constructor=reference(cp),
                physical_proof=dict(
                    path=binding["physical_proof_path"],
                    sha256=binding["physical_proof_sha256"],
                ),
                runtime_lock=reference(a.output / "runtime-lock.json"),
                scope="native_preexecution_cpu_constructor",
                launch_ready=False,
            )
            write(a.output / "PREFLIGHT_RESULT.json", preflight)
        else:
            preflight = checked(
                os.environ["AF_PREFLIGHT_RESULT"],
                os.environ["AF_EXPECTED_PREFLIGHT_SHA256"],
            )
            assert (
                preflight["status"] == "PASS"
                and preflight["profile"] == PROFILE
                and preflight["model_contract_sha256"] == fingerprint
            ), "native preflight scientific identity mismatch"
            assert (
                preflight["source_commit"] == binding["source_commit"]
                and preflight["normalization_sha256"] == binding["file_sha256"]
                and preflight["split_manifest_sha256"]
                == binding["split_receipt_sha256"]
            )
            runtime_equivalent(
                checked(
                    preflight["runtime_lock"]["path"],
                    preflight["runtime_lock"]["sha256"],
                ),
                runtime,
            )
            if a.phase == "full":
                smoke = checked(
                    os.environ["AF_SMOKE_RESULT"],
                    os.environ["AF_EXPECTED_SMOKE_SHA256"],
                )
                assert (
                    smoke["status"] == "PASS"
                    and smoke["profile"] == PROFILE
                    and smoke["model_contract_sha256"] == fingerprint
                ), "native full run lacks exact genuine smoke"
                assert (
                    smoke["preflight"]["sha256"]
                    == os.environ["AF_EXPECTED_PREFLIGHT_SHA256"]
                )
                assert (
                    smoke["checkpoint"]["parameter_count"] == 39750391
                    and smoke["strict_online_and_ema_reload"] is True
                    and smoke["native_artifacts"]["same_pass_verified"] is True
                )
            write(
                a.output / "NATIVE_RUN_PREPARED.json",
                dict(
                    status="PASS",
                    phase=a.phase,
                    model_contract_sha256=fingerprint,
                    resolved_config_sha256=sha(a.output / "resolved.yaml"),
                    preflight=reference(os.environ["AF_PREFLIGHT_RESULT"]),
                ),
            )
    else:
        root = a.output
        from egomimic.benchmarks.libero.saved_hydra_context import (
            load_saved_native_config,
        )

        cfg = OmegaConf.to_container(
            load_saved_native_config(root, a.repo), resolve=True
        )
        preflight = checked(
            os.environ["AF_PREFLIGHT_RESULT"],
            os.environ["AF_EXPECTED_PREFLIGHT_SHA256"],
        )
        assert contract_sha(cfg) == preflight["model_contract_sha256"]
        artifact = (
            root / "validation_predictions/native_action_flow/valid-step-000002.json"
        )
        assert artifact.is_file()
        evidence = dict(
            schema="libero-native-postsmoke-evidence/v1",
            preflight=reference(os.environ["AF_PREFLIGHT_RESULT"]),
            native_metrics_artifact=reference(artifact),
            resolved_config_sha256=sha(root / ".hydra/config.yaml"),
            model_contract_sha256=contract_sha(cfg),
        )
        write(root / "provenance/NATIVE_SMOKE_EVIDENCE.json", evidence)


if __name__ == "__main__":
    main()
