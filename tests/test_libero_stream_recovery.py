import hashlib
import io
import json

import pytest

from egomimic.benchmarks.libero.arc_streams import campaign, candidates, source_receipt
from scripts.benchmarks.launch_libero_stream_recovery import recovery_workflow
from scripts.benchmarks.resume_libero_stream import (
    restore_resume,
    validate_resume_payload,
)
from scripts.benchmarks.wait_libero_stream_reference import reference_ready
from tests.test_libero_stream_campaign import experiment

COMMIT = "a" * 40


def request(
    run="old-component-time-dur",
    suite="libero_spatial",
    variant="component_time",
    mode="dur",
):
    sha = "b" * 64
    return {
        "source_run": run,
        "source_commit": COMMIT,
        "suite": suite,
        "method": f"arc_{mode}",
        "arc_stream_variant": variant,
        "epochs": 5001,
        "total_optimizer_steps": campaign()["optimizer_steps"][suite],
        "checkpoint": {
            "sha256": sha,
            "bytes": 100,
            "uri": f"s3://rldb/experiments/arc-oat-20260919/{run}/checkpoints/{sha}/last.ckpt",
        },
    }


@pytest.fixture
def partial():
    _, tree = experiment("component_time", "dur")
    return {
        "hyper_parameters": {"config_tree": tree},
        "training_budget": {
            "epochs": 5001,
            "total_optimizer_steps": 270054,
            "global_batch_size": 1024,
            "world_size": 4,
            "microbatch_size": 256,
            "gradient_accumulation": 1,
            "optimizer_steps_per_epoch": 54,
        },
        "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 10}}}},
        "global_step": 540,
        "ema_num_updates": 540,
        "ema_state_dict": {"weight": 1},
        "optimizer_states": [{"state": {"weight": 1}}],
        "normalizer_state": {},
        "benchmark_data_context": {"suite": "libero_spatial"},
    }


def test_partial_resume_preserves_existing_training_and_ema_counts(partial):
    result = validate_resume_payload(partial, request())
    assert result["global_step"] == result["ema_num_updates"] == 540
    assert result["epochs_completed"] == 10
    assert result["training_budget"]["total_optimizer_steps"] == 270054


def test_periodic_checkpoint_retains_lightning_epoch_boundary_state(partial):
    loop = partial["loops"]["fit_loop"]
    loop["epoch_progress"]["current"] = {"completed": 9, "ready": 10, "started": 10}
    loop["epoch_loop.batch_progress"] = {
        "is_last_batch": True,
        "current": {"completed": 54},
    }
    assert validate_resume_payload(partial, request())["epochs_completed"] == 10
    loop["epoch_loop.batch_progress"]["is_last_batch"] = False
    with pytest.raises(ValueError, match="last batch"):
        validate_resume_payload(partial, request())


@pytest.mark.parametrize(
    "problem",
    [
        "batch",
        "gpu_layout",
        "budget",
        "ema",
        "optimizer",
        "normalizer",
        "complete",
        "mid_epoch",
        "suite",
        "representation",
    ],
)
def test_partial_resume_rejects_incompatible_state(partial, problem):
    if problem == "batch":
        partial["training_budget"]["global_batch_size"] = 512
    elif problem == "gpu_layout":
        partial["training_budget"]["world_size"] = 8
    elif problem == "budget":
        partial["training_budget"]["total_optimizer_steps"] = 100
    elif problem == "ema":
        partial["ema_num_updates"] -= 1
    elif problem == "optimizer":
        partial["optimizer_states"] = []
    elif problem == "normalizer":
        del partial["normalizer_state"]
    elif problem == "complete":
        partial["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] = 5001
    elif problem == "mid_epoch":
        partial["global_step"] += 1
    elif problem == "suite":
        partial["benchmark_data_context"]["suite"] = "libero_goal"
    else:
        partial["hyper_parameters"]["config_tree"]["model"]["benchmark_protocol"][
            "arc_mode"
        ] = "stk"
    with pytest.raises(ValueError):
        validate_resume_payload(partial, request())


class Storage:
    def __init__(self, objects, bad_hash=False):
        self.objects = objects
        self.bad_hash = bad_hash

    def get_object(self, *, Bucket, Key):
        raw = json.dumps(self.objects[Key.rsplit("/", 1)[-1]]).encode()
        return {
            "Body": io.BytesIO(raw),
            "Metadata": {
                "sha256": "bad" if self.bad_hash else hashlib.sha256(raw).hexdigest()
            },
        }

    def download_file(self, *args):
        raise AssertionError("Invalid provenance must be rejected before download")


def test_restore_refuses_wrong_source_and_unverified_metadata(tmp_path):
    client = Storage({"runtime.json": {}}, bad_hash=True)
    with pytest.raises(ValueError, match="original training source"):
        restore_resume(client, request(), tmp_path, "c" * 40)
    with pytest.raises(ValueError, match="checksum"):
        restore_resume(client, request(), tmp_path, COMMIT)
    client.bad_hash = False
    with pytest.raises(ValueError, match="runtime"):
        restore_resume(client, request(), tmp_path, COMMIT)


@pytest.mark.parametrize("suite", campaign()["suites"])
def test_recovery_resumes_partial_cells_and_preserves_completed_references(
    tmp_path, suite
):
    complete = suite in ("libero_spatial", "libero_goal")
    rows = []
    for variant in campaign()["variants"]:
        for mode in ("stk", "dur"):
            run = f"old-{variant.replace('_', '-')}-{mode}"
            rows.append(
                {
                    "suite": suite,
                    "variant": variant,
                    "mode": mode,
                    "training_run_id": run,
                }
            )
            directory = tmp_path / run
            directory.mkdir()
            (directory / "runtime.json").write_text(
                json.dumps({"source_commit": COMMIT})
            )
            (directory / "status.json").write_text(
                json.dumps(
                    {
                        "state": "TRAINING_COMPLETE"
                        if complete and (variant, mode) == ("reference", "stk")
                        else "TRAINING"
                    }
                )
            )
            receipt = request(run, suite, variant, mode)["checkpoint"]
            (directory / "checkpoint-receipts.json").write_text(
                json.dumps({f"training/arc_{mode}/checkpoints/last.ckpt": receipt})
            )
    proof = {
        "source_commit": COMMIT,
        "suite": suite,
        "sources": source_receipt(),
        "candidates": candidates(),
        "controls_passed": True,
        "episodes": 30,
    }
    spec = recovery_workflow(
        COMMIT,
        "recovery-test",
        suite,
        proof,
        {"runs": rows},
        tmp_path,
        completed_evaluation="new-reference-stk-eval" if complete else None,
    )["workflow"]
    tasks = {t["name"]: t for t in spec["tasks"]}
    resume_tasks = [
        t
        for t in tasks.values()
        if any(f["path"] == "/tmp/resume-request.json" for f in t["files"])
    ]
    assert len(resume_tasks) == (3 if complete else 4)
    for task in resume_tasks:
        assert task.get("inputs", []) == []
        env = task["environment"]
        pinned = json.loads(
            next(
                f["contents"]
                for f in task["files"]
                if f["path"] == "/tmp/resume-request.json"
            )
        )
        assert pinned["source_commit"] == env["SOURCE_COMMIT"] == COMMIT
        assert pinned["method"] == f"arc_{env['ARC_STREAM_MODE']}"
        assert pinned["arc_stream_variant"] == env["ARC_STREAM_VARIANT"]
    assert sum(
        t["environment"].get("RUN_KIND") == "arc_stream_train"
        for t in tasks.values()
        if "environment" in t
    ) == (21 if complete else 22)
    if complete:
        assert (
            tasks["reference-stk-ready"]["image"]
            == "docker.io/library/python:3.11-slim"
        )
        assert (
            "--training-run old-reference-stk"
            in tasks["reference-stk-ready"]["args"][1]
        )
        assert "reference-train" not in tasks
    else:
        assert tasks["reference-stk-ready"]["resource"] == "evaluation"
        assert tasks["reference-stk-ready"]["inputs"] == [{"task": "reference-train"}]
    assert {"task": "reference-stk-ready"} in tasks["evaluate-01"]["inputs"]


def test_recovered_reference_evaluation_pins_original_training_run():
    client = Storage(
        {
            "status.json": {
                "state": "POLICY_EVALUATION_COMPLETE",
                "episodes": 2500,
                "complete_protocol": True,
            },
            "runtime.json": {
                "source_commit": COMMIT,
                "suite": "libero_spatial",
                "method": "arc_stk",
                "run_kind": "policy_evaluation",
                "mode": "full",
                "evaluate_from_run": "old-reference-stk",
            },
        }
    )
    args = (client, "new-reference-stk-eval", "libero_spatial", COMMIT)
    with pytest.raises(ValueError, match="protocol"):
        reference_ready(*args)
    assert (
        reference_ready(*args, training_run="old-reference-stk")["reference_run"]
        == args[1]
    )


def test_real_trainer_resume_continues_optimizer_ema_and_epoch_counts(tmp_path):
    import torch
    from hydra import compose, initialize_config_dir

    from egomimic.benchmarks.libero.arc_streams import ROOT, stream_training_arguments
    from egomimic.trainHydra import train
    from tests.test_libero_benchmark import make_replay

    torch.set_num_threads(1)
    data = tmp_path / "data.zarr"
    make_replay(data)
    argv = stream_training_arguments(
        "libero_10",
        data,
        tmp_path,
        "smoke",
        variant="component_time",
        arc_mode="dur",
        gpus=1,
    )
    overrides = [
        v
        for v in argv[3:]
        if not v.startswith(("benchmark.batch_size=", "trainer.precision="))
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=overrides
            + [
                "benchmark.batch_size=2",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                f"norm_stats.save_cache_dir={tmp_path}",
                f"paths.work_dir={tmp_path}",
            ],
        )
    cfg.model.pipeline.stages[3].policy.model.down_dims = [16, 32]
    cfg.model.pipeline.stages[3].policy.model.diffusion_step_embed_dim = 16
    cfg.model.pipeline.stages[3].policy.num_inference_steps = 2
    _, first = train(cfg)
    assert first["trainer"].global_step == 2
    checkpoint = tmp_path / "training/arc_dur/checkpoints/epoch_epoch=0.ckpt"
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert before["ema_num_updates"] == 2
    assert before["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 0
    # Checkpoint remains a distinct input while resumed output goes to a new directory.
    restored = tmp_path / "recovered.ckpt"
    import shutil

    shutil.copyfile(checkpoint, restored)
    cfg.ckpt_path = str(restored)
    cfg.trainer.max_epochs = 2
    cfg.paths.output_dir = str(tmp_path / "resumed")
    _, resumed = train(cfg)
    assert resumed["trainer"].global_step == 4
    after = torch.load(
        tmp_path / "resumed/checkpoints/last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert after["ema_num_updates"] == 4
    assert after["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 2
    assert all(
        int(s["step"]) == 4 for s in after["optimizer_states"][0]["state"].values()
    )
    assert after["benchmark_data_context"] == before["benchmark_data_context"]

    def same(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for k in left:
                same(left[k], right[k])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                same(a, b)
        elif isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            import numpy as np

            np.testing.assert_equal(left, right)

    same(after["normalizer_state"], before["normalizer_state"])
    assert (
        hashlib.sha256(restored.read_bytes()).hexdigest()
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
