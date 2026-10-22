"""Admit two fresh native image/proprio temporal-representation trials.

This v1 contract is deliberately narrow: frozen535d, constant1e-4, four ranks,
300k updates and all15 scheduled checkpoints. It validates both real GPU smokes;
a PASS string alone is never an admission. Different D/C training draws and
whole-model initial hashes are retained, not presented as paired controls.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from scripts.train.joint.contract import (
    bound, compare_configs, need, sha, verify_evidence, verify_model_execution,
    write_once,
)

SCHEMA = "native-image-proprio-temporal-pair-v1"
SOURCE = "535d4fc26c4c85cf57bbd2a776cc79cfec085c4e"
SOURCE_MANIFEST = "f29118056a2f7824b5215e503bed99b77264b1598bfe154ba20c5412be106184"
SOURCE_ARCHIVE = "f23b041bb2e8f250e406cbc24ef7be06b463221e8a004125d175f006661d8325"
DATA_MANIFEST = "8c40d792a5cedd4105ff8270ff3141767b147513c52f11292f888d9b1527bfe2"
MODES = ("temporal_comparison", "optimizer_comparison", "modality_comparison",
         "factorial_controls")
PROFILES = {
    "discrete": dict(strategy="discrete_window", pose=36, tokens=3216,
                     parameters=102135076, proprio_parameters=13956388),
    "continuous": dict(strategy="full_episode_uniform", pose=9, tokens=12864,
                       parameters=102121225, proprio_parameters=13942537),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def canonical_request(request):
    """The receipt binding is the only field excluded from the request identity."""
    value = copy.deepcopy(request)
    value.pop("temporal_comparison", None)
    return digest(value)


def admission_mode(request):
    modes = [name for name in MODES if request.get(name) is not None]
    need(len(modes) == 1, "Exactly one production comparison contract required")
    return modes[0]


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def evidence_binding(value):
    need(isinstance(value, dict) and set(value) == {"path", "sha256"}
         and isinstance(value["path"], str) and Path(value["path"]).is_absolute()
         and valid_hash(value["sha256"]), "Invalid immutable evidence binding")
    return value


def request_shape(request):
    need(request.get("schema") == "joint-training-v2" and request.get("phase") == "full"
         and request.get("arm") in PROFILES, "Temporal pair requires D/C full requests")
    need(not any(k.startswith("_") for k in request), "Request still contains preparation fields")
    need(not any(name in request for name in MODES), "Prototype already has an admission contract")
    need(not any(name in request for name in (
        "continuation", "allocation_stop_step", "allocation_boundaries", "allocation_recovery")),
        "Temporal v1 requires fresh single-allocation requests")
    need(request.get("source_revision") == SOURCE
         and request.get("source_manifest_sha256") == SOURCE_MANIFEST
         and request.get("source_archive_sha256") == SOURCE_ARCHIVE
         and request.get("data_manifest_sha256") == DATA_MANIFEST,
         "Temporal v1 source/data identity changed")
    need(request.get("active_streams") == ["image", "proprio"]
         and request.get("experiment") == "image_proprio"
         and request.get("action_contract") is None,
         "Temporal pair supports image/proprio only")
    profile = PROFILES[request["arm"]]
    need(request.get("parameters") == profile["parameters"]
         and request.get("expert_parameters") == {
             "image": 88178688, "proprio": profile["proprio_parameters"]}
         and request.get("expert_widths") == [768, 256]
         and request.get("vector_expert_parameter_budget") == 20000000,
         "Temporal expert architecture changed")
    need(request.get("world_size") == 4 and request.get("per_rank_batch") == 8
         and request.get("global_batch") == 32 and request.get("target_steps") == 300000
         and request.get("lr_policy") == {
             "peak_lr": 1e-4, "schedule": "constant", "warmup_optimizer_steps": 0},
         "Temporal topology, budget or LR recipe changed")
    for name in ("source", "source_manifest", "source_archive", "data_manifest", "output", "local_logs"):
        need(isinstance(request.get(name), str) and Path(request[name]).is_absolute(),
             "Absolute bound path required: " + name)
    need(isinstance(request.get("run_id"), str) and bool(request["run_id"]), "Missing run identity")
    need(request.get("intended_full_run_id", request["run_id"]) == request["run_id"]
         and request.get("intended_full_output", request["output"]) == request["output"]
         and request.get("intended_full_target_steps", 300000) == 300000,
         "Full request contradicts intended identity")
    return profile



def verify_overrides(request):
    """The narrow v1 recipe permits no additional scientific override on both arms."""
    arm, p = request["arm"], PROFILES[request["arm"]]
    expected = [
        f"+experiment=pixel_proprio/native_joint/{arm}_image_proprio", "hydra/launcher=basic",
        "data.manifest_path=" + request["data_manifest"], "paths.work_dir=" + request["source"],
        "paths.output_dir=" + request["output"], "hydra.run.dir=" + request["output"],
        "hydra.sweep.dir=" + request["output"], "hydra.job.chdir=false",
        "+trainer.enable_progress_bar=false", "+trainer.enable_model_summary=false",
        "logger=[wandb,csv]", "logger.wandb.id=" + request["run_id"], "logger.wandb.resume=never",
        "logger.csv.name=csv", "logger.csv.flush_logs_every_n_steps=1",
        "logger.csv.save_dir=" + request["local_logs"], "evaluator.pixel_diagnostics.log_wandb=true",
        "modality_ablation.screening_budget_updates=300000", "trainer.devices=4",
        "data.distributed_world_size=4", "data.batch_size=8", "model.optimizer.lr=1e-4",
        "model.scheduler=null", "data.mode=" + arm, "data.sampling_strategy=" + p["strategy"],
        "data.min_frames=2", "data.max_frames=64", "data.discrete_frames=16", "data.stride=4",
        "joint.max_tokens=" + str(p["tokens"]), "joint.pose_dims.libero_gripper=" + str(p["pose"]),
        "trainer.max_steps=300000", "callbacks.model_checkpoint.save_last=link",
    ]
    actual = request.get("overrides")
    need(isinstance(actual, list) and all(isinstance(v, str) for v in actual)
         and sorted(actual) == sorted(expected), "Additional or changed temporal v1 override")


def compose_request(request):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path(request["source"])/"egomimic/hydra_configs"),
                               version_base=None):
        cfg = compose(config_name="train_zarr_cartesian", overrides=request["overrides"])
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def normalize_config(request, cfg):
    """Check every allowed substitution at its exact executed/resolved path."""
    profile, arm = PROFILES[request["arm"]], request["arm"]
    cfg = copy.deepcopy(cfg)
    need(verify_model_execution(request, cfg) == "visual_diffusion", "Wrong native graph")
    need(cfg["seed"] == 42 and cfg["data"]["seed"] == 42
         and cfg["model"]["optimizer"]["lr"] == 1e-4
         and cfg["model"]["optimizer"]["weight_decay"] == 0.01
         and cfg["model"]["scheduler"] is None and not cfg["model"]["enable_grad_norm"]
         and cfg["ckpt_path"] is None
         and not cfg["model"]["pipeline"].get("initialization")
         and not cfg["model"]["pipeline"].get("trainability"), "Wrong fresh optimizer recipe")
    need(cfg["runtime"]["slurm_requeue_owner"] == "none" and cfg["runtime"]["disable_tf32"]
         and cfg["trainer"]["precision"] == "bf16-mixed"
         and cfg["trainer"]["devices"] == cfg["data"]["distributed_world_size"] == 4
         and cfg["data"]["batch_size"] == 8 and cfg["trainer"]["accumulate_grad_batches"] == 1,
         "Wrong precision/topology/runtime")
    need(set(cfg["callbacks"]) == {"model_checkpoint", "learning_rate_monitor", "wandb_profiler"}
         and cfg["trainer"]["limit_val_batches"] == 1.0
         and cfg["trainer"]["limit_train_batches"] == 1.0
         and cfg["val_at_end"] is True,
         "Unexpected checkpoint callback or partial/final-validation policy")
    data = cfg["data"]
    need(data["mode"] == arm and data["sampling_strategy"] == profile["strategy"]
         and data["discrete_frames"] == 16 and data["stride"] == 4
         and data["min_frames"] == 2 and data["max_frames"] == 64
         and data["max_past_seconds"] is None and data["max_future_seconds"] is None
         and data["checkpointable_sampling"] is True and data["couple_state_observations"] is True
         and data["image_only"] is False and data["manifest_path"] == request["data_manifest"],
         "Unexpected temporal sampling contract")
    need(data["validation_sampling_strategy"] == "discrete_window"
         and data["validation_discrete_frames"] == 16 and data["validation_stride"] == 4
         and data["validation_anchors_per_episode"] == 2
         and data["validation_batch_size"] == 1
         and data["validation_batches_per_embodiment"] is None
         and cfg["evaluator"]["objective"] == "visual_proprio"
         and cfg["evaluator"]["sample_seed"] == 42,
         "Scheduled held-out validation changed")
    data["mode"], data["sampling_strategy"] = "<temporal>", "<temporal>"
    need(cfg["name"] == f"native_mot_{arm}_image_proprio_s42_300k", "Unbound recipe name")
    cfg["name"] = "<temporal>"
    need(cfg["joint"]["pose_dims"] == {"libero_gripper": profile["pose"]}
         and cfg["joint"]["max_tokens"] == profile["tokens"]
         and cfg["pixel_video"]["max_tokens"] == profile["tokens"], "Wrong temporal dimensions")
    cfg["joint"]["pose_dims"], cfg["joint"]["max_tokens"] = "<pose>", "<tokens>"
    cfg["pixel_video"]["max_tokens"] = "<tokens>"
    ids = cfg["model"]["pipeline"]["stage_ids"]
    need(ids == {"joint_noising": 0, "joint_denoiser": 1, "image_loss": 2,
                 "proprio_loss": 3, "joint_sampler": 4}
         and cfg["modality_ablation"]["stage_ids"] == ids,
         "Unexpected native stage graph")
    models = [cfg["model"]["pipeline"]["stages"][1]["model"],
              cfg["modality_ablation"]["nodes"]["joint_denoiser"]["model"],
              cfg["modality_ablation"]["graph"][1]["model"]]
    image_models = [cfg["native_joint"]["reference"]["pipeline"]["stages"][1]["model"]]
    for model in models:
        need(model["pose_dims"] == {"libero_gripper": profile["pose"]}
             and model["active_streams"] == ["image", "proprio"]
             and model["proprio_width"] == 256, "Wrong resolved proprio model")
        model["pose_dims"] = "<pose>"
        image_models.append(model["image_model"])
    for model in image_models:
        need(model["time_mode"] == arm and model["max_tokens"] == profile["tokens"],
             "Unexpected image temporal representation")
        model["time_mode"], model["max_tokens"] = "<temporal>", "<tokens>"
    need(cfg["paths"]["output_dir"] == request["output"]
         and cfg["logger"]["csv"]["save_dir"] == request["local_logs"]
         and cfg["logger"]["wandb"]["id"] == request["run_id"], "Unbound output/log identity")
    cfg["logger"]["wandb"]["id"] = "<run_id>"
    def paths(value):
        if isinstance(value, dict):
            return {key: paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        if isinstance(value, str):
            for old, replacement in ((request["output"], "<output>"),
                                     (request["local_logs"], "<local_logs>")):
                if value == old or value.startswith(old + "/"):
                    return replacement + value[len(old):]
        return value
    return paths(cfg)


def disjoint(paths):
    normalized = [Path(value) for value in paths]
    need(all(p.is_absolute() and ".." not in p.parts for p in normalized), "Unsafe artifact path")
    for i, left in enumerate(normalized):
        need(all(left != right and left not in right.parents and right not in left.parents
                 for right in normalized[i+1:]), "Overlapping experiment output/log paths")


def verify(prototypes):
    """Return reproducible evidence from two immutable full-request prototypes."""
    need(isinstance(prototypes, list) and len(prototypes) == 2, "Exactly two prototypes required")
    pairs = [(evidence_binding(binding), bound(binding)) for binding in prototypes]
    need({request.get("arm") for _, request in pairs} == set(PROFILES), "Missing or duplicate temporal arm")
    pairs.sort(key=lambda row: row[1]["arm"])
    requests = [request for _, request in pairs]
    for request in requests:
        request_shape(request)
        verify_overrides(request)
        need(sha(request["source_manifest"]) == SOURCE_MANIFEST
             and sha(request["data_manifest"]) == DATA_MANIFEST, "Source/data manifest bytes changed")
    for key in ("source", "source_revision", "source_manifest", "source_manifest_sha256",
                "source_archive", "source_archive_sha256", "data_manifest", "data_manifest_sha256",
                "active_streams", "experiment", "world_size", "per_rank_batch", "global_batch",
                "target_steps", "lr_policy", "model_execution", "operational_service",
                "lambda_runtime", "runtime_localization", "data_cache", "allocation"):
        need(requests[0].get(key) == requests[1].get(key), "Temporal control changed: " + key)
    need(requests[0]["run_id"] != requests[1]["run_id"], "Duplicate full W&B identity")
    need(requests[0]["smoke"]["job_id"] != requests[1]["smoke"]["job_id"], "Shared GPU smoke job")
    outputs, image_hashes, records, normalized = [], [], [], []
    for binding, request in pairs:
        completion, preflight = verify_evidence(request)
        smoke_request = bound(request["smoke"]["request"])
        need(smoke_request.get("intended_full_run_id") == request["run_id"]
             and smoke_request.get("intended_full_output") == request["output"]
             and smoke_request.get("intended_full_target_steps") == request["target_steps"],
             "Smoke belongs to another full experiment")
        outputs.extend((request["output"], request["local_logs"], smoke_request["output"], smoke_request["local_logs"]))
        from omegaconf import OmegaConf
        smoke_cfg = OmegaConf.to_container(OmegaConf.create(bound(request["smoke"]["resolved"], False)), resolve=True)
        cfg = compose_request(request)
        compare_configs(cfg, smoke_cfg, request)
        normalized.append(normalize_config(request, cfg))
        ranks = []
        for rank in completion["ranks"]:
            value = rank["initial_image_weights_sha256"]
            need(valid_hash(value) and valid_hash(rank["data_context_sha256"])
                 and all(valid_hash(v) for v in rank["initial_weights"] + rank["paired_rgb_batch_sha256"]),
                 "Missing actual initial/batch/context hash")
            image_hashes.append(value)
            ranks.append({key: rank[key] for key in (
                "rank", "initial_image_weights_sha256", "initial_weights", "paired_rgb_batch_sha256", "data_context_sha256")})
        records.append(dict(arm=request["arm"], prototype=binding,
                            canonical_request_sha256=canonical_request(request),
                            resolved_sha256=digest(cfg), smoke_job_id=request["smoke"]["job_id"],
                            smoke_completion_sha256=request["smoke"]["completion"]["sha256"],
                            smoke_resolved_sha256=request["smoke"]["resolved"]["sha256"],
                            data_context_sha256=preflight["provenance"]["data_context_sha256"],
                            ranks=ranks))
    disjoint(outputs)
    for request in requests:
        disjoint([request["output"], request["local_logs"], request["source"], request["data_cache"]["root"]])
    need(len(image_hashes) == 8 and len(set(image_hashes)) == 1, "Actual initial image weights differ across ranks/arms")
    need(normalized[0] == normalized[1], "Unexpected D/C scientific configuration difference")
    return dict(schema=SCHEMA, status="PASS", source_revision=SOURCE,
                source_manifest_sha256=SOURCE_MANIFEST, data_manifest_sha256=DATA_MANIFEST,
                initial_image_weights_sha256=image_hashes[0], normalized_config_sha256=digest(normalized[0]),
                fairness="FAIR WITH CAVEATS",
                caveats=["Equal optimizer updates and global batch; temporal representation, frame/token exposure and compute differ.",
                         "Actual training draws, full-model initialization and data-context hashes are recorded separately, not paired."],
                evidence=records)


def admit(request):
    need(admission_mode(request) == "temporal_comparison", "Wrong temporal admission mode")
    need(not any(name in request for name in ("continuation", "allocation_stop_step", "allocation_boundaries", "allocation_recovery")),
         "Temporal v1 supports fresh single-allocation requests only")
    receipt = bound(evidence_binding(request["temporal_comparison"]))
    need(receipt.get("schema") == SCHEMA and receipt.get("status") == "PASS"
         and len(receipt.get("evidence", [])) == 2, "Missing temporal GPU evidence")
    expected = verify([row["prototype"] for row in receipt["evidence"]])
    need(receipt == expected, "Temporal receipt differs from freshly verified evidence")
    members = [row for row in receipt["evidence"]
               if row["arm"] == request["arm"] and row["canonical_request_sha256"] == canonical_request(request)]
    need(len(members) == 1, "Full request is not an exact admitted prototype member")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs=2, type=Path, help="Two immutable full prototypes without temporal_comparison")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prototypes = [dict(path=str(path.resolve()), sha256=sha(path)) for path in args.requests]
    write_once(args.output, verify(prototypes))


if __name__ == "__main__":
    main()
