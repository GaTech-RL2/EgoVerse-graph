#!/bin/bash
# Portable one-GPU launcher for the approved U-Socket Action Flow pair.
#
# Cluster-specific account, partition, QoS, source, runtime, data, and output
# locations are explicit inputs.  The executable model contract remains shared
# across Phoenix and ICE so the two queue candidates cannot silently drift.
# Submit with one H100 or H200 and command-line Slurm resource arguments.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

# Resource selection is intentionally supplied by the submitter.  Preflight
# runs on one CPU node; smoke/full must pass exactly
# ``--gres=gpu:1 --constraint='H100|H200'`` and is checked again on-node.

set -Eeuo pipefail
umask 077

die() {
  printf 'launch_action_flow_usocket: %s\n' "$*" >&2
  exit 64
}

required() {
  local name=$1
  test -n "${!name:-}" || die "set $name"
}

absolute_path() {
  local label=$1 value=$2
  [[ "$value" = /* ]] || die "$label must be an absolute path"
  test "$value" != / || die "$label must identify a specific path"
}

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

for variable in \
  AF_REPO AF_EXPECTED_HEAD AF_PYTHON AF_DATASET_DIR AF_OUTPUT_DIR \
  AF_EXPERIMENT AF_WANDB_ENTITY AF_WANDB_PROJECT AF_WANDB_RUN_ID \
  AF_EXPECTED_SPLIT_MANIFEST_SHA256 AF_EXPECTED_LAUNCHER_SHA256 \
  AF_CONTENT_MANIFEST AF_EXPECTED_CONTENT_MANIFEST_SHA256 \
  AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256 \
  AF_CLUSTER_LABEL AF_EXPECTED_ACCOUNT AF_EXPECTED_PARTITION AF_EXPECTED_QOS; do
  required "$variable"
done

AF_LAUNCH_MODE=${AF_LAUNCH_MODE:-preflight}
AF_RUN_KIND=${AF_RUN_KIND:-smoke}
AF_MAX_RESTARTS=${AF_MAX_RESTARTS:-4}
AF_WANDB_NAME=${AF_WANDB_NAME:-$AF_WANDB_RUN_ID}
AF_WANDB_GROUP=${AF_WANDB_GROUP:-usocket-action-flow-bc-v1}
AF_WANDB_TAGS=${AF_WANDB_TAGS:-"[pushshapes,u-socket,action-flow,h16,z8,k14,single-gpu,$AF_RUN_KIND]"}
AF_TOOL_DIR=${AF_TOOL_DIR:-$AF_REPO/scripts/ice}
AF_GPU_PROBE=${AF_GPU_PROBE:-$AF_TOOL_DIR/ice_gpu_probe.py}
AF_REQUEUE_RUNNER=${AF_REQUEUE_RUNNER:-$AF_TOOL_DIR/ice_requeue_runner.py}
AF_DATASET_VALIDATOR=${AF_DATASET_VALIDATOR:-$AF_TOOL_DIR/validate_planar_dataset.py}
AF_CHECKPOINT_VALIDATOR=${AF_CHECKPOINT_VALIDATOR:-$AF_TOOL_DIR/validate_lightning_checkpoint.py}
AF_RUNTIME_LOCK_CAPTURE=${AF_RUNTIME_LOCK_CAPTURE:-$AF_TOOL_DIR/capture_runtime_lock.py}
AF_CONFIG_VALIDATOR=${AF_CONFIG_VALIDATOR:-$AF_REPO/tools/validate_action_flow_config.py}
AF_SMOKE_VERIFIER=${AF_SMOKE_VERIFIER:-$AF_REPO/scripts/train/verify_action_flow_training_smoke.py}
AF_SLURM_CONTRACT_VALIDATOR=${AF_SLURM_CONTRACT_VALIDATOR:-$AF_REPO/scripts/train/validate_slurm_job_contract.py}
AF_STORAGE_VALIDATOR=${AF_STORAGE_VALIDATOR:-$AF_REPO/scripts/train/check_checkpoint_storage.py}
AF_SPLIT_MANIFEST=${AF_SPLIT_MANIFEST:-$AF_REPO/egomimic/hydra_configs/data/pusht/planar_v2_usocket_dp_3k_split_seed42_v1.json}
AF_PRECISION=${AF_PRECISION:-bf16}
AF_SEED=${AF_SEED:-42}
AF_STORAGE_SAFETY_RESERVE_BYTES=${AF_STORAGE_SAFETY_RESERVE_BYTES:-2147483648}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
LAUNCHER=${BASH_SOURCE[0]}
CANONICAL_CONTENT_MANIFEST_SHA256=a1c81fb0ce8967aba795383a293180f9ba08a0ecfdd6f4a878afb20b39733761
CANONICAL_DATASET_CONTENT_AGGREGATE_SHA256=80f835ad37c3d5c5b7b2d5c3e1656c307ee567a1f63f51081165bf404b8ceb52

case "$AF_LAUNCH_MODE" in
  preflight|run) ;;
  *) die "AF_LAUNCH_MODE must be preflight or run" ;;
esac
case "$AF_RUN_KIND" in
  smoke|full) ;;
  *) die "AF_RUN_KIND must be smoke or full" ;;
esac
if test "$AF_LAUNCH_MODE" = run; then
  required AF_EXPECTED_MEMORY
  required AF_EXPECTED_TIME_LIMIT
fi
case "$AF_EXPERIMENT" in
  pusht/action_flow_bc_usocket_recon1_s42)
    AF_RECONSTRUCTION_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon1_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_recon10_s42)
    AF_RECONSTRUCTION_WEIGHT=10
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon10_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_recon10_warmup10k_s42)
    AF_RECONSTRUCTION_WEIGHT=10
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon10_warmup10k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=10000
    ;;
  pusht/action_flow_bc_usocket_recon100_s42)
    AF_RECONSTRUCTION_WEIGHT=100
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon100_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  *) die "AF_EXPERIMENT must select an approved recon-1, recon-10, or recon-100 config" ;;
esac

if test "$AF_EXPERIMENT" = \
  pusht/action_flow_bc_usocket_recon10_warmup10k_s42; then
  AF_COMPARISON_EXPERIMENT=
elif test "$AF_EXPERIMENT" = pusht/action_flow_bc_usocket_recon1_s42; then
  AF_COMPARISON_EXPERIMENT=pusht/action_flow_bc_usocket_recon10_s42
else
  AF_COMPARISON_EXPERIMENT=pusht/action_flow_bc_usocket_recon1_s42
fi

[[ "$AF_EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || \
  die "AF_EXPECTED_HEAD must be a full lowercase commit"
[[ "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "AF_EXPECTED_SPLIT_MANIFEST_SHA256 must be a lowercase SHA-256"
[[ "$AF_EXPECTED_LAUNCHER_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "AF_EXPECTED_LAUNCHER_SHA256 must be a lowercase SHA-256"
[[ "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "AF_EXPECTED_CONTENT_MANIFEST_SHA256 must be a lowercase SHA-256"
[[ "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256 must be a lowercase SHA-256"
test "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" = \
  "$CANONICAL_CONTENT_MANIFEST_SHA256" || \
  die "content manifest is not the canonical USocket corpus manifest"
test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = \
  "$CANONICAL_DATASET_CONTENT_AGGREGATE_SHA256" || \
  die "dataset aggregate is not the canonical USocket corpus content"
[[ "$AF_WANDB_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe W&B run ID"
[[ "$AF_CLUSTER_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe cluster label"
[[ "$AF_MAX_RESTARTS" =~ ^[1-9][0-9]*$ ]] || die "invalid AF_MAX_RESTARTS"
[[ "$AF_SEED" =~ ^[0-9]+$ ]] || die "invalid AF_SEED"
[[ "$AF_STORAGE_SAFETY_RESERVE_BYTES" =~ ^[1-9][0-9]*$ ]] || \
  die "invalid AF_STORAGE_SAFETY_RESERVE_BYTES"
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || die "invalid SLURM_RESTART_COUNT"
test "$RESTART_COUNT" -le "$AF_MAX_RESTARTS" || die "restart limit exceeded"
test "$AF_PRECISION" = bf16 || die "AF_PRECISION must be bf16"
if test "$AF_LAUNCH_MODE" = run && test "$AF_RUN_KIND" = smoke \
  && test "$RESTART_COUNT" -ne 0; then
  die "a preempted two-step smoke must be relaunched with a fresh immutable identity"
fi

for binding in \
  "AF_REPO:$AF_REPO" "AF_PYTHON:$AF_PYTHON" \
  "AF_DATASET_DIR:$AF_DATASET_DIR" "AF_OUTPUT_DIR:$AF_OUTPUT_DIR" \
  "AF_TOOL_DIR:$AF_TOOL_DIR" "AF_GPU_PROBE:$AF_GPU_PROBE" \
  "AF_REQUEUE_RUNNER:$AF_REQUEUE_RUNNER" \
  "AF_DATASET_VALIDATOR:$AF_DATASET_VALIDATOR" \
  "AF_CHECKPOINT_VALIDATOR:$AF_CHECKPOINT_VALIDATOR" \
  "AF_RUNTIME_LOCK_CAPTURE:$AF_RUNTIME_LOCK_CAPTURE" \
  "AF_CONFIG_VALIDATOR:$AF_CONFIG_VALIDATOR" \
  "AF_SMOKE_VERIFIER:$AF_SMOKE_VERIFIER" \
  "AF_SLURM_CONTRACT_VALIDATOR:$AF_SLURM_CONTRACT_VALIDATOR" \
  "AF_STORAGE_VALIDATOR:$AF_STORAGE_VALIDATOR" \
  "AF_CONTENT_MANIFEST:$AF_CONTENT_MANIFEST" \
  "AF_SPLIT_MANIFEST:$AF_SPLIT_MANIFEST"; do
  absolute_path "${binding%%:*}" "${binding#*:}"
done

test -d "$AF_REPO" || die "AF_REPO is unavailable"
test -x "$AF_PYTHON" || die "AF_PYTHON is not executable"
test -d "$AF_DATASET_DIR" || die "AF_DATASET_DIR is unavailable"
test -d "$AF_TOOL_DIR" || die "AF_TOOL_DIR is unavailable"
test -s "$AF_GPU_PROBE" || die "GPU probe is unavailable"
test -s "$AF_REQUEUE_RUNNER" || die "requeue runner is unavailable"
test -s "$AF_DATASET_VALIDATOR" || die "dataset validator is unavailable"
test -x "$AF_CHECKPOINT_VALIDATOR" || die "checkpoint validator is not executable"
test -s "$AF_RUNTIME_LOCK_CAPTURE" || die "runtime-lock capture is unavailable"
test -s "$AF_CONFIG_VALIDATOR" || die "Action Flow config validator is unavailable"
test -s "$AF_SMOKE_VERIFIER" || die "Action Flow smoke verifier is unavailable"
test -s "$AF_SLURM_CONTRACT_VALIDATOR" || \
  die "Slurm contract validator is unavailable"
test -s "$AF_STORAGE_VALIDATOR" || die "storage validator is unavailable"
test -s "$AF_CONTENT_MANIFEST" || die "dataset content manifest is unavailable"
test -s "$AF_SPLIT_MANIFEST" || die "split manifest is unavailable"
test "$(sha256 "$LAUNCHER")" = "$AF_EXPECTED_LAUNCHER_SHA256" || \
  die "launcher SHA-256 mismatch"
test "$(sha256 "$AF_SPLIT_MANIFEST")" = "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" || \
  die "split manifest SHA-256 mismatch"
test "$(sha256 "$AF_CONTENT_MANIFEST")" = \
  "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" || \
  die "content manifest SHA-256 mismatch"
if test -n "${AF_CACHED_DATASET_VALIDATION:-}"; then
  required AF_EXPECTED_CACHED_DATASET_VALIDATION_SHA256
  absolute_path AF_CACHED_DATASET_VALIDATION "$AF_CACHED_DATASET_VALIDATION"
  test -s "$AF_CACHED_DATASET_VALIDATION" || \
    die "cached dataset validation is unavailable"
  test "$(sha256 "$AF_CACHED_DATASET_VALIDATION")" = \
    "$AF_EXPECTED_CACHED_DATASET_VALIDATION_SHA256" || \
    die "cached dataset validation SHA-256 mismatch"
fi
test "$(git -C "$AF_REPO" rev-parse HEAD)" = "$AF_EXPECTED_HEAD" || \
  die "source HEAD mismatch"
test -z "$(git -C "$AF_REPO" status --porcelain=v1 --untracked-files=all)" || \
  die "source repository must be exactly clean"
test -s "$AF_REPO/egomimic/hydra_configs/experiment/$AF_EXPERIMENT.yaml" || \
  die "selected experiment config is unavailable"
if test -n "${AF_EXTRA_PYTHONPATH:-}"; then
  absolute_path AF_EXTRA_PYTHONPATH "$AF_EXTRA_PYTHONPATH"
  test -d "$AF_EXTRA_PYTHONPATH" || die "extra Python path is unavailable"
fi

export PATH="$(dirname "$AF_PYTHON"):$PATH"
export PYTHONPATH="$AF_REPO${AF_EXTRA_PYTHONPATH:+:$AF_EXTRA_PYTHONPATH}"
export AF_EXTRA_PYTHONPATH
export AF_REPO AF_PYTHON AF_EXPERIMENT AF_OUTPUT_DIR AF_DATASET_DIR
export AF_WANDB_ENTITY AF_WANDB_PROJECT AF_WANDB_RUN_ID AF_WANDB_NAME
export AF_WANDB_GROUP AF_WANDB_TAGS
export ICE_REQUIRE_LR_SCHEDULERS=1
export ICE_EXPECTED_SOURCE_COMMIT=$AF_EXPECTED_HEAD
export ICE_EXPECTED_RUN_ID=$AF_WANDB_RUN_ID
export ICE_EXPECTED_SPLIT_SHA256=$AF_EXPECTED_SPLIT_MANIFEST_SHA256
if test -n "${AF_EXPECTED_NORM_SHA256:-}"; then
  export ICE_EXPECTED_NORMALIZATION_SHA256=$AF_EXPECTED_NORM_SHA256
fi

if test "$AF_RUN_KIND" = smoke; then
  MAX_STEPS=2
  VALIDATE_EVERY=1
  LIMIT_TRAIN_BATCHES=2
  LIMIT_VAL_BATCHES=1
  CHECKPOINT_EVERY=1
  LOG_EVERY=1
  TELEMETRY_EVERY=2
  if test "$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS" -gt 0; then
    RECONSTRUCTION_ONLY_WARMUP_STEPS=1
  else
    RECONSTRUCTION_ONLY_WARMUP_STEPS=0
  fi
else
  MAX_STEPS=240000
  VALIDATE_EVERY=10000
  LIMIT_TRAIN_BATCHES=1.0
  LIMIT_VAL_BATCHES=1.0
  CHECKPOINT_EVERY=40000
  LOG_EVERY=10
  TELEMETRY_EVERY=100
  RECONSTRUCTION_ONLY_WARMUP_STEPS=$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS
fi

BASE_OVERRIDES=(
  --config-name=train_zarr_cartesian
  hydra/launcher=basic
  "+experiment=$AF_EXPERIMENT"
  mode=train
  ckpt_path=null
  "hydra.run.dir=$AF_OUTPUT_DIR"
  "++paths.root_dir=$AF_REPO"
  "paths.output_dir=$AF_OUTPUT_DIR"
  "paths.work_dir=$AF_REPO"
  launch_params.gpus_per_node=1
  launch_params.nodes=1
  trainer.accelerator=gpu
  trainer.devices=1
  trainer.num_nodes=1
  trainer.strategy=auto
  trainer.sync_batchnorm=false
  "trainer.precision=$AF_PRECISION"
  trainer.accumulate_grad_batches=1
  "trainer.max_steps=$MAX_STEPS"
  trainer.max_epochs=-1
  trainer.min_epochs=null
  "trainer.val_check_interval=$VALIDATE_EVERY"
  trainer.check_val_every_n_epoch=null
  "trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES"
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES"
  trainer.num_sanity_val_steps=0
  "trainer.log_every_n_steps=$LOG_EVERY"
  model.gradient_telemetry_cadence="$TELEMETRY_EVERY"
  ++model.reconstruction_only_warmup_steps="$RECONSTRUCTION_ONLY_WARMUP_STEPS"
  "++run_provenance.objective.requested_full_reconstruction_only_warmup_steps=$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS"
  "++run_provenance.objective.effective_reconstruction_only_warmup_steps=$RECONSTRUCTION_ONLY_WARMUP_STEPS"
  ++callbacks.model_checkpoint.monitor=null
  callbacks.model_checkpoint.save_top_k=-1
  callbacks.model_checkpoint.save_last=link
  callbacks.model_checkpoint.every_n_epochs=null
  "++callbacks.model_checkpoint.every_n_train_steps=$CHECKPOINT_EVERY"
  ++callbacks.model_checkpoint.train_time_interval=null
  ++callbacks.model_checkpoint.save_on_train_epoch_end=false
  "callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'"
  runtime.slurm_requeue_owner=runner
  runtime.slurm_save_signal=SIGUSR2
  'runtime.slurm_signal_checkpoint_dir=${paths.output_dir}/checkpoints'
  "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path=$AF_DATASET_DIR"
  "data.valid_datasets.pushshapes_sim_u_socket.resolver.folder_path=$AF_DATASET_DIR"
  data.train_dataloader_params.pushshapes_sim_u_socket.batch_size=32
  data.valid_dataloader_params.pushshapes_sim_u_socket.batch_size=16
  evaluator.energy_score_validation_view.world_size=1
  evaluator.energy_score_validation_view.per_rank_batch_size=16
  ++logger.wandb.offline=false
  "++logger.wandb.entity=$AF_WANDB_ENTITY"
  "++logger.wandb.project=$AF_WANDB_PROJECT"
  "++logger.wandb.group=$AF_WANDB_GROUP"
  "++logger.wandb.id=$AF_WANDB_RUN_ID"
  "++logger.wandb.name=$AF_WANDB_NAME"
  "++logger.wandb.resume=allow"
  "++logger.wandb.tags=$AF_WANDB_TAGS"
  "seed=$AF_SEED"
  "++run_provenance.execution_cluster=$AF_CLUSTER_LABEL"
  "++run_provenance.run_kind=$AF_RUN_KIND"
  "++run_provenance.source_commit=$AF_EXPECTED_HEAD"
  "++run_provenance.launcher_sha256=$AF_EXPECTED_LAUNCHER_SHA256"
  "++run_provenance.content_manifest_path=$AF_CONTENT_MANIFEST"
  "++run_provenance.content_manifest_sha256=$AF_EXPECTED_CONTENT_MANIFEST_SHA256"
  "++run_provenance.dataset_content_aggregate_sha256=$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256"
)

if test -n "${AF_NORM_STATS_PATH:-}"; then
  absolute_path AF_NORM_STATS_PATH "$AF_NORM_STATS_PATH"
  if test -d "$AF_NORM_STATS_PATH"; then
    EFFECTIVE_NORM_FILE=$AF_NORM_STATS_PATH/norm_stats.json
  else
    EFFECTIVE_NORM_FILE=$AF_NORM_STATS_PATH
  fi
  test -s "$EFFECTIVE_NORM_FILE" || die "precomputed norm_stats.json is unavailable"
  required AF_EXPECTED_NORM_SHA256
  test "$(sha256 "$EFFECTIVE_NORM_FILE")" = "$AF_EXPECTED_NORM_SHA256" || \
    die "normalization SHA-256 mismatch"
  BASE_OVERRIDES+=(
    "norm_stats.precomputed_norm_path=$AF_NORM_STATS_PATH"
    norm_stats.save_cache_dir=null
    "++run_provenance.normalization_sha256=$AF_EXPECTED_NORM_SHA256"
  )
elif test "$AF_LAUNCH_MODE" = preflight; then
  EFFECTIVE_NORM_FILE=$AF_OUTPUT_DIR/norm_stats/norm_stats.json
  BASE_OVERRIDES+=(norm_stats.precomputed_norm_path=null "norm_stats.save_cache_dir=$AF_OUTPUT_DIR")
else
  die "run mode requires AF_NORM_STATS_PATH and AF_EXPECTED_NORM_SHA256 from a passed preflight"
fi

if test "$AF_LAUNCH_MODE" = preflight; then
  for variable in \
    SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_JOB_QOS \
    SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK; do
    required "$variable"
  done
  test "$SLURM_JOB_ACCOUNT" = "$AF_EXPECTED_ACCOUNT" || die "unexpected Slurm account"
  test "$SLURM_JOB_PARTITION" = "$AF_EXPECTED_PARTITION" || die "unexpected Slurm partition"
  test "$SLURM_JOB_QOS" = "$AF_EXPECTED_QOS" || die "unexpected Slurm QoS"
  test "$SLURM_JOB_NUM_NODES" = 1 || die "preflight requires one node"
  test "$SLURM_NTASKS" = 1 || die "preflight requires one task"
  test "$SLURM_CPUS_PER_TASK" -ge 8 || die "preflight requires at least eight CPUs"
  test "$RESTART_COUNT" = 0 || die "preflight cannot be requeued"
  test ! -e "$AF_OUTPUT_DIR" || die "unique preflight output already exists"
  mkdir -p "$AF_OUTPUT_DIR"
  "$AF_PYTHON" "$AF_RUNTIME_LOCK_CAPTURE" \
    --repo "$AF_REPO" \
    --expected-head "$AF_EXPECTED_HEAD" \
    --lock-input "$AF_REPO/pyproject.toml" \
    --lock-input "$AF_REPO/uv.lock" \
    --output "$AF_OUTPUT_DIR/runtime-lock.json"
  if test -n "${AF_CACHED_DATASET_VALIDATION:-}"; then
    cp -- "$AF_CACHED_DATASET_VALIDATION" \
      "$AF_OUTPUT_DIR/dataset_validation.json"
    "$AF_PYTHON" - "$AF_OUTPUT_DIR/dataset_validation.json" \
      "$AF_DATASET_DIR" "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
      "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
      "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" <<'PY'
import json
import sys

path, dataset, split_sha, content_sha, aggregate_sha = sys.argv[1:]
payload = json.load(open(path))
inventory = payload["physical_inventory"]
content = payload["content_identity"]
assert payload["status"] == "DATASET_VALIDATED"
assert payload["manifest_sha256"] == split_sha
assert inventory["dataset_root"] == dataset
assert inventory["id_overlap_count"] == 0
assert inventory["resolved_path_overlap_count"] == 0
assert inventory["union_matches_inventory"] is True
assert content["manifest_sha256"] == content_sha
assert content["aggregate_sha256"] == aggregate_sha
PY
  else
    "$AF_PYTHON" "$AF_DATASET_VALIDATOR" \
      --repo "$AF_REPO" \
      --dataset-root "$AF_DATASET_DIR" \
      --manifest "$AF_SPLIT_MANIFEST" \
      --expected-manifest-sha256 "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
      --content-manifest "$AF_CONTENT_MANIFEST" \
      --expected-content-manifest-sha256 \
        "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
      --output "$AF_OUTPUT_DIR/dataset_validation.json"
  fi
  (
    cd "$AF_REPO"
    CONFIG_VALIDATOR_ARGS=(
      --experiment "$AF_EXPERIMENT"
      --output "$AF_OUTPUT_DIR/action_flow_pair_preflight.json"
    )
    if test -n "$AF_COMPARISON_EXPERIMENT"; then
      CONFIG_VALIDATOR_ARGS+=(--compare-experiment "$AF_COMPARISON_EXPERIMENT")
    fi
    "$AF_PYTHON" "$AF_CONFIG_VALIDATOR" "${CONFIG_VALIDATOR_ARGS[@]}"
    HYDRA_FULL_ERROR=1 "$AF_PYTHON" -m egomimic.trainHydra \
      "${BASE_OVERRIDES[@]}" --cfg job --resolve \
      > "$AF_OUTPUT_DIR/resolved_config.yaml"
    if test -z "${AF_NORM_STATS_PATH:-}"; then
      HYDRA_FULL_ERROR=1 "$AF_PYTHON" -m egomimic.trainHydra \
        "${BASE_OVERRIDES[@]}" norm_stats_only=true \
        trainer.accelerator=cpu trainer.devices=1 '~logger' \
        "hydra.run.dir=$AF_OUTPUT_DIR/norm_materialize"
    fi
  )
  test -s "$EFFECTIVE_NORM_FILE" || die "normalization materialization failed"
  "$AF_PYTHON" - "$AF_OUTPUT_DIR" "$AF_REPO" "$AF_EXPECTED_HEAD" \
    "$AF_EXPERIMENT" "$AF_DATASET_DIR" "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
    "$LAUNCHER" "$AF_PYTHON" "$EFFECTIVE_NORM_FILE" "$AF_CLUSTER_LABEL" \
    "$SLURM_JOB_ID" "$AF_OUTPUT_DIR/runtime-lock.json" \
    "$AF_CONTENT_MANIFEST" "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
    "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" <<'PY'
import hashlib
import json
import pathlib
import platform
import sys

(
    root,
    repo,
    head,
    experiment,
    dataset,
    split_sha,
    launcher,
    python,
    norm,
    cluster,
    slurm_job_id,
    runtime_lock_path,
    content_manifest_path,
    content_manifest_sha256,
    content_aggregate_sha256,
) = sys.argv[1:]
root = pathlib.Path(root)
norm = pathlib.Path(norm)
runtime_lock_path = pathlib.Path(runtime_lock_path)

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

dataset_report = json.loads((root / "dataset_validation.json").read_text())
pair_report = json.loads((root / "action_flow_pair_preflight.json").read_text())
runtime_lock = json.loads(runtime_lock_path.read_text())
assert dataset_report["status"] == "DATASET_VALIDATED"
assert dataset_report["content_identity"]["manifest_sha256"] == (
    content_manifest_sha256
)
assert dataset_report["content_identity"]["aggregate_sha256"] == (
    content_aggregate_sha256
)
assert pair_report["status"] == "PASS"
assert runtime_lock["status"] == "RUNTIME_LOCKED"
payload = {
    "schema_version": 1,
    "status": "PASS",
    "source": {"repo": repo, "head": head},
    "experiment": experiment,
    "dataset_root": dataset,
    "content_manifest_path": content_manifest_path,
    "content_manifest_sha256": content_manifest_sha256,
    "dataset_content_aggregate_sha256": content_aggregate_sha256,
    "execution_cluster": cluster,
    "slurm_job_id": slurm_job_id,
    "split_manifest_sha256": split_sha,
    "normalization_path": str(norm),
    "normalization_sha256": digest(norm),
    "dataset_validation_sha256": digest(root / "dataset_validation.json"),
    "pair_preflight_sha256": digest(root / "action_flow_pair_preflight.json"),
    "resolved_config_sha256": digest(root / "resolved_config.yaml"),
    "launcher_sha256": digest(launcher),
    "runtime": {
        "python": python,
        "python_version": platform.python_version(),
        "runtime_lock_sha256": digest(runtime_lock_path),
        "python_executable_sha256": runtime_lock["python"][
            "resolved_executable_sha256"
        ],
        "package_inventory_sha256": runtime_lock["installed_distributions"][
            "canonical_sha256"
        ],
        "core_distribution_versions": runtime_lock[
            "core_distribution_versions"
        ],
        "dependency_inputs": runtime_lock["dependency_inputs"],
        "source_tree": runtime_lock["source"]["tree"],
        "torch_runtime": runtime_lock["torch_runtime"],
    },
}
temporary = root / "PREFLIGHT_RESULT.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "PREFLIGHT_RESULT.json")
PY
  sha256sum "$AF_OUTPUT_DIR"/*.json "$AF_OUTPUT_DIR/resolved_config.yaml" \
    "$EFFECTIVE_NORM_FILE" "$AF_CONTENT_MANIFEST" \
    > "$AF_OUTPUT_DIR/SHA256SUMS"
  printf '[preflight] ALL_PASS result=%s\n' "$AF_OUTPUT_DIR/PREFLIGHT_RESULT.json"
  exit 0
fi

required AF_PREFLIGHT_RESULT
required AF_EXPECTED_PREFLIGHT_SHA256
absolute_path AF_PREFLIGHT_RESULT "$AF_PREFLIGHT_RESULT"
test -s "$AF_PREFLIGHT_RESULT" || die "preflight result is unavailable"
[[ "$AF_EXPECTED_PREFLIGHT_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "AF_EXPECTED_PREFLIGHT_SHA256 must be a lowercase SHA-256"
test "$(sha256 "$AF_PREFLIGHT_RESULT")" = "$AF_EXPECTED_PREFLIGHT_SHA256" || \
  die "preflight result SHA-256 mismatch"
BASE_OVERRIDES+=(
  "++run_provenance.preflight_result_sha256=$AF_EXPECTED_PREFLIGHT_SHA256"
)
"$AF_PYTHON" - "$AF_PREFLIGHT_RESULT" "$AF_EXPECTED_HEAD" "$AF_EXPERIMENT" \
  "$AF_DATASET_DIR" "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
  "$AF_EXPECTED_NORM_SHA256" "$AF_EXPECTED_LAUNCHER_SHA256" \
  "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
  "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" \
  "$AF_CLUSTER_LABEL" <<'PY'
import json
import sys

(
    path,
    head,
    experiment,
    dataset,
    split_sha,
    norm_sha,
    launcher_sha,
    content_sha,
    content_aggregate_sha,
    cluster,
) = sys.argv[1:]
payload = json.load(open(path))
assert payload["status"] == "PASS"
assert payload["source"]["head"] == head
assert payload["experiment"] == experiment
assert payload["dataset_root"] == dataset
assert payload["execution_cluster"] == cluster
assert payload["split_manifest_sha256"] == split_sha
assert payload["normalization_sha256"] == norm_sha
assert payload["launcher_sha256"] == launcher_sha
assert payload["content_manifest_sha256"] == content_sha
assert payload["dataset_content_aggregate_sha256"] == content_aggregate_sha
PY

AF_EFFECTIVE_SMOKE_SHA256=
if test "$AF_RUN_KIND" = full; then
  required AF_SMOKE_RESULT
  required AF_EXPECTED_SMOKE_SHA256
  absolute_path AF_SMOKE_RESULT "$AF_SMOKE_RESULT"
  test -s "$AF_SMOKE_RESULT" || die "smoke result is unavailable"
  [[ "$AF_EXPECTED_SMOKE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "AF_EXPECTED_SMOKE_SHA256 must be a lowercase SHA-256"
  test "$(sha256 "$AF_SMOKE_RESULT")" = "$AF_EXPECTED_SMOKE_SHA256" || \
    die "smoke result SHA-256 mismatch"
  "$AF_PYTHON" - "$AF_SMOKE_RESULT" "$AF_EXPECTED_HEAD" "$AF_EXPERIMENT" \
    "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" "$AF_EXPECTED_NORM_SHA256" \
    "$AF_EXPECTED_PREFLIGHT_SHA256" \
    "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
    "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" <<'PY'
import json
import sys

(
    path,
    head,
    experiment,
    split_sha,
    norm_sha,
    preflight_sha,
    content_sha,
    content_aggregate_sha,
) = sys.argv[1:]
payload = json.load(open(path))
assert payload["schema_version"] == 1 and payload["status"] == "PASS"
assert payload["experiment"] == experiment
identities = payload["identities"]
assert identities["repo_head"] == head
assert identities["split_manifest_sha256"] == split_sha
assert identities["normalization_sha256"] == norm_sha
assert identities["content_manifest_sha256"] == content_sha
assert identities["dataset_content_aggregate_sha256"] == content_aggregate_sha
assert payload["preflight"]["sha256"] == preflight_sha
assert payload["checkpoint"]["global_step"] == 2
assert payload["checkpoint"]["file_size_bytes"] > 0
assert payload["checkpoint"]["gradient_routes"]["manifest_sha256"]
if experiment.endswith("_warmup10k_s42"):
    assert payload["checkpoint"]["loss_schedule"] == {
        "joint_objective_begins_at_global_step": 1,
        "reconstruction_only_optimizer_steps": 1,
        "schema_version": 1,
    }
    assert payload["metrics"]["warmup_step"] is not None
assert payload["wandb"]["exit_code"] == 0
assert payload["gpu_probes"]
assert all(
    "H100" in record["gpu_name"] or "H200" in record["gpu_name"]
    for record in payload["gpu_probes"]
)
PY
  AF_EFFECTIVE_SMOKE_SHA256=$AF_EXPECTED_SMOKE_SHA256
  BASE_OVERRIDES+=(
    "++run_provenance.smoke_result_sha256=$AF_EXPECTED_SMOKE_SHA256"
  )
fi

for variable in \
  SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_JOB_QOS \
  SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK; do
  required "$variable"
done
test "$SLURM_JOB_ACCOUNT" = "$AF_EXPECTED_ACCOUNT" || die "unexpected Slurm account"
test "$SLURM_JOB_PARTITION" = "$AF_EXPECTED_PARTITION" || die "unexpected Slurm partition"
test "$SLURM_JOB_QOS" = "$AF_EXPECTED_QOS" || die "unexpected Slurm QoS"
test "$SLURM_JOB_NUM_NODES" = 1 || die "launcher requires one node"
test "$SLURM_NTASKS" = 1 || die "launcher requires one task"
test "$SLURM_CPUS_PER_TASK" -eq 8 || die "launcher requires exactly eight CPUs"

if test "$RESTART_COUNT" = 0; then
  test ! -e "$AF_OUTPUT_DIR" || die "unique run output already exists"
else
  test -d "$AF_OUTPUT_DIR" || die "requeued run output is unavailable"
fi
mkdir -p "$AF_OUTPUT_DIR/checkpoints" "$AF_OUTPUT_DIR/provenance" \
  "$AF_OUTPUT_DIR/runner-state"
ATTEMPT=$AF_OUTPUT_DIR/provenance/restart-$RESTART_COUNT
test ! -e "$ATTEMPT" || die "attempt provenance already exists"
mkdir -p "$ATTEMPT"
cp -- "$AF_PREFLIGHT_RESULT" "$ATTEMPT/PREFLIGHT_RESULT.json"
cp -- "$AF_SPLIT_MANIFEST" "$ATTEMPT/split_manifest.json"
cp -- "$AF_CONTENT_MANIFEST" "$ATTEMPT/content_manifest.json"
cp -- "$LAUNCHER" "$ATTEMPT/launch_action_flow_usocket.sbatch"
if test "$AF_RUN_KIND" = full; then
  cp -- "$AF_SMOKE_RESULT" "$ATTEMPT/SMOKE_RESULT.json"
fi
sha256sum "$EFFECTIVE_NORM_FILE" > "$ATTEMPT/normalization.sha256"
git -C "$AF_REPO" status --short --branch > "$ATTEMPT/source_status.txt"
"$AF_PYTHON" "$AF_RUNTIME_LOCK_CAPTURE" \
  --repo "$AF_REPO" \
  --expected-head "$AF_EXPECTED_HEAD" \
  --lock-input "$AF_REPO/pyproject.toml" \
  --lock-input "$AF_REPO/uv.lock" \
  --output "$ATTEMPT/runtime-lock.json"
"$AF_PYTHON" - "$AF_PREFLIGHT_RESULT" "$ATTEMPT/runtime-lock.json" <<'PY'
import json
import pathlib
import sys

preflight_path, current_path = map(pathlib.Path, sys.argv[1:])
preflight = json.loads(preflight_path.read_text())
expected = preflight["runtime"]
current = json.loads(current_path.read_text())
assert current["status"] == "RUNTIME_LOCKED"
assert current["python"]["resolved_executable_sha256"] == expected[
    "python_executable_sha256"
]
assert current["installed_distributions"]["canonical_sha256"] == expected[
    "package_inventory_sha256"
]
assert current["core_distribution_versions"] == expected[
    "core_distribution_versions"
]
assert current["dependency_inputs"] == expected["dependency_inputs"]
assert current["source"]["head"] == preflight["source"]["head"]
assert current["source"]["tree"] == expected["source_tree"]
expected_torch = dict(expected["torch_runtime"])
current_torch = dict(current["torch_runtime"])
expected_torch.pop("cuda_available_on_capture_node", None)
current_torch.pop("cuda_available_on_capture_node", None)
assert current_torch == expected_torch
PY
nvidia-smi -q > "$ATTEMPT/nvidia_smi_q.txt"
scontrol show job -dd -o "$SLURM_JOB_ID" > "$ATTEMPT/slurm_job.txt"
"$AF_PYTHON" "$AF_SLURM_CONTRACT_VALIDATOR" \
  --record "$ATTEMPT/slurm_job.txt" \
  --expected-job-id "$SLURM_JOB_ID" \
  --expected-account "$AF_EXPECTED_ACCOUNT" \
  --expected-partition "$AF_EXPECTED_PARTITION" \
  --expected-qos "$AF_EXPECTED_QOS" \
  --expected-cpus 8 \
  --expected-memory "$AF_EXPECTED_MEMORY" \
  --expected-time-limit "$AF_EXPECTED_TIME_LIMIT" \
  --expected-constraint 'H100|H200' \
  --output "$ATTEMPT/SLURM_JOB_CONTRACT.json"

SRUN=$(command -v srun)
test -x "$SRUN" || die "srun is unavailable"
export SRUN
"$SRUN" --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 --unbuffered \
  "$AF_PYTHON" "$AF_GPU_PROBE" \
  --expected-world-size 1 \
  --allowed-gpu-name "NVIDIA H100 80GB HBM3" \
  --allowed-gpu-name "NVIDIA H200" \
  --output "$ATTEMPT/gpu_probe.json"

(
  cd "$AF_REPO"
  HYDRA_FULL_ERROR=1 "$AF_PYTHON" -m egomimic.trainHydra \
    "${BASE_OVERRIDES[@]}" --cfg job --resolve \
    > "$ATTEMPT/resolved_config.yaml"
)
"$AF_PYTHON" - "$ATTEMPT/resolved_config.yaml" "$AF_RUN_KIND" \
  "$AF_EXPERIMENT" "$AF_EXPECTED_CONFIG_NAME" \
  "$AF_RECONSTRUCTION_WEIGHT" "$AF_DATASET_DIR" \
  "$AF_NORM_STATS_PATH" "$AF_CLUSTER_LABEL" "$AF_EXPECTED_HEAD" \
  "$AF_EXPECTED_LAUNCHER_SHA256" "$AF_EXPECTED_NORM_SHA256" \
  "$AF_EXPECTED_PREFLIGHT_SHA256" "$AF_EFFECTIVE_SMOKE_SHA256" \
  "$AF_CONTENT_MANIFEST" "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
  "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" \
  "$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS" \
  "$RECONSTRUCTION_ONLY_WARMUP_STEPS" <<'PY'
import math
import pathlib
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf

(
    path,
    run_kind,
    experiment,
    expected_name,
    reconstruction,
    dataset,
    norm,
    cluster,
    source_head,
    launcher_sha,
    normalization_sha,
    preflight_sha,
    smoke_sha,
    content_manifest,
    content_manifest_sha,
    content_aggregate_sha,
    full_warmup_steps,
    effective_warmup_steps,
) = sys.argv[1:]
cfg = OmegaConf.load(pathlib.Path(path))
reconstruction = float(reconstruction)
full_warmup_steps = int(full_warmup_steps)
effective_warmup_steps = int(effective_warmup_steps)
assert cfg.name == expected_name
assert cfg.mode == "train" and cfg.ckpt_path is None and cfg.seed == 42
assert cfg.run_provenance.execution_cluster == cluster
assert cfg.run_provenance.run_kind == run_kind
assert cfg.run_provenance.source_commit == source_head
assert cfg.run_provenance.launcher_sha256 == launcher_sha
assert cfg.run_provenance.normalization_sha256 == normalization_sha
assert cfg.run_provenance.preflight_result_sha256 == preflight_sha
assert str(cfg.run_provenance.content_manifest_path) == content_manifest
assert cfg.run_provenance.content_manifest_sha256 == content_manifest_sha
assert cfg.run_provenance.dataset_content_aggregate_sha256 == content_aggregate_sha
if run_kind == "full":
    assert cfg.run_provenance.smoke_result_sha256 == smoke_sha
else:
    assert smoke_sha == ""
assert cfg.model._target_.endswith("ActionFlowModelWrapper")
assert cfg.model.action_horizon == 16 and cfg.model.action_dim == 4
assert cfg.model.latent_dim == 8 and cfg.model.condition_dim == 67
assert cfg.model.flow_samples_per_content == 14
assert math.isclose(cfg.model.condition_dropout_probability, 0.3)
assert cfg.model.num_inference_steps == 16
assert math.isclose(cfg.model.reconstruction_weight, reconstruction)
assert cfg.model.reconstruction_only_warmup_steps == effective_warmup_steps
assert (
    cfg.run_provenance.objective.effective_reconstruction_only_warmup_steps
    == effective_warmup_steps
)
assert (
    cfg.run_provenance.objective.requested_full_reconstruction_only_warmup_steps
    == full_warmup_steps
)
stages = cfg.model.pipeline.stages
assert [str(stage._target_).rsplit(".", 1)[-1] for stage in stages] == [
    "FusedObsEncoder",
    "GaussianLatentNoise",
    "ActionTargetBuilder",
    "ContentEncoderStage",
    "LatentBridgeStage",
    "ConditionalVelocityStage",
    "ContentDecoderStage",
    "ActionFlowObjectiveStage",
]
assert stages[1].num_tokens == 16 and stages[1].latent_dim == 8
assert stages[4].samples_per_content == 14
assert math.isclose(stages[4].condition_dropout_probability, 0.3)
assert stages[5].num_inference_steps == 16
assert stages[5].field.time_scale == 1000.0
assert stages[7].flow_weight == stages[7].action_velocity_weight == 1.0
assert math.isclose(stages[7].reconstruction_weight, reconstruction)
optimizer, scheduler = cfg.model.optimizer, cfg.model.scheduler
assert optimizer._target_ == "torch.optim.AdamW" and optimizer._partial_ is True
assert optimizer.lr == 3e-5 and list(optimizer.betas) == [0.9, 0.999]
assert optimizer.eps == 1e-8 and optimizer.weight_decay == 1e-4
assert scheduler.max_steps == 240000 and scheduler.warmup_steps == 8000
assert scheduler.warmup_start_factor == 0.1 and scheduler.eta_min == 3e-6
assert cfg.trainer.accelerator == "gpu"
assert cfg.trainer.devices == cfg.trainer.num_nodes == 1
assert cfg.trainer.strategy == "auto" and cfg.trainer.sync_batchnorm is False
assert str(cfg.trainer.precision) == "bf16"
assert cfg.trainer.accumulate_grad_batches == 1
if run_kind == "smoke":
    assert cfg.trainer.max_steps == cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.val_check_interval == cfg.trainer.limit_val_batches == 1
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 1
    assert cfg.model.gradient_telemetry_cadence == 2
else:
    assert run_kind == "full"
    assert cfg.trainer.max_steps == 240000
    assert cfg.trainer.limit_train_batches == cfg.trainer.limit_val_batches == 1.0
    assert cfg.trainer.val_check_interval == 10000
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 40000
    assert cfg.model.gradient_telemetry_cadence == 100
assert cfg.callbacks.model_checkpoint.save_top_k == -1
assert cfg.callbacks.model_checkpoint.save_last == "link"
assert "{step}" in str(cfg.callbacks.model_checkpoint.filename)
source = "pushshapes_sim_u_socket"
assert set(cfg.data.train_datasets) == set(cfg.data.valid_datasets) == {source}
assert cfg.data.train_datasets[source].resolver.folder_path == dataset
assert cfg.data.valid_datasets[source].resolver.folder_path == dataset
assert cfg.data.train_dataloader_params[source].batch_size == 32
assert cfg.data.valid_dataloader_params[source].batch_size == 16
assert str(cfg.norm_stats.precomputed_norm_path) == norm
assert cfg.norm_stats.save_cache_dir is None
diagnostics = cfg.evaluator.action_flow_diagnostics
assert diagnostics.enabled is True and diagnostics.max_batches_per_rank == 1
assert list(diagnostics.raw_noise_levels) == [0.0, 0.25, 0.5, 0.75, 1.0]
assert diagnostics.max_samples == 16 and diagnostics.jacobian_samples == 2
assert diagnostics.capture_activations is True and diagnostics.cknna_k == 10
assert diagnostics.validation_view.world_size == 1
assert diagnostics.validation_view.per_rank_batch_size == 16
pipeline = instantiate(cfg.model.pipeline, device="cpu")
count = sum(parameter.numel() for parameter in pipeline.nets.parameters())
assert count == 50_725_221, count
print(f"[run-preflight] PASS kind={run_kind} parameters={count}")
PY
sha256sum "$ATTEMPT/resolved_config.yaml" > "$ATTEMPT/resolved_config.sha256"

FULL_STORAGE_ARTIFACT=$AF_OUTPUT_DIR/provenance/restart-0/CHECKPOINT_STORAGE.json
if test "$AF_RUN_KIND" = full; then
  if test "$RESTART_COUNT" = 0; then
    read -r AF_SMOKE_CHECKPOINT_BYTES < <(
      "$AF_PYTHON" - "$AF_SMOKE_RESULT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
value = payload["checkpoint"]["file_size_bytes"]
assert isinstance(value, int) and value > 0
print(value)
PY
    )
    [[ "$AF_SMOKE_CHECKPOINT_BYTES" =~ ^[1-9][0-9]*$ ]] || \
      die "smoke checkpoint size is invalid"
    "$AF_PYTHON" "$AF_STORAGE_VALIDATOR" \
      --target "$AF_OUTPUT_DIR" \
      --planned-checkpoint-count 14 \
      --measured-checkpoint-bytes "$AF_SMOKE_CHECKPOINT_BYTES" \
      --safety-reserve-bytes "$AF_STORAGE_SAFETY_RESERVE_BYTES" \
      --output "$FULL_STORAGE_ARTIFACT"
  else
    test -s "$FULL_STORAGE_ARTIFACT" || \
      die "restarted full run lacks initial storage proof"
  fi
fi

read -r -d '' CHILD_CODE <<'BASH' || true
set -Eeuo pipefail
checkpoint_override=ckpt_path=null
if test -n "${ICE_RESUME_CHECKPOINT:-}"; then
  "$AF_PYTHON" - "$ICE_RESUME_CHECKPOINT" \
    "${ICE_RESUME_CHECKPOINT_SHA256:?}" "${ICE_RESUME_GLOBAL_STEP:?}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected, raw_step = sys.argv[2:]
assert int(raw_step) >= 0
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == expected
PY
  checkpoint_override="ckpt_path=$ICE_RESUME_CHECKPOINT"
else
  test -z "${ICE_RESUME_CHECKPOINT_SHA256:-}${ICE_RESUME_GLOBAL_STEP:-}"
fi
cd "$AF_REPO"
export PYTHONPATH="$AF_REPO${AF_EXTRA_PYTHONPATH:+:$AF_EXTRA_PYTHONPATH}"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_SILENT=true MUJOCO_GL=egl
exec "$SRUN" --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --kill-on-bad-exit=1 --unbuffered \
  "$AF_PYTHON" -m egomimic.trainHydra "$@" \
  "$checkpoint_override"
BASH

RUNNER_ARGS=(
  --state-dir "$AF_OUTPUT_DIR/runner-state"
  --checkpoint-glob "$AF_OUTPUT_DIR/checkpoints/*.ckpt"
  --checkpoint-validator "$AF_CHECKPOINT_VALIDATOR"
  --checkpoint-signal USR2
  --checkpoint-forwarding slurm-steps
  --checkpoint-grace-seconds 480
  --max-restarts "$AF_MAX_RESTARTS"
  --requeue-owner runner
  --confirm-child-requeue-disabled
  --completion-sentinel "$AF_OUTPUT_DIR/COMPLETE.json"
)
"$AF_PYTHON" "$AF_REQUEUE_RUNNER" "${RUNNER_ARGS[@]}" -- \
  /usr/bin/env bash -c "$CHILD_CODE" action-flow-usocket "${BASE_OVERRIDES[@]}"

if test "$AF_RUN_KIND" = smoke; then
  SMOKE_CHECKPOINT=$AF_OUTPUT_DIR/checkpoints/last.ckpt
  test -s "$SMOKE_CHECKPOINT" || die "smoke checkpoint is unavailable"
  test -L "$SMOKE_CHECKPOINT" || die "smoke last.ckpt must be a symlink"
  read -r SMOKE_CHECKPOINT_BYTES < <(stat -Lc %s "$SMOKE_CHECKPOINT")
  [[ "$SMOKE_CHECKPOINT_BYTES" =~ ^[1-9][0-9]*$ ]] || \
    die "smoke checkpoint size is invalid"
  "$AF_PYTHON" "$AF_STORAGE_VALIDATOR" \
    --target "$AF_OUTPUT_DIR" \
    --planned-checkpoint-count 14 \
    --measured-checkpoint-bytes "$SMOKE_CHECKPOINT_BYTES" \
    --safety-reserve-bytes "$AF_STORAGE_SAFETY_RESERVE_BYTES" \
    --output "$ATTEMPT/CHECKPOINT_STORAGE.json"
  "$AF_PYTHON" "$AF_SMOKE_VERIFIER" "$AF_OUTPUT_DIR" \
    --expected-head "$AF_EXPECTED_HEAD" \
    --expected-experiment "$AF_EXPERIMENT" \
    --expected-reconstruction-weight "$AF_RECONSTRUCTION_WEIGHT" \
    --expected-split-sha256 "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
    --expected-normalization-sha256 "$AF_EXPECTED_NORM_SHA256" \
    --expected-content-manifest-sha256 \
      "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
    --expected-dataset-content-aggregate-sha256 \
      "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" \
    --expected-config-sha256 "$(sha256 "$AF_OUTPUT_DIR/.hydra/config.yaml")" \
    --expected-preflight-sha256 "$AF_EXPECTED_PREFLIGHT_SHA256"
  printf '[smoke] ALL_PASS result=%s wandb=https://wandb.ai/%s/%s/runs/%s\n' \
    "$AF_OUTPUT_DIR/SMOKE_RESULT.json" "$AF_WANDB_ENTITY" \
    "$AF_WANDB_PROJECT" "$AF_WANDB_RUN_ID"
else
  test -s "$AF_OUTPUT_DIR/COMPLETE.json" || die "full run lacks COMPLETE.json"
  printf '[full] ALL_PASS wandb=https://wandb.ai/%s/%s/runs/%s\n' \
    "$AF_WANDB_ENTITY" "$AF_WANDB_PROJECT" "$AF_WANDB_RUN_ID"
fi
