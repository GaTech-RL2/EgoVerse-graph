#!/bin/bash
# Maintained, typed Lambda entry point for YAM Search Sort Stationary Action Flow.
# Slurm account/partition/QoS/GPU/time/output are supplied at submission.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
set -Eeuo pipefail
umask 077

fail() { printf 'yam-action-flow-launch: %s\n' "$*" >&2; exit 64; }
for name in AF_YAM_MODE AF_YAM_PHASE AF_YAM_REPO AF_YAM_SOURCE_COMMIT \
  AF_YAM_PYTHON AF_YAM_CONTAINER_IMAGE AF_YAM_DATA_ROOT AF_YAM_SPLIT_MANIFEST \
  AF_YAM_SPLIT_SHA AF_YAM_CONTENT_SUMMARY AF_YAM_CONTENT_SUMMARY_SHA \
  AF_YAM_NORM_JSON AF_YAM_NORM_SHA AF_YAM_RUN_DIR AF_YAM_PREFLIGHT_DIR \
  AF_YAM_WANDB_ENTITY AF_YAM_WANDB_PROJECT AF_YAM_WANDB_ID; do
  test -n "${!name:-}" || fail "missing $name"
done
case "$AF_YAM_MODE" in preflight|run) ;; *) fail 'mode must be preflight or run';; esac
case "$AF_YAM_PHASE" in smoke|full) ;; *) fail 'phase must be smoke or full';; esac
for name in AF_YAM_REPO AF_YAM_PYTHON AF_YAM_DATA_ROOT AF_YAM_SPLIT_MANIFEST \
  AF_YAM_CONTENT_SUMMARY AF_YAM_NORM_JSON AF_YAM_RUN_DIR AF_YAM_PREFLIGHT_DIR; do
  value=${!name}
  [[ "$value" = /* && "$value" != / ]] || fail "$name must be an exact absolute path"
done
for name in AF_YAM_SOURCE_COMMIT AF_YAM_SPLIT_SHA AF_YAM_CONTENT_SUMMARY_SHA AF_YAM_NORM_SHA; do
  [[ "${!name}" =~ ^[0-9a-f]{40,64}$ ]] || fail "$name is not a lowercase Git/SHA identity"
done
test -x "$AF_YAM_PYTHON" || fail 'pinned Python runtime absent'
test -d "$AF_YAM_DATA_ROOT" || fail 'frozen YAM corpus absent'
test -f "$AF_YAM_SPLIT_MANIFEST" || fail 'split manifest absent'
test -f "$AF_YAM_CONTENT_SUMMARY" || fail 'content receipt absent'
test -f "$AF_YAM_NORM_JSON" || fail 'train-only normalizer absent'
test "$(sha256sum "$AF_YAM_SPLIT_MANIFEST" | cut -d' ' -f1)" = "$AF_YAM_SPLIT_SHA" || fail 'split SHA mismatch'
test "$(sha256sum "$AF_YAM_CONTENT_SUMMARY" | cut -d' ' -f1)" = "$AF_YAM_CONTENT_SUMMARY_SHA" || fail 'content receipt SHA mismatch'
test "$(sha256sum "$AF_YAM_NORM_JSON" | cut -d' ' -f1)" = "$AF_YAM_NORM_SHA" || fail 'normalizer SHA mismatch'
test "$(git -C "$AF_YAM_REPO" rev-parse HEAD)" = "$AF_YAM_SOURCE_COMMIT" || fail 'source commit mismatch'
test -z "$(git -C "$AF_YAM_REPO" status --porcelain)" || fail 'source checkout is dirty'

validator=$AF_YAM_REPO/scripts/train/verify_yam_action_flow_launch.py
test -f "$validator" || fail 'reviewed YAM launch validator absent'
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$AF_YAM_REPO:$AF_YAM_REPO/tools
container=(
  --container-image="$AF_YAM_CONTAINER_IMAGE"
  --container-writable
  --container-mounts=/workspace:/workspace:rw
  --container-workdir="$AF_YAM_REPO"
)

identities=$(srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" "$validator" \
  --split "$AF_YAM_SPLIT_MANIFEST" --split-sha "$AF_YAM_SPLIT_SHA" \
  --content-summary "$AF_YAM_CONTENT_SUMMARY" \
  --content-summary-sha "$AF_YAM_CONTENT_SUMMARY_SHA")
read -r train_sha valid_sha content_sha <<< "$identities"
for value in "$train_sha" "$valid_sha" "$content_sha"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail 'derived corpus identity invalid'
done
export YAM_TRAIN_IDS_SHA256=$train_sha
export YAM_VALID_IDS_SHA256=$valid_sha

overrides=(
  '+experiment=e1/yam_stationary_action_flow_h816_s42'
  "e1.data_root=$AF_YAM_DATA_ROOT"
  "norm_stats.precomputed_norm_path=$AF_YAM_NORM_JSON"
  "hydra.run.dir=$AF_YAM_RUN_DIR"
  "paths.output_dir=$AF_YAM_RUN_DIR"
  "run_provenance.source_commit=$AF_YAM_SOURCE_COMMIT"
  "run_provenance.normalization_sha256=$AF_YAM_NORM_SHA"
  "run_provenance.dataset_content_aggregate_sha256=$content_sha"
  "++logger.wandb.entity=$AF_YAM_WANDB_ENTITY"
  "++logger.wandb.project=$AF_YAM_WANDB_PROJECT"
  "++logger.wandb.id=$AF_YAM_WANDB_ID"
  "++logger.wandb.name=$AF_YAM_WANDB_ID"
  '++logger.wandb.resume=allow'
  'trainer.devices=1'
  'trainer.num_nodes=1'
)
if test "$AF_YAM_PHASE" = smoke; then
  overrides+=(
    'trainer.max_steps=1'
    'trainer.val_check_interval=1'
    'trainer.limit_val_batches=1'
    'callbacks.model_checkpoint.every_n_train_steps=1'
  )
fi

resolved=$AF_YAM_PREFLIGHT_DIR/resolved-$AF_YAM_PHASE.yaml
receipt=$AF_YAM_PREFLIGHT_DIR/receipt-$AF_YAM_PHASE.json
if test "$AF_YAM_MODE" = preflight; then
  test ! -e "$resolved" && test ! -e "$receipt" || fail 'preflight output exists'
  mkdir -p "$AF_YAM_PREFLIGHT_DIR"
  srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" -m egomimic.trainHydra \
    --cfg job --resolve "${overrides[@]}" > "$resolved"
  srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" "$validator" \
    --split "$AF_YAM_SPLIT_MANIFEST" --split-sha "$AF_YAM_SPLIT_SHA" \
    --content-summary "$AF_YAM_CONTENT_SUMMARY" \
    --content-summary-sha "$AF_YAM_CONTENT_SUMMARY_SHA" \
    --config "$resolved" --receipt "$receipt" --phase "$AF_YAM_PHASE" \
    --source-commit "$AF_YAM_SOURCE_COMMIT" --data-root "$AF_YAM_DATA_ROOT" \
    --norm-json "$AF_YAM_NORM_JSON" --norm-sha "$AF_YAM_NORM_SHA" \
    --wandb-entity "$AF_YAM_WANDB_ENTITY" --wandb-project "$AF_YAM_WANDB_PROJECT" \
    --wandb-id "$AF_YAM_WANDB_ID" --run-dir "$AF_YAM_RUN_DIR"
  exit 0
fi

test -f "$resolved" && test -f "$receipt" || fail 'official preflight missing'
test -n "${AF_YAM_PREFLIGHT_RECEIPT_SHA:-}" || fail 'pin official preflight receipt SHA'
test "$(sha256sum "$receipt" | cut -d' ' -f1)" = "$AF_YAM_PREFLIGHT_RECEIPT_SHA" || fail 'preflight receipt changed'
temp_config=$(mktemp "$AF_YAM_PREFLIGHT_DIR/run-resolved-$AF_YAM_PHASE.XXXXXX.yaml")
trap 'test -z "${temp_config:-}" || rm -f -- "$temp_config"' EXIT
srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" -m egomimic.trainHydra \
  --cfg job --resolve "${overrides[@]}" > "$temp_config"
cmp -s "$resolved" "$temp_config" || fail 'run resolved config differs from official preflight'
if test "${SLURM_RESTART_COUNT:-0}" = 0; then
  test ! -e "$AF_YAM_RUN_DIR" || fail 'new run output already exists'
else
  test -f "$AF_YAM_RUN_DIR/checkpoints/last.ckpt" || fail 'restart checkpoint absent'
fi
# Pyxis can expose all physical devices to nvidia-smi even when Slurm assigns
# one GPU. Check the CUDA view actually used by the model, and bind both child
# steps to the single assigned GPU explicitly.
srun --overlap --ntasks=1 --gpus-per-task=1 "${container[@]}" \
  "$AF_YAM_PYTHON" -c 'import torch; assert torch.cuda.device_count() == 1, f"expected one CUDA GPU, got {torch.cuda.device_count()}"; name = torch.cuda.get_device_name(0); assert name.startswith("NVIDIA H100"), f"expected H100, got {name}"; print("YAM_ACTION_FLOW_GPU_PASS", name, flush=True)'
export HYDRA_FULL_ERROR=1
srun --overlap --ntasks=1 --gpus-per-task=1 "${container[@]}" "$AF_YAM_PYTHON" -m egomimic.trainHydra "${overrides[@]}"
