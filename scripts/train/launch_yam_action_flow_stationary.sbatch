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
case "$AF_YAM_PHASE" in norm|smoke|full) ;; *) fail 'phase must be norm, smoke or full';; esac
AF_YAM_RECIPE=${AF_YAM_RECIPE:-yam_only}
if test "$AF_YAM_PHASE" = norm; then
  test "$AF_YAM_RECIPE" = yam_human_keypoints || fail 'normalization phase requires maintained co-train recipe'
  test -n "${AF_YAM_NORM_CACHE_DIR:-}" || fail 'normalization cache directory missing'
  [[ "$AF_YAM_NORM_CACHE_DIR" = /* && "$AF_YAM_NORM_CACHE_DIR" != / ]] || fail 'normalization cache must be absolute'
fi
case "$AF_YAM_RECIPE" in yam_only|yam_human_keypoints) ;; *) fail 'unknown YAM Action Flow recipe';; esac
AF_YAM_CHECKPOINT_POLICY=${AF_YAM_CHECKPOINT_POLICY:-all}
AF_YAM_INFERENCE_METHOD=${AF_YAM_INFERENCE_METHOD:-dopri5}
case "$AF_YAM_CHECKPOINT_POLICY" in all|dit_half) ;; *) fail 'unknown native checkpoint policy';; esac
case "$AF_YAM_INFERENCE_METHOD" in euler|dopri5) ;; *) fail 'unknown native inference method';; esac
if test "$AF_YAM_RECIPE" != yam_human_keypoints; then
  test "$AF_YAM_CHECKPOINT_POLICY" = all && test "$AF_YAM_INFERENCE_METHOD" = dopri5 || fail 'execution extension requires maintained YAM+human recipe'
fi
AF_YAM_AUGMENTATION=${AF_YAM_AUGMENTATION:-none}
case "$AF_YAM_AUGMENTATION" in
  none) AF_YAM_EXPERIMENT=yam_human_keypoints_action_flow_h816_private512_s42 ;;
  human_speed_v1)
    test "$AF_YAM_RECIPE" = yam_human_keypoints || fail 'human speed requires maintained two-source recipe'
    for name in AF_STATIONARY_SPEED_REFERENCE AF_STATIONARY_SPEED_REFERENCE_RECEIPT AF_STATIONARY_SPEED_REFERENCE_RECEIPT_SHA; do
      test -n "${!name:-}" || fail "missing $name"
    done
    test -f "$AF_STATIONARY_SPEED_REFERENCE_RECEIPT" || fail 'train-only speed reference receipt absent'
    test "$(sha256sum "$AF_STATIONARY_SPEED_REFERENCE_RECEIPT" | cut -d' ' -f1)" = "$AF_STATIONARY_SPEED_REFERENCE_RECEIPT_SHA" || fail 'speed reference receipt SHA changed'
    AF_YAM_EXPERIMENT=yam_human_keypoints_speed_h816_private512_s42 ;;
  *) fail 'unknown augmentation contract' ;;
esac
export AF_YAM_EXPERIMENT AF_YAM_AUGMENTATION
AF_YAM_ACCUMULATION=${AF_YAM_ACCUMULATION:-16}
case "$AF_YAM_ACCUMULATION" in 1|16) ;; *) fail 'accumulation must be 1 or 16';; esac
if test "$AF_YAM_RECIPE" = yam_human_keypoints; then
  test "$AF_YAM_ACCUMULATION" = 1 || fail 'YAM+human co-train requires accumulation 1'
  for name in AF_HUMAN_DATA_ROOT AF_HUMAN_CONTENT_MANIFEST AF_HUMAN_CONTENT_SHA; do
    test -n "${!name:-}" || fail "missing $name"
  done
  for name in AF_HUMAN_DATA_ROOT AF_HUMAN_CONTENT_MANIFEST; do
    value=${!name}
    [[ "$value" = /* && "$value" != / ]] || fail "$name must be an exact absolute path"
  done
  [[ "$AF_HUMAN_CONTENT_SHA" =~ ^[0-9a-f]{64}$ ]] || fail 'invalid human content SHA'
  test -d "$AF_HUMAN_DATA_ROOT" || fail 'frozen human corpus absent'
  test -f "$AF_HUMAN_CONTENT_MANIFEST" || fail 'frozen human content manifest absent'
  test "$(sha256sum "$AF_HUMAN_CONTENT_MANIFEST" | cut -d' ' -f1)" = "$AF_HUMAN_CONTENT_SHA" || fail 'human content SHA mismatch'
  export AF_COTRAIN_NORM=$AF_YAM_NORM_JSON
fi
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
if test "$AF_YAM_PHASE" != norm; then
  test -f "$AF_YAM_NORM_JSON" || fail 'train-only normalizer absent'
fi
test "$(sha256sum "$AF_YAM_SPLIT_MANIFEST" | cut -d' ' -f1)" = "$AF_YAM_SPLIT_SHA" || fail 'split SHA mismatch'
test "$(sha256sum "$AF_YAM_CONTENT_SUMMARY" | cut -d' ' -f1)" = "$AF_YAM_CONTENT_SUMMARY_SHA" || fail 'content receipt SHA mismatch'
if test "$AF_YAM_PHASE" != norm; then
  test "$(sha256sum "$AF_YAM_NORM_JSON" | cut -d' ' -f1)" = "$AF_YAM_NORM_SHA" || fail 'normalizer SHA mismatch'
fi
test "$(git -C "$AF_YAM_REPO" rev-parse HEAD)" = "$AF_YAM_SOURCE_COMMIT" || fail 'source commit mismatch'
test -z "$(git -C "$AF_YAM_REPO" status --porcelain)" || fail 'source checkout is dirty'

if test "$AF_YAM_RECIPE" = yam_human_keypoints; then
  validator=$AF_YAM_REPO/scripts/train/verify_yam_human_action_flow_launch.py
else
  validator=$AF_YAM_REPO/scripts/train/verify_yam_action_flow_launch.py
fi
test -f "$validator" || fail 'reviewed YAM launch validator absent'
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$AF_YAM_REPO:$AF_YAM_REPO/tools
container=(
  --container-image="$AF_YAM_CONTAINER_IMAGE"
  --container-writable
  --container-mounts=/workspace:/workspace:rw
  --container-workdir="$AF_YAM_REPO"
)

if test "$AF_YAM_RECIPE" = yam_human_keypoints; then
  validator_args=(
    --split "$AF_YAM_SPLIT_MANIFEST" --split-sha "$AF_YAM_SPLIT_SHA"
    --yam-content-summary "$AF_YAM_CONTENT_SUMMARY" --yam-content-sha "$AF_YAM_CONTENT_SUMMARY_SHA"
    --human-content-manifest "$AF_HUMAN_CONTENT_MANIFEST" --human-content-sha "$AF_HUMAN_CONTENT_SHA"
  )
  if test -n "${AF_YAM_CORPUS_CONTRACT:-}${AF_YAM_CORPUS_CONTRACT_SHA:-}"; then
    test -n "${AF_YAM_CORPUS_CONTRACT:-}" && test -n "${AF_YAM_CORPUS_CONTRACT_SHA:-}" || fail 'complete corpus contract path/SHA pair required'
    [[ "$AF_YAM_CORPUS_CONTRACT" = /* && "$AF_YAM_CORPUS_CONTRACT" != / ]] || fail 'corpus contract must be absolute'
    [[ "$AF_YAM_CORPUS_CONTRACT_SHA" =~ ^[0-9a-f]{64}$ ]] || fail 'corpus contract SHA invalid'
    test "$(sha256sum "$AF_YAM_CORPUS_CONTRACT" | cut -d' ' -f1)" = "$AF_YAM_CORPUS_CONTRACT_SHA" || fail 'corpus contract SHA changed'
    validator_args+=(--corpus-contract "$AF_YAM_CORPUS_CONTRACT" --corpus-contract-sha "$AF_YAM_CORPUS_CONTRACT_SHA")
  fi
  identities=$(srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" "$validator" "${validator_args[@]}")
  corpus_counts=$(srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" "$validator" "${validator_args[@]}" --emit-corpus-counts)
  [[ "$corpus_counts" =~ ^[0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+\ [0-9]+$ ]] || fail 'invalid derived corpus counts'
  read -r yam_train_count yam_valid_count yam_train_windows yam_valid_windows human_train_count human_valid_count human_train_windows human_valid_windows <<< "$corpus_counts"
  read -r train_sha valid_sha human_train_sha human_valid_sha content_sha <<< "$identities"
  export HUMAN_TRAIN_IDS_SHA256=$human_train_sha
  export HUMAN_VALID_IDS_SHA256=$human_valid_sha
else
  validator_args=(
    --split "$AF_YAM_SPLIT_MANIFEST" --split-sha "$AF_YAM_SPLIT_SHA"
    --content-summary "$AF_YAM_CONTENT_SUMMARY" --content-summary-sha "$AF_YAM_CONTENT_SUMMARY_SHA"
  )
  identities=$(srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" "$validator" "${validator_args[@]}")
  read -r train_sha valid_sha content_sha <<< "$identities"
fi
for value in $identities; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail 'derived corpus identity invalid'
done
export YAM_TRAIN_IDS_SHA256=$train_sha
export YAM_VALID_IDS_SHA256=$valid_sha

source "$AF_YAM_REPO/scripts/train/yam_action_flow_arguments.sh"
yam_action_flow_arguments

resolved=$AF_YAM_PREFLIGHT_DIR/resolved-$AF_YAM_PHASE.yaml
receipt=$AF_YAM_PREFLIGHT_DIR/receipt-$AF_YAM_PHASE.json
if test "$AF_YAM_MODE" = preflight; then
  test ! -e "$resolved" && test ! -e "$receipt" || fail 'preflight output exists'
  mkdir -p "$AF_YAM_PREFLIGHT_DIR"
  srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" -m egomimic.trainHydra \
    --cfg job --resolve "${overrides[@]}" > "$resolved"
  config_verify_args=(
    --config "$resolved" --receipt "$receipt" --phase "$AF_YAM_PHASE" \
    --source-commit "$AF_YAM_SOURCE_COMMIT" --data-root "$AF_YAM_DATA_ROOT" \
    --norm-json "$AF_YAM_NORM_JSON" --norm-sha "$AF_YAM_NORM_SHA" \
    --wandb-entity "$AF_YAM_WANDB_ENTITY" --wandb-project "$AF_YAM_WANDB_PROJECT" \
    --wandb-id "$AF_YAM_WANDB_ID" --run-dir "$AF_YAM_RUN_DIR"
  )
  if test "$AF_YAM_RECIPE" = yam_human_keypoints; then
    config_verify_args=(
      --config "$resolved" --receipt "$receipt" --phase "$AF_YAM_PHASE"
      --source-commit "$AF_YAM_SOURCE_COMMIT" --yam-root "$AF_YAM_DATA_ROOT"
      --human-root "$AF_HUMAN_DATA_ROOT" --norm-json "$AF_YAM_NORM_JSON"
      --norm-sha "$AF_YAM_NORM_SHA" --wandb-entity "$AF_YAM_WANDB_ENTITY"
      --wandb-project "$AF_YAM_WANDB_PROJECT" --wandb-id "$AF_YAM_WANDB_ID"
      --run-dir "$AF_YAM_RUN_DIR"
      --inference-method "$AF_YAM_INFERENCE_METHOD" --checkpoint-policy "$AF_YAM_CHECKPOINT_POLICY"
    )
    config_verify_args+=(--augmentation "$AF_YAM_AUGMENTATION")
    if test "$AF_YAM_AUGMENTATION" = human_speed_v1; then
      config_verify_args+=(--speed-reference "$AF_STATIONARY_SPEED_REFERENCE" --speed-reference-receipt "$AF_STATIONARY_SPEED_REFERENCE_RECEIPT" --speed-reference-receipt-sha "$AF_STATIONARY_SPEED_REFERENCE_RECEIPT_SHA")
    fi
    if test "$AF_YAM_PHASE" = norm; then config_verify_args+=(--norm-cache-dir "$AF_YAM_NORM_CACHE_DIR"); fi
  else
    config_verify_args+=(--accumulation "$AF_YAM_ACCUMULATION")
  fi
  srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" "$validator" \
    "${validator_args[@]}" "${config_verify_args[@]}"
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
if test "$AF_YAM_PHASE" = norm; then
  test ! -e "$AF_YAM_NORM_CACHE_DIR" || fail 'normalization output exists'
  srun --overlap --ntasks=1 "${container[@]}" "$AF_YAM_PYTHON" -m egomimic.trainHydra "${overrides[@]}"
  exit 0
fi
# Pyxis can expose all physical devices to nvidia-smi even when Slurm assigns
# one GPU. Check the CUDA view actually used by the model, and bind both child
# steps to the single assigned GPU explicitly.
srun --overlap --ntasks=1 --gpus-per-task=1 "${container[@]}" \
  "$AF_YAM_PYTHON" "$AF_YAM_REPO/scripts/ice/ice_gpu_probe.py" \
    --allowed-gpu-name H100 --allowed-gpu-name H200 \
    --output "$AF_YAM_PREFLIGHT_DIR/gpu-health-$AF_YAM_PHASE-${SLURM_JOB_ID:?}-${SLURM_RESTART_COUNT:-0}.json"
export HYDRA_FULL_ERROR=1
srun --overlap --ntasks=1 --gpus-per-task=1 "${container[@]}" "$AF_YAM_PYTHON" -m egomimic.trainHydra "${overrides[@]}"
