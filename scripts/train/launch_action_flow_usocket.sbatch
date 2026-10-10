#!/bin/bash
#SBATCH --kill-on-invalid-dep=yes
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

partition_allowed() {
  local requested=$1 selected=$2
  [[ "$requested" =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]] || return 1
  [[ "$selected" =~ ^[A-Za-z0-9_-]+$ ]] || return 1
  case ",$requested," in
    *",$selected,"*) return 0 ;;
    *) return 1 ;;
  esac
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
AF_SIGNAL_RELAY=${AF_SIGNAL_RELAY:-$AF_TOOL_DIR/relay_batch_signals.sh}
AF_DATASET_VALIDATOR=${AF_DATASET_VALIDATOR:-$AF_TOOL_DIR/validate_planar_dataset.py}
AF_CHECKPOINT_VALIDATOR=${AF_CHECKPOINT_VALIDATOR:-$AF_TOOL_DIR/validate_lightning_checkpoint.py}
AF_RUNTIME_LOCK_CAPTURE=${AF_RUNTIME_LOCK_CAPTURE:-$AF_TOOL_DIR/capture_runtime_lock.py}
AF_CONFIG_VALIDATOR=${AF_CONFIG_VALIDATOR:-$AF_REPO/tools/validate_action_flow_config.py}
AF_EXTRA_PYTHONPATH=${AF_EXTRA_PYTHONPATH:-$(dirname "$AF_CONFIG_VALIDATOR")}
AF_SMOKE_VERIFIER=${AF_SMOKE_VERIFIER:-$AF_REPO/scripts/train/verify_action_flow_training_smoke.py}
AF_SLURM_CONTRACT_VALIDATOR=${AF_SLURM_CONTRACT_VALIDATOR:-$AF_REPO/scripts/train/validate_slurm_job_contract.py}
AF_STORAGE_VALIDATOR=${AF_STORAGE_VALIDATOR:-$AF_REPO/scripts/train/check_checkpoint_storage.py}
AF_SPLIT_MANIFEST=${AF_SPLIT_MANIFEST:-}
AF_PRECISION=${AF_PRECISION:-bf16}
AF_SEED=${AF_SEED:-42}
AF_EXPECTED_GPU_CONSTRAINT=${AF_EXPECTED_GPU_CONSTRAINT:-H100|H200}
AF_STORAGE_SAFETY_RESERVE_BYTES=${AF_STORAGE_SAFETY_RESERVE_BYTES:-2147483648}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
LAUNCHER=${BASH_SOURCE[0]}
CANONICAL_CONTENT_MANIFEST_SHA256=a1c81fb0ce8967aba795383a293180f9ba08a0ecfdd6f4a878afb20b39733761
CANONICAL_DATASET_CONTENT_AGGREGATE_SHA256=80f835ad37c3d5c5b7b2d5c3e1656c307ee567a1f63f51081165bf404b8ceb52
CHAIN_CONTENT_MANIFEST_SHA256=65d22f70d75a7e8a2ddd80f5cef989b4cd31fc608b44452726a1b0e0ef1df865
CHAIN_DATASET_CONTENT_AGGREGATE_SHA256=2d75e0cc446f60b0fc7160444b5027778fd1f1155e0516b18d637a1a89399efb
CHAIN_4920_CONTENT_MANIFEST_SHA256=d47337ec051351c37def6078ce96a9d4187d44ca6d2d89867e716aea8b5573a9
CHAIN_4920_DATASET_CONTENT_AGGREGATE_SHA256=241ef7cab22303088c46ef1dca4379c4c4932409b494fc88db413fcb687e1afe
CHAIN_MANUAL4919_CONTENT_MANIFEST_SHA256=9920658e35a6cffc3dd4dde82129b7d1ae08f77664f0cbb06498ebc13ca3eaa1
CHAIN_MANUAL4919_DATASET_CONTENT_AGGREGATE_SHA256=f2762688fdcfb6481f841c9939a0723426e97cb31260454d532c0312e278bb7c

case "$AF_LAUNCH_MODE" in
  preflight|run|schema) ;;
  *) die "AF_LAUNCH_MODE must be preflight or run" ;;
esac
# The portable launcher carries a ten-minute USR1 checkpoint warning for long
# GPU runs.  A CPU preflight has no requeue runner yet, so inheriting that
# warning would terminate a legitimate normalization pass before it can write
# PREFLIGHT_RESULT.json.  Ignore only the scheduler boundary signal in
# preflight mode; run mode installs the maintained relay before training.
if test "$AF_LAUNCH_MODE" = preflight; then
  trap '' USR1
fi
case "$AF_RUN_KIND" in
  smoke|full) ;;
  *) die "AF_RUN_KIND must be smoke or full" ;;
esac
if test "$AF_LAUNCH_MODE" = run; then
  required AF_EXPECTED_MEMORY
  required AF_EXPECTED_TIME_LIMIT
fi
AF_METHOD=action_flow_joint
AF_FULL_MAX_STEPS=240000
AF_FULL_VALIDATE_EVERY=10000
AF_FULL_CHECKPOINT_EVERY=40000
AF_FULL_LIMIT_VAL_BATCHES=1.0
AF_VALID_BATCH_SIZE=16
AF_SOURCE=pushshapes_sim_u_socket
AF_ACTION_DIM=4
AF_EXPECTED_PARAMETER_COUNT=97956100
AF_COTRAIN=false
AF_SCALED_H512=false
AF_FROZEN_CHAIN_4920=false
AF_PAIRED_CHAIN_MANUAL4919=false
AF_SPEED_DIAGNOSTIC=false
AF_H696_EULER=false
AF_NATIVE_LIBERO=false
AF_NATIVE_RECIPE_BASE=$AF_EXPERIMENT
case "$AF_EXPERIMENT" in
  libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_av0_s42|libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_av0_s42|libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_av0_s42|libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_av0_s42)
    AF_NATIVE_RECIPE_BASE=${AF_EXPERIMENT/_80k_av0_/_80k_}
    ;;
esac
case "$AF_NATIVE_RECIPE_BASE" in
  pusht/action_flow_cotrain_uc_manual4919_avln_80k_s42)
    test "${AF_RESUME_TYPED_MODEL:-}" = noaug && test -n "${AF_INITIAL_CHECKPOINT:-}" || die "noaug route is checkpoint continuation only"
    test "$AF_EXPECTED_HEAD" = fb397446056a73ebcc2cf546217a6284d582727a || die "H696 Euler source mismatch"
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_cotrain_uc_manual4919_avln_80k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=80000
    AF_FULL_VALIDATE_EVERY=15000
    AF_FULL_CHECKPOINT_EVERY=5000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_COTRAIN=true
    AF_H696_EULER=true
    required AF_SAMPLER_AUDITOR
    required AF_SAMPLER_AUDITOR_SHA256
    test "$(sha256 "$AF_SAMPLER_AUDITOR")" = "$AF_SAMPLER_AUDITOR_SHA256" || die "sampler auditor hash mismatch"
    AF_SECOND_SOURCE=pushshapes_sim_chain_gripper
    AF_EXPECTED_PARAMETER_COUNT=471545114
    AF_EXPECTED_SMOKE_PARAMETER_COUNT=471545114
    AF_SMOKE_VERIFIER=${AF_H696_SMOKE_VERIFIER:?}
    test "$(sha256 "$AF_SMOKE_VERIFIER")" = "${AF_H696_SMOKE_VERIFIER_SHA256:?}" || die "H696 smoke verifier SHA mismatch"
    AF_WANDB_TAGS="[pushshapes,u-socket,chain-gripper,manual4919,cotrain,h696,euler50,bf16,fresh,single-gpu,$AF_RUN_KIND]"
    ;;
  libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42|libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_s42|libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_s42|libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_s42)
    AF_NATIVE_LIBERO=true
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=${AF_EXPERIMENT#libero_historical/}
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=80000
    AF_FULL_VALIDATE_EVERY=15000
    AF_FULL_CHECKPOINT_EVERY=5000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_SOURCE=libero_panda
    AF_ACTION_DIM=7
    AF_EXPECTED_PARAMETER_COUNT=39750391
    # Native downstream dataset/preflight/smoke adapters are required before run.
    if test "$AF_LAUNCH_MODE" != schema; then
      if test "$AF_LAUNCH_MODE" = preflight; then AF_NATIVE_PHASE=preflight; else AF_NATIVE_PHASE=$AF_RUN_KIND; fi
      AF_NATIVE_SCHEMA_DRIVER=$AF_REPO/scripts/train/libero_native_schema_contract.py
    fi
    for variable in AF_NATIVE_SCHEMA_DRIVER AF_NATIVE_SCHEMA_DRIVER_SHA256 \
      AF_NATIVE_BINDING_OVERRIDE AF_NATIVE_BINDING_OVERRIDE_SHA256 \
      AF_NATIVE_DATASET_VALIDATOR AF_NATIVE_CONFIG_VALIDATOR AF_NATIVE_SMOKE_VERIFIER \
      AF_NATIVE_DATASET_SHA256 AF_NATIVE_HISTORICAL_SPLIT_SHA256 \
      AF_SPLIT_MANIFEST AF_NORM_STATS_PATH AF_EXPECTED_NORM_SHA256 AF_NATIVE_PHASE; do
      required "$variable"
    done
    case "$AF_NATIVE_PHASE" in preflight|smoke|full) ;; *) die "invalid native phase" ;; esac
    case "$AF_NATIVE_RECIPE_BASE:$AF_SEED" in
      libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_s42:42|libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_s42:43|libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_s42:42|libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_s42:43|libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42:42|libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_s42:42) ;;
      *) die "unsupported native suite/training seed" ;;
    esac
    AF_DATASET_VALIDATOR=$AF_NATIVE_DATASET_VALIDATOR
    AF_CONFIG_VALIDATOR=$AF_NATIVE_CONFIG_VALIDATOR
    test "$AF_NATIVE_SMOKE_VERIFIER" = "$AF_REPO/scripts/train/verify_action_flow_training_smoke.py" || die "native smoke requires the maintained shared CLI typed dispatcher"
    AF_SMOKE_VERIFIER=$AF_NATIVE_SMOKE_VERIFIER
    ;;

  pusht/action_flow_cotrain_uc_speed_interpolation)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_cotrain_uc_speed_interpolation
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=80000
    AF_FULL_VALIDATE_EVERY=15000
    AF_FULL_CHECKPOINT_EVERY=5000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_COTRAIN=true
    AF_SPEED_DIAGNOSTIC=true
    AF_SECOND_SOURCE=pushshapes_sim_chain_gripper
    required AF_SPEED_ENCODING
    required AF_SPEED_REFERENCE
    case "$AF_SPEED_ENCODING" in
      scalar) AF_EXPECTED_PARAMETER_COUNT=471549402 ;;
      fourier) AF_EXPECTED_PARAMETER_COUNT=471549530 ;;
      *) die "speed encoding must be scalar or fourier" ;;
    esac
    test "$AF_SPEED_REFERENCE" = 84.11600368466028 || die "unapproved speed reference"
    required AF_SPEED_CONFIG_VALIDATOR
    required AF_SPEED_SMOKE_VERIFIER
    required AF_SPEED_CONFIG_VALIDATOR_SHA256
    required AF_SPEED_SMOKE_VERIFIER_SHA256
    test "$AF_EXPECTED_HEAD" = 344dd3baf5ba942d947e0a0e136c8990025d91e5 || die "unapproved speed diagnostic source"
    test "$(sha256 "$AF_SPEED_CONFIG_VALIDATOR")" = "$AF_SPEED_CONFIG_VALIDATOR_SHA256" || die "speed config validator hash mismatch"
    test "$(sha256 "$AF_SPEED_SMOKE_VERIFIER")" = "$AF_SPEED_SMOKE_VERIFIER_SHA256" || die "speed smoke verifier hash mismatch"
    AF_CONFIG_VALIDATOR=$AF_SPEED_CONFIG_VALIDATOR
    AF_SMOKE_VERIFIER=$AF_SPEED_SMOKE_VERIFIER
    export AF_SPEED_ENCODING AF_SPEED_REFERENCE
    AF_WANDB_TAGS="[pushshapes,cotrain,speed-$AF_SPEED_ENCODING,NON-PROTOCOL_INTERPOLATION_ONLY_DIAGNOSTIC,h696d12h12,bf16,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_usocket_refactored_c12_compat_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_refactored_c12_compat_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=30000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=5000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    ;;
  pusht/action_flow_usocket_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_EXPECTED_PARAMETER_COUNT=190208924
    AF_SCALED_H512=true
    AF_WANDB_TAGS="[pushshapes,u-socket,action-flow,unite-h512d14h16,sum14,cfg4,val8x32,h16,z8x16,k14,dopri5,muon-adamw,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_chain_points6_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_chain_points6_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_SOURCE=pushshapes_sim_chain_gripper
    AF_ACTION_DIM=6
    AF_EXPECTED_PARAMETER_COUNT=190210206
    AF_SCALED_H512=true
    AF_FROZEN_CHAIN_4920=true
    AF_WANDB_TAGS="[pushshapes,chain-gripper,points6,frozen4920,action-flow,unite-h512d14h16,sum14,cfg4,val8x32,h16,z8x16,k14,dopri5,muon-adamw,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_cotrain_uc_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_cotrain_uc_latent_fm_sg_unite_h512d14h16_sum14_cfg4_val10k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_EXPECTED_PARAMETER_COUNT=301792030
    AF_COTRAIN=true
    AF_SCALED_H512=true
    AF_FROZEN_CHAIN_4920=true
    AF_SECOND_SOURCE=pushshapes_sim_chain_gripper
    AF_WANDB_TAGS="[pushshapes,u-socket,chain-gripper,points6,frozen4920,cotrain,action-flow,private-codecs,shared-field,unite-h512d14h16,h16,z8x16,k14,dopri5,cfg4,muon-adamw,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_cotrain_uc_latent_fm_sg_unite_h384d12h12_sum14_cfg4_val10k_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_cotrain_uc_latent_fm_sg_unite_h384d12h12_sum14_cfg4_val10k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_COTRAIN=true
    AF_SECOND_SOURCE=pushshapes_sim_chain_gripper
    AF_WANDB_TAGS="[pushshapes,u-socket,chain-gripper,frozen4920,cotrain,action-flow,private-codecs,shared-field,h384d12h12,h16,z8x16,k14,dopri5,cfg4,muon-adamw,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_chain_native4_points6_manual4919_h384_150k_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_chain_native4_points6_manual4919_h384_150k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=15000
    AF_FULL_CHECKPOINT_EVERY=5000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_SOURCE=pushshapes_sim_chain_gripper_native4
    AF_SECOND_SOURCE=pushshapes_sim_chain_gripper
    AF_ACTION_DIM=4
    AF_EXPECTED_PARAMETER_COUNT=86539098
    AF_EXPECTED_SMOKE_PARAMETER_COUNT=151990970
    AF_COTRAIN=true
    AF_PAIRED_CHAIN_MANUAL4919=true
    AF_WANDB_TAGS="[pushshapes,chain-gripper,native4,points6,manual4919,cotrain,action-flow,private-codecs,shared-field,h384d12h12,h16,z8x16,k14,dopri5,cfg4,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_cotrain_usocket_chain_obstacle_unite_arch_denoiser90m_sum14_adamw_lr1e5_s42)
    AF_METHOD=latent_fm_stopgrad_cotrain
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_cotrain_usocket_chain_obstacle_unite_arch_denoiser90m_sum14_adamw_lr1e5_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=240000
    AF_FULL_VALIDATE_EVERY=240000
    AF_FULL_CHECKPOINT_EVERY=40000
    AF_FULL_LIMIT_VAL_BATCHES=0
    AF_VALID_BATCH_SIZE=16
    AF_COTRAIN=true
    AF_SECOND_SOURCE=pushshapes_sim_chain_gripper
    AF_WANDB_TAGS="[pushshapes,u-socket,chain-gripper,obstacle,cotrain,action-flow,route-local-codecs,shared-field,h16,z8x16,k14,adamw,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_chain_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_chain_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_SOURCE=pushshapes_sim_chain_gripper
    AF_ACTION_DIM=5
    AF_EXPECTED_PARAMETER_COUNT=97956613
    AF_WANDB_TAGS="[pushshapes,chain-gripper,action-flow,unite-h384,sum14,cfg4,val8x32,h16,z8x16,k14,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_WANDB_TAGS="[pushshapes,u-socket,action-flow,unite-h384,sum14,cfg4,val8x32,h16,z8x16,k14,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_deterministic_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_latent_fm_sg_unite_h384_sum14_cfg4_val8_deterministic_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=10000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=8
    AF_VALID_BATCH_SIZE=32
    AF_WANDB_TAGS="[pushshapes,u-socket,action-flow,unite-h384,sum14,cfg4,val8x32,h16,z8x16,k14,deterministic-data,legacy-bounds,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_usocket_latent_fm_sg_unite_denoiser90m_sum14_cfg4_finalval1_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_latent_fm_sg_unite_denoiser90m_sum14_cfg4_finalval1_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=240000
    AF_FULL_VALIDATE_EVERY=240000
    AF_FULL_CHECKPOINT_EVERY=40000
    AF_FULL_LIMIT_VAL_BATCHES=0
    AF_VALID_BATCH_SIZE=32
    AF_EXPECTED_PARAMETER_COUNT=155626932
    AF_WANDB_TAGS="[pushshapes,u-socket,action-flow,unite-denoiser90m,codec-h384,sum14,cfg4,smoke-val1x32,full-no-val,h16,z8x16,k14,no-checkpointing,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_usocket_latent_fm_sg_unite_arch_denoiser90m_sum14_cfg4_adamw_lr1e5_finalval1_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_latent_fm_sg_unite_arch_denoiser90m_sum14_cfg4_adamw_lr1e5_finalval1_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=150000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_FULL_LIMIT_VAL_BATCHES=0
    AF_VALID_BATCH_SIZE=32
    AF_EXPECTED_PARAMETER_COUNT=155626932
    AF_WANDB_TAGS="[pushshapes,u-socket,action-flow,unite-architecture-only,unite-denoiser90m,codec-h384,adamw-lr1e-5,sum14,cfg4,smoke-val1x32,full-no-val,h16,z8x16,k14,no-checkpointing,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_chain_latent_fm_sg_unite_denoiser90m_sum14_cfg4_finalval1_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_chain_latent_fm_sg_unite_denoiser90m_sum14_cfg4_finalval1_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=150000
    AF_FULL_CHECKPOINT_EVERY=20000
    AF_FULL_LIMIT_VAL_BATCHES=0
    AF_VALID_BATCH_SIZE=32
    AF_SOURCE=pushshapes_sim_chain_gripper
    AF_ACTION_DIM=5
    AF_EXPECTED_PARAMETER_COUNT=155627445
    AF_WANDB_TAGS="[pushshapes,chain-gripper,action-flow,unite-denoiser90m,codec-h384,sum14,cfg4,smoke-val1x32,full-no-val,ckpt20k,h16,z8x16,k14,no-checkpointing,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_usocket_latent_fm_sg_unite_h384_s42)
    AF_METHOD=latent_fm_stopgrad_unite
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_usocket_latent_fm_sg_unite_h384_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    AF_FULL_MAX_STEPS=150000
    AF_FULL_VALIDATE_EVERY=30000
    AF_FULL_CHECKPOINT_EVERY=30000
    AF_WANDB_TAGS="[pushshapes,u-socket,action-flow,unite-h384,h16,z8x16,k14,single-gpu,$AF_RUN_KIND]"
    ;;
  pusht/action_flow_bc_usocket_latent_fm_sg_recon1_s42)
    AF_METHOD=latent_fm_stopgrad
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_latent_fm_sg_recon1_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_latent_fm_sg_recon1_codec98k_s42)
    AF_METHOD=latent_fm_stopgrad
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_latent_fm_sg_recon1_codec98k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_latent_fm_sg_recon1_200m_muon_lr1e5_s42)
    AF_METHOD=latent_fm_stopgrad
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_latent_fm_sg_recon1_200m_muon_lr1e5_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_latent_fm_sg_recon1_200m_adamw_lr1e5_s42)
    AF_METHOD=latent_fm_stopgrad
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_latent_fm_sg_recon1_200m_adamw_lr1e5_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_bridge_likelihood_s42)
    AF_METHOD=gaussian_bridge_likelihood
    # Transport placeholders only; no FM/reconstruction flags enter this model.
    AF_RECONSTRUCTION_WEIGHT=0
    AF_FLOW_WEIGHT=0
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_bridge_likelihood_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_graph_section_s42)
    AF_METHOD=graph_section_diagnostic
    AF_RECONSTRUCTION_WEIGHT=0
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_graph_section_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_recon1_s42)
    AF_RECONSTRUCTION_WEIGHT=1
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon1_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_recon10_s42)
    AF_RECONSTRUCTION_WEIGHT=10
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon10_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  pusht/action_flow_bc_usocket_recon10_warmup10k_s42)
    AF_RECONSTRUCTION_WEIGHT=10
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon10_warmup10k_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=10000
    ;;
  pusht/action_flow_bc_usocket_recon10_warmup10k_flow001_s42)
    AF_RECONSTRUCTION_WEIGHT=10
    AF_FLOW_WEIGHT=0.01
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon10_warmup10k_flow001_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=10000
    ;;
  pusht/action_flow_bc_usocket_recon100_s42)
    AF_RECONSTRUCTION_WEIGHT=100
    AF_FLOW_WEIGHT=1
    AF_EXPECTED_CONFIG_NAME=action_flow_bc_usocket_recon100_s42
    AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS=0
    ;;
  *) die "AF_EXPERIMENT must select an explicitly approved Action Flow config" ;;
esac

if test "$AF_NATIVE_LIBERO" = true && test -z "${SLURM_STEP_ID:-}"; then
  required SLURM_JOB_ID
  required SLURM_CPUS_PER_TASK
  # Re-enter through a real CPU step; nested GPU child/probe steps overlap.
  exec srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gres=none /usr/bin/env bash "$LAUNCHER"
fi
if test "$AF_NATIVE_LIBERO" = true; then export AF_NATIVE_SCHEDULED_STEP=true; fi

if test "$AF_SCALED_H512" = true; then
  AF_SMOKE_VERIFIER=$AF_REPO/scripts/train/verify_scaled_action_flow_training_smoke.py
fi
if test "$AF_PAIRED_CHAIN_MANUAL4919" = true; then
  required AF_PAIRED_CHAIN_SMOKE_VERIFIER
  AF_SMOKE_VERIFIER=$AF_PAIRED_CHAIN_SMOKE_VERIFIER
fi

if test "$AF_COTRAIN" = true; then
  for variable in \
    AF_SECOND_DATASET_DIR AF_SECOND_SPLIT_MANIFEST \
    AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256 AF_SECOND_CONTENT_MANIFEST \
    AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256 \
    AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256; do
    required "$variable"
  done
fi
if test "$AF_PAIRED_CHAIN_MANUAL4919" = true; then
  test "$AF_COTRAIN" = true || die "paired ChainGripper requires both routed views"
  test -n "$AF_SPLIT_MANIFEST" || die "paired ChainGripper requires an explicit split manifest"
  test "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" = 1fe77d9c0320965fd973711c4b2803938bb9a59ce5e842eb18316b9e71730ecc || \
    die "paired ChainGripper split is not the pinned manual4919 split"
  test "$AF_SOURCE" = pushshapes_sim_chain_gripper_native4 || die "wrong native4 route"
  test "$AF_SECOND_SOURCE" = pushshapes_sim_chain_gripper || die "wrong points6 route"
  test "$AF_DATASET_DIR" = "$AF_SECOND_DATASET_DIR" || die "paired routes must use one physical corpus"
  test "$AF_SPLIT_MANIFEST" = "$AF_SECOND_SPLIT_MANIFEST" || die "paired routes must share one physical episode split"
  test "$AF_CONTENT_MANIFEST" = "$AF_SECOND_CONTENT_MANIFEST" || die "paired routes must share one content manifest"
  test "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" = "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256" || die "paired split SHA mismatch"
  test "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" = "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" || die "paired content SHA mismatch"
  test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" || die "paired aggregate SHA mismatch"
fi

if test -z "$AF_SPLIT_MANIFEST" && test "$AF_FROZEN_CHAIN_4920" = true \
  && test "$AF_COTRAIN" = false; then
  die "AF_SPLIT_MANIFEST is required for the frozen 4,920-episode ChainGripper row"
elif test -z "$AF_SPLIT_MANIFEST"; then
  if test "$AF_SOURCE" = pushshapes_sim_chain_gripper; then
    AF_SPLIT_MANIFEST=$AF_REPO/egomimic/hydra_configs/data/pusht/planar_v2_chain_gripper_dp_3k_split_seed42_v1.json
  else
    AF_SPLIT_MANIFEST=$AF_REPO/egomimic/hydra_configs/data/pusht/planar_v2_usocket_dp_3k_split_seed42_v1.json
  fi
fi

if test "$AF_METHOD" != action_flow_joint || [[ "$AF_EXPERIMENT" = *warmup10k* ]]; then
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
if test "$AF_NATIVE_LIBERO" = true; then
  # Suite corpus identities are immutable; destination receipt paths may differ.
  case "$AF_NATIVE_RECIPE_BASE" in
    libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42)
      native_logical_sha=3ed2b86cf97eaa27045af81e6d6c4834c8659d15eda69a2394d835e602014e1d
      native_historical_split_sha=c588c831ef0deab61a59a21840ea9e11dafe92a170d9023a87e0acee831df236 ;;
    libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_s42)
      native_logical_sha=ee28d02e8c6cd28fb0a235fab2ccc649b662e6f3e10eb096d89be082d2871860
      native_historical_split_sha=15640eada8102c7071dcfcd153ef438cb98c61835474a2fb2c3c9415cf6d714e ;;
    libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_s42)
      native_logical_sha=c190385dfc8a7d05b966a4b5f94651826b8a906b1e63d8b348edc70df52c1f60
      native_historical_split_sha=207c24115104bf0545e7f165abb91cf052eb99f6568446f11e835f9a3e4c6af8 ;;
    libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_s42)
      native_logical_sha=40b5e3ccf4febcbae470690755adc8aa6a39471a4fcb7fa585a9bdb8e59c325d
      native_historical_split_sha=5ade2a86fad7552457e093944fd60efec8b51fbe9dfd20831f627d1c56e9b5d7 ;;
    *) die "unsupported native corpus identity profile" ;;
  esac
  test "$AF_NATIVE_DATASET_SHA256" = "$native_logical_sha" || die "native dataset identity mismatch"
  test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = "$native_logical_sha" || die "native dataset aggregate identity mismatch"
  test "$AF_NATIVE_HISTORICAL_SPLIT_SHA256" = "$native_historical_split_sha" || die "historical split identity mismatch"
elif test "$AF_PAIRED_CHAIN_MANUAL4919" = true; then
  test "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" = "$CHAIN_MANUAL4919_CONTENT_MANIFEST_SHA256" || \
    die "content manifest is not the paired manual4919 ChainGripper manifest"
  test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = "$CHAIN_MANUAL4919_DATASET_CONTENT_AGGREGATE_SHA256" || \
    die "dataset aggregate is not the paired manual4919 ChainGripper content"
elif test "$AF_SOURCE" = pushshapes_sim_chain_gripper; then
  if test "$AF_FROZEN_CHAIN_4920" = true; then
    test "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" = "$CHAIN_4920_CONTENT_MANIFEST_SHA256" || \
      die "content manifest is not the frozen 4,920-episode ChainGripper manifest"
    test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = "$CHAIN_4920_DATASET_CONTENT_AGGREGATE_SHA256" || \
      die "dataset aggregate is not the frozen 4,920-episode ChainGripper content"
  else
    test "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" = "$CHAIN_CONTENT_MANIFEST_SHA256" || \
      die "content manifest is not the canonical ChainGripper corpus manifest"
    test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = "$CHAIN_DATASET_CONTENT_AGGREGATE_SHA256" || \
      die "dataset aggregate is not the canonical ChainGripper corpus content"
  fi
else
  test "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" = "$CANONICAL_CONTENT_MANIFEST_SHA256" || \
    die "content manifest is not the canonical USocket corpus manifest"
  test "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" = "$CANONICAL_DATASET_CONTENT_AGGREGATE_SHA256" || \
    die "dataset aggregate is not the canonical USocket corpus content"
fi
if test "$AF_COTRAIN" = true; then
  [[ "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256 must be a lowercase SHA-256"
  [[ "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256 must be a lowercase SHA-256"
  [[ "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
    die "AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256 must be a lowercase SHA-256"
  if test "$AF_PAIRED_CHAIN_MANUAL4919" = true || test "$AF_SPEED_DIAGNOSTIC" = true || test "$AF_H696_EULER" = true; then
    test "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" = "$CHAIN_MANUAL4919_CONTENT_MANIFEST_SHA256" || \
      die "second content manifest is not the paired manual4919 ChainGripper manifest"
    test "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" = "$CHAIN_MANUAL4919_DATASET_CONTENT_AGGREGATE_SHA256" || \
      die "second dataset aggregate is not the paired manual4919 ChainGripper content"
  elif test "$AF_FROZEN_CHAIN_4920" = true; then
    test "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" = "$CHAIN_4920_CONTENT_MANIFEST_SHA256" || \
      die "second content manifest is not the frozen 4,920-episode ChainGripper manifest"
    test "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" = "$CHAIN_4920_DATASET_CONTENT_AGGREGATE_SHA256" || \
      die "second dataset aggregate is not the frozen 4,920-episode ChainGripper content"
  else
    test "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" = "$CHAIN_CONTENT_MANIFEST_SHA256" || \
      die "second content manifest is not the canonical ChainGripper corpus manifest"
    test "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" = "$CHAIN_DATASET_CONTENT_AGGREGATE_SHA256" || \
      die "second dataset aggregate is not the canonical ChainGripper corpus content"
  fi
fi
[[ "$AF_WANDB_RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe W&B run ID"
[[ "$AF_CLUSTER_LABEL" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe cluster label"
if test "$AF_NATIVE_LIBERO" = true && test "$AF_LAUNCH_MODE" = run && test "$AF_RUN_KIND" = smoke; then
  test "$AF_MAX_RESTARTS" = 0 || die "native smoke requires AF_MAX_RESTARTS=0 (no requeue)"
else
  [[ "$AF_MAX_RESTARTS" =~ ^[1-9][0-9]*$ ]] || die "invalid AF_MAX_RESTARTS"
fi
[[ "$AF_SEED" =~ ^[0-9]+$ ]] || die "invalid AF_SEED"
[[ "$AF_STORAGE_SAFETY_RESERVE_BYTES" =~ ^[1-9][0-9]*$ ]] || \
  die "invalid AF_STORAGE_SAFETY_RESERVE_BYTES"
[[ "$RESTART_COUNT" =~ ^[0-9]+$ ]] || die "invalid SLURM_RESTART_COUNT"
test "$RESTART_COUNT" -le "$AF_MAX_RESTARTS" || die "restart limit exceeded"
if test "$AF_NATIVE_LIBERO" = true; then
  test "$AF_PRECISION" = bf16-mixed || die "native BF16-mixed required"
else
  test "$AF_PRECISION" = bf16 || die "AF_PRECISION must be bf16"
fi
if test "$AF_NATIVE_LIBERO" = true && test "$AF_LAUNCH_MODE" = run && test "$AF_EXPECTED_GPU_CONSTRAINT" = "(null)"; then
  # Native Lambda GPU identity is established by actual allocation + H100 BF16
  # probe in the maintained phase adapter; this cluster has no Features value.
  test "$AF_EXPECTED_GPU_CONSTRAINT" = '(null)' || die "native GPU allocation requires AF_EXPECTED_GPU_CONSTRAINT=(null)"
else
  case "$AF_EXPECTED_GPU_CONSTRAINT" in
    H100|H200|'H100|H200') ;;
    *) die "AF_EXPECTED_GPU_CONSTRAINT must be H100, H200, or H100|H200" ;;
  esac
fi
if test "$AF_LAUNCH_MODE" = run && test "$AF_RUN_KIND" = smoke \
  && test "$RESTART_COUNT" -ne 0; then
  die "a preempted two-step smoke must be relaunched with a fresh immutable identity"
fi

for binding in \
  "AF_REPO:$AF_REPO" "AF_PYTHON:$AF_PYTHON" \
  "AF_DATASET_DIR:$AF_DATASET_DIR" "AF_OUTPUT_DIR:$AF_OUTPUT_DIR" \
  "AF_TOOL_DIR:$AF_TOOL_DIR" "AF_GPU_PROBE:$AF_GPU_PROBE" \
  "AF_REQUEUE_RUNNER:$AF_REQUEUE_RUNNER" \
  "AF_SIGNAL_RELAY:$AF_SIGNAL_RELAY" \
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
if test "$AF_COTRAIN" = true; then
  for binding in \
    "AF_SECOND_DATASET_DIR:$AF_SECOND_DATASET_DIR" \
    "AF_SECOND_SPLIT_MANIFEST:$AF_SECOND_SPLIT_MANIFEST" \
    "AF_SECOND_CONTENT_MANIFEST:$AF_SECOND_CONTENT_MANIFEST"; do
    absolute_path "${binding%%:*}" "${binding#*:}"
  done
fi

test -d "$AF_REPO" || die "AF_REPO is unavailable"
test -x "$AF_PYTHON" || die "AF_PYTHON is not executable"
test -d "$AF_DATASET_DIR" || die "AF_DATASET_DIR is unavailable"
test -d "$AF_TOOL_DIR" || die "AF_TOOL_DIR is unavailable"
test -s "$AF_GPU_PROBE" || die "GPU probe is unavailable"
test -s "$AF_REQUEUE_RUNNER" || die "requeue runner is unavailable"
test -s "$AF_SIGNAL_RELAY" || die "signal relay is unavailable"
test -s "$AF_DATASET_VALIDATOR" || die "dataset validator is unavailable"
test -x "$AF_CHECKPOINT_VALIDATOR" || die "checkpoint validator is not executable"
test -s "$AF_RUNTIME_LOCK_CAPTURE" || die "runtime-lock capture is unavailable"
test -s "$AF_CONFIG_VALIDATOR" || die "Action Flow config validator is unavailable"
if test "$AF_NATIVE_LIBERO" = true && test "$AF_LAUNCH_MODE" = schema; then
  # Read-only composition does not consume a smoke verifier; keep its exact
  # future path recorded and leave the operational gate explicitly pending.
  export AF_NATIVE_OPERATIONAL_VERIFIER_PENDING=true
else
  test -s "$AF_SMOKE_VERIFIER" || die "Action Flow smoke verifier is unavailable"
fi
test -s "$AF_SLURM_CONTRACT_VALIDATOR" || \
  die "Slurm contract validator is unavailable"
test -s "$AF_STORAGE_VALIDATOR" || die "storage validator is unavailable"
test -s "$AF_CONTENT_MANIFEST" || die "dataset content manifest is unavailable"
test -s "$AF_SPLIT_MANIFEST" || die "split manifest is unavailable"
if test "$AF_COTRAIN" = true; then
  test -d "$AF_SECOND_DATASET_DIR" || die "second dataset is unavailable"
  test -s "$AF_SECOND_SPLIT_MANIFEST" || die "second split manifest is unavailable"
  test -s "$AF_SECOND_CONTENT_MANIFEST" || die "second content manifest is unavailable"
  test "$(sha256 "$AF_SECOND_SPLIT_MANIFEST")" = \
    "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256" || die "second split manifest SHA-256 mismatch"
  test "$(sha256 "$AF_SECOND_CONTENT_MANIFEST")" = \
    "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" || die "second content manifest SHA-256 mismatch"
  if test -n "${AF_CACHED_SECOND_DATASET_VALIDATION:-}"; then
    required AF_EXPECTED_CACHED_SECOND_DATASET_VALIDATION_SHA256
    absolute_path AF_CACHED_SECOND_DATASET_VALIDATION "$AF_CACHED_SECOND_DATASET_VALIDATION"
    test -s "$AF_CACHED_SECOND_DATASET_VALIDATION" || \
      die "cached second dataset validation is unavailable"
    test "$(sha256 "$AF_CACHED_SECOND_DATASET_VALIDATION")" = \
      "$AF_EXPECTED_CACHED_SECOND_DATASET_VALIDATION_SHA256" || \
      die "cached second dataset validation SHA-256 mismatch"
  fi
fi
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
export PYTHONPATH="$AF_REPO:${AF_RESUME_HELPER_DIR:+$AF_RESUME_HELPER_DIR:}$(dirname "$AF_CONFIG_VALIDATOR")${AF_EXTRA_PYTHONPATH:+:$AF_EXTRA_PYTHONPATH}"
export AF_CONFIG_VALIDATOR
export AF_EXTRA_PYTHONPATH
export AF_REPO AF_PYTHON AF_EXPERIMENT AF_OUTPUT_DIR AF_DATASET_DIR
export AF_WANDB_ENTITY AF_WANDB_PROJECT AF_WANDB_RUN_ID AF_WANDB_NAME
export AF_WANDB_GROUP AF_WANDB_TAGS
export ICE_REQUIRE_LR_SCHEDULERS=1
export ICE_EXPECTED_SOURCE_COMMIT=$AF_EXPECTED_HEAD
export ICE_EXPECTED_RUN_ID=$AF_WANDB_RUN_ID
export ICE_EXPECTED_SPLIT_SHA256=$AF_EXPECTED_SPLIT_MANIFEST_SHA256
if test "$AF_NATIVE_LIBERO" = true; then
  case "$AF_NATIVE_RECIPE_BASE" in
    libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42) export LIBERO10_REPLAY_ROOT=$AF_DATASET_DIR ;;
    libero_historical/action_flow_libero_spatial_h240_euler50_dithalf_80k_s42) export LIBERO_SPATIAL_REPLAY_ROOT=$AF_DATASET_DIR ;;
    libero_historical/action_flow_libero_goal_h240_euler50_dithalf_80k_s42) export LIBERO_GOAL_REPLAY_ROOT=$AF_DATASET_DIR ;;
    libero_historical/action_flow_libero_object_h240_euler50_dithalf_80k_s42) export LIBERO_OBJECT_REPLAY_ROOT=$AF_DATASET_DIR ;;
    *) die "unsupported native replay environment" ;;
  esac
elif test "$AF_SOURCE" = pushshapes_sim_u_socket; then
  export PUSHSHAPES_USOCKET_ROOT=$AF_DATASET_DIR
else
  export PUSHSHAPES_CHAIN_GRIPPER_ROOT=$AF_DATASET_DIR
fi
if test "$AF_COTRAIN" = true; then
  export PUSHSHAPES_CHAIN_GRIPPER_ROOT=$AF_SECOND_DATASET_DIR
fi
if test -n "${AF_EXPECTED_NORM_SHA256:-}"; then
  export ICE_EXPECTED_NORMALIZATION_SHA256=$AF_EXPECTED_NORM_SHA256
fi

if test "$AF_NATIVE_LIBERO" = true; then
  if test "$AF_NATIVE_PHASE" = smoke; then AF_RUN_KIND=smoke; else AF_RUN_KIND=full; fi
fi
if test "$AF_RUN_KIND" = smoke; then
  MAX_STEPS=2
  # Released UNITE initializes both DiT output projections to exactly zero and
  # the first warmup optimizer update has LR=0. Validate after the second real
  # update so cosine diagnostics observe a learned, non-degenerate latent.
  VALIDATE_EVERY=2
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
  MAX_STEPS=$AF_FULL_MAX_STEPS
  VALIDATE_EVERY=$AF_FULL_VALIDATE_EVERY
  LIMIT_TRAIN_BATCHES=1.0
  LIMIT_VAL_BATCHES=$AF_FULL_LIMIT_VAL_BATCHES
  CHECKPOINT_EVERY=$AF_FULL_CHECKPOINT_EVERY
  LOG_EVERY=10
  TELEMETRY_EVERY=100
  RECONSTRUCTION_ONLY_WARMUP_STEPS=$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS
fi

if test -n "${AF_INITIAL_CHECKPOINT:-}"; then
  required AF_RESUME_DRIVER_REPO
  required AF_RESUME_CPU_PROOF
  required AF_RESUME_CPU_PROOF_SHA256
  required AF_RESUME_START_STEP
  required AF_INITIAL_CHECKPOINT_SHA256
  required AF_PARENT_WANDB_RUN_ID
  test "$(sha256 "$AF_RESUME_CPU_PROOF")" = "$AF_RESUME_CPU_PROOF_SHA256" || die "resume CPU proof SHA mismatch"
  test "$(sha256 "$AF_INITIAL_CHECKPOINT")" = "$AF_INITIAL_CHECKPOINT_SHA256" || die "initial checkpoint SHA mismatch"
  if test "$AF_RUN_KIND" = smoke; then
    MAX_STEPS=$((AF_RESUME_START_STEP + 4))
    VALIDATE_EVERY=4
    LIMIT_TRAIN_BATCHES=1.0
  fi
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
  "data.train_datasets.$AF_SOURCE.resolver.folder_path=$AF_DATASET_DIR"
  "data.valid_datasets.$AF_SOURCE.resolver.folder_path=$AF_DATASET_DIR"
  "data.train_dataloader_params.$AF_SOURCE.batch_size=32"
  "data.valid_dataloader_params.$AF_SOURCE.batch_size=$AF_VALID_BATCH_SIZE"
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
  "++run_provenance.split_manifest_path=$AF_SPLIT_MANIFEST"
  "++run_provenance.split_manifest_sha256=$AF_EXPECTED_SPLIT_MANIFEST_SHA256"
  "++run_provenance.content_manifest_path=$AF_CONTENT_MANIFEST"
  "++run_provenance.content_manifest_sha256=$AF_EXPECTED_CONTENT_MANIFEST_SHA256"
  "++run_provenance.dataset_content_aggregate_sha256=$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256"
)

if test "$AF_NATIVE_LIBERO" = true; then
  BASE_OVERRIDES+=(
    evaluator.energy_sample_count=32
    evaluator.energy_seed_bank_sha256=88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6
    "++run_provenance.dataset_sha256=$AF_NATIVE_DATASET_SHA256"
    "++run_provenance.historical_split_manifest_sha256=$AF_NATIVE_HISTORICAL_SPLIT_SHA256"
  )
else
  BASE_OVERRIDES+=(
    evaluator.energy_score_validation_view.world_size=1
    "evaluator.energy_score_validation_view.per_rank_batch_size=$AF_VALID_BATCH_SIZE"
  )
fi

if test "$AF_EXPERIMENT" = pusht/action_flow_cotrain_uc_multiplier_interpolation; then
  BASE_OVERRIDES+=(model.pipeline.stages.6.inference_method=euler model.pipeline.stages.6.num_inference_steps=50 ++run_provenance.validation_inference_sampler=euler ++run_provenance.validation_inference_steps=50)
fi
if test "$AF_SPEED_DIAGNOSTIC" = true; then
  # Full-corpus retimed normalization retains shared-memory batch buffers.
  # The gateway's inherited 1024 descriptor limit failed before completion.
  ulimit -Sn 131072 || die "speed diagnostic requires RLIMIT_NOFILE >=131072"
  test "$(ulimit -Sn)" -ge 131072 || die "open-file limit guard failed"
  printf '[speed-contract] open_file_soft_limit=%s\n' "$(ulimit -Sn)"
  BASE_OVERRIDES+=(
    "speed_diagnostic.encoding=$AF_SPEED_ENCODING"
    $([[ "$AF_EXPERIMENT" = pusht/action_flow_cotrain_uc_speed_interpolation ]] && printf "%s" "speed_diagnostic.speed_reference=$AF_SPEED_REFERENCE" || true)
    "run_provenance.architecture.family=action_flow_private_codecs_shared_field_h696d12h12"
    "run_provenance.architecture.tokenizer=two_private_UniteDiTBackbone_h696_d12_h12"
    "run_provenance.architecture.denoiser=one_shared_UniteDiTBackbone_h696_d12_h12"
    "run_provenance.architecture.decoder=two_private_UniteActionDecoder_h696_d12_h12"
    "run_provenance.validation.interval_optimizer_steps=$VALIDATE_EVERY"
    "run_provenance.validation.limit_batches=$LIMIT_VAL_BATCHES"
  )
fi

# This paired-Chain experiment inherits the U-Socket baseline. Remove the
# inactive U-Socket planar dimension explicitly and make the exact config
# validator require only the two ChainGripper routes.
if test "$AF_PAIRED_CHAIN_MANUAL4919" = true; then
  BASE_OVERRIDES+=( '~planar.action_dims.pushshapes_sim_u_socket' )
fi

if test "$AF_COTRAIN" = true; then
  BASE_OVERRIDES+=(
    "data.train_datasets.$AF_SECOND_SOURCE.resolver.folder_path=$AF_SECOND_DATASET_DIR"
    "data.valid_datasets.$AF_SECOND_SOURCE.resolver.folder_path=$AF_SECOND_DATASET_DIR"
    "data.train_dataloader_params.$AF_SECOND_SOURCE.batch_size=32"
    "data.valid_dataloader_params.$AF_SECOND_SOURCE.batch_size=$AF_VALID_BATCH_SIZE"
    "++run_provenance.chain_split_manifest_path=$AF_SECOND_SPLIT_MANIFEST"
    "++run_provenance.chain_split_manifest_sha256=$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256"
    "++run_provenance.content_manifests.$AF_SOURCE.path=$AF_CONTENT_MANIFEST"
    "++run_provenance.content_manifests.$AF_SOURCE.manifest_sha256=$AF_EXPECTED_CONTENT_MANIFEST_SHA256"
    "++run_provenance.content_manifests.$AF_SOURCE.aggregate_sha256=$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256"
    "++run_provenance.content_manifests.$AF_SECOND_SOURCE.path=$AF_SECOND_CONTENT_MANIFEST"
    "++run_provenance.content_manifests.$AF_SECOND_SOURCE.manifest_sha256=$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256"
    "++run_provenance.content_manifests.$AF_SECOND_SOURCE.aggregate_sha256=$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256"
  )
fi
if test "$AF_SCALED_H512" = true; then
  BASE_OVERRIDES+=(
    'evaluator.action_flow_diagnostics.activation_layer_map={0:0,1:13}'
  )
fi

if test "$AF_METHOD" != gaussian_bridge_likelihood && test "$AF_NATIVE_LIBERO" = false; then
  BASE_OVERRIDES+=(
    ++model.reconstruction_only_warmup_steps="$RECONSTRUCTION_ONLY_WARMUP_STEPS"
    "++run_provenance.objective.requested_full_reconstruction_only_warmup_steps=$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS"
    "++run_provenance.objective.effective_reconstruction_only_warmup_steps=$RECONSTRUCTION_ONLY_WARMUP_STEPS"
  )
fi

if test "$AF_NATIVE_LIBERO" = true; then
  test -n "${SLURM_STEP_ID:-}" || die "native schema requires scheduled srun"
  for binding in "AF_NATIVE_SCHEMA_DRIVER:$AF_NATIVE_SCHEMA_DRIVER" "AF_NATIVE_BINDING_OVERRIDE:$AF_NATIVE_BINDING_OVERRIDE"; do
    absolute_path "${binding%%:*}" "${binding#*:}"
    test -s "${binding#*:}" || die "native dispatch input unavailable"
  done
  test "$(sha256 "$AF_NATIVE_SCHEMA_DRIVER")" = "$AF_NATIVE_SCHEMA_DRIVER_SHA256" || die "native schema driver SHA mismatch"
  test "$(sha256 "$AF_NATIVE_BINDING_OVERRIDE")" = "$AF_NATIVE_BINDING_OVERRIDE_SHA256" || die "native saved-state override SHA mismatch"
  test -s "$AF_NORM_STATS_PATH" || die "native saved-state payload unavailable"
  test "$(sha256 "$AF_NORM_STATS_PATH")" = "$AF_EXPECTED_NORM_SHA256" || die "native payload SHA mismatch"
  EFFECTIVE_NORM_FILE=$AF_NORM_STATS_PATH
  IFS= read -r native_binding < "$AF_NATIVE_BINDING_OVERRIDE"
  [[ "$native_binding" = ++norm_stats.native_saved_state_binding=* ]] || die "exact native binding override required"
  test "$(wc -l < "$AF_NATIVE_BINDING_OVERRIDE")" -eq 1 || die "binding must be one complete override"
  BASE_OVERRIDES+=("$native_binding" norm_stats.precomputed_norm_path=null norm_stats.save_cache_dir=null
    "++run_provenance.normalization_sha256=$AF_EXPECTED_NORM_SHA256")
  # Structural-only sentinels are never valid SHA-256 proof identities.
  # They are allowed only in schema mode and keep READY permanently false.
  export AF_NATIVE_PENDING_PROOFS=""
  if test "$AF_NATIVE_PHASE" != preflight; then
    if test -n "${AF_PREFLIGHT_RESULT:-}" && test -n "${AF_EXPECTED_PREFLIGHT_SHA256:-}"; then
      test "$(sha256 "$AF_PREFLIGHT_RESULT")" = "$AF_EXPECTED_PREFLIGHT_SHA256" || die "native preflight SHA mismatch"
      preflight_identity=$AF_EXPECTED_PREFLIGHT_SHA256
    elif test "${AF_NATIVE_SCHEMA_ALLOW_PENDING_PROOFS:-false}" = true; then
      preflight_identity=PENDING_SCHEMA_ONLY_NO_VALID_PREFLIGHT_PROOF
      AF_NATIVE_PENDING_PROOFS=preflight
    else
      die "exact native preflight proof required"
    fi
    BASE_OVERRIDES+=("++run_provenance.preflight_result_sha256=$preflight_identity")
  fi
  if test "$AF_NATIVE_PHASE" = full; then
    if test -n "${AF_SMOKE_RESULT:-}" && test -n "${AF_EXPECTED_SMOKE_SHA256:-}"; then
      test "$(sha256 "$AF_SMOKE_RESULT")" = "$AF_EXPECTED_SMOKE_SHA256" || die "native smoke SHA mismatch"
      smoke_identity=$AF_EXPECTED_SMOKE_SHA256
    elif test "${AF_NATIVE_SCHEMA_ALLOW_PENDING_PROOFS:-false}" = true; then
      smoke_identity=PENDING_SCHEMA_ONLY_NO_VALID_SMOKE_PROOF
      AF_NATIVE_PENDING_PROOFS="$AF_NATIVE_PENDING_PROOFS smoke"
    else
      die "exact native smoke proof required"
    fi
    BASE_OVERRIDES+=("++run_provenance.smoke_result_sha256=$smoke_identity")
  fi
elif test -n "${AF_NORM_STATS_PATH:-}"; then
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

if test -n "${AF_INITIAL_CHECKPOINT:-}" && test -n "${AF_RESUME_TYPED_MODEL:-}"; then
  case "$AF_RESUME_TYPED_MODEL" in noaug|av0) ;; *) die "unsupported typed continuation";; esac
  test "${AF_HOMOGENEOUS_NUMERICAL_DELTA_ACCEPTED:-}" = false || die "typed continuation requires source-specific raw proof"
  required AF_RESUME_HELPER_DIR
  required AF_RESUME_TYPED_CALLBACK_SHA256
  test "$(sha256 "$AF_RESUME_HELPER_DIR/homogeneous_typed_native_training.py")" = "$AF_RESUME_TYPED_CALLBACK_SHA256" || die "typed callback hash mismatch"
  BASE_OVERRIDES+=("trainer.deterministic=true" "++trainer.benchmark=false")
  if test "$AF_RESUME_TYPED_MODEL" = noaug; then
    test "$AF_H696_EULER" = true || die "noaug model mismatch"
    BASE_OVERRIDES+=(
      "+callbacks.homogeneous_dithalf._target_=homogeneous_dithalf_training_typed_native_rows_proposal_v6.HomogeneousDiTHalf"
      "+callbacks.homogeneous_dithalf.task=$AF_NOAUG_TASK"
      "+callbacks.homogeneous_dithalf.reference_task=$AF_NOAUG_REFERENCE_TASK"
      "+callbacks.homogeneous_dithalf.contract_path=$AF_NOAUG_TASK/NOAUG_EXECUTION_CONTRACT_VERIFIED_V2.json"
      model.hidden_dim=696
      model.pipeline.stages.6.inference_method=euler
      model.num_inference_steps=50
      ++run_provenance.scaled_from_hidden_dim=384
      ++run_provenance.inference_parameter_count=257829594
      ++run_provenance.validation_inference_sampler=euler
      ++run_provenance.validation_inference_updates=50
    )
  else
    test "$AF_SPEED_DIAGNOSTIC" = true && test "$AF_ACTION_VELOCITY_WEIGHT" = 0 || die "AV0 model mismatch"
    BASE_OVERRIDES+=(
      "+callbacks.homogeneous_dithalf._target_=av0_native_rows_callback_v5.HomogeneousDiTHalf"
      "+callbacks.homogeneous_dithalf.task=$AF_AV_ZERO_TASK"
      "+callbacks.homogeneous_dithalf.reference_task=$AF_HOMOGENEOUS_TASK"
      "+callbacks.homogeneous_dithalf.contract_path=$AF_EXECUTION_CONTRACT_PATH"
      "++run_provenance.execution_numerical_contract=raw-required"
      "++run_provenance.execution_contract_sha256=$AF_EXECUTION_CONTRACT_SHA256"
      "model.pipeline.stages.8.action_velocity_weight=0"
      "++run_provenance.objective.action_velocity_weight=0"
      model.pipeline.stages.6.inference_method=euler
      model.pipeline.stages.6.num_inference_steps=50
      ++run_provenance.validation_inference_sampler=euler
      ++run_provenance.validation_inference_steps=50
    )
  fi
elif test -n "${AF_INITIAL_CHECKPOINT:-}"; then
  # Task-local typed extension of canonical launcher; user accepts the known
  # grouped BF16 numerical delta, but never a silent sampling/augmentation delta.
  test "${AF_HOMOGENEOUS_NUMERICAL_DELTA_ACCEPTED:-}" = true || die "explicit numerical authorization required"
  required AF_HOMOGENEOUS_TASK
  required AF_HOMOGENEOUS_CALLBACK_SHA256
  test "$(sha256 "$AF_HOMOGENEOUS_TASK/homogeneous_dithalf_training.py")" = "$AF_HOMOGENEOUS_CALLBACK_SHA256" || die "callback SHA mismatch"
  # Full-corpus retimed normalization retains shared-memory batch buffers.
  # The gateway's inherited 1024 descriptor limit failed before completion.
  ulimit -Sn 131072 || die "speed diagnostic requires RLIMIT_NOFILE >=131072"
  test "$(ulimit -Sn)" -ge 131072 || die "open-file limit guard failed"
  printf '[speed-contract] open_file_soft_limit=%s\n' "$(ulimit -Sn)"
  BASE_OVERRIDES+=(
    "+callbacks.homogeneous_dithalf._target_=homogeneous_dithalf_training.HomogeneousDiTHalf"
    "+callbacks.homogeneous_dithalf.task=$AF_HOMOGENEOUS_TASK"
    "model.pipeline.stages.6.inference_method=euler"
    "model.pipeline.stages.6.num_inference_steps=50"
    "++run_provenance.validation_inference_sampler=euler"
    "++run_provenance.validation_inference_steps=50"
    "speed_diagnostic.encoding=$AF_SPEED_ENCODING"
    $([[ "$AF_EXPERIMENT" = pusht/action_flow_cotrain_uc_speed_interpolation ]] && printf "%s" "speed_diagnostic.speed_reference=$AF_SPEED_REFERENCE" || true)
    "run_provenance.architecture.family=action_flow_private_codecs_shared_field_h696d12h12"
    "run_provenance.architecture.tokenizer=two_private_UniteDiTBackbone_h696_d12_h12"
    "run_provenance.architecture.denoiser=one_shared_UniteDiTBackbone_h696_d12_h12"
    "run_provenance.architecture.decoder=two_private_UniteActionDecoder_h696_d12_h12"
    "run_provenance.validation.interval_optimizer_steps=$VALIDATE_EVERY"
    "run_provenance.validation.limit_batches=$LIMIT_VAL_BATCHES"
  )
fi


if test -n "${AF_INITIAL_CHECKPOINT:-}"; then
  BASE_OVERRIDES+=("ckpt_path='$AF_INITIAL_CHECKPOINT'")
  if test "$AF_RUN_KIND" = full; then BASE_OVERRIDES+=("++logger.wandb.resume=must");
  else BASE_OVERRIDES+=("++logger.wandb.resume=never" "+callbacks.resume_batch_probe._target_=homogeneous_dithalf_training.ResumeBatchProbe"); fi
  source "$AF_RESUME_DRIVER_REPO/scripts/train/action_flow_checkpoint_resume.sh"
  exit 0
fi

if test "$AF_NATIVE_LIBERO" = true && test "$AF_LAUNCH_MODE" != schema; then
  if test -n "${AF_NATIVE_CONTEXT_RELEASE:-}"; then
    test "$AF_EXPECTED_HEAD" = 26a3dce73cf6a770c040d3433424915a525c745b || die "external native context source mismatch"
    case "$AF_NATIVE_CONTEXT_RELEASE" in /*) ;; *) die "absolute external context release required";; esac
    test "$(sha256 "$AF_NATIVE_CONTEXT_RELEASE/libero_native_launch_phases_context_v3.sh")" = f51a5f2590e7bb48d65b008a2a0e4db0dde70542211359403c08525b5b260f07 || die "external native phase SHA mismatch"
    test "$(sha256 "$AF_NATIVE_CONTEXT_RELEASE/saved_hydra_context_dataset_identity_v1.py")" = 76713364ac3245ab0bc2434b3d38e1e25f7c2dc2e49627dff9b7b3e0678e35ca || die "external native context SHA mismatch"
    test "$(sha256 "$AF_NATIVE_CONTEXT_RELEASE/bootstrap_saved_dataset_identity_v1.py")" = c6e95242e8b5b182edb35b57a0b1526fecb6d74ca054269bf6a36e30be249050 || die "external native bootstrap SHA mismatch"
    printf '[native external certification] model-source=%s phase=f51a5f2590e7bb48d65b008a2a0e4db0dde70542211359403c08525b5b260f07 context=76713364ac3245ab0bc2434b3d38e1e25f7c2dc2e49627dff9b7b3e0678e35ca bootstrap=c6e95242e8b5b182edb35b57a0b1526fecb6d74ca054269bf6a36e30be249050\n' "$AF_EXPECTED_HEAD"
    source "$AF_NATIVE_CONTEXT_RELEASE/libero_native_launch_phases_context_v3.sh"
  else
    source "$AF_REPO/scripts/train/libero_native_launch_phases.sh"
  fi
  exit 0
fi

if test "$AF_LAUNCH_MODE" = schema; then
  test "$AF_NATIVE_LIBERO" = true || die "schema mode is native LIBERO only"
  test ! -e "$AF_OUTPUT_DIR" || die "schema output collision"
  mkdir -p "$AF_OUTPUT_DIR"
  (set -o noclobber; printf '%s\0' "${BASE_OVERRIDES[@]}" > "$AF_OUTPUT_DIR/exact-phase.argv0")
  export AF_NATIVE_PHASE
  "$AF_PYTHON" "$AF_NATIVE_SCHEMA_DRIVER" --repo "$AF_REPO" --phase "$AF_NATIVE_PHASE" \
    --output "$AF_OUTPUT_DIR" --argv "$AF_OUTPUT_DIR/exact-phase.argv0" \
    --expected-argv-sha256 "$(sha256 "$AF_OUTPUT_DIR/exact-phase.argv0")"
  exit 0
fi

if test "$AF_LAUNCH_MODE" = preflight; then
  for variable in \
    SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_JOB_QOS \
    SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK; do
    required "$variable"
  done
  test "$SLURM_JOB_ACCOUNT" = "$AF_EXPECTED_ACCOUNT" || die "unexpected Slurm account"
  partition_allowed "$AF_EXPECTED_PARTITION" "$SLURM_JOB_PARTITION" || die "unexpected Slurm partition"
  test "$SLURM_JOB_QOS" = "$AF_EXPECTED_QOS" || die "unexpected Slurm QoS"
  test "$SLURM_JOB_NUM_NODES" = 1 || die "preflight requires one node"
  test "$SLURM_NTASKS" = 1 || die "preflight requires one task"
  test "$SLURM_CPUS_PER_TASK" -ge 8 || die "preflight requires at least eight CPUs"
  test "$RESTART_COUNT" = 0 || die "preflight cannot be requeued"
  test ! -e "$AF_OUTPUT_DIR" || die "unique preflight output already exists"
  mkdir -p "$AF_OUTPUT_DIR"
  # Compose the real argument vector before runtime scans or data traversal.
  if test "$AF_SPEED_DIAGNOSTIC" = true; then
    (
      cd "$AF_REPO"
      "$AF_PYTHON" -m egomimic.trainHydra "${BASE_OVERRIDES[@]}" \
        "++run_provenance.normalization_sha256=${AF_EXPECTED_NORM_SHA256:-$(printf '0%.0s' {1..64})}" \
        --cfg job --resolve > "$AF_OUTPUT_DIR/exact-phase-before-data.yaml"
    )
  fi
  "$AF_PYTHON" "$AF_RUNTIME_LOCK_CAPTURE" \
    --repo "$AF_REPO" \
    --expected-head "$AF_EXPECTED_HEAD" \
    --lock-input "$AF_REPO/pyproject.toml" \
    --lock-input "$AF_REPO/uv.lock" \
    --output "$AF_OUTPUT_DIR/runtime-lock.json"
  USE_CACHED_DATASET_VALIDATION=false
  if test -n "${AF_CACHED_DATASET_VALIDATION:-}"; then
    if "$AF_PYTHON" - "$AF_CACHED_DATASET_VALIDATION" \
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
    then
      USE_CACHED_DATASET_VALIDATION=true
    else
      # A receipt binds one physical dataset root.  Matching content hashes are
      # insufficient after a cross-account or task-local copy: validate the
      # actual ICE path rather than treating a stale cache as evidence.
      printf '[preflight] cached dataset receipt does not bind %s; validating the task-local dataset\n' \
        "$AF_DATASET_DIR" >&2
    fi
  fi
  if test "$USE_CACHED_DATASET_VALIDATION" = true; then
    cp -- "$AF_CACHED_DATASET_VALIDATION" \
      "$AF_OUTPUT_DIR/dataset_validation.json"
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
  if test "$AF_COTRAIN" = true; then
    USE_CACHED_SECOND_DATASET_VALIDATION=false
    if test "$AF_PAIRED_CHAIN_MANUAL4919" = true; then
      # Both logical routes are the same physical manual4919 corpus and split.
      # The primary receipt was just verified or materialized above; copying
      # those exact bytes avoids a second 4,919-episode inventory traversal.
      cp -- "$AF_OUTPUT_DIR/dataset_validation.json" \
        "$AF_OUTPUT_DIR/dataset_validation_secondary.json"
      USE_CACHED_SECOND_DATASET_VALIDATION=true
    fi
    if test "$AF_PAIRED_CHAIN_MANUAL4919" != true \
      && test -n "${AF_CACHED_SECOND_DATASET_VALIDATION:-}"; then
      if "$AF_PYTHON" - "$AF_CACHED_SECOND_DATASET_VALIDATION" \
        "$AF_SECOND_DATASET_DIR" "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256" \
        "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" \
        "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" <<'PY'
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
      then
        USE_CACHED_SECOND_DATASET_VALIDATION=true
      fi
    fi
    if test "$USE_CACHED_SECOND_DATASET_VALIDATION" = true; then
      if test "$AF_PAIRED_CHAIN_MANUAL4919" != true; then
        cp -- "$AF_CACHED_SECOND_DATASET_VALIDATION" \
          "$AF_OUTPUT_DIR/dataset_validation_secondary.json"
      fi
    else
      "$AF_PYTHON" "$AF_DATASET_VALIDATOR" \
        --repo "$AF_REPO" \
        --dataset-root "$AF_SECOND_DATASET_DIR" \
        --manifest "$AF_SECOND_SPLIT_MANIFEST" \
        --expected-manifest-sha256 "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256" \
        --content-manifest "$AF_SECOND_CONTENT_MANIFEST" \
        --expected-content-manifest-sha256 \
          "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" \
        --output "$AF_OUTPUT_DIR/dataset_validation_secondary.json"
    fi
  fi
  (
    cd "$AF_REPO"
    CONFIG_VALIDATOR_ARGS=(
      --experiment "$AF_EXPERIMENT"
      --config-root "$AF_REPO/egomimic/hydra_configs"
      --output "$AF_OUTPUT_DIR/action_flow_pair_preflight.json"
    )
    if test "$AF_PAIRED_CHAIN_MANUAL4919" = true || test "$AF_SPEED_DIAGNOSTIC" = true; then
      for exact_override in "${BASE_OVERRIDES[@]}"; do
        case "$exact_override" in
          +experiment=*|--config-name=*) continue ;;
        esac
        CONFIG_VALIDATOR_ARGS+=("--override=$exact_override")
      done
    fi
    if test "$AF_SCALED_H512" = true; then
      CONFIG_VALIDATOR_ARGS+=(
        --override "data.train_datasets.$AF_SOURCE.resolver.folder_path=$AF_DATASET_DIR"
        --override "data.valid_datasets.$AF_SOURCE.resolver.folder_path=$AF_DATASET_DIR"
        --override "++run_provenance.split_manifest_path=$AF_SPLIT_MANIFEST"
        --override "++run_provenance.split_manifest_sha256=$AF_EXPECTED_SPLIT_MANIFEST_SHA256"
        --override "++run_provenance.content_manifest_path=$AF_CONTENT_MANIFEST"
        --override "++run_provenance.content_manifest_sha256=$AF_EXPECTED_CONTENT_MANIFEST_SHA256"
        --override "++run_provenance.dataset_content_aggregate_sha256=$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256"
        --override 'evaluator.action_flow_diagnostics.activation_layer_map={0:0,1:13}'
      )
      if test "$AF_COTRAIN" = true; then
        CONFIG_VALIDATOR_ARGS+=(
          --override "data.train_datasets.$AF_SECOND_SOURCE.resolver.folder_path=$AF_SECOND_DATASET_DIR"
          --override "data.valid_datasets.$AF_SECOND_SOURCE.resolver.folder_path=$AF_SECOND_DATASET_DIR"
          --override "++run_provenance.chain_split_manifest_path=$AF_SECOND_SPLIT_MANIFEST"
          --override "++run_provenance.chain_split_manifest_sha256=$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256"
          --override "++run_provenance.content_manifests.$AF_SOURCE.path=$AF_CONTENT_MANIFEST"
          --override "++run_provenance.content_manifests.$AF_SOURCE.manifest_sha256=$AF_EXPECTED_CONTENT_MANIFEST_SHA256"
          --override "++run_provenance.content_manifests.$AF_SOURCE.aggregate_sha256=$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256"
          --override "++run_provenance.content_manifests.$AF_SECOND_SOURCE.path=$AF_SECOND_CONTENT_MANIFEST"
          --override "++run_provenance.content_manifests.$AF_SECOND_SOURCE.manifest_sha256=$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256"
          --override "++run_provenance.content_manifests.$AF_SECOND_SOURCE.aggregate_sha256=$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256"
        )
      fi
    fi
    if test -n "$AF_COMPARISON_EXPERIMENT"; then
      CONFIG_VALIDATOR_ARGS+=(--compare-experiment "$AF_COMPARISON_EXPERIMENT")
    fi
    if test "$AF_COTRAIN" = true && test "$AF_SCALED_H512" = false \
      && test "$AF_PAIRED_CHAIN_MANUAL4919" = false && test "$AF_SPEED_DIAGNOSTIC" = false; then
      # Preserve the pre-existing historical H384 co-train path byte-for-byte.
      # It is not authority for the new paired-Chain route, which must execute
      # the real fail-closed validator below.
      "$AF_PYTHON" - "$AF_OUTPUT_DIR/action_flow_pair_preflight.json" \
        "$AF_EXPERIMENT" "$AF_CONFIG_VALIDATOR" <<'PY'
import hashlib
import json
import pathlib
import sys

output, experiment, validator = map(pathlib.Path, sys.argv[1:])
payload = {
    "schema_version": 1,
    "status": "PASS",
    "experiment": str(experiment),
    "validation_mode": "launcher_strict_routed_cotrain",
    "note": (
        "The historical external validator does not accept arbitrary routed "
        "cotrain experiment names/topologies; the launcher composes, "
        "instantiates, and fail-closed audits the exact resolved config."
    ),
    "external_validator_path": str(validator),
    "external_validator_sha256": hashlib.sha256(validator.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    else
      "$AF_PYTHON" "$AF_CONFIG_VALIDATOR" "${CONFIG_VALIDATOR_ARGS[@]}"
    fi
    PREFLIGHT_NORM_ARGS=()
    if test -z "${AF_NORM_STATS_PATH:-}"; then
      PREFLIGHT_NORM_SENTINEL=$(printf '0%.0s' {1..64})
      PREFLIGHT_NORM_ARGS+=(
        "++run_provenance.normalization_sha256=$PREFLIGHT_NORM_SENTINEL"
      )
    fi
    HYDRA_FULL_ERROR=1 "$AF_PYTHON" -m egomimic.trainHydra \
      "${BASE_OVERRIDES[@]}" "${PREFLIGHT_NORM_ARGS[@]}" \
      --cfg job --resolve \
      > "$AF_OUTPUT_DIR/resolved_config_pre_normalization.yaml"
    if test -z "${AF_NORM_STATS_PATH:-}"; then
      HYDRA_FULL_ERROR=1 "$AF_PYTHON" -m egomimic.trainHydra \
        "${BASE_OVERRIDES[@]}" \
        "++run_provenance.normalization_sha256=$PREFLIGHT_NORM_SENTINEL" \
        norm_stats_only=true \
        trainer.accelerator=cpu trainer.devices=1 '~logger' \
        "hydra.run.dir=$AF_OUTPUT_DIR/norm_materialize"
    fi
  )
  test -s "$EFFECTIVE_NORM_FILE" || die "normalization materialization failed"
  (
    cd "$AF_REPO"
    FINAL_NORM_ARGS=()
    if test -z "${AF_NORM_STATS_PATH:-}"; then
      FINAL_NORM_ARGS+=(
        "++run_provenance.normalization_sha256=$(sha256 "$EFFECTIVE_NORM_FILE")"
      )
    fi
    HYDRA_FULL_ERROR=1 "$AF_PYTHON" -m egomimic.trainHydra \
      "${BASE_OVERRIDES[@]}" "${FINAL_NORM_ARGS[@]}" \
      --cfg job --resolve > "$AF_OUTPUT_DIR/resolved_config.yaml"
  )
  LOADER_SRUN_ARGS=()
  if test "$AF_PAIRED_CHAIN_MANUAL4919" = true; then
    LOADER_SRUN_ARGS+=(--overlap)
  fi
  srun "${LOADER_SRUN_ARGS[@]}" --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    /bin/bash -c 'set -Eeuo pipefail; cd "$AF_REPO"; exec "$AF_PYTHON" - "$@"' \
    action-flow-loader "$AF_OUTPUT_DIR/resolved_config.yaml" \
    "$AF_OUTPUT_DIR/dataset_loader_probe.json" "$AF_REPO" <<'PY'
import json
import pathlib
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf

config_path, output_path, repo_path = map(pathlib.Path, sys.argv[1:])
if pathlib.Path.cwd().resolve() != repo_path.resolve():
    raise SystemExit("nested Slurm container did not enter the pinned source checkout")
cfg = OmegaConf.load(config_path)
rows = []
for split_name, expected_field in (
    ("train_datasets", "expected_train_episode_count"),
    ("valid_datasets", "expected_valid_episode_count"),
):
    for source, dataset_cfg in cfg.data[split_name].items():
        dataset = instantiate(dataset_cfg)
        observed = len(dataset.datasets)
        expected = int(dataset_cfg[expected_field])
        if observed != expected:
            raise SystemExit(
                f"{split_name}.{source}: expected {expected} loaded episodes, "
                f"found {observed}"
            )
        rows.append(
            {
                "split": split_name,
                "source": str(source),
                "expected_episode_count": expected,
                "loaded_episode_count": observed,
                "sample_count": len(dataset),
            }
        )
payload = {"schema_version": 1, "status": "PASS", "datasets": rows}
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
PY
  "$AF_PYTHON" - "$AF_OUTPUT_DIR" "$AF_REPO" "$AF_EXPECTED_HEAD" \
    "$AF_EXPERIMENT" "$AF_DATASET_DIR" "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
    "$LAUNCHER" "$AF_PYTHON" "$EFFECTIVE_NORM_FILE" "$AF_CLUSTER_LABEL" \
    "$SLURM_JOB_ID" "$AF_OUTPUT_DIR/runtime-lock.json" \
    "$AF_CONTENT_MANIFEST" "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
    "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" "$AF_COTRAIN" \
    "${AF_SECOND_DATASET_DIR:-}" "${AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256:-}" \
    "${AF_SECOND_CONTENT_MANIFEST:-}" \
    "${AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256:-}" \
    "${AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256:-}" "${AF_SPEED_ENCODING:-}" <<'PY'
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
    cotrain,
    second_dataset,
    second_split_sha,
    second_content_manifest_path,
    second_content_manifest_sha256,
    second_content_aggregate_sha256,
    speed_encoding,
) = sys.argv[1:]
root = pathlib.Path(root)
norm = pathlib.Path(norm)
runtime_lock_path = pathlib.Path(runtime_lock_path)

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

dataset_report = json.loads((root / "dataset_validation.json").read_text())
pair_report = json.loads((root / "action_flow_pair_preflight.json").read_text())
loader_report = json.loads((root / "dataset_loader_probe.json").read_text())
runtime_lock = json.loads(runtime_lock_path.read_text())
assert dataset_report["status"] == "DATASET_VALIDATED"
assert dataset_report["content_identity"]["manifest_sha256"] == (
    content_manifest_sha256
)
assert dataset_report["content_identity"]["aggregate_sha256"] == (
    content_aggregate_sha256
)
assert pair_report["status"] == "PASS"
assert loader_report["status"] == "PASS"
assert runtime_lock["status"] == "RUNTIME_LOCKED"
datasets = {
    "primary": {
        "root": dataset,
        "split_manifest_sha256": split_sha,
        "content_manifest_path": content_manifest_path,
        "content_manifest_sha256": content_manifest_sha256,
        "content_aggregate_sha256": content_aggregate_sha256,
        "validation_sha256": digest(root / "dataset_validation.json"),
    }
}
if cotrain == "true":
    secondary_report = json.loads(
        (root / "dataset_validation_secondary.json").read_text()
    )
    assert secondary_report["status"] == "DATASET_VALIDATED"
    assert secondary_report["manifest_sha256"] == second_split_sha
    assert secondary_report["physical_inventory"]["dataset_root"] == second_dataset
    assert secondary_report["content_identity"]["manifest_sha256"] == (
        second_content_manifest_sha256
    )
    assert secondary_report["content_identity"]["aggregate_sha256"] == (
        second_content_aggregate_sha256
    )
    datasets["secondary"] = {
        "root": second_dataset,
        "split_manifest_sha256": second_split_sha,
        "content_manifest_path": second_content_manifest_path,
        "content_manifest_sha256": second_content_manifest_sha256,
        "content_aggregate_sha256": second_content_aggregate_sha256,
        "validation_sha256": digest(root / "dataset_validation_secondary.json"),
    }
payload = {
    "schema_version": 1,
    "status": "PASS",
    "source": {"repo": repo, "head": head},
    "experiment": experiment,
    "speed_encoding": speed_encoding,
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
    "dataset_loader_probe_sha256": digest(root / "dataset_loader_probe.json"),
    "datasets": datasets,
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
  "$AF_CLUSTER_LABEL" "$AF_COTRAIN" "${AF_SECOND_DATASET_DIR:-}" \
  "${AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256:-}" \
  "${AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256:-}" \
  "${AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256:-}" "${AF_SPEED_ENCODING:-}" <<'PY'
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
    cotrain,
    second_dataset,
    second_split_sha,
    second_content_sha,
    second_content_aggregate_sha,
    speed_encoding,
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
if experiment.endswith(("action_flow_cotrain_uc_speed_interpolation", "action_flow_cotrain_uc_multiplier_interpolation")):
    assert payload["speed_encoding"] == speed_encoding
if cotrain == "true":
    secondary = payload["datasets"]["secondary"]
    assert secondary["root"] == second_dataset
    assert secondary["split_manifest_sha256"] == second_split_sha
    assert secondary["content_manifest_sha256"] == second_content_sha
    assert secondary["content_aggregate_sha256"] == second_content_aggregate_sha
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
    "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" "$AF_SCALED_H512" \
    "$AF_PAIRED_CHAIN_MANUAL4919" \
    "${AF_EXPECTED_SMOKE_PARAMETER_COUNT:-$AF_EXPECTED_PARAMETER_COUNT}" "$AF_COTRAIN" \
    "${AF_SECOND_SOURCE:-}" \
    "${AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256:-}" \
    "${AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256:-}" \
    "${AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256:-}" "${AF_SPEED_ENCODING:-}" <<'PY'
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
    scaled_h512,
    paired_chain,
    expected_parameter_count,
    cotrain,
    second_source,
    second_split_sha,
    second_content_sha,
    second_content_aggregate_sha,
    speed_encoding,
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
if experiment.endswith(("action_flow_cotrain_uc_speed_interpolation", "action_flow_cotrain_uc_multiplier_interpolation")):
    assert identities["speed_conditioning"]["encoding"] == speed_encoding
    if experiment.endswith("action_flow_cotrain_uc_multiplier_interpolation"):
        assert identities["speed_conditioning"]["conditioning_input"] == "retiming_multiplier"
        assert identities["speed_conditioning"]["input_key"] == "retiming_rate"
    else:
        assert identities["speed_conditioning"]["speed_reference"] == 84.11600368466028
assert payload["preflight"]["sha256"] == preflight_sha
assert payload["checkpoint"]["global_step"] == 2
assert payload["checkpoint"]["file_size_bytes"] > 0
assert payload["checkpoint"]["gradient_routes"]["manifest_sha256"]
if scaled_h512 == "true" or paired_chain == "true" or experiment.endswith(("action_flow_cotrain_uc_speed_interpolation", "action_flow_cotrain_uc_multiplier_interpolation")):
    assert payload["checkpoint"]["parameter_count"] == int(expected_parameter_count)
    assert payload["artifacts"]["energy_score"]["sha256"]
    assert payload["artifacts"]["action_flow_diagnostics"]["sha256"]
    if cotrain == "true":
        secondary = identities["datasets"][second_source]
        assert secondary["split_manifest_sha256"] == second_split_sha
        assert secondary["content_manifest_sha256"] == second_content_sha
        assert secondary["dataset_content_aggregate_sha256"] == second_content_aggregate_sha
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
partition_allowed "$AF_EXPECTED_PARTITION" "$SLURM_JOB_PARTITION" || die "unexpected Slurm partition"
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
if test "$AF_COTRAIN" = true; then
  cp -- "$AF_SECOND_SPLIT_MANIFEST" "$ATTEMPT/split_manifest_secondary.json"
  cp -- "$AF_SECOND_CONTENT_MANIFEST" "$ATTEMPT/content_manifest_secondary.json"
fi
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
  --expected-partition "$SLURM_JOB_PARTITION" \
  --expected-qos "$AF_EXPECTED_QOS" \
  --expected-cpus 8 \
  --expected-memory "$AF_EXPECTED_MEMORY" \
  --expected-time-limit "$AF_EXPECTED_TIME_LIMIT" \
  --expected-constraint "$AF_EXPECTED_GPU_CONSTRAINT" \
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
  "$AF_RECONSTRUCTION_WEIGHT" "$AF_FLOW_WEIGHT" "$AF_DATASET_DIR" \
  "$AF_NORM_STATS_PATH" "$AF_CLUSTER_LABEL" "$AF_EXPECTED_HEAD" \
  "$AF_EXPECTED_LAUNCHER_SHA256" "$AF_EXPECTED_NORM_SHA256" \
  "$AF_EXPECTED_PREFLIGHT_SHA256" "$AF_EFFECTIVE_SMOKE_SHA256" \
  "$AF_CONTENT_MANIFEST" "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
  "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" \
  "$AF_FULL_RECONSTRUCTION_ONLY_WARMUP_STEPS" \
  "$RECONSTRUCTION_ONLY_WARMUP_STEPS" "$AF_FULL_MAX_STEPS" \
  "$AF_FULL_VALIDATE_EVERY" "$AF_FULL_CHECKPOINT_EVERY" \
  "$AF_FULL_LIMIT_VAL_BATCHES" "$AF_VALID_BATCH_SIZE" "$AF_SOURCE" \
  "$AF_ACTION_DIM" "$AF_EXPECTED_PARAMETER_COUNT" "$AF_COTRAIN" "$AF_SCALED_H512" \
  "${AF_SECOND_SOURCE:-}" "${AF_SECOND_DATASET_DIR:-}" <<'PY'
import math
import os
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
    flow_weight,
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
    full_max_steps,
    full_validate_every,
    full_checkpoint_every,
    full_limit_val_batches,
    valid_batch_size,
    source,
    action_dim,
    expected_parameter_count,
    cotrain,
    scaled_h512,
    second_source,
    second_dataset,
) = sys.argv[1:]
cfg = OmegaConf.load(pathlib.Path(path))
reconstruction = float(reconstruction)
flow_weight = float(flow_weight)
full_warmup_steps = int(full_warmup_steps)
effective_warmup_steps = int(effective_warmup_steps)
full_max_steps = int(full_max_steps)
full_validate_every = int(full_validate_every)
full_checkpoint_every = int(full_checkpoint_every)
full_limit_val_batches = float(full_limit_val_batches)
valid_batch_size = int(valid_batch_size)
action_dim = int(action_dim)
expected_parameter_count = int(expected_parameter_count)
cotrain = cotrain == "true"
scaled_h512 = scaled_h512 == "true"
speed_diagnostic = experiment.endswith(("action_flow_cotrain_uc_speed_interpolation", "action_flow_cotrain_uc_multiplier_interpolation"))
if speed_diagnostic:
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("typed_condition_validator", os.environ["AF_SPEED_CONFIG_VALIDATOR"])
    typed = module_from_spec(spec); spec.loader.exec_module(typed)
    validate_speed_contract = typed.validate_speed_contract
    validate_speed_contract(cfg, os.environ["AF_SPEED_ENCODING"])
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
# AF_CONFIG_VALIDATOR may be a task-local staged tool rather than a module in
# the immutable source checkout.  AF_EXTRA_PYTHONPATH therefore exposes its
# directory directly, not a synthetic ``tools`` package.
if speed_diagnostic or experiment.endswith("action_flow_chain_native4_points6_manual4919_h384_150k_s42"):
    # The paired route uses the separately pinned fail-closed config validator.
    # Historical topology helpers hard-code the old U-Socket/Chain routes.
    LEGACY_METHOD = "action_flow_joint"
    LIKELIHOOD_METHOD = "gaussian_bridge_likelihood"
    GRAPH_METHOD = "graph_section_diagnostic"
    STOPGRAD_METHOD = "latent_fm_stopgrad"
    STOPGRAD_UNITE_METHOD = "latent_fm_stopgrad_unite"
else:
    from validate_action_flow_config import (
        LEGACY_METHOD, LIKELIHOOD_METHOD, GRAPH_METHOD, STOPGRAD_METHOD,
        STOPGRAD_UNITE_METHOD,
        validate_method_contract, _validate_dimensions_and_modules, _validate_topology,
    )
if scaled_h512:
    method = STOPGRAD_UNITE_METHOD
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_.endswith(
        "ActionFlowTrainingBehavior"
    )
    targets = [str(stage._target_) for stage in cfg.model.pipeline.stages]
    assert targets[4].endswith(
        "RoutedContentEncoderStage" if cotrain else "ContentEncoderStage"
    )
    assert targets[7].endswith(
        "RoutedContentDecoderStage" if cotrain else "ContentDecoderStage"
    )
    assert cfg.model.hidden_dim == 512
    assert cfg.model.flow_mini_batch == 14
elif cotrain:
    method = STOPGRAD_UNITE_METHOD
    assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.model.training_behavior._target_.endswith("ActionFlowTrainingBehavior")
    targets = [str(stage._target_) for stage in cfg.model.pipeline.stages]
    if speed_diagnostic or experiment.endswith("action_flow_chain_native4_points6_manual4919_h384_150k_s42"):
        assert targets[4].endswith("RoutedContentEncoderStage")
        assert targets[7].endswith("RoutedContentDecoderStage")
    else:
        assert targets[3].endswith("RoutedContentEncoderStage")
        assert targets[6].endswith("RoutedContentDecoderStage")
else:
    method = validate_method_contract(cfg, experiment)
assert cfg.model.action_horizon == 16
if not cotrain:
    assert cfg.model.action_dim == action_dim
if scaled_h512:
    assert cfg.model.num_latent_tokens == 8
    assert cfg.model.latent_dim == 16 and cfg.model.condition_dim == 128
    assert math.isclose(cfg.model.condition_dropout_probability, 0.1)
elif cotrain:
    assert cfg.model.num_latent_tokens == 8
    assert cfg.model.latent_dim == 16 and cfg.model.condition_dim == 128
    assert math.isclose(cfg.model.condition_dropout_probability, 0.1)
elif method == STOPGRAD_UNITE_METHOD:
    assert cfg.model.num_latent_tokens == 8
    assert cfg.model.latent_dim == 16 and cfg.model.condition_dim == 128
    assert math.isclose(cfg.model.condition_dropout_probability, 0.1)
else:
    assert cfg.model.latent_dim == 8 and cfg.model.condition_dim == 67
    assert math.isclose(cfg.model.condition_dropout_probability, 0.3)
if method != LIKELIHOOD_METHOD:
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.num_inference_steps == (
        50 if method == STOPGRAD_UNITE_METHOD else 16
    )
    assert math.isclose(cfg.model.reconstruction_weight, reconstruction)
    assert math.isclose(cfg.model.flow_weight, flow_weight)
    assert cfg.model.reconstruction_only_warmup_steps == effective_warmup_steps
    assert cfg.run_provenance.objective.effective_reconstruction_only_warmup_steps == effective_warmup_steps
    assert cfg.run_provenance.objective.requested_full_reconstruction_only_warmup_steps == full_warmup_steps
else:
    assert effective_warmup_steps == full_warmup_steps == 0
optimizer, scheduler = cfg.model.optimizer, cfg.model.scheduler
unite_recipe = method == STOPGRAD_UNITE_METHOD
scaled_muon = experiment.endswith("_200m_muon_lr1e5_s42")
scaled_adamw = experiment.endswith("_200m_adamw_lr1e5_s42")
scaled_200m = scaled_muon or scaled_adamw
if unite_recipe:
    assert optimizer._target_ == (
        "egomimic.utils.unite_optim.ReleasedUniteCompositeOptimizer"
    )
    assert optimizer._partial_ is True and optimizer.lr == 1e-4
    assert list(optimizer.betas) == [0.9, 0.999] and optimizer.eps == 1e-6
    assert optimizer.adamw_weight_decay == optimizer.muon_weight_decay == 0.0
    assert optimizer.muon_momentum == 0.95
    assert optimizer.muon_adjust_lr_fn == "match_rms_adamw"
    assert cfg.model.optimizer_named_parameters is True
    assert scheduler._target_ == (
        "egomimic.utils.unite_optim.released_unite_two_stage_scheduler"
    )
    assert scheduler.warmup_steps == 8000
    assert scheduler.decay_start_1_steps == 12000
    assert scheduler.decay_end_1_steps == 20000
    assert scheduler.decay_start_2_steps == scheduler.decay_end_2_steps == 1200000
    assert scheduler.base_lr_1 == 1e-4
    assert scheduler.base_lr_2 == scheduler.final_lr == 5e-5
elif scaled_muon:
    assert optimizer._target_ == (
        "egomimic.utils.unite_optim.ReleasedUniteCompositeOptimizer"
    )
    assert optimizer._partial_ is True and optimizer.lr == 1e-5
    assert list(optimizer.betas) == [0.9, 0.999] and optimizer.eps == 1e-8
    assert optimizer.adamw_weight_decay == optimizer.muon_weight_decay == 1e-4
    assert optimizer.muon_momentum == 0.95
    assert optimizer.muon_adjust_lr_fn == "match_rms_adamw"
    assert cfg.model.optimizer_named_parameters is True
else:
    assert optimizer._target_ == "torch.optim.AdamW" and optimizer._partial_ is True
    assert optimizer.lr == (1e-5 if scaled_adamw else 3e-5)
    assert list(optimizer.betas) == [0.9, 0.999]
    assert optimizer.eps == 1e-8 and optimizer.weight_decay == 1e-4
    if scaled_adamw:
        assert cfg.model.optimizer_named_parameters is False
if not unite_recipe and not cotrain:
    assert scheduler.max_steps == 240000 and scheduler.warmup_steps == 8000
    assert scheduler.warmup_start_factor == 0.1
    assert scheduler.eta_min == (1e-6 if scaled_200m else 3e-6)
assert cfg.trainer.accelerator == "gpu"
assert cfg.trainer.devices == cfg.trainer.num_nodes == 1
assert cfg.trainer.strategy == "auto" and cfg.trainer.sync_batchnorm is False
assert str(cfg.trainer.precision) == "bf16"
assert cfg.trainer.accumulate_grad_batches == 1
if run_kind == "smoke":
    assert cfg.trainer.max_steps == cfg.trainer.limit_train_batches == 2
    assert cfg.trainer.val_check_interval == 2
    assert cfg.trainer.limit_val_batches == 1
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 1
    assert cfg.model.gradient_telemetry_cadence == 2
else:
    assert run_kind == "full"
    assert cfg.trainer.max_steps == full_max_steps
    assert cfg.trainer.limit_train_batches == 1.0
    assert math.isclose(float(cfg.trainer.limit_val_batches), full_limit_val_batches)
    assert cfg.trainer.val_check_interval == full_validate_every
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == full_checkpoint_every
    assert cfg.model.gradient_telemetry_cadence == 100
assert cfg.callbacks.model_checkpoint.save_top_k == -1
assert cfg.callbacks.model_checkpoint.save_last == "link"
assert "{step}" in str(cfg.callbacks.model_checkpoint.filename)
expected_sources = {source, second_source} if cotrain else {source}
assert set(cfg.data.train_datasets) == set(cfg.data.valid_datasets) == expected_sources
assert cfg.data.train_datasets[source].resolver.folder_path == dataset
assert cfg.data.valid_datasets[source].resolver.folder_path == dataset
assert cfg.data.train_dataloader_params[source].batch_size == 32
assert cfg.data.valid_dataloader_params[source].batch_size == valid_batch_size
if cotrain:
    assert cfg.data.train_datasets[second_source].resolver.folder_path == second_dataset
    assert cfg.data.valid_datasets[second_source].resolver.folder_path == second_dataset
    assert cfg.data.train_dataloader_params[second_source].batch_size == 32
    assert cfg.data.valid_dataloader_params[second_source].batch_size == valid_batch_size
assert str(cfg.norm_stats.precomputed_norm_path) == norm
assert cfg.norm_stats.save_cache_dir is None
diagnostics = cfg.evaluator.action_flow_diagnostics
if method != LIKELIHOOD_METHOD:
    assert diagnostics.enabled is True and diagnostics.max_batches_per_rank == 1
    assert list(diagnostics.raw_noise_levels) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert diagnostics.max_samples == 16 and diagnostics.jacobian_samples == 2
    assert diagnostics.capture_activations is (method != GRAPH_METHOD)
    assert diagnostics.cknna_k == (0 if method == GRAPH_METHOD else 10)
    assert diagnostics.validation_view.world_size == 1
    assert diagnostics.validation_view.per_rank_batch_size == valid_batch_size
    if scaled_h512:
        assert {int(k): int(v) for k, v in diagnostics.activation_layer_map.items()} == {
            0: 0,
            1: 13,
        }
pipeline = instantiate(cfg.model.pipeline, device="cpu")
if scaled_h512:
    stages = tuple(pipeline.pipeline.stages)
    encoder_stage, field_stage, decoder_stage = stages[4], stages[6], stages[7]
    assert field_stage.flow_clean_gradient_mode == "all_stopgrad"
    assert field_stage.field.backbone.hidden_dim == 512
    assert field_stage.field.backbone.depth == 14
    assert field_stage.field.backbone.num_heads == 16
    if cotrain:
        assert tuple(encoder_stage.encoder) == (source, second_source)
        assert tuple(decoder_stage.decoder) == (source, second_source)
        assert encoder_stage.encoder[source] is not encoder_stage.encoder[second_source]
        assert decoder_stage.decoder[source] is not decoder_stage.decoder[second_source]
        assert encoder_stage.encoder[source].input_dim == 4
        assert encoder_stage.encoder[second_source].input_dim == 6
        assert decoder_stage.decoder[source].action_dim == 4
        assert decoder_stage.decoder[second_source].action_dim == 6
    else:
        assert encoder_stage.encoder.input_dim == action_dim
        assert decoder_stage.decoder.action_dim == action_dim
    for module in (
        (
            *encoder_stage.encoder.values(),
            *decoder_stage.decoder.values(),
        )
        if cotrain
        else (encoder_stage.encoder, decoder_stage.decoder)
    ):
        width = (
            module.backbone.hidden_dim
            if hasattr(module, "backbone")
            else module.hidden_dim
        )
        assert width == 512
elif cotrain:
    stages = tuple(pipeline.pipeline.stages)
    if speed_diagnostic:
        validate_speed_pipeline = typed.validate_speed_pipeline
        validate_speed_pipeline(cfg, pipeline)
        encoder_stage = next(s for s in stages if type(s).__name__ == "RoutedContentEncoderStage")
        decoder_stage = next(s for s in stages if type(s).__name__ == "RoutedContentDecoderStage")
    elif experiment.endswith("action_flow_chain_native4_points6_manual4919_h384_150k_s42"):
        encoder_stage, decoder_stage = stages[4], stages[7]
    else:
        encoder_stage, decoder_stage = stages[3], stages[6]
    assert tuple(encoder_stage.encoder) == (source, second_source)
    assert tuple(decoder_stage.decoder) == (source, second_source)
    assert encoder_stage.encoder[source] is not encoder_stage.encoder[second_source]
    assert decoder_stage.decoder[source] is not decoder_stage.decoder[second_source]
else:
    _validate_topology(cfg, pipeline)
    _validate_dimensions_and_modules(cfg, tuple(pipeline.pipeline.stages))
count = sum(parameter.numel() for parameter in pipeline.nets.parameters())
if scaled_h512:
    assert expected_parameter_count > 0
    assert count == expected_parameter_count, count
elif cotrain:
    if speed_diagnostic:
        assert count == expected_parameter_count, count
    elif experiment.endswith("action_flow_chain_native4_points6_manual4919_h384_150k_s42"):
        inference_parameters = {
            id(parameter): parameter
            for stage in pipeline.pipeline.stages
            if not getattr(stage, "train_only", False)
            for parameter in stage.parameters()
        }
        inference_count = sum(parameter.numel() for parameter in inference_parameters.values())
        assert count == 151_990_970, count
        assert inference_count == expected_parameter_count == 86_539_098, inference_count
        assert source == "pushshapes_sim_chain_gripper_native4"
        assert second_source == "pushshapes_sim_chain_gripper"
        assert dataset == second_dataset
        assert set(cfg.run_provenance.action_contract.representations) == {source, second_source}
        assert list(cfg.run_provenance.architecture.action_shapes[source]) == [16, 4]
        assert list(cfg.run_provenance.architecture.action_shapes[second_source]) == [16, 6]
    else:
        assert count > 90_000_000, count
elif method == STOPGRAD_UNITE_METHOD:
    assert count == expected_parameter_count, count
elif method in (LEGACY_METHOD, STOPGRAD_METHOD):
    if scaled_200m:
        expected_count = 199_754_837
    elif experiment.endswith("_codec98k_s42"):
        expected_count = 50_801_685
    else:
        expected_count = 50_725_221
    assert count == expected_count, count
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
    AF_PLANNED_CHECKPOINT_COUNT=$((
      (AF_FULL_MAX_STEPS + AF_FULL_CHECKPOINT_EVERY - 1)
        / AF_FULL_CHECKPOINT_EVERY
    ))
    test "$AF_PLANNED_CHECKPOINT_COUNT" -ge 1 || \
      die "planned checkpoint count is invalid"
    "$AF_PYTHON" "$AF_STORAGE_VALIDATOR" \
      --target "$AF_OUTPUT_DIR" \
      --planned-checkpoint-count "$AF_PLANNED_CHECKPOINT_COUNT" \
      --measured-checkpoint-bytes "$AF_SMOKE_CHECKPOINT_BYTES" \
      --safety-reserve-bytes "$AF_STORAGE_SAFETY_RESERVE_BYTES" \
      --output "$FULL_STORAGE_ARTIFACT"
  else
    test -s "$FULL_STORAGE_ARTIFACT" || \
      die "restarted full run lacks initial storage proof"
  fi
fi

source "$AF_REPO/scripts/train/action_flow_run_child.sh"

# A successful requeue request also returns zero, but is not training completion.
# The runner writes COMPLETE.json only for successful child completion.
if test "$ICE_BATCH_BOUNDARY_FORWARDED" = 1 && test ! -s "$AF_OUTPUT_DIR/COMPLETE.json"; then
  printf '[requeue] boundary handled; completion verification belongs to the final attempt\n'
  exit 0
fi

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
  SMOKE_WEIGHT_ARGS=()
  if test "$AF_METHOD" != gaussian_bridge_likelihood; then
    SMOKE_WEIGHT_ARGS=(--expected-reconstruction-weight "$AF_RECONSTRUCTION_WEIGHT" --expected-flow-weight "$AF_FLOW_WEIGHT")
  fi
  ROUTED_SMOKE_ARGS=()
  if test "$AF_SCALED_H512" = true || test "$AF_PAIRED_CHAIN_MANUAL4919" = true || test "$AF_SPEED_DIAGNOSTIC" = true; then
    ROUTED_SMOKE_ARGS+=(--expected-parameter-count "${AF_EXPECTED_SMOKE_PARAMETER_COUNT:-$AF_EXPECTED_PARAMETER_COUNT}")
    if test "$AF_COTRAIN" = true; then
      ROUTED_SMOKE_ARGS+=(
        --expected-second-split-sha256 "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256"
        --expected-second-content-manifest-sha256 "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256"
        --expected-second-dataset-content-aggregate-sha256 "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256"
      )
    fi
  fi
  "$AF_PYTHON" "$AF_SMOKE_VERIFIER" "$AF_OUTPUT_DIR" \
    --expected-head "$AF_EXPECTED_HEAD" \
    --expected-experiment "$AF_EXPERIMENT" \
    "${SMOKE_WEIGHT_ARGS[@]}" \
    --expected-split-sha256 "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
    --expected-normalization-sha256 "$AF_EXPECTED_NORM_SHA256" \
    --expected-content-manifest-sha256 \
      "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
    --expected-dataset-content-aggregate-sha256 \
      "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" \
    --expected-config-sha256 "$(sha256 "$AF_OUTPUT_DIR/.hydra/config.yaml")" \
    --expected-preflight-sha256 "$AF_EXPECTED_PREFLIGHT_SHA256" \
    "${ROUTED_SMOKE_ARGS[@]}"
  printf '[smoke] ALL_PASS result=%s wandb=https://wandb.ai/%s/%s/runs/%s\n' \
    "$AF_OUTPUT_DIR/SMOKE_RESULT.json" "$AF_WANDB_ENTITY" \
    "$AF_WANDB_PROJECT" "$AF_WANDB_RUN_ID"
else
  test -s "$AF_OUTPUT_DIR/COMPLETE.json" || die "full run lacks COMPLETE.json"
  printf '[full] ALL_PASS wandb=https://wandb.ai/%s/%s/runs/%s\n' \
    "$AF_WANDB_ENTITY" "$AF_WANDB_PROJECT" "$AF_WANDB_RUN_ID"
fi
