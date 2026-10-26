#!/bin/bash
set -euo pipefail
root=${1:?Operations root}
mode=${2:?Mode}
spec=${3:?Execution specification}
spec_sha=${4:?Specification hash}
files_sha=${5:?Operations manifest hash}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export PYTHONDONTWRITEBYTECODE=1
# Verify private client bytes and actual authenticated scontrol+squeue before
# touching the data cache or entering runtime_bootstrap (which imports Torch).
exports=$(python3 "$root/runtime_guard.py" inside "$spec" "$spec_sha" "$mode" "$files_sha")
eval "$exports"
if [[ "$mode" != capture ]]; then
  runtime_exports=$(python3 "$root/lambda_runtime.py" "$ELMO_PARENT_REQUEST" --inside)
  eval "$runtime_exports"
  export PIXEL_SOURCE_ROOT="$ELMO_SOURCE" PIXEL_FA4_ROOT="$JOINT_FA4_ROOT" PIXEL_FA3_ROOT="$JOINT_FA3_ROOT" PIXEL_TASK="$JOINT_RUNTIME_TASK"
  export PYTHONPATH="$root:$PIXEL_SOURCE_ROOT:$PIXEL_FA3_ROOT:$PIXEL_FA4_ROOT/runtime/nvidia_cutlass_dsl/dsl_packages:$PIXEL_FA4_ROOT/torch-runtime:$PIXEL_FA4_ROOT/runtime:$JOINT_EXTRA_PYTHONPATH"
  export PATH="$(dirname "$JOINT_ACTUAL_PYTHON"):$PATH"
  export NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
  export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
  cache="$JOINT_NODE_SCRATCH/jit/$SLURM_JOB_ID"
  export CUTE_DSL_CACHE_DIR="$cache/cute" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
  export TMPDIR="$JOINT_JOB_TMP" XDG_CACHE_HOME="$cache/cache" WANDB_CACHE_DIR="$cache/cache/wandb" WANDB_DATA_DIR="$cache/data/wandb"
  mkdir -p "$CUTE_DSL_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
fi
if [[ "$mode" != diagnostic && "$mode" != full ]]; then
  [[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
  export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
fi
if [[ "$mode" == full ]]; then
  export PIXEL_WANDB_NETRC=/run/pixel-wandb.netrc
else
  unset WANDB_API_KEY PIXEL_WANDB_NETRC
  export WANDB_MODE=disabled
fi
exec python3 "$root/runtime_guard.py" dispatch "$spec" "$spec_sha" "$mode" "$files_sha"
