#!/bin/bash
# Immutable held submission/release is owned by execution_spec.py, never this script.
set -euo pipefail
root=${1:?Exact operations root}
mode=${2:?capture inspect diagnostic finalize full}
spec=${3:?Exact execution specification}
spec_sha=${4:?Exact specification SHA256}
files_sha=${5:?Exact operations manifest SHA256}
[[ "$root" == /workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/* || "$root" == /workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/* ]]
[[ "$mode" == capture || "$mode" == inspect || "$mode" == diagnostic || "$mode" == finalize || "$mode" == full ]]
[[ "$spec" == "$root"/execution/*.json && "$spec_sha" =~ ^[0-9a-f]{64}$ && "$files_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
timeout --kill-after=10s 120s bash -c '
  set -euo pipefail
  printf "%s  %s\n" "$2" "$1/FILES.sha256" | sha256sum --check --status
  cd "$1" && sha256sum --check --status FILES.sha256
' elmo-wrapper-manifest "$root" "$files_sha"
printf '%s  %s\n' "$spec_sha" "$spec" | timeout --kill-after=10s 120s sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
source "$root/launcher_common.sh"
child= mirror=
receipt="$root/status/$mode-$SLURM_JOB_ID.exit"
finish() {
  code=$?
  trap - EXIT
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
  if [[ -n "$mirror" ]]; then kill "$mirror" 2>/dev/null || true; wait "$mirror" 2>/dev/null || true; fi
  if [[ "$mode" == full && -d "${ELMO_LOCAL_LOGS:-}" ]]; then
    destination="$ELMO_ATTEMPT/startup-log-mirror"
    [[ ! -f "$ELMO_OUTPUT/.training-admitted" ]] || destination="$ELMO_OUTPUT/log-mirror"
    mirror_once "$ELMO_LOCAL_LOGS" "$destination" || code=1
  fi
  (set -o noclobber; printf '%s\n' "$code" > "$receipt")
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
# Allocated host must authenticate actual native ownership before source/cache I/O.
exports=$(timeout --kill-after=10s 300s python3 "$root/runtime_guard.py" host "$spec" "$spec_sha" "$mode" "$files_sha")
eval "$exports"
native_exports=$(timeout --kill-after=10s 600s python3 "$root/native_client.py" "$spec" "$spec_sha" "$mode" "$files_sha")
eval "$native_exports"
mounts="/workspace:/workspace:rw,$ELMO_SCRATCH:$ELMO_SCRATCH:rw,$ELMO_NATIVE_MOUNTS"
workdir="$root"
if [[ "$mode" != capture ]]; then
  runtime_exports=$(timeout --kill-after=10s 300s python3 "$root/lambda_runtime.py" "$ELMO_PARENT_REQUEST")
  eval "$runtime_exports"
  source_cache="$JOINT_NODE_SCRATCH/source/$SLURM_JOB_ID"
  timeout --kill-after=10s 600s bash "$root/lambda_stage_source.sh" "$ELMO_ARCHIVE" "$ELMO_ARCHIVE_SHA" "$ELMO_SOURCE_MANIFEST" "$ELMO_SOURCE_MANIFEST_SHA" "$source_cache"
  cp -- "$source_cache/source-stage.json" "$ELMO_ATTEMPT/source-stage.json"
  mounts+=",$JOINT_JOB_TMP:$JOINT_JOB_TMP:rw,$(cat "$JOINT_RUNTIME_MOUNTS"),$ELMO_CACHE_ROOT:$ELMO_DATA_ROOT:ro,$source_cache/source:$ELMO_SOURCE:ro"
  workdir="$ELMO_SOURCE"
fi
if [[ "$mode" == full ]]; then
  credentials_mount=
  if [[ -f "$HOME/.netrc" ]]; then credentials_mount=",$HOME/.netrc:/run/pixel-wandb.netrc:ro"; fi
  [[ -n "${WANDB_API_KEY:-}" || -n "$credentials_mount" ]] || fail 'Production W&B credential unavailable'
  mounts+="$credentials_mount"
fi
gpu_args=(--gres=none)
if [[ "$mode" == diagnostic || "$mode" == full ]]; then gpu_args=(--gres=gpu:h100:4 --gpu-bind=none); fi
/usr/local/bin/srun --jobid="$SLURM_JOB_ID" --exclusive --exact --unbuffered --nodes=1 --ntasks=1 \
  --cpus-per-task="$ELMO_CPUS" --mem="$ELMO_MEMORY" "${gpu_args[@]}" \
  --container-name="elmo-recovery-$mode-$SLURM_JOB_ID" --container-image="$ELMO_CONTAINER_IMAGE" --container-writable \
  --container-mounts="$mounts" --container-workdir="$workdir" \
  --output="$root/logs/$mode-$SLURM_JOB_ID.out" --error="$root/logs/$mode-$SLURM_JOB_ID.err" \
  bash "$root/inside.sh" "$root" "$mode" "$spec" "$spec_sha" "$files_sha" &
child=$!
if [[ "$mode" == full ]]; then
  watch_log_mirror "$child" "$ELMO_LOCAL_LOGS" "$ELMO_OUTPUT/log-mirror" "$ELMO_OUTPUT/.training-admitted" "$ELMO_ATTEMPT/unused" &
  mirror=$!
fi
status=0
wait "$child" || status=$?
child=
exit "$status"
