#!/bin/bash
set -euo pipefail
root=${1:?Operations root}
request=${2:?Exact request}
request_sha=${3:?Bound request hash}
files_sha=${4:?Bound operational manifest}
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$request" == "$root"/requests/*.json && "$request_sha" =~ ^[0-9a-f]{64}$ && "$files_sha" =~ ^[0-9a-f]{64}$ ]]
printf '%s  %s\n' "$request_sha" "$request" | sha256sum --check --status
printf '%s  %s\n' "$files_sha" "$root/FILES.sha256" | sha256sum --check --status
(cd "$root" && sha256sum --check --status FILES.sha256)
source "$root/launcher_common.sh"
export PIXEL_SOURCE_ROOT
PIXEL_SOURCE_ROOT=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source"])' "$request")
world=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["world_size"])' "$request")
output=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["output"])' "$request")
local_logs=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["local_logs"])' "$request")
archive=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_archive"])' "$request")
archive_sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_archive_sha256"])' "$request")
source_manifest=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_manifest"])' "$request")
source_manifest_sha=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_manifest_sha256"])' "$request")
[[ "$world" == 2 || "$world" == 4 ]]
[[ ! -e "$local_logs" ]]
resume=$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1])).get("continuation") is not None))' "$request")
if [[ "$resume" == 0 ]]; then [[ ! -e "$output" ]]; else [[ "$resume" == 1 && -f "$output/.training-admitted" ]]; fi
# The shared native service checks the complete prior-request/checkpoint chain.
runtime_exports=$(python3 "$root/lambda_runtime.py" "$request")
eval "$runtime_exports"
mounts=$(cat "$JOINT_RUNTIME_MOUNTS")
credentials_mount=
if [[ -f "$HOME/.netrc" ]]; then credentials_mount=",$HOME/.netrc:/run/pixel-wandb.netrc:ro"; fi
[[ -n "${WANDB_API_KEY:-}" || -n "$credentials_mount" ]] || fail "W&B credentials unavailable"
mkdir -p "$JOINT_NODE_SCRATCH/jit" "$JOINT_NODE_SCRATCH/logs" "$JOINT_NODE_SCRATCH/source"
name="$(basename "$request" .json)-$SLURM_JOB_ID"
receipt="$root/status/$name.exit"
attempt="$root/status/$name.attempt"
mkdir "$attempt"
child= mirror=
finish() {
  code=$?
  trap - EXIT
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi
  if [[ -n "$mirror" ]]; then kill "$mirror" 2>/dev/null || true; wait "$mirror" 2>/dev/null || true; fi
  if [[ -d "$local_logs" ]]; then
    destination="$output/log-mirror"
    [[ -f "$output/.training-admitted" ]] || destination="$attempt/startup-log-mirror"
    mirror_once "$local_logs" "$destination" || code=1
  fi
  (set -o noclobber; printf '%s\n' "$code" > "$receipt")
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
python3 "$root/verify_joint_data_cache.py" "$request" "$attempt/data-cache-stage.json"
cache_root=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["data_cache"]["root"])' "$request")
data_root=$(python3 -c 'import json,sys,pathlib;print(pathlib.Path(json.load(open(sys.argv[1]))["data_manifest"]).parent)' "$request")
mounts+=",$cache_root:$data_root:ro"
source_cache="$JOINT_NODE_SCRATCH/source/$SLURM_JOB_ID"
bash "$root/lambda_stage_source.sh" "$archive" "$archive_sha" "$source_manifest" "$source_manifest_sha" "$source_cache"
cp -- "$source_cache/source-stage.json" "$attempt/source-stage.json"
# Bind broad writable parents before the verified read-only source child.
# Otherwise a parent node-scratch mount hides the virtual source workdir.
srun --jobid="$SLURM_JOB_ID" --exclusive --exact --unbuffered --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --gres="gpu:h100:$world" --mem=192G --gpu-bind=none \
  --container-name="joint-$name" --container-image="$JOINT_CONTAINER_IMAGE" --container-writable \
  --container-mounts="/workspace:/workspace:rw,$JOINT_NODE_SCRATCH:$JOINT_NODE_SCRATCH:rw,$JOINT_JOB_TMP:$JOINT_JOB_TMP:rw,$mounts$credentials_mount,$source_cache/source:$PIXEL_SOURCE_ROOT:ro" \
  --container-workdir="$PIXEL_SOURCE_ROOT" --output="$root/logs/$name.out" --error="$root/logs/$name.err" \
  bash "$root/lambda_inside.sh" "$root" "$request" &
child=$!
watch_log_mirror "$child" "$local_logs" "$output/log-mirror" "$output/.training-admitted" "$attempt/unused" &
mirror=$!
status=0
wait "$child" || status=$?
child=
exit "$status"
