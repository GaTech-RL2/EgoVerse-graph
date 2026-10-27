#!/bin/bash
set -euo pipefail
umask 022
root=${1:?Immutable preparation root}; request_sha=${2:?Exact request SHA}; files_sha=${3:?Exact payload SHA}
[[ "$root" == /workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/full-v5 ]]
[[ "$request_sha" =~ ^[0-9a-f]{64}$ && "$files_sha" =~ ^[0-9a-f]{64}$ ]]
finish() {
  code=$?; trap - EXIT
  if ! (set -o noclobber; printf '%s\n' "$code" > "$root/host-exit-$SLURM_JOB_ID.txt"); then
    [[ "$code" != 0 ]] || code=74
  fi
  /usr/bin/sync -f "$root/host-exit-$SLURM_JOB_ID.txt" || { [[ "$code" != 0 ]] || code=74; }
  exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$files_sha" "$root/FILES.sha256" | sha256sum --check --status
cd "$root"; sha256sum --check --status FILES.sha256
request="$root/ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json"
printf '%s  %s\n' "$request_sha" "$request" | sha256sum --check --status
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B "$root/guard_full.py" --root "$root"
ops_sha=$(sha256sum "$root/ops/FILES.sha256"); ops_sha=${ops_sha%% *}
# The unchanged maintained launcher performs source/cache/runtime admission,
# all fifteen20k checkpoints/validation and strict300kTOTAL full postflight.
bash "$root/ops/full.sbatch" "$root/ops" "$request" "$request_sha" "$ops_sha"
