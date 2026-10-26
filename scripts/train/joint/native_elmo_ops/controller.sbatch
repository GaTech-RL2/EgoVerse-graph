#!/bin/bash
# CPU orchestration only. Host exit never certifies any child native terminal.
set -euo pipefail
root=${1:?Operations root}
spec=${2:?Controller specification}
spec_sha=${3:?Controller SHA256}
files_sha=${4:?Operations manifest SHA256}
[[ "$root" == /workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/* ]]
[[ "$spec" == "$root"/controller-specs/*.json ]]
[[ "$spec_sha" =~ ^[0-9a-f]{64}$ && "$files_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$(id -u)" == 41003 && -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
finish() {
  code=$?
  trap - EXIT
  (set -o noclobber; printf '%s\n' "$code" > "$root/status/controller-$SLURM_JOB_ID.exit") || code=74
  exit "$code"
}
trap finish EXIT
printf '%s  %s\n' "$files_sha" "$root/FILES.sha256" | sha256sum --check --status
(cd "$root" && timeout --kill-after=5s 120s sha256sum --check --status FILES.sha256)
printf '%s  %s\n' "$spec_sha" "$spec" | sha256sum --check --status
[[ -x /usr/bin/python3 ]]
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME WANDB_API_KEY PIXEL_WANDB_NETRC
timeout --kill-after=10s 900s /usr/bin/python3 -I -B "$root/controller.py" allocated "$spec" "$spec_sha" "$files_sha"
