#!/bin/bash
# Evidence only: no model/container/cache work, training or GPU request.
set -euo pipefail
root=${1:?Exact operations root}
spec=${2:?Exact observer spec}
spec_sha=${3:?Observer spec SHA256}
files_sha=${4:?Operations manifest SHA256}
[[ "$root" == /workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/* || "$root" == /workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/* ]]
[[ "$spec" == "$root"/observer-specs/*.json && "$spec_sha" =~ ^[0-9a-f]{64}$ && "$files_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" ]]
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
cd "$root"
timeout --kill-after=5s 120s sha256sum --check --status < <(printf '%s  %s\n' "$files_sha" "$root/FILES.sha256")
timeout --kill-after=5s 120s sha256sum --check --status FILES.sha256
timeout --kill-after=5s 120s sha256sum --check --status < <(printf '%s  %s\n' "$spec_sha" "$spec")
# The allocated host, not the Python-free gateway, supplies this interpreter.
timeout --kill-after=10s 480s /usr/bin/python3 -B "$root/phase_observer.py" capture "$spec" "$spec_sha" "$files_sha"
