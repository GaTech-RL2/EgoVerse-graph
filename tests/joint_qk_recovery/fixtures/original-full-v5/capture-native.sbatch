#!/bin/bash
set -euo pipefail
stage=${1:?Immutable GPU full root}
parent=${2:?Exact GPU full parent job}
request_sha=${3:?Exact request hash}
files_sha=${4:?Exact operations manifest hash}
[[ "$stage" == /workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/full-v5 && "$parent" =~ ^[0-9]+$ ]]
[[ "$request_sha" =~ ^[0-9a-f]{64}$ && "$files_sha" =~ ^[0-9a-f]{64}$ ]]
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
# Actual parent/full-steps/observer raw queries precede slow NFS proof reads.
# These raw files have NO terminal authority until the later authentication.
/usr/bin/python3 -B "$stage/early_native_capture.py" collect "$stage" "$parent" "$request_sha" "$files_sha"
printf '%s  %s\n' "$request_sha" "$stage/ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json" | sha256sum --check --status
printf '%s  %s\n' "$files_sha" "$stage/FILES.sha256" | sha256sum --check --status
cd "$stage"
sha256sum --check --status FILES.sha256
# Unknown/purged queries retain raw evidence, but this fails closed.
/usr/bin/python3 -B "$stage/early_native_capture.py" authenticate "$stage" "$parent" "$request_sha" "$files_sha"
