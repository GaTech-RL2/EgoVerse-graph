#!/bin/bash
#SBATCH --job-name=unite-candidate-stage
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

set -Eeuo pipefail

ROOT=${ROOT:?set exact ICE experiment root}
PYTHON=${PYTHON:?set exact Python interpreter}
STAGER=${STAGER:?set immutable candidate stager}
MANIFEST=${MANIFEST:?set immutable candidate manifest}
ACCESS_SELECTION=${ACCESS_SELECTION:?set sealed ICE-to-Skynet access selection}
SSH_WRAPPER=${SSH_WRAPPER:?set immutable Skynet SSH wrapper}
EXPECTED_STAGER_SHA256=${EXPECTED_STAGER_SHA256:?set stager SHA256}
EXPECTED_MANIFEST_SHA256=${EXPECTED_MANIFEST_SHA256:?set manifest SHA256}
EXPECTED_SSH_WRAPPER_SHA256=${EXPECTED_SSH_WRAPPER_SHA256:?set SSH wrapper SHA256}
EXPECTED_ACCOUNT=${EXPECTED_ACCOUNT:?set expected Slurm account}
EXPECTED_QOS=${EXPECTED_QOS:?set expected Slurm QOS}
EXPECTED_PARTITION=${EXPECTED_PARTITION:-ice-cpu}

for path in "$ROOT" "$PYTHON" "$STAGER" "$MANIFEST" "$ACCESS_SELECTION" "$SSH_WRAPPER"; do
  [[ "$path" = /* ]]
done
test -x "$PYTHON"
test -f "$STAGER"
test -r "$STAGER"
test -x "$SSH_WRAPPER"
test -s "$MANIFEST"
test -s "$ACCESS_SELECTION"
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_ACCOUNT"
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_PARTITION"
test "${SLURM_JOB_QOS:?}" = "$EXPECTED_QOS"
test -z "${CUDA_VISIBLE_DEVICES:-}"

assert_sha() {
  local expected=$1 path=$2
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]]
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}
assert_sha "$EXPECTED_STAGER_SHA256" "$STAGER"
assert_sha "$EXPECTED_MANIFEST_SHA256" "$MANIFEST"
assert_sha "$EXPECTED_SSH_WRAPPER_SHA256" "$SSH_WRAPPER"

PROOF_DIR=$ROOT/provenance/candidate-transfers
exec "$PYTHON" "$STAGER" \
  --manifest "$MANIFEST" \
  --access-selection "$ACCESS_SELECTION" \
  --authorized-root "$ROOT" \
  --ssh-wrapper "$SSH_WRAPPER" \
  --proof-dir "$PROOF_DIR"
