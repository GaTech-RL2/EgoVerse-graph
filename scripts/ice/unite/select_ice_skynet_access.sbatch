#!/bin/bash
#SBATCH --job-name=ice-skynet-access
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --open-mode=append

set -Eeuo pipefail

PYTHON=${PYTHON:?set exact Python interpreter}
PROBE=${PROBE:?set immutable access-selection probe}
PROBE_SHA256=${PROBE_SHA256:?set exact probe SHA256}
ROOT=${ROOT:?set exact immutable ICE experiment root}
SSH_WRAPPER=${SSH_WRAPPER:?set immutable Skynet SSH wrapper}
SSH_WRAPPER_SHA256=${SSH_WRAPPER_SHA256:?set exact wrapper SHA256}
OUTPUT=${OUTPUT:?set unique access-selection output}
CANONICAL_REMOTE=${CANONICAL_REMOTE:?set canonical user@host}
CANONICAL_HOSTNAME=${CANONICAL_HOSTNAME:?set canonical remote hostname}
FALLBACK_REMOTE=${FALLBACK_REMOTE:?set fallback user@host}
FALLBACK_HOSTNAME=${FALLBACK_HOSTNAME:?set fallback remote hostname}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:?set expected remote username}
EXPECTED_ACCOUNT=${EXPECTED_ACCOUNT:?set expected Slurm account}
EXPECTED_QOS=${EXPECTED_QOS:?set expected Slurm QOS}
EXPECTED_PARTITION=${EXPECTED_PARTITION:-ice-cpu}

for path in "$PYTHON" "$PROBE" "$ROOT" "$SSH_WRAPPER" "$OUTPUT"; do
  [[ "$path" = /* ]]
done
test -x "$PYTHON"
test -d "$ROOT"
test -f "$PROBE"
test -r "$PROBE"
test -x "$SSH_WRAPPER"
test ! -e "$OUTPUT"
test ! -e "$OUTPUT.sha256"
test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_ACCOUNT"
test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_PARTITION"
test "${SLURM_JOB_QOS:?}" = "$EXPECTED_QOS"
test -z "${CUDA_VISIBLE_DEVICES:-}"
[[ "$PROBE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$SSH_WRAPPER_SHA256" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$PROBE" | awk '{print $1}')" = "$PROBE_SHA256"
test "$(sha256sum "$SSH_WRAPPER" | awk '{print $1}')" = "$SSH_WRAPPER_SHA256"

exec "$PYTHON" "$PROBE" \
  --ssh-wrapper "$SSH_WRAPPER" \
  --ssh-wrapper-sha256 "$SSH_WRAPPER_SHA256" \
  --authorized-root "$ROOT" \
  --output "$OUTPUT" \
  --canonical-remote "$CANONICAL_REMOTE" \
  --canonical-hostname "$CANONICAL_HOSTNAME" \
  --fallback-remote "$FALLBACK_REMOTE" \
  --fallback-hostname "$FALLBACK_HOSTNAME" \
  --expected-remote-user "$EXPECTED_REMOTE_USER" \
  --expected-account "$EXPECTED_ACCOUNT" \
  --expected-partition "$EXPECTED_PARTITION" \
  --expected-qos "$EXPECTED_QOS"
