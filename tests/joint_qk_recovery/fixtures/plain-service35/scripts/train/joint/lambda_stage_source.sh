#!/bin/bash
# Run only on the allocated node. Preserve exact source bytes and all file checks.
set -euo pipefail
archive=${1:?Pinned source archive}
archive_sha=${2:?Archive SHA256}
manifest=${3:?Pinned source manifest}
manifest_sha=${4:?Manifest SHA256}
cache=${5:?Unique allocated node-local destination}
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
[[ "${SLURMD_NODENAME:-}" =~ ^[a-zA-Z0-9._-]+$ ]]
[[ "$archive_sha" =~ ^[0-9a-f]{64}$ && "$manifest_sha" =~ ^[0-9a-f]{64}$ ]]
: "${JOINT_NODE_SCRATCH:?explicit admitted node-local scratch required}"
[[ "$cache" == "$JOINT_NODE_SCRATCH/source/$SLURM_JOB_ID" ]]
[[ ! -e "$cache" ]]
mkdir -p "$JOINT_NODE_SCRATCH/source"
mkdir "$cache" "$cache/source"
cp -- "$archive" "$cache/source.tar.gz"
cp -- "$manifest" "$cache/source.files.sha256"
printf '%s  %s\n' "$archive_sha" "$cache/source.tar.gz" | sha256sum --check --status
printf '%s  %s\n' "$manifest_sha" "$cache/source.files.sha256" | sha256sum --check --status
tar -xzf "$cache/source.tar.gz" -C "$cache/source"
(cd "$cache/source" && sha256sum --check --status "$cache/source.files.sha256")
chmod -R a-w "$cache/source"
(set -o noclobber; printf '{"schema":"joint-node-local-source-v2","status":"PASS","job_id":%s,"node":"%s","archive_sha256":"%s","manifest_sha256":"%s","source":"%s/source","all_manifest_files_verified":true}\n' \
  "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$archive_sha" "$manifest_sha" "$cache" > "$cache/source-stage.json")
cat "$cache/source-stage.json"
