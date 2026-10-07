#!/usr/bin/env bash
# Sourced by canonical launcher AFTER its single shared exact BASE array.
set -Eeuo pipefail
test "$AF_NATIVE_LIBERO" = true || die "native phase adapter profile mismatch"
test "${AF_NATIVE_SCHEMA_ALLOW_PENDING_PROOFS:-false}" = false || die "operational dispatch rejects pending proof sentinels"
for variable in SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION SLURM_JOB_QOS SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK; do required "$variable"; done
test "$SLURM_JOB_ACCOUNT" = "$AF_EXPECTED_ACCOUNT" || die "unexpected Slurm account"
partition_allowed "$AF_EXPECTED_PARTITION" "$SLURM_JOB_PARTITION" || die "unexpected Slurm partition"
test "$SLURM_JOB_QOS" = "$AF_EXPECTED_QOS" || die "unexpected Slurm QoS"
test "$SLURM_JOB_NUM_NODES" = 1 && test "$SLURM_NTASKS" = 1 || die "one node/task required"
test "$SLURM_CPUS_PER_TASK" -eq 8 || die "exact eight CPUs required"
phase=$AF_NATIVE_PHASE
if test "$AF_LAUNCH_MODE" = preflight; then
 test "$phase" = preflight && test "$RESTART_COUNT" = 0 || die "native preflight phase/restart mismatch"
 test ! -e "$AF_OUTPUT_DIR" || die "preflight output collision"
 mkdir -p "$AF_OUTPUT_DIR"
 ATTEMPT=$AF_OUTPUT_DIR
else
 test "$AF_LAUNCH_MODE" = run || die "unsupported native mode"
 test "$phase" = "$AF_RUN_KIND" || die "native phase/run-kind mismatch"
 if test "$RESTART_COUNT" = 0; then test ! -e "$AF_OUTPUT_DIR" || die "unique native output collision"; else test -d "$AF_OUTPUT_DIR" || die "native restart output missing"; fi
 mkdir -p "$AF_OUTPUT_DIR/checkpoints" "$AF_OUTPUT_DIR/provenance" "$AF_OUTPUT_DIR/runner-state"
 ATTEMPT=$AF_OUTPUT_DIR/provenance/restart-$RESTART_COUNT
 test ! -e "$ATTEMPT" || die "native attempt collision"
 mkdir "$ATTEMPT"
fi
(set -o noclobber; printf '%s\0' "${BASE_OVERRIDES[@]}" > "$ATTEMPT/exact-phase.argv0")
"$AF_PYTHON" "$AF_RUNTIME_LOCK_CAPTURE" --repo "$AF_REPO" --expected-head "$AF_EXPECTED_HEAD" --lock-input "$AF_REPO/pyproject.toml" --lock-input "$AF_REPO/uv.lock" --output "$ATTEMPT/runtime-lock.json"
export AF_PREFLIGHT_RESULT AF_EXPECTED_PREFLIGHT_SHA256 AF_SMOKE_RESULT AF_EXPECTED_SMOKE_SHA256
if test "$AF_LAUNCH_MODE" = preflight; then
 "$AF_PYTHON" "$AF_REPO/scripts/train/libero_native_launch_contract.py" --action preflight --repo "$AF_REPO" --output "$ATTEMPT" --argv "$ATTEMPT/exact-phase.argv0" --phase "$phase"
 printf '[native preflight] CPU contract passed: %s\n' "$ATTEMPT/PREFLIGHT_RESULT.json"
 return 0
fi
# Prove official phase composition/cache identity before GPU initialization.
"$AF_PYTHON" "$AF_REPO/scripts/train/libero_native_launch_contract.py" --action prepare --repo "$AF_REPO" --output "$ATTEMPT" --argv "$ATTEMPT/exact-phase.argv0" --phase "$phase"
scontrol show job -dd -o "$SLURM_JOB_ID" > "$ATTEMPT/slurm_job.txt"
"$AF_PYTHON" "$AF_SLURM_CONTRACT_VALIDATOR" --record "$ATTEMPT/slurm_job.txt" --expected-job-id "$SLURM_JOB_ID" --expected-account "$AF_EXPECTED_ACCOUNT" --expected-partition "$SLURM_JOB_PARTITION" --expected-qos "$AF_EXPECTED_QOS" --expected-cpus 8 --expected-memory "$AF_EXPECTED_MEMORY" --expected-time-limit "$AF_EXPECTED_TIME_LIMIT" --expected-constraint "$AF_EXPECTED_GPU_CONSTRAINT" --output "$ATTEMPT/SLURM_JOB_CONTRACT.json"
SRUN=$(command -v srun);test -x "$SRUN" || die "srun unavailable";export SRUN
"$SRUN" --overlap --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 --unbuffered "$AF_PYTHON" "$AF_GPU_PROBE" --expected-world-size 1 --allowed-gpu-name 'NVIDIA H100 80GB HBM3' --allowed-gpu-name 'NVIDIA H200' --output "$ATTEMPT/gpu_probe.json"
if test "$phase" = full; then
 measured=$($AF_PYTHON -c 'import json,os; p=json.load(open(os.environ["AF_SMOKE_RESULT"])); v=p["checkpoint"]["file_size_bytes"]; assert isinstance(v,int) and v>0; print(v)')
 "$AF_PYTHON" "$AF_STORAGE_VALIDATOR" --target "$AF_OUTPUT_DIR" --planned-checkpoint-count 16 --measured-checkpoint-bytes "$measured" --safety-reserve-bytes "$AF_STORAGE_SAFETY_RESERVE_BYTES" --output "$ATTEMPT/CHECKPOINT_STORAGE.json"
fi
# Reuse the exact shared canonical training/requeue/signal code, extracted once.
source "$AF_REPO/scripts/train/action_flow_run_child.sh"
if test "${ICE_BATCH_BOUNDARY_FORWARDED:-0}" = 1 && test ! -s "$AF_OUTPUT_DIR/COMPLETE.json"; then return 0; fi
test -s "$AF_OUTPUT_DIR/COMPLETE.json" || die "native child completion missing"
if test "$phase" = smoke; then
 "$AF_PYTHON" "$AF_REPO/scripts/train/libero_native_launch_contract.py" --action finalize --repo "$AF_REPO" --output "$AF_OUTPUT_DIR"
 "$AF_PYTHON" "$AF_SMOKE_VERIFIER" "$AF_OUTPUT_DIR" --experiment "$AF_EXPERIMENT" --expected-head "$AF_EXPECTED_HEAD" --expected-config-sha256 "$(sha256 "$AF_OUTPUT_DIR/.hydra/config.yaml")" --expected-split-sha256 "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" --expected-normalization-sha256 "$AF_EXPECTED_NORM_SHA256" --expected-preflight-sha256 "$AF_EXPECTED_PREFLIGHT_SHA256"
 test -s "$AF_OUTPUT_DIR/SMOKE_RESULT.json" || die "native smoke verifier did not publish a nonempty SMOKE_RESULT.json"
fi
