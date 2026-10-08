# Typed checkpoint continuation from the canonical launcher's shared argument vector.
set -Eeuo pipefail
required AF_RESUME_PHASE
case "$AF_RESUME_PHASE" in preflight|smoke|full) ;; *) die 'invalid resume phase';; esac
required AF_RESUME_CACHED_PREFLIGHT
required AF_RESUME_CACHED_PREFLIGHT_SHA256
required AF_RESUME_BASE_CONFIG
required AF_RESUME_BASE_CONFIG_SHA256
required AF_RESUME_INITIAL_VALIDATOR
required AF_RESUME_INITIAL_VALIDATOR_SHA256
for spec in "$AF_RESUME_CACHED_PREFLIGHT_SHA256:$AF_RESUME_CACHED_PREFLIGHT" "$AF_RESUME_BASE_CONFIG_SHA256:$AF_RESUME_BASE_CONFIG" "$AF_RESUME_INITIAL_VALIDATOR_SHA256:$AF_RESUME_INITIAL_VALIDATOR"; do
  test "$(sha256 "${spec#*:}")" = "${spec%%:*}" || die 'cached resume authority mismatch'
done
ATTEMPT=$AF_OUTPUT_DIR/provenance/restart-0
mkdir -p "$ATTEMPT"
cd "$AF_REPO"
BASE_OVERRIDES+=("++run_provenance.preflight_result_sha256=$AF_RESUME_CACHED_PREFLIGHT_SHA256")
if test "$AF_RUN_KIND" = full && test -n "${AF_EXPECTED_SMOKE_SHA256:-}"; then
  BASE_OVERRIDES+=("++run_provenance.smoke_result_sha256=$AF_EXPECTED_SMOKE_SHA256")
fi
"$AF_PYTHON" -m egomimic.trainHydra "${BASE_OVERRIDES[@]}" --cfg job --resolve > "$ATTEMPT/resolved_config.yaml"
"$AF_PYTHON" "$AF_RESUME_DRIVER_REPO/scripts/train/validate_action_flow_checkpoint_resume.py" "$ATTEMPT/resolved_config.yaml" "$ATTEMPT/RESUME_PREFLIGHT.json"
if test "$AF_RESUME_PHASE" = preflight; then exit 0; fi
required AF_PREFLIGHT_RESULT
required AF_EXPECTED_PREFLIGHT_SHA256
test "$(sha256 "$AF_PREFLIGHT_RESULT")" = "$AF_EXPECTED_PREFLIGHT_SHA256" || die 'resume preflight gate mismatch'
if test "$AF_RESUME_PHASE" = full; then
  required AF_SMOKE_RESULT
  required AF_EXPECTED_SMOKE_SHA256
  test "$(sha256 "$AF_SMOKE_RESULT")" = "$AF_EXPECTED_SMOKE_SHA256" || die 'resume smoke gate mismatch'
  "$AF_PYTHON" - "$AF_SMOKE_RESULT" <<'PY'
import json,os,sys
s=json.load(open(sys.argv[1]));assert s['status']=='PASS'
assert s['checkpoint']['strict_checkpoint_reload']=='passed'
assert s['checkpoint']['global_step']==int(os.environ['AF_RESUME_START_STEP'])+4
assert s['identities']['repo_head']==os.environ['AF_EXPECTED_HEAD']
PY
fi
cp "$AF_RESUME_CACHED_PREFLIGHT" "$ATTEMPT/PREFLIGHT_RESULT.json"
cp "$LAUNCHER" "$ATTEMPT/launch_action_flow_usocket.sbatch"
if test "$AF_RUN_KIND" = full; then cp "$AF_SMOKE_RESULT" "$ATTEMPT/SMOKE_RESULT.json"; fi
scontrol show job -dd -o "$SLURM_JOB_ID" > "$ATTEMPT/slurm_job.txt"
"$AF_PYTHON" "$AF_SLURM_CONTRACT_VALIDATOR" --record "$ATTEMPT/slurm_job.txt" --expected-job-id "$SLURM_JOB_ID" --expected-account "$AF_EXPECTED_ACCOUNT" --expected-partition "$SLURM_JOB_PARTITION" --expected-qos "$AF_EXPECTED_QOS" --expected-cpus 8 --expected-memory "$AF_EXPECTED_MEMORY" --expected-time-limit "$AF_EXPECTED_TIME_LIMIT" --expected-constraint "$AF_EXPECTED_GPU_CONSTRAINT" --output "$ATTEMPT/SLURM_JOB_CONTRACT.json"
"$AF_PYTHON" "$AF_RUNTIME_LOCK_CAPTURE" --repo "$AF_REPO" --expected-head "$AF_EXPECTED_HEAD" --lock-input "$AF_REPO/pyproject.toml" --lock-input "$AF_REPO/uv.lock" --output "$ATTEMPT/runtime-lock.json"
# The scientific preflight receipt remains the verified dataset/runtime cache;
# the exact changed argument vector is independently audited above.
"$AF_PYTHON" "$AF_GPU_PROBE" --expected-world-size 1 --allowed-gpu-name 'NVIDIA H100 80GB HBM3' --allowed-gpu-name 'NVIDIA H200' --output "$ATTEMPT/gpu_probe.json"
export AF_CHECKPOINT_VALIDATOR=$AF_RESUME_INITIAL_VALIDATOR
SRUN=$(command -v srun)
source "$AF_RESUME_DRIVER_REPO/scripts/train/action_flow_run_child_resume.sh"
if test "$AF_RESUME_PHASE" = smoke; then
  export AF_RESUME_SMOKE_TARGET_STEP=$MAX_STEPS
  "$AF_PYTHON" "$AF_RESUME_DRIVER_REPO/scripts/train/verify_speed_smoke_resume.py" "$AF_OUTPUT_DIR" \
   --expected-experiment "$AF_EXPERIMENT" --expected-head "$AF_EXPECTED_HEAD" \
   --expected-config-sha256 "$(sha256 "$AF_OUTPUT_DIR/.hydra/config.yaml")" \
   --expected-split-sha256 "$AF_EXPECTED_SPLIT_MANIFEST_SHA256" \
   --expected-normalization-sha256 "$AF_EXPECTED_NORM_SHA256" \
   --expected-content-manifest-sha256 "$AF_EXPECTED_CONTENT_MANIFEST_SHA256" \
   --expected-dataset-content-aggregate-sha256 "$AF_EXPECTED_DATASET_CONTENT_AGGREGATE_SHA256" \
   --expected-second-split-sha256 "$AF_EXPECTED_SECOND_SPLIT_MANIFEST_SHA256" \
   --expected-second-content-manifest-sha256 "$AF_EXPECTED_SECOND_CONTENT_MANIFEST_SHA256" \
   --expected-second-dataset-content-aggregate-sha256 "$AF_EXPECTED_SECOND_DATASET_CONTENT_AGGREGATE_SHA256" \
   --expected-parameter-count "$AF_EXPECTED_PARAMETER_COUNT" \
   --expected-preflight-sha256 "$AF_RESUME_CACHED_PREFLIGHT_SHA256" \
   --expected-reconstruction-weight 1 --expected-flow-weight 1
fi
