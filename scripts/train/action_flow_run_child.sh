read -r -d '' CHILD_CODE <<'BASH' || true
set -Eeuo pipefail
checkpoint_override=ckpt_path=null
if test -n "${ICE_RESUME_CHECKPOINT:-}"; then
  "$AF_PYTHON" - "$ICE_RESUME_CHECKPOINT" \
    "${ICE_RESUME_CHECKPOINT_SHA256:?}" "${ICE_RESUME_GLOBAL_STEP:?}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
expected, raw_step = sys.argv[2:]
assert int(raw_step) >= 0
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
assert digest.hexdigest() == expected
PY
  case "$ICE_RESUME_CHECKPOINT" in
    *"'"*|*$'\n'*|*$'\r'*)
      printf 'Unsafe checkpoint path for Hydra override: %q\n' \
        "$ICE_RESUME_CHECKPOINT" >&2
      exit 2
      ;;
  esac
  # Preserve '=' and other Hydra grammar characters as literal path content.
  checkpoint_override="ckpt_path='$ICE_RESUME_CHECKPOINT'"
else
  test -z "${ICE_RESUME_CHECKPOINT_SHA256:-}${ICE_RESUME_GLOBAL_STEP:-}"
fi
cd "$AF_REPO"
export PYTHONPATH="$AF_REPO:$(dirname "$AF_CONFIG_VALIDATOR")${AF_EXTRA_PYTHONPATH:+:$AF_EXTRA_PYTHONPATH}"
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_SILENT=true MUJOCO_GL=egl
overlap_args=()
if test "${AF_NATIVE_SCHEDULED_STEP:-false}" = true; then overlap_args+=(--overlap); fi
exec "$SRUN" "${overlap_args[@]}" --nodes=1 --ntasks=1 --gpus-per-task=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --cpu-bind=none --kill-on-bad-exit=1 --unbuffered \
  /bin/bash -c 'set -Eeuo pipefail; cd "$AF_REPO"; exec "$AF_PYTHON" -m egomimic.trainHydra "$@"' \
  action-flow-train "$@" \
  "$checkpoint_override"
BASH

RUNNER_ARGS=(
  --state-dir "$AF_OUTPUT_DIR/runner-state"
  --checkpoint-glob "$AF_OUTPUT_DIR/checkpoints/*.ckpt"
  --checkpoint-validator "$AF_CHECKPOINT_VALIDATOR"
  --checkpoint-signal USR2
  --checkpoint-forwarding slurm-steps
  --checkpoint-grace-seconds 480
  --max-restarts "$AF_MAX_RESTARTS"
  --requeue-owner runner
  --confirm-child-requeue-disabled
  --completion-sentinel "$AF_OUTPUT_DIR/COMPLETE.json"
)
# Slurm B:USR1 reaches this batch shell only. Preserve the post-run verifier
# while relaying boundary/cancellation signals to the runner's installed handlers.
source "$AF_SIGNAL_RELAY"
ice_run_with_signal_relay "$ATTEMPT/runner-signal-ready.json" \
  "$AF_PYTHON" "$AF_REQUEUE_RUNNER" "${RUNNER_ARGS[@]}" -- \
  /usr/bin/env bash -c "$CHILD_CODE" action-flow-usocket "${BASE_OVERRIDES[@]}"
