# The explicit joint route may never inherit the historical step-wide save transport.
if [[ "${AF_EXPERIMENT:-}" = libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42 || "${AF_EXPERIMENT:-}" = libero/action_flow_libero_spatial_sharpa_mot_s42 ]]; then
  test "${AF_CHECKPOINT_TRANSPORT:-}" = trainer-pidfd || {
    printf '%s\n' 'joint child requires audited trainer-pidfd checkpoint transport' >&2; exit 2;
  }
  for joint_signal_variable in AF_REQUEUE_RUNNER AF_EXPECTED_PID_RUNNER_SHA256 \
    ICE_CANONICAL_REQUEUE_RUNNER ICE_NATIVE_RUNNER_SHA256 ICE_TRAINER_REPO \
    ICE_TRAINER_PYTHON ICE_TRAINER_HEAD ICE_TRAINER_SIGNAL_ADAPTER_SHA256; do
    test -n "${!joint_signal_variable:-}" || {
      printf 'joint PIDfd binding missing: %s\n' "$joint_signal_variable" >&2; exit 2;
    }
  done
  test "$ICE_TRAINER_REPO" = "$AF_REPO" && test "$ICE_TRAINER_PYTHON" = "$AF_PYTHON" \
    && test "$ICE_TRAINER_HEAD" = "$AF_EXPECTED_HEAD" || {
    printf '%s\n' 'joint PIDfd runtime/source binding differs' >&2; exit 2;
  }
  test "$ICE_CANONICAL_REQUEUE_RUNNER" = "$AF_REPO/scripts/ice/ice_requeue_runner.py" || exit 2
  test "$(sha256 "$AF_REQUEUE_RUNNER")" = "$AF_EXPECTED_PID_RUNNER_SHA256" || exit 2
  test "$(sha256 "$ICE_CANONICAL_REQUEUE_RUNNER")" = "$ICE_NATIVE_RUNNER_SHA256" || exit 2
  joint_signal_adapter=$(dirname "$AF_REQUEUE_RUNNER")/trainer_pid_signal.py
  test "$(sha256 "$joint_signal_adapter")" = "$ICE_TRAINER_SIGNAL_ADAPTER_SHA256" || exit 2
  unset joint_signal_variable joint_signal_adapter
fi

# The phase adapter can check transport before output creation or a GPU probe.
if test "${AF_CHILD_TRANSPORT_AUDIT_ONLY:-false}" = true; then return 0; fi

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
