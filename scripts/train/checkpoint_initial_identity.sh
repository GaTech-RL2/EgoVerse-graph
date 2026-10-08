#!/usr/bin/env bash
set -Eeuo pipefail
if test "$(realpath "$1")" = "$(realpath "$AF_INITIAL_CHECKPOINT")"; then
  test "$(sha256sum "$1" | cut -d' ' -f1)" = "$AF_INITIAL_CHECKPOINT_SHA256"
  export ICE_EXPECTED_RUN_ID="$AF_PARENT_WANDB_RUN_ID"
fi
exec "$AF_PYTHON" "$AF_REPO/scripts/ice/validate_lightning_checkpoint.py" "$@"
