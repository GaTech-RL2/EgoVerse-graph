#!/bin/bash
# Bind a checkpoint to an explicit immutable UNITE row contract.

set -Eeuo pipefail

checkpoint=${1:?usage: unite_checkpoint_validate_dispatch.sh CHECKPOINT}
PYTHON=${ICE_PYTHON:?set exact Python interpreter}
VALIDATOR=${ICE_CHECKPOINT_VALIDATOR:?set immutable UNITE validator}
VALIDATOR_SHA256=${ICE_CHECKPOINT_VALIDATOR_SHA256:?set validator SHA-256}
SOURCE_CHECKOUT=${ICE_SOURCE_CHECKOUT:?set exact clean source checkout}
EXPECTED_HEAD=${ICE_EXPECTED_SOURCE_COMMIT:?set exact source commit}

for path in "$checkpoint" "$PYTHON" "$VALIDATOR" "$SOURCE_CHECKOUT"; do
  [[ "$path" = /* ]]
done
test -x "$PYTHON"
test -f "$VALIDATOR"
test -e "$SOURCE_CHECKOUT/.git"
[[ "$VALIDATOR_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]]
test "$(sha256sum "$VALIDATOR" | awk '{print $1}')" = "$VALIDATOR_SHA256"
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain --untracked-files=all)"

required_contract=(
  ICE_EXPECTED_TASK_ID ICE_EXPECTED_TOPOLOGY ICE_EXPECTED_NUM_LATENT_TOKENS
  ICE_EXPECTED_PARAMETER_COUNT ICE_EXPECTED_MIN_STEP ICE_EXPECTED_MAX_STEP
  ICE_EXPECTED_SPLIT_SHA256 ICE_EXPECTED_NORM_SHA256 ICE_EXPECTED_WANDB_ENTITY
  ICE_EXPECTED_WANDB_PROJECT ICE_EXPECTED_WANDB_GROUP ICE_EXPECTED_WANDB_ID
  ICE_EXPECTED_WORLD_SIZE ICE_EXPECTED_LR_START ICE_EXPECTED_LR_FINAL
  ICE_EXPECTED_LR_START_STEP ICE_EXPECTED_LR_END_STEP
  ICE_REQUIRE_REBASED_SCHEDULE ICE_REQUIRE_RUNTIME_REQUEUE_CONTRACT ICE_STRICT_LOAD
)
for name in "${required_contract[@]}"; do
  test -n "${!name:-}"
done

export PYTHONPATH=$SOURCE_CHECKOUT
exec "$PYTHON" "$VALIDATOR" "$checkpoint"
