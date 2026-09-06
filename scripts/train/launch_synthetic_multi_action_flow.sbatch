#!/bin/bash
# Generic immutable launcher for SyntheticMultiActionAdapterFlow configs.

set -Eeuo pipefail

SOURCE=${SOURCE:?set exact clean source checkout}
EXPECTED_SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT:?set exact source commit}
CONFIG=${CONFIG:?set absolute config path}
EXPECTED_CONFIG_SHA256=${EXPECTED_CONFIG_SHA256:?set config SHA256}
PYTHON=${PYTHON:-/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python}

test "$SOURCE" = "$(cd "$SOURCE" && pwd -P)"
test "$CONFIG" = "$(cd "$(dirname "$CONFIG")" && pwd -P)/$(basename "$CONFIG")"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain=v1 --untracked-files=all)"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256"
test -x "$PYTHON"

OUTPUT_DIR=$(sed -n 's/^[[:space:]]*"output_dir": "\([^"]*\)",$/\1/p' "$CONFIG")
test -n "$OUTPUT_DIR"
test ! -e "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_DIR")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$SOURCE"
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec "$PYTHON" scripts/train/train_synthetic_multi_action_flow.py --config "$CONFIG"
