#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTIVATE="$SOURCE_DIR/.venv/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
  echo "missing repository activation file: $ACTIVATE" >&2
  exit 1
fi

cd "$SOURCE_DIR"
source "$ACTIVATE"
python -m pytest -q \
  tests/test_synthetic_decoder_inversion_flow.py \
  tests/test_synthetic_trajectory_eval.py \
  tests/test_synthetic_trainer_resume.py
ruff check \
  egomimic/synthetic/decoder_inversion_flow.py \
  scripts/experiments/build_decoder_inversion_torus_configs.py \
  scripts/experiments/pilot_decoder_inversion.py \
  scripts/eval/export_synthetic_trajectory_npz.py \
  scripts/train/train_synthetic_manifold.py \
  tests/test_synthetic_decoder_inversion_flow.py \
  tests/test_synthetic_trajectory_eval.py \
  tests/test_synthetic_trainer_resume.py
