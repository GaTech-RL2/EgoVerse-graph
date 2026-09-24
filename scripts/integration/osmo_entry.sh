#!/usr/bin/env bash
# Runs only inside an allocated OSMO task. No dataset or existing job is modified.
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
unset PIP_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.10.4/install.sh -o /tmp/install-uv.sh
bash /tmp/install-uv.sh
export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/tmp/egoverse/emimic
export GIT_ASKPASS=/tmp/git-askpass.sh GIT_TERMINAL_PROMPT=0
cat > "$GIT_ASKPASS" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
esac
ASKPASS
chmod 700 "$GIT_ASKPASS"
git init -q /tmp/egoverse
cd /tmp/egoverse
git remote add origin https://github.com/GaTech-RL2/EgoVerse-graph.git
git fetch --quiet --depth=1 origin "$SOURCE_COMMIT"
git checkout --quiet --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"

if [[ "$GATE_SUITE" == stage ]]; then
  uv venv emimic --python 3.11
  source emimic/bin/activate
  uv pip install boto3==1.43.6
  python -m scripts.integration.prepare_inputs \
    --data-manifest docs/integration/evidence/real-data-inputs.json \
    --weight-manifest docs/integration/evidence/weight-inputs.json \
    --tokenizer-manifest docs/integration/evidence/pi-tokenizer-inputs.json \
    --tokenizer-root /tmp/input-tokenizer --output "$GATE_OUTPUT"
else
  uv sync --locked --python 3.11 --extra pi05 --extra alignment --extra diagnostics
  source emimic/bin/activate
  mkdir -p "$GATE_OUTPUT"
  uv pip list --format json > "$GATE_OUTPUT/packages.json"
  if [[ "$GATE_SUITE" == pi ]]; then
    python scripts/install_pi05_source.py
    cp emimic/pi05-source-receipt.json "$GATE_OUTPUT/pi05-source-receipt.json"
  fi
  export SMOKE_INPUT_ROOT="$GATE_INPUT" TORCH_HOME="$GATE_INPUT/torch"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
  export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
  suite_status=0
  python -m scripts.integration.run_suite \
    --matrix scripts/integration/matrix.yaml --suite "$GATE_SUITE" \
    --inputs "$GATE_INPUT" --output "$GATE_OUTPUT" || suite_status=$?
  python -m scripts.integration.upload_artifacts \
    --root "$GATE_OUTPUT" --prefix "$ARTIFACT_PREFIX"
  exit "$suite_status"
fi
