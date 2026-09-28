#!/usr/bin/env bash
# Runs only in the allocated single-GPU OSMO task.
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
unset PIP_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libegl1 libglfw3 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.10.4/install.sh -o /tmp/install-uv.sh
bash /tmp/install-uv.sh
export PATH="/root/.local/bin:$PATH" UV_LINK_MODE=copy
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
export UV_PROJECT_ENVIRONMENT=/tmp/egoverse/emimic
uv sync --locked --python 3.11
source emimic/bin/activate
mkdir -p "$GATE_OUTPUT"
git rev-parse HEAD > "$GATE_OUTPUT/source-commit.txt"
cp uv.lock "$GATE_OUTPUT/learner.lock"
cp scripts/astra_push/simulator.lock "$GATE_OUTPUT/simulator.lock"
uv pip list --format json > "$GATE_OUTPUT/learner-packages.json"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$GATE_OUTPUT/gpu.csv"
upload() {
  source /tmp/egoverse/emimic/bin/activate
  python -m scripts.integration.upload_artifacts --root "$GATE_OUTPUT" --prefix "$ARTIFACT_PREFIX"
}
trap upload EXIT
status=0
if [[ "${RUN_PROVIDER_PROBE:-1}" == "1" ]]; then
  python -m egomimic.experiments.astra_push.provider --output "$GATE_OUTPUT/provider" 2>&1 | tee "$GATE_OUTPUT/provider.log" || status=1
else
  python - <<'PY'
import os
from pathlib import Path
from egomimic.experiments.astra_push.artifacts import publish_json
publish_json(Path(os.environ['GATE_OUTPUT'])/'provider-skipped.json', {'reason':'confirmed_gateway_budget_exhaustion', 'provider_gate_passed':False, 'generation_calls':0})
PY
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=4
# The synthetic update is already measured in v1. This revision verifies real
# three-stage teacher data after constructing the separate simulator below.
git init -q /tmp/libero
git -C /tmp/libero remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git
git -C /tmp/libero fetch --quiet --depth=1 origin f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
git -C /tmp/libero checkout --quiet --detach FETCH_HEAD
test "$(git -C /tmp/libero rev-parse HEAD)" = f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
uv venv /tmp/simulator-env --python 3.11
source /tmp/simulator-env/bin/activate
uv pip install --require-hashes -r scripts/astra_push/simulator.lock
uv pip install --no-deps -e /tmp/libero
export PYTHONPATH="/tmp/libero:/tmp/egoverse"
git -C /tmp/libero rev-parse HEAD > "$GATE_OUTPUT/libero-source-commit.txt"
uv pip list --format json > "$GATE_OUTPUT/simulator-packages.json"
export LIBERO_CONFIG_PATH=/tmp/astrapush-libero-config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python - <<'PY'
import os
from pathlib import Path
import yaml
root=Path('/tmp/libero/libero/libero')
cfg=Path(os.environ['LIBERO_CONFIG_PATH'])
cfg.mkdir(exist_ok=False)
paths={'benchmark_root':str(root),'bddl_files':str(root/'bddl_files'),'init_states':str(root/'init_files'),'datasets':str(root.parent/'datasets'),'assets':str(root/'assets')}
(cfg/'config.yaml').write_text(yaml.safe_dump(paths))
PY
for stage in S1 S2 S3; do
  python -m egomimic.experiments.astra_push.render_probe --stage "$stage" --output "$GATE_OUTPUT/render-$stage" 2>&1 | tee "$GATE_OUTPUT/render-$stage.log" || status=1
  python -m egomimic.experiments.astra_push.teacher_probe --stage "$stage" --output "$GATE_OUTPUT/teacher-$stage" 2>&1 | tee "$GATE_OUTPUT/teacher-$stage.log" || status=1
done
python -m egomimic.experiments.astra_push.calibration --output "$GATE_OUTPUT/calibration" 2>&1 | tee "$GATE_OUTPUT/calibration.log" || status=1
source /tmp/egoverse/emimic/bin/activate
python -m egomimic.experiments.astra_push.init_audit --output "$GATE_OUTPUT/real-learner-audit" --real-episodes "$GATE_OUTPUT/teacher-S1/receipt.json" "$GATE_OUTPUT/teacher-S2/receipt.json" "$GATE_OUTPUT/teacher-S3/receipt.json" 2>&1 | tee "$GATE_OUTPUT/real-learner-audit.log" || status=1
exit "$status"
