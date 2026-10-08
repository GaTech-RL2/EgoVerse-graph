#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=42
export SDL_VIDEODRIVER=dummy MUJOCO_GL=egl
unset PIP_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 libegl1 libosmesa6 ffmpeg
curl --retry 4 --retry-all-errors -LsSf https://astral.sh/uv/0.10.4/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/cotrain/source /workspace/cotrain/evidence
cat > /tmp/planar-git-askpass.sh <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
esac
ASKPASS
chmod 700 /tmp/planar-git-askpass.sh
export GIT_ASKPASS=/tmp/planar-git-askpass.sh GIT_TERMINAL_PROMPT=0
cd /workspace/cotrain/source
git init -q
git remote add origin https://github.com/GaTech-RL2/EgoVerse-graph.git
git fetch --depth 1 origin "$SOURCE_COMMIT"
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
export UV_PROJECT_ENVIRONMENT=/workspace/cotrain/emimic
uv sync --frozen --python 3.11 --project /workspace/cotrain/source
source /workspace/cotrain/emimic/bin/activate
export PYTHONPATH=/workspace/cotrain/source
ulimit -n 8192
if [[ "$RUN_KIND" == train ]]; then
    python scripts/benchmarks/train_planar_streams.py --arm "$ARM" --run-id "$RUN_ID" --output "$WORKFLOW_OUTPUT"
elif [[ "$RUN_KIND" == evaluation ]]; then
    # Retain the exact dependencies of the previous frozen simulator benchmark.
    uv pip install --python "$VIRTUAL_ENV/bin/python" numpy==2.4.4 gymnasium==1.1.1 pygame==2.6.1 pymunk==7.1.0 shapely==2.1.1 opencv-python==4.13.0.92
    python scripts/benchmarks/prepare_planar_evaluation.py --training /tmp/training-complete.json --domain "$EVAL_DOMAIN" --shard "$EVAL_SHARD"
    python scripts/benchmarks/evaluate_planar_streams.py
else
    printf '%s\n' 'Unknown run kind' >&2
    exit 2
fi
