#!/usr/bin/env bash
set -Eeuo pipefail
set +x
export DEBIAN_FRONTEND=noninteractive
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PIP_CONSTRAINT UV_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
apt-get update -qq
apt-get install -y --no-install-recommends git curl ca-certificates libgl1 libglib2.0-0 libegl1 libosmesa6 ffmpeg
curl -LsSf https://astral.sh/uv/0.10.4/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /workspace/libero/source
cat > /tmp/libero-git-askpass.sh <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
esac
ASKPASS
chmod 700 /tmp/libero-git-askpass.sh
export GIT_ASKPASS=/tmp/libero-git-askpass.sh GIT_TERMINAL_PROMPT=0
cd /workspace/libero/source
git init -q
git remote add origin https://github.com/GaTech-RL2/EgoVerse-graph.git
git fetch --depth 1 origin "$SOURCE_COMMIT"
git checkout --detach FETCH_HEAD
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
uv venv --python 3.11 emimic
source emimic/bin/activate
# Select only the OAT/simulator extras; unrelated PI/Yam constraints conflict.
uv pip install -e '.[oat,libero]'
# Upstream's wheel does not declare its simulator assets. Keep the pinned
# checkout available and install it editable so meshes and BDDL files exist.
export LIBERO_SOURCE_ROOT=/workspace/libero/LIBERO
git init -q "$LIBERO_SOURCE_ROOT"
git -C "$LIBERO_SOURCE_ROOT" remote add origin https://github.com/Chaoqi-LIU/LIBERO.git
git -C "$LIBERO_SOURCE_ROOT" fetch --depth 1 origin 6090ff21837566fed47b7c9061c9899f4749d36b
git -C "$LIBERO_SOURCE_ROOT" checkout --detach FETCH_HEAD
uv pip install --no-deps -e "$LIBERO_SOURCE_ROOT"
export LIBERO_CONFIG_PATH=/workspace/libero/config
if [[ "${RUN_KIND:-benchmark}" == replay ]]; then
    python -m egomimic.benchmarks.libero.replay --root /workspace/libero --suite "$SUITE" --run-id "$RUN_ID"
else
    python -m egomimic.benchmarks.libero.cluster --root /workspace/libero --suite "$SUITE" --mode "$RUN_MODE" --run-id "$RUN_ID" --epochs "$EPOCHS"
fi
