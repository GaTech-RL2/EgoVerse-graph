#!/bin/bash
# One-time Lambda setup for the towels394 campaign: import the pytorch image to a squashfs (so packed steps do
# not each pull 20 GB), install uv, and build the repo venv from uv.lock inside that image.
#SBATCH --job-name=setup_runtime
#SBATCH --account=loaner
#SBATCH --partition=loaner
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=224
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/workspace/users/%u/logs/%x-%j.out
set -euo pipefail
SRC=/workspace/users/$USER/EgoVerse-graph
IMAGE=/workspace/users/$USER/images/nvidia-pytorch-25.06-py3.sqsh
mkdir -p "$(dirname "$IMAGE")"
if [ ! -s "$IMAGE" ]; then
  srun --ntasks=1 enroot import -o "$IMAGE.partial" 'docker://nvcr.io#nvidia/pytorch:25.06-py3'
  mv "$IMAGE.partial" "$IMAGE"
fi
ls -la "$IMAGE"
srun --ntasks=1 --container-image="$IMAGE" --container-mounts=/workspace:/workspace:rw --container-workdir="$SRC" \
  bash -c 'set -euo pipefail
    export HOME=/workspace/users/$USER UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/workspace/users/$USER/.local/share/uv/python
    if ! command -v ~/.local/bin/uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh; fi
    ~/.local/bin/uv --version
    ~/.local/bin/uv sync --frozen
    .venv/bin/python -c "import torch, torchdiffeq, lightning, hydra, zarr; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.device_count(), \"torchdiffeq\", torchdiffeq.__version__)"'
