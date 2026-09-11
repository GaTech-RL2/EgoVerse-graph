#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 /absolute/source/repo /absolute/runtime" >&2
  exit 2
fi

repo=$1
runtime=$2

for path in "$repo" "$runtime"; do
  if [[ "$path" != /* ]]; then
    echo "source repo and runtime must be absolute paths: $path" >&2
    exit 2
  fi
done

if [[ ! -f "$repo/pyproject.toml" || ! -f "$repo/uv.lock" ]]; then
  echo "source repo must contain pyproject.toml and uv.lock: $repo" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "runtime sync must run inside a scheduled allocation (set SLURM_JOB_ID)" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to install the frozen EgoVerse runtime" >&2
  exit 2
fi

mkdir -p "$(dirname "$runtime")"
UV_PROJECT_ENVIRONMENT="$runtime" uv sync --frozen --project "$repo"

python_bin="$runtime/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "frozen sync did not create an executable Python: $python_bin" >&2
  exit 1
fi

"$python_bin" - "$repo" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
import tomllib

project = tomllib.loads((Path(sys.argv[1]) / "pyproject.toml").read_text())
pins = [
    item.split("==", 1)[1]
    for item in project["project"]["dependencies"]
    if item.startswith("torchdiffeq==")
]
if len(pins) != 1:
    raise SystemExit("pyproject.toml must contain one exact torchdiffeq pin")
expected = pins[0]
actual = version("torchdiffeq")
if actual != expected:
    raise SystemExit(
        f"torchdiffeq runtime mismatch: expected {expected}, found {actual}"
    )
print(f"RUNTIME_READY torchdiffeq={actual}")
PY
