#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash web_demo/setup_env.sh
#   bash web_demo/setup_env.sh <env_name> <python_version>
#
# Example:
#   bash web_demo/setup_env.sh robot_vision_demo 3.10

ENV_NAME="${1:-robot_vision_demo}"
PY_VER="${2:-3.10}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REQ_FILE="${PROJECT_ROOT}/web_demo/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found."
  echo "Please initialize conda first (e.g., source ~/miniconda3/etc/profile.d/conda.sh)."
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] requirements.txt not found: ${REQ_FILE}"
  exit 1
fi

echo "[1/5] Creating conda env: ${ENV_NAME} (python=${PY_VER})"
conda create -y -n "${ENV_NAME}" "python=${PY_VER}"

echo "[2/5] Activating env"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[3/5] Installing compatible pip/setuptools/wheel"
# SC-Depth's pytorch-lightning(1.7.x) metadata is rejected by pip>=24.1.
python -m pip install --upgrade "pip<24.1" "setuptools<81" wheel

echo "[4/5] Installing web demo dependencies"
python -m pip install -r "${REQ_FILE}"

echo "[5/5] Installing SC-Depth compatibility deps"
python -m pip install \
  "pytorch-lightning==1.7.3" \
  "torchmetrics==0.9.3" \
  "kornia>=0.7" \
  tqdm imageio path scipy

echo "[Done] Environment ready"
echo
echo "Run demo with:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${PROJECT_ROOT}"
echo "  uvicorn web_demo.main:app --host 0.0.0.0 --port 8000 --reload"
