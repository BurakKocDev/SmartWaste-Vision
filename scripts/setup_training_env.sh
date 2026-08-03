#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${SMARTWASTE_TRAIN_VENV:-${HOME}/.cache/smartwaste-vision/train-venv}"
python_bin="${venv_root}/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  python3 -m virtualenv "${venv_root}"
fi

"${python_bin}" -m pip install \
  --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.12.1+cu130' \
  'torchvision==0.27.1+cu130'

"${python_bin}" -m pip install \
  --no-cache-dir \
  --index-url https://pypi.org/simple \
  -r "${project_root}/requirements-training.in"

"${python_bin}" -m pip check
"${python_bin}" -m pip freeze --all | sort > "${project_root}/requirements-training.lock"

"${python_bin}" - <<'PY'
import json

import torch
import torchvision

if not torch.cuda.is_available():
  raise SystemExit("CUDA verification failed: torch.cuda.is_available() is false")

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        indent=2,
    )
)
PY
