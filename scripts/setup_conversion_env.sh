#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${SMARTWASTE_VENV:-${HOME}/.cache/smartwaste-vision/venv}"
python_bin="${venv_root}/bin/python"
pip_cache="${project_root}/.cache/pip"

if [[ ! -x "${python_bin}" ]]; then
  python3 -m virtualenv "${venv_root}"
fi

mkdir -p "${pip_cache}"

"${python_bin}" -m pip install \
  --cache-dir "${pip_cache}" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.12.1+cpu' \
  'torchvision==0.27.1+cpu'

"${python_bin}" -m pip install \
  --cache-dir "${pip_cache}" \
  --index-url https://pypi.org/simple \
  -r "${project_root}/requirements-conversion.in"

"${python_bin}" -m pip check
"${python_bin}" -m pip freeze --all | sort > "${project_root}/requirements-conversion.lock"

"${python_bin}" - <<'PY'
import json
import platform

import litert_torch
import onnx
import onnxruntime
import torch
import torchvision

print(
    json.dumps(
        {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "litert_torch": getattr(litert_torch, "__version__", "unknown"),
        },
        indent=2,
    )
)
PY
