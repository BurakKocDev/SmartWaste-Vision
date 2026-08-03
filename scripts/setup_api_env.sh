#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="${SMARTWASTE_API_VENV:-${HOME}/.cache/smartwaste-vision/api-venv}"
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
  -r "${project_root}/requirements-api-dev.in"

"${python_bin}" -m pip check
"${python_bin}" -m pip freeze --all | sort > "${project_root}/requirements-api.lock"
