#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
python_bin="${1:-python3}"
install_packages() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python_bin" "$@"
  else
    "$python_bin" -m pip install "$@"
  fi
}
install_packages -r validation/requirements-v1.txt
if ! "$python_bin" - <<'PY'
import torch
assert torch.__version__.startswith("2.7.0")
PY
then
  install_packages --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.0+cpu'
fi
