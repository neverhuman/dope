#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
python_bin="${1:-python3}"
"$python_bin" -m pip install -r validation/requirements-v1.txt
if ! "$python_bin" - <<'PY'
import torch
assert torch.__version__.startswith("2.7.0")
PY
then
  "$python_bin" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.0+cpu'
fi
