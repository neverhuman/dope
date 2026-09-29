#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
ci_require_command python3
mkdir -p target
python3 - <<'PY'
import pathlib
import torch
assert torch.__version__ == '2.7.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available()
with open('target/torch-lib-path.txt', 'w') as output:
    output.write(str(pathlib.Path(torch.__file__).parent / 'lib'))
PY
export LIBTORCH_USE_PYTORCH=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$(cat target/torch-lib-path.txt):${LD_LIBRARY_PATH:-}"
cargo test --locked --features gpu-training --lib
if [[ ! -d "${SEALED_CORPUS_DIR:-}" || ! -f "${SEALED_ARTIFACT:-}" ]]; then
  echo "Sealed corpus directory and candidate artifact are required" >&2
  exit 1
fi
cargo run --locked --release --features gpu-training -- certify \
  --real-dir "$SEALED_CORPUS_DIR" \
  --kernel "$SEALED_ARTIFACT" \
  --out target/sealed-certification.json
ci_require_artifact target/sealed-certification.json
