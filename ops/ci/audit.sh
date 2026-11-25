#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
verify_report() {
  ci_require_artifact "$1"
  ci_require_artifact "$2"
  cargo run --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked \
    --bin audit-proof -- verify --report "$1" --expected-head "$(git rev-parse HEAD)"
}
case "${1:-run}" in
  prepare)
    python3 -m venv --system-site-packages target/ci-python
    bash ops/ci/install-python-v1.sh target/ci-python/bin/python
    export PATH="$PWD/target/ci-python/bin:$PATH"
    bash agent/check-python-v1-boundary.sh
    bash agent/check-external-boundary.sh
    target/ci-python/bin/python - <<'PY'
import json
assert json.load(open("research/benchmark/sdv-runtime.lock.json"))["catboost"] == "1.2.10"
PY
    ci_require_command uv
    uv pip install --python target/ci-python/bin/python \
      --constraint validation/requirements-v1.txt 'catboost==1.2.10' 'jsonschema==4.26.0' \
      --requirement research/benchmark/train-export-requirements.txt
    bash agent/check-benchmark-boundary.sh
    ;;
  verify)
    verify_report "$2" "$3"
    ;;
  run)
    bash agent/check-python-v1-boundary.sh
    bash agent/check-external-boundary.sh
    bash agent/check-benchmark-boundary.sh
    bash agent/run-jankurai.sh
    verify_report target/jankurai/repo-score.json target/jankurai/repo-score.md
    ;;
  *) echo "Usage: $0 {prepare|verify JSON MD|run}" >&2; exit 2 ;;
esac
