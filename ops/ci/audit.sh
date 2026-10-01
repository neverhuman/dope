#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
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
    target/ci-python/bin/python -m pip install 'catboost==1.2.10'
    bash agent/check-benchmark-boundary.sh
    ;;
  verify)
    ci_require_artifact "$2"
    ci_require_artifact "$3"
    python3 - "$2" <<'PY'
import json
import sys
with open(sys.argv[1]) as source:
    report = json.load(source)
assert report['auditor_version'] == '1.7.1'
assert report['score'] >= 85
assert report['decision']['status'] == 'pass'
assert not report.get('caps_applied')
assert not any(item['severity'] in {'high', 'critical'} for item in report.get('findings', []))
PY
    ;;
  run)
    bash agent/check-python-v1-boundary.sh
    bash agent/check-external-boundary.sh
    bash agent/check-benchmark-boundary.sh
    bash agent/run-jankurai.sh
    ;;
  *) echo "Usage: $0 {prepare|verify JSON MD|run}" >&2; exit 2 ;;
esac
