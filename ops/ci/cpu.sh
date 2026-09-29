#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
ci_require_command cargo
ci_require_command python3
bash ops/ci/check-budgets.sh
bash ops/ci/check-contracts.sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
python3 -m venv --system-site-packages target/ci-python
bash ops/ci/install-python-v1.sh target/ci-python/bin/python
PYTHONPATH=src target/ci-python/bin/python -m pytest -q tests
ci_require_command uv
cd validation/external
UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --locked --extra xgboost --extra lightgbm
UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --offline --locked --extra xgboost --extra lightgbm
PYTHONPATH=. ../../target/external-venv/bin/python -m unittest discover -s tests -v
