#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"
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
cd "$REPO_ROOT"
python3 research/benchmark/verify_paper_numbers.py
python3 docs/whitepaper/scripts/compute_panel.py
git diff --exit-code -- docs/whitepaper/generated/fit-trace.json docs/whitepaper/generated/numbers.tex docs/whitepaper/generated/density-table.tex docs/whitepaper/generated/neural-table.tex docs/whitepaper/generated/forest-table.tex docs/whitepaper/generated/byte-table.tex docs/whitepaper/generated/threshold-table.tex docs/whitepaper/generated/beyond-fit.tex docs/whitepaper/generated/provenance-table.tex
python3 -B -m unittest research.benchmark.tests.test_verify_paper_numbers research.benchmark.tests.test_paper_log
bash docs/whitepaper/scripts/build_pdf.sh
python3 docs/whitepaper/scripts/check_paper.py
