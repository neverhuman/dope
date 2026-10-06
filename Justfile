set shell := ["bash", "-euo", "pipefail", "-c"]

export CARGO_INCREMENTAL := "0"
export CARGO_PROFILE_DEV_DEBUG := "line-tables-only"

setup:
    cargo fetch --locked
    mkdir -p target/tmp
    export TMPDIR="$PWD/target/tmp"; python3 -m venv --system-site-packages target/venv
    bash ops/ci/install-python-v1.sh target/venv/bin/python
    TMPDIR="$PWD/target/tmp" target/venv/bin/python -m pip install --no-deps -e .
    mkdir -p target/python-metadata
    if test -d src/dope_kernel.egg-info; then mv src/dope_kernel.egg-info "target/python-metadata/dope_kernel.$(date +%s%N).egg-info"; fi

fast:
    cargo fmt --all -- --check
    cargo check -p dope-kernel --locked
    cargo test --locked --lib
    PYTHONPATH=src python3 -m pytest -q tests

focus filter:
    cargo check -p dope-kernel --locked
    cargo test -p dope-kernel --locked --lib {{filter}}

fast-audit:
    bash agent/run-jankurai-fast.sh

check: fast
    bash ops/ci/check-budgets.sh
    bash ops/ci/check-contracts.sh
    cargo clippy --all-targets --locked -- -D warnings
    cargo test --locked --test rust_cli
    git diff --check

security:
    bash ops/ci/security.sh

contract-drift:
    bash ops/ci/check-contracts.sh

score:
    bash ops/ci/audit.sh run

paper:
    python3 research/benchmark/verify_paper_numbers.py --write-evidence
    python3 docs/whitepaper/scripts/local_receipts.py
    python3 docs/whitepaper/scripts/compute_panel.py
    python3 research/benchmark/verify_paper_numbers.py
    python3 docs/whitepaper/scripts/render_figures.py
    bash docs/whitepaper/scripts/build_pdf.sh
    python3 docs/whitepaper/scripts/check_paper.py

paper-check:
    python3 research/benchmark/verify_paper_numbers.py
    python3 docs/whitepaper/scripts/compute_panel.py
    git diff --exit-code -- docs/whitepaper/generated/fit-trace.json docs/whitepaper/generated/numbers.tex docs/whitepaper/generated/density-table.tex docs/whitepaper/generated/neural-table.tex docs/whitepaper/generated/forest-table.tex docs/whitepaper/generated/byte-table.tex docs/whitepaper/generated/threshold-table.tex docs/whitepaper/generated/beyond-fit.tex docs/whitepaper/generated/provenance-table.tex
    bash docs/whitepaper/scripts/build_pdf.sh
    python3 docs/whitepaper/scripts/check_paper.py
