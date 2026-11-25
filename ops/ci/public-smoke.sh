#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m pip install --constraint validation/requirements-v1.txt numpy scipy pandas 'matplotlib==3.10.8' 'jsonschema==4.26.0'

# 1. Pure-Rust v3 DOPE engine formatting, lints, unit/CLI tests, and real-dataset quick start.
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo clippy --manifest-path engine/Cargo.toml --locked -p dope-core -p dope-cli --all-targets -- -D warnings
cargo test --manifest-path engine/Cargo.toml --locked -p dope-core -p dope-cli
cargo build --manifest-path engine/Cargo.toml --locked --release -p dope-cli --bin dope
mkdir -p target/public-smoke
bash examples/v3-quickstart/run.sh --binary engine/target/release/dope --out target/public-smoke/v3-airfoil --auditors linear,gbdt

# 2. Historical dope-kernel compatibility smoke and golden parity.
cargo build --locked --release -p dope-kernel --bin dope
target/release/dope compile --dataset-dir examples/readme-regression --task regression --out target/public-smoke/toy.dpk
target/release/dope inspect --kernel target/public-smoke/toy.dpk
target/release/dope sample --kernel target/public-smoke/toy.dpk --rows 200 --out target/public-smoke/synth.csv
cargo build --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin audit-proof
target/debug/audit-proof focus metrics::golden

# 3. Public claims firewall and rendered surfaces.
python3 -m unittest discover -s tools/claims -p 'test_*.py'
python3 tools/claims/check.py
python3 tools/claims/render.py --check
