#!/usr/bin/env bash
# Bounded CPU engineering conformance. No training or scientific campaign.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --manifest-path deploy/Cargo.toml -- --check
cargo clippy --manifest-path deploy/Cargo.toml --locked --all-targets -- -D warnings
cargo test --manifest-path deploy/Cargo.toml --locked
mkdir -p target/deployment-evidence
cargo tree --manifest-path deploy/Cargo.toml --locked -e normal --prefix none \
  > target/deployment-evidence/normal-dependencies.txt
if rg -q '^(tch|torch-sys|pyo3|cpython) v' target/deployment-evidence/normal-dependencies.txt; then
  echo 'deployment graph contains a forbidden training/interpreter dependency' >&2
  exit 1
fi
echo 'Rust deployment CPU conformance passed; CUDA and scientific qualification remain unmeasured'
