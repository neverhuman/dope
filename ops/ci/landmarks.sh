#!/usr/bin/env bash
# Same native landmark/release gate locally and on hosted CI. No sealed data.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo clippy --manifest-path engine/Cargo.toml --all-targets --locked -- -D warnings
cargo test --manifest-path engine/Cargo.toml --locked --test landmarks --test landmark_cli
# Development-probe receipts: NumPy-exact split and replay of the numbers the plans cite.
cargo test --manifest-path engine/Cargo.toml --locked -p dope-probes
