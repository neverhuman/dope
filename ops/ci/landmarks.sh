#!/usr/bin/env bash
# Same native landmark/release gate locally and on hosted CI. No sealed data.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo clippy -j2 --manifest-path engine/Cargo.toml --all-targets --locked -- -D warnings
cargo test -j2 --manifest-path engine/Cargo.toml --locked --test landmarks --test landmark_cli
# Development-probe receipts: NumPy-exact split and replay of the numbers the plans cite.
cargo test -j2 --manifest-path engine/Cargo.toml --locked -p dope-probes
