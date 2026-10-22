#!/usr/bin/env bash
# The DOPE engine (engine/): a standalone pure-Rust workspace with its own lockfile.
# Formatting, lints as errors, and the unit, end-to-end and CLI tests (CPU backend).
set -euo pipefail
cd "$(dirname "$0")/../.."
manifest=engine/Cargo.toml
cargo fmt --manifest-path "$manifest" --all -- --check
cargo clippy --manifest-path "$manifest" --all-targets --locked -- -D warnings
cargo test --manifest-path "$manifest" --locked
python3 -B -m unittest discover -s bench/real-world -p 'test_*.py' -v
python3 -B bench/real-world/elo.py --check
python3 -B bench/real-world/elo.py --methods dope,rows,kernel,copula,arf --out bench/real-world/results/elo-cpu --check
