#!/usr/bin/env bash
# The DOPE engine (engine/): a standalone pure-Rust workspace with its own lockfile.
# Formatting, lints as errors, and the unit, end-to-end and CLI tests (CPU backend).
set -euo pipefail
cd "$(dirname "$0")/../.."
manifest=engine/Cargo.toml
cargo fmt --manifest-path "$manifest" --all -- --check
cargo clippy --manifest-path "$manifest" --all-targets --locked -- -D warnings
cargo test --manifest-path "$manifest" --locked
