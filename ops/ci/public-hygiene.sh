#!/usr/bin/env bash
# Pure-Rust public repository hygiene lane (root cleanliness, leak scan, map & link integrity).
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo fmt --manifest-path ops/merge_guard/Cargo.toml --all -- --check
cargo clippy --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin repo-hygiene -- -D warnings
cargo test --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin repo-hygiene
cargo run --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin repo-hygiene -- --root .
