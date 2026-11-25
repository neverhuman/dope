#!/usr/bin/env bash
# Keep the existing CI entry point for the Rust smoke tests.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo test --manifest-path engine/Cargo.toml --target-dir target --locked -p dope-cli \
  --test readme_install_smoke paper_examples_match_smoke_commands -- --exact
exec cargo test --manifest-path engine/Cargo.toml --target-dir target --locked -p dope-cli \
  --test readme_install_smoke "${1:-published_readme_quick_start}" -- --ignored --exact --nocapture
