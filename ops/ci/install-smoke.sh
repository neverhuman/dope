#!/usr/bin/env bash
# Keep the existing CI entry point for the Rust smoke tests.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec cargo test --manifest-path engine/Cargo.toml --locked -p dope-cli \
  --test readme_install_smoke "${1:-published_readme_quick_start}" -- --ignored --exact --nocapture
