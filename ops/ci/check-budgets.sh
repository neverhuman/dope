#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec cargo run --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked \
  --bin ci-shard -- budgets "$PWD"
