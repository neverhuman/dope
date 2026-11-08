#!/usr/bin/env bash
set -euo pipefail

ci_shard_init() {
  local shard="$1"
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  source ops/ci/lib.sh
  CI_SHARD="$shard"
  CI_LOG="$PWD/target/ci-shards/$shard.log"
  CI_TMP="$PWD/target/ci-shards/$shard-command.log"
  mkdir -p "$(dirname "$CI_LOG")"
  : > "$CI_LOG"
  trap ci_shard_finish EXIT
}

ci_run() {
  local id="$1" kind="$2" status=0
  shift 2
  printf '@@CI-BEGIN %s %s@@\n' "$id" "$kind" >> "$CI_LOG"
  if "$@" > "$CI_TMP" 2>&1; then status=0; else status=$?; fi
  cat "$CI_TMP"
  cat "$CI_TMP" >> "$CI_LOG"
  printf '@@CI-END %s %s@@\n' "$id" "$status" >> "$CI_LOG"
  return "$status"
}

ci_frozen_check() {
  ci_run frozen-evidence check cargo run --manifest-path ops/merge_guard/Cargo.toml \
    --target-dir target --locked --bin check-frozen-evidence -- --root "$PWD"
}

ci_prepare_receipts() {
  ci_run ci-shard-build check cargo build --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin ci-shard
  ci_run ci-shard-coverage check target/debug/ci-shard coverage ops/merge_guard/fixtures/ci-sharding/command-map.json \
    ops/merge_guard/fixtures/ci-sharding/source ops/ci .github/workflows/ci.yml
}

ci_shard_finish() {
  local status=$?
  trap - EXIT
  if [[ -n "${GITHUB_RUN_ID:-}" && -n "${GITHUB_RUN_ATTEMPT:-}" && -n "${GITHUB_SHA:-}" ]]; then
    local receipt_status=success
    [[ "$status" == 0 ]] || receipt_status=failed
    target/debug/ci-shard emit "$CI_SHARD" "$GITHUB_SHA" "$GITHUB_RUN_ID" \
      "$GITHUB_RUN_ATTEMPT" "$receipt_status" "$CI_LOG" "target/ci-shards/$CI_SHARD.json" || status=1
  fi
  exit "$status"
}
