#!/usr/bin/env bash
# Prepare the full locked graphs before the native proof resolves them offline.
set -euo pipefail
cd "$(dirname "$0")/../.."
metadata_dir="${CARGO_TARGET_DIR:-target}/merge-guard-preparation"
mkdir -p "$metadata_dir"
export CARGO_NET_RETRY=0
for entry in root:Cargo.toml tool:ops/merge_guard/Cargo.toml; do
  label=${entry%%:*}
  manifest=${entry#*:}
  if timeout --signal=KILL 120s cargo metadata --locked --manifest-path "$manifest" --format-version 1 \
    > "$metadata_dir/$label.json" 2> "$metadata_dir/$label.stderr"; then
    test -s "$metadata_dir/$label.json"
  else
    status=$?
    printf 'locked merge-guard preparation failed: manifest=%s status=%s; stderr (up to 64 KiB):\n' "$manifest" "$status" >&2
    head -c 65536 "$metadata_dir/$label.stderr" >&2
    exit "$status"
  fi
done
