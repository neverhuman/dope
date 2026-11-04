#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  cpu)
    cargo build --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin native-process -j1
    export DOPE_NATIVE_PROCESS="$PWD/target/debug/native-process"
    bash ops/ci/cpu.sh
    ;;
  security|audit|gpu)
    bash "ops/ci/$1.sh"
    ;;
  *) echo "Usage: $0 {cpu|security|audit|gpu}" >&2; exit 2 ;;
esac
