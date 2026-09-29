#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  cpu|security|audit|gpu)
    bash "ops/ci/$1.sh"
    ;;
  *) echo "Usage: $0 {cpu|security|audit|gpu}" >&2; exit 2 ;;
esac
