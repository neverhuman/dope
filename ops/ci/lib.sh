#!/usr/bin/env bash
set -euo pipefail

RUST_VERSION=1.88.0
PYTHON_VERSION=3.12
UV_VERSION=0.11.28
JANKURAI_VERSION=1.7.1
CARGO_AUDIT_VERSION=0.22.1
ACTIONLINT_VERSION=1.7.8
GITLEAKS_VERSION=8.21.2

export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_DEV_DEBUG="${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}"
mkdir -p "$PWD/target/tmp"
export TMPDIR="$PWD/target/tmp"

ci_require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required CI tool is unavailable: $1" >&2
    exit 1
  fi
}

ci_require_artifact() {
  if [[ ! -s "$1" ]]; then
    echo "Required CI artifact is absent or empty: $1" >&2
    exit 1
  fi
}
