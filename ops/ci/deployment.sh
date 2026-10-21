#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
ensure_auditor() {
  if command -v cargo-audit >/dev/null 2>&1 && [[ "$(cargo-audit --version)" == "cargo-audit $CARGO_AUDIT_VERSION" ]]; then
    return
  fi
  local tools="$PWD/target/deployment-tools"
  local archive="$tools/cargo-audit-${CARGO_AUDIT_VERSION}.tgz"
  mkdir -p "$tools"
  if [[ ! -e "$archive" ]]; then
    curl --fail --location --silent --show-error --retry 3 \
      "https://github.com/rustsec/rustsec/releases/download/cargo-audit/v${CARGO_AUDIT_VERSION}/cargo-audit-x86_64-unknown-linux-gnu-v${CARGO_AUDIT_VERSION}.tgz" \
      --output "$archive"
  fi
  echo "1890badd5f15831a9af4b074399fcd21e6f7c0fe42c84e9254cdffc9f813765c  $archive" | sha256sum --check --status
  if [[ ! -x "$tools/cargo-audit" ]]; then
    tar -xzf "$archive" -C "$tools" --strip-components=1 \
      "cargo-audit-x86_64-unknown-linux-gnu-v${CARGO_AUDIT_VERSION}/cargo-audit"
  fi
  export PATH="$tools:$PATH"
}
case "${1:-check}" in
  provision)
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y --no-install-recommends rustup build-essential pkg-config ripgrep
    rustup toolchain install 1.88.0 --profile minimal --component rustfmt --component clippy
    ensure_auditor
    ;;
  check) bash deploy/check.sh ;;
  audit)
    ensure_auditor
    test "$(cargo-audit --version)" = "cargo-audit $CARGO_AUDIT_VERSION"
    mkdir -p target/deployment-evidence
    if ! cargo audit --file deploy/Cargo.lock --json > target/deployment-evidence/cargo-audit.json; then
      echo "deployment dependency advisory scan failed; see target/deployment-evidence/cargo-audit.json" >&2
      exit 1
    fi
    test -s target/deployment-evidence/cargo-audit.json
    ;;
  *) echo 'unsupported deployment check mode' >&2; exit 2 ;;
esac
