#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ops/ci/lib.sh
for tool in cargo rustup python3 uv just gitleaks actionlint syft; do
  ci_require_command "$tool"
done
[[ "$(rustc --version)" == "rustc $RUST_VERSION"* ]]
[[ "$(python3 --version)" == "Python $PYTHON_VERSION"* ]]
[[ "$(uv --version)" == "uv $UV_VERSION"* ]]
echo "Local CI tools and language pins are available"
