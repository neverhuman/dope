#!/usr/bin/env bash
# ARM qualification is native product compute, never publication rendering.
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
for tool in uname cargo rustc cc ar python3; do
  ci_require_command "$tool"
done
if [[ "$(uname -m)" != aarch64 ]]; then
  echo "ARM64 core gate requires a native aarch64 host" >&2
  exit 1
fi
uname -a
rustc -vV
cargo --version
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-2}"
# All native library tests include numerical goldens and historical decoders.
cargo test --locked --lib
# Real packet, CLI, target-last intake and source-free sampling regressions.
cargo test --locked --test codec_golden --test rust_cli \
  --test vnext_raw_schema --test vnext_raw_workflow --test vnext_artifact_only \
  --test vnext_bundle --test vnext_bundle_isolation --test vnext_geometry_binding
