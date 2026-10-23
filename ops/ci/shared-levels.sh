#!/usr/bin/env bash
# Shared local/CI conformance entry; finite jobs run only on the admitted disposable Linux class.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-all}" in
  contracts)
    python3 -B -m unittest research.levels.tests.test_contracts research.levels.tests.test_operations -v
    ;;
  prepare-torch)
    python3 -m venv target/shared-python
    target/shared-python/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0
    target/shared-python/bin/pip install numpy==1.26.4 safetensors==0.7.0
    ;;
  torch)
    case "${2:?test shard required}" in test_models|test_training) ;; *) exit 2;; esac
    "${SHARED_PYTHON:-target/shared-python/bin/python}" -B -m unittest "research.levels.tests.$2" -v
    ;;
  prepare-native)
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rustup
    rustup toolchain install 1.88.0 --profile minimal --component rustfmt --component clippy
    ;;
  native)
    cargo +1.88.0 fmt --manifest-path engine/Cargo.toml --all -- --check
    cargo +1.88.0 clippy --manifest-path engine/Cargo.toml --locked -p dope-core -p dope-cli --all-targets -j2 -- -D warnings
    cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-core --test learned_parity -j2
    cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-cli --test learned_cli -j2
    ;;
  all)
    bash ops/ci/shared-levels.sh contracts
    bash ops/ci/shared-levels.sh torch test_models
    bash ops/ci/shared-levels.sh torch test_training
    bash ops/ci/shared-levels.sh native
    ;;
  *) echo 'Usage: shared-levels.sh contracts|prepare-torch|torch SHARD|prepare-native|native|all' >&2;exit 2;;
esac
