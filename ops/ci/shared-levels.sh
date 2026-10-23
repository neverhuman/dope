#!/usr/bin/env bash
# Shared local/CI conformance entry; finite jobs run only on the admitted disposable Linux class.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$HOME/.cargo/bin:$PATH"
case "${1:-all}" in
  contracts)
    python3 -B -m unittest research.levels.tests.test_contracts research.levels.tests.test_operations -v
    ;;
  prepare-torch)
    python3 -m venv target/shared-python
    target/shared-python/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0
    target/shared-python/bin/pip install --requirement research/levels/requirements.txt
    ;;
  torch)
    case "${2:?test shard required}" in test_models|test_training|test_comparison) ;; *) exit 2;; esac
    "${SHARED_PYTHON:-target/shared-python/bin/python}" -B -m unittest "research.levels.tests.$2" -v
    ;;
  prepare-native)
    if ! command -v rustup >/dev/null 2>&1; then
      case "$(uname -m)" in
        x86_64) target_triple=x86_64-unknown-linux-gnu;;
        aarch64) target_triple=aarch64-unknown-linux-gnu;;
        *) echo 'Unsupported native CI architecture' >&2;exit 2;;
      esac
      mkdir -p target/shared-tools
      installer_url="https://static.rust-lang.org/rustup/dist/$target_triple/rustup-init"
      curl --fail --location --retry 3 "$installer_url" --output target/shared-tools/rustup-init
      curl --fail --location --retry 3 "$installer_url.sha256" --output target/shared-tools/rustup-init.sha256
      installer_sha=$(cut -d ' ' -f1 target/shared-tools/rustup-init.sha256)
      printf '%s  %s\n' "$installer_sha" target/shared-tools/rustup-init | sha256sum --check -
      chmod u+x target/shared-tools/rustup-init
      target/shared-tools/rustup-init -y --no-modify-path --default-toolchain none
    fi
    rustup toolchain install 1.88.0 --profile minimal --component rustfmt --component clippy
    ;;
  native)
    cargo +1.88.0 fmt --manifest-path engine/Cargo.toml --all -- --check
    cargo +1.88.0 clippy --manifest-path engine/Cargo.toml --locked -p dope-core -p dope-cli --all-targets -j2 -- -D warnings
    cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-core --test learned_parity -j2
    cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-cli --test learned_cli -j2
    cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-probes --test private_errors -j2
    cargo +1.88.0 test --manifest-path engine/Cargo.toml --locked -p dope-core --test seeded_auditors -j2
    ;;
  all)
    bash ops/ci/shared-levels.sh contracts
    bash ops/ci/shared-levels.sh torch test_models
    bash ops/ci/shared-levels.sh torch test_training
    bash ops/ci/shared-levels.sh torch test_comparison
    bash ops/ci/shared-levels.sh native
    ;;
  *) echo 'Usage: shared-levels.sh contracts|prepare-torch|torch SHARD|prepare-native|native|all' >&2;exit 2;;
esac
