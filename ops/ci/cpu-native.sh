#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/shard-runtime.sh
ci_shard_init cpu-native
for tool in cargo python3 rg; do ci_require_command "$tool"; done
ci_frozen_check
ci_prepare_receipts
# ci-command:cpu-native:landmark-cache cc478598c8d2250be85b236103a38617cfde15303db5a26c37807c3fbd4e7b59
ci_run landmark-cache unittest python3 -B -m unittest discover -s research/landmark_cache/tests -v
# ci-command:cpu-native:legal-inputs cd7f2fcc9e6a809d19f0394a2559c48aa176d84e7456b37bfdb9f490b92d83f9
ci_run legal-inputs check sha256sum --check --status scripts/native-legal.sha256
# ci-command:cpu-native:budgets 45d2b37503b2c3dc39a9b5213ab2b3e606d63a8eedb6078979322a8661d336c0
ci_run budgets check bash ops/ci/check-budgets.sh
# ci-command:cpu-native:contracts b1196c8511d80538050f5bac0d0c761122661840ec9f606bb40d20bdc9fc5471
ci_run contracts check bash ops/ci/check-contracts.sh
# ci-command:cpu-native:root-format 3d8b66678a0ac20b1fbff53bd1120edcb7a8640941b5586535bdd31630a4afcc
ci_run root-format check cargo fmt --all -- --check
# ci-command:cpu-native:root-clippy 1ab3c6176e008c738c825108f176299ac75100aa3c3ecb4d8a7cf1f950dc5a4d
ci_run root-clippy check cargo clippy --all-targets --locked -- -D warnings
# ci-command:cpu-native:merge-guard-prepare d1b48262b2aa3823b8aabc359f0f77991aa046b67ce803e6ea73829a6fdaafac
ci_run merge-guard-prepare check bash ops/ci/prepare-merge-guard.sh
# ci-command:cpu-native:root-cargo-test 39431e057e668c629ea7e1fa993b2ec4bdaddce7a1b02303bc22ba37d327a248
ci_run root-cargo-test cargo-test cargo test --locked
# ci-command:cpu-native:engine-test 937093a35003f117f371bcfbb25e5e63ccff503d147db693d0fb82fa599034d1
ci_run engine-test cargo-test bash ops/ci/engine.sh
# ci-command:cpu-native:native-notices ee2edbf0dc94c17d853860c4eba5f09e629d92208011ba243ceb2d0d70a1b066
ci_run native-notices check cargo run --locked --offline --bin dope-native-notices -- --check
# ci-command:cpu-native:merge-guard-format 2da8989742345c533f86c9b626b37039f355c187315bce1b38e2023cd6185baa
ci_run merge-guard-format check cargo fmt --manifest-path ops/merge_guard/Cargo.toml --all -- --check
# ci-command:cpu-native:merge-guard-clippy 94a58c2296c95cf887e805e5e06d891890e2395e88d12e2000c9b6db43df2b07
ci_run merge-guard-clippy check cargo clippy --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin dope-merge-guard --bin audit-proof --bin check-frozen-evidence --bin ci-shard -- -D warnings
# ci-command:cpu-native:merge-guard-test 129dfb7b509d1d3da8d1edbe8c0c71ace0c93ae73de9381966b76d5db3313d4e
ci_run merge-guard-test cargo-test cargo test --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin dope-merge-guard --bin audit-proof --bin check-frozen-evidence --bin ci-shard
# ci-command:cpu-native:native-process-clippy 5c96b3d4fe49317e10979092b69bb529d5291110fe7b3fccc84b5345e3c7ad7c
ci_run native-process-clippy check cargo clippy --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin native-process -j1 -- -D warnings
# ci-command:cpu-native:native-process-test 62144d2633e8d853615a0e7ece6ec9ca27e624c000b607f79ee645b41a351d27
ci_run native-process-test cargo-test cargo test --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin native-process -j1 -- --test-threads=1
