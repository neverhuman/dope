#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
source ops/ci/shard-runtime.sh
ci_shard_init cpu-compat
for tool in cargo python3 uv; do ci_require_command "$tool"; done
ci_frozen_check
ci_prepare_receipts
# ci-command:cpu-compat:python-v1-venv 5d99194eb3c43f3c411b64a193c56695ffd1245a1f072014d7ad17a504ccab3b
ci_run python-v1-venv check /usr/bin/python3 -m venv --system-site-packages target/ci-python
# ci-command:cpu-compat:python-v1-install 21865de22cd96ed38c1077970a1eccc930e8d7d65d57d5357c6475a24aa196e7
ci_run python-v1-install check bash ops/ci/install-python-v1.sh target/ci-python/bin/python
# ci-command:cpu-compat:engine-benchmark ce728c26f336f5a414d2c0a538f5c74864eac444dd13fe027aafacb044637d7e
ci_run engine-benchmark unittest bash ops/ci/engine-benchmark.sh target/ci-python/bin/python
# ci-command:cpu-compat:project-pytest 237dd9730f9acd84acf68ae42f267e1232e6b4e18a90afda683c81ed4e2930e9
ci_run project-pytest pytest env PYTHONPATH=src target/ci-python/bin/python -m pytest -q tests
# ci-command:cpu-compat:jsonschema-install 7d578618510fbcc73b04365ff8f0b76a80395cd5f5c1b56bbbffe2747220010b
ci_run jsonschema-install check uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt jsonschema==4.26.0
# ci-command:cpu-compat:benchmark-tests 088bd9c2f0073d382f5a0c693feedc094bbd714213324d5ce248079180e2bab5
ci_run benchmark-tests unittest env PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_checkpoint_index research.benchmark.tests.test_validation_figure research.benchmark.tests.test_retained_metrics research.benchmark.tests.test_tabddpm_retained research.benchmark.tests.test_tabddpm_matched_coverage research.benchmark.tests.test_baseline_completion research.benchmark.tests.test_forest_fit_variance research.benchmark.tests.test_forest_complete_cohort research.benchmark.tests.test_forest_projected_cohort research.benchmark.tests.test_retained_comparison research.benchmark.tests.test_arf_tabsyn_followon research.benchmark.tests.test_cpu_fit_batch research.benchmark.tests.test_arf_contract research.benchmark.tests.test_local_checkpoint research.benchmark.tests.test_public_demo research.benchmark.tests.test_equal_bytes -q
cd validation/external
# ci-command:cpu-compat:external-online 50f715182395cc5494a0071edfd676b0fb120443ea09a52ed08d6d646e0bed75
ci_run external-online check env UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --locked --extra xgboost --extra lightgbm
# ci-command:cpu-compat:external-offline 6869ef8304ffe5c309e9a41cf7e296d0c6b0029bd19582a90735e661f2127e56
ci_run external-offline check env UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --offline --locked --extra xgboost --extra lightgbm
# ci-command:cpu-compat:external-tests 32075a3dcbe895fd91fa3d1d0849e1b3fca58710111a20eaba198ded373f33d2
ci_run external-tests unittest env PYTHONPATH=. ../../target/external-venv/bin/python -m unittest discover -s tests -v
cd "$root"
# ci-command:cpu-compat:tabsyn-tests c173ee1958c67de0cb5eaac682da67701b902515cb90bcb9e6cd4d100e77dfe5
ci_run tabsyn-tests unittest env PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_tabsyn_admission research.benchmark.tests.test_tabsyn_owned_lifecycle -q
# ci-command:cpu-compat:mfs-tests cea3667209337dbd548f11fba972200f8e1a963c777f8f656aef01c792ee6c42
ci_run mfs-tests unittest env PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_mfs_v3 research.benchmark.tests.test_mfs_v3_panel research.benchmark.tests.test_fetch_jope research.benchmark.tests.test_fetch_pmlb -q
# ci-command:cpu-compat:fit-ledger-test 30e4c7f856bfef4b516bf8bdfa9f6878a6f13999d8ecb0f05a0134244dafa5d5
ci_run fit-ledger-test unittest env PYTHONPATH=. python3 -m unittest research.benchmark.tests.test_dope_refinement_fits.RefinementFits.test_committed_fit_ledger_tracks_publisher -v
# ci-command:cpu-compat:shared-levels-prepare 164a9c5bb6dacd80c7f6b2bbff5d86076ae24b3de1756873b613d17cdb94ebca
ci_run shared-levels-prepare check bash ops/ci/shared-levels.sh prepare-torch
# ci-command:cpu-compat:next-version-reference 04aeb858ee9f7714d22e6540431a71842dd5dddab58c77ebfc15699431d04303
ci_run next-version-reference unittest-required env SHARED_PYTHON=target/shared-python/bin/python bash ops/ci/shared-levels.sh torch test_next_version
# ci-command:cpu-compat:r10-tests 97e8d3665264c7218b1f0b07f9c0cbaa8b57877c1f58915d19fd9e4e67940af9
ci_run r10-tests unittest target/ci-python/bin/python -B -m unittest research.transfer.privacy.test_summary_study -v
# ci-command:cpu-compat:u04-admissions 5732c95b52c02335262fb2a77bcb46750f304f4f4666c4901d2aaa34b4490f92
ci_run u04-admissions unittest target/ci-python/bin/python -B examples/public-regression/bikeshare-observational/test_runner.py
out=target/healthcare-sim
# ci-command:cpu-compat:u03-prepare a5175ae4ae1e955e33c2a4f00531b7ae9b1e28a4c63830d4bec6276cfd27c745
ci_run u03-prepare check target/ci-python/bin/python -B examples/healthcare-sim/demo.py prepare --out "$out"
# ci-command:cpu-compat:u03-verify 857344f8ce4b514a2478a8d0342f5cd66775fe2b7fccfb49fdc9d631273023a4
ci_run u03-verify check target/ci-python/bin/python -B examples/healthcare-sim/demo.py verify --out "$out"
# ci-command:cpu-compat:u03-build 310c08d507f169540deb39dedde1c9715c26e10ba20484c44060e73446c8afad
ci_run u03-build check cargo build --locked --bin dope-kernel
# ci-command:cpu-compat:u03-compile b5a77991e1c3a8317f4a0e7c62cce0586e8b5eb78facad8dca6674579d2f2f92
ci_run u03-compile check target/debug/dope-kernel compile-csv --csv "$out/fit.csv" --target event_30d --candidate chow_liu --seed 1729 --deadline-seconds 60 --out "$out/model.dpk"
# ci-command:cpu-compat:u03-sample abe9c9c02958e18164c27e3e8be84e7f27e09ab4f0b7ac6a286ecb556020ed2a
ci_run u03-sample check target/debug/dope-kernel sample-mixed --artifact "$out/model.dpk" --rows 640 --seed 1729 --out "$out/sampled.json"
# ci-command:cpu-compat:u03-evaluate cc797242a6d14e30868f87538f2187fc604d2ec2b5b5afcc60dfae761320f19c
ci_run u03-evaluate check target/ci-python/bin/python -B examples/healthcare-sim/demo.py evaluate --out "$out" --sampled "$out/sampled.json"
# ci-command:cpu-compat:u03-cleanup 399cd1e593fffdee4029d52a05cd1f26514c510f426128fd89c8e3759321b7c7
ci_run u03-cleanup check rm -rf "$out"
# ci-command:cpu-compat:public-claims af6f2d1bdfb25fa98897c6acb3e2ab0527d5b53ed88269513c50b5ba0cdc34ec
ci_run public-claims check target/ci-python/bin/python tools/claims/check.py
# ci-command:cpu-compat:claims-render 8e17b62df24defabb8a31b6cc4291668c3a53894f163b4d0c565390fcd7b14d0
ci_run claims-render check target/ci-python/bin/python tools/claims/render.py --check
# ci-command:cpu-compat:golden-parity a64bf7b68c5c7fa15e41812346893518b6598da5c890901d4e4d2d08c22b8566
ci_run golden-parity check target/ci-python/bin/python -m research.transfer.tdl.golden --check
