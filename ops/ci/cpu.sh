#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
source ops/ci/lib.sh
for tool in cargo python3 uv just latexmk pdffonts pdfinfo pdftotext rg strace tar gzip; do
  ci_require_command "$tool"
done
python3 -B -m unittest discover -s ops/ci -p test_frozen_evidence.py -v
python3 -B ops/ci/check-frozen-evidence.py
bash ops/ci/test-cpu-tool-readiness.sh
sha256sum --check --status scripts/native-legal.sha256
bash ops/ci/check-budgets.sh
bash ops/ci/check-contracts.sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
bash ops/ci/prepare-merge-guard.sh
cargo test --locked
bash ops/ci/engine.sh
cargo run --locked --offline --bin dope-native-notices -- --check
cargo fmt --manifest-path ops/merge_guard/Cargo.toml --all -- --check
cargo clippy --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin dope-merge-guard --bin audit-proof -- -D warnings
cargo test --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin dope-merge-guard --bin audit-proof
/usr/bin/python3 -m venv --system-site-packages target/ci-python
bash ops/ci/install-python-v1.sh target/ci-python/bin/python
PYTHONPATH=src target/ci-python/bin/python -m pytest -q tests
uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt 'jsonschema==4.26.0'
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_checkpoint_index research.benchmark.tests.test_validation_figure research.benchmark.tests.test_retained_metrics research.benchmark.tests.test_tabddpm_retained research.benchmark.tests.test_tabddpm_matched_coverage research.benchmark.tests.test_baseline_completion research.benchmark.tests.test_forest_fit_variance research.benchmark.tests.test_forest_complete_cohort research.benchmark.tests.test_forest_projected_cohort research.benchmark.tests.test_retained_comparison research.benchmark.tests.test_arf_tabsyn_followon research.benchmark.tests.test_cpu_fit_batch research.benchmark.tests.test_arf_contract research.benchmark.tests.test_local_checkpoint research.benchmark.tests.test_public_demo research.benchmark.tests.test_equal_bytes -q
cd validation/external
UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --locked --extra xgboost --extra lightgbm
UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --offline --locked --extra xgboost --extra lightgbm
PYTHONPATH=. ../../target/external-venv/bin/python -m unittest discover -s tests -v
cd "$root"
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_tabsyn_admission research.benchmark.tests.test_tabsyn_owned_lifecycle -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_mfs_v3 research.benchmark.tests.test_mfs_v3_panel research.benchmark.tests.test_fetch_jope research.benchmark.tests.test_fetch_pmlb -q
# A publisher edit that leaves the committed fit ledger stale fails before campaign publication.
PYTHONPATH=. python3 -m unittest research.benchmark.tests.test_dope_refinement_fits.RefinementFits.test_committed_fit_ledger_tracks_publisher -v
python3 docs/whitepaper/scripts/local_receipts.py --check
python3 research/benchmark/verify_paper_numbers.py
python3 docs/whitepaper/scripts/compute_cost.py
uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt 'matplotlib==3.10.8' 'catboost==1.2.10'
mkdir -p target/transfer-test-tmp
# Explicit packet qualification or structured not_run; full transfer discovery runs once.
PYTHONPATH=. target/ci-python/bin/python -B -m research.transfer.workplan_packet
target/ci-python/bin/python -B research/transfer/native_control_prepare.py
TMPDIR="$PWD/target/transfer-test-tmp" PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 target/ci-python/bin/python -B -m unittest discover -s research/transfer/tests -p 'test_*.py' -v
target/ci-python/bin/python docs/whitepaper/scripts/retained_evidence.py
target/ci-python/bin/python docs/whitepaper/scripts/compute_panel.py
# Required paper-tests gate: same commands as Justfile, using the V1 interpreter.
# Contains the independent committed-input replay exactly once; do not also run its old selector.
target/ci-python/bin/python -B -m unittest discover -s docs/whitepaper/scripts -p 'test_*.py'
target/ci-python/bin/python -B -m unittest discover -s research/benchmark/tests -p test_paper_build_pipeline.py -v
target/ci-python/bin/python -B -m unittest discover -s tools/claims -p 'test_*.py' -v
bash ops/ci/check-review-fixes.sh target/ci-python/bin/python
python3 docs/whitepaper/scripts/publication_rows.py
git diff --exit-code -- docs/whitepaper/generated/retained-evidence.json docs/whitepaper/generated/retained-*.tex docs/whitepaper/generated/retained-figure-hashes.json docs/whitepaper/generated/baseline-coverage.json docs/whitepaper/generated/baseline-coverage.tex docs/whitepaper/figures/retained-matched-eight.pdf docs/whitepaper/generated/published-baseline-kpis.json docs/whitepaper/generated/published-baseline-kpis.csv docs/whitepaper/generated/published-baseline-figure-hashes.json docs/whitepaper/figures/published-baseline-cohorts.pdf
target/ci-python/bin/python docs/whitepaper/scripts/render_figures.py --check
target/ci-python/bin/python docs/whitepaper/scripts/mfs_v3_table.py --check
target/ci-python/bin/python docs/whitepaper/scripts/v2_figures.py --check
target/ci-python/bin/python docs/whitepaper/scripts/readme_benchmark.py --check
target/ci-python/bin/python -B docs/whitepaper/scripts/compression_frontier.py --check
target/ci-python/bin/python -B docs/whitepaper/scripts/quickstart_block.py --check
target/ci-python/bin/python -B -m unittest discover -s examples/quickstart -p 'test_*.py' -v
git diff --exit-code -- docs/whitepaper/generated/fit-trace.json docs/whitepaper/generated/panel-stats.json docs/whitepaper/generated/numbers.tex docs/whitepaper/generated/figure-spec.tex docs/whitepaper/generated/density-table.tex docs/whitepaper/generated/neural-table.tex docs/whitepaper/generated/forest-table.tex docs/whitepaper/generated/headline-table.tex docs/whitepaper/generated/byte-table.tex docs/whitepaper/generated/threshold-table.tex docs/whitepaper/generated/threshold-counts.tex docs/whitepaper/generated/beyond-fit.tex docs/whitepaper/generated/provenance-table.tex docs/whitepaper/generated/fidelity-privacy-table.tex docs/whitepaper/generated/loss-beside-retention.tex docs/whitepaper/generated/artifact-hashes.tex docs/whitepaper/generated/coreset-paragraph.tex docs/whitepaper/generated/compute-cost.json docs/whitepaper/generated/compute-cost-receipts.json docs/whitepaper/generated/compute-cost-table.tex docs/whitepaper/generated/licenses.json docs/whitepaper/generated/availability.tex docs/whitepaper/generated/publication-rows.tex docs/whitepaper/generated/figure-hashes.json docs/whitepaper/figures/retention-bytes.pdf docs/whitepaper/figures/retention-bars.pdf docs/whitepaper/figures/paired-cdf.pdf docs/whitepaper/figures/loss-curves.pdf docs/whitepaper/figures/teaser.pdf docs/whitepaper/generated/teaser-figure-hashes.json docs/whitepaper/figures/pareto.pdf docs/whitepaper/generated/pareto-figure-hashes.json docs/whitepaper/generated/readme-figure-hashes.json docs/readme/pareto.png docs/readme/paired.png README.md HISTORY.md docs/whitepaper/generated/v2-numbers.tex docs/whitepaper/generated/v2-headline.json docs/whitepaper/generated/v2-headline-table.tex docs/whitepaper/generated/v2-size-table.tex docs/whitepaper/generated/v2-fidelity-table.tex docs/whitepaper/generated/v2-fidelity-levels.tex docs/whitepaper/generated/v2-contrasts.tex docs/whitepaper/generated/v2-levels.tex docs/whitepaper/generated/v2-seeds.tex docs/whitepaper/generated/v2-family-numbers.tex docs/whitepaper/generated/v2-family-table.tex docs/whitepaper/generated/v2-all-five-numbers.tex docs/whitepaper/generated/v2-all-five-table.tex docs/whitepaper/generated/mfs-v3-density-table.tex docs/whitepaper/generated/mfs-v3-scalar.tex docs/whitepaper/generated/mfs-v3-layers.tex docs/whitepaper/generated/mfs-v3-figure-hashes.json docs/whitepaper/figures/mfs-v3-retention-bars.pdf docs/whitepaper/figures/mfs-v3-paired.pdf research/benchmark/results/mfs-v3-density-panel.json
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_verify_paper_numbers research.benchmark.tests.test_paper_log
# PDF admission replays scientific inputs using the provisioned V1 interpreter.
export PATH="$PWD/target/ci-python/bin:$PATH"
bash docs/whitepaper/scripts/build_pdf.sh
python3 docs/whitepaper/scripts/check_paper.py
rm -rf target/external-venv target/paper-render target/paper-build target/transfer-test-tmp target/cpu-tool-readiness target/review-test-tmp target/claims-tests target/tmp
find target/debug/deps -maxdepth 1 -type f -executable ! -name '*.so' -delete || true
