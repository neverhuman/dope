#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
source ops/ci/lib.sh
ci_require_command cargo
ci_require_command python3
bash ops/ci/check-budgets.sh
bash ops/ci/check-contracts.sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
python3 -m venv --system-site-packages target/ci-python
bash ops/ci/install-python-v1.sh target/ci-python/bin/python
PYTHONPATH=src target/ci-python/bin/python -m pytest -q tests
ci_require_command uv
uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt 'jsonschema==4.26.0'
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_checkpoint_index -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_validation_figure -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_retained_metrics -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_tabddpm_retained -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_forest_fit_variance -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_forest_complete_cohort -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_forest_projected_cohort -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_retained_comparison -q
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_cpu_fit_batch research.benchmark.tests.test_arf_contract -q
cd validation/external
UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --locked --extra xgboost --extra lightgbm
UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --offline --locked --extra xgboost --extra lightgbm
PYTHONPATH=. ../../target/external-venv/bin/python -m unittest discover -s tests -v
cd "$root"
PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_tabsyn_admission research.benchmark.tests.test_tabsyn_owned_lifecycle -q
# A publisher edit that leaves the committed fit ledger stale fails before campaign publication.
PYTHONPATH=. python3 -m unittest research.benchmark.tests.test_dope_refinement_fits.RefinementFits.test_committed_fit_ledger_tracks_publisher -v
python3 research/benchmark/verify_paper_numbers.py
python3 docs/whitepaper/scripts/compute_cost.py
uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt 'matplotlib==3.10.8'
target/ci-python/bin/python docs/whitepaper/scripts/retained_evidence.py
target/ci-python/bin/python docs/whitepaper/scripts/compute_panel.py
target/ci-python/bin/python -B -m unittest discover -s docs/whitepaper/scripts -p 'test_retained_evidence.py'
target/ci-python/bin/python -B -m unittest discover -s docs/whitepaper/scripts -p 'test_published_baseline_inventory.py'
python3 -B -m unittest discover -s docs/whitepaper/scripts -p 'test_publication_rows.py'
python3 docs/whitepaper/scripts/publication_rows.py
git diff --exit-code -- docs/whitepaper/generated/retained-evidence.json docs/whitepaper/generated/retained-*.tex docs/whitepaper/generated/retained-figure-hashes.json docs/whitepaper/generated/baseline-coverage.json docs/whitepaper/generated/baseline-coverage.tex docs/whitepaper/figures/retained-matched-eight.pdf docs/whitepaper/generated/published-baseline-kpis.json docs/whitepaper/generated/published-baseline-kpis.csv docs/whitepaper/generated/published-baseline-figure-hashes.json docs/whitepaper/figures/published-baseline-cohorts.pdf
if python3 -c 'import matplotlib' >/dev/null 2>&1; then
  python3 docs/whitepaper/scripts/render_figures.py --check
fi
git diff --exit-code -- docs/whitepaper/generated/fit-trace.json docs/whitepaper/generated/panel-stats.json docs/whitepaper/generated/numbers.tex docs/whitepaper/generated/density-table.tex docs/whitepaper/generated/neural-table.tex docs/whitepaper/generated/forest-table.tex docs/whitepaper/generated/headline-table.tex docs/whitepaper/generated/byte-table.tex docs/whitepaper/generated/threshold-table.tex docs/whitepaper/generated/threshold-counts.tex docs/whitepaper/generated/beyond-fit.tex docs/whitepaper/generated/provenance-table.tex docs/whitepaper/generated/fidelity-privacy-table.tex docs/whitepaper/generated/loss-beside-retention.tex docs/whitepaper/generated/artifact-hashes.tex docs/whitepaper/generated/coreset-paragraph.tex docs/whitepaper/generated/compute-cost.json docs/whitepaper/generated/compute-cost-receipts.json docs/whitepaper/generated/compute-cost-table.tex docs/whitepaper/generated/licenses.json docs/whitepaper/generated/availability.tex docs/whitepaper/generated/publication-rows.tex docs/whitepaper/generated/figure-hashes.json docs/whitepaper/figures/retention-bytes.pdf docs/whitepaper/figures/retention-bars.pdf docs/whitepaper/figures/paired-cdf.pdf docs/whitepaper/figures/loss-curves.pdf
python3 -B -m unittest research.benchmark.tests.test_verify_paper_numbers research.benchmark.tests.test_paper_log
if ! command -v latexmk >/dev/null 2>&1 || ! command -v pdffonts >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y latexmk poppler-utils texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-pictures texlive-publishers
fi
bash docs/whitepaper/scripts/build_pdf.sh
python3 docs/whitepaper/scripts/check_paper.py
