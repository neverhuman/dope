set shell := ["bash", "-euo", "pipefail", "-c"]

export CARGO_INCREMENTAL := "0"
export CARGO_PROFILE_DEV_DEBUG := "line-tables-only"

setup:
    cargo fetch --locked
    mkdir -p target/tmp
    export TMPDIR="$PWD/target/tmp"; python3 -m venv --system-site-packages target/venv
    bash ops/ci/install-python-v1.sh target/venv/bin/python
    TMPDIR="$PWD/target/tmp" target/venv/bin/python -m pip install --no-deps -e .
    mkdir -p target/python-metadata
    if test -d src/dope.egg-info; then mv src/dope.egg-info "target/python-metadata/dope.$(date +%s%N).egg-info"; fi
    if test -d src/dope_kernel.egg-info; then mv src/dope_kernel.egg-info "target/python-metadata/dope_kernel.$(date +%s%N).egg-info"; fi

native-guard-check:
    bash ops/ci/prepare-merge-guard.sh
    cargo fmt --manifest-path ops/merge_guard/Cargo.toml --all -- --check
    cargo clippy --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin dope-merge-guard -- -D warnings
    cargo test --manifest-path ops/merge_guard/Cargo.toml --target-dir target --locked --bin dope-merge-guard

fast: native-guard-check
    bash ops/ci/test-cpu-tool-readiness.sh
    sha256sum --check --status scripts/native-legal.sha256
    cargo fmt --all -- --check
    cargo check -p dope-kernel --locked
    cargo test --locked --lib
    cargo test --locked --bin dope-native-notices
    cargo run --locked --offline --bin dope-native-notices -- --check
    PYTHONPATH=src python3 -m pytest -q tests
    python3 -B -m unittest discover -s research/transfer/tests -p 'test_*.py' -v
    python3 -B -m unittest research.benchmark.tests.test_local_checkpoint research.benchmark.tests.test_public_demo -q
    python3 -B -m unittest research.benchmark.tests.test_fetch_jope research.benchmark.tests.test_fetch_pmlb -q
    PYTHONPATH=docs/whitepaper/scripts python3 -B -m unittest test_rust_closure -q

focus filter:
    cargo check -p dope-kernel --locked
    cargo test -p dope-kernel --locked --lib {{filter}}

release-local-native:
    RUSTFLAGS="-C target-cpu=native" cargo build --locked --release --offline -p dope-kernel --bin dope

native-release-smoke:
    bash ops/ci/native-release-smoke.sh

fast-audit:
    bash agent/run-jankurai-fast.sh

check: fast
    bash ops/ci/check-budgets.sh
    bash ops/ci/check-contracts.sh
    cargo clippy --all-targets --locked -- -D warnings
    cargo test --locked --test rust_cli
    cargo test --locked --test vnext_raw_schema --test vnext_raw_workflow --test vnext_artifact_only --test vnext_native_install --test vnext_byte_allocation --test vnext_native_successors --test vnext_public_demo --test vnext_geometry_binding --test vnext_interactions
    git diff --check

security:
    bash ops/ci/security.sh

contract-drift:
    bash ops/ci/check-contracts.sh

score:
    bash ops/ci/audit.sh run

golden-parity:
    python3 -B -m research.transfer.tdl.golden --check
    cargo test --locked --lib metrics::golden

paper-tdl:
    bash paper/build.sh

paper:
    python3 ops/ci/paper-prepare.py
    python3 -B docs/whitepaper/scripts/native_byte_conformance.py --write
    python3 docs/whitepaper/scripts/local_receipts.py
    python3 research/benchmark/verify_paper_numbers.py --write-evidence
    python3 docs/whitepaper/scripts/compute_cost.py
    python3 docs/whitepaper/scripts/licenses.py
    python3 docs/whitepaper/scripts/retained_evidence.py
    python3 docs/whitepaper/scripts/compute_panel.py
    python3 docs/whitepaper/scripts/publication_rows.py
    python3 docs/whitepaper/scripts/mfs_v3_table.py
    python3 research/benchmark/verify_paper_numbers.py
    python3 docs/whitepaper/scripts/render_figures.py
    python3 -B docs/whitepaper/scripts/paper_build.py

paper-check:
    python3 docs/whitepaper/scripts/local_receipts.py --check
    python3 research/benchmark/verify_paper_numbers.py
    python3 docs/whitepaper/scripts/compute_cost.py
    python3 docs/whitepaper/scripts/retained_evidence.py
    python3 docs/whitepaper/scripts/compute_panel.py
    python3 docs/whitepaper/scripts/publication_rows.py
    python3 docs/whitepaper/scripts/mfs_v3_table.py
    python3 docs/whitepaper/scripts/render_figures.py --check
    git diff --exit-code -- docs/whitepaper/generated/retained-evidence.json docs/whitepaper/generated/retained-*.tex docs/whitepaper/generated/retained-figure-hashes.json docs/whitepaper/generated/baseline-coverage.json docs/whitepaper/generated/baseline-coverage.tex docs/whitepaper/figures/retained-matched-eight.pdf docs/whitepaper/generated/published-baseline-kpis.json docs/whitepaper/generated/published-baseline-kpis.csv docs/whitepaper/generated/published-baseline-figure-hashes.json docs/whitepaper/figures/published-baseline-cohorts.pdf
    git diff --exit-code -- docs/whitepaper/generated/fit-trace.json docs/whitepaper/generated/panel-stats.json docs/whitepaper/generated/numbers.tex docs/whitepaper/generated/figure-spec.tex docs/whitepaper/generated/density-table.tex docs/whitepaper/generated/neural-table.tex docs/whitepaper/generated/forest-table.tex docs/whitepaper/generated/headline-table.tex docs/whitepaper/generated/byte-table.tex docs/whitepaper/generated/threshold-table.tex docs/whitepaper/generated/threshold-counts.tex docs/whitepaper/generated/beyond-fit.tex docs/whitepaper/generated/provenance-table.tex docs/whitepaper/generated/fidelity-privacy-table.tex docs/whitepaper/generated/loss-beside-retention.tex docs/whitepaper/generated/artifact-hashes.tex docs/whitepaper/generated/coreset-paragraph.tex docs/whitepaper/generated/compute-cost.json docs/whitepaper/generated/compute-cost-receipts.json docs/whitepaper/generated/compute-cost-table.tex docs/whitepaper/generated/licenses.json docs/whitepaper/generated/availability.tex docs/whitepaper/generated/publication-rows.tex docs/whitepaper/generated/figure-hashes.json docs/whitepaper/figures/retention-bytes.pdf docs/whitepaper/figures/retention-bars.pdf docs/whitepaper/figures/paired-cdf.pdf docs/whitepaper/figures/loss-curves.pdf docs/whitepaper/figures/teaser.pdf docs/whitepaper/generated/teaser-figure-hashes.json docs/whitepaper/figures/pareto.pdf docs/whitepaper/generated/pareto-figure-hashes.json docs/whitepaper/generated/readme-figure-hashes.json docs/readme/pareto.png docs/readme/paired.png README.md docs/whitepaper/generated/v2-numbers.tex docs/whitepaper/generated/v2-headline.json docs/whitepaper/generated/v2-headline-table.tex docs/whitepaper/generated/v2-size-table.tex docs/whitepaper/generated/v2-fidelity-table.tex docs/whitepaper/generated/v2-fidelity-levels.tex docs/whitepaper/generated/v2-contrasts.tex docs/whitepaper/generated/v2-levels.tex docs/whitepaper/generated/v2-seeds.tex docs/whitepaper/generated/v2-family-numbers.tex docs/whitepaper/generated/v2-family-table.tex docs/whitepaper/generated/v2-all-five-numbers.tex docs/whitepaper/generated/v2-all-five-table.tex docs/whitepaper/generated/mfs-v3-density-table.tex docs/whitepaper/generated/mfs-v3-scalar.tex docs/whitepaper/generated/mfs-v3-layers.tex docs/whitepaper/generated/mfs-v3-figure-hashes.json docs/whitepaper/figures/mfs-v3-retention-bars.pdf docs/whitepaper/figures/mfs-v3-paired.pdf research/benchmark/results/mfs-v3-density-panel.json
    python3 -B docs/whitepaper/scripts/paper_build.py --check
    git diff --exit-code -- docs/whitepaper/generated/architecture-figure-hashes.json docs/whitepaper/figures/architecture.pdf

# Preserve the independent paper-script suite as its own explicit gate.
# Its committed-input authentication test is a full replay; admission is separate.
paper-tests:
    python3 -B -m unittest discover -s docs/whitepaper/scripts -p 'test_*.py'
    python3 -B -m unittest discover -s research/benchmark/tests -p test_paper_build_pipeline.py -v
