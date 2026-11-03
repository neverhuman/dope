#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/shard-runtime.sh
ci_shard_init cpu-transfer-paper
for tool in cargo python3 uv just latexmk pdffonts pdfinfo pdftotext rg strace tar gzip; do ci_require_command "$tool"; done
ci_frozen_check
ci_prepare_receipts
# ci-command:cpu-transfer-paper:cpu-tool-readiness f414d47701dc5f42592be8111e602f176347e276c32eb9715c6fe4b5eba5d81b
ci_run cpu-tool-readiness readiness bash ops/ci/test-cpu-tool-readiness.sh
# ci-command:cpu-transfer-paper:python-v1-venv 5d99194eb3c43f3c411b64a193c56695ffd1245a1f072014d7ad17a504ccab3b
ci_run python-v1-venv check /usr/bin/python3 -m venv --system-site-packages target/ci-python
# ci-command:cpu-transfer-paper:python-v1-install 21865de22cd96ed38c1077970a1eccc930e8d7d65d57d5357c6475a24aa196e7
ci_run python-v1-install check bash ops/ci/install-python-v1.sh target/ci-python/bin/python
# ci-command:cpu-transfer-paper:jsonschema-install 7d578618510fbcc73b04365ff8f0b76a80395cd5f5c1b56bbbffe2747220010b
ci_run jsonschema-install check uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt jsonschema==4.26.0
# ci-command:cpu-transfer-paper:local-receipts 98c16613280083aae2e249c33484e8a02fd5bcabe114ccd273e8b6f1f28e3ecd
ci_run local-receipts check python3 docs/whitepaper/scripts/local_receipts.py --check
# ci-command:cpu-transfer-paper:paper-numbers bdc603a6aa0273b4af7c36cd6d62024d51a4143efc26f3c83969944427a167ce
ci_run paper-numbers check python3 research/benchmark/verify_paper_numbers.py
# ci-command:cpu-transfer-paper:compute-cost 13e411791173ff11d96791a5a4543937f7f89ff51a20236bc301f3a64003a84c
ci_run compute-cost check python3 docs/whitepaper/scripts/compute_cost.py
# ci-command:cpu-transfer-paper:paper-dependencies 012b84a5b306ca3418577510d087e36c8dc620099082ffc0c24873f6b7b22d1b
ci_run paper-dependencies check uv pip install --python target/ci-python/bin/python --constraint validation/requirements-v1.txt matplotlib==3.10.8 catboost==1.2.10
# ci-command:cpu-transfer-paper:transfer-test-tmp f91fa118f36735ca80efb061fe4ca2ce16bfb0c83ee782119d0255d33c98376d
ci_run transfer-test-tmp check mkdir -p target/transfer-test-tmp
# ci-command:cpu-transfer-paper:workplan-packet b9226e47959039326a2e80e5daefd99dcd96921b1c5511dbaa8d69d46b5e9b20
ci_run workplan-packet check env PYTHONPATH=. target/ci-python/bin/python -B -m research.transfer.workplan_packet
# ci-command:cpu-transfer-paper:native-control-cache-warm c2dcc3791143da79a6d13a559a633e8a514790de14ff7d810a9c160e73c90d4e
ci_run native-control-cache-warm check cargo fetch --manifest-path ops/native_control/Cargo.toml --locked
# Cargo verifies every fetched crate against Cargo.lock and refuses lock changes.
# ci-command:cpu-transfer-paper:native-control-prepare c590f690703ecd8159089f051e0ea0d34fa17bb034856bea653e550adbf1d52a
ci_run native-control-prepare check target/ci-python/bin/python -B research/transfer/native_control_prepare.py
# ci-command:cpu-transfer-paper:transfer-tests dac2bc433e8e9d0a48a5ea6f8e6092cb8f04c76ac89846a9102e7c3c5d16ea67
ci_run transfer-tests unittest env TMPDIR="$PWD/target/transfer-test-tmp" PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 target/ci-python/bin/python -B -m unittest discover -s research/transfer/tests -p 'test_*.py' -v
# ci-command:cpu-transfer-paper:retained-evidence 2669e9486d67805926999ff635c51d71d8c38d18afb06c9f614217f74cf0425d
ci_run retained-evidence check target/ci-python/bin/python docs/whitepaper/scripts/retained_evidence.py
# ci-command:cpu-transfer-paper:compute-panel c9c6058b836f79f853b5fa1eda6ddcd2a90c16ddcd908557f5e8c3e568027cb4
ci_run compute-panel check target/ci-python/bin/python docs/whitepaper/scripts/compute_panel.py
# ci-command:cpu-transfer-paper:paper-script-tests a74e15e18d8f0468306cea8099ef8a19f890ad2634053c9e37efa742a2eb4dd1
ci_run paper-script-tests unittest target/ci-python/bin/python -B -m unittest discover -s docs/whitepaper/scripts -p 'test_*.py'
# ci-command:cpu-transfer-paper:pipeline-tests adc99c383aa1e39f150ad2ff8fb23d04d368d15d0812bae5e3dc17a141799e12
ci_run pipeline-tests unittest target/ci-python/bin/python -B -m unittest discover -s research/benchmark/tests -p test_paper_build_pipeline.py -v
# ci-command:cpu-transfer-paper:claims-tests dc361e9398eb00c269256196a8fd47b21c0fb13f75eb676dc04cad728227d7e8
ci_run claims-tests unittest target/ci-python/bin/python -B -m unittest discover -s tools/claims -p 'test_*.py' -v
# ci-command:cpu-transfer-paper:review-fixes 5f23313dd0d7772d21104fb2e730017544ab18acad22e31083d854f1f763fcdb
ci_run review-fixes unittest bash ops/ci/check-review-fixes.sh target/ci-python/bin/python
# ci-command:cpu-transfer-paper:publication-rows 6dde6be72d18bb6efba4c628080ab3ba0d804dce1402f98dbc8f02f361831b1a
ci_run publication-rows check python3 docs/whitepaper/scripts/publication_rows.py
# ci-command:cpu-transfer-paper:retained-drift cda52a08061857458aaf3f9d44fa7866695a326d75b14144fb19c01db133a68b
ci_run retained-drift check bash -c 'git diff --exit-code -- docs/whitepaper/generated/retained-evidence.json docs/whitepaper/generated/retained-*.tex docs/whitepaper/generated/retained-figure-hashes.json docs/whitepaper/generated/baseline-coverage.json docs/whitepaper/generated/baseline-coverage.tex docs/whitepaper/figures/retained-matched-eight.pdf docs/whitepaper/generated/published-baseline-kpis.json docs/whitepaper/generated/published-baseline-kpis.csv docs/whitepaper/generated/published-baseline-figure-hashes.json docs/whitepaper/figures/published-baseline-cohorts.pdf'
# ci-command:cpu-transfer-paper:render-figures c90e87d4cacd63f768f0da1156d0eae15a0367b8e2c2389452d64f6e85db1eff
ci_run render-figures check target/ci-python/bin/python docs/whitepaper/scripts/render_figures.py --check
# ci-command:cpu-transfer-paper:mfs-table 9f44e2bc73ff04885827ce0de6dc5ffcfd8d4b9a13a9955e731bef4b1c7f7834
ci_run mfs-table check target/ci-python/bin/python docs/whitepaper/scripts/mfs_v3_table.py --check
# ci-command:cpu-transfer-paper:v2-figures 33e344ea63475b33eeeeab9d39925817da9e329a3d4d5b92f6f71656487813fa
ci_run v2-figures check target/ci-python/bin/python docs/whitepaper/scripts/v2_figures.py --check
# ci-command:cpu-transfer-paper:readme-benchmark ddfe735776170584bdd89aeeb9d48215e1b1ab88d8b213c6dfbb22f85240426b
ci_run readme-benchmark check target/ci-python/bin/python docs/whitepaper/scripts/readme_benchmark.py --check
# ci-command:cpu-transfer-paper:compression-frontier b65a68a2be1a0d7e4fe077f5a96260ffbf0f40996e17363ad6b30b5488a4dd3e
ci_run compression-frontier check target/ci-python/bin/python -B docs/whitepaper/scripts/compression_frontier.py --check
# ci-command:cpu-transfer-paper:quickstart-block 4136269c2021fd5cb6056f1000ab58a61aa62f721e2a06e381de8e6604b22002
ci_run quickstart-block check target/ci-python/bin/python -B docs/whitepaper/scripts/quickstart_block.py --check
# ci-command:cpu-transfer-paper:quickstart-tests 11b5216907e3dd32009801702869196f8e792e141f169fc1d6b262a736ac8088
ci_run quickstart-tests unittest target/ci-python/bin/python -B -m unittest discover -s examples/quickstart -p 'test_*.py' -v
# ci-command:cpu-transfer-paper:main-output-drift 88be8afb527a1dcd7ff467479d6eecdfd7af312adbc1ee666752778e3688a889
ci_run main-output-drift check bash -c 'git diff --exit-code -- docs/whitepaper/generated/fit-trace.json docs/whitepaper/generated/panel-stats.json docs/whitepaper/generated/numbers.tex docs/whitepaper/generated/figure-spec.tex docs/whitepaper/generated/density-table.tex docs/whitepaper/generated/neural-table.tex docs/whitepaper/generated/forest-table.tex docs/whitepaper/generated/headline-table.tex docs/whitepaper/generated/byte-table.tex docs/whitepaper/generated/threshold-table.tex docs/whitepaper/generated/threshold-counts.tex docs/whitepaper/generated/beyond-fit.tex docs/whitepaper/generated/provenance-table.tex docs/whitepaper/generated/fidelity-privacy-table.tex docs/whitepaper/generated/loss-beside-retention.tex docs/whitepaper/generated/artifact-hashes.tex docs/whitepaper/generated/coreset-paragraph.tex docs/whitepaper/generated/compute-cost.json docs/whitepaper/generated/compute-cost-receipts.json docs/whitepaper/generated/compute-cost-table.tex docs/whitepaper/generated/licenses.json docs/whitepaper/generated/availability.tex docs/whitepaper/generated/publication-rows.tex docs/whitepaper/generated/figure-hashes.json docs/whitepaper/figures/retention-bytes.pdf docs/whitepaper/figures/retention-bars.pdf docs/whitepaper/figures/paired-cdf.pdf docs/whitepaper/figures/loss-curves.pdf docs/whitepaper/figures/teaser.pdf docs/whitepaper/generated/teaser-figure-hashes.json docs/whitepaper/figures/pareto.pdf docs/whitepaper/generated/pareto-figure-hashes.json docs/whitepaper/generated/readme-figure-hashes.json docs/readme/pareto.png docs/readme/paired.png README.md HISTORY.md docs/whitepaper/generated/v2-numbers.tex docs/whitepaper/generated/v2-headline.json docs/whitepaper/generated/v2-headline-table.tex docs/whitepaper/generated/v2-size-table.tex docs/whitepaper/generated/v2-fidelity-table.tex docs/whitepaper/generated/v2-fidelity-levels.tex docs/whitepaper/generated/v2-contrasts.tex docs/whitepaper/generated/v2-levels.tex docs/whitepaper/generated/v2-seeds.tex docs/whitepaper/generated/v2-family-numbers.tex docs/whitepaper/generated/v2-family-table.tex docs/whitepaper/generated/v2-all-five-numbers.tex docs/whitepaper/generated/v2-all-five-table.tex docs/whitepaper/generated/mfs-v3-density-table.tex docs/whitepaper/generated/mfs-v3-scalar.tex docs/whitepaper/generated/mfs-v3-layers.tex docs/whitepaper/generated/mfs-v3-figure-hashes.json docs/whitepaper/figures/mfs-v3-retention-bars.pdf docs/whitepaper/figures/mfs-v3-paired.pdf research/benchmark/results/mfs-v3-density-panel.json'
# ci-command:cpu-transfer-paper:paper-number-tests 13f781bcdd59b4cf2c47bc39dac96b041bbd7b1418dbc2a25e982691ebffe276
ci_run paper-number-tests unittest env PYTHONPATH=. target/ci-python/bin/python -B -m unittest research.benchmark.tests.test_verify_paper_numbers research.benchmark.tests.test_paper_log
export PATH="$PWD/target/ci-python/bin:$PATH"
# ci-command:cpu-transfer-paper:pdf-build c446ec3d994cce96329e8069738a297f69080b8f0d76f2ce5629f35dbd0d692f
ci_run pdf-build check bash docs/whitepaper/scripts/build_pdf.sh
# ci-command:cpu-transfer-paper:paper-admission 3541714de08074d81dd13da2cf775cd8d6e9e9d1ac75d0c55d89357fa37e77d4
ci_run paper-admission check python3 docs/whitepaper/scripts/check_paper.py
# ci-command:cpu-transfer-paper:cleanup 50f2c6495d7077be873f04ebadd27901938dfe01f9e12224d06f80d0ca50ea39
ci_run cleanup check rm -rf target/external-venv target/paper-render target/paper-build target/transfer-test-tmp target/cpu-tool-readiness target/review-test-tmp target/claims-tests target/tmp
# ci-command:cpu-transfer-paper:cleanup-cargo-deps 14d5cf8417fbd0255c0a4d660ba1bafe0bf2c21f90fcbfa0eecc673d42becd13
ci_run cleanup-cargo-deps check bash -c "find target/debug/deps -maxdepth 1 -type f -executable ! -name '*.so' -delete || true"
