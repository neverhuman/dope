#!/usr/bin/env bash
# Small protocol fixtures only; no benchmark partitions or registry are opened.
set -euo pipefail
cd "$(dirname "$0")/../.."
review_python="${1:-python3}"
mkdir -p target/review-test-tmp
TMPDIR="$PWD/target/review-test-tmp" PYTHONPATH=. "$review_python" -B -m unittest \
  research.benchmark.review_fixes.test_review_stats \
  research.benchmark.review_fixes.test_sealed_gate \
  research.benchmark.review_fixes.test_sample_paths \
  research.benchmark.review_fixes.test_review_ingest \
  research.benchmark.review_fixes.test_source_pins \
  research.benchmark.review_fixes.test_fit_seed_singleton \
  research.benchmark.review_fixes.test_tabsyn_fidelity_cells \
  research.benchmark.review_fixes.test_tabsyn_sample_evidence \
  research.benchmark.review_fixes.test_stats_v2 \
  research.benchmark.review_fixes.test_predeclare_v2 \
  research.benchmark.review_fixes.test_rescore_cells -q
TMPDIR="$PWD/target/review-test-tmp" PYTHONPATH=. "$review_python" -B -m unittest discover \
  -s docs/whitepaper/scripts -p 'test_paper_control_inputs.py' -q
