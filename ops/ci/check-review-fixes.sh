#!/usr/bin/env bash
# Small protocol fixtures only; no benchmark partitions or registry are opened.
set -euo pipefail
cd "$(dirname "$0")/../.."
review_python="${1:-python3}"
mkdir -p target/review-test-tmp
TMPDIR="$PWD/target/review-test-tmp" PYTHONPATH=. "$review_python" -B -m unittest \
  research.benchmark.review_fixes.test_review_stats \
  research.benchmark.review_fixes.test_sealed_gate \
  research.benchmark.review_fixes.test_sample_paths -q
