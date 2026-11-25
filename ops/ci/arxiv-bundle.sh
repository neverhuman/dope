#!/usr/bin/env bash
# Verify turnkey arXiv v3 bundle parity + isolated compile, and 2-page CEO/CTO executive brief PDFs.
set -euo pipefail
cd "$(dirname "$0")/../.."
bash docs/whitepaper-v3/scripts/build_arxiv_bundle.sh --check --verify
bash docs/briefs/scripts/build_briefs.sh --check
