#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
bash ops/ci/install-jankurai.sh
auditor=target/jankurai/jankurai-1.7.1-x86_64-unknown-linux-gnu/jankurai
comparison_base="${JANKURAI_COMPARISON_BASE:-origin/main}"
if ! git rev-parse --verify "${comparison_base}^{commit}" >/dev/null 2>&1; then
  comparison_base=origin/main
fi
if [[ "$(git rev-parse "${comparison_base}^{commit}")" == "$(git rev-parse HEAD)" ]]; then
  comparison_base=HEAD^
fi
git rev-parse --verify "${comparison_base}^{commit}" >/dev/null
"$auditor" proof . --changed-from "$comparison_base" \
  --out target/jankurai/proof-plan.json --md target/jankurai/proof-plan.md
"$auditor" copy-code . --json target/jankurai/copy-code.json --md target/jankurai/copy-code.md
"$auditor" rust witness build . --out target/jankurai/rust/witness-graph.json
ci_require_artifact target/jankurai/proof-plan.json
ci_require_artifact target/jankurai/copy-code.json
ci_require_artifact target/jankurai/rust/witness-graph.json
