#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
bash ops/ci/install-jankurai.sh
auditor=target/jankurai/jankurai-1.7.1-x86_64-unknown-linux-gnu/jankurai
if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
  # synchronize events also have a before SHA, which can be unreachable after
  # an amended push. Proof routing must cover the PR from its frozen base.
  JANKURAI_COMPARISON_BASE="$(python3 - "$GITHUB_EVENT_PATH" <<'PY'
import json
import re
import sys
with open(sys.argv[1]) as stream:
    base = json.load(stream)["pull_request"]["base"]["sha"]
if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}", base):
    raise ValueError("invalid pull request comparison base")
print(base)
PY
  )"
fi
if [[ -n "${JANKURAI_COMPARISON_BASE:-}" ]]; then
  comparison_base="$JANKURAI_COMPARISON_BASE"
  if [[ "$comparison_base" =~ ^0{40}$ ]]; then
    comparison_base="$(git merge-base HEAD origin/main)" || {
      echo "Cannot determine the base of a new push" >&2
      exit 1
    }
  fi
  if ! git rev-parse --verify "${comparison_base}^{commit}" >/dev/null 2>&1; then
    # A force-with-lease push reports the previous tip in github.event.before.
    # That object is not in the new history. Compare against main instead.
    requested="$comparison_base"
    if git fetch --no-tags origin main && comparison_base="$(git merge-base HEAD FETCH_HEAD)"; then
      echo "Jankurai comparison base ${requested} is not in this clone; using merge-base ${comparison_base}"
    else
      echo "Explicit Jankurai comparison base is unavailable: ${requested}" >&2
      exit 1
    fi
  fi
  if [[ "$(git rev-parse "${comparison_base}^{commit}")" == "$(git rev-parse HEAD)" ]]; then
    echo "Explicit Jankurai comparison base equals HEAD" >&2
    exit 1
  fi
else
  comparison_base=origin/main
  if ! git rev-parse --verify "${comparison_base}^{commit}" >/dev/null 2>&1; then
    comparison_base=main
  fi
  if [[ "$(git rev-parse "${comparison_base}^{commit}")" == "$(git rev-parse HEAD)" ]]; then
    comparison_base=HEAD^
  fi
  git rev-parse --verify "${comparison_base}^{commit}" >/dev/null
fi
"$auditor" proof . --changed-from "$comparison_base" \
  --out target/jankurai/proof-plan.json --md target/jankurai/proof-plan.md
"$auditor" copy-code . --json target/jankurai/copy-code.json --md target/jankurai/copy-code.md
"$auditor" rust witness build . --out target/jankurai/rust/witness-graph.json
ci_require_artifact target/jankurai/proof-plan.json
ci_require_artifact target/jankurai/copy-code.json
ci_require_artifact target/jankurai/rust/witness-graph.json
