#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ops/ci/lib.sh
cargo test --locked contract::tier_policy_tests::generated_tier_limits_match_the_embedded_v2_contract
cargo test --locked production::tests::embedded_contract_is_canonical_and_bound
mkdir -p target
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

active = Path("production/kpi-contract.json").read_bytes()
historical = Path("production/kpi-contract-v1.json").read_bytes()
contract = json.loads(active)
assert contract["version"] == 2
assert contract["tier_byte_limits"] == {"l2": 32768, "l3": 10240}
receipt = {
    "format": "dope-contract-drift-proof",
    "version": 1,
    "active_sha256": hashlib.sha256(active.rstrip(b"\n")).hexdigest(),
    "historical_sha256": hashlib.sha256(historical.rstrip(b"\n")).hexdigest(),
    "checks": ["generated-tier-limits", "embedded-v2-digest", "v1-read-only-fixture"],
    "status": "passed",
}
Path("target/contract-drift.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
PY
ci_require_artifact target/contract-drift.json
