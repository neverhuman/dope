#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
import re
import tomllib
from pathlib import Path

budget = tomllib.loads(Path("agent/cost-budget.toml").read_text())
assert budget["version"] == 1
for workflow, expected in (
    # The clean paper rebuild is a second CPU-only PR job with the same cap.
    (".github/workflows/ci.yml", [budget["cpu_pr_minutes"], budget["cpu_pr_minutes"], budget["security_minutes"]]),
    (".github/workflows/jankurai.yml", [budget["audit_minutes"]]),
    (".github/workflows/gpu.yml", [budget["scheduled_gpu_minutes"]]),
):
    actual = [int(value) for value in re.findall(r"^\s+timeout-minutes:\s*(\d+)$", Path(workflow).read_text(), re.M)]
    assert actual == expected, (workflow, actual, expected)
def source_tree(path):
    root = Path(path)
    parts = [root, *sorted(root.with_suffix("").rglob("*.rs"))]
    return "\n".join(part.read_text() for part in parts)

assert "Duration::from_secs(600)" in source_tree("rust/compiler/neural_candidates.rs")
assert "GPU_MEMORY_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024" in source_tree("rust/neural_train.rs")
assert '"${SEALED_CORPUS_DIR:-}"' in Path("ops/ci/gpu.sh").read_text()
assert '"${SEALED_ARTIFACT:-}"' in Path("ops/ci/gpu.sh").read_text()
fitness = Path("rust/fitness_metrics.rs").read_text()
assert "const MAX_ROWS: usize = 128" in fitness
assert "const MAX_DIMENSIONS: usize = 32" in fitness
assert budget["single_neural_fit_seconds"] == 600
assert budget["gpu_memory_bytes"] == 16 * 1024**3
assert budget["fitness_sample_rows"] == 128
assert budget["fitness_sample_dimensions"] == 32
print("Cost, time, admission, and kill-switch budget proof passed")
PY
