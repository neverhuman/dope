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
    (".github/workflows/ci.yml", [budget["cpu_pr_minutes"], budget["security_minutes"]]),
    (".github/workflows/jankurai.yml", [budget["audit_minutes"]]),
    (".github/workflows/gpu.yml", [budget["scheduled_gpu_minutes"]]),
):
    actual = [int(value) for value in re.findall(r"^\s+timeout-minutes:\s*(\d+)$", Path(workflow).read_text(), re.M)]
    assert actual == expected, (workflow, actual, expected)
assert "Duration::from_secs(600)" in Path("rust/compiler/neural_candidates.rs").read_text()
assert "GPU_MEMORY_LIMIT_BYTES: u128 = 16 * 1024 * 1024 * 1024" in Path("rust/neural_train.rs").read_text()
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
