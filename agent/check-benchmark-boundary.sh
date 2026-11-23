#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
evidence_dir=target/jankurai/boundaries/offline-benchmark-evidence
rm -rf "$evidence_dir/modules"
mkdir -p "$evidence_dir/modules"
# The suite has no shared state between modules (each passes alone), so every module runs in
# its own process, one per CPU. The log keeps discovery order; any failing module fails the
# check, and a module with no recorded result counts as failed. Exit 5 ("NO TESTS RAN") marks a
# pytest-style module that unittest discovery never collected; it added no tests to the old
# single-process run either. At least one module must run tests.
mapfile -t modules < <(cd research/benchmark/tests && ls test_*.py)
[[ ${#modules[@]} -gt 0 ]]
printf '%s\n' "${modules[@]}" | xargs -P "$(nproc)" -I{} sh -c \
  'cd target && PYTHONPATH=.. python3 -m unittest discover -s ../research/benchmark/tests -p "$1" -v \
     > "../$2/modules/$1.log" 2>&1; echo $? > "../$2/modules/$1.status"' _ {} "$evidence_dir" || true
failed=0
ran=0
for module in "${modules[@]}"; do
  cat "$evidence_dir/modules/$module.log"
  status="$(cat "$evidence_dir/modules/$module.status" 2>/dev/null || true)"
  if [[ "$status" == 0 ]]; then ran=1; elif [[ "$status" != 5 ]]; then failed=1; fi
done > "$evidence_dir/unittest.log"
[[ "$ran" == 1 ]] || failed=1
if [[ "$failed" != 0 ]]; then
  cat "$evidence_dir/unittest.log" >&2
  exit 1
fi
python3 - <<'PY'
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from research.benchmark.admission import assess

root = Path("research/benchmark")
files = sorted(root.rglob("*.py"))
tracked = subprocess.check_output(
    ["git", "ls-files", "--", "research/benchmark"], text=True
).splitlines()
tracked_python = {Path(name) for name in tracked if name.endswith(".py")}
assert files and set(files) == tracked_python, "benchmark Python source coverage differs from Git"
violations = []
records = []
for path in files:
    content = path.read_bytes()
    tree = ast.parse(content, filename=str(path))
    if path == root / "adapter_worker.py":
        records.append({"path": path.as_posix(),
                        "sha256": "sha256:" + hashlib.sha256(content).hexdigest()})
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [item.name.split(".")[0] for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module.split(".")[0]] if node.module else []
        else:
            modules = []
        if any(module in {"sqlite3", "psycopg", "sqlalchemy", "django"} for module in modules):
            violations.append((path.as_posix(), node.lineno, "database ownership"))
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(decorator, ast.Attribute) and
            decorator.attr in {"route", "get", "post", "put", "delete"}
            for decorator in node.decorator_list
        ):
            violations.append((path.as_posix(), node.lineno, "product route"))
for path in Path("src").rglob("*.py"):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            item.name.startswith("research.benchmark") for item in node.names
        ):
            violations.append((path.as_posix(), node.lineno, "product import"))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "research.benchmark"
        ):
            violations.append((path.as_posix(), node.lineno, "product import"))
assert not violations, f"offline benchmark boundary violations: {violations}"
assert len(records) == 1
assert "subprocess" not in (root / "adapter_worker.py").read_text()
admission = assess(Path.cwd(), root)
assert admission["admitted"] is False and all(
    any(f"{name}:{state}" in admission["blockers"]
        for state in ("not_frozen", "missing_or_invalid"))
    for name in ("methods.lock.json", "datasets.lock.json", "budget.lock.json",
                 "method-dataset-matrix.lock.json", "evaluator.lock.json")
), "public-test admission did not fail closed"
checks = ["manifest-coverage", "payload-hash-match", "benchmark-unit-tests",
          "sealed-test-fail-closed", "no-product-imports", "no-direct-db-access",
          "no-product-routes", "no-subprocess"]
evidence = {
    "boundary_id": "offline-benchmark-evidence",
    "classification": "advanced-data",
    "runtime_language": "python",
    "paths": ["research/benchmark/adapter_worker.py"],
    "files": records,
    "checks": [{"id": check, "status": "passed"} for check in checks],
    "summary": {"passed": True, "failed_count": 0},
}
destination = Path("target/jankurai/boundaries/offline-benchmark-evidence/evidence.json")
destination.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
print(f"Offline benchmark boundary: {len(files)} source files checked, {len(checks)} checks passed")
PY
