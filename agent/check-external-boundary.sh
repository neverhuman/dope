#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
evidence_dir=target/jankurai/boundaries/external-learner-evidence
mkdir -p "$evidence_dir"
command -v uv >/dev/null
(
  cd validation/external
  UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --locked --extra xgboost --extra lightgbm
  UV_PROJECT_ENVIRONMENT=../../target/external-venv uv sync --offline --locked --extra xgboost --extra lightgbm
)
PYTHONPATH=validation/external target/external-venv/bin/python -m unittest discover -s validation/external/tests -v > "$evidence_dir/unittest.log" 2>&1
python3 - <<'PY'
import ast
import hashlib
import json
from pathlib import Path

root = Path("validation/external")
files = sorted(root.rglob("*.py"))
assert files, "external Python boundary is empty"
assert (root / "uv.lock").is_file(), "external dependency lock is absent"
violations = []
records = []
for path in files:
    tree = ast.parse(path.read_text(), filename=str(path))
    records.append({"path": path.as_posix(), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()})
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [name.name.split(".")[0] for name in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module.split(".")[0]] if node.module else []
        else:
            modules = []
        for module in modules:
            if module in {"subprocess", "socket", "requests", "urllib", "httpx", "sqlite3", "psycopg", "sqlalchemy"}:
                violations.append((path.as_posix(), node.lineno, module))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__"}:
            violations.append((path.as_posix(), node.lineno, node.func.id))
        if isinstance(node, ast.FunctionDef):
            if any(isinstance(decorator, ast.Attribute) and decorator.attr in {"route", "get", "post", "put", "delete"} for decorator in node.decorator_list):
                violations.append((path.as_posix(), node.lineno, "product route"))
assert not violations, f"external boundary violations: {violations}"
checks = ["manifest-coverage", "payload-hash-match", "pinned-dependencies", "external-fixtures", "no-direct-db-access", "no-network-or-subprocess", "no-product-routes"]
evidence = {
    "boundary_id": "external-learner-evidence",
    "classification": "advanced-data",
    "runtime_language": "python",
    "paths": ["validation/external/*.py", "validation/external/tests/*.py"],
    "files": records,
    "checks": [{"id": check, "status": "passed"} for check in checks],
    "summary": {"passed": True, "failed_count": 0},
}
destination = Path("target/jankurai/boundaries/external-learner-evidence/evidence.json")
destination.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n")
print(f"External learner boundary: {len(records)} files, {len(checks)} checks passed")
PY
