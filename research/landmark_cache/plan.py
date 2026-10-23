"""Freeze a complete catalog schedule without opening source rows."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from research.benchmark.fetch_jope import CATALOG_SHA256, validate_bucket

OPTIONS = dict(panel="landmarks-native-v1", folds=5, ridge_rows=8192,
               small_rows=4096, knn_reference=2048, spectrum_features=64,
               seed=0, work_budget=20_000_000_000)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def make_plan(catalog, *, bucket, engine_sha256, source_commit, options=None):
    raw = Path(catalog).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == CATALOG_SHA256, "catalog_digest_mismatch")
    require(re.fullmatch(r"[0-9a-f]{64}", engine_sha256), "invalid_engine_digest")
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit), "invalid_source_commit")
    validate_bucket(bucket)
    options = dict(OPTIONS if options is None else options)
    require(set(options) == set(OPTIONS) and options["panel"] == OPTIONS["panel"], "invalid_recipe")
    preparation = {name: file_sha(Path(__file__).with_name(name))
                   for name in ("plan.py", "runner.py", "storage.py", "exec_native.py")}
    cells, seen = [], set()
    for row in (json.loads(line) for line in raw.splitlines()):
        identity = row["dataset_hash"]
        require(re.fullmatch(r"[0-9a-f]{16,64}", identity) and identity not in seen,
                "invalid_or_duplicate_source_identity")
        seen.add(identity)
        train = row["files"]["train"]
        require(re.fullmatch(r"[0-9a-f]{64}", train["sha256"]), "invalid_TRAIN_digest")
        require(type(train["bytes"]) is int and train["bytes"] > 0, "invalid_TRAIN_size")
        provenance = row.get("provenance", {})
        family = (f"{provenance['platform']}:{provenance['dataset_id']}"
                  if provenance.get("platform") and provenance.get("dataset_id") is not None
                  else "unresolved:" + row["canonical_sha256"])
        task = {"binary": "classification", "regression": "regression"}.get(row["task"])
        binding = dict(catalog_sha256=CATALOG_SHA256, catalog_entry_sha256=digest(row),
                       dataset_id=identity, source_family_id=family,
                       source_family_resolved=not family.startswith("unresolved:"),
                       source_role="official_train", source_bucket=bucket,
                       source_key="blobs/sha256/" + train["sha256"],
                       train_sha256=train["sha256"], train_bytes=train["bytes"],
                       rows=train["rows"], columns=train["columns"],
                       has_header=train["has_header"], target=row["target"], task=task,
                       options=options, engine_sha256=engine_sha256, source_commit=source_commit,
                       preparation_recipe_sha256=digest(preparation),
                       source_license=provenance.get("license", {"status": "unknown"}))
        reason = None
        if task is None:
            reason = "unsupported_task"
        elif train["rows"] < 20:
            reason = "insufficient_training_rows"
        elif train["bytes"] > 200_000_000:
            reason = "input_byte_cap"
        elif train["columns"] < 2:
            reason = "no_predictor_columns"
        cells.append(dict(key=digest(binding), binding=binding,
                          initial_state="unsupported" if reason else "pending", reason=reason))
    cells.sort(key=lambda cell: cell["binding"]["dataset_id"])
    plan = dict(format="dope-landmark-cache-plan-v1", catalog_sha256=CATALOG_SHA256,
                cells=cells, official_test_access=False, model_training=False,
                external_certification=False, private_measurements=True,
                source_rights_policy="recorded_verbatim_not_upgraded; consumer_admission_separate",
                family_policy="unresolved_families_require_reconciliation_before_meta_training")
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan):
    value = dict(plan)
    checksum = value.pop("plan_sha256")
    require(checksum == digest(value), "plan_digest_mismatch")
    require(plan["format"] == "dope-landmark-cache-plan-v1"
            and plan["official_test_access"] is False and plan["model_training"] is False,
            "unsupported_plan")
    keys = []
    for cell in plan["cells"]:
        require(cell["key"] == digest(cell["binding"]), "cell_binding_mismatch")
        require(cell["binding"]["source_role"] == "official_train", "non_training_source")
        require(cell["binding"]["source_key"] == "blobs/sha256/" + cell["binding"]["train_sha256"],
                "non_training_object")
        keys.append(cell["key"])
    require(len(keys) == len(set(keys)), "duplicate_cell")
    return plan
