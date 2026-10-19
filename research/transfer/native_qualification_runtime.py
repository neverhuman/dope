"""Offline adapter comparison and honest implementation origins for W01.

The runner retains CLI admission and process supervision. This helper performs
its existing SDK/adapter comparison only when invoked by that admitted worker.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
from typing import Any

from research.transfer.native_qualification import MODEL_FILES, QualificationError


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()



def _implementation_origins() -> dict:
    """Pin both the actual entry and this extracted helper, without aliasing them."""
    from research.transfer.adapters.auditors.conventional import file_sha256

    module = Path(__file__)
    directory = module.parent
    return {
        label: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for label, path in (
            ("tabicl_adapter", directory / "adapters/auditors/tabicl.py"),
            ("kumo_adapter", directory / "adapters/auditors/kumo_tabular.py"),
            ("native_qualification", directory / "native_qualification.py"),
            ("run_native_qualification", directory / "run_native_qualification.py"),
            ("native_qualification_runtime", module),
        )
    }


def _adapter_parity(case: dict, root: Path, context: dict, query: dict,
                    recipe: dict, sdk_values: Any, sdk_classes: list[str]) -> dict:
    """Run the real offline DOPE adapter on the exact native SDK inputs."""
    import numpy as np
    import psutil
    from research.transfer.adapters.base import (
        AdapterAdmission, FeatureBatch, ModelIdentity, ResourceLimits, ResourceUsage, TrainingBatch,
    )
    from research.transfer.adapters.auditors.conventional import (
        OfflineArtifacts, OfflineRuntime, file_sha256,
    )
    if case["model"] == "tabicl_v2":
        from research.transfer.adapters.auditors.tabicl import TabICLAdapter, TabICLRecipe
        adapter_file = Path(inspect.getfile(TabICLAdapter))
        adapter_recipe = TabICLRecipe(
            case["task"], estimators=recipe["n_estimators"], estimator_batch_size=1,
            class_order=tuple(case["class_order"]), device="cpu",
        )
        adapter_cls = TabICLAdapter
        code_license_name, weights_license_name = "tabicl_code_license.txt", "tabicl_weights_license.txt"
        license_id = "bsd-3-clause"
    else:
        from research.transfer.adapters.auditors.kumo_tabular import KumoRecipe, KumoTabularAdapter
        adapter_file = Path(inspect.getfile(KumoTabularAdapter))
        adapter_recipe = KumoRecipe(
            case["task"], size="large", estimators=recipe["n_estimators"],
            estimator_batch_size=1,
            output="quantile_grid_mean" if case["task"] == "regression" else "probability",
            class_order=tuple(case["class_order"]), device="cpu",
        )
        adapter_cls = KumoTabularAdapter
        code_license_name, weights_license_name = "kumo_code_license.txt", "kumo_weights_license.txt"
        license_id = "openmdw-1.1"
    closure_path = root / "w01-cert-venv-closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    dependencies, dependency_paths = [], []
    for item in closure["distributions"]:
        dist = metadata.distribution(item["name"])
        record = next((entry for entry in (dist.files or ()) if str(entry).endswith("RECORD")), None)
        if (dist.version != item["version"] or record is None or
                _hash(dist.locate_file(record).read_bytes()) != item["record_sha256"]):
            raise QualificationError("adapter_dependency_closure_mismatch")
        dependencies.append((item["name"], item["version"], item["record_sha256"]))
        dependency_paths.append(Path(dist.locate_file(record)))
    licenses_dir = root / "licenses"
    code_license = licenses_dir / code_license_name
    weights_license = licenses_dir / weights_license_name
    weight_path = root / case["model"] / case["checkpoint"]
    expected_weight_sha256 = MODEL_FILES[case["model"]]["files"][case["checkpoint"]][1]
    if (file_sha256(code_license) != case["code_license_sha256"] or
            file_sha256(weights_license) != case["weights_license_sha256"] or
            file_sha256(weight_path) != expected_weight_sha256):
        raise QualificationError("adapter_artifact_pin_mismatch")
    identity = ModelIdentity(
        name=case["model"], mode="frozen_context", code_revision=case["code_revision"],
        code_sha256=file_sha256(adapter_file), weights_sha256=file_sha256(weight_path),
        dependencies=tuple(dependencies),
        licenses=(("code", license_id, file_sha256(code_license)),
                  ("weights", license_id, file_sha256(weights_license))),
        preprocessing_sha256=adapter_recipe.preprocessing_sha256,
        recipe_sha256=adapter_recipe.sha256,
        output_aggregation_sha256=adapter_recipe.output_sha256,
        environment_sha256=file_sha256(closure_path), precision="float32",
        context_cap=len(context["rows"]), context_policy="all", query_policy="batch",
        query_batch_size=len(query["rows"]), order_sensitive=True,
        determinism="seeded_best_effort",
        resources=ResourceLimits(float(case["resource_caps"]["wall_seconds"]),
                                 case["resource_caps"]["memory_bytes"],
                                 case["resource_caps"]["gpu_bytes"]),
    )
    # Admission is a local identity binding for this comparison, not a registry
    # decision. Registry promotion remains a separate reviewed gate.
    admission = AdapterAdmission(identity.sha256, _hash(b"W01 parity eligibility"),
                                 _hash(b"W01 parity reference"), "learner")
    artifacts = OfflineArtifacts(adapter_file, tuple(dependency_paths),
                                 (code_license, weights_license), weight_path)
    process = psutil.Process()
    runtime = OfflineRuntime(
        artifacts,
        lambda event: ResourceUsage(len(context["rows"]), process.memory_info().rss, 0),
        factory=None,
    )
    columns = context["feature_columns"]
    query_columns = query["feature_columns"]
    train_rows = tuple(tuple(float(row[column]) for column in columns) for row in context["rows"])
    train_targets = tuple(row[context["target_column"]] for row in context["rows"])
    query_rows = tuple(tuple(float(row[column]) for column in query_columns) for row in query["rows"])
    schema_sha = _hash(json.dumps({"feature_columns": columns}, sort_keys=True,
                                  separators=(",", ":")).encode())
    context_ids = tuple(_hash(f"{case['model']}:{case['task']}:context:{i}".encode())
                        for i in range(len(train_rows)))
    query_ids = tuple(_hash(f"{case['model']}:{case['task']}:query:{i}".encode())
                      for i in range(len(query_rows)))
    training = TrainingBatch(FeatureBatch(context_ids, train_rows, schema_sha),
                             train_targets, "development_train")
    query_batch = FeatureBatch(query_ids, query_rows, schema_sha)
    adapter = adapter_cls(identity, admission, adapter_recipe, runtime)
    seed = int(recipe["seed"])
    adapter.prepare(training, seed)
    prediction = adapter.predict(query_batch, seed)
    adapter_values = np.asarray(prediction.values, dtype=np.float64)
    native_values = np.asarray(sdk_values, dtype=np.float64)
    delta = float(np.max(np.abs(adapter_values - native_values)))
    tolerance = 1e-6
    adapter_classes = list(prediction.class_order)
    sdk_class_order = list(sdk_classes)
    modules = _implementation_origins()
    if case["model"] == "tabicl_v2":
        import tabicl
        modules["vendor_tabicl"] = {"path": str(Path(inspect.getfile(tabicl)).resolve()),
                                    "sha256": file_sha256(inspect.getfile(tabicl))}
    else:
        import sdm
        modules["vendor_sdm"] = {"path": str(Path(inspect.getfile(sdm)).resolve()),
                                "sha256": file_sha256(inspect.getfile(sdm))}
    modules["vendor_torch"] = {"path": str(Path(inspect.getfile(__import__("torch"))).resolve()),
                              "sha256": file_sha256(inspect.getfile(__import__("torch")))}
    return {
        "adapter_output_parity": "established" if (
            adapter_values.shape == native_values.shape and adapter_classes == sdk_class_order and
            delta <= tolerance
        ) else "mismatch",
        "sdk_output_shape": list(native_values.shape),
        "adapter_output_shape": list(adapter_values.shape),
        "sdk_class_order": sdk_class_order,
        "adapter_class_order": adapter_classes,
        "sdk_output_sha256": _hash(np.asarray(native_values, dtype="<f8").tobytes()),
        "adapter_output_sha256": _hash(np.asarray(adapter_values, dtype="<f8").tobytes()),
        "sdk_output_values": native_values.tolist(),
        "adapter_output_values": adapter_values.tolist(),
        "adapter_vs_sdk_max_abs_delta": delta,
        "tolerance": tolerance,
        "recipe_sha256": case["recipe_sha256"],
        "adapter_recipe_sha256": adapter_recipe.sha256,
        "preprocessing_sha256": adapter_recipe.preprocessing_sha256,
        "output_aggregation_sha256": adapter_recipe.output_sha256,
        "seed": seed,
        "precision": adapter_recipe.precision,
        "class_order": list(case["class_order"]),
        "module_origins": modules,
    }
