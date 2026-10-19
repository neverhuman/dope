"""Offline W01 native SDK and DOPE adapter parity runtime helpers.

Provides deterministic case input verification, official DOPE adapter parity
comparison, and single-threaded CPU native SDK execution for the signed W01
qualification harness. Execution alone never confers registry admission.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
from pathlib import Path
from typing import Any

from research.transfer.native_qualification import (
    MODEL_FILES, QualificationError, sha256_file,
)

RUNTIME_FILE = Path(__file__).resolve()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _case_inputs(case: dict, root: Path) -> tuple[dict, dict, dict]:
    base = root / "inputs" / f"{case['model']}_{case['task']}"
    values = [json.loads(Path(str(base) + suffix).read_text())
              for suffix in ("_context.json", "_query.json", "_recipe.json")]
    expected = (case["context_sha256"], case["query_sha256"], case["recipe_sha256"])
    for suffix, pin in zip(("_context.json", "_query.json", "_recipe.json"), expected):
        if sha256_file(Path(str(base) + suffix)) != pin:
            raise QualificationError("case_input_pin_mismatch")
    return tuple(values)  # type: ignore[return-value]


def _adapter_parity(case: dict, root: Path, context: dict, query: dict,
                    recipe: dict, sdk_values: Any, sdk_classes: list[str],
                    *, runner_file: Path | str | None = None) -> dict:
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
    entry_file = (
        Path(runner_file).resolve()
        if runner_file is not None
        else RUNTIME_FILE.with_name("run_native_qualification.py").resolve()
    )
    modules = {}
    for label, path in (
        ("tabicl_adapter", RUNTIME_FILE.parent / "adapters/auditors/tabicl.py"),
        ("kumo_adapter", RUNTIME_FILE.parent / "adapters/auditors/kumo_tabular.py"),
        ("native_qualification", RUNTIME_FILE.with_name("native_qualification.py")),
        ("native_qualification_runtime", RUNTIME_FILE),
        ("run_native_qualification", entry_file),
    ):
        modules[label] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
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


def _run_native(case: dict, root: Path, repeat: bool = False,
                *, runner_file: Path | str | None = None) -> dict:
    context, query, recipe = _case_inputs(case, root)
    if recipe["allow_auto_download"] is not False or recipe["device"] != "cpu" or recipe["threads"] != 1:
        raise QualificationError("recipe_resource_policy_mismatch")
    import numpy as np
    rows = context["rows"]
    cols = context["feature_columns"]
    x = np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=np.float64)
    y = np.asarray([row[context["target_column"]] for row in rows])
    qrows = query["rows"]
    qcols = query["feature_columns"]
    qx = np.asarray([[float(row[col]) for col in qcols] for row in qrows], dtype=np.float64)
    seed = int(recipe["seed"])
    checkpoint = root / case["model"] / case["checkpoint"]
    if case["model"] == "tabicl_v2":
        from tabicl import TabICLClassifier, TabICLRegressor
        cls = TabICLClassifier if case["task"] == "classification" else TabICLRegressor
        name = Path(case["checkpoint"]).name
        model = cls(n_estimators=recipe["n_estimators"], norm_methods=["none", "power"],
                    feat_shuffle_method="latin", outlier_threshold=4.0,
                    batch_size=1, kv_cache=False, model_path=str(checkpoint),
                    allow_auto_download=False, checkpoint_version=name, device="cpu",
                    use_amp=False, use_fa3=False, offload_mode=False, n_jobs=1,
                    random_state=seed, verbose=False)

        def predict(rows_x):
            if not hasattr(model, "n_samples_in_"):
                model.fit(x, y)
            return model.predict_proba(rows_x) if case["task"] == "classification" else model.predict(rows_x)

        model.fit(x, y)
        values = np.asarray(model.predict_proba(qx) if case["task"] == "classification" else model.predict(qx))
        if case["task"] == "regression":
            values = values.reshape(len(qrows), 1)
        classes = [str(c) for c in getattr(model, "classes_", [])]
        if case["task"] == "classification" and classes != case["class_order"]:
            raise QualificationError("native_class_order_mismatch")
        if repeat:
            replay_model = cls(n_estimators=recipe["n_estimators"], norm_methods=["none", "power"],
                               feat_shuffle_method="latin", outlier_threshold=4.0,
                               batch_size=1, kv_cache=False, model_path=str(checkpoint),
                               allow_auto_download=False, checkpoint_version=name, device="cpu",
                               use_amp=False, use_fa3=False, offload_mode=False, n_jobs=1,
                               random_state=seed, verbose=False)
            replay_model.fit(x, y)
            replay = np.asarray(replay_model.predict_proba(qx) if case["task"] == "classification" else replay_model.predict(qx))
            if case["task"] == "regression":
                replay = replay.reshape(len(qrows), 1)
        else:
            replay = values
        permuted = np.asarray(model.predict_proba(qx[::-1]) if case["task"] == "classification" else model.predict(qx[::-1]))[::-1]
        if case["task"] == "regression":
            permuted = permuted.reshape(len(qrows), 1)
    elif case["model"] == "kumo_tabular":
        import pandas as pd
        import torch
        from sdm import Stype, TableTensor, Task
        from sdm.models import KumoTabular
        torch.set_num_threads(1)
        task = Task(case["task"])
        model = KumoTabular(task=task, size="large", pretrained=False, device="meta")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.models[task].load_state_dict(state, strict=True, assign=True)
        model.eval()
        tx = torch.as_tensor(x, dtype=torch.float64, device="cpu")
        if case["task"] == "classification":
            class_codes = np.asarray([case["class_order"].index(str(label)) for label in y], dtype=np.int64)
            ty = TableTensor.from_pandas(pd.DataFrame({"target": class_codes}), {"target": Stype.categorical}, device="cpu")
        else:
            ty = torch.as_tensor(y.astype(np.float32).reshape(-1, 1), dtype=torch.float32, device="cpu")
        model.fit(x=tx, y=ty, num_estimators=recipe["n_estimators"],
                  estimator_batch_size=1, generator=torch.Generator(device="cpu").manual_seed(seed))
        qtx = torch.as_tensor(qx, dtype=torch.float64, device="cpu")
        native = model.predict(x=qtx)
        values = native.numerical.detach().cpu().numpy()
        cols_native = tuple(native.columns[Stype.numerical])
        if case["task"] == "classification":
            expected = tuple(str(i) for i in range(len(case["class_order"])))
            if set(cols_native) != set(expected):
                raise QualificationError("native_class_order_mismatch")
            values = values[:, [cols_native.index(c) for c in expected]]
            classes = list(case["class_order"])
        else:
            expected = tuple(f"q{i:03d}" for i in range(1, 1000))
            if cols_native != expected:
                raise QualificationError("native_quantile_grid_mismatch")
            if not np.all(np.diff(values, axis=1) >= 0):
                raise QualificationError("native_quantile_order_invalid")
            values = np.asarray([math.fsum(float(v) for v in row) / len(row)
                                 for row in values], dtype=np.float64).reshape(-1, 1)
            classes = []
        if repeat:
            # The signed seed is replayed with a fresh, independently loaded model.
            replay_model = KumoTabular(task=task, size="large", pretrained=False, device="meta")
            replay_model.models[task].load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True, assign=True)
            replay_model.eval()
            replay_model.fit(x=tx, y=ty, num_estimators=recipe["n_estimators"],
                             estimator_batch_size=1, generator=torch.Generator(device="cpu").manual_seed(seed))
            replay_native = replay_model.predict(x=qtx)
            replay_values = replay_native.numerical.detach().cpu().numpy()
            if case["task"] == "classification":
                replay_cols = tuple(replay_native.columns[Stype.numerical])
                replay = replay_values[:, [replay_cols.index(str(i)) for i in range(len(case["class_order"]))]]
            else:
                replay = np.asarray([math.fsum(float(v) for v in row) / len(row)
                                     for row in replay_values], dtype=np.float64).reshape(-1, 1)
        else:
            replay = values
        perm_native = model.predict(x=qtx.flip(0))
        perm_values = perm_native.numerical.detach().cpu().numpy()[::-1]
        if case["task"] == "classification":
            perm_cols = tuple(perm_native.columns[Stype.numerical])
            permuted = perm_values[:, [perm_cols.index(str(i)) for i in range(len(case["class_order"]))]]
        else:
            permuted = np.asarray([math.fsum(float(v) for v in row) / len(row)
                                   for row in perm_values], dtype=np.float64).reshape(-1, 1)
    else:
        raise QualificationError("unsupported_model")
    if values.shape[0] != len(qrows) or not np.isfinite(values).all():
        raise QualificationError("native_output_invalid")
    if case["task"] == "classification":
        if values.shape != (len(qrows), len(case["class_order"])) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise QualificationError("native_output_invalid")
        kind = "probability"
    else:
        if values.shape != (len(qrows), 1):
            raise QualificationError("native_output_invalid")
        kind = "point"
    tol = 1e-6
    replay_ok = replay.shape == values.shape and bool(np.allclose(replay, values, rtol=tol, atol=tol))
    parity = _adapter_parity(case, root, context, query, recipe, values, classes,
                             runner_file=runner_file)
    return {"output_shape": list(values.shape), "output_kind": kind,
            "output_sha256": _hash(np.asarray(values, dtype="<f8").tobytes()),
            "class_order": classes, "context_rows": len(y), "query_rows": len(qrows),
            "query_permutation_max_abs_delta": float(np.max(np.abs(values - permuted))),
            "seeded_replay_max_abs_delta": float(np.max(np.abs(values - replay))),
            "seeded_replay_passed": replay_ok, "tolerance": tol,
            "weights_loaded_strictly": True, "network_disabled": True,
            "q2_parity": parity}
