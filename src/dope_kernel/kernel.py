from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .copula import fit_dependence, sample_uniforms
from .dsl import DSLNode, build_node, canonicalize
from .io import apply_order, canonical_feature_order, load_train
from .marginals import fit_marginals, sample_marginals
from .target import fit_target, sample_target
from .utils import (
    bits_for_jsonish,
    canonical_bytes,
    canonical_dumps,
    clip01,
    normalize_task,
    stable_dataset_seed,
    stable_hash_bytes,
)


def _program_for_kernel(kernel: dict[str, Any]) -> str:
    shell = kernel["shell"]
    marginal_nodes = [
        build_node(m["kind"], {"i": i, **_marginal_program_attrs(m)})
        for i, m in enumerate(kernel["marginals"])
    ]
    dep = kernel["dependence"]
    target = kernel["target"]
    residual = kernel["residual"]
    root = DSLNode(
        "dk",
        {
            "v": 1,
            "task": shell["task"],
            "n": shell["n"],
            "p": shell["p"],
            "seed": shell["seed"],
        },
        (
            build_node("marginals", {"count": shell["p"]}, marginal_nodes),
            build_node("dependence", _dependence_program_attrs(dep)),
            build_node("target", {"kind": target["kind"]}),
            build_node("residual", {"kind": residual["kind"]}),
        ),
    )
    return canonicalize(root.to_sexpr())


def _marginal_program_attrs(marginal: dict[str, Any]) -> dict[str, Any]:
    params = marginal.get("params", {})
    attrs: dict[str, Any] = {}
    if marginal["kind"] == "empq":
        attrs["k"] = params.get("k", len(params.get("values", [])))
    if marginal["kind"] in {"spike_grid", "ordinal"}:
        attrs["k"] = len(params.get("values", []))
    return attrs


def _dependence_program_attrs(dependence: dict[str, Any]) -> dict[str, Any]:
    attrs = {"kind": dependence["kind"]}
    if dependence["kind"] == "gauss_copula":
        attrs["rank"] = int(dependence["params"].get("rank", 0))
    return attrs


def _estimate_description_bits(kernel: dict[str, Any]) -> int:
    payload = {
        "program": kernel["program"],
        "marginals": kernel["marginals"],
        "dependence": kernel["dependence"],
        "target": kernel["target"],
        "residual": kernel["residual"],
        "quantization": kernel["quantization"],
    }
    return int(bits_for_jsonish(payload))


def _finalize_kernel(kernel: dict[str, Any]) -> dict[str, Any]:
    kernel = deepcopy(kernel)
    kernel["program"] = _program_for_kernel(kernel)
    kernel["description_bits"] = _estimate_description_bits(kernel)
    cells = max(1, int(kernel["shell"]["n"]) * (int(kernel["shell"]["p"]) + 1))
    kernel["description_bits_per_cell"] = kernel["description_bits"] / cells
    kernel["content_hash"] = stable_hash_bytes(canonical_bytes(_without_size_fields(kernel)))
    kernel["kernel_bytes"] = 0
    for _ in range(8):
        size = len(canonical_bytes(kernel))
        if size == kernel["kernel_bytes"]:
            break
        kernel["kernel_bytes"] = size
    return kernel


def _without_size_fields(kernel: dict[str, Any]) -> dict[str, Any]:
    clone = deepcopy(kernel)
    for key in ("content_hash", "kernel_bytes"):
        clone.pop(key, None)
    return clone


def fit_kernel_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    seed: int | None = None,
    dependence_kind: str = "auto",
    target_kind: str = "auto",
    empq_k: int | None = None,
) -> dict[str, Any]:
    task = normalize_task(task)
    X = clip01(X)
    y = (clip01(y) >= 0.5).astype(float) if task == "binary" else clip01(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(y) != X.shape[0]:
        raise ValueError("X and y row counts differ")
    order = canonical_feature_order(X)
    Xc = apply_order(X, order)
    full_data = np.column_stack([Xc, y])
    seed = int(stable_dataset_seed(full_data) if seed is None else seed)

    marginals = fit_marginals(Xc, empq_k=empq_k)
    dependence = fit_dependence(Xc, kind=dependence_kind)
    target, residual = fit_target(Xc, y, task, seed=seed, target_kind=target_kind)

    kernel = {
        "format": "dope-kernel",
        "version": 1,
        "program": "",
        "shell": {
            "v": 1,
            "task": task,
            "n": int(X.shape[0]),
            "p": int(X.shape[1]),
            "seed": seed,
            "column_identity": "anonymous_positional_canonical_target_free",
            "target_position": "last",
        },
        "marginals": marginals,
        "dependence": dependence,
        "target": target,
        "residual": residual,
        "quantization": {
            "float_decimals": 6,
            "clip": [0.0, 1.0],
            "parameter_bits": "deterministic_mdl_surrogate",
        },
        "teacher": {
            "used": False,
            "policy": "TabICL/TabPFN may guide fitness externally but hidden states and embeddings are never serialized.",
        },
        "metadata_policy": {
            "column_names_stored": False,
            "source_metadata_stored": False,
            "feature_order_stored": False,
        },
    }
    return _finalize_kernel(kernel)


def fit_kernel_from_dir(
    dataset_dir: str | Path,
    task: str,
    seed: int | None = None,
    dependence_kind: str = "auto",
    target_kind: str = "auto",
    empq_k: int | None = None,
) -> dict[str, Any]:
    task = normalize_task(task)
    X, y, _ = load_train(dataset_dir, task)
    return fit_kernel_from_arrays(
        X,
        y,
        task,
        seed=seed,
        dependence_kind=dependence_kind,
        target_kind=target_kind,
        empq_k=empq_k,
    )


def sample_kernel(kernel: dict[str, Any], rows: int, seed: int | None = None) -> np.ndarray:
    if int(kernel.get("version", 0)) != 1:
        raise ValueError("unsupported kernel version")
    rows = int(rows)
    if rows < 0:
        raise ValueError("rows must be non-negative")
    p = int(kernel["shell"]["p"])
    task = normalize_task(kernel["shell"]["task"])
    rng = np.random.default_rng(int(kernel["shell"]["seed"]) if seed is None else int(seed))
    U = sample_uniforms(kernel["dependence"], rows, p, rng)
    X = sample_marginals(kernel["marginals"], U)
    y = sample_target(kernel["target"], kernel["residual"], X, task, rng)
    return clip01(np.column_stack([X, y]))


def save_kernel(kernel: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finalized = _finalize_kernel(kernel)
    path.write_text(canonical_dumps(finalized) + "\n", encoding="utf-8")


def load_kernel(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    import json

    kernel = json.loads(text)
    if kernel.get("format") != "dope-kernel":
        raise ValueError("not a dope-kernel JSON file")
    if canonicalize(kernel["program"]) != kernel["program"]:
        raise ValueError("kernel program is not canonical")
    return kernel


def kernel_to_canonical_json(kernel: dict[str, Any]) -> str:
    return canonical_dumps(_finalize_kernel(kernel))
