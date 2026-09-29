from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .eval import rbcs, score_synthetic_dataset
from .io import load_real_split, load_train
from .kernel import fit_kernel_from_arrays, sample_kernel
from .utils import infer_task_from_target, normalize_task


def gp_search(
    dataset_dir: str | Path,
    budget_secs: float,
    task: str | None = None,
    seed: int = 1729,
) -> dict[str, Any]:
    X, y, inferred = load_train(dataset_dir, None)
    task = normalize_task(task) if task is not None else inferred or infer_task_from_target(y)
    X_train, y_train, X_test, y_test, _ = load_real_split(dataset_dir, task)

    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + max(0.0, float(budget_secs))
    configs = _initial_configs(task)
    best: tuple[float, dict[str, Any]] | None = None
    seen: set[tuple[str, str, int | None]] = set()

    while configs and (time.monotonic() <= deadline or best is None):
        config = configs.pop(0)
        key = (config["dependence_kind"], config["target_kind"], config["empq_k"])
        if key in seen:
            continue
        seen.add(key)
        kernel = fit_kernel_from_arrays(
            X_train,
            y_train,
            task,
            seed=seed,
            dependence_kind=config["dependence_kind"],
            target_kind=config["target_kind"],
            empq_k=config["empq_k"],
        )
        synth = sample_kernel(kernel, len(X_train), seed=seed)
        report = score_synthetic_dataset(X_train, y_train, X_test, y_test, synth[:, :-1], synth[:, -1], task, seed)
        fitness = rbcs(report["fidelity_real"], kernel["kernel_bytes"], len(X_train), X_train.shape[1] + 1)
        kernel["gp_search"] = {
            "fitness": fitness,
            "fitness_components": report["components"],
            "config": config,
            "budget_secs": float(budget_secs),
            "selection": "typed_tournament_mutation_v1",
        }
        if best is None or fitness > best[0]:
            best = (fitness, kernel)

        if time.monotonic() <= deadline:
            parent = config if best is None else best[1]["gp_search"]["config"]
            configs.extend(_mutations(parent, task, rng))

    if best is None:
        raise RuntimeError("gp-search did not evaluate any candidates")
    return best[1]


def _initial_configs(task: str) -> list[dict[str, Any]]:
    target_options = ["linear_sparse", "tree_piecewise"] if task == "regression" else ["lift_logit", "tree_piecewise"]
    configs = []
    for dep in ["auto", "ind", "gauss_copula"]:
        for target in target_options:
            configs.append({"dependence_kind": dep, "target_kind": target, "empq_k": None})
    return configs


def _mutations(parent: dict[str, Any], task: str, rng: np.random.Generator) -> list[dict[str, Any]]:
    dep_options = ["auto", "ind", "gauss_copula"]
    target_options = ["linear_sparse", "tree_piecewise"] if task == "regression" else ["lift_logit", "tree_piecewise"]
    k_options: list[int | None] = [None, 9, 17, 33]
    children = []
    for _ in range(2):
        child = dict(parent)
        choice = int(rng.integers(0, 3))
        if choice == 0:
            child["dependence_kind"] = str(rng.choice(dep_options))
        elif choice == 1:
            child["target_kind"] = str(rng.choice(target_options))
        else:
            child["empq_k"] = k_options[int(rng.integers(0, len(k_options)))]
        children.append(child)
    return children
