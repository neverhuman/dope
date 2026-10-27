"""Export MLC-v3 preparation goldens for the Rust port (engine/dope-core/src/prepare).

Runs the UNMODIFIED Python reference functions
(`scaling.FitOnlyTableTransform`, `descriptors.extract_report_descriptors`,
`descriptors.compute_table_descriptors`, `building_block_landmarks.*`) on
deterministic synthetic tables and native-shaped reports, and writes exact
float64/float32 tensors to SafeTensors plus the report dicts to JSON:

    engine/dope-core/tests/fixtures/prepare-v1/manifest.json
    engine/dope-core/tests/fixtures/prepare-v1/<case>.safetensors
    engine/dope-core/tests/fixtures/prepare-v1/<case>.json   (reports + raw cells)

Report-shape overrides (`rows=`/`features=`) are expressed by injecting the
explicit shape into a copy of the report before calling the unmodified
function; the Rust `ShapeOverride` must reproduce exactly that.

Usage (repo root):  python3 -B research/levels/tools/export_prepare_goldens.py [--out DIR]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.levels import building_block_landmarks as bbl  # noqa: E402
from research.levels import descriptors as desc  # noqa: E402
from research.levels.scaling import FitOnlyTableTransform  # noqa: E402

DEFAULT_OUT = ROOT / "engine/dope-core/tests/fixtures/prepare-v1"
A_LEVELS = ("small", "medium", "high", "ultra")
B_LEVELS = ("high", "ultra")
MAX_COLS = 64
COL_FEATS = 17
STATES = ("complete", "uninformative", "failed", "unsupported", "over_budget")


def _r6(v: float) -> float:
    return float(round(float(v), 6))


def _numeric_table(rng: np.random.Generator, n: int, p: int, *, missing: bool) -> np.ndarray:
    cols = []
    for j in range(p):
        kind = j % 8
        if kind == 0:
            c = rng.standard_normal(n)
        elif kind == 1:
            c = np.exp(rng.standard_normal(n) * 1.5)  # heavy right skew -> signed_log1p
        elif kind == 2:
            c = rng.integers(0, 2, n).astype(np.float64)  # binary
        elif kind == 3:
            c = np.full(n, 3.25)  # constant
        elif kind == 4:
            c = rng.integers(0, 4, n).astype(np.float64)  # heavy ties
        elif kind == 5:
            c = rng.standard_t(2, n)  # heavy tails
        elif kind == 6:
            c = np.round(rng.uniform(-5, 5, n), 1)  # ties + uniform
        else:
            c = rng.standard_normal(n) * 1e-3 + 100.0
        cols.append(np.round(c, 4))
    X = np.column_stack(cols).astype(np.float64)
    if missing:
        mask = rng.random(X.shape) < 0.07
        mask[:, 3 % p] = False
        X[mask] = np.nan
        if p > 1:
            X[: max(1, n // 10), 1] = np.inf if p > 1 else X[0, 0]
    return X


def _target(rng: np.random.Generator, X: np.ndarray, task: str, n_classes: int = 3) -> np.ndarray:
    Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xf = (Xf - Xf.mean(0)) / np.maximum(Xf.std(0), 1e-9)
    w = np.linspace(1.0, -0.5, X.shape[1])
    s = Xf @ w + 0.5 * np.sin(Xf[:, 0] * 2.0) + rng.standard_normal(X.shape[0]) * 0.5
    if task == "regression":
        return np.round(s, 4)
    if task == "classification":
        return (s > np.median(s)).astype(np.float64)
    qs = np.quantile(s, np.linspace(0, 1, n_classes + 1)[1:-1])
    return np.searchsorted(qs, s).astype(np.float64)


def _report(rng: np.random.Generator, variant: str, n: int, p: int, task: str, classes: int) -> Dict[str, Any]:
    if variant in ("empty",):
        return {}
    if variant == "empty_shaped":
        return {"rows": n, "features": p, "task": task, "classes": classes}
    if variant == "full":
        ids = list(range(1, 24))
    else:  # partial / partial_override: macro tier + a couple of others
        ids = [i + 1 for i in desc.MACRO_INDICES] + [8, 16]
    base = [_r6(v) for v in rng.uniform(-0.05, 0.6, 23)]
    base[5] = base[2]  # tie for midranks
    base[17] = base[2]
    cells = []
    for i in ids:
        r = rng.random()
        state = "complete" if r < 0.7 else ("uninformative" if r < 0.8 else STATES[int(rng.integers(2, 5))])
        cell = {
            "id": i,
            "state": state,
            "gain": base[i - 1] if state in ("complete", "uninformative") or rng.random() < 0.5 else None,
            "se": _r6(rng.uniform(0.0, 0.1)) if rng.random() < 0.9 else None,
            "stable": bool(rng.random() < 0.6),
            "lambda_indices": [int(v) for v in rng.integers(0, 9, int(rng.integers(0, 4)))],
        }
        cells.append(cell)
    rep: Dict[str, Any] = {
        "panel": "landmarks-native-v1",
        "cells": cells,
        "target_stats": [_r6(rng.normal()), _r6(abs(rng.normal())), None],
    }
    if variant != "partial_override":
        rep.update({"rows": n, "features": p, "task": task, "classes": classes})
    else:
        rep.update({"task": task, "classes": classes})
    spec = []
    for slot in range(p + 2):  # includes out-of-range slots
        if rng.random() < 0.2:
            continue
        vals = [(_r6(v) if rng.random() < 0.85 else None) for v in rng.uniform(-0.2, 0.9, 9)]
        spec.append({"slot": slot, "values": vals})
    if variant == "partial":
        spec = spec[: max(1, len(spec) // 2)]
    rep["spectrum"] = spec
    return rep


def _f64(a: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _f32(a: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def _export_case(out: Path, name: str, spec: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    task, n, p = spec["task"], spec["n"], spec["p"]
    tensors: Dict[str, np.ndarray] = {}
    doc: Dict[str, Any] = {"name": name, "task": task, "n": n, "p": p, "input": spec.get("input", "numeric")}
    if doc["input"] == "cells":
        X_num = _numeric_table(rng, n, p, missing=False)
        y_num = _target(rng, X_num, task)
        cats = ["red", "green", "blue", "Cyan", "teal", "zz"]
        rows: List[List[str]] = []
        for i in range(n):
            row = []
            for j in range(p):
                if j % 3 == 1:
                    k = int(abs(X_num[i, 0]) * 3 + (i % 7 == 0) * 5) % len(cats)
                    tok = cats[k] if rng.random() > 0.05 else ("NA" if rng.random() < 0.5 else " ")
                    row.append(f" {tok}" if i % 11 == 0 else tok)
                elif j % 3 == 2 and rng.random() < 0.05:
                    row.append("?")
                else:
                    row.append(repr(float(X_num[i, j])))
            rows.append(row)
        if task == "classification":
            y_cells = ["yes" if v > 0.5 else "no" for v in y_num]
        else:
            y_cells = [repr(float(v)) for v in y_num]
        doc["x_cells"], doc["y_cells"] = rows, y_cells
        scaler = FitOnlyTableTransform.fit(rows, y_cells, task=task)
        Xs, ys = scaler.transform(rows, y_cells)
    else:
        X = _numeric_table(rng, n, p, missing=spec.get("missing", False))
        if spec.get("y") == "many_classes":
            y = rng.integers(0, 40, n).astype(np.float64)
            y[: n // 3] = 7.0
        else:
            y = _target(rng, X, task, spec.get("classes", 3))
        tensors["x_raw"], tensors["y_raw"] = _f64(X), _f64(y)
        scaler = FitOnlyTableTransform.fit(X, y, task=task)
        Xs, ys = scaler.transform(X, y)
    Xs = _f64(Xs).reshape(n, p)
    ys = _f64(ys)
    tensors["x_scaled"], tensors["y_scaled"] = Xs, ys
    ts = scaler.target_scaler
    classes = 0 if task == "regression" else (2 if task == "classification" else len(ts.class_labels))
    doc["classes"] = classes
    doc["kinds"] = list(scaler.kinds)
    doc["target_scaler"] = {"mean": ts.mean, "std": ts.std, "class_labels": list(ts.class_labels),
                            "positive_label": ts.positive_label}
    if task == "regression" or ys.size == 0:
        doc["observed_target_stats"] = None
    else:
        labels = np.rint(ys).astype(np.int64)
        cls_list = [0, 1] if task in ("classification", "binary") else sorted(set(int(v) for v in labels))
        if len(cls_list) < 2:
            doc["observed_target_stats"] = None
        else:
            slot_map = {c: i for i, c in enumerate(cls_list)}
            counts = np.zeros(len(cls_list), dtype=np.float64)
            for lab in labels:
                if int(lab) in slot_map:
                    counts[slot_map[int(lab)]] += 1.0 / ys.size
            pos = counts[counts > 0.0]
            ent = -float(np.sum(pos * np.log(pos))) / float(np.log(counts.size))
            doc["observed_target_stats"] = [ent, float(np.min(counts))]

    for lvl in A_LEVELS:
        td = desc.compute_table_descriptors(Xs, ys, level=lvl)
        tensors[f"table_vec24.{lvl}"] = _f32(td["table_vec24"])
        tensors[f"column_desc.{lvl}"] = _f32(td["column_desc"])
        tensors[f"emp_w.{lvl}"] = _f32(td["empirical_driver_weights"])

    variants = ["full", "partial", "partial_override", "empty", "empty_shaped"]
    reports = {}
    for v in variants:
        rep = _report(rng, v, n, p, task, classes)
        rep = json.loads(json.dumps(rep))  # exactly what Rust will parse
        reports[v] = rep
        for lvl in A_LEVELS:
            calls = {"": rep}
            if v in ("partial_override", "empty"):
                ov = copy.deepcopy(rep)
                ov["rows"], ov["features"] = n, p
                calls["ovr"] = ov
            for tag, r in calls.items():
                rd = desc.extract_report_descriptors(r, level=lvl)
                key = f"{v}{'+' + tag if tag else ''}.{lvl}"
                tensors[f"vector116.{key}"] = _f32(rd["vector116"])
                tensors[f"column_spectrum.{key}"] = _f32(rd["column_spectrum"])
                tensors[f"col_prior.{key}"] = _f32(rd["column_prior_importance"])
        for lvl in B_LEVELS:
            b = bbl.build_v3_conditioning_bundle(rep, Xs, ys, level=lvl)
            key = f"{v}.{lvl}"
            tensors[f"vec256.{key}"] = _f32(b["vec256"])
            tensors[f"vec140.{key}"] = _f32(b["vec140"])
            tensors[f"inter48.{key}"] = _f32(b["inter_landmark_vec48"])
            tensors[f"bb68.{key}"] = _f32(b["building_block_vec68"])
            cl24 = _f32(b["col_landmarks24"])
            tensors[f"col_landmarks24.{key}"] = cl24
            pad = np.zeros((MAX_COLS, COL_FEATS), dtype=np.float32)
            take = min(cl24.shape[0], MAX_COLS)
            pad[:take] = cl24[:take, :COL_FEATS]
            tensors[f"col_landmarks17x64.{key}"] = pad
            tensors[f"v3_drivers.{key}"] = _f32(b["v3_driver_weights"])
            for k, arr in b["bb_desc"].items():
                tensors[f"bb.{k}.{key}"] = _f32(arr)
    doc["reports"] = reports
    doc["variants"] = variants
    save_file(tensors, str(out / f"{name}.safetensors"))
    (out / f"{name}.json").write_text(json.dumps(doc, sort_keys=True, separators=(",", ":")))
    return {"name": name, "task": task, "n": n, "p": p, "input": doc["input"], "seed": seed,
            "keys": sorted(tensors)}


CASES = {
    "reg_small": dict(task="regression", n=40, p=5),
    "reg_missing": dict(task="regression", n=300, p=12, missing=True),
    "bin_medium": dict(task="classification", n=200, p=8),
    "mc_medium": dict(task="multiclass", n=240, p=7, classes=4),
    "mc_many": dict(task="multiclass", n=150, p=6, y="many_classes"),
    "reg_wide": dict(task="regression", n=120, p=70),
    "reg_p1": dict(task="regression", n=30, p=1),
    "bin_tiny": dict(task="classification", n=5, p=3),
    "reg_n3": dict(task="regression", n=3, p=4),
    "cells_reg": dict(task="regression", n=90, p=6, input="cells"),
    "cells_bin": dict(task="classification", n=80, p=5, input="cells"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "dope.prepare-goldens/v1", "numpy": np.__version__,
                "a_levels": list(A_LEVELS), "b_levels": list(B_LEVELS), "cases": []}
    for i, (name, spec) in enumerate(CASES.items()):
        manifest["cases"].append(_export_case(args.out, name, spec, seed=20261027 + i))
    for c in manifest["cases"]:
        c.pop("keys")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(manifest['cases'])} cases to {args.out}")


if __name__ == "__main__":
    main()
