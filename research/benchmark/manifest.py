"""Rights-aware dataset preparation with sealed test output.

Input is a reviewed JSON selection file; no dataset is downloaded implicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .score import sha256


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_csv(path: Path, header: bool) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        first = next(reader)
        columns = first if header else [f"c{i}" for i in range(len(first))]
        rows = list(reader) if header else [first, *reader]
    if not rows or len(columns) < 2 or any(len(row) != len(columns) for row in rows):
        raise ValueError("empty, ragged, or target-only CSV")
    return columns, rows


def row_digest(row: list[str]) -> str:
    return digest(row)


def split_rows(rows: list[list[str]], target_index: int, task: str, seed: int = 1729) -> dict:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row_digest(row)].append(index)
    strata = defaultdict(list)
    for key, indices in groups.items():
        label = rows[indices[0]][target_index] if task == "binary" else "regression"
        strata[label].append((key, indices))
    splits = {name: [] for name in ("train", "validation", "test")}
    for label, members in sorted(strata.items()):
        random.Random(digest([seed, label])).shuffle(members)
        total = sum(len(indices) for _, indices in members)
        cut_train, cut_valid = 0.6 * total, 0.8 * total
        seen = 0
        for _, indices in members:
            name = "train" if seen < cut_train else "validation" if seen < cut_valid else "test"
            splits[name].extend(indices)
            seen += len(indices)
    for indices in splits.values():
        indices.sort()
    if any(not indices for indices in splits.values()):
        raise ValueError("split has an empty partition")
    return splits


def check_nonoverlap(parts: dict[str, list[list[str]]]) -> None:
    seen = {}
    for name, rows in parts.items():
        for row in rows:
            key = row_digest(row)
            if key in seen and seen[key] != name:
                raise ValueError("overlapping source row across partitions")
            seen[key] = name


def _float(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def fit_projection(train: list[list[str]], columns: list[str], target: str, task: str) -> dict:
    if target not in columns:
        raise ValueError("target absent")
    target_index = columns.index(target)
    features = []
    for index, name in enumerate(columns):
        if index == target_index:
            continue
        values = [row[index] for row in train]
        numeric = [_float(value) for value in values]
        if all(value is not None or raw == "" for value, raw in zip(numeric, values)):
            finite = [value for value in numeric if value is not None]
            if not finite:
                raise ValueError("all-missing numeric feature")
            features.append({"name": name, "kind": "numeric", "min": min(finite), "max": max(finite), "missing": "0.5"})
        else:
            levels = sorted(set(values) - {""})
            features.append({"name": name, "kind": "categorical", "levels": levels, "unknown": "all_zero"})
    labels = sorted({row[target_index] for row in train}) if task == "binary" else None
    if task == "binary" and len(labels) != 2:
        raise ValueError("binary training target needs two classes")
    targets = [_float(row[target_index]) for row in train] if task == "regression" else None
    if task == "regression" and any(value is None for value in targets):
        raise ValueError("non-numeric regression target")
    output_features = sum(1 if feature["kind"] == "numeric" else len(feature["levels"]) for feature in features)
    if not 1 <= output_features <= 2000:
        raise ValueError("projected feature count outside DOPE range")
    return {
        "version": 1, "target": target, "task": task, "input_columns": columns,
        "features": features, "output_features": output_features,
        "target_map": {"labels": labels} if task == "binary" else {"min": min(targets), "max": max(targets)},
        "fit_partition": "train", "unknown_category_rule": "all_zero",
        "numeric_rule": "train_minmax_clip_0_1", "missing_numeric_rule": "0.5",
    }


def project(rows: list[list[str]], mapping: dict) -> list[list[float]]:
    columns = mapping["input_columns"]
    projected = []
    for row in rows:
        result = []
        for feature in mapping["features"]:
            value = row[columns.index(feature["name"])]
            if feature["kind"] == "numeric":
                number = _float(value)
                spread = feature["max"] - feature["min"]
                result.append(0.5 if number is None or spread == 0 else max(0.0, min(1.0, (number - feature["min"]) / spread)))
            else:
                result.extend(float(value == level) for level in feature["levels"])
        target = row[columns.index(mapping["target"])]
        if mapping["task"] == "binary":
            if target not in mapping["target_map"]["labels"]:
                raise ValueError("unseen target label")
            result.append(float(target == mapping["target_map"]["labels"][1]))
        else:
            number = _float(target)
            if number is None:
                raise ValueError("non-numeric regression target")
            limits = mapping["target_map"]
            spread = limits["max"] - limits["min"]
            result.append(0.5 if spread == 0 else max(0.0, min(1.0, (number - limits["min"]) / spread)))
        projected.append(result)
    return projected


def write_numeric(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows([[format(value, ".17g") for value in row] for row in rows])


def prepare(entry: dict, output_root: Path, seen: dict | None = None) -> dict:
    license_info = entry.get("license", {})
    if license_info.get("status") != "recorded" or not license_info.get("spdx") or not license_info.get("evidence_url"):
        raise ValueError("license evidence incomplete")
    if entry["task"] not in ("binary", "regression"):
        raise ValueError("unsupported task")
    raw = entry["raw"]
    header = entry.get("header", True)
    if isinstance(raw, (str, list)):
        paths = [Path(raw)] if isinstance(raw, str) else [Path(item) for item in raw]
        if not paths:
            raise ValueError("empty source file list")
        rows = []
        raw_files = {}
        for index, path in enumerate(paths):
            file_columns, file_rows = read_csv(path, header)
            if index == 0:
                columns = file_columns
            elif file_columns != columns:
                raise ValueError("source file columns differ")
            rows.extend(file_rows)
            raw_files[str(index)] = {"path": str(path), "sha256": sha256(path)}
        indices = split_rows(rows, columns.index(entry["target"]), entry["task"])
        parts = {name: [rows[i] for i in ids] for name, ids in indices.items()}
        split_kind = "deterministic_grouped_60_20_20"
    else:
        if not entry.get("official_split_id") or not entry.get("official_split_source"):
            raise ValueError("official split provenance incomplete")
        parts, raw_files = {}, {}
        for name in ("train", "validation", "test"):
            path = Path(raw[name])
            part_columns, rows = read_csv(path, header)
            if name == "train":
                columns = part_columns
            elif part_columns != columns:
                raise ValueError("official split columns differ")
            parts[name] = rows
            raw_files[name] = {"path": str(path), "sha256": sha256(path)}
        split_kind = "official"
    check_nonoverlap(parts)
    if entry.get("transformed_sha256") and (len(raw_files) != 1 or next(iter(raw_files.values()))["sha256"] != entry["transformed_sha256"]):
        raise ValueError("transformed source digest mismatch")
    all_rows = [row for rows in parts.values() for row in rows]
    if entry["panel"] == "public_extension" and not (500 <= len(all_rows) <= 100000 and 1 <= len(columns) - 1 <= 2000):
        raise ValueError("public extension size outside preregistered range")
    mapping = fit_projection(parts["train"], columns, entry["target"], entry["task"])
    source_row_hash = digest(sorted(row_digest(row) for row in all_rows))
    source_key = (entry["source_identity"], source_row_hash)
    if seen is not None and source_key in seen:
        return {"format": "dope-benchmark-deduplication", "duplicate_id": entry["id"],
                "canonical_id": seen[source_key], "source_row_hash": source_row_hash}
    split_hashes = {name: digest(sorted(row_digest(row) for row in rows)) for name, rows in parts.items()}
    dataset_id = entry["id"]
    worker = output_root / "worker" / dataset_id
    evaluator = output_root / "evaluator" / dataset_id
    if worker.exists() or evaluator.exists():
        raise FileExistsError("prepared dataset already exists; freeze is immutable")
    worker.mkdir(parents=True)
    evaluator.mkdir(parents=True)
    for name, rows in parts.items():
        destination = (evaluator if name == "test" else worker) / f"{name}.csv"
        write_numeric(destination, project(rows, mapping))
    (worker / "projection.json").write_text(json.dumps(mapping, sort_keys=True, indent=2) + "\n")
    manifest = {
        "format": "dope-benchmark-dataset", "version": 1, "id": dataset_id,
        "panel": entry["panel"], "source": entry["source"],
        "source_identity": entry["source_identity"], "license": license_info,
        "task": entry["task"], "target": entry["target"],
        "raw_files": raw_files, "source_row_hash": source_row_hash,
        "source_archive_sha256": entry.get("source_archive_sha256"),
        "transformation_sha256": entry.get("transformation_sha256"),
        "rows": len(all_rows), "columns": len(columns),
        "split": {"kind": split_kind, "seed": None if split_kind == "official" else 1729,
                  "hashes": split_hashes, "rows": {name: len(rows) for name, rows in parts.items()}},
        "projection_sha256": sha256(worker / "projection.json"),
        "projected_files": {name: sha256((evaluator if name == "test" else worker) / f"{name}.csv") for name in parts},
    }
    (evaluator / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (worker / "worker-manifest.json").write_text(json.dumps({
        "dataset_id": dataset_id, "split_hashes": {name: split_hashes[name] for name in ("train", "validation")},
        "projection_sha256": manifest["projection_sha256"], "train_rows": len(parts["train"]),
        "projected_hashes": {name: manifest["projected_files"][name] for name in ("train", "validation")},
    }, sort_keys=True, indent=2) + "\n")
    if seen is not None:
        seen[source_key] = dataset_id
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.output_root.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or args.output_root.resolve().is_relative_to(Path("/tmp")):
        parser.error("bulk data must live outside the worktree and /tmp")
    selection = json.loads(args.selection.read_text())
    seen = {}
    for entry in selection["datasets"]:
        print(json.dumps(prepare(entry, args.output_root, seen), sort_keys=True))


if __name__ == "__main__":
    main()
