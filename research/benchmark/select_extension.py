"""Select rights-reviewed OpenML-CC18 and CTR23 extension tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_BY_TASK = {"binary": "OpenML-CC18", "regression": "CTR23"}


def select(candidates: list[dict], limit: int = 10) -> dict:
    if limit < 1:
        raise ValueError("selection limit must be positive")
    eligible = {task: [] for task in SOURCE_BY_TASK}
    exclusions = []
    for row in candidates:
        task = row.get("task")
        identity = str(row.get("official_task_id", "missing"))
        reasons = []
        if task not in SOURCE_BY_TASK or row.get("suite") != SOURCE_BY_TASK.get(task):
            reasons.append("task_or_suite")
        if not identity or identity == "missing":
            reasons.append("official_task_id")
        if not isinstance(row.get("rows"), int) or not 500 <= row["rows"] <= 100000:
            reasons.append("row_count")
        if not isinstance(row.get("features"), int) or not 1 <= row["features"] <= 2000:
            reasons.append("feature_count")
        if not row.get("target"):
            reasons.append("target")
        license_info = row.get("license", {})
        if (license_info.get("status") != "recorded" or not license_info.get("spdx")
                or not license_info.get("evidence_url")):
            reasons.append("redistribution_rights")
        if not row.get("source_identity") or not row.get("source_row_hash"):
            reasons.append("source_identity_or_row_hash")
        if reasons:
            exclusions.append({"official_task_id": identity, "reasons": reasons})
        else:
            eligible[task].append(row)
    ordered_all = sorted((row for rows in eligible.values() for row in rows),
                         key=lambda row: (hashlib.sha256(str(row["official_task_id"]).encode()).hexdigest(),
                                          str(row["official_task_id"])))
    seen = set()
    deduplicated = {task: [] for task in SOURCE_BY_TASK}
    for row in ordered_all:
        key = (row["source_identity"], row["source_row_hash"])
        if key in seen:
            exclusions.append({"official_task_id": str(row["official_task_id"]),
                               "reasons": ["duplicate_source_rows"]})
        else:
            seen.add(key)
            deduplicated[row["task"]].append(row)
    selected = {}
    shortfall = {}
    for task in ("binary", "regression"):
        selected[task] = []
        for row in deduplicated[task]:
            if len(selected[task]) == limit:
                exclusions.append({"official_task_id": str(row["official_task_id"]),
                                   "reasons": ["after_first_ten_hash_order"]})
            else:
                selected[task].append(row)
        shortfall[task] = limit - len(selected[task])
    return {"format": "dope-benchmark-extension-selection", "version": 1,
            "selected": selected, "shortfall": shortfall, "exclusions": exclusions,
            "selection_rule": "sha256_official_task_id_ascending_after_rights_filter_and_dedup"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewed_candidates", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(json.loads(args.reviewed_candidates.read_text())["candidates"]),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
