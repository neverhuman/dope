"""Pure, prospective BeyondArena preparation from supplied training rows.

No files, partitions or datasets are opened here. The caller must authenticate
a dedicated official TRAIN export before splitting it into inner fit/validation.
These helpers do not certify that custody or admit a family for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict

WIDE_FAMILIES = (
    "early_learning_predictors", "lending_club", "kick", "wine_world_cost",
)
OVERLAP_FAMILIES = (
    "heart_disease_va_long_beach", "qsar_biodeg", "hotel_booking_demand",
)
FEATURE_CAP = 2000
RAW_INPUT_CAP = 12
LOCKED_LICENSES = frozenset({"CC-BY-4.0", "CC0-1.0", "LicenseRef-Public-Domain", "MIT"})
LOCKED_STRATA = frozenset({"iid", "grouped", "temporal"})
CATALOG_BYTE_CAP = 1 << 20


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _table(rows: list[list[str]]) -> int:
    _require(type(rows) is list and bool(rows), "training_table_required")
    _require(type(rows[0]) is list and len(rows[0]) >= 2,
             "training_table_required")
    width = len(rows[0])
    _require(all(type(row) is list and len(row) == width
                 and all(type(value) is str for value in row) for row in rows),
             "training_table_required")
    return width


def _float(value: str) -> float | None:
    # Same finite-number classification as manifest.fit_projection.
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def screen_inner_fit(family: str, inner_fit_rows: list[list[str]],
                     columns: list[str], target: str) -> dict:
    """Freeze raw input positions in source order before fitted projection.

    Numeric inputs cost one feature; categorical inputs cost the number of
    nonempty inner-fit levels. A whole raw input is skipped if it would exceed
    the original 2000-feature cap. At most 12 raw inputs are kept. No levels
    are truncated, no target values
    rank inputs, and later inputs may still fit after an oversized input.
    """
    _require(type(family) is str and family in WIDE_FAMILIES, "family_outside_wide_screen")
    width = _table(inner_fit_rows)
    _require(type(columns) is list and len(columns) == width
             and all(type(c) is str and bool(c) for c in columns)
             and len(set(columns)) == width and type(target) is str
             and target in columns, "training_schema_required")
    target_index = columns.index(target)
    selected, omitted, output_width = [], [], 0
    for index in range(width):
        if index == target_index:
            continue
        if len(selected) >= RAW_INPUT_CAP:
            omitted.append({"column_index": index, "reason": "raw_input_cap"})
            continue
        values = [row[index] for row in inner_fit_rows]
        numeric = [_float(value) for value in values]
        if all(value is not None or raw == ""
               for value, raw in zip(numeric, values)):
            cost = 1 if any(value is not None for value in numeric) else 0
        else:
            cost = len(set(values) - {""})
        if cost == 0:
            omitted.append({"column_index": index, "reason": "all_missing_inner_fit"})
        elif output_width + cost > FEATURE_CAP:
            omitted.append({"column_index": index, "reason": "projected_width_cap"})
        else:
            selected.append(index)
            output_width += cost
    _require(bool(selected), "no_usable_training_inputs")
    indices = sorted([*selected, target_index])
    return {
        "family": family, "rule": "source_order12_full_input_width_budget_v1",
        "input_width": width, "column_indices": indices,
        "raw_input_cap": RAW_INPUT_CAP, "selected_raw_inputs": len(selected),
        "feature_indices": selected, "target_index": target_index,
        "target_index_in_selected": indices.index(target_index),
        "projected_feature_width_estimate": output_width,
        "target_output_width": 1, "projected_row_width_estimate": output_width + 1,
        "projected_feature_cap": FEATURE_CAP, "omitted_inputs": omitted,
        "learned_partition": "inner_fit_only",
        "validation_used_for_selection": False, "official_tests_opened": False,
        "actual_projection_ref": None, "measured_utility_cost": None,
        "complete_generator_byte_charge": None,
    }


def row_group_sha256(row: list[str]) -> str:
    """Original full raw-row hash, including the target and every input."""
    _require(type(row) is list and all(type(value) is str for value in row),
             "raw_string_row_required")
    return _sha(json.dumps(row, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8"))


def split_training_groups(train_rows: list[list[str]]) -> dict:
    """Original seed1729 80/20 split of complete official TRAIN row groups."""
    _table(train_rows)
    groups = defaultdict(list)
    for index, row in enumerate(train_rows):
        groups[row_group_sha256(row)].append(index)
    members = sorted(groups.items())
    seed_bytes = json.dumps([1729, "official_training"], sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False).encode()
    random.Random(_sha(seed_bytes)).shuffle(members)
    split = {"train": [], "validation": []}
    seen = 0
    for _, indices in members:
        name = "train" if seen < 0.8 * len(train_rows) else "validation"
        split[name].extend(indices)
        seen += len(indices)
    for indices in split.values():
        indices.sort()
    _require(all(split.values()), "insufficient_distinct_TRAIN_groups")
    return split


def drop_train_overlap_groups(train_rows: list[list[str]], receipt_bytes: bytes,
                              external_receipt_sha256: str,
                              source_binding: dict) -> tuple[list[list[str]], dict]:
    """Apply an already authenticated evaluator TRAIN-intersection receipt.

    The external pin must come from existing sealed evaluator custody. This
    helper neither creates a collision catalog nor reads an official TEST.
    Remove every TRAIN occurrence of a full-row group, before inner splitting.
    """
    _table(train_rows)
    _require(type(receipt_bytes) is bytes and _pin(external_receipt_sha256)
             and _sha(receipt_bytes) == external_receipt_sha256,
             "unbound_evaluator_overlap_receipt")
    receipt = json.loads(receipt_bytes)
    _require(type(receipt) is dict
             and receipt.get("format") == "dope-beyond-precomputed-overlap-groups"
             and type(receipt.get("version")) is int and receipt["version"] == 1,
             "overlap_receipt_format_changed")
    _require(type(source_binding) is dict, "overlap_source_identity_changed")
    for key in ("family", "source_manifest_sha256", "source_parquet_sha256",
                "source_split_metadata_sha256"):
        _require(key in source_binding and receipt.get(key) == source_binding[key],
                 "overlap_source_identity_changed")
    _require(receipt["family"] in OVERLAP_FAMILIES
             and all(_pin(source_binding[k]) for k in
                     ("source_manifest_sha256", "source_parquet_sha256",
                      "source_split_metadata_sha256"))
             and receipt.get("canonical_row_hash") ==
             "SHA256_JSON_strings_sortkeys_compact_UTF8_ensure_ascii_false"
             and receipt.get("from_existing_authenticated_evaluator_hash_catalog") is True
             and receipt.get("only_TRAIN_intersection_group_hashes") is True
             and type(receipt.get("official_test_rows_decoded_during_this_operation")) is int
             and receipt["official_test_rows_decoded_during_this_operation"] == 0
             and receipt.get("official_test_indices_or_values_in_worker") is False
             and _pin(receipt.get("evaluator_catalog_sha256")),
             "overlap_evaluator_scope_rejected")
    hashes = receipt.get("TRAIN_intersection_group_hashes")
    _require(type(hashes) is list and bool(hashes)
             and all(_pin(value) for value in hashes)
             and len(set(hashes)) == len(hashes), "overlap_group_hashes_invalid")
    blocked = set(hashes)
    _require(blocked <= {row_group_sha256(row) for row in train_rows},
             "overlap_group_not_present_in_TRAIN")
    kept = [row for row in train_rows if row_group_sha256(row) not in blocked]
    _require(len({row_group_sha256(row) for row in kept}) >= 2,
             "insufficient_TRAIN_groups_after_overlap_removal")
    return kept, {
        "removed_groups": len(blocked), "removed_TRAIN_rows": len(train_rows) - len(kept),
        "original_TRAIN_rows": len(train_rows), "retained_TRAIN_rows": len(kept),
        "only_TRAIN_changed": True, "entire_groups_removed": True,
        "external_overlap_receipt_sha256": external_receipt_sha256,
    }


def plan_catalog_successors(panel: dict, catalog_bytes: bytes | None,
                            external_catalog_sha256: str | None) -> dict:
    """Plan a separate cohort from finite supplied metadata, without fetching.

    A catalog is a JSON object with a datasets list of objects carrying name.
    Presence proves only name membership, never TRAIN export availability.
    All original lock families remain in their original denominator.
    """
    _require(type(panel) is dict and panel.get("format") == "dope-beyondarena-panel-lock"
             and type(panel.get("version")) is int and panel["version"] == 3,
             "panel_lock_required")
    families = panel.get("datasets")
    _require(type(families) is list and type(panel.get("selected_count")) is int
             and len(families) == panel["selected_count"], "panel_lock_required")
    locked = {}
    for item in families:
        _require(type(item) is dict and type(item.get("name")) is str
                 and type(item.get("source_family_id")) is str
                 and item["source_family_id"] == "beyondarena:" + item["name"]
                 and item["name"] not in locked
                 and type(item.get("license_spdx")) is str
                 and item["license_spdx"] in LOCKED_LICENSES
                 and type(item.get("stratum")) is str
                 and item["stratum"] in LOCKED_STRATA
                 and item.get("task") in ("binary_classification", "regression")
                 and _pin(item.get("parquet_sha256")), "panel_family_identity_required")
        locked[item["name"]] = item
    pilot = panel.get("historical_pilot_families")
    _require(type(pilot) is list and all(type(name) is str and name in locked for name in pilot)
             and len(set(pilot)) == len(pilot)
             and all(name in locked for name in OVERLAP_FAMILIES), "panel_pilot_required")
    order = sorted((item for name, item in locked.items() if name not in pilot),
                   key=lambda item: (_sha(item["source_family_id"].encode("utf-8")), item["name"]))
    result = {
        "scope": "supplied_catalog_dataset_name_membership_only",
        "catalog_sha256": external_catalog_sha256, "catalog_names_count": None,
        "successor_order_sha256": _sha(json.dumps(
            [item["source_family_id"] for item in order], separators=(",", ":")
        ).encode("utf-8")),
        "original_selected_count": panel["selected_count"],
        "original_lock_or_denominator_changed": False, "choices": [],
        "training_exports_verified": False, "prepared_families": 0,
        "official_tests_opened": False, "mfs_v2": None, "retention": None,
    }
    if catalog_bytes is None:
        _require(external_catalog_sha256 is None, "unbound_catalog_identity")
        return {**result, "status": "input_gap_catalog_not_supplied"}
    _require(type(catalog_bytes) is bytes and len(catalog_bytes) <= CATALOG_BYTE_CAP
             and _pin(external_catalog_sha256)
             and _sha(catalog_bytes) == external_catalog_sha256, "unbound_catalog_identity")
    catalog = json.loads(catalog_bytes)
    _require(type(catalog) is dict and type(catalog.get("datasets")) is list
             and len(catalog["datasets"]) <= 142, "catalog_metadata_shape_required")
    names = []
    for item in catalog["datasets"]:
        _require(type(item) is dict and type(item.get("name")) is str
                 and bool(item["name"]) and len(item["name"]) <= 128,
                 "catalog_metadata_shape_required")
        names.append(item["name"])
    _require(len(names) == len(set(names)), "duplicate_catalog_family")
    present = set(names)
    available = iter(item for item in order if item["name"] in present)
    choices = []
    for original in OVERLAP_FAMILIES:
        if original in present:
            choices.append({"requested_family": original, "status": "catalog_entry_present",
                            "prospective_successor": None})
        else:
            replacement = next(available, None)
            choices.append({
                "requested_family": original, "status": "entry_not_in_supplied_catalog",
                "prospective_successor": None if replacement is None else replacement["name"],
                "successor_parquet_sha256": None if replacement is None else replacement["parquet_sha256"],
                "separate_cohort_required": replacement is not None,
                "original_stratum": locked[original]["stratum"],
                "successor_stratum": None if replacement is None else replacement["stratum"],
                "strata_preserved": None if replacement is None else
                locked[original]["stratum"] == replacement["stratum"],
            })
    return {**result, "status": "prospective_catalog_plan_only",
            "catalog_names_count": len(names), "choices": choices}
