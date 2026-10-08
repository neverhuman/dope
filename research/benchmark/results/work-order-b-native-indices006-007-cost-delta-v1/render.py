#!/usr/bin/env python3
"""Render only frozen public scalar metadata; private inputs are never reopened."""
from pathlib import Path
import argparse, csv, hashlib, io, json, math
HERE = Path(__file__).resolve().parent
PINS = {'delta.json': (5981, '432d4dab78ff8cee28779a69a99b32af22435319c85b627e36ff4fedf574308e'), 'source-proof.json': (15470, 'c060cab77aa6c4c5ed57ab917303e9e6b978b6812cc76bb607cdafcf25ac5180')}

def read(name):
    size, digest = PINS[name]
    blob = (HERE / name).read_bytes()
    if len(blob) != size or hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("public ledger digest mismatch")
    return json.loads(blob)

def validate(d, p):
    if d["format"] != "work-order-b-native-indices006-007-paid-infrastructure-cost-delta-v1" or p["format"] != "work-order-b-native-indices006-007-paid-cost-source-proof-v1":
        raise ValueError("unexpected public ledger format")
    rows = d["operations"]
    if len(rows) != 2 or any(r["status"] != "infrastructure_failure" or r["paid_trial_increment"] != 1 for r in rows) or len({r["job_sha256"] for r in rows}) != 2:
        raise ValueError("operation count or status mismatch")
    if not math.isclose(math.fsum(r["parent_elapsed_seconds"] for r in rows), d["cost"]["new_paid_parent_seconds"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("parent clock mismatch")
    if not math.isclose(d["baseline"]["paid_parent_seconds"] + d["cost"]["new_paid_parent_seconds"], d["cost"]["current_cumulative_native_and_prefit_paid_parent_seconds"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("cumulative clock mismatch")
    if d["progress"]["current_inclusive_paid_trial_count"] - d["baseline"]["paid_trial_count"] != 2:
        raise ValueError("trial reconciliation mismatch")
    if [{k:r[k] for k in ("id","job_sha256","parent_ref","receipt_ref")} for r in rows] != p["operation_joins"]:
        raise ValueError("source custody mismatch")
    c = d["progress"]["current_logical_progress"]
    if c["accepted"] + c["historical_artifact_cap"] + c["distinct_unresolved_infrastructure_slots"] + c["unstarted_at_paid_budget_snapshot"] != c["scheduled"]:
        raise ValueError("logical roster mismatch")

def csv_text(d):
    columns = ['id', 'phase', 'status', 'seed', 'attempt', 'parent_elapsed_seconds', 'metadata_hold_seconds', 'paid_trial_increment', 'charged_model_plus_projection_bytes', 'co_tenant', 'shared_GPU_cost', 'maximum_observed_own_GPU_resident_mib', 'job_sha256']
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in d["operations"]:
        writer.writerow(["" if row[k] is None else str(row[k]).lower() if isinstance(row[k],bool) else row[k] for k in columns])
    return buf.getvalue()

def md_text(d):
    r, c = d["operations"][0], d["cost"]
    g = d["progress"]["current_logical_progress"]
    return '# Closed indices006–007 infrastructure costs\n\n| Operation | Scientific status | Parent seconds | Charged bytes | Sharing |\n| --- | --- | ---: | --- | --- |\n| paid_index006 | infrastructure_failure | 54.268754 | unavailable | unknown |\n| paid_index007 | infrastructure_failure | 52.069365 | unavailable | unknown |\n\nTwo paid infrastructure attempts add **106.338119 s** to PR181: **1730.760269 s**, **118 inclusive cell-budget trial units**. Their parent exits 0 do not establish successful native fits.\n\nLogical snapshot: **110 accepted + 2 historical caps + 5 distinct unresolved infrastructure slots + 383 snapshot-unstarted = 500**. The resolved prior infrastructure attempt remains an extra paid trial.\n\nNested timers, the prior1670.322090s metadata hold and the current800.174717s whole operator timer are nonadditive/excluded. The latter includes its paid operation wait; it is not a measured metadata-only cost. Unpaid deferral elapsed remains unknown. No accepted artifact, GPU observation, true peak, energy or kernel-only fit time is inferred.\n\n| Cost/count basis | Frozen value |\n| --- | ---: |\n| New native/prefit parent aggregate, excluding selected seed11 costs | 1730.7602687231265 s |\n| Historical-inclusive cell ledger, including selected seed11 costs | 2525.3003210125025 s |\n| Selected historical seed11 initial cell ledger | 794.540052289376 s |\n| Inclusive cell-budget trial units | 118 |\n| New native/prefit physical attempts since selected seed11 history | 18 |\n| Distinct canonical accounted logical slots | 117 |\n\nThe two clock bases are alternatives; do not add them. The118 budget units preserve100 selected historical units plus18 new physical attempts. They do not mean118 distinct logical fits. The resolved prior infrastructure attempt explains the one-unit difference from117 accounted canonical slots. This is the DOPE500-slot snapshot; caps from other methods/cohorts are excluded.\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    d, p = read("delta.json"), read("source-proof.json")
    validate(d, p)
    for name, text in (("costs.csv",csv_text(d)),("costs.md",md_text(d))):
        path = HERE / name
        if args.check:
            if path.read_text() != text:
                raise ValueError("public projection mismatch")
        else:
            path.write_text(text)

if __name__ == "__main__":
    main()
