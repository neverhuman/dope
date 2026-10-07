#!/usr/bin/env python3
"""Render a frozen rights-safe scalar ledger; no private inputs are read."""
from pathlib import Path
import argparse, csv, hashlib, io, json, math
HERE = Path(__file__).resolve().parent
PINS = {'delta.json': (9576, 'e5377bb0db549cf7be91f376810239dac9ce852e2323c154923c6b2a8b337e30'), 'source-proof.json': (6111, 'e32014ffb250e249375f3bafcbb5bb067740d023163a31f068b9987c118014f2')}

def read(name):
    size, sha = PINS[name]
    raw = (HERE / name).read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("public ledger digest mismatch")
    return json.loads(raw)

def validate(d, p):
    if d["format"] != "work-order-b-native-paid-cost-delta-v1" or p["format"] != "work-order-b-native-paid-cost-source-proof-v1":
        raise ValueError("unexpected public ledger format")
    native = [r for r in d["operations"] if r["phase"] == "native_population"]
    if len(native) != 5 or len({r["job_sha256"] for r in native}) != 5:
        raise ValueError("physical operation count mismatch")
    if not math.isclose(math.fsum(r["parent_elapsed_seconds"] for r in native), d["cost"]["new_paid_parent_seconds"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("parent clock mismatch")
    if not math.isclose(d["baseline"]["paid_parent_seconds"] + d["cost"]["new_paid_parent_seconds"], d["cost"]["current_cumulative_native_and_prefit_paid_parent_seconds"], abs_tol=1e-9, rel_tol=0):
        raise ValueError("cumulative clock mismatch")
    if d["progress"]["current_inclusive_paid_trial_count"] - d["baseline"]["paid_trial_count"] != sum(r["paid_trial_increment"] for r in native):
        raise ValueError("trial reconciliation mismatch")
    joins = [{k:r[k] for k in ("id","job_sha256","parent_ref","receipt_ref")} for r in d["operations"]]
    if joins != p["operation_joins"] or d["progress"]["current_logical_progress"] is not None:
        raise ValueError("custody or progress mismatch")

def csv_text(d):
    columns = ["id","phase","status","seed","attempt","parent_elapsed_seconds","metadata_hold_seconds","paid_trial_increment","charged_model_plus_projection_bytes","co_tenant","shared_GPU_cost","maximum_observed_own_GPU_resident_mib","job_sha256"]
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in d["operations"]:
        w.writerow(["" if r[k] is None else str(r[k]).lower() if isinstance(r[k],bool) else r[k] for k in columns])
    return buf.getvalue()

def md_text(d):
    lines = ["# Closed native operation costs", "", "| Operation | Status | Parent seconds | Charged bytes | Co-tenant | Observed own GPU MiB |", "| --- | --- | ---: | ---: | --- | ---: |"]
    for r in d["operations"]:
        if r["phase"] != "native_population":
            continue
        vals = [r["id"], r["status"], f'{r["parent_elapsed_seconds"]:.6f}', r["charged_model_plus_projection_bytes"], r["co_tenant"], r["maximum_observed_own_GPU_resident_mib"]]
        lines.append("| " + " | ".join("unknown" if v is None else str(v).lower() if isinstance(v,bool) else str(v) for v in vals) + " |")
    c = d["cost"]
    lines.extend(["", f'Five paid operations add **{c["new_paid_parent_seconds"]:.6f} s** to the fixed PR177 cutoff: **{c["current_cumulative_native_and_prefit_paid_parent_seconds"]:.6f} s**, **116 paid trials**.', "", f'Released sampler metadata hold: **{c["separate_metadata_hold_seconds_excluded_from_native_total"]:.6f} s**, zero science and zero paid trials; excluded from native time.', "", "Parent time includes guarded operation overhead. Nested timers are nonadditive. Residency is an observed maximum, not a true peak. Logical progress and gated scientific claims remain unavailable."])
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    d, p = read("delta.json"), read("source-proof.json")
    validate(d, p)
    for name, text in (("costs.csv",csv_text(d)), ("costs.md",md_text(d))):
        path = HERE / name
        if args.check:
            if path.read_text() != text:
                raise ValueError("public projection mismatch")
        else:
            path.write_text(text)

if __name__ == "__main__":
    main()
