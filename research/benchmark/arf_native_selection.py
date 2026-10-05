"""Select ARF by its own FORDE objective with complete trial accounting."""
import hashlib
import json
import math


def require(ok,reason):
    if not ok:raise ValueError(reason)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def select(jobs,receipts):
    require(1 <= len(jobs) <= 8 and len(receipts)==len(jobs),"incomplete native trial ledger")
    by_job={r["job_sha256"]:r for r in receipts}
    require(len(by_job)==len(receipts) and set(by_job)=={identity(j) for j in jobs},"native job identity differs")
    require(len({j["dataset"] for j in jobs})==1 and len({j["fit_seed"] for j in jobs})==1,
            "native method-dataset cell differs")
    require(all(j["method"]=="ARF" and j["track"]=="common_numeric" and j["dp_budget"] is None
            and type(j["fit_seed"]) is int and 0<=j["fit_seed"]<2**32 for j in jobs),
            "native method identity differs")
    common=[{k:v for k,v in j.items() if k not in ("configuration","configuration_sha256","trial_index")} for j in jobs]
    require(len({identity(j) for j in common})==1,"native data or source binding differs")
    elapsed=[];candidates=[];counts={};cutoffs=0
    for job in jobs:
        job_id=identity(job);row=by_job[job_id];status=row["status"]
        seconds=row["elapsed_seconds"]
        require(type(seconds) in (float,int) and math.isfinite(seconds) and 0<=seconds,
                "native attempt time unavailable")
        elapsed.append(seconds);counts[status]=counts.get(status,0)+1
        if status in ("deadline_unstarted","deadline_truncated"):
            cutoffs+=1;continue
        require(status in ("ok","timeout","infra_interrupted","method_error","adapter_incompatible"),
                "unclassified native trial")
        if status!="ok":continue
        result=row["result"];kpi=result["native_kpi"];value=kpi["value"]
        require(kpi["objective"]=="heldout_forde_mean_log_density" and kpi["direction"]=="maximize"
                and kpi["partition"]=="validation" and type(kpi["fit_seed"]) is int
                and kpi["fit_seed"]==job["fit_seed"]
                and kpi["validation_sha256"]==job["worker"]["files"]["validation.csv"]
                and kpi["not_comparable_across_methods"] is True
                and type(value) in (float,int) and math.isfinite(value),"native objective binding differs")
        charged=result["artifact_bytes"]
        require(type(charged) is int and charged>=0 and all(type(x["bytes"]) is int and x["bytes"]>=0 for x in result["artifact_inventory"].values())
                and charged==sum(x["bytes"] for x in result["artifact_inventory"].values()),
                "native artifact charges differ")
        configuration_sha256=identity(job["configuration"])
        require(configuration_sha256==job["configuration_sha256"],"native configuration differs")
        candidates.append((-value,charged,configuration_sha256,job_id))
    require(sum(elapsed)<=43200,"native cell time cap exceeded")
    winner=min(candidates)[-1] if candidates else None
    return dict(selected_job_sha256=winner,completed_search_without_scheduling_cutoff=(cutoffs==0),
        successful_trials=len(candidates),statuses=counts,elapsed_seconds=sum(elapsed),
        selected_on="heldout_forde_mean_log_density",shared_outcomes_used_for_selection=False,
        native_values_ranked_across_methods=False,mfs_v2=None,ptf_v1=None,superiority=None)
