"""Normalize probe records and aggregate fleet-wide DOPE KPI metrics and history."""
from __future__ import annotations

from collections import deque
from math import isfinite
from statistics import median
from typing import Any


def _num(value: Any, allow_negative: bool = False) -> float | None:
    if type(value) in (int, float) and isfinite(value) and abs(value) < 1e15:
        if not allow_negative and value < 0:
            return None
        return float(value)
    return None


def _normalize_run(raw: dict[str, Any]) -> dict[str, Any]:
    train = []
    for row in (raw.get("train") or [])[-180:]:
        if not isinstance(row, dict) or _num(row.get("step")) is None:
            continue
        train.append(
            {
                "step": int(row["step"]),
                "time": _num(row.get("time")),
                "wall_s": _num(row.get("wall_s")),
                "lr": _num(row.get("lr")),
                "loss": _num(row.get("loss")),
                "tstr_loss": _num(row.get("tstr_loss")),
                "driver_loss": _num(row.get("driver_loss")),
                "realism_loss": _num(row.get("realism_loss")),
                "infonce_loss": _num(row.get("infonce_loss")),
                "train_unclipped_retention": _num(row.get("train_unclipped_retention"), allow_negative=True),
                "tables_per_s": _num(row.get("tables_per_s")),
                "unique_splits_seen": _num(row.get("unique_splits_seen")),
                "avg_n_fit": _num(row.get("avg_n_fit")),
                "avg_n_cols": _num(row.get("avg_n_cols")),
                "avg_noise_cols": _num(row.get("avg_noise_cols")),
                "avg_derived_cols": _num(row.get("avg_derived_cols")),
                "gpu_vram_mib": _num(row.get("gpu_vram_mib")),
                "world_size": int(row.get("world_size", 1)) if type(row.get("world_size", 1)) is int else 1,
                "task_counts": row.get("task_counts") if isinstance(row.get("task_counts"), dict) else {},
            }
        )
    evals = []
    for row in (raw.get("eval") or [])[-100:]:
        if not isinstance(row, dict) or _num(row.get("step")) is None:
            continue
        evals.append(
            {
                "step": int(row["step"]),
                "time": _num(row.get("time")),
                "main_kpi_retention_median": _num(row.get("main_kpi_retention_median"), allow_negative=True),
                "main_kpi_retention_mean": _num(row.get("main_kpi_retention_mean"), allow_negative=True),
                "regression_retention": _num(row.get("regression_retention"), allow_negative=True),
                "binary_retention": _num(row.get("binary_retention"), allow_negative=True),
                "multiclass_retention": _num(row.get("multiclass_retention"), allow_negative=True),
                "multilabel_retention": _num(row.get("multilabel_retention"), allow_negative=True),
                "parity_coverage": _num(row.get("parity_coverage")),
                "realism_coverage": _num(row.get("realism_coverage")),
                "privacy_pass_coverage": _num(row.get("privacy_pass_coverage")),
                "driver_spearman_median": _num(row.get("driver_spearman_median"), allow_negative=True),
                "c2st_auc_median": _num(row.get("c2st_auc_median")),
                "dcr_share_median": _num(row.get("dcr_share_median")),
                "tables_evaluated": int(row.get("tables_evaluated", 0)) if type(row.get("tables_evaluated", 0)) is int else 0,
                "formal_dp_state": str(row.get("formal_dp_state", "not_claimed")),
            }
        )
    latest_tr = train[-1] if train else None
    latest_ev = evals[-1] if evals else None
    cfg = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    return {
        "name": str(raw.get("name", "run"))[:80],
        "config": cfg,
        "train": train,
        "eval": evals,
        "latest_train": latest_tr,
        "latest_eval": latest_ev,
    }


def host_view(host: str, sample: dict[str, Any] | None, seen: float | None, detail: str, now: float) -> dict[str, Any]:
    if sample is None:
        return {
            "host": host,
            "status": "offline" if detail else "connecting",
            "detail": detail,
            "seen": seen,
            "resources": {},
            "gpus": [],
            "procs": [],
            "runs": [],
            "role_badge": "OFFLINE" if detail else "INIT",
        }
    stale = seen is None or (now - seen) > 15.0
    procs = sample.get("procs") if isinstance(sample.get("procs"), list) else []
    gpus = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
    runs = [_normalize_run(r) for r in (sample.get("runs") or []) if isinstance(r, dict)]

    # Determine concise operational badge for node list
    roles = {p.get("role") for p in procs if isinstance(p, dict)}
    ddp_ranks = [p["ddp"]["rank"] for p in procs if isinstance(p, dict) and isinstance(p.get("ddp"), dict)]
    if stale:
        badge = "STALE"
    elif ddp_ranks:
        badge = f"DDP-{ddp_ranks[0]}"
    elif "ddp_trainer" in roles:
        badge = "TRAIN"
    elif "kpi_evaluator" in roles:
        badge = "EVAL"
    elif "rust_arena" in roles:
        badge = "ARENA"
    elif any((g.get("other_vram_mib") or 0) > 1000 for g in gpus):
        badge = "JSR-HOST"
    elif host == "xbabe0":
        badge = "CTRL"
    else:
        badge = "IDLE"

    return {
        "host": host,
        "status": "stale" if stale else "live",
        "detail": detail if stale else "",
        "seen": seen,
        "resources": sample.get("resources") if isinstance(sample.get("resources"), dict) else {},
        "gpus": gpus,
        "procs": procs,
        "runs": runs,
        "role_badge": badge,
    }


def fleet_summary(hosts: list[dict[str, Any]], history: deque[dict[str, Any]], now: float) -> dict[str, Any]:
    live_nodes = sum(1 for h in hosts if h["status"] == "live")
    total_nodes = len(hosts)
    total_tables_s = 0.0
    total_splits = 0
    losses, tstr_losses, train_rets = [], [], []
    avg_noise_vals, avg_fit_vals = [], []
    dope_vram_mib, other_vram_mib, free_vram_mib = 0.0, 0.0, 0.0
    active_dope_procs = 0
    ddp_groups: dict[str, int] = {}
    leaderboard: list[dict[str, Any]] = []

    for h in hosts:
        if h["status"] != "live":
            continue
        for g in h.get("gpus") or []:
            dope_vram_mib += float(g.get("dope_vram_mib") or 0.0)
            other_vram_mib += float(g.get("other_vram_mib") or 0.0)
            free_vram_mib += float(g.get("memory_free_mib") or 0.0)
        for p in h.get("procs") or []:
            active_dope_procs += 1
            if isinstance(p.get("ddp"), dict):
                grp = str(p["ddp"].get("group", ""))
                ddp_groups[grp] = ddp_groups.get(grp, 0) + 1
        for run in h.get("runs") or []:
            lt = run.get("latest_train")
            le = run.get("latest_eval")
            if lt:
                if lt.get("tables_per_s") is not None:
                    total_tables_s += lt["tables_per_s"]
                if lt.get("unique_splits_seen") is not None:
                    total_splits += int(lt["unique_splits_seen"])
                if lt.get("loss") is not None:
                    losses.append(lt["loss"])
                if lt.get("tstr_loss") is not None:
                    tstr_losses.append(lt["tstr_loss"])
                if lt.get("train_unclipped_retention") is not None:
                    train_rets.append(lt["train_unclipped_retention"])
                if lt.get("avg_noise_cols") is not None:
                    avg_noise_vals.append(lt["avg_noise_cols"])
                if lt.get("avg_n_fit") is not None:
                    avg_fit_vals.append(lt["avg_n_fit"])
            if le:
                leaderboard.append(
                    {
                        "host": h["host"],
                        "run": run["name"],
                        "step": le["step"],
                        "main_kpi_retention_median": le["main_kpi_retention_median"],
                        "regression_retention": le["regression_retention"],
                        "binary_retention": le["binary_retention"],
                        "multiclass_retention": le["multiclass_retention"],
                        "multilabel_retention": le["multilabel_retention"],
                        "parity_coverage": le["parity_coverage"],
                        "realism_coverage": le["realism_coverage"],
                        "privacy_pass_coverage": le["privacy_pass_coverage"],
                        "driver_spearman_median": le["driver_spearman_median"],
                        "c2st_auc_median": le["c2st_auc_median"],
                        "dcr_share_median": le["dcr_share_median"],
                        "tables_evaluated": le["tables_evaluated"],
                    }
                )

    leaderboard.sort(
        key=lambda r: (r["main_kpi_retention_median"] if r["main_kpi_retention_median"] is not None else -999.0),
        reverse=True,
    )
    best_eval = leaderboard[0] if leaderboard else None
    med_loss = round(float(median(losses)), 4) if losses else None
    med_tstr = round(float(median(tstr_losses)), 4) if tstr_losses else None
    med_train_ret = round(float(median(train_rets)), 4) if train_rets else None
    best_kpi = best_eval["main_kpi_retention_median"] if best_eval else med_train_ret

    point = {
        "at": round(now, 1),
        "tables_per_s": round(total_tables_s, 2),
        "median_loss": med_loss,
        "median_tstr_loss": med_tstr,
        "main_kpi_retention": best_kpi,
        "unique_splits": total_splits,
    }
    if not history or (now - history[-1]["at"]) >= 4.0:
        history.append(point)

    return {
        "live_nodes": live_nodes,
        "total_nodes": total_nodes,
        "active_dope_procs": active_dope_procs,
        "ddp_groups": ddp_groups,
        "tables_per_s": round(total_tables_s, 2),
        "unique_splits_seen": total_splits,
        "median_loss": med_loss,
        "median_tstr_loss": med_tstr,
        "median_train_retention": med_train_ret,
        "avg_noise_cols": round(float(median(avg_noise_vals)), 2) if avg_noise_vals else None,
        "avg_n_fit": round(float(median(avg_fit_vals)), 1) if avg_fit_vals else None,
        "dope_vram_gib": round(dope_vram_mib / 1024.0, 2),
        "other_vram_gib": round(other_vram_mib / 1024.0, 2),
        "free_vram_gib": round(free_vram_mib / 1024.0, 2),
        "best_eval": best_eval,
        "leaderboard": leaderboard,
        "history": list(history),
        "formal_dp_state": "not_claimed",
    }
