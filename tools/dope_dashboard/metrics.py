"""Normalize probe records, attach shared DDP runs, and aggregate DOPE v2.0.0 KPI & checkpoint telemetry."""
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
        tret = row.get("train_retention")
        if tret is None:
            tret = row.get("train_unclipped_retention")
        train.append(
            {
                "step": int(row["step"]),
                "total_steps": int(row["total_steps"]) if type(row.get("total_steps")) is int else None,
                "stage": int(row["stage"]) if type(row.get("stage")) is int else None,
                "arch": str(row.get("arch", ""))[:32],
                "time": _num(row.get("time")),
                "wall_s": _num(row.get("wall_s")),
                "lr": _num(row.get("lr")),
                "loss": _num(row.get("loss")),
                "tstr_loss": _num(row.get("tstr_loss")),
                "distill_loss": _num(row.get("distill_loss")),
                "driver_loss": _num(row.get("driver_loss")),
                "realism_loss": _num(row.get("realism_loss")),
                "mmd_loss": _num(row.get("mmd_loss")),
                "infonce_loss": _num(row.get("infonce_loss")),
                "grad_norm": _num(row.get("grad_norm")),
                "skipped_spikes": int(row.get("skipped_spikes", 0)) if type(row.get("skipped_spikes", 0)) is int else 0,
                "cohost_yields": int(row.get("cohost_yields", 0)) if type(row.get("cohost_yields", 0)) is int else 0,
                "train_unclipped_retention": _num(tret, allow_negative=True),
                "tables_per_s": _num(row.get("tables_per_s")),
                "unique_splits_seen": _num(row.get("unique_splits_seen")),
                "encoding_bytes_avg": _num(row.get("encoding_bytes_avg")),
                "regret_boost_mean": _num(row.get("regret_boost_mean")),
                "avg_n_fit": _num(row.get("avg_n_fit")),
                "avg_n_cols": _num(row.get("avg_n_cols")),
                "avg_noise_cols": _num(row.get("avg_noise_cols")),
                "avg_derived_cols": _num(row.get("avg_derived_cols")),
                "gpu_vram_mib": _num(row.get("gpu_vram_mib")),
                "world_size": int(row.get("world_size", 1)) if type(row.get("world_size", 1)) is int else 1,
                "task_counts": row.get("task_totals") or row.get("task_counts") or {},
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
                "arch": str(row.get("arch") or row.get("engine") or "")[:32],
                "checkpoint": str(row.get("checkpoint") or "")[:64],
                "eval_role": str(row.get("eval_role") or "step_ema")[:32],
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
                "marg_auc_median": _num(row.get("marg_auc_median")),
                "dcr_share_median": _num(row.get("dcr_share_median")),
                "mia_auc_median": _num(row.get("mia_auc_median")),
                "exact_copy_excess_median": _num(row.get("exact_copy_excess_median")),
                "encoding_bytes_median": _num(row.get("encoding_bytes_median")),
                "tables_evaluated": int(row.get("tables_evaluated", 0)) if type(row.get("tables_evaluated", 0)) is int else 0,
                "formal_dp_state": str(row.get("formal_dp_state", "not_claimed")),
            }
        )
    eval_by_step = {e["step"]: e for e in evals}
    ckpts = []
    for c in (raw.get("checkpoints") or [])[:32]:
        if not isinstance(c, dict):
            continue
        st = int(c["step"]) if type(c.get("step")) is int else None
        ev = eval_by_step.get(st, {}) if st is not None else {}
        ckpts.append(
            {
                "checkpoint": str(c.get("checkpoint", ""))[:64],
                "role": str(c.get("role", "step_ema"))[:24],
                "arch": str(c.get("arch") or ev.get("arch") or "")[:32],
                "step": st,
                "time": _num(c.get("time")),
                "bytes": int(c["bytes"]) if type(c.get("bytes")) is int else None,
                "sha256": str(c.get("sha256", ""))[:16],
                "train_loss": _num(c.get("train_loss")),
                "train_tstr": _num(c.get("train_tstr")),
                "train_mmd": _num(c.get("train_mmd")),
                "main_kpi_retention_median": _num(c.get("main_kpi_retention_median", ev.get("main_kpi_retention_median")), allow_negative=True),
                "regression_retention": _num(c.get("regression_retention", ev.get("regression_retention")), allow_negative=True),
                "binary_retention": _num(c.get("binary_retention", ev.get("binary_retention")), allow_negative=True),
                "multiclass_retention": _num(c.get("multiclass_retention", ev.get("multiclass_retention")), allow_negative=True),
                "multilabel_retention": _num(c.get("multilabel_retention", ev.get("multilabel_retention")), allow_negative=True),
                "parity_coverage": _num(c.get("parity_coverage", ev.get("parity_coverage"))),
                "c2st_auc_median": _num(c.get("c2st_auc_median", ev.get("c2st_auc_median"))),
                "marg_auc_median": _num(c.get("marg_auc_median", ev.get("marg_auc_median"))),
                "driver_spearman_median": _num(c.get("driver_spearman_median", ev.get("driver_spearman_median")), allow_negative=True),
                "dcr_share_median": _num(c.get("dcr_share_median", ev.get("dcr_share_median"))),
                "mia_auc_median": _num(c.get("mia_auc_median", ev.get("mia_auc_median"))),
                "exact_copy_excess_median": _num(c.get("exact_copy_excess_median", ev.get("exact_copy_excess_median"))),
                "encoding_bytes_median": _num(c.get("encoding_bytes_median", ev.get("encoding_bytes_median"))),
                "delta_vs_prev": _num(c.get("delta_vs_prev"), allow_negative=True),
            }
        )
    return {
        "name": str(raw.get("name", "run"))[:80],
        "config": raw.get("config") if isinstance(raw.get("config"), dict) else {},
        "regret": raw.get("regret") if isinstance(raw.get("regret"), dict) else {},
        "checkpoints": ckpts,
        "train": train,
        "eval": evals,
        "latest_train": train[-1] if train else None,
        "latest_eval": evals[-1] if evals else None,
    }


def host_view(host: str, sample: dict[str, Any] | None, seen: float | None, detail: str, now: float) -> dict[str, Any]:
    if sample is None:
        return {
            "host": host, "status": "offline" if detail else "connecting", "detail": detail,
            "seen": seen, "resources": {}, "gpus": [], "procs": [], "runs": [],
            "role_badge": "OFFLINE" if detail else "INIT",
        }
    stale = seen is None or (now - seen) > 15.0
    procs = sample.get("procs") if isinstance(sample.get("procs"), list) else []
    gpus = sample.get("gpus") if isinstance(sample.get("gpus"), list) else []
    runs = [_normalize_run(r) for r in (sample.get("runs") or []) if isinstance(r, dict)]
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
    elif host in ("xbabe0", "xbabe2"):
        badge = "CTRL"
    else:
        badge = "IDLE"
    return {
        "host": host, "status": "stale" if stale else "live", "detail": detail if stale else "",
        "seen": seen, "resources": sample.get("resources") if isinstance(sample.get("resources"), dict) else {},
        "gpus": gpus, "procs": procs, "runs": runs, "role_badge": badge, "shared_from": None,
    }


def attach_shared_runs(hosts: list[dict[str, Any]]) -> None:
    """Mirror xbabe3:~/jsr/: link DDP follower hosts to their rank-0 leader run telemetry."""
    leaders: dict[str, tuple[str, dict[str, Any]]] = {}
    for h in hosts:
        if h["status"] != "live":
            continue
        for p in h.get("procs") or []:
            ddp = p.get("ddp")
            if isinstance(ddp, dict) and ddp.get("rank") == 0 and h.get("runs"):
                leaders[str(ddp.get("group", ""))] = (h["host"], h["runs"][0])
    for h in hosts:
        if h["status"] != "live":
            continue
        for p in h.get("procs") or []:
            ddp = p.get("ddp")
            if isinstance(ddp, dict) and (ddp.get("rank") or 0) > 0:
                grp = str(ddp.get("group", ""))
                if grp in leaders and not h.get("runs"):
                    h["shared_from"] = {"host": leaders[grp][0], "run": leaders[grp][1]["name"]}


def fleet_summary(hosts: list[dict[str, Any]], history: deque[dict[str, Any]], now: float) -> dict[str, Any]:
    attach_shared_runs(hosts)
    live_nodes = sum(1 for h in hosts if h["status"] == "live")
    total_nodes = len(hosts)
    total_tables_s = 0.0
    total_splits = 0
    losses, tstr_losses, mmd_losses, train_rets, grad_norms = [], [], [], [], []
    avg_noise_vals, avg_fit_vals, enc_bytes_vals = [], [], []
    dope_vram_mib = other_vram_mib = free_vram_mib = 0.0
    active_dope_procs = 0
    ddp_groups: dict[str, int] = {}
    leaderboard: list[dict[str, Any]] = []
    all_checkpoints: list[dict[str, Any]] = []

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
            # Consider run active for throughput if train log updated within 180s
            is_fresh = lt is not None and lt.get("time") is not None and (now - lt["time"]) <= 180.0
            if lt:
                if is_fresh and lt.get("tables_per_s") is not None:
                    total_tables_s += lt["tables_per_s"]
                if lt.get("unique_splits_seen") is not None:
                    total_splits += int(lt["unique_splits_seen"])
                if is_fresh and lt.get("loss") is not None:
                    losses.append(lt["loss"])
                if is_fresh and lt.get("tstr_loss") is not None:
                    tstr_losses.append(lt["tstr_loss"])
                if is_fresh and lt.get("mmd_loss") is not None:
                    mmd_losses.append(lt["mmd_loss"])
                if is_fresh and lt.get("grad_norm") is not None:
                    grad_norms.append(lt["grad_norm"])
                if is_fresh and lt.get("train_unclipped_retention") is not None:
                    train_rets.append(lt["train_unclipped_retention"])
                if lt.get("avg_noise_cols") is not None:
                    avg_noise_vals.append(lt["avg_noise_cols"])
                if lt.get("avg_n_fit") is not None:
                    avg_fit_vals.append(lt["avg_n_fit"])
                if lt.get("encoding_bytes_avg") is not None and lt["encoding_bytes_avg"] > 0:
                    enc_bytes_vals.append(lt["encoding_bytes_avg"])
            if le:
                # Pick best eval row for leaderboard + also report latest step
                best_ev = max(
                    run["eval"],
                    key=lambda r: r["main_kpi_retention_median"] if r["main_kpi_retention_median"] is not None else -999.0,
                )
                leaderboard.append(
                    {
                        "host": h["host"], "run": run["name"], "arch": le.get("arch") or "",
                        "step": le["step"], "best_step": best_ev["step"],
                        "best_main_kpi": best_ev["main_kpi_retention_median"],
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
                        "marg_auc_median": le["marg_auc_median"],
                        "dcr_share_median": le["dcr_share_median"],
                        "mia_auc_median": le["mia_auc_median"],
                        "exact_copy_excess_median": le["exact_copy_excess_median"],
                        "encoding_bytes_median": le["encoding_bytes_median"],
                        "tables_evaluated": le["tables_evaluated"],
                        "active": is_fresh,
                    }
                )
            for ck in run.get("checkpoints") or []:
                all_checkpoints.append({"host": h["host"], "run": run["name"], **ck})

    leaderboard.sort(
        key=lambda r: (r["main_kpi_retention_median"] if r["main_kpi_retention_median"] is not None else -999.0),
        reverse=True,
    )
    role_pri = {"best_main_kpi": 0, "valley_soup": 1, "step_ema": 2}
    all_checkpoints.sort(
        key=lambda c: (role_pri.get(str(c.get("role")), 9), -(c.get("main_kpi_retention_median") or -999.0), -(c.get("time") or 0.0))
    )
    best_eval = leaderboard[0] if leaderboard else None
    med_loss = round(float(median(losses)), 4) if losses else None
    med_tstr = round(float(median(tstr_losses)), 4) if tstr_losses else None
    med_mmd = round(float(median(mmd_losses)), 4) if mmd_losses else None
    med_train_ret = round(float(median(train_rets)), 4) if train_rets else None
    best_kpi = best_eval["main_kpi_retention_median"] if best_eval else med_train_ret

    point = {
        "at": round(now, 1), "tables_per_s": round(total_tables_s, 2), "median_loss": med_loss,
        "median_tstr_loss": med_tstr, "median_mmd_loss": med_mmd, "main_kpi_retention": best_kpi,
        "unique_splits": total_splits,
    }
    if not history or (now - history[-1]["at"]) >= 4.0:
        history.append(point)

    return {
        "live_nodes": live_nodes, "total_nodes": total_nodes, "active_dope_procs": active_dope_procs,
        "ddp_groups": ddp_groups, "tables_per_s": round(total_tables_s, 2), "unique_splits_seen": total_splits,
        "median_loss": med_loss, "median_tstr_loss": med_tstr, "median_mmd_loss": med_mmd,
        "median_grad_norm": round(float(median(grad_norms)), 3) if grad_norms else None,
        "median_train_retention": med_train_ret,
        "avg_noise_cols": round(float(median(avg_noise_vals)), 2) if avg_noise_vals else None,
        "avg_n_fit": round(float(median(avg_fit_vals)), 1) if avg_fit_vals else None,
        "encoding_bytes_median": round(float(median(enc_bytes_vals)), 0) if enc_bytes_vals else None,
        "dope_vram_gib": round(dope_vram_mib / 1024.0, 2),
        "other_vram_gib": round(other_vram_mib / 1024.0, 2),
        "free_vram_gib": round(free_vram_mib / 1024.0, 2),
        "best_eval": best_eval, "leaderboard": leaderboard,
        "checkpoints": all_checkpoints[:80], "history": list(history),
        "formal_dp_state": "not_claimed",
    }
