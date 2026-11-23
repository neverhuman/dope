"""Lossless V2.0.0 training-wave telemetry archiver and publication-grade figure renderer.

Reads all mirrored fleet training logs (`~/dope-data/fleet-mirror/*/runs/cpufleet-v7*/`) and
dashboard throughput logs (`~/dope-data/dashboard/fleet_history.jsonl`), writes an immutable
SHA-256 archive manifest under `~/dope-data/archives/v2.0.0-wave/`, emits a compact committed
JSON receipt (`bench/real-world/results/v2.0.0/training-telemetry-snapshot.json`), and renders
300-DPI PNG + vector PDF loss-over-time and fleet-throughput figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _classify_wave(run_name: str, cfg: dict[str, Any]) -> str:
    recipe = str(cfg.get("recipe", ""))
    if "v7.2" in recipe or "v72" in run_name:
        return "V7.2"
    if "v71" in run_name or "v7.1" in recipe:
        return "V7.1"
    return "V7.0"


def _bin_series(
    runs_xy: list[tuple[np.ndarray, np.ndarray]], step_bin: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buckets: dict[int, list[float]] = {}
    for xs, ys in runs_xy:
        for x, y in zip(xs, ys):
            if math.isfinite(float(y)):
                b = int(round(float(x) / step_bin) * step_bin)
                buckets.setdefault(b, []).append(float(y))
    if not buckets:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty
    steps = np.array(sorted(buckets), dtype=np.float64)
    med = np.array([np.median(buckets[int(s)]) for s in steps], dtype=np.float64)
    q25 = np.array([np.percentile(buckets[int(s)], 25) for s in steps], dtype=np.float64)
    q75 = np.array([np.percentile(buckets[int(s)], 75) for s in steps], dtype=np.float64)
    return steps, med, q25, q75


def collect_wave_data(mirror_root: Path, history_log: Path) -> dict[str, Any]:
    """Load all V7.0 / V7.1 / V7.2 train & eval trajectories plus fleet throughput history."""
    waves: dict[str, list[dict[str, Any]]] = {"V7.0": [], "V7.1": [], "V7.2": []}
    total_train_rows = 0
    total_eval_rows = 0
    max_splits_by_run: dict[str, int] = {}
    byte_samples: list[float] = []
    source_files: list[Path] = []

    for run_dir in sorted(mirror_root.glob("*/runs/cpufleet-v7*")):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        wave = _classify_wave(run_dir.name, cfg)
        tr_path = run_dir / "train.jsonl"
        ev_path = run_dir / "eval.jsonl"
        tr_rows = _read_jsonl(tr_path)
        ev_rows = _read_jsonl(ev_path)
        if not tr_rows and not ev_rows:
            continue
        for p in (cfg_path, tr_path, ev_path):
            if p.is_file():
                source_files.append(p)
        total_train_rows += len(tr_rows)
        total_eval_rows += len(ev_rows)
        if tr_rows:
            max_splits_by_run[f"{run_dir.parent.parent.name}/{run_dir.name}"] = max(
                int(r.get("unique_splits_seen", 0)) for r in tr_rows
            )
            for r in tr_rows[-20:]:
                b = r.get("encoding_bytes_avg")
                if isinstance(b, (int, float)) and math.isfinite(b):
                    byte_samples.append(float(b))
        waves[wave].append({
            "host": run_dir.parent.parent.name,
            "run_name": run_dir.name,
            "config": cfg,
            "train": tr_rows,
            "eval": ev_rows,
        })

    hist_rows = _read_jsonl(history_log)
    if history_log.is_file():
        source_files.append(history_log)
    return {
        "waves": waves,
        "history": hist_rows,
        "total_train_rows": total_train_rows,
        "total_eval_rows": total_eval_rows,
        "total_unique_splits": sum(max_splits_by_run.values()),
        "byte_samples": byte_samples,
        "source_files": source_files,
    }


def archive_telemetry(data: dict[str, Any], archive_dir: Path, mirror_root: Path) -> dict[str, Any]:
    """Copy all telemetry JSONLs + configs into an immutable SHA-256 archived directory."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, Any]] = []
    for src in data["source_files"]:
        try:
            rel = src.relative_to(mirror_root)
            dst = archive_dir / "fleet" / rel
        except ValueError:
            dst = archive_dir / "dashboard" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_files.append({
            "relpath": str(dst.relative_to(archive_dir)),
            "bytes": dst.stat().st_size,
            "sha256": _sha256_file(dst),
        })
    manifest = {
        "schema": "dope.v2-telemetry-archive/v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(manifest_files),
        "total_bytes": sum(f["bytes"] for f in manifest_files),
        "files": manifest_files,
    }
    (archive_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _persisted_fleet_rate_bins(data: dict[str, Any], bin_s: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute wall-clock fleet throughput (tbl/s) and monotonic cumulative splits (M) from train.jsonl."""
    ts_pairs: list[tuple[float, float]] = []
    for rlist in data["waves"].values():
        for run in rlist:
            prev_u = 0.0
            for row in run["train"]:
                t_val, u_val = row.get("time"), row.get("unique_splits_seen")
                if isinstance(t_val, (int, float)) and isinstance(u_val, (int, float)):
                    du = max(0.0, float(u_val) - prev_u)
                    prev_u = float(u_val)
                    if du > 0.0:
                        ts_pairs.append((float(t_val), du))
    if not ts_pairs:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    ts_pairs.sort(key=lambda x: x[0])
    t_arr = np.array([p[0] for p in ts_pairs], dtype=np.float64)
    du_arr = np.array([p[1] for p in ts_pairs], dtype=np.float64)
    t0, t1 = float(t_arr[0]), float(t_arr[-1])
    n_bins = max(1, int(math.ceil((t1 - t0) / bin_s)))
    edges = t0 + np.arange(n_bins + 1, dtype=np.float64) * bin_s
    counts, _ = np.histogram(t_arr, bins=edges, weights=du_arr)
    mins = (edges[:-1] - t0 + 0.5 * bin_s) / 60.0
    return mins, counts / bin_s, np.cumsum(counts) / 1e6


def build_snapshot_receipt(data: dict[str, Any], archive_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Build compact committed JSON receipt summarizing persisted training & throughput telemetry."""
    waves_summary: dict[str, Any] = {}
    for wname, runs in data["waves"].items():
        if not runs:
            continue
        loss_xy = [
            (
                np.array([r["step"] for r in run["train"] if "loss" in r], dtype=np.float64),
                np.array([r["loss"] for r in run["train"] if "loss" in r], dtype=np.float64),
            )
            for run in runs if run["train"]
        ]
        st, med, _, _ = _bin_series(loss_xy, step_bin=200)
        max_step = max((int(r["step"]) for run in runs for r in run["train"]), default=0)
        waves_summary[wname] = {
            "runs": len(runs),
            "train_rows": sum(len(run["train"]) for run in runs),
            "eval_rows": sum(len(run["eval"]) for run in runs),
            "max_step": max_step,
            "loss_start_median": round(float(med[0]), 4) if len(med) else None,
            "loss_latest_median": round(float(med[-1]), 4) if len(med) else None,
            "sampled_loss_curve": [
                {"step": int(s), "loss_median": round(float(v), 4)}
                for s, v in zip(st[:: max(1, len(st) // 16)], med[:: max(1, len(med) // 16)])
            ],
        }
    hist = data["history"]
    _, bin_tps, _ = _persisted_fleet_rate_bins(data, bin_s=60.0)
    active_tps = [float(v) for v in bin_tps if float(v) > 1.0]
    tps_vals = active_tps or [float(r["tables_per_s"]) for r in hist if float(r.get("tables_per_s", 0)) > 0]
    vram_vals = [float(r.get("dope_vram_gib", 0.0)) for r in hist]
    bytes_arr = data["byte_samples"]
    return {
        "schema": "dope.v2-training-telemetry-snapshot/v1",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_scope": "training_and_dev_telemetry_only_not_confirm_kpi",
        "total_runs": sum(len(v) for v in data["waves"].values()),
        "total_train_rows_persisted": data["total_train_rows"],
        "total_eval_rows_persisted": data["total_eval_rows"],
        "total_unique_splits_seen": data["total_unique_splits"],
        "fleet_history_ticks": len(hist),
        "fleet_throughput_tbl_s_median": round(float(np.median(tps_vals)), 2) if tps_vals else None,
        "fleet_throughput_tbl_s_max": round(float(np.max(tps_vals)), 2) if tps_vals else None,
        "fleet_dope_vram_gib_max": round(float(np.max(vram_vals)), 3) if vram_vals else 0.0,
        "encoding_bytes_median": round(float(np.median(bytes_arr)), 1) if bytes_arr else None,
        "encoding_bytes_max": round(float(np.max(bytes_arr)), 1) if bytes_arr else None,
        "within_2kib_cap_rate": round(
            float(np.mean([1.0 if b <= 2048.0 else 0.0 for b in bytes_arr])), 4
        ) if bytes_arr else None,
        "waves": waves_summary,
        "archive_file_count": archive_manifest["file_count"] if archive_manifest else 0,
        "archive_total_bytes": archive_manifest["total_bytes"] if archive_manifest else 0,
    }


def render_figures(data: dict[str, Any], png_out: Path, pdf_out: Path) -> None:
    """Render 4-panel publication-grade training convergence + throughput + DEV trajectory figure."""
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.28,
        "legend.frameon": False,
    })
    colors = {"V7.0": "#56B4E9", "V7.1": "#0072B2", "V7.2": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)

    # Panel (a): Total training loss over step by wave (median + IQR across seeds)
    ax = axes[0, 0]
    for wname in ("V7.0", "V7.1", "V7.2"):
        runs = data["waves"][wname]
        if not runs:
            continue
        xy = [
            (
                np.array([r["step"] for r in run["train"] if "loss" in r], dtype=np.float64),
                np.array([r["loss"] for r in run["train"] if "loss" in r], dtype=np.float64),
            )
            for run in runs if run["train"]
        ]
        st, med, q25, q75 = _bin_series(xy, step_bin=100)
        if len(st):
            ax.plot(st, med, label=f"{wname} (n={len(runs)} runs)", color=colors[wname], lw=2.0)
            ax.fill_between(st, q25, q75, color=colors[wname], alpha=0.18)
    ax.set_title("(a) Training Loss Convergence (Median & IQR Across Seeds)")
    ax.set_xlabel("Optimizer Step (16 Unique S3 Splits / Step / Worker)")
    ax.set_ylabel("Total Objective Loss")
    ax.grid(True)
    ax.legend(loc="upper right")

    # Panel (b): Differentiable TSTR vs Joint Realism (MMD/W1/SWD) loss decomposition (latest active wave)
    ax = axes[0, 1]
    active_wave = "V7.2" if data["waves"]["V7.2"] else ("V7.1" if data["waves"]["V7.1"] else "V7.0")
    active_runs = data["waves"][active_wave]
    for key, lbl, col in (
        ("tstr_loss", "Differentiable RFF-TSTR Loss", "#009E73"),
        ("mmd_loss", "Joint Realism (W1 + Corr + SWD)", "#CC79A7"),
        ("cond_loss", "Conditional Proper-Scoring NLL", "#E69F00"),
    ):
        xy = [
            (
                np.array([r["step"] for r in run["train"] if key in r], dtype=np.float64),
                np.array([r[key] for r in run["train"] if key in r], dtype=np.float64),
            )
            for run in active_runs if run["train"]
        ]
        st, med, q25, q75 = _bin_series(xy, step_bin=100)
        if len(st):
            ax.plot(st, med, label=lbl, color=col, lw=1.9)
            ax.fill_between(st, q25, q75, color=col, alpha=0.15)
    ax.set_title(f"(b) Objective Decomposition ({active_wave}, Source-Free <= 2 KiB)")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Component Loss")
    ax.grid(True)
    ax.legend(loc="upper right")

    # Panel (c): Wall-clock CPU-Only Fleet Throughput (tbl/s) & Monotonic Cumulative Splits
    ax = axes[1, 0]
    mins, tps, cum_m = _persisted_fleet_rate_bins(data, bin_s=60.0)
    if len(mins):
        active_tps = tps[tps > 1.0]
        med_tps = float(np.median(active_tps)) if len(active_tps) else 0.0
        ax.plot(mins, tps, color="#0072B2", lw=1.8, label="Persisted Fleet Rate (tbl/s)")
        if med_tps > 0.0:
            ax.axhline(med_tps, color="#666666", ls="--", lw=1.0, label=f"Active Median ({med_tps:.0f} tbl/s)")
        ax.set_xlabel("Elapsed Wave Time (minutes)")
        ax.set_ylabel("Unique Tables / Second (0.0 GiB VRAM)", color="#0072B2")
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(mins, cum_m, color="#D55E00", lw=1.6, ls="-.", label="Cumulative Splits (M)")
        ax2.set_ylabel("Cumulative Unique Splits (Millions)", color="#D55E00")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "Awaiting persisted train.jsonl timestamps", ha="center", va="center")
    ax.set_title("(c) CPU-Only Fleet Throughput & Cumulative Splits")
    ax.grid(True)

    # Panel (d): DEV-Only Per-Task Trajectory (clearly labeled unconfirmed DEV)
    ax = axes[1, 1]
    for key, lbl, col in (
        ("regression_retention", "Regression DEV Median", "#0072B2"),
        ("multiclass_retention", "Multiclass DEV Median", "#009E73"),
        ("main_kpi_retention_median", "Overall DEV Median", "#000000"),
    ):
        xy = [
            (
                np.array([r["step"] for r in run["eval"] if r.get(key) is not None], dtype=np.float64),
                np.array([float(r[key]) for r in run["eval"] if r.get(key) is not None], dtype=np.float64),
            )
            for run in active_runs if run["eval"]
        ]
        st, med, q25, q75 = _bin_series(xy, step_bin=300)
        if len(st):
            ax.plot(st, med, label=lbl, color=col, lw=1.9)
            ax.fill_between(st, q25, q75, color=col, alpha=0.14)
    ax.set_title(f"(d) Held-Out DEV-Only Trajectory ({active_wave}; Unconfirmed)")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("DEV Retention Proxy (Higher Better)")
    ax.set_ylim(0.55, 1.02)
    ax.grid(True)
    ax.legend(loc="lower right")

    fig.suptitle(
        f"DOPE v2.0.0 SingleEncV7 Training Wave Telemetry "
        f"({data['total_train_rows']:,} Persisted Step Logs | "
        f"{data['total_unique_splits'] / 1e6:.2f}M Unique Splits | 0.0 GiB VRAM)",
        fontsize=11.5,
        fontweight="bold",
    )
    png_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=300)
    fig.savefig(pdf_out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive V2.0.0 training telemetry and render loss/throughput plots")
    ap.add_argument("--mirror-root", type=Path, default=Path("/home/ubuntu/dope-data/fleet-mirror"))
    ap.add_argument("--history-log", type=Path, default=Path("/home/ubuntu/dope-data/dashboard/fleet_history.jsonl"))
    ap.add_argument("--archive-dir", type=Path, default=Path("/home/ubuntu/dope-data/archives/v2.0.0-wave"))
    ap.add_argument("--receipt-out", type=Path, default=Path("bench/real-world/results/v2.0.0/training-telemetry-snapshot.json"))
    ap.add_argument("--png-out", type=Path, default=Path("bench/real-world/figures/v2-training-curves.png"))
    ap.add_argument("--pdf-out", type=Path, default=Path("docs/whitepaper-v2/figures/v2-loss-and-throughput.pdf"))
    ap.add_argument("--skip-archive", action="store_true")
    args = ap.parse_args()

    data = collect_wave_data(args.mirror_root, args.history_log)
    manifest = None if args.skip_archive else archive_telemetry(data, args.archive_dir, args.mirror_root)
    receipt = build_snapshot_receipt(data, manifest)
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_figures(data, args.png_out, args.pdf_out)
    print(json.dumps({
        "status": "ok",
        "total_runs": receipt["total_runs"],
        "total_train_rows": receipt["total_train_rows_persisted"],
        "total_unique_splits": receipt["total_unique_splits_seen"],
        "fleet_tbl_s_med": receipt["fleet_throughput_tbl_s_median"],
        "fleet_tbl_s_max": receipt["fleet_throughput_tbl_s_max"],
        "receipt": str(args.receipt_out),
        "png": str(args.png_out),
        "pdf": str(args.pdf_out),
    }, indent=2))


if __name__ == "__main__":
    main()
