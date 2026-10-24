"""Render publication figures, tables, and structured summary from the DOPE-MLC-v2 S3 study receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def render_all(
    stage0_path: Path = Path("target/rapid-validation-20261023/stage0_rapid_validation_receipt.json"),
    train_path: Path = Path("target/s3-multilevel-training-20261023/s3_training_convergence_receipt.json"),
    eval_path: Path = Path("target/s3-multilevel-training-20261023/world_class_transfer_and_elo_receipt.json"),
    out_dir: Path = Path("target/s3-multilevel-training-20261023/figures"),
) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.edgecolor": "#d8d7d2",
        "axes.labelcolor": "#2b2a27",
        "xtick.color": "#2b2a27",
        "ytick.color": "#2b2a27",
        "axes.facecolor": "#fcfcfb",
        "figure.facecolor": "#fcfcfb",
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
    train = json.loads(train_path.read_text(encoding="utf-8"))
    ev = json.loads(eval_path.read_text(encoding="utf-8"))

    written: dict[str, str] = {}

    # 1. S3 19,079-split Training & Validation Convergence Curves across 3 Levels
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    history = train["convergence_history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["mean_total_loss"] for row in history]
    val_mean = [row["val"]["mean_total_loss"] for row in history]
    val_med = [row["val"]["medium"]["total_loss"] for row in history]
    val_high = [row["val"]["high"]["total_loss"] for row in history]
    val_ultra = [row["val"]["ultra"]["total_loss"] for row in history]

    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", lw=2.0, color="#62615d", linestyle="--", label="Train Loss (17,188 splits, 3-level avg)")
    ax.plot(epochs, val_mean, marker="s", lw=2.3, color="#0b0b0b", label="Val Loss (1,891 unseen dataset splits, avg)")
    ax.set_xlabel("Epoch (Full-S3 19,079 Regression + Binary Splits)")
    ax.set_ylabel("Multi-Task Objective Loss")
    ax.set_title("A. Training vs Held-Out Dataset Validation Convergence", fontweight="bold", fontsize=9.8)
    ax.legend(frameon=True, facecolor="#fcfcfb", edgecolor="#d8d7d2", fontsize=8.0)

    ax2 = axes[1]
    ax2.plot(epochs, val_med, marker="^", lw=2.1, color="#eb6834", label="Medium Tier (Macro Landmarks, 4 KiB / 4,096 coords)")
    ax2.plot(epochs, val_high, marker="o", lw=2.1, color="#2a78d6", label="High Tier (Larger Landmarks, 16 KiB / 8,192 coords)")
    ax2.plot(epochs, val_ultra, marker="*", markersize=8, lw=2.3, color="#15803d", label="Ultra Tier (Fine + Chatterjee ξ, 64 KiB / 10,240 coords)")
    ax2.set_xlabel("Epoch (Evaluated on Unseen Datasets Across All 6 Splits)")
    ax2.set_ylabel("Validation Objective Loss (Lower is Better)")
    ax2.set_title("B. Monotonic Multi-Level Hierarchy (Ultra < High < Medium)", fontweight="bold", fontsize=9.8)
    ax2.legend(frameon=True, facecolor="#fcfcfb", edgecolor="#d8d7d2", fontsize=7.8)

    fig.tight_layout()
    conv_png = out_dir / "s3_convergence_curves.png"
    conv_pdf = out_dir / "s3_convergence_curves.pdf"
    fig.savefig(conv_png, dpi=220)
    fig.savefig(conv_pdf)
    plt.close(fig)
    written["s3_convergence_png"] = str(conv_png)
    written["s3_convergence_pdf"] = str(conv_pdf)

    # 2. Stage 0 Meta-Feature Importance & Ablation across 19,190 S3 Splits
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    s0a = stage0["stage_0a"]
    top_feats = s0a["top_25_features_for_nonlinear_uplift"][:12][::-1]
    f_names = [row["feature"] for row in top_feats]
    f_vals = [row["importance_pct"] for row in top_feats]
    colors = ["#2a78d6" if not n.startswith("gain_") and not n.startswith("valid_") else "#8a8984" for n in f_names]
    axes[0].barh(f_names, f_vals, color=colors, height=0.68)
    axes[0].set_xlabel("CatBoost Meta-Feature Importance (%) across 19,190 Splits")
    axes[0].set_title("A. Top Landmark & Data Descriptor Drivers (Blue = Newly Unlocked)", fontweight="bold", fontsize=9.5)

    tiers = [
        ("Medium\n(Raw 9)", "Medium_Raw_GainsOnly"),
        ("Medium\n(+Rich)", "Medium_Rich_MacroDescriptors"),
        ("High\n(Raw 18)", "High_Raw_GainsOnly"),
        ("High\n(+Rich)", "High_Rich_LargerDescriptors"),
        ("Ultra\n(+Rich+ξ)", "Ultra_Rich_WithChatterjeeSpectrum"),
    ]
    x_pos = np.arange(len(tiers))
    abl = s0a["hierarchy_and_descriptor_ablation"]
    ceil_r2 = [abl[k]["catboost_ceiling_r2"] for _, k in tiers]
    uplift_r2 = [abl[k]["catboost_nonlin_uplift_r2"] for _, k in tiers]
    width = 0.36
    axes[1].bar(x_pos - width / 2, ceil_r2, width, color="#2a78d6", label="Signal Ceiling R² (Unseen Datasets)")
    axes[1].bar(x_pos + width / 2, uplift_r2, width, color="#eb6834", label="Nonlinear Uplift R² (Unseen Datasets)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([t[0] for t in tiers], fontsize=8.2)
    axes[1].set_ylim(0.65, 1.01)
    axes[1].set_ylabel("Out-of-Dataset Holdout R²")
    axes[1].set_title("B. Rapid CatBoost Probe Validation (19,190 S3 Splits)", fontweight="bold", fontsize=9.5)
    axes[1].legend(frameon=True, facecolor="#fcfcfb", edgecolor="#d8d7d2", fontsize=8.0, loc="lower right")

    fig.tight_layout()
    s0_png = out_dir / "stage0_feature_importance.png"
    s0_pdf = out_dir / "stage0_feature_importance.pdf"
    fig.savefig(s0_png, dpi=220)
    fig.savefig(s0_pdf)
    plt.close(fig)
    written["stage0_png"] = str(s0_png)
    written["stage0_pdf"] = str(s0_pdf)

    # 3. World-Class Model Transfer, Feature Importance Fidelity, and Elo Leaderboard
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    leaderboard = ev["elo_leaderboard_overall"][::-1]
    m_names = [row["method"] for row in leaderboard]
    m_elos = [row["elo"] for row in leaderboard]
    bar_colors = [
        "#15803d" if m == "dope_mlc_v2" else ("#2a78d6" if m in ("rows", "kernel") else "#8a8984")
        for m in m_names
    ]
    y_pos = np.arange(len(m_names))
    axes[0].barh(y_pos, m_elos, color=bar_colors, height=0.65)
    axes[0].axvline(1500.0, color="#62615d", linestyle="--", lw=1.0)
    for idx_y, val_x in enumerate(m_elos):
        axes[0].text(val_x + 6.0, idx_y, f"{val_x:.1f}", va="center", fontsize=8.3, fontweight="bold")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(m_names, fontsize=8.5)
    axes[0].set_xlim(min(1250.0, min(m_elos) - 60), max(m_elos) + 75)
    axes[0].set_xlabel("Bradley–Terry Elo Rating (Centered at 1500)")
    axes[0].set_title("A. Overall Multi-Auditor Transfer & Driver Elo Leaderboard", fontweight="bold", fontsize=9.5)

    auditors = ["catboost", "kumo_tabpfn", "lightgbm", "xgboost", "mlp", "linear"]
    aud_labels = ["CatBoost", "Tabular Kumo", "LightGBM", "XGBoost", "MLP", "Linear"]
    aud_ultra = ev["world_class_auditors_at_ultra"]
    x_aud = np.arange(len(auditors))
    w = 0.36
    ret_vals = [aud_ultra[a]["mean_retention_R"] for a in auditors]
    drv_vals = [aud_ultra[a]["mean_driver_fidelity_D"] for a in auditors]
    axes[1].bar(x_aud - w / 2, ret_vals, w, label="Holdout Retention R (Unseen Scaled D_test)", color="#15803d")
    axes[1].bar(x_aud + w / 2, drv_vals, w, label="Driver Attribution Fidelity D (Feature Importance)", color="#2a78d6")
    axes[1].set_xticks(x_aud)
    axes[1].set_xticklabels(aud_labels, fontsize=8.3)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].axhline(0.80, color="#62615d", linestyle=":", lw=1.0, label="D80 Driver Gate (0.80)")
    axes[1].set_ylabel("Mean Score Across Unseen Holdout Datasets")
    axes[1].set_title("B. DOPE-MLC-v2 (Ultra) Transfer & Feature Importance", fontweight="bold", fontsize=9.3)
    axes[1].legend(frameon=True, facecolor="#fcfcfb", edgecolor="#d8d7d2", fontsize=7.8, loc="lower right")

    fig.tight_layout()
    elo_png = out_dir / "world_class_elo_and_transfer.png"
    elo_pdf = out_dir / "world_class_elo_and_transfer.pdf"
    fig.savefig(elo_png, dpi=220)
    fig.savefig(elo_pdf)
    plt.close(fig)
    written["elo_png"] = str(elo_png)
    written["elo_pdf"] = str(elo_pdf)

    return written


if __name__ == "__main__":
    print(json.dumps(render_all(), indent=2))
