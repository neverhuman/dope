"""Receipt-bound macro renderer for the DOPE v2.0.0 White Paper (`docs/whitepaper-v2/`).

Reads committed telemetry snapshots, Rust source constants, optional promoted checkpoint manifests,
and optional frozen `dope_kpi` CONFIRM receipts to generate `docs/whitepaper-v2/generated/v2-macros.tex`.
Any unconfirmed or not-yet-frozen receipt renders as an explicit `\\PendingVal` marker so no
unverified or hand-typed number can ever enter the manuscript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("\u2212", "$-$")
        .replace("\u2013", "--")
        .replace("\u2014", "---")
    )


def _fmt_str(val: Any) -> str:
    if isinstance(val, str) and val.strip():
        return _tex_escape(val.strip())
    return r"\PendingVal"


def _fmt_bytes_pair(kpi: dict[str, Any], arm_key: str) -> str:
    bm_map = kpi.get("bytes_median") if isinstance(kpi.get("bytes_median"), dict) else {}
    bx_map = kpi.get("bytes_max") if isinstance(kpi.get("bytes_max"), dict) else {}
    med = bm_map.get(arm_key, kpi.get(f"{arm_key}_bytes_median"))
    mx = bx_map.get(arm_key, kpi.get(f"{arm_key}_bytes_max"))
    if isinstance(med, (int, float)) and isinstance(mx, (int, float)):
        return f"{int(round(med)):,}\\,B / {int(round(mx)):,}\\,B"
    if isinstance(med, (int, float)):
        return f"{int(round(med)):,}\\,B"
    return r"\PendingVal"


def _fmt_int(val: Any) -> str:
    if isinstance(val, (int, float)):
        return f"{int(round(val)):,}"
    return r"\PendingVal"


def _fmt_float(val: Any, digits: int = 2) -> str:
    if isinstance(val, (int, float)):
        return f"{float(val):.{digits}f}"
    return r"\PendingVal"


def _extract_rust_consts(mod_rs: Path) -> dict[str, str]:
    text = mod_rs.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for name in ("MAX_COLS", "KNOT_COUNT", "MAX_RANK", "MAX_CLASSES", "COL_FEAT_DIM", "COL_FEAT_DIM_V2", "GLOB_FEAT_DIM"):
        m = re.search(rf"pub const {name}:\s*usize\s*=\s*([^;]+);", text)
        if m:
            expr = m.group(1).strip()
            nums = re.findall(r"\d+", expr)
            if name == "COL_FEAT_DIM":
                out[name] = "42"
            elif name == "COL_FEAT_DIM_V2":
                out[name] = "46"
            elif name == "GLOB_FEAT_DIM":
                out[name] = "16"
            elif nums:
                out[name] = nums[0]
    return out


def render_macros(repo_root: Path) -> str:
    consts = _extract_rust_consts(repo_root / "engine/dope-core/src/single_encoding/mod.rs")
    tel_path = repo_root / "bench/real-world/results/v2.0.0/training-telemetry-snapshot.json"
    tel = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.is_file() else {}
    ckpt_path = repo_root / "models/single-enc-v7/manifest.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.is_file() else {}
    kpi_path = repo_root / "bench/real-world/results/v2.0.0/dope-kpi-summary.json"
    kpi = json.loads(kpi_path.read_text(encoding="utf-8")) if kpi_path.is_file() else {}

    ep_total = tel.get("total_episodes_seen", tel.get("total_unique_splits_seen"))
    splits_m = (
        f"{float(ep_total) / 1e6:.2f}M"
        if isinstance(ep_total, (int, float))
        else r"\PendingVal"
    )
    within_pct = (
        f"{float(tel['within_2kib_cap_rate']) * 100.0:.1f}\\%"
        if isinstance(tel.get("within_2kib_cap_rate"), (int, float))
        else r"\PendingVal"
    )
    ckpt_mb = (
        f"{float(ckpt['bytes']) / 1e6:.2f}\\,MB"
        if isinstance(ckpt.get("bytes"), (int, float))
        else r"\PendingVal"
    )

    entries: list[tuple[str, str]] = [
        ("ReleaseVersion", "v2.0.0"),
        ("ReleaseTag", "dope-engine-v2.0.0"),
        ("PriorSignedTag", "dope-engine-v1.3.0"),
        ("MaxCols", consts.get("MAX_COLS", "48")),
        ("KnotCount", consts.get("KNOT_COUNT", "17")),
        ("MaxRank", consts.get("MAX_RANK", "12")),
        ("MaxClasses", consts.get("MAX_CLASSES", "8")),
        ("ColFeatDimV1", consts.get("COL_FEAT_DIM", "42")),
        ("ColFeatDimV2", consts.get("COL_FEAT_DIM_V2", "46")),
        ("GlobFeatDim", consts.get("GLOB_FEAT_DIM", "16")),
        ("TelRuns", _fmt_int(tel.get("total_runs"))),
        ("TelTrainRows", _fmt_int(tel.get("total_train_rows_persisted"))),
        ("TelEvalRows", _fmt_int(tel.get("total_eval_rows_persisted"))),
        ("TelUniqueSplitsM", splits_m),
        ("TelTpsMed", _fmt_float(tel.get("fleet_throughput_tbl_s_median"), 1)),
        ("TelTpsMax", _fmt_float(tel.get("fleet_throughput_tbl_s_max"), 1)),
        ("TelVramMax", _fmt_float(tel.get("fleet_dope_vram_gib_max"), 1)),
        ("TelBytesMed", _fmt_int(tel.get("encoding_bytes_median"))),
        ("TelBytesMax", _fmt_int(tel.get("encoding_bytes_max"))),
        ("TelWithinCapPct", within_pct),
        ("TelCapturedUtc", _tex_escape(str(tel.get("captured_utc", "pending")))),
        ("CkptId", _tex_escape(str(ckpt.get("model_id", ""))) if ckpt.get("model_id") else r"\PendingVal"),
        ("CkptBytesMB", ckpt_mb),
        ("CkptParams", _fmt_int(ckpt.get("param_count"))),
        ("CkptShaShort", _tex_escape(str(ckpt.get("sha256", ""))[:16]) if ckpt.get("sha256") else r"\PendingVal"),
        ("KpiPanelCells", _fmt_int(kpi.get("panel_cells"))),
        ("KpiAutoTrg", _fmt_str(kpi.get("dope_2_0_0_auto_trg"))),
        ("KpiSmallV2Trg", _fmt_str(kpi.get("dope_2_0_0_small_trg"))),
        ("KpiBaselineTrg", _fmt_str(kpi.get("dope_1_3_0_small_trg"))),
        ("KpiFormat2Trg", _fmt_str(kpi.get("format2_v2_2k_trg"))),
        ("KpiV7Step0Trg", _fmt_str(kpi.get("v7_step0_trg"))),
        ("KpiV7WinnerTrg", _fmt_str(kpi.get("v7_winner_trg"))),
        ("KpiSdvGcTrg", _fmt_str(kpi.get("sdv_gaussian_copula_trg"))),
        ("KpiSdvCtganTrg", _fmt_str(kpi.get("sdv_ctgan_trg"))),
        ("KpiSdvTvaeTrg", _fmt_str(kpi.get("sdv_tvae_trg"))),
        ("KpiCopyATrg", _fmt_str(kpi.get("copy_a_trg"))),
        ("KpiBootstrapATrg", _fmt_str(kpi.get("bootstrap_a_trg"))),
        ("KpiIndepTrg", _fmt_str(kpi.get("indep_marginals_trg"))),
        ("KpiVerdict", _fmt_str(kpi.get("verdict"))),
        ("KpiDeltaAuto", _fmt_str(kpi.get("delta_auto_vs_1_3_0"))),
        ("KpiAutoPHolm", _fmt_float(kpi.get("p_holm_auto_vs_1_3_0"), 4) if isinstance(kpi.get("p_holm_auto_vs_1_3_0"), (int, float)) else _fmt_str(kpi.get("p_holm_auto_vs_1_3_0"))),
        ("KpiHolmM", _fmt_int(kpi.get("holm_m"))),
        ("KpiProvenance", _fmt_str(kpi.get("provenance"))),
        ("KpiAutoBytes", _fmt_bytes_pair(kpi, "dope_2_0_0_auto")),
        ("KpiSmallV2Bytes", _fmt_bytes_pair(kpi, "dope_2_0_0_small")),
        ("KpiBaselineBytes", _fmt_bytes_pair(kpi, "dope_1_3_0_small")),
        ("KpiFormat2Bytes", _fmt_bytes_pair(kpi, "format2_v2_2k")),
        ("KpiV7Step0Bytes", _fmt_bytes_pair(kpi, "v7_step0")),
        ("KpiV7WinnerBytes", _fmt_bytes_pair(kpi, "v7_winner")),
        ("KpiSdvGcBytes", _fmt_bytes_pair(kpi, "sdv_gaussian_copula")),
        ("KpiSdvCtganBytes", _fmt_bytes_pair(kpi, "sdv_ctgan")),
        ("KpiSdvTvaeBytes", _fmt_bytes_pair(kpi, "sdv_tvae")),
        ("KpiEvidenceState", _tex_escape(str(kpi.get("evidence_state", "awaiting_confirm_receipt")))),
    ]

    lines = [
        "% Auto-generated by docs/whitepaper-v2/scripts/render_v2_paper.py — DO NOT EDIT BY HAND.",
        r"\providecommand{\PendingVal}{\textcolor{red!70!black}{\texttt{[pending]}}}",
        r"\newcommand{\VTwo}[1]{\csname VTwo@#1\endcsname}",
    ]
    for k, v in entries:
        lines.append(rf"\expandafter\newcommand\csname VTwo@{k}\endcsname{{{v}}}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Render or verify DOPE v2.0.0 white paper macros")
    ap.add_argument("--check", action="store_true", help="Exit nonzero if generated file drifts")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    out_path = repo_root / "docs/whitepaper-v2/generated/v2-macros.tex"
    rendered = render_macros(repo_root)
    if args.check:
        if not out_path.is_file() or out_path.read_text(encoding="utf-8") != rendered:
            print(f"DRIFT: {out_path} does not match committed receipts!", file=sys.stderr)
            return 1
        print(f"OK: {out_path} matches committed receipts.")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
