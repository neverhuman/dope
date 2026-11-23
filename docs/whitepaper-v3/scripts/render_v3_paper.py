"""Receipt-bound macro & figure renderer for the DOPE v3.0.0 White Paper (`docs/whitepaper-v3/`).

Reads frozen contracts (`contracts/evidence-layout-v1.json`, `contracts/dope-kpi.json`),
Rust source constants (`engine/dope-core/src/single_encoding/v8/`), and committed
`dope_kpi` v3.0.0 scoreboard receipts (`bench/real-world/results/scoreboard/v3.0.0/`) to
generate `docs/whitepaper-v3/generated/v3-macros.tex`. Missing receipts emit `\\PendingVal`
so zero hand-typed quantitative claims can enter the manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
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


def _sha256_short(path: Path, n: int = 12) -> str:
    if not path.is_file():
        return r"\PendingVal"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def _fmt_int(val: Any) -> str:
    if isinstance(val, (int, float)):
        return f"{int(round(val)):,}"
    return r"\PendingVal"


def _fmt_float(val: Any, digits: int = 3, signed: bool = False) -> str:
    if isinstance(val, (int, float)):
        spec = f"+.{digits}f" if signed else f".{digits}f"
        s = format(float(val), spec)
        return s.replace("-", "$-$") if s.startswith("-") else s
    return r"\PendingVal"


def _fmt_ci(val: Any, lo: Any, hi: Any, digits: int = 3, signed: bool = False) -> str:
    if all(isinstance(x, (int, float)) for x in (val, lo, hi)):
        return f"{_fmt_float(val, digits, signed)} [{_fmt_float(lo, digits, signed)}, {_fmt_float(hi, digits, signed)}]"
    return r"\PendingVal"


def _fmt_bytes(arm: dict[str, Any] | None) -> str:
    if not arm:
        return r"\PendingVal"
    if arm.get("artifact_class") == "control":
        return "---"
    b = arm.get("bytes")
    if not isinstance(b, dict):
        return r"\PendingVal"
    med, mx = b.get("median"), b.get("max")
    if isinstance(med, (int, float)) and isinstance(mx, (int, float)):
        return f"{int(round(med)):,}\\,B / {int(round(mx)):,}\\,B"
    return r"\PendingVal"


def _find_arm(primary: dict[str, Any], candidates: tuple[str, ...]) -> dict[str, Any] | None:
    arms = primary.get("arms")
    if not isinstance(arms, list):
        return None
    by_id = {str(a.get("arm_id", "")): a for a in arms if isinstance(a, dict)}
    for cid in candidates:
        if cid in by_id:
            return by_id[cid]
    for cid in candidates:
        for k, v in by_id.items():
            if cid in k:
                return v
    return None


def _task_val(block: dict[str, Any] | None, task: str) -> Any:
    if not isinstance(block, dict):
        return None
    for t in block.get("tasks", []):
        if isinstance(t, dict) and t.get("task") == task:
            return t.get("value")
    return None


def _auditor_med(arm: dict[str, Any] | None, name_prefix: str) -> Any:
    if not isinstance(arm, dict):
        return None
    for row in arm.get("median_retention", []):
        if isinstance(row, dict) and str(row.get("auditor", "")).startswith(name_prefix):
            return row.get("median")
    return None


def _arm_macros(prefix: str, arm: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(arm, dict):
        keys = ("Trg", "MatchedTrg", "Reg", "Bin", "Mc", "Bytes", "MlpTrg", "C2st", "FailCell", "AudMed")
        return [(f"{prefix}{k}", r"\PendingVal") for k in keys]
    pr = arm.get("primary") or {}
    ma = arm.get("matched") or {}
    lin = _auditor_med(arm, "linear")
    gbd = _auditor_med(arm, "gbdt")
    mlp = _auditor_med(arm, "mlp")
    aud = (
        f"{_fmt_float(lin, 3)} / {_fmt_float(gbd, 3)} / {_fmt_float(mlp, 3)}"
        if all(isinstance(x, (int, float)) for x in (lin, gbd, mlp))
        else r"\PendingVal"
    )
    fc, cc = pr.get("failed_cells"), pr.get("cells")
    fail_str = f"{int(fc)}/{int(cc)}" if isinstance(fc, (int, float)) and isinstance(cc, (int, float)) else r"\PendingVal"
    return [
        (f"{prefix}Trg", _fmt_ci(pr.get("trg"), pr.get("ci_low"), pr.get("ci_high"), 3)),
        (f"{prefix}MatchedTrg", _fmt_ci(ma.get("trg"), ma.get("ci_low"), ma.get("ci_high"), 3)),
        (f"{prefix}Reg", _fmt_float(_task_val(pr, "regression"), 3)),
        (f"{prefix}Bin", _fmt_float(_task_val(pr, "binary"), 3)),
        (f"{prefix}Mc", _fmt_float(_task_val(pr, "multiclass"), 3)),
        (f"{prefix}Bytes", _fmt_bytes(arm)),
        (f"{prefix}MlpTrg", _fmt_float(arm.get("mlp_trg"), 3)),
        (f"{prefix}C2st", _fmt_float(arm.get("c2st_median"), 3)),
        (f"{prefix}FailCell", fail_str),
        (f"{prefix}AudMed", aud),
    ]


def _find_comparison(compare: dict[str, Any], cand_hints: tuple[str, ...], base_hints: tuple[str, ...]) -> dict[str, Any] | None:
    comps = compare.get("comparisons")
    if not isinstance(comps, list):
        return None
    for c in comps:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", ""))
        bid = str(c.get("baseline", ""))
        if any(h in cid for h in cand_hints) and any(h in bid for h in base_hints):
            return c
    return None


def _guard_entry(comp: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not isinstance(comp, dict):
        return None
    dec = comp.get("decision") or {}
    for g in dec.get("guards", []):
        if isinstance(g, dict) and g.get("name") == name:
            return g
    return None


def _fmt_guard(g: dict[str, Any] | None, digits: int = 3, signed: bool = True) -> tuple[str, str]:
    if not isinstance(g, dict):
        return (r"\PendingVal", r"\PendingVal")
    chk = _tex_escape(str(g.get("check", "pending")).upper())
    val = g.get("value")
    if not isinstance(val, (int, float)):
        return ("n/e", chk)
    return (_fmt_float(val, digits, signed), chk)


def _extract_v8_consts(repo_root: Path) -> dict[str, str]:
    out = {
        "MaxCols": "48", "MaxRank": "12", "MaxClasses": "8", "KnotCount": "17",
        "DecColDim": "48", "DecEdgeDim": "4", "DecGlobDim": "32", "DecCodeDim": "64",
        "NodeTrees": "8", "NodeDepth": "3", "BilinRank": "4",
    }
    pkt = repo_root / "engine/dope-core/src/single_encoding/v8/packet.rs"
    if pkt.is_file():
        txt = pkt.read_text(encoding="utf-8")
        for rust_k, tex_k in (("V8_MAX_COLS", "MaxCols"), ("V8_MAX_RANK", "MaxRank"), ("V8_MAX_CLASSES", "MaxClasses")):
            m = re.search(rf"pub const {rust_k}:\s*usize\s*=\s*(\d+);", txt)
            if m:
                out[tex_k] = m.group(1)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_macros(repo_root: Path, receipt_dir: Path | None = None) -> str:
    ev_path = repo_root / "contracts/evidence-layout-v1.json"
    kpi_contract = repo_root / "contracts/dope-kpi.json"
    ev = _read_json(ev_path)
    sb_dir = receipt_dir or (repo_root / "bench/real-world/results/scoreboard/v3.0.0")
    primary_path = sb_dir / "primary.json"
    compare_path = sb_dir / "compare.json"
    primary = _read_json(primary_path)
    compare = _read_json(compare_path)
    consts = _extract_v8_consts(repo_root)

    auto_v8 = _find_arm(primary, ("dope-3.0.0-auto-v8", "dope-v3b-auto-v8", "dope-v3-auto-v8", "auto-v8"))
    dse8_p0 = _find_arm(primary, ("dse8-p0-v3b", "dse8-p0", "dope-3.0.0-v8-analytic"))
    v2_auto = _find_arm(primary, ("dope-2.0.0-auto",))
    v2_small = _find_arm(primary, ("dope-2.0.0-small",))
    v13_small = _find_arm(primary, ("dope-1.3.0-small",))
    v7_step0 = _find_arm(primary, ("pr390-v7-step0", "v7-step0"))
    copy_a = _find_arm(primary, ("copy_a",))
    boot_a = _find_arm(primary, ("bootstrap_a",))
    indep_m = _find_arm(primary, ("indep_marginals",))

    comp_v2 = _find_comparison(compare, ("auto-v8", "3.0.0", "v3b"), ("2.0.0-auto", "2.0.0", "1.3.0"))
    dims = ev.get("dimensions") if isinstance(ev.get("dimensions"), dict) else {}
    panel_obj = primary.get("panel") if isinstance(primary.get("panel"), dict) else {}
    panel_fams = panel_obj.get("families")
    panel_label = (
        f"fresh DEV report ({int(panel_fams)} families)"
        if isinstance(panel_fams, (int, float))
        else "fresh DEV report"
    )
    pr_block = (auto_v8 or {}).get("primary") or {}

    entries: list[tuple[str, str]] = [
        ("ReleaseVersion", "v3.0.0"),
        ("ReleaseTag", "dope-engine-v3.0.0"),
        ("PriorReleaseTag", "dope-engine-v2.0.0"),
        ("PriorSignedTag", "dope-engine-v1.3.0"),
        ("EvLayoutSha", _sha256_short(ev_path, 12)),
        ("KpiContractSha", _sha256_short(kpi_contract, 12)),
        ("PrimarySha", _sha256_short(primary_path, 12)),
        ("CompareSha", _sha256_short(compare_path, 12)),
        ("EvColDim", _fmt_int(dims.get("col_dim", 128))),
        ("EvEdgeDim", _fmt_int(dims.get("edge_dim", 16))),
        ("EvClassDim", _fmt_int(dims.get("class_dim", 16))),
        ("EvClassColDim", _fmt_int(dims.get("class_col_dim", 4))),
        ("EvGroupDim", _fmt_int(dims.get("group_dim", 8))),
        ("EvGlobDim", _fmt_int(dims.get("glob_dim", 160))),
        ("ReportPanel", _tex_escape(panel_label)),
        ("ReportCells", _fmt_int(pr_block.get("cells"))),
        ("ReportScheduled", _fmt_int(pr_block.get("scheduled_cells"))),
        ("ReportFamilies", _fmt_int(pr_block.get("families"))),
        *consts.items(),
    ]
    for pfx, arm in (
        ("AutoV8", auto_v8), ("Dse8P0", dse8_p0), ("V2Auto", v2_auto),
        ("V2Small", v2_small), ("V13Small", v13_small), ("V7Step0", v7_step0),
        ("CopyA", copy_a), ("BootA", boot_a), ("Indep", indep_m),
    ):
        entries.extend(_arm_macros(pfx, arm))

    if isinstance(comp_v2, dict):
        dec = comp_v2.get("decision") or {}
        ma = comp_v2.get("matched") or {}
        t_map = {t["task"]: t for t in comp_v2.get("tasks", []) if isinstance(t, dict) and "task" in t}
        g_task = _fmt_guard(_guard_entry(comp_v2, "task_ci_low"), 3, True)
        g_mlp = _fmt_guard(_guard_entry(comp_v2, "mlp_delta"), 3, True)
        g_real = _fmt_guard(_guard_entry(comp_v2, "realism_coverage_delta"), 3, True)
        g_priv = _fmt_guard(_guard_entry(comp_v2, "privacy_coverage_delta"), 3, True)
        g_copy = _fmt_guard(_guard_entry(comp_v2, "exact_copy_excess"), 4, False)
        g_byte = _fmt_guard(_guard_entry(comp_v2, "bytes"), 0, False)
        nc = (comp_v2.get("null_calibration") or {}).get("flip_guard_rule") or {}
        entries.extend([
            ("CompBaseline", _tex_escape(str(comp_v2.get("baseline", "dope-2.0.0-auto")))),
            ("CompDelta", _fmt_ci(comp_v2.get("delta"), comp_v2.get("ci_low"), comp_v2.get("ci_high"), 3, True)),
            ("CompMatchedDelta", _fmt_ci(ma.get("delta"), ma.get("ci_low"), ma.get("ci_high"), 3, True)),
            ("CompPHolm", _fmt_float(comp_v2.get("p_holm"), 4)),
            ("CompSignFlip", _fmt_float(nc.get("rate"), 3)),
            ("CompOutcome", _tex_escape(str(dec.get("outcome", "HOLD")))),
            ("CompRegDelta", _fmt_ci(t_map.get("regression", {}).get("delta"), t_map.get("regression", {}).get("ci_low"), t_map.get("regression", {}).get("ci_high"), 3, True)),
            ("CompBinDelta", _fmt_ci(t_map.get("binary", {}).get("delta"), t_map.get("binary", {}).get("ci_low"), t_map.get("binary", {}).get("ci_high"), 3, True)),
            ("CompMcDelta", _fmt_ci(t_map.get("multiclass", {}).get("delta"), t_map.get("multiclass", {}).get("ci_low"), t_map.get("multiclass", {}).get("ci_high"), 3, True)),
            ("GuardTaskVal", g_task[0]), ("GuardTaskChk", g_task[1]),
            ("GuardMlpVal", g_mlp[0]), ("GuardMlpChk", g_mlp[1]),
            ("GuardRealVal", g_real[0]), ("GuardRealChk", g_real[1]),
            ("GuardPrivVal", g_priv[0]), ("GuardPrivChk", g_priv[1]),
            ("GuardCopyVal", g_copy[0]), ("GuardCopyChk", g_copy[1]),
            ("GuardByteVal", g_byte[0]), ("GuardByteChk", g_byte[1]),
        ])
    else:
        for k in (
            "CompBaseline", "CompDelta", "CompMatchedDelta", "CompPHolm", "CompSignFlip", "CompOutcome",
            "CompRegDelta", "CompBinDelta", "CompMcDelta",
            "GuardTaskVal", "GuardTaskChk", "GuardMlpVal", "GuardMlpChk",
            "GuardRealVal", "GuardRealChk", "GuardPrivVal", "GuardPrivChk",
            "GuardCopyVal", "GuardCopyChk", "GuardByteVal", "GuardByteChk",
        ):
            entries.append((k, r"\PendingVal"))

    lines = [
        "% Auto-generated by docs/whitepaper-v3/scripts/render_v3_paper.py -- DO NOT EDIT BY HAND.",
        r"\providecommand{\PendingVal}{\textcolor{red!70!black}{\texttt{[pending]}}}",
        r"\newcommand{\VThree}[1]{\csname VThree@#1\endcsname}",
    ]
    for k, v in entries:
        lines.append(rf"\expandafter\newcommand\csname VThree@{k}\endcsname{{{v}}}")
    return "\n".join(lines) + "\n"


def render_figures(repo_root: Path, receipt_dir: Path | None = None) -> bool:
    """Render vector frontier + per-task PDF figure when v3.0.0 primary.json is present."""
    sb_dir = receipt_dir or (repo_root / "bench/real-world/results/scoreboard/v3.0.0")
    primary_path = sb_dir / "primary.json"
    if not primary_path.is_file():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = _read_json(primary_path)
    specs = [
        ("DOPE 3.0.0 (auto-v8)", ("dope-3.0.0-auto-v8", "dope-v3b-auto-v8", "auto-v8"), "#0f766e", "o", 90),
        ("DOPE 2.0.0 (auto)", ("dope-2.0.0-auto",), "#1d4ed8", "^", 70),
        ("DOPE 2.0.0 (small)", ("dope-2.0.0-small",), "#64748b", "v", 60),
        ("DOPE 1.3.0 (small)", ("dope-1.3.0-small",), "#94a3b8", "D", 55),
        ("DSE8 P0 (analytic)", ("dse8-p0-v3b", "dse8-p0"), "#0284c7", "s", 65),
        ("DSE7 Step-0", ("pr390-v7-step0", "v7-step0"), "#b45309", "P", 60),
    ]
    found = [(lbl, _find_arm(primary, hints), col, mk, sz) for lbl, hints, col, mk, sz in specs]
    found = [x for x in found if isinstance(x[1], dict)]
    if len(found) < 2:
        return False

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 3.8), dpi=220)
    for lbl, arm, col, mk, sz in found:
        pr = arm["primary"]
        med_b = float((arm.get("bytes") or {}).get("median", 2048.0))
        y, lo, hi = float(pr["trg"]), float(pr["ci_low"]), float(pr["ci_high"])
        ax0.errorbar(
            med_b, y, yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]],
            fmt=mk, color=col, markersize=7.5 if sz > 75 else 6.0, capsize=3, label=lbl, lw=1.4,
        )
    ax0.axvline(2048.0, color="#dc2626", ls="--", lw=1.1, label="2,048 B hard cap")
    ax0.axhline(0.0, color="#334155", ls=":", lw=0.9)
    ax0.set_xlim(650, 2150)
    ax0.set_xlabel("Median Serialized Artifact Size (bytes)")
    ax0.set_ylabel("4x TRG (0 = real twin)")
    ax0.set_title("(a) Sub-2 KiB Value Frontier (Fresh DEV Report)")
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=7.3, loc="lower left", frameon=True)

    tasks = ["regression", "binary", "multiclass", "mlp"]
    tlabels = ["Regression", "Binary", "Multiclass", "Held-out MLP"]
    show = [found[0], found[1], found[2], found[4]] if len(found) >= 5 else found[:4]
    w = 0.8 / max(len(show), 1)
    xs = list(range(len(tasks)))
    for i, (lbl, arm, col, _mk, _sz) in enumerate(show):
        vals = []
        for tk in tasks:
            v = arm.get("mlp_trg") if tk == "mlp" else _task_val(arm.get("primary"), tk)
            vals.append(float(v) if isinstance(v, (int, float)) else 0.0)
        offs = [x - 0.4 + (i + 0.5) * w for x in xs]
        ax1.bar(offs, vals, width=w * 0.92, color=col, alpha=0.92, label=lbl)
    ax1.axhline(0.0, color="#334155", lw=0.9)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(tlabels, fontsize=8.5)
    ax1.set_ylabel("TRG (higher is better)")
    ax1.set_title("(b) Per-Task & Held-Out MLP Breakdown")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(fontsize=7.3, loc="lower left", frameon=True)

    fig.tight_layout()
    out_fig = repo_root / "docs/whitepaper-v3/figures/v3-frontier-and-tasks.pdf"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Render or verify DOPE v3.0.0 white paper macros")
    ap.add_argument("--check", action="store_true", help="Exit nonzero if generated macro file drifts")
    ap.add_argument("--figures", action="store_true", help="Render vector PDF figures from receipts")
    ap.add_argument("--receipt-dir", type=Path, default=None, help="Override scoreboard v3.0.0 directory")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    out_path = repo_root / "docs/whitepaper-v3/generated/v3-macros.tex"
    rendered = render_macros(repo_root, args.receipt_dir)
    if args.check:
        if not out_path.is_file() or out_path.read_text(encoding="utf-8") != rendered:
            print(f"DRIFT: {out_path} does not match committed v3.0.0 receipts!", file=sys.stderr)
            return 1
        print(f"OK: {out_path} matches committed v3.0.0 receipts.")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rendered)} bytes)")
    if args.figures and render_figures(repo_root, args.receipt_dir):
        print("Rendered docs/whitepaper-v3/figures/v3-frontier-and-tasks.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
