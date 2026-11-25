"""Receipt-bound renderer and checker for the DOPE v3.0.0 white paper (`docs/whitepaper-v3/`).

Every number in the manuscript is a `\\VThree{...}` macro written here from committed inputs:
the `dope_kpi` receipts under `bench/real-world/results/scoreboard/v3.0.0/`, the contracts
under `contracts/`, and constants of the Rust sources. Nothing falls back to a default: a
missing receipt, contract key or source constant is an error.

  render_v3_paper.py            write generated/{v3-macros.tex, v3-render-manifest.json}
  render_v3_paper.py --figures  also convert the methods hero SVG to figures/hero-v3-methods.pdf
  render_v3_paper.py --check    write nothing; fail on drift of any generated file, on a hero
                                PDF whose provenance does not match, on any digit typed into
                                the manuscript body, on a paper table that cites an earlier DOPE
                                release, and on a comparative claim of the prose that the receipts
                                contradict.

The paper's Tables 1 and 2 (generated/v3-{ranking,paired}-table.tex) are written by the Rust
`dope_kpi render-paper-tables`; this script records their SHA-256 and lints them.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_paper_final_look as final_mod  # noqa: E402
import v3_paper_methods as methods_mod  # noqa: E402
import v3_paper_tables as tables  # noqa: E402
from v3_paper_numbers import RenderError, ci, grouped, num, pct, short, sweep_macros, tt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SCOREBOARD = "bench/real-world/results/scoreboard/v3.0.0"
HERO_SVG = "bench/real-world/figures/hero-v3-comparison-light.svg"
HERO_PDF = "docs/whitepaper-v3/figures/hero-v3-methods.pdf"
GEN = "docs/whitepaper-v3/generated"
RUST_TABLES = (f"{GEN}/v3-ranking-table.tex", f"{GEN}/v3-paired-table.tex")
# The paper reports this release against third-party generators only (owner, 2026-11-24): no section
# may cite an earlier release's arms, the comparison with it, or its final look.
PRIOR_MACRO = re.compile(r"(Base|Old|PrevSmall|V7|Fl|Wide)[A-Z]|Delta|Guard|Flip|StatPass|Matched[DP]|PHolm|HolmM|Outcome|Comp[CFa]")
PRIOR_ARM = re.compile(r"dope-\d|pr\d{3}|\bv7\b", re.I)
OUT = {"macros": f"{GEN}/v3-macros.tex", "manifest": f"{GEN}/v3-render-manifest.json"}
HERO_EPOCH = "1795392000"  # 2026-11-23T00:00:00Z: fixes the PDF creation date of the conversion.
V8 = "engine/dope-core/src/single_encoding/v8"
RUST = {  # macro: (source, regex); the first group is the value.
    "DseHeaderBytes": (f"{V8}/planner.rs", r"pub const DSE8_HEADER_BYTES: usize = (\d+);"),
    "CrcBytes": (f"{V8}/planner.rs", r"pub const DSE8_CRC_BYTES: usize = (\d+);"),
    "TlvBytes": (f"{V8}/planner.rs", r"pub const TLV_HDR_BYTES: usize = (\d+);"),
    "MargBaseBytes": (f"{V8}/planner.rs", r"let marg_b = p \* \((\d+) \+ kl\)"),
    "LinFixedBytes": (f"{V8}/planner.rs", r"let lin_b = (\d+) \+ o_dim \* \(\d+ \+ p\);"),
    "LinPerOutBytes": (f"{V8}/planner.rs", r"let lin_b = \d+ \+ o_dim \* \((\d+) \+ p\);"),
    "EffectBytes": (f"{V8}/packet.rs", r"effects\.len\(\) \* (\d+)\)"),
    "PairBytes": (f"{V8}/packet.rs", r"pairs\.len\(\) \* (\d+)\)"),
    "TreeBytes": (f"{V8}/trees_v8.rs", r"pub const V8_TREE_WIRE_BYTES: usize = (\d+);"),
    "TreeLeaves": (f"{V8}/trees_v8.rs", r"pub leaves: \[f32; (\d+)\]"),
    "MaxTrees": (f"{V8}/trees_v8.rs", r"pub const V8_MAX_TREES: usize = (\d+);"),
    "EffectBins": (f"{V8}/target_v8.rs", r"pub bins8: \[f32; (\d+)\]"),
    "MaxEffects": (f"{V8}/target_v8.rs", r"pub const V8_MAX_EFFECTS: usize = (\d+);"),
    "MaxPairs": (f"{V8}/target_v8.rs", r"pub const V8_MAX_PAIRS: usize = (\d+);"),
    "MaxCols": (f"{V8}/scale_v8.rs", r"pub const V8_MAX_COLS: usize = (\d+);"),
    "MaxClasses": (f"{V8}/scale_v8.rs", r"pub const V8_MAX_CLASSES: usize = (\d+);"),
    "MaxRank": (f"{V8}/copula_regimes.rs", r"pub const V8_MAX_RANK: usize = (\d+);"),
    "MaxRegimes": (f"{V8}/copula_regimes.rs", r"pub const V8_MAX_REGIMES: usize = (\d+);"),
    "KnotCount": ("engine/dope-core/src/single_encoding/mod.rs", r"pub const KNOT_COUNT: usize = (\d+);"),
    "DpaHeaderBytes": ("engine/dope-core/src/auto/container.rs", r"pub const DPA1_HEADER_BYTES: usize = (\d+);"),
    "MinPacketRoom": ("engine/dope-core/src/auto/v8_dpa.rs", r"if packet_room < (\d+) \{"),
    "CliBudget": ("engine/dope-core/src/auto/v8.rs", r"budget: (\d+),"),
    "FoldVeto": ("engine/dope-core/src/auto/v8.rs", r"x >= a - ([0-9.]+) && y >= b - [0-9.]+"),
    "MinTestPerClass": ("engine/dope-core/src/auto/v8/folds.rs", r"any\(\|&c\| c < (\d+)\)"),
    "MinCheckGroups": ("engine/dope-core/src/auto/check.rs", r"pub const MIN_CHECK_GROUPS: usize = (\d+);"),
    "CheckSynthFactor": ("engine/dope-core/src/auto/check.rs", r"pub const CHECK_SYNTH_FACTOR: usize = (\d+);"),
    "SelectionMargin": ("engine/dope-core/src/auto/mod.rs", r"pub const SELECTION_MARGIN: f64 = ([0-9.]+);"),
    "SupportFloorK": ("engine/dope-core/src/evidence/primitives.rs", r"pub const SUPPORT_FLOOR_K: usize = (\d+);"),
    "TailMinDen": ("engine/dope-core/src/evidence/primitives.rs", r"\(n_obs as f64 \+ 1\.0\)\.max\((\d+)\.0\)\)\.min\("),
    "TailCap": ("engine/dope-core/src/evidence/primitives.rs", r"\.max\(\d+\.0\)\)\.min\(([0-9.]+)\)"),
}
LINT_ALLOW = (
    "SHA-256", "Format~1", r"-{}-format 1", "x86-64",
    "TabPFN-3.5", "TabPFN-3", "LimiX-2", "landmarks-native-v1",
    "evidence-layout-v1", "dope-kpi/v2",
)
# Inside math, a lone 0 or 1 is structure (N(0, I), 1/|T|, 1 - R_B), not a measured quantity.
LINT_MATH = re.compile(r"\$[^$]*\$|\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", re.S)
LINT_STRUCTURAL = re.compile(r"(?<![\d.])[01](?![\d.])")
LINT_STRIP = (
    r"\\VThree\{[^{}]*\}",
    r"\\(?:cite[a-z]*|ref|eqref|autoref|label|input|includegraphics|path|url|href|bibliography"
    r"|bibliographystyle)\*?(?:\[[^\]]*\])*\{[^{}]*\}",
    r"-?\d*\.?\d+(?:em|ex|pt|bp|mm|cm|in)\b",
    r"[\d.]*\\(?:linewidth|textwidth|columnwidth|baselineskip|relax)",
    r"[_^]\{?\d+\}?",
    r"[A-Za-z]+\d[A-Za-z0-9]*",
    r"\\fontsize\{[^{}]*\}\{[^{}]*\}",
    r"\\(?:vspace|hspace|setlength|addtolength|raisebox|rule|qrcode)\*?(?:\[[^\]]*\])?\{[^{}]*\}",
    r"\\(?:textsuperscript|makebox|parbox|minipage|fbox|colorbox)(?:\[[^\]]*\])?\{[^{}]*\}",
)
LINT_HIDE = ("tikzpicture", "tabularx", "tabular\\*", "tabular", "codebox")


def sha256(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes()).hexdigest()


def read_json(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        raise RenderError(f"missing input {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


def rust_constants() -> tuple[dict[str, str], set[str]]:
    out, files = {}, set()
    for name, (rel, pattern) in RUST.items():
        match = re.search(pattern, (REPO / rel).read_text(encoding="utf-8"))
        if not match:
            raise RenderError(f"{rel}: constant for {name} not found (pattern {pattern})")
        out[name] = match.group(1)
        files.add(rel)
    return out, files


def role_ids(primary: dict[str, Any], comp: dict[str, Any]) -> dict[str, str]:
    ids = [a["arm_id"] for a in primary["arms"]]
    base_version = re.match(r"dope-(\d+\.\d+\.\d+)-", comp["baseline"])
    if not base_version:
        raise RenderError(f"baseline {comp['baseline']} names no release")

    def one(pattern: str) -> str:
        hits = [i for i in ids if re.search(pattern, i)]
        if len(hits) != 1:
            raise RenderError(f"expected one arm matching {pattern}, found {hits}")
        return hits[0]
    return {"Cand": comp["id"], "Base": comp["baseline"], "Analytic": one(r"dse8"), "V7": one(r"v7"),
            "PrevSmall": one(rf"^dope-{re.escape(base_version.group(1))}-small$"),
            "OldSmall": one(rf"^dope-(?!{re.escape(base_version.group(1))})[\d.]+-small$"),
            "CopyA": "copy_a", "BootA": "bootstrap_a", "Indep": "indep_marginals",
            "LabelShuffle": "label_shuffle", "TwinC": "twin_c"}


SCORE_SCALE = 100  # Transfer Score = SCORE_SCALE x (1 + TRG), the README headline (owner-approved 2026-11-24)


def score(trg: float) -> float:
    """Transfer Score in points; SCORE_SCALE is a real twin of the fitting split."""
    return SCORE_SCALE * (1.0 + float(trg))


def arm_macros(role: str, arm: dict[str, Any]) -> list[tuple[str, str]]:
    p, m, b = arm["primary"], arm["matched"], arm["bytes"]
    out = [(f"{role}Id", tt(arm["arm_id"])), (f"{role}Trg", ci(p["trg"], p["ci_low"], p["ci_high"])),
           (f"{role}TrgPoint", num(p["trg"])), (f"{role}Matched", ci(m["trg"], m["ci_low"], m["ci_high"])),
           # Transfer Score = 100 x (1 + TRG), the README headline: a fixed relabel, same ranks, intervals x 100.
           (f"{role}Score", ci(score(p["trg"]), score(p["ci_low"]), score(p["ci_high"]), digits=1)),
           (f"{role}ScorePoint", num(score(p["trg"]), 1)), (f"{role}MatchedScorePoint", num(score(m["trg"]), 1)),
           (f"{role}Mlp", num(arm["mlp_trg"])), (f"{role}Failed", str(p["failed_cells"])),
           (f"{role}Cells", str(p["cells"])), (f"{role}NotApplicable", str(p["not_applicable_cells"]))]
    if arm.get("c2st_median") is not None:
        out.append((f"{role}C2st", num(arm["c2st_median"])))
    for task in p["tasks"]:
        out += [(f"{role}{task['task'].capitalize()}", num(task["value"])),
                (f"{role}{task['task'].capitalize()}Score", num(score(task["value"]), 1))]
    if b.get("median") is not None:
        out += [(f"{role}BytesMed", grouped(b["median"])), (f"{role}BytesMax", grouped(b["max"]))]
    for key, name in (("realism_coverage", "Realism"), ("privacy_coverage", "Privacy")):
        if arm.get(key) is not None:
            out.append((f"{role}{name}", num(arm[key])))
    for row in arm["median_retention"]:
        out.append((f"{role}Ret{row['auditor'].split('-')[0].capitalize()}", num(row["median"])))
    return out


def comparison_macros(comp: dict[str, Any], compare: dict[str, Any]) -> list[tuple[str, str]]:
    m, cal = comp["matched"], comp["null_calibration"]
    out = [("Delta", ci(comp["delta"], comp["ci_low"], comp["ci_high"], True)),
           ("DeltaPoint", num(comp["delta"], 3, True)), ("PHolm", num(comp["p_holm"])),
           ("DeltaPts", ci(SCORE_SCALE * comp["delta"], SCORE_SCALE * comp["ci_low"], SCORE_SCALE * comp["ci_high"], True, 1)),
           ("DeltaPtsPoint", num(SCORE_SCALE * comp["delta"], 1, True)), ("ScoreScale", str(SCORE_SCALE)),
           ("HolmM", str(comp["holm_m"])), ("Outcome", comp["decision"]["outcome"]),
           ("MatchedDelta", ci(m["delta"], m["ci_low"], m["ci_high"], True)),
           ("MatchedP", num(m["p_one_sided"])), ("MatchedMult", str(m["multiplier"])),
           ("CompCells", str(comp["cells"])), ("CompFamilies", str(comp["families"])),
           ("MlpCells", str(comp["mlp_cells"])), ("NoiseSd", short(compare["noise_sd"]["value"])),
           ("NullReplicates", grouped(cal["replicates"])),
           ("FlipRatePct", pct(cal["flip_guard_rule"]["rate"])),
           ("FlipHighPct", pct(cal["flip_guard_rule"]["wilson95_high"])),
           ("StatPassPct", pct(cal["statistical_pass"]["rate"])),
           ("StatPassLowPct", pct(cal["statistical_pass"]["wilson95_low"])),
           ("StatPassHighPct", pct(cal["statistical_pass"]["wilson95_high"]))]
    for t in comp["tasks"]:
        out.append((f"Delta{t['task'].capitalize()}", ci(t["delta"], t["ci_low"], t["ci_high"], True)))
    guards = comp["decision"]["guards"]
    for g in guards:
        name = "".join(w.capitalize() for w in tables.GUARD_NAMES.get(g["name"], g["name"]).split("_"))
        value = grouped(g["value"]) if g["name"] == "bytes" else num(g["value"], 3, g["name"] != "exact_copy_excess")
        out += [(f"Guard{name}", value), (f"Guard{name}Check", tables.CHECK_WORDS[g["check"]])]
    failed = sum(g["check"] == "fail" for g in guards)
    return out + [("GuardsFailed", str(failed)), ("GuardsTotal", str(len(guards)))]


def contract_macros(kpi: dict[str, Any], fid: dict[str, Any], layout: dict[str, Any]) -> list[tuple[str, str]]:
    st, pr, bs, nl = kpi["statistic"], kpi["promotion"], kpi["bootstrap"], kpi["null"]
    gates = fid["gates"]
    priv, real = gates["validity_and_empirical_privacy"], gates["realism_coverage"]
    tpr_key = next(k for k in priv if k.startswith("mia_tpr_at_"))
    if layout["support_floor_k"] != priv["support_floor_k"]:
        raise RenderError("evidence layout and fidelity contract disagree on the support floor")
    dims = layout["dims"]
    return [("ContractId", tt(kpi["contract_id"])), ("ContractStatus", kpi["status"]),
            ("ClipLow", short(st["clip_low"])), ("ClipHigh", short(st["clip_high"])),
            ("FailedValue", short(st["failed_value"])), ("FailedScore", num(score(st["failed_value"]), 1)),
            ("PrimaryMult", str(st["primary_multiplier"])),
            ("CiLevel", f"{round(100 * (bs['upper_quantile'] - bs['lower_quantile']))}\\%"),
            ("LowerPct", short(100 * bs["lower_quantile"])), ("UpperPct", short(100 * bs["upper_quantile"])),
            ("HolmAlpha", short(pr["holm_p_lt"])), ("MinDelta", short(pr["min_delta"])),
            ("CiLowGt", short(pr["ci_low_gt"])), ("TaskCiLowGt", short(pr["task_ci_low_gt"])),
            ("MlpDeltaGe", short(pr["mlp_delta_ge"])), ("RealismDeltaGt", short(pr["realism_coverage_delta_gt"])),
            ("PrivacyDeltaGt", short(pr["privacy_coverage_delta_gt"])), ("CopyLe", short(pr["exact_copy_excess_le"])),
            ("NullMaxRatePct", pct(nl["max_false_promote_rate"], 0)),
            ("BudgetBytes", grouped(kpi["bytes"]["headline"])),
            ("BudgetBytesPlain", str(int(kpi["bytes"]["headline"]))),
            ("BudgetKiB", short(kpi["bytes"]["headline"] / 1024)),
            ("C2stMax", short(real["c2st_auc_max"])), ("MargAucMax", short(real["marg_auc_max"])),
            ("RuleViolationMax", short(real["rule_violation_ratio_max"])), ("DcrMax", short(priv["dcr_share_max"])),
            ("MiaAucMax", short(priv["mia_auc_max"])), ("MiaTprMax", short(priv[tpr_key])),
            ("MiaFprPct", re.search(r"at_(\d+)pct", tpr_key).group(1) + r"\%"),
            ("SupportK", str(layout["support_floor_k"])), ("EvTiers", str(len(layout["tiers"]))),
            *[(f"Ev{k.title().replace('_', '')}Dim", str(v)) for k, v in dims.items()]]


def check_same_run(primary: dict[str, Any], other: dict[str, Any], name: str) -> None:
    for key in ("contract", "panel", "bootstrap"):
        if primary[key] != other[key]:
            raise RenderError(f"{name} and primary.json differ in {key}")
    for key in ("schedule_sha256", "realref_sha256"):
        if primary["inputs"].get(key) != other["inputs"].get(key):
            raise RenderError(f"{name} and primary.json differ in inputs.{key}")


def prose_errors(primary: dict[str, Any], comp: dict[str, Any], roles: dict[str, str],
                 methods: dict[str, Any] | None, failed_value: float, final: dict[str, Any],
                 second: dict[str, Any] | None) -> list[str]:
    """The comparative wording of the sections, checked against the receipts.

    The sections cite every value through a macro, but words such as "fails", "below", "the same
    as" or "not yet scored" are typed. Each claim here names the sentence it protects; when a new
    receipt falsifies one, --check fails until the prose is rewritten."""
    arm = {role: next(a for a in primary["arms"] if a["arm_id"] == arm_id) for role, arm_id in roles.items()}
    trg = {role: a["primary"]["trg"] for role, a in arm.items()}
    task = {role: {t["task"]: t["value"] for t in a["primary"]["tasks"]} for role, a in arm.items()}
    delta = {t["task"]: t for t in comp["tasks"]}
    dec, cal, m = comp["decision"], comp["null_calibration"], comp["matched"]
    failed = [g["name"] for g in dec["guards"] if g["check"] == "fail"]
    privacy = next(g for g in dec["guards"] if g["name"] == "privacy_coverage_delta")
    families = {t["task"]: t["families"] for t in arm["Cand"]["primary"]["tasks"]}
    claims = [
        ("outcome is HOLD", dec["outcome"] == "HOLD"),
        ("conditions (i)-(iii) hold", all(c["check"] == "pass" for c in dec["conditions"])),
        ("the failing guards are tasks, mlp, realism and privacy",
         failed == ["task_ci_low", "mlp_delta", "realism_coverage_delta", "privacy_coverage_delta"]),
        ("auto-v8 improves TRG with a positive lower bound", comp["delta"] > 0 and comp["ci_low"] > 0),
        ("the matched-size interval contains zero", m["ci_low"] < 0 < m["ci_high"]),
        ("the gain comes from classification; regression favours the baseline",
         delta["regression"]["delta"] < 0 < min(delta["binary"]["delta"], delta["multiclass"]["delta"])
         and task["Cand"]["regression"] < task["Base"]["regression"]),
        ("the MLP auditor prefers the baseline", arm["Cand"]["mlp_trg"] < arm["Base"]["mlp_trg"]),
        ("realism coverage is lower than the baseline's", arm["Cand"]["realism_coverage"] < arm["Base"]["realism_coverage"]),
        ("privacy coverage is just below its guard", privacy["threshold"] - 0.005 < privacy["value"] < privacy["threshold"]),
        ("statistical conditions alone exceed the null rate; with guards they do not",
         not cal["statistical_pass"]["acceptable"] and cal["flip_guard_rule"]["acceptable"]),
        ("analytic alone is worse than the baseline, better on regression, worse on multiclass",
         trg["Analytic"] < trg["Base"] and task["Analytic"]["regression"] > task["Base"]["regression"]
         and task["Analytic"]["multiclass"] < task["Base"]["multiclass"]),
        ("both small arms fail the same number of cells",
         arm["PrevSmall"]["primary"]["failed_cells"] == arm["OldSmall"]["primary"]["failed_cells"]),
        ("copy of A scores below zero", trg["CopyA"] < 0),
        ("twin_c is not applicable on some cells", arm["TwinC"]["primary"]["not_applicable_cells"] > 0),
        ("auto-v8 C2ST is the same as independent marginals at three decimals",
         num(arm["Cand"]["c2st_median"]) == num(arm["Indep"]["c2st_median"])),
        ("binary has the fewest families", families["binary"] == min(families.values())),
        ("Holm covers a single comparison", comp["holm_m"] == 1),
        ("the contract is not frozen", primary["contract"]["protocol_frozen"] is False),
    ] + methods_mod.claims(methods, arm, failed_value) + final_mod.claims(final, comp) + methods_mod.second_claims(
        second, methods and methods["versus"]["arfpy-arf"]["matched"]["ci_low"])
    return [f"prose claim no longer holds for the receipts: {text}" for text, ok in claims if not ok]


def render(hero: dict[str, Any] | None) -> tuple[dict[str, str], list[str]]:
    """The generated files, and the prose claims the receipts no longer support."""
    primary, compare = read_json(f"{SCOREBOARD}/primary.json"), read_json(f"{SCOREBOARD}/compare.json")
    kpi, fid = read_json("contracts/dope-kpi.json"), read_json("contracts/dope-fidelity-v3-candidate.json")
    layout = read_json("contracts/evidence-layout-v1.json")
    check_same_run(primary, compare, "compare.json")
    if len(compare["comparisons"]) != 1:
        raise RenderError("compare.json must hold exactly the one pre-registered comparison")
    comp = compare["comparisons"][0]
    if kpi["contract_id"] != primary["contract"]["id"] or sha256("contracts/dope-kpi.json") != primary["contract"]["sha256"]:
        raise RenderError("contracts/dope-kpi.json is not the contract the receipts were scored under")
    methods = methods_mod.load(SCOREBOARD, read_json, lambda rel: (REPO / rel).is_file(), primary,
                               sha256(f"{SCOREBOARD}/primary.json"), check_same_run)
    third = methods["third"] if methods else []
    roles = role_ids(primary, comp)
    final = final_mod.load(SCOREBOARD, read_json, lambda rel: (REPO / rel).read_text(encoding="utf-8"),
                           lambda rel: (REPO / rel).is_file(), sha256, primary, roles["Base"], roles["Cand"], check_same_run)
    second = methods_mod.load_second(SCOREBOARD, read_json, lambda rel: (REPO / rel).is_file(), sha256,
                                     read_json(f"{SCOREBOARD}/{final_mod.LOOK}/primary.json"), check_same_run)
    readme = (REPO / f"{SCOREBOARD}/readme-block.md").read_text(encoding="utf-8")
    if methods:
        readme_arms = [a for a in primary["arms"] if a["arm_id"] == roles["Cand"] or a["artifact_class"] == "control"] + third
        tables.cross_check(readme, readme_arms, list(methods["versus"].values()), set())
    else:
        tables.cross_check(readme, primary["arms"], compare["comparisons"], set())
    consts, rust_files = rust_constants()
    by_id = {a["arm_id"]: a for a in primary["arms"]}
    cand = by_id[roles["Cand"]]["primary"]
    entries = [("ReleaseVersion", Path(SCOREBOARD).name), ("BaseVersion", roles["PrevSmall"].split("-")[1]),
               ("OldVersion", roles["OldSmall"].split("-")[1]), ("PrimarySha", sha256(f"{SCOREBOARD}/primary.json")[:12]),
               ("CompareSha", sha256(f"{SCOREBOARD}/compare.json")[:12]), ("KpiContractSha", primary["contract"]["sha256"][:12]),
               ("EvLayoutSha", sha256("contracts/evidence-layout-v1.json")[:12]),
               ("ScheduleSha", compare["inputs"]["schedule_sha256"][:12]), ("ToolVersion", primary["tool"]["version"]),
               ("ToolGitSha", primary["tool"]["git_sha"]), ("ProtocolFrozen", str(primary["contract"]["protocol_frozen"]).lower()),
               ("BootDraws", grouped(primary["bootstrap"]["draws"])), ("Scheduled", str(cand["scheduled_cells"])),
               ("Uninformative", str(cand["uninformative_cells"])), ("Families", str(cand["families"])),
               ("ArmCount", str(len(primary["arms"]))), ("ThirdPartyCount", str(len(third)))]
    entries += [(k, grouped(float(v)) if v.isdigit() else v) for k, v in consts.items()]
    entries.append(("TreeDepth", str(int(math.log2(int(consts["TreeLeaves"]))))))
    for t in cand["tasks"]:
        entries.append((f"{t['task'].capitalize()}Families", str(t["families"])))
    entries += contract_macros(kpi, fid, layout)
    if consts["SupportFloorK"] != str(layout["support_floor_k"]):
        raise RenderError("SUPPORT_FLOOR_K in the Rust sources differs from the evidence layout's support_floor_k")
    for role, arm_id in roles.items():
        entries += arm_macros(role, by_id[arm_id])
    entries += comparison_macros(comp, compare)
    if methods:
        entries += methods_mod.macros(methods, arm_macros, num, ci, grouped, sha256, pct, SCORE_SCALE)
    entries += final_mod.macros(final, num, sha256)
    entries += methods_mod.second_macros(second, ci, sha256, SCORE_SCALE)
    entries += sweep_macros()
    if hero:
        entries += [("HeroSvgSha", hero["source_sha256"][:12]), ("HeroPdfSha", hero["output_sha256"][:12])]
    names = [k for k, _ in entries]
    if len(names) != len(set(names)):
        raise RenderError(f"duplicate macro names: {sorted({n for n in names if names.count(n) > 1})}")
    lines = ["% Generated by docs/whitepaper-v3/scripts/render_v3_paper.py -- DO NOT EDIT BY HAND.",
             r"\newcommand{\VThree}[1]{\ifcsname VThree@#1\endcsname\csname VThree@#1\endcsname"
             r"\else\PackageError{dope-v3}{undefined receipt macro #1}{Re-run render_v3_paper.py}\fi}"]
    lines += [rf"\expandafter\newcommand\csname VThree@{k}\endcsname{{{v}}}" for k, v in entries]
    out = {"macros": "\n".join(lines) + "\n"}
    missing = [rel for rel in RUST_TABLES if not (REPO / rel).is_file()]
    if missing:
        raise RenderError(f"missing {missing}: run dope_kpi render-paper-tables")
    inputs = sorted({f"{SCOREBOARD}/{n}" for n in ("primary.json", "compare.json", "readme-block.md", "render-manifest.json", "evidence/addendum-k/rows.csv", "evidence/addendum-k/served.csv")}
                    | set(RUST_TABLES)
                    | {"contracts/dope-kpi.json", "contracts/dope-fidelity-v3-candidate.json",
                       "contracts/evidence-layout-v1.json"} | rust_files | set(methods["paths"] if methods else []) | set(final["paths"])
                    | set(second["paths"] if second else []))
    manifest = {"schema": "dope.whitepaper-v3-render/v1", "hero": hero,
                "inputs": {rel: sha256(rel) for rel in inputs},
                "outputs": {OUT[k]: hashlib.sha256(v.encode()).hexdigest() for k, v in out.items()}}
    out["manifest"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return out, prose_errors(primary, comp, roles, methods, float(kpi["statistic"]["failed_value"]), final, second)


def convert_hero() -> dict[str, Any]:
    tool = shutil.which("rsvg-convert")
    if not tool:
        raise RenderError("--figures needs rsvg-convert (librsvg2-bin)")
    cmd = ["rsvg-convert", "-f", "pdf", "-o", HERO_PDF, HERO_SVG]
    (REPO / HERO_PDF).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=REPO, check=True, env=dict(os.environ, SOURCE_DATE_EPOCH=HERO_EPOCH))
    version = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    return {"source": HERO_SVG, "source_sha256": sha256(HERO_SVG), "output": HERO_PDF,
            "output_sha256": sha256(HERO_PDF), "tool": version,
            "command": f"SOURCE_DATE_EPOCH={HERO_EPOCH} " + " ".join(cmd)}


def hero_errors(hero: dict[str, Any] | None) -> list[str]:
    if not hero:
        return ["no hero provenance recorded; run render_v3_paper.py --figures"]
    errors = []
    scoreboard = read_json(f"{SCOREBOARD}/render-manifest.json")["files"].get(HERO_SVG)
    if sha256(HERO_SVG) != hero["source_sha256"]:
        errors.append(f"{HERO_SVG} changed since {HERO_PDF} was converted; run --figures")
    if scoreboard != hero["source_sha256"]:
        errors.append("hero SVG SHA-256 differs from the scoreboard render-manifest.json entry")
    if not (REPO / HERO_PDF).is_file() or sha256(HERO_PDF) != hero["output_sha256"]:
        return errors + [f"{HERO_PDF} is not the recorded conversion output; run --figures"]
    if not shutil.which("pdftotext"):
        return errors + ["pdftotext (poppler-utils) is required to verify the hero PDF text"]
    text = subprocess.run(["pdftotext", "-raw", str(REPO / HERO_PDF), "-"], capture_output=True,
                          text=True, check=True).stdout
    flat = re.sub(r"\s+", "", text)
    svg = (REPO / HERO_SVG).read_text(encoding="utf-8")
    for item in re.findall(r"<text[^>]*>([^<]*)</text>", svg):
        if re.sub(r"\s+", "", html.unescape(item)) not in flat:
            errors.append(f"hero PDF lacks the SVG text {html.unescape(item)!r}")
    return errors


def hide_measured_displays(text: str) -> str:
    """Drop figure, table and code interiors from the digit scan.

    Display math stays visible. A digit there must be a macro or a structural 0 or 1.
    """
    def blank(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    for env in LINT_HIDE:
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", blank, text, flags=re.S)
    return text


def lint_tex(rel: str, text: str) -> list[str]:
    """Digits typed into prose; every quantity must come from a \\VThree macro."""
    originals = text.splitlines()
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in originals)
    body = hide_measured_displays(body)
    body = LINT_MATH.sub(lambda m: LINT_STRUCTURAL.sub(" ", m.group(0)), body)
    errors = []
    for no, line in enumerate(body.splitlines(), 1):
        for phrase in LINT_ALLOW:
            line = line.replace(phrase, " ")
        for pattern in LINT_STRIP:
            line = re.sub(pattern, " ", line)
        if re.search(r"\d", line):
            errors.append(f"{rel}:{no}: hand-typed digit outside a \\VThree macro: {originals[no - 1].strip()[:90]}")
    return errors


def paper_errors(macros: str) -> list[str]:
    defined = set(re.findall(r"\\csname VThree@([^\\]+)\\endcsname\{", macros))
    errors, used = [], set()
    main = (REPO / "docs/whitepaper-v3/dope-v3.tex").read_text(encoding="utf-8")
    body = main[main.index(r"\begin{document}"):]
    sources = [("docs/whitepaper-v3/dope-v3.tex", body)]
    sources += [(p.relative_to(REPO).as_posix(), p.read_text(encoding="utf-8"))
                for p in sorted((REPO / "docs/whitepaper-v3/sections").glob("*.tex"))]
    for rel, text in sources:
        errors += lint_tex(rel, text)
        cited = set(re.findall(r"\\VThree\{([^{}]*)\}", text))
        errors += [f"{rel}: \\VThree{{{n}}} cites an earlier DOPE release" for n in sorted(cited) if PRIOR_MACRO.match(n)]
        used |= cited
    for rel in RUST_TABLES:
        if PRIOR_ARM.search((REPO / rel).read_text(encoding="utf-8")):
            errors.append(f"{rel} lists an arm of an earlier DOPE release")
    return errors + [f"undefined macro \\VThree{{{n}}}" for n in sorted(used - defined)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="write nothing; fail on drift, provenance or lint")
    ap.add_argument("--figures", action="store_true", help="convert the README hero SVG to PDF first")
    args = ap.parse_args()
    manifest_path = REPO / OUT["manifest"]
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    try:
        hero = convert_hero() if args.figures and not args.check else old.get("hero")
        errors = hero_errors(hero)
        out, prose = render(hero)
    except (RenderError, KeyError, subprocess.CalledProcessError, tables.TableError, methods_mod.MethodsError,
            final_mod.FinalLookError) as err:
        print(f"render_v3_paper: {err}", file=sys.stderr)
        return 1
    errors += prose + paper_errors(out["macros"])
    if args.check:
        for key, text in out.items():
            path = REPO / OUT[key]
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                errors.append(f"DRIFT: {OUT[key]} differs from its rendering; run render_v3_paper.py")
    else:
        for key, text in out.items():
            (REPO / OUT[key]).parent.mkdir(parents=True, exist_ok=True)
            (REPO / OUT[key]).write_text(text, encoding="utf-8")
            print(f"wrote {OUT[key]}")
    for err in errors:
        print(f"render_v3_paper: {err}", file=sys.stderr)
    if not errors and args.check:
        print("OK: generated files, hero provenance, prose claims and the digit lint match the committed inputs")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
