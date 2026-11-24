"""Third-party baseline arms of the DOPE v3.0.0 paper: SDV GaussianCopula, CTGAN and TVAE, and arfpy ARF.

They are read only when `primary-methods.json` is committed beside the scoreboard receipts, and
then the three files of that run must all be there:

  primary-methods.json   `dope_kpi primary` over the release arms and the four methods
  compare-methods.json   `dope_kpi compare`: each method against the release encoder, Holm across them
  MANIFEST.json  library versions, recipes, licences and commands of the run
  evidence/paper/v3-vs-<method>.json
                 `dope_kpi compare` re-run on the same scores with the release encoder as the
                 candidate: the paper's direction, which must be the exact sign reversal of
                 the method's row in compare-methods.json

The run must be the release receipts' run (schedule, realref, panel, contract, bootstrap), and
`primary-methods.json` must repeat every committed release arm field for field. Anything else is
an error, never a silent omission.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import v3_paper_tables as tables

PRIMARY, COMPARE, MANIFEST = "primary-methods.json", "compare-methods.json", "MANIFEST.json"
VERSUS = "evidence/paper/v3-vs-{}.json"
ROLES = {"sdv-gaussian-copula": "Gc", "sdv-ctgan": "Ctgan", "sdv-tvae": "Tvae", "arfpy-arf": "Arf"}
VERSIONS = {"SdvVersion": "sdv", "CtganVersion": "ctgan", "CopulasVersion": "copulas", "RdtVersion": "rdt",
            "ArfpyVersion": "arfpy", "TorchVersion": "torch", "SklearnVersion": "scikit-learn"}


class MethodsError(ValueError):
    """The third-party receipts are incomplete or are not the release receipts' run."""


def load(scoreboard: str, read: Callable[[str], dict], exists: Callable[[str], bool], primary: dict[str, Any],
         primary_sha: str, same_run: Callable[[dict, dict, str], None]) -> dict[str, Any] | None:
    paths = [f"{scoreboard}/{name}" for name in (PRIMARY, COMPARE, MANIFEST)]
    present = [exists(p) for p in paths]
    if not any(present):
        return None
    if not all(present):
        raise MethodsError(f"third-party receipts must be committed together: {paths}")
    methods, compare, manifest = (read(p) for p in paths)
    same_run(primary, methods, PRIMARY)
    same_run(primary, compare, COMPARE)
    release = {a["arm_id"]: a for a in primary["arms"]}
    if {a["arm_id"]: a for a in methods["arms"] if a["arm_id"] in release} != release:
        raise MethodsError(f"{PRIMARY} does not repeat every committed release arm field for field")
    third = [a for a in methods["arms"] if a["arm_id"] not in release]
    ids = sorted(a["arm_id"] for a in third)
    if ids != sorted(ROLES):
        raise MethodsError(f"{PRIMARY}: expected the arms {sorted(ROLES)}, found {ids}")
    comps = compare["comparisons"]
    if sorted(c["id"] for c in comps) != ids or any(c["holm_m"] != len(comps) for c in comps):
        raise MethodsError(f"{COMPARE}: expected one comparison per method, Holm across all of them")
    run = manifest.get("schedule", {}).get("sha256"), manifest.get("realref_sha256")
    if manifest.get("schema") != "dope.baselines-manifest/v1" or run != (
            primary["inputs"]["schedule_sha256"], primary["inputs"]["realref_sha256"]):
        raise MethodsError(f"{MANIFEST} is not the manifest of the release receipts' run")
    sanity = manifest["sanity"]
    if not sanity["identical"] or sanity["committed_primary_sha256"] != primary_sha:
        raise MethodsError(f"{MANIFEST}: the run's sanity replay does not reproduce the committed primary.json")
    packages = {p["name"].lower(): p["version"] for p in
                manifest["runtime"]["venv_packages"] + manifest["runtime"]["shared_packages"]}
    for method in manifest["methods"]:
        if packages.get(method["package"]) != method["version"]:
            raise MethodsError(f"{MANIFEST}: {method['arm_id']} names {method['package']} {method['version']}")
    if sorted(m["arm_id"] for m in manifest["methods"]) != ids:
        raise MethodsError(f"{MANIFEST}: its methods are not the arms of {PRIMARY}")
    versus = {}
    for c in comps:
        rel = f"{scoreboard}/{VERSUS.format(c['id'])}"
        if not exists(rel):
            raise MethodsError(f"missing {rel}: the release encoder against {c['id']}")
        rec = read(rel)
        same_run(primary, rec, rel)
        if len(rec["comparisons"]) != 1:
            raise MethodsError(f"{rel} must hold exactly one comparison")
        versus[c["id"]] = v = rec["comparisons"][0]
        if (v["id"], v["baseline"]) != (c["baseline"], c["id"]) or not sign_reversed(v, c):
            raise MethodsError(f"{rel} is not the sign reversal of {c['id']} in {COMPARE}")
        paths.append(rel)
    return {"third": third, "comparisons": comps, "manifest": manifest, "packages": packages, "paths": paths,
            "versus": versus, "cand": release[comps[0]["baseline"]]}


def sign_reversed(v: dict[str, Any], c: dict[str, Any]) -> bool:
    """Same cells and families, and every difference and interval negated (bit-exact up to rounding)."""
    def flipped(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return all(abs(a[x] + b[y]) < 1e-9 for x, y in (("delta", "delta"), ("ci_low", "ci_high"), ("ci_high", "ci_low")))
    tasks = {t["task"]: t for t in c["tasks"]}
    return (v["cells"], v["families"]) == (c["cells"], c["families"]) and flipped(v, c) \
        and flipped(v["matched"], c["matched"]) and sorted(tasks) == sorted(t["task"] for t in v["tasks"]) \
        and all(flipped(t, tasks[t["task"]]) for t in v["tasks"])


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjusted p-values over the given comparisons."""
    order, out, running = sorted(p, key=p.get), {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[key]))
        out[key] = running
    return out


def macros(info: dict[str, Any], arm_macros: Callable, num: Callable, ci: Callable, grouped: Callable,
           sha256: Callable[[str], str], pct: Callable, scale: float) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    cal = {}
    for method, v in info["versus"].items():
        role, m = ROLES[method], v["matched"]
        cal[role] = v["null_calibration"]
        out += [(f"{role}VsPts", ci(scale * v["delta"], scale * v["ci_low"], scale * v["ci_high"], True, 1)),
                (f"{role}VsPtsMatched", ci(scale * m["delta"], scale * m["ci_low"], scale * m["ci_high"], True, 1)),
                (f"{role}VsRule", v["decision"]["outcome"])]
    stat = [c["statistical_pass"]["rate"] for c in cal.values()]
    ratio = [a["bytes"]["median"] / info["cand"]["bytes"]["median"] for a in info["third"]]
    replicates = {c["replicates"] for c in cal.values()}
    if len(replicates) != 1:
        raise MethodsError("the four sign-flip checks differ in their replicate counts")
    out += [("MethodsStatPassMinPct", pct(min(stat), 2)), ("MethodsStatPassMaxPct", pct(max(stat), 2)),
            ("MethodsFlipMaxPct", pct(max(c["flip_guard_rule"]["rate"] for c in cal.values()), 2)),
            ("MethodsNullReplicates", grouped(float(replicates.pop()))),
            ("MethodsRatioMin", grouped(float(round(min(ratio))))), ("MethodsRatioMax", grouped(float(round(max(ratio)))))]
    for arm in info["third"]:
        role = ROLES[arm["arm_id"]]
        out += arm_macros(role, arm)
        out += [(f"{role}CopyMean", num(arm["exact_copy"]["mean"])),
                (f"{role}TrgAtBudget", num(arm["trg_at_budget"]))]
    for comp in info["comparisons"]:
        role = ROLES[comp["id"]]
        out += [(f"{role}VsDelta", ci(comp["delta"], comp["ci_low"], comp["ci_high"], True)),
                (f"{role}VsPHolm", num(comp["p_holm"])), (f"{role}VsOutcome", comp["decision"]["outcome"])]
    manifest, packages = info["manifest"], info["packages"]
    out += [(name, packages[pkg]) for name, pkg in VERSIONS.items()]
    budget = re.search(r"--budget-bytes (\d+)", manifest["commands"]["score"])
    if not budget:
        raise MethodsError(f"{MANIFEST}: the score command names no --budget-bytes")
    evidence = manifest["artifact_class_evidence"]
    scheduled = {m["cells"]["scheduled"] for m in manifest["methods"]}
    if len(scheduled) != 1 or sorted(evidence["per_method"]) != sorted(ROLES) or any(
            len(files) != len(evidence["smoke_cells"]) for files in evidence["per_method"].values()):
        raise MethodsError(f"{MANIFEST}: the audit does not cover the same smoke cells for every method")
    primary_path, compare_path, manifest_path = info["paths"][:3]
    return out + [("MethodsAuditFiles", str(len(evidence["smoke_cells"]))), ("MethodsScheduledCells", str(scheduled.pop())),("MethodsHolmM", str(len(info["comparisons"]))), ("MethodsCount", str(len(info["third"]))),
                  ("MethodsPythonVersion", manifest["runtime"]["python"]),
                  ("SdvLicense", manifest["licenses"]["sdv"]), ("ArfpyLicense", manifest["licenses"]["arfpy"]),
                  ("MethodsScoreBudget", grouped(float(budget.group(1)))),
                  ("MethodsPrimarySha", sha256(primary_path)[:12]), ("MethodsCompareSha", sha256(compare_path)[:12]),
                  ("MethodsManifestSha", sha256(manifest_path)[:12])]


def claims(info: dict[str, Any] | None, release: dict[str, dict[str, Any]], failed_value: float) -> list[tuple[str, bool]]:
    """The prose about the third-party arms (sections 0, 2, 6, 8 and appendix C), against the receipts."""
    if info is None:
        return [("third-party receipts are committed", False)]
    arm = {ROLES[a["arm_id"]]: a for a in info["third"]}
    comp = {ROLES[c["id"]]: c for c in info["comparisons"]}
    outcome = {r: c["decision"]["outcome"] for r, c in comp.items()}
    failed = {r: {g["name"] for g in c["decision"]["guards"] if g["check"] == "fail"} for r, c in comp.items()}
    trg = {r: a["primary"]["trg"] for r, a in arm.items()}
    task = {r: {t["task"]: t["value"] for t in a["primary"]["tasks"]} for r, a in arm.items()}
    cand = {t["task"]: t["value"] for t in release["Cand"]["primary"]["tasks"]}
    return [*_versus_claims(info, release, arm), *[
        ("every third-party method has a lower TRG than auto-v8 and a negative difference",
         all(c["delta"] < 0 and trg[r] < release["Cand"]["primary"]["trg"] for r, c in comp.items())),
        ("ARF is HOLD with an interval containing zero; the other three are REJECT",
         outcome == {"Gc": "REJECT", "Ctgan": "REJECT", "Tvae": "REJECT", "Arf": "HOLD"}
         and comp["Arf"]["ci_low"] < 0 < comp["Arf"]["ci_high"]),
        ("ARF has the highest TRG of the four", max(trg, key=trg.get) == "Arf"),
        ("ARF beats auto-v8 on regression and trails it on binary and multiclass targets",
         task["Arf"]["regression"] > cand["regression"] and task["Arf"]["binary"] < cand["binary"]
         and task["Arf"]["multiclass"] < cand["multiclass"]),
        ("CTGAN scores below independently permuted columns", trg["Ctgan"] < release["Indep"]["primary"]["trg"]),
        ("every method fails the byte guard, no file is within the budget, and at the budget each scores the floor",
         all("bytes" in f for f in failed.values())
         and all(a["bytes"]["within_budget"] == 0 and a["trg_at_budget"] == failed_value for a in arm.values())),
        ("CTGAN alone fails the exact-copy guard", [r for r, f in failed.items() if "exact_copy_excess" in f] == ["Ctgan"]),
        ("only the Gaussian copula and CTGAN have single entries above the exact-copy bound",
         sorted(r for r, a in {**arm, "Cand": release["Cand"]}.items() if a["exact_copy"]["over_bound"] > 0)
         == ["Ctgan", "Gc"]),
        ("TVAE alone has failed cells", [r for r, a in arm.items() if a["primary"]["failed_cells"] > 0] == ["Tvae"]),
        ("the packet audit found no row container and no column copy in any audited file",
         all(f["audit_row_containers"] == 0 and f["audit_column_copies"] == 0
             for files in info["manifest"]["artifact_class_evidence"]["per_method"].values() for f in files)),
    ]]


def _versus_claims(info: dict[str, Any], release: dict[str, dict[str, Any]],
                   arm: dict[str, dict[str, Any]]) -> list[tuple[str, bool]]:
    """The prose of sections 0, 1, 6 and 8 and appendix D, which ranks the release encoder against the methods."""
    every = {**arm, "Cand": release["Cand"]}
    v = {ROLES[k]: c for k, c in info["versus"].items()}
    failed = {r: {g["name"] for g in c["decision"]["guards"] if g["check"] == "fail"} for r, c in v.items()}
    cond = {r: {g["name"] for g in c["decision"]["conditions"] if g["check"] == "fail"} for r, c in v.items()}
    alpha = {g["threshold"] for c in v.values() for g in c["decision"]["conditions"] if g["name"] == "holm_p"}
    adjusted = holm({r: c["p_one_sided"] for r, c in v.items()})
    cal = {r: c["null_calibration"] for r, c in v.items()}

    def rank(key: str) -> list[str]:
        return sorted(every, key=lambda r: every[r][key]["trg"], reverse=True)

    def of(key: str) -> dict[str, float]:
        return {r: a[key] for r, a in every.items()}
    real, priv = of("realism_coverage"), of("privacy_coverage")
    c2st = {r: arm[r]["c2st_median"] for r in v}
    size = {r: arm[r]["bytes"]["median"] for r in v}
    task = {r: {t["task"]: t["value"] for t in a["primary"]["tasks"]} for r, a in
            (("Cand", release["Cand"]), ("Analytic", release["Analytic"]))}
    return [
        ("auto-v8 ranks first at both sizes, then ARF, the Gaussian copula, TVAE and CTGAN",
         rank("primary") == rank("matched") == ["Cand", "Arf", "Gc", "Tvae", "Ctgan"]),
        ("the gains over the three SDV methods have lower bounds above zero; the gain over ARF has an interval containing zero",
         all(v[r]["ci_low"] > 0 for r in ("Gc", "Tvae", "Ctgan")) and v["Arf"]["ci_low"] < 0 < v["Arf"]["delta"]),
        ("the rule is PROMOTE against TVAE alone with every guard passing; GC and CTGAN meet (i)-(iii) and miss one guard "
         "(tasks, privacy); against ARF the interval misses and three guards fail",
         {r: c["decision"]["outcome"] for r, c in v.items()} == {"Tvae": "PROMOTE", "Gc": "HOLD", "Ctgan": "HOLD", "Arf": "HOLD"}
         and failed == {"Tvae": set(), "Gc": {"task_ci_low"}, "Ctgan": {"privacy_coverage_delta"},
                        "Arf": {"task_ci_low", "mlp_delta", "realism_coverage_delta"}}
         and cond["Gc"] == cond["Ctgan"] == cond["Tvae"] == set() and "ci_low" in cond["Arf"]),
        ("no outcome changes when Holm covers the four pairs instead of one",
         len(alpha) == 1 and all((adjusted[r] < min(alpha)) == (c["p_holm"] < min(alpha)) for r, c in v.items())),
        ("in every pair the statistical conditions alone exceed the null rate and the full rule does not",
         all(not c["statistical_pass"]["acceptable"] and c["flip_guard_rule"]["acceptable"] for c in cal.values())),
        ("realism coverage: ARF above auto-v8 above the Gaussian copula; TVAE and CTGAN zero",
         real["Arf"] > real["Cand"] > real["Gc"] > 0 == real["Tvae"] == real["Ctgan"]),
        ("privacy coverage: CTGAN above auto-v8 above the Gaussian copula and TVAE; ARF lowest",
         priv["Ctgan"] > priv["Cand"] > max(priv["Gc"], priv["Tvae"]) and min(priv, key=priv.get) == "Arf"),
        ("the methods' C2ST medians run from ARF's to CTGAN's", (min(c2st, key=c2st.get), max(c2st, key=c2st.get)) == ("Arf", "Ctgan")),
        ("the methods' median files run from the Gaussian copula's to ARF's", (min(size, key=size.get), max(size, key=size.get)) == ("Gc", "Arf")),
        ("CTGAN scores below label shuffle too", every["Ctgan"]["primary"]["trg"] < release["LabelShuffle"]["primary"]["trg"]),
        ("every auto-v8 file is within the budget and auto-v8 scores every cell",
         release["Cand"]["bytes"]["within_budget"] == release["Cand"]["bytes"]["files"] and release["Cand"]["primary"]["failed_cells"] == 0),
        ("without selection the packet scores lower, higher on regression and lower on binary and multiclass",
         release["Analytic"]["primary"]["trg"] < release["Cand"]["primary"]["trg"]
         and task["Analytic"]["regression"] > task["Cand"]["regression"]
         and task["Analytic"]["binary"] < task["Cand"]["binary"] and task["Analytic"]["multiclass"] < task["Cand"]["multiclass"]),
    ]


# Addendum H part (a): the four methods on the DEV row splits of the final selection look (data seeds 9-12),
# reduced as `served-auto-v8 - method`, Holm (m = 4) over the four pairs. Descriptive; a second block of
# Table 2 and a results paragraph. The two looks are never pooled.
SECOND = "evidence/addendum-h"
SECOND_CAND = "served-auto-v8"
SECOND_REUSED = (SECOND_CAND, "twin_c", "copy_a", "bootstrap_a", "indep_marginals", "label_shuffle")
SECOND_TEXT = ("ADDENDUM_H.md", "PREREG_LOG-H.txt", "MANIFEST-h.json")


def holm_outcome(c: dict[str, Any], p_holm: float) -> str:
    """The contract rule with the Holm-adjusted p in place of the pair's own (render_paper_paired.rs)."""
    dec = c["decision"]
    alpha = next(x["threshold"] for x in dec["conditions"] if x["name"] == "holm_p")
    if p_holm < alpha and all(x["check"] == "pass" for x in dec["conditions"] if x["name"] != "holm_p") \
            and all(g["check"] == "pass" for g in dec["guards"]):
        return "PROMOTE"
    return "REJECT" if dec["outcome"] == "REJECT" else "HOLD"


def load_second(scoreboard: str, read: Callable[[str], dict], exists: Callable[[str], bool],
                sha256: Callable[[str], str], look: dict[str, Any],
                same_run: Callable[[dict, dict, str], None]) -> dict[str, Any] | None:
    """`family-h.json` and what it binds, checked against the committed final look's `primary.json` (`look`)."""
    base = f"{scoreboard}/{SECOND}"
    fam_rel, prim_rel = f"{base}/family-h.json", f"{base}/primary-h.json"
    if not exists(fam_rel):
        return None
    paths = [fam_rel, prim_rel] + [f"{base}/{n}" for n in SECOND_TEXT]
    missing = [p for p in paths if not exists(p)]
    if missing:
        raise MethodsError(f"second-look evidence is incomplete: {missing}")
    fam, prim = read(fam_rel), read(prim_rel)
    same_run(look, prim, prim_rel)
    reused = {a["arm_id"]: a for a in look["arms"] if a["arm_id"] in SECOND_REUSED}
    if sorted(reused) != sorted(SECOND_REUSED) or \
            {a["arm_id"]: a for a in prim["arms"] if a["arm_id"] in reused} != reused:
        raise MethodsError(f"{prim_rel} does not repeat the final look's DOPE and control rows field for field")
    if sorted(a["arm_id"] for a in prim["arms"] if a["arm_id"] not in reused) != sorted(ROLES):
        raise MethodsError(f"{prim_rel}: expected the arms {sorted(ROLES)} beside the reused rows")
    if fam.get("schema") != "dope.addendum-h-family/v1" or fam.get("m") != len(ROLES) \
            or sorted(r["baseline"] for r in fam["comparisons"]) != sorted(ROLES):
        raise MethodsError(f"{fam_rel} is not the Holm family over the four methods")
    versus = {}
    for row in fam["comparisons"]:
        rel = f"{base}/{row['receipt']}"
        if row["receipt"] != f"v3-vs-{row['baseline']}-h.json" or not exists(rel) or sha256(rel) != row["receipt_sha256"]:
            raise MethodsError(f"{fam_rel} does not bind {rel}")
        rec = read(rel)
        same_run(prim, rec, rel)
        if len(rec["comparisons"]) != 1 or (rec["comparisons"][0]["id"], rec["comparisons"][0]["baseline"]) \
                != (SECOND_CAND, row["baseline"]):
            raise MethodsError(f"{rel} must hold exactly {SECOND_CAND} against {row['baseline']}")
        for arm in (SECOND_CAND, row["baseline"]):
            for key in ("rows_sha256", "guards_sha256", "guards_summary_sha256"):
                if rec["inputs"][key].get(arm) != prim["inputs"][key].get(arm):
                    raise MethodsError(f"{rel} and {prim_rel} differ on {key} for {arm}")
        versus[row["baseline"]] = rec["comparisons"][0]
        paths.append(rel)
    adjusted = holm({k: c["p_one_sided"] for k, c in versus.items()})
    rule = {k: holm_outcome(c, adjusted[k]) for k, c in versus.items()}
    for row in fam["comparisons"]:
        k, c = row["baseline"], versus[row["baseline"]]
        if abs(row["p_holm_m4"] - adjusted[k]) > 1e-12 or row["outcome_holm_m4"] != rule[k] \
                or row["replicates"] != (c["ci_low"] > 0 and adjusted[k] < row["alpha"]):
            raise MethodsError(f"{fam_rel}: the row for {k} differs from Holm over the four receipts")
    cand = next(a for a in prim["arms"] if a["arm_id"] == SECOND_CAND)
    if any(read(rel)["tool"] != prim["tool"] for rel in paths[len(SECOND_TEXT) + 2:]):
        raise MethodsError("the second look's receipts name more than one dope_kpi build")
    return {"versus": versus, "holm": adjusted, "rule": rule, "family": fam, "cand": cand, "paths": paths,
            "tool": prim["tool"]}


def second_macros(info: dict[str, Any] | None, ci: Callable, sha256: Callable[[str], str],
                  scale: float) -> list[tuple[str, str]]:
    if info is None:
        return []
    out = [("SecondLookCells", str(info["cand"]["primary"]["cells"])),
           ("SecondLookFamilies", str(info["cand"]["primary"]["families"])),
           ("SecondLookPrimarySha", sha256(info["paths"][1])[:12]), ("SecondLookFamilySha", sha256(info["paths"][0])[:12]),
           ("SecondLookToolGitSha", info["tool"]["git_sha"]), ("SecondLookToolVersion", info["tool"]["version"])]
    for method, v in info["versus"].items():
        role = ROLES[method]
        out += [(f"SecondLook{role}VsPts", ci(scale * v["delta"], scale * v["ci_low"], scale * v["ci_high"], True, 1)),
                (f"SecondLook{role}VsRule", info["rule"][method])]
    return out


def second_claims(info: dict[str, Any] | None) -> list[tuple[str, bool]]:
    """The second-look prose of sections 0, 1, 5, 6 and 8, against the Addendum H part (a) receipts."""
    if info is None:
        return [("the second-look receipts are committed", False)]
    v = {ROLES[k]: c for k, c in info["versus"].items()}
    rule = {ROLES[k]: o for k, o in info["rule"].items()}
    rep = {ROLES[r["baseline"]]: r["replicates"] for r in info["family"]["comparisons"]}
    failed = {r: {g["name"] for g in c["decision"]["guards"] if g["check"] == "fail"} for r, c in v.items()}
    return [
        ("on the second look the SDV gains replicate (lower bound above zero, Holm p below alpha) and the ARF "
         "interval contains zero", rep == {"Gc": True, "Tvae": True, "Ctgan": True, "Arf": False}
         and v["Arf"]["ci_low"] < 0 < v["Arf"]["delta"]),
        ("on the second look the rule is PROMOTE against TVAE and CTGAN, and HOLD against the Gaussian copula "
         "(tasks, privacy) and ARF (interval, tasks, mlp, realism)",
         rule == {"Tvae": "PROMOTE", "Ctgan": "PROMOTE", "Gc": "HOLD", "Arf": "HOLD"}
         and failed == {"Tvae": set(), "Ctgan": set(), "Gc": {"task_ci_low", "privacy_coverage_delta"},
                        "Arf": {"task_ci_low", "mlp_delta", "realism_coverage_delta"}}),
    ]
