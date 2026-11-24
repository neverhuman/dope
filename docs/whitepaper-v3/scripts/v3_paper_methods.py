"""Third-party baseline arms of the DOPE v3.0.0 paper: SDV GaussianCopula, CTGAN and TVAE, and arfpy ARF.

They are read only when `primary-methods.json` is committed beside the scoreboard receipts, and
then the three files of that run must all be there:

  primary-methods.json   `dope_kpi primary` over the release arms and the four methods
  compare-methods.json   `dope_kpi compare`: each method against the release encoder, Holm across them
  MANIFEST.json  library versions, recipes, licences and commands of the run

The run must be the release receipts' run (schedule, realref, panel, contract, bootstrap), and
`primary-methods.json` must repeat every committed release arm field for field. Anything else is
an error, never a silent omission.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import v3_paper_tables as tables

PRIMARY, COMPARE, MANIFEST = "primary-methods.json", "compare-methods.json", "MANIFEST.json"
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
    return {"third": third, "comparisons": comps, "manifest": manifest, "packages": packages, "paths": paths}


def macros(info: dict[str, Any], arm_macros: Callable, num: Callable, ci: Callable, grouped: Callable,
           sha256: Callable[[str], str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
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
    primary_path, compare_path, manifest_path = info["paths"]
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
    return [
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
        ("TVAE alone has failed cells", [r for r, a in arm.items() if a["primary"]["failed_cells"] > 0] == ["Tvae"]),
        ("the packet audit found no row container and no column copy in any audited file",
         all(f["audit_row_containers"] == 0 and f["audit_column_copies"] == 0
             for files in info["manifest"]["artifact_class_evidence"]["per_method"].values() for f in files)),
    ]
