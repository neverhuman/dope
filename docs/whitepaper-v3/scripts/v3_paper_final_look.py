"""The final DEV look of DOPE v3.0.0 (Addendum F of the pre-registration), for the v3 paper.

Inputs, all under the scoreboard's `evidence/` directory and all required:

  final-look-s9-12/   the look's `dope_kpi` receipts, the Holm family, the disclosures and
                      `SHA256SUMS`, which must list exactly those files with their hashes
  selection/V3_SELECTION_PREREG.md, PREREG_LOG.txt
                      the pre-registration, whose last log entry must hash it; its Addendum F
                      freeze must name the look's schedule, the candidate's source and binaries,
                      and its result must name the look's primary and family receipts
  selection/wide-dev/ the Addendum F eligibility receipts of the builds tried on wide DEV

The look must be scored by the frozen `dope_kpi` build under the frozen schedule and contract.
The Holm family, the family sums and the exact sign-flip p values are recomputed here. Any
disagreement is an error, never a silent omission.
"""
from __future__ import annotations

import itertools
import re
from typing import Any, Callable

LOOK, SELECTION = "evidence/final-look-s9-12", "evidence/selection"
PREREG, LOG = f"{SELECTION}/V3_SELECTION_PREREG.md", f"{SELECTION}/PREREG_LOG.txt"
FILES = ("primary.json", "family.json", "compare-a.json", "compare-b.json", "null-a.json", "arms.json",
         "family-disclosure-rc-vs-v2.json", "family-disclosure-rc-vs-served.json", "served-vs-v2-descriptive.json")
# The Addendum F freeze: the look's schedule, the contract and the scoring build. The PREREG must agree.
SCHEDULE = "beaa2a0fa3e2bcc44c37f23bf15adca0c2ff5b4adb5f212a198c99f3d23f71a2"
CONTRACT = "200ea0ed1ba23a25be89de2dac2a4101656119514e0d69b4ff86a98fa4857623"
TOOL = {"name": "dope_kpi", "version": "3.0.0", "git_sha": "bc92df0b1beae93254718e270e4e7ef7b05b38a3"}
RC, SERVED, BUILDS = "rc-auto-v8", "served-auto-v8", ("v3c-cand6", "v3c-cand6b")
DESCRIPTIVE = "served-vs-v2-descriptive.json"
TOL = 1e-9


class FinalLookError(ValueError):
    """The final-look evidence is incomplete, altered, or not the run the pre-registration froze."""


def holm(ps: list[float]) -> list[float]:
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(len(ps)), key=lambda j: ps[j])):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def sign_flip_p(contributions: list[float], observed: float) -> float:
    """Exact one-sided p: the share of sign assignments of the nonzero families with a sum >= observed."""
    nonzero = [c for c in contributions if c != 0]
    hits = sum(sum(s * c for s, c in zip(signs, nonzero)) >= observed - TOL
               for signs in itertools.product((1, -1), repeat=len(nonzero)))
    return hits / 2 ** len(nonzero)


def _section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        raise FinalLookError(f"{PREREG} has no section {heading!r}")
    end = text.find("\n### ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


def _match(pattern: str, text: str, what: str) -> re.Match:
    found = re.search(pattern, text)
    if not found:
        raise FinalLookError(f"{PREREG} does not state {what}")
    return found


def _one(receipt: dict[str, Any], name: str, cand: str, base: str) -> dict[str, Any]:
    comps = receipt["comparisons"]
    if len(comps) != 1 or (comps[0]["id"], comps[0]["baseline"]) != (cand, base):
        raise FinalLookError(f"{name} must hold the one comparison {cand} vs {base}")
    return comps[0]


def _disclosure(disc: dict[str, Any], comp: dict[str, Any], name: str) -> None:
    contributions = [f["contribution"] for f in disc["family_contributions"]]
    if ((disc["candidate"], disc["baseline"]) != (comp["id"], comp["baseline"]) or abs(disc["delta"] - comp["delta"]) > TOL
            or disc["families"] != comp["families"] or len(contributions) != comp["families"]
            or disc["contract_null_calibration"] != comp["null_calibration"]
            or abs(sum(contributions) - disc["delta"]) > TOL
            or disc["nonzero_families"] != sum(c != 0 for c in contributions)
            or abs(sign_flip_p(contributions, disc["delta"]) - disc["exact_sign_flip_p"]) > TOL):
        raise FinalLookError(f"{name} does not decompose its comparison (families, sum, null or sign-flip p)")


def _descriptive(primary: dict[str, Any], served: dict[str, Any], same_run: Callable[[dict, dict, str], None]) -> None:
    """The descriptive receipt was scored without the families file: its panel lacks the families hash
    and counts the tables outside the schedule as unmatched. Everything else, and both arms, must be
    the primary receipt's."""
    loose = {"families_sha256", "dev_tables_unmatched"}
    panel = {k: v for k, v in primary["panel"].items() if k not in loose}
    if {k: v for k, v in served["panel"].items() if k not in loose} != panel:
        raise FinalLookError(f"{DESCRIPTIVE} differs from primary.json in its panel")
    same_run({**primary, "panel": served["panel"]}, served, DESCRIPTIVE)
    arms = {a["arm_id"]: a for a in primary["arms"]}
    if any(arms.get(a["arm_id"]) != a for a in served["arms"]):
        raise FinalLookError(f"{DESCRIPTIVE}: its arms are not primary.json's, field for field")


def _family_outcome(comp: dict[str, Any], p_family: float) -> str:
    """The pair decision with Holm p taken over the family (the frozen rule)."""
    dec = comp["decision"]
    passed = all(c["check"] == "pass" if c["name"] != "holm_p" else p_family < c["threshold"]
                 for c in dec["conditions"]) and all(g["check"] == "pass" for g in dec["guards"])
    return "PROMOTE" if passed else ("HOLD" if dec["outcome"] == "PROMOTE" else dec["outcome"])


def load(scoreboard: str, read_json: Callable[[str], dict], read_text: Callable[[str], str],
         exists: Callable[[str], bool], sha256: Callable[[str], str], report: dict[str, Any],
         base: str, cand: str, same_run: Callable[[dict, dict, str], None]) -> dict[str, Any]:
    look = {name: f"{scoreboard}/{LOOK}/{name}" for name in FILES}
    wide = {(b, a): f"{scoreboard}/{SELECTION}/wide-dev/cmp-{b}-vs-{a}.json" for b in BUILDS for a in (base, cand)}
    wide_disc = f"{scoreboard}/{SELECTION}/wide-dev/family-disclosure-cand6b-vs-v2.json"
    sums, prereg, log = (f"{scoreboard}/{rel}" for rel in (f"{LOOK}/SHA256SUMS", PREREG, LOG))
    paths = [*look.values(), sums, prereg, log, *wide.values(), wide_disc]
    missing = [p for p in paths if not exists(p)]
    if missing:
        raise FinalLookError(f"final-look evidence missing: {missing}")
    listed = {name: sha for sha, name in (line.split("  ", 1) for line in read_text(sums).splitlines() if line)}
    if set(listed) != set(FILES) or any(sha256(look[n]) != listed[n] for n in FILES):
        raise FinalLookError(f"{sums} does not list exactly the look's files with their SHA-256")
    text, entries = read_text(prereg), read_text(log).splitlines()
    if not entries or not entries[-1].endswith(f"V3_SELECTION_PREREG.md sha256 {sha256(prereg)}"):
        raise FinalLookError(f"the last entry of {LOG} does not hash the committed {PREREG}")
    freeze = _section(text, "### Addendum F freeze")
    result = _section(text, "### Addendum F result")
    seeds = _match(rf"schedule `{SCHEDULE}` \(\d+ cells, data seeds (\d+)-(\d+)", freeze, "the look's schedule")
    report_seeds = _match(r"seeds (\d+)-(\d+) reproduces the report schedule `([0-9a-f]{8,})", freeze, "the report seeds")
    wide_seeds = _match(r"Wide DEV seeds (\d+)-(\d+) ledgers \(schedule `([0-9a-f]{8,})", text, "the wide DEV seeds")
    holm_m = int(_match(r"\*\*Holm family:\*\* \{[^}]*\}, m = (\d+)", freeze, "the Holm family").group(1))
    if not report["inputs"]["schedule_sha256"].startswith(report_seeds.group(3)) or TOOL["git_sha"] not in freeze:
        raise FinalLookError(f"{PREREG}: the freeze names another report schedule or candidate source")
    if not all(sha256(look[n]) in result for n in ("primary.json", "family.json")):
        raise FinalLookError(f"{PREREG}: the Addendum F result does not name the committed look receipts")
    rec = {n: read_json(p) for n, p in look.items()}
    primary = rec["primary.json"]
    for name in ("primary.json", "compare-a.json", "compare-b.json", "null-a.json", "served-vs-v2-descriptive.json"):
        r = rec[name]
        if (r["schema"] != "dope.kpi-receipt/v2" or r["tool"] != TOOL or r["contract"]["sha256"] != CONTRACT
                or r["inputs"]["schedule_sha256"] != SCHEDULE):
            raise FinalLookError(f"{name} is not scored by the frozen dope_kpi, contract and schedule")
        if name != DESCRIPTIVE:
            same_run(primary, r, name)
    _descriptive(primary, rec[DESCRIPTIVE], same_run)
    if primary["contract"] != report["contract"] or primary["bootstrap"] != report["bootstrap"]:
        raise FinalLookError("the look is not scored under the report look's contract and bootstrap")
    comp_a = _one(rec["compare-a.json"], "compare-a.json", RC, base)
    comp_b = _one(rec["compare-b.json"], "compare-b.json", RC, SERVED)
    served = _one(rec["served-vs-v2-descriptive.json"], "served-vs-v2-descriptive.json", SERVED, base)
    family = rec["family.json"]
    pairs = list(zip(family["comparisons"], (("compare-a.json", comp_a), ("compare-b.json", comp_b))))
    if family["schema"] != "dope.v3-final-look-family/v2" or family["m"] != holm_m or len(family["comparisons"]) != holm_m:
        raise FinalLookError("family.json is not the pre-registered Holm family")
    for entry, (name, comp) in pairs:
        if (entry["receipt"], entry["candidate"], entry["baseline"], entry["pair_decision"]) != (
                name, comp["id"], comp["baseline"], comp["decision"]) or any(
                abs(entry[k] - comp[k]) > TOL for k in ("delta", "ci_low", "ci_high", "p_one_sided")):
            raise FinalLookError(f"family.json does not repeat {name}")
    adjusted = holm([e["p_one_sided"] for e in family["comparisons"]])
    if any(abs(e["p_holm_family"] - p) > TOL for e, p in zip(family["comparisons"], adjusted)):
        raise FinalLookError("family.json: p_holm_family is not Holm over the family")
    null = rec["null-a.json"]["null"]
    if (null["candidate"], null["baseline"]) != (RC, base) or null["flip_guard_rule"] != comp_a["null_calibration"]["flip_guard_rule"]:
        raise FinalLookError("null-a.json is not the null of compare-a.json")
    _disclosure(rec["family-disclosure-rc-vs-v2.json"], comp_a, "family-disclosure-rc-vs-v2.json")
    _disclosure(rec["family-disclosure-rc-vs-served.json"], comp_b, "family-disclosure-rc-vs-served.json")
    arms = {a["arm_id"]: a for a in rec["arms.json"]["arms"]}
    served_build = re.fullmatch(r"served product ([0-9a-f]{8}): .*", arms.get(SERVED, {}).get("note", ""))
    if served_build and f"`{cand}` (= `{served_build.group(1)}`)" not in text:
        raise FinalLookError(f"{PREREG} does not identify the report's {cand} with the served engine")
    if (set(arms) != {RC, SERVED, base} or not served_build or f"{BUILDS[1]}-{TOOL['git_sha'][:8]}" not in arms[RC]["binary"]
            or any(a["binary_sha256"][:8] not in freeze for a in arms.values())):
        raise FinalLookError("arms.json does not describe the frozen candidate, served and baseline binaries")
    wide_rec = {k: read_json(p) for k, p in wide.items()}
    wide_comp = {k: _one(r, wide[k], *k) for k, r in wide_rec.items()}
    for k, r in wide_rec.items():
        if r["contract"]["sha256"] != CONTRACT or not r["inputs"]["schedule_sha256"].startswith(wide_seeds.group(3)):
            raise FinalLookError(f"{wide[k]} is not a wide DEV receipt under the contract")
        same_run(wide_rec[(BUILDS[0], base)], r, wide[k])
    _disclosure(read_json(wide_disc), wide_comp[(BUILDS[1], base)], wide_disc)
    return {"rec": rec, "a": comp_a, "b": comp_b, "served": served, "family": family, "holm_m": holm_m,
            "outcome_a": _family_outcome(comp_a, family["comparisons"][0]["p_holm_family"]),
            "outcome_b": _family_outcome(comp_b, family["comparisons"][1]["p_holm_family"]),
            "served_build": served_build.group(1), "seeds": seeds.groups(), "report_seeds": report_seeds.groups()[:2],
            "wide_seeds": wide_seeds.groups()[:2], "wide": wide_comp, "wide_disc": read_json(wide_disc),
            "wide_p": dict(zip(BUILDS, holm([wide_comp[(b, base)]["p_one_sided"] for b in BUILDS]))),
            "base": base, "cand": cand, "paths": paths, "look": look, "prereg": prereg}


def macros(info: dict[str, Any], num: Callable, sha256: Callable[[str], str]) -> list[tuple[str, str]]:
    def ci4(c: dict[str, Any]) -> str:
        return f"{num(c['delta'], 4, True)} [{num(c['ci_low'], 4, True)}, {num(c['ci_high'], 4, True)}]"

    def guard(c: dict[str, Any], name: str) -> str:
        return num(next(g for g in c["decision"]["guards"] if g["name"] == name)["value"], 3, True)
    a, b, served, base, cand = info["a"], info["b"], info["served"], info["base"], info["cand"]
    disc = info["rec"]["family-disclosure-rc-vs-v2.json"]
    top = max(disc["family_contributions"], key=lambda f: f["contribution"])
    cal = a["null_calibration"]
    rc = next(x for x in info["rec"]["primary.json"]["arms"] if x["arm_id"] == RC)["primary"]
    six, sixb = (info["wide"][(bld, base)] for bld in BUILDS)
    wdisc = info["wide_disc"]
    return [("FlRcId", r"\texttt{" + RC + "}"), ("FlServedId", r"\texttt{" + SERVED + "}"),
            ("FlRcBuild", TOOL["git_sha"][:8]), ("FlServedBuild", info["served_build"]),
            ("FlSeeds", "--".join(info["seeds"])), ("ReportSeeds", "--".join(info["report_seeds"])),
            ("WideSeeds", "--".join(info["wide_seeds"])), ("FlScheduleSha", SCHEDULE[:12]),
            ("FlPrimarySha", sha256(info["look"]["primary.json"])[:12]),
            ("FlFamilySha", sha256(info["look"]["family.json"])[:12]),
            ("FlPreregSha", sha256(info["prereg"])[:12]),
            ("FlScheduled", str(rc["scheduled_cells"])), ("FlUninformative", str(rc["uninformative_cells"])),
            ("FlCells", str(a["cells"])), ("FlFamilies", str(a["families"])), ("FlHolmM", str(info["holm_m"])),
            ("FlVsBaseDelta", ci4(a)), ("FlVsBasePHolm", num(info["family"]["comparisons"][0]["p_holm_family"])),
            ("FlVsBaseOutcome", info["outcome_a"]), ("FlVsBaseGuards", str(len(a["decision"]["guards"]))),
            ("FlVsServedDelta", ci4(b)), ("FlVsServedMatched", num(b["matched"]["delta"], 3, True)),
            ("FlMatchedMult", str(b["matched"]["multiplier"])), ("FlVsServedOutcome", info["outcome_b"]),
            ("FlNonzeroFamilies", str(disc["nonzero_families"])), ("FlTopFamily", top["family_id"][3:11]),
            ("FlTopSharePct", f"{round(100 * top['contribution'] / disc['delta'])}\\%"),
            ("FlSignFlipP", num(disc["exact_sign_flip_p"])), ("FlNullFlipRate", num(cal["flip_guard_rule"]["rate"])),
            ("FlNullMaxRate", num(cal["max_false_promote_rate"], 2)),
            ("FlServedVsBaseDelta", ci4(served)), ("FlServedVsBaseOutcome", served["decision"]["outcome"]),
            ("FlServedRealism", guard(served, "realism_coverage_delta")), ("FlServedTaskLow", guard(served, "task_ci_low")),
            ("WideBuilds", str(len(BUILDS))), ("WideSixId", r"\texttt{" + BUILDS[0] + "}"),
            ("WideSixbId", r"\texttt{" + BUILDS[1] + "}"), ("WideSixOutcome", six["decision"]["outcome"]),
            ("WideSixPrivacy", guard(six, "privacy_coverage_delta")),
            ("WideSixbDelta", num(sixb["delta"], 3, True) + f" [{num(sixb['ci_low'], 3, True)}, {num(sixb['ci_high'], 3, True)}]"),
            ("WideSixbOutcome", sixb["decision"]["outcome"]), ("WideSixbPHolm", num(info["wide_p"][BUILDS[1]])),
            ("WideSixbVsCand", num(info["wide"][(BUILDS[1], cand)]["delta"], 3, True)),
            ("WideSixbNonzero", str(wdisc["nonzero_families"])), ("WideSixbFamilies", str(wdisc["families"])),
            ("WideSixbSignFlipP", num(wdisc["exact_sign_flip_p"]))]


def claims(info: dict[str, Any], report_comp: dict[str, Any]) -> list[tuple[str, bool]]:
    """The final-look prose (abstract, sections 6 and 8, appendix C), against the receipts."""
    a, b, served, base, cand = info["a"], info["b"], info["served"], info["base"], info["cand"]
    disc = info["rec"]["family-disclosure-rc-vs-v2.json"]
    top = max(disc["family_contributions"], key=lambda f: f["contribution"])
    cal = a["null_calibration"]
    failing = {g["name"] for g in served["decision"]["guards"] if g["check"] == "fail"}
    six, sixb = (info["wide"][(bld, base)] for bld in BUILDS)
    return [
        ("the candidate is PROMOTE against the previous release on the family Holm p, every guard passing",
         info["outcome_a"] == "PROMOTE" and all(g["check"] == "pass" for g in a["decision"]["guards"])),
        ("against the released engine the candidate is below zero, its interval contains zero, HOLD",
         b["delta"] < 0 and b["ci_low"] < 0 < b["ci_high"] and b["matched"]["delta"] < 0 and info["outcome_b"] == "HOLD"),
        ("the ship rule's second condition alone fails", info["outcome_a"] == "PROMOTE" and b["delta"] < 0),
        ("the gain comes from a few families, all positive, the largest multiclass",
         all(f["contribution"] >= 0 for f in disc["family_contributions"]) and top["task"] == "multiclass"
         and disc["nonzero_families"] < disc["families"]),
        ("the contract's null promotes at the same rate with or without the flip guards, above its limit",
         cal["flip_guard_rule"]["rate"] == cal["statistical_pass"]["rate"] > cal["max_false_promote_rate"]
         and not cal["flip_guard_rule"]["acceptable"]),
        ("the released engine is HOLD against the previous release on the look; realism and tasks fail",
         served["decision"]["outcome"] == "HOLD" and failing == {"task_ci_low", "realism_coverage_delta"}
         and served["ci_low"] < 0 < served["ci_high"]),
        ("the released engine is also HOLD against the previous release on the report look",
         (report_comp["id"], report_comp["baseline"], report_comp["decision"]["outcome"]) == (cand, base, "HOLD")),
        ("on wide DEV cand6 failed only privacy and cand6b was PROMOTE, at or above the released engine",
         six["decision"]["outcome"] != "PROMOTE" and [g["name"] for g in six["decision"]["guards"] if g["check"] == "fail"]
         == ["privacy_coverage_delta"] and sixb["decision"]["outcome"] == "PROMOTE" and info["wide_p"][BUILDS[1]] < 0.05
         and info["wide"][(BUILDS[1], cand)]["delta"] >= 0),
        ("on wide DEV cand6b's gain also came from a minority of families",
         info["wide_disc"]["nonzero_families"] < info["wide_disc"]["families"] / 2),
    ]
