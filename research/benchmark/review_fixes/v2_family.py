"""Source-family sensitivity of the v2 panel.

predeclare_v2 inherits the v1 family rules unchanged (supersedes.inherited_unchanged):
the four simulated name prefixes are the families; the excluding-simulated rows drop
them and repeat the paired analysis on the remaining lineages; inside one family the
family is fixed and its lineages are resampled. The v2 nested bootstrap also resamples
fit seeds inside each drawn lineage, separately for each arm. The remaining lineages
are exactly the excluding-simulated rows, so no separate remainder family is
reported. Every row is a sensitivity outside every Holm family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from research.benchmark.review_fixes import stats_v2 as S
from research.benchmark.review_fixes import v2_panel as V

SIMULATED = ("feynman", "strogatz", "fri", "bng")
SCOPES = ("excluding_simulated",) + SIMULATED
COMPARATOR_FAMILIES = ("primary", "density")


def in_scope(scope: str, family: str) -> bool:
    if scope == "excluding_simulated":
        return family not in SIMULATED
    if scope not in SIMULATED:
        raise ValueError(f"unknown family scope {scope}")
    return family == scope


def scoped(items: list[dict], scope: str) -> list[dict]:
    return [item for item in items if in_scope(scope, item["family"])]


def comparators(values: dict) -> list[tuple[str, str]]:
    """Declared primary and density arms that have fit values, in declared order."""
    present = {key[0] for key in values}
    return [arm for name in COMPARATOR_FAMILIES for arm in V.FAMILIES[name] if arm in present]


def reduce(cells: list[dict], names: dict[str, str], draws: int = S.DRAWS, jobs: int | None = None,
           predeclare: bytes | None = None) -> dict:
    jobs = min(os.cpu_count() or 1, 16) if jobs is None else jobs
    values = V.fit_values(cells, names)
    arms = comparators(values)
    level_tasks, contrast_tasks = [], []
    for scope in SCOPES:
        for size in V.SIZES:
            for auditor in V.AUDITORS:
                left = scoped(values.get((V.HEADLINE, size, auditor), []), scope)
                if not left:
                    continue
                level_tasks.append(("level", (scope, size, auditor), left,
                                    f"family|{scope}|level|{size}|{auditor}", S.MIN_FITS_FIVE_SEED, draws))
                for arm in arms:
                    right = scoped(values.get((arm, size, auditor), []), scope)
                    if right:
                        contrast_tasks.append(((scope, arm, size, auditor), left, right,
                                               f"family|{scope}|paired|{arm}|{size}|{auditor}", arm, draws))
    levels = []
    for result in V._run(V._level_task, level_tasks, jobs):
        result.pop("kind")
        scope, size, auditor = result.pop("key")
        levels.append({"scope": scope, "method": V.HEADLINE[0], "configuration": V.HEADLINE[1],
                       "size": size, "auditor": auditor, **result})
    contrasts = []
    for result in V._run(V._contrast_task, contrast_tasks, jobs):
        scope, arm, size, auditor = result.pop("key")
        contrasts.append({"scope": scope, "method": arm[0], "configuration": arm[1], "size": size,
                          "auditor": auditor, "holm_family": None, **result})
    return {
        "format": "dope-review-fix-v2-family", "version": 1, "draws": draws, "scopes": list(SCOPES),
        "predeclare_sha256": hashlib.sha256(V.PREDECLARE.read_bytes() if predeclare is None else predeclare).hexdigest(),
        "levels": levels, "contrasts": contrasts, "official_tests_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the v2 source-family sensitivity from the v2 scalar ledger.")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--lineage-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    names = V.names_of(json.loads(args.lineage_record.read_text()))
    args.out.write_text(V.canonical(reduce(V.load_ledger(args.ledger), names, jobs=args.jobs)))
    print(f"wrote {args.out} sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
