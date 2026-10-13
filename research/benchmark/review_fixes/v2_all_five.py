"""The all-five-fits lineage sensitivity of the v2 panel.

predeclare_v2 estimands.lineage_value: a five-seed arm's lineage value is the median of its
complete fits and needs at least three; "Sensitivity: lineages with all five fits." Here a lineage
enters an arm's row only when every declared fit seed has a complete fit value (three ok,
informative sample seeds) at that size and auditor. Levels cover every five-seed arm. Each
contrast pairs the DOPE headline arm with one five-seed arm on lineages where both have all five
fits. The nested bootstrap is the panel's. Every row is a sensitivity outside every Holm family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from research.benchmark.review_fixes import stats_v2 as S
from research.benchmark.review_fixes import v2_panel as V

FIT_SEEDS = (11, 23, 37, 53, 71)


def all_five(items: list[dict]) -> list[dict]:
    """The fit items of lineages that have a complete fit at every declared fit seed."""
    seeds: dict[str, set] = defaultdict(set)
    for item in items:
        seeds[item["dataset"]].add(item["fit_seed"])
    keep = {dataset for dataset, found in seeds.items() if found == set(FIT_SEEDS)}
    return [item for item in items if item["dataset"] in keep]


def arms_of(values: dict) -> list[tuple[str, str]]:
    present = {key[0] for key in values}
    return sorted(arm for arm in V.FIVE_SEED if arm in present)


def reduce(cells: list[dict], names: dict[str, str], draws: int = S.DRAWS, jobs: int | None = None,
           predeclare: bytes | None = None) -> dict:
    jobs = min(os.cpu_count() or 1, 16) if jobs is None else jobs
    values = V.fit_values(cells, names)
    arms = arms_of(values)
    level_tasks, contrast_tasks = [], []
    for size in V.SIZES:
        for auditor in V.AUDITORS:
            left = all_five(values.get((V.HEADLINE, size, auditor), []))
            for arm in arms:
                items = all_five(values.get((arm, size, auditor), []))
                if items:
                    level_tasks.append(("level", (arm, size, auditor), items,
                                        f"all-five|level|{arm}|{size}|{auditor}", len(FIT_SEEDS), draws))
                if arm != V.HEADLINE and left and items:
                    contrast_tasks.append(((arm, size, auditor), left, items,
                                           f"all-five|paired|{arm}|{size}|{auditor}", arm, draws))
    levels = []
    for result in V._run(V._level_task, level_tasks, jobs):
        result.pop("kind")
        arm, size, auditor = result.pop("key")
        levels.append({"method": arm[0], "configuration": arm[1], "size": size, "auditor": auditor, **result})
    contrasts = []
    for result in V._run(V._contrast_task, contrast_tasks, jobs):
        arm, size, auditor = result.pop("key")
        contrasts.append({"method": arm[0], "configuration": arm[1], "size": size, "auditor": auditor,
                          "holm_family": None, **result})
    return {
        "format": "dope-review-fix-v2-all-five", "version": 1, "draws": draws, "fit_seeds": list(FIT_SEEDS),
        "predeclare_sha256": hashlib.sha256(V.PREDECLARE.read_bytes() if predeclare is None else predeclare).hexdigest(),
        "levels": levels, "contrasts": contrasts, "official_tests_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the v2 all-five-fits sensitivity from the v2 scalar ledger.")
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
