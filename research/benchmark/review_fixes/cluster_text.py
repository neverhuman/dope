"""Prose for the family-cluster bootstrap, rendered from the code that runs it.

stats.cluster_of gives each simulated name family in stats.SIMULATED one
cluster and every other lineage its own cluster. stats._hier_median_ci then
draws clusters with replacement and, inside each drawn cluster, lineages with
replacement. describe_clusters states that rule from the same tuple, so the
manuscript sentence follows the code.
"""

from __future__ import annotations

from research.benchmark.review_fixes.stats import SIMULATED

_COUNT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _series(names) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def describe_clusters(simulated=SIMULATED) -> str:
    """Manuscript sentences for the two-stage family-cluster bootstrap."""
    names = tuple(simulated)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("simulated families must be non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("simulated families repeat a name")
    shown = [name.replace("_", "\\_") for name in names]
    count = _COUNT.get(len(names), str(len(names)))
    noun = "family" if len(names) == 1 else "families"
    return (
        "The printed family-cluster intervals use a two-stage bootstrap. "
        f"Lineages in the {count} simulated name {noun} ({_series(shown)}) form one cluster "
        f"per family, and every other lineage is a singleton cluster. Each draw resamples "
        "clusters with replacement, then resamples lineages with replacement inside each "
        "drawn cluster."
    )
