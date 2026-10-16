"""Family means, worst-family gate, and paired cluster bootstrap (T-031/T-036).

Resample source family -> dataset -> seed, preserving every auditor family in
one paired draw. Input rows are seed-level paired retention observations.
"""
import math

from .common import MetricError, count, failure, mean, number, require, unit

MASK = (1 << 64) - 1


class XorShift64:
    """Explicit portable RNG; zero seed maps to one. Modulo indexing is frozen."""
    def __init__(self, seed):
        self.state = count(seed) or 1

    def index(self, length):
        x = self.state
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        self.state = x & MASK
        return self.state % length


def _names(values):
    require(type(values) is list and bool(values), "invalid_families")
    require(all(type(v) is str and v for v in values), "invalid_families")
    require(len(values) == len(set(values)), "duplicate_family")
    return sorted(values)


def evaluate(data):
    try:
        require(type(data) is dict)
        families = _names(data.get("required_families"))
        rows = data.get("rows")
        if not rows:
            return failure("missing_family_or_cell", "incomplete")
        require(type(rows) is list, "invalid_rows")
        tree, seen = {}, set()
        for row in rows:
            require(type(row) is dict, "invalid_rows")
            for key in ("source_family", "dataset", "family", "auditor"):
                require(type(row.get(key)) is str and bool(row[key]), "invalid_cluster")
            require(row["family"] in families, "unexpected_family")
            seed = count(row.get("seed"))
            key = (row["source_family"], row["dataset"], seed, row["family"], row["auditor"])
            require(key not in seen, "duplicate_observation")
            seen.add(key)
            if row.get("transfer") is None:
                return failure("missing_family_or_cell", "incomplete")
            value = number(row["transfer"])
            cell = tree.setdefault(key[0], {}).setdefault(key[1], {}).setdefault(seed, {})
            cell.setdefault(row["family"], {})[row["auditor"]] = value
        # Every seed is a paired block containing every required family.
        for datasets in tree.values():
            for seeds in datasets.values():
                for cell in seeds.values():
                    if set(cell) != set(families):
                        return failure("missing_family_or_cell", "incomplete")
                    for family in families:
                        cell[family] = mean([cell[family][a] for a in sorted(cell[family])])
        sources = sorted(tree)

        def estimate(rng=None):
            source_estimates = []
            for i in range(len(sources)):
                datasets = tree[sources[i if rng is None else rng.index(len(sources))]]
                names = sorted(datasets)
                dataset_estimates = []
                for j in range(len(names)):
                    seeds = datasets[names[j if rng is None else rng.index(len(names))]]
                    ids = sorted(seeds)
                    cells = [seeds[ids[k if rng is None else rng.index(len(ids))]] for k in range(len(ids))]
                    dataset_estimates.append({f: mean([c[f] for c in cells]) for f in families})
                source_estimates.append({f: mean([d[f] for d in dataset_estimates]) for f in families})
            return {f: mean([s[f] for s in source_estimates]) for f in families}

        points = estimate()
        replicates = count(data.get("bootstrap_replicates", 1000), 1)
        require(replicates <= 100000, "bootstrap_budget_exceeded")
        alpha = unit(data.get("alpha", 0.05))
        require(0 < alpha < 0.5, "invalid_alpha")
        rng = XorShift64(data.get("bootstrap_seed", 1729))
        samples = sorted(min(estimate(rng).values()) for _ in range(replicates))
        lcb = samples[math.floor(alpha * (replicates - 1))]
        return {"status": "complete", "reason": None, "family_means": points,
                "robust_transfer": min(points.values()), "mean_transfer": mean(list(points.values())),
                "lcb": lcb}
    except MetricError as error:
        return failure(str(error))
