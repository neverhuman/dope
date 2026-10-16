"""Family means, worst-family gate, and paired cluster bootstrap (T-031/T-036).

Resample source family -> dataset -> seed, preserving every auditor family in
one paired draw. Input rows are seed-level paired retention observations.
"""
import math

from .common import MetricError, count, failure, mean, number, opt_default, require, unit

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


def _names(values, invalid="invalid_families", dup="duplicate_family"):
    require(type(values) is list and bool(values), invalid)
    require(all(type(v) is str and v for v in values), invalid)
    require(len(values) == len(set(values)), dup)
    return sorted(values)


def _parse_required_auditors(raw, families):
    if raw is None:
        return None
    require(type(raw) is dict and set(raw) == set(families), "invalid_auditors")
    return {family: set(_names(raw[family], "invalid_auditors", "duplicate_auditor")) for family in families}


def _parse_required_seeds(raw):
    if raw is None:
        return None
    require(type(raw) is list and bool(raw), "invalid_seeds")
    seeds = [count(s, 0) for s in raw]
    require(len(seeds) == len(set(seeds)), "duplicate_seed")
    return set(seeds)


def evaluate(data):
    try:
        require(type(data) is dict)
        families = _names(data.get("required_families"))
        declared_auditors = _parse_required_auditors(data.get("required_auditors"), families)
        declared_seeds = _parse_required_seeds(data.get("required_seeds"))
        rows = data.get("rows")
        if rows is None:
            return failure("missing_family_or_cell", "incomplete")
        require(type(rows) is list, "invalid_rows")
        if not rows:
            return failure("missing_family_or_cell", "incomplete")
        tree, seen = {}, set()
        observed_auditors = {f: set() for f in families}
        for row in rows:
            require(type(row) is dict, "invalid_rows")
            for key in ("source_family", "dataset", "family", "auditor"):
                require(type(row.get(key)) is str and bool(row[key]), "invalid_cluster")
            require(row["family"] in families, "unexpected_family")
            if declared_auditors is not None:
                require(row["auditor"] in declared_auditors[row["family"]], "unexpected_auditor")
            seed = count(row.get("seed"))
            if declared_seeds is not None:
                require(seed in declared_seeds, "unexpected_seed")
            key = (row["source_family"], row["dataset"], seed, row["family"], row["auditor"])
            require(key not in seen, "duplicate_observation")
            seen.add(key)
            row_status = row.get("status")
            if row_status is not None:
                require(type(row_status) is str, "invalid_row_status")
                if row_status == "failed":
                    return failure("row_failed", "failed")
                if row_status in ("incomplete", "uninformative"):
                    return failure("missing_family_or_cell", "incomplete")
                require(row_status in ("complete", "informative"), "invalid_row_status")
            if row.get("informative") is False or row.get("transfer") is None:
                return failure("missing_family_or_cell", "incomplete")
            value = min(1.0, number(row["transfer"]))
            observed_auditors[row["family"]].add(row["auditor"])
            cell = tree.setdefault(key[0], {}).setdefault(key[1], {}).setdefault(seed, {})
            cell.setdefault(row["family"], {})[row["auditor"]] = value
        required_auditors = declared_auditors if declared_auditors is not None else observed_auditors
        if any(not required_auditors[f] for f in families):
            return failure("missing_family_or_cell", "incomplete")
        # Every seed is a paired block containing every required family and every required auditor.
        for datasets in tree.values():
            for seeds in datasets.values():
                expected_seeds = declared_seeds if declared_seeds is not None else set(seeds)
                if set(seeds) != expected_seeds:
                    return failure("missing_family_or_cell", "incomplete")
                for cell in seeds.values():
                    if set(cell) != set(families):
                        return failure("missing_family_or_cell", "incomplete")
                    for family in families:
                        if set(cell[family]) != required_auditors[family]:
                            return failure("missing_family_or_cell", "incomplete")
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
        replicates = count(opt_default(data, "bootstrap_replicates", 1000), 1)
        require(replicates <= 100000, "bootstrap_budget_exceeded")
        alpha = unit(opt_default(data, "alpha", 0.05))
        require(0 < alpha < 0.5, "invalid_alpha")
        rng = XorShift64(opt_default(data, "bootstrap_seed", 1729))
        samples = sorted(min(estimate(rng).values()) for _ in range(replicates))
        lcb = samples[math.floor(alpha * (replicates - 1))]
        return {"status": "complete", "reason": None, "family_means": points,
                "robust_transfer": min(points.values()), "mean_transfer": mean(list(points.values())),
                "lcb": lcb}
    except MetricError as error:
        return failure(str(error))
