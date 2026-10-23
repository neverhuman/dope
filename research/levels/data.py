"""Fit-only mixed-type preprocessing; physical rows and opaque feature slots stay bound."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import numpy as np

from .contracts import PREPROCESSING, digest, require


@dataclass(frozen=True)
class Column:
    slot: int
    kind: str
    center: float = 0.
    scale: float = 1.
    labels: tuple = field(default=(), repr=False)

    def public(self):
        return dict(slot=self.slot, kind=self.kind, center=self.center, scale=self.scale,
                    categories=len(self.labels) + 2 if self.kind == "categorical" else 0)


@dataclass(frozen=True)
class Prepared:
    columns: tuple[Column, ...]
    values: np.ndarray = field(repr=False)
    observed: np.ndarray = field(repr=False)
    row_ids: tuple[str, ...] = field(repr=False)
    task: str

    @property
    def schema(self):
        return dict(preprocessing=PREPROCESSING, task=self.task, target_index=len(self.columns)-1,
                    output="positional_numeric_and_category_ids", columns=[c.public() for c in self.columns])


def role_indices(row_ids, seed, *, policy="iid", groups=None, times=None):
    """Four physical roles, selected without targets; group/time blocks are never shuffled apart."""
    require(len(row_ids) == len(set(row_ids)) and len(row_ids) >= 80, "insufficient_or_duplicate_rows")
    require(policy in ("iid", "group", "time"), "unknown_row_policy")
    def key(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).digest()
    if policy == "iid":
        blocks = [[i] for i in sorted(range(len(row_ids)), key=lambda i: key(row_ids[i]))]
    else:
        values = groups if policy == "group" else times
        require(values is not None and len(values) == len(row_ids), "physical_blocks_required")
        units = {}
        for i, value in enumerate(values):
            units.setdefault(value, []).append(i)
        require(len(units) >= 8, "insufficient_physical_blocks")
        order = sorted(units, key=key) if policy == "group" else sorted(units)
        blocks = [units[value] for value in order]
    # Boundaries are in whole independent units. Unequal group sizes remain visible.
    n = len(blocks)
    cuts = [0, int(n*.65), int(n*.80), int(n*.90), n]
    result = {name: [i for b in blocks[cuts[k]:cuts[k+1]] for i in b]
              for k, name in enumerate(("fit", "tune", "calibration", "confirmation"))}
    require(all(len(v) >= 4 for v in result.values()), "row_roles_unsupported")
    return result


def fit_transform(rows, kinds, slots, task):
    require(task in ("regression", "classification") and len(kinds) == len(slots)
            and len(kinds) >= 2 and len(set(slots)) == len(slots) and slots[-1] == 0,
            "invalid_target_last_schema")
    require(all(len(row) == len(kinds) for row in rows) and len(rows) >= 4, "invalid_context_shape")
    columns = []
    for j, (kind, slot) in enumerate(zip(kinds, slots)):
        values = [row[j] for row in rows if not missing(row[j])]
        require(kind in ("numeric", "categorical") and values, "unsupported_empty_or_typed_column")
        if kind == "numeric":
            values = np.asarray(values, dtype=np.float64)
            require(np.isfinite(values).all(), "nonfinite_context")
            q25, center, q75 = np.quantile(values, [.25, .5, .75])
            scale = max(float(q75-q25), float(np.std(values)), 1e-6)
            columns.append(Column(slot, kind, float(center), scale))
        else:
            columns.append(Column(slot, kind, labels=tuple(sorted(set(str(v) for v in values)))))
    return tuple(columns)


def missing(value):
    return value is None or (isinstance(value, str) and value.strip() in ("", "?", "NA", "NaN", "null"))


def transform(rows, columns, row_ids, task):
    require(len(rows) == len(row_ids) and all(len(r) == len(columns) for r in rows), "row_schema_drift")
    values, observed = np.zeros((len(rows), len(columns)), np.float32), np.zeros((len(rows), len(columns)), bool)
    for j, column in enumerate(columns):
        mapping = {label: k+2 for k, label in enumerate(column.labels)}
        for i, row in enumerate(rows):
            value = row[j]
            if missing(value):
                require(j != len(columns)-1, "missing_target_unsupported")
                continue
            observed[i, j] = True
            if column.kind == "numeric":
                number = float(value)
                require(np.isfinite(number), "nonfinite_value")
                values[i, j] = (number-column.center)/column.scale
            else:
                values[i, j] = mapping.get(str(value), 1)
                require(j != len(columns)-1 or values[i, j] >= 2, "unsupported_target_class")
    require(np.isfinite(values).all(), "normalized_value_overflow")
    return Prepared(columns, values, observed, tuple(row_ids), task)


def episode(rows, kinds, slots, row_ids, task, roles, config, cursor, seed):
    """Gradient episodes use fit context / tune inner queries; calibration/confirmation remain outside this module."""
    fit, query = roles["fit"], roles["tune"]
    require(not set(fit).intersection(query), "context_query_overlap")
    rng = np.random.default_rng(np.random.SeedSequence([seed, cursor]))
    chosen = sorted(rng.choice(fit, min(len(fit), config.max_context), replace=False).tolist())
    queries = sorted(rng.choice(query, min(len(query), config.max_query), replace=False).tolist())
    require(len(kinds)-1 <= config.max_features
            and len(chosen)*(len(kinds)-1) <= config.cell_cap, "episode_shape_unsupported")
    context_rows = [rows[i] for i in chosen]
    columns = fit_transform(context_rows, kinds, slots, task)
    require(all(len(c.labels)+2 <= config.max_categories for c in columns if c.kind == "categorical"),
            "category_envelope_unsupported")
    context = transform(context_rows, columns, [row_ids[i] for i in chosen], task)
    inner = transform([rows[i] for i in queries], columns, [row_ids[i] for i in queries], task)
    identity = digest(dict(context=context.row_ids, query=inner.row_ids, schema=context.schema,
                           recipe=PREPROCESSING, cursor=cursor, seed=seed))
    return context, inner, identity
