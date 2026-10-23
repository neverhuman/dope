"""Deterministic development readout; failed cells stay in the scheduled denominator."""
from collections import defaultdict
import numpy as np
from .contracts import LEVELS, require


KEY = ('family', 'table_seed', 'level', 'sample_seed', 'method')


def schedule(selected, table_seeds, sample_seeds):
    require(selected and len(set(selected)) == len(selected)
            and all(level in LEVELS for level in selected), 'comparison_levels_invalid')
    for seeds in (table_seeds, sample_seeds):
        require(1 <= len(seeds) <= 8 and len(set(seeds)) == len(seeds)
                and all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds),
                'comparison_seeds_invalid')
    # Family 9 is reserved; this command cannot select it or discover real datasets.
    return [dict(family=family, table_seed=table_seed, level=level,
                 sample_seed=sample_seed, method=method)
            for family in (7, 8) for table_seed in table_seeds for level in selected
            for sample_seed in sample_seeds for method in ('shared', 'baseline')]


def design(values, observed):
    values, observed = np.asarray(values), np.asarray(observed)
    require(values.ndim == 2 and values.shape == observed.shape and len(values) > 0
            and values.shape[1] >= 2 and observed.dtype == np.bool_
            and np.isfinite(values).all() and observed[:, -1].all(), 'comparison_rows_invalid')
    return np.column_stack((np.ones(len(values)), np.where(observed[:, :-1], values[:, :-1], 0),
                            ~observed[:, :-1])).astype(np.float64)


def ridge_loss(values, observed, query, classification):
    """Numeric predictors in the two owned dev mechanisms; binary Brier or regression MSE."""
    x, q = design(values, observed), design(query.values, query.observed)
    require(x.shape[1] == q.shape[1], 'comparison_schema_mismatch')
    if classification:
        require(np.isin(values[:, -1], [2, 3]).all()
                and np.isin(query.values[:, -1], [2, 3]).all(), 'comparison_binary_labels_invalid')
    y = values[:, -1].astype(np.float64) - (2 if classification else 0)
    gram = x.T @ x
    penalty = np.eye(x.shape[1]) * 1e-3 * max(np.mean(np.diag(gram)[1:]), 1e-12)
    penalty[0, 0] = 0
    penalty += np.eye(len(penalty)) * 1e-9
    prediction = q @ np.linalg.solve(gram + penalty, x.T @ y)
    if classification:
        prediction = np.clip(prediction, 0, 1)
    target = query.values[:, -1] - (2 if classification else 0)
    result = float(np.mean((target - prediction)**2))
    require(np.isfinite(result), 'comparison_loss_nonfinite')
    return result


def retention(null_loss, real_loss, synthetic_loss):
    require(all(np.isfinite(value) and value >= 0 for value in
                (null_loss, real_loss, synthetic_loss)), 'comparison_loss_invalid')
    gain = null_loss - real_loss
    return (null_loss - synthetic_loss) / gain if gain > max(1e-8, .01 * null_loss) else None


def summarize(cases, records):
    expected = {tuple(case[k] for k in KEY) for case in cases}
    actual = [tuple(record[k] for k in KEY) for record in records]
    require(len(expected) == len(cases) and len(actual) == len(set(actual))
            and set(actual) == expected, 'comparison_schedule_incomplete_or_duplicate')
    groups = defaultdict(list)
    for record in records:
        require(record['state'] in ('complete', 'failed'), 'comparison_state_invalid')
        if record['state'] == 'complete':
            require(all(np.isfinite(record[k]) and record[k] >= 0 for k in
                        ('synthetic_loss', 'real_loss', 'null_loss')), 'comparison_loss_invalid')
        groups[record['family'], record['level']].append(record)
    output = []
    for (family, level), group in sorted(groups.items()):
        methods = {}
        for method in ('baseline', 'shared'):
            selected = [r for r in group if r['method'] == method]
            valid = [r for r in selected if r['state'] == 'complete']
            informative = [r['retention'] for r in valid if r['retention'] is not None]
            require(all(np.isfinite(v) for v in informative), 'comparison_retention_nonfinite')
            methods[method] = dict(scheduled=len(selected), complete=len(valid), failures=len(selected)-len(valid),
                mean_loss=float(np.mean([r['synthetic_loss'] for r in valid])) if valid else None,
                informative_cells=len(informative), mean_retention=float(np.mean(informative)) if informative else None)
        pairs = defaultdict(dict)
        for row in group:
            if row['state'] == 'complete':
                pairs[row['table_seed'], row['sample_seed']][row['method']] = row['synthetic_loss']
        table_deltas = defaultdict(list)
        for (seed, _), pair in pairs.items():
            if set(pair) == {'baseline', 'shared'}:
                table_deltas[seed].append(pair['shared']-pair['baseline'])
        deltas = [float(np.mean(values)) for values in table_deltas.values()]
        output.append(dict(family=family, level=level, methods=methods,
            complete_pairs=sum(len(v) for v in table_deltas.values()),
            table_mean_loss_deltas=deltas,
            mean_paired_loss_delta=float(np.mean(deltas)) if deltas else None))
    failures = sum(r['state'] != 'complete' for r in records)
    return dict(format='dope-owned-comparison-summary-v2', scheduled=len(cases),
        complete=len(records)-failures, failures=failures, groups=output,
        all_cells_complete=failures == 0, superiority_qualified=False,
        qualification='owned_development_exploration_only',
        uncertainty='Two mechanisms; three default tables each. Sampling seeds share tables. '
                    'Paired deltas are descriptive, without external generalization or Elo claims.')
