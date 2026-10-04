"""Aggregate complete supplied density cells; verify receipt custody separately."""
from collections import Counter, defaultdict
import math
from statistics import median

METHODS = ('GaussianCopula', 'independent_marginals', 'Chow-Liu')
CONFIGS = ('default', 'native_selected')
SEEDS = (101, 211, 307)
SIZES = (1, 4)
AUDITORS = ('catboost', 'linear', 'mlp')
GATED_KEYS = ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def aggregate(cells, expected_dataset_ids):
    """Caller binds expected IDs, metrics, bytes and native choices to closed receipts.

    Check all 3,600 logical cells, including unavailable outcomes. This pure
    aggregation does not authenticate inputs, inspect files, select a generator,
    infer method failures, perform paired inference, or certify any release gate.
    """
    require(type(expected_dataset_ids) is tuple and len(expected_dataset_ids) == 100
            and all(type(d) is str and bool(d) for d in expected_dataset_ids)
            and len(set(expected_dataset_ids)) == 100, 'invalid expected lineage set')
    require(type(cells) is list and all(type(c) is dict for c in cells), 'invalid cell container')
    for c in cells:
        require(type(c['dataset']) is str and bool(c['dataset']), 'invalid dataset identity')
        require(type(c['method']) is str and c['method'] in METHODS, 'invalid method identity')
        require(type(c['configuration']) is str and c['configuration'] in CONFIGS,
                'invalid configuration identity')
        require(type(c['fit_seed']) is int and c['fit_seed'] == 11, 'invalid fit seed')
        require(type(c['sample_seed']) is int and c['sample_seed'] in SEEDS, 'invalid sample seed')
        require(type(c['size_multiplier']) is int and c['size_multiplier'] in SIZES,
                'invalid sample size')
        require(type(c['status']) is str and c['status'] in ('ok', 'failed', 'fit_unavailable'),
                'invalid status')
        require(c['counts_as_dope_win'] is False and all(c[k] is None for k in GATED_KEYS),
                'gated or win claim present')
        require(type(c['artifact_bytes']) is int and c['artifact_bytes'] >= 0,
                'invalid artifact bytes')
        require(c['utility'] is None or type(c['utility']) is dict, 'invalid utility container')
    ids = {c['dataset'] for c in cells}
    identities = [(c['dataset'], c['method'], c['configuration'], c['sample_seed'],
                   c['size_multiplier']) for c in cells]
    expected = {(d, m, k, s, z) for d in expected_dataset_ids for m in METHODS
                for k in CONFIGS for s in SEEDS for z in SIZES}
    require(ids == set(expected_dataset_ids)
            and len(identities) == len(set(identities)) == 3600 and set(identities) == expected,
            'complete expected all100 density matrix required')
    groups = defaultdict(list)
    for c in cells:
        groups[(c['dataset'], c['method'], c['configuration'], c['size_multiplier'])].append(c)
    lineages = []
    for (d, m, k, z), values in sorted(groups.items()):
        values.sort(key=lambda c: c['sample_seed'])
        require(len({c['artifact_bytes'] for c in values}) == 1,
                'artifact charges disagree within lineage')
        utility = {}
        for auditor in AUDITORS:
            scores = []
            for c in values:
                u = (c['utility'] or {}).get(auditor, {})
                require(type(u) is dict, 'invalid auditor utility')
                x = u.get('retention') if c['status'] == 'ok' and u.get('informative') is True else None
                require(x is None or (type(x) in (int, float) and math.isfinite(x)),
                        'invalid common utility')
                scores.append(x)
            complete = all(x is not None for x in scores)
            utility[auditor] = {'complete_informative_sample_group': complete,
                                'sample_retention_values': scores,
                                'median_retention': median(scores) if complete else None}
        lineages.append({'dataset': d, 'method': m, 'configuration': k, 'size_multiplier': z,
                         'fit_seed': 11, 'sample_cells': 3, 'artifact_bytes': values[0]['artifact_bytes'],
                         'artifact_within_l3_cap': values[0]['artifact_bytes'] <= 10240,
                         'statuses': dict(Counter(c['status'] for c in values)), 'utility': utility})
    panels = []
    for method in METHODS:
        for configuration in CONFIGS:
            for size in SIZES:
                selected = [g for g in lineages if (g['method'], g['configuration'], g['size_multiplier'])
                            == (method, configuration, size)]
                audits = {}
                for auditor in AUDITORS:
                    scores = [g['utility'][auditor]['median_retention'] for g in selected
                              if g['utility'][auditor]['complete_informative_sample_group']]
                    audits[auditor] = {'complete_informative_lineages': len(scores),
                                      'median_of_lineage_sample_medians': median(scores) if scores else None}
                panels.append({'method': method, 'configuration': configuration, 'size_multiplier': size,
                    'lineages': 100,
                    'lineage_status_counts': dict(Counter('ok' if g['statuses'] == {'ok': 3}
                        else 'incomplete_or_unavailable' for g in selected)),
                    'lineages_within_l3_byte_cap': sum(g['artifact_within_l3_cap'] for g in selected),
                    'utility': audits,
                    'comparison_scope': 'descriptive official-training-derived validation only; no paired inference, fit uncertainty or release score',
                    'study_reference_implementation': method in ('independent_marginals', 'Chow-Liu'),
                    'counts_as_dope_win': False})
    return lineages, panels
