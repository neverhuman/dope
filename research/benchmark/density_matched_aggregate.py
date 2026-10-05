"""Complete descriptive matched aggregation; no selection or release scores."""
from collections import Counter, defaultdict
import math
from statistics import median

from research.benchmark.density_publication_inputs import digest_identity, require

PROFILES = ('features12_steps512', 'features12_steps2048',
            'features24_steps512', 'features24_steps2048')
METHODS = ('GaussianCopula', 'independent_marginals', 'Chow-Liu')
VARIANTS = ('default', 'native_selected')
SEEDS, SIZES = (101, 211, 307), (1, 4)
AUDITORS = ('catboost', 'linear', 'mlp')
CONFIGS = [('DOPE', p) for p in PROFILES] + [(m, v) for m in METHODS for v in VARIANTS]


def aggregate(cells):
    """Caller must first verify complete frozen closure, metrics and matched views."""
    datasets = {c['dataset'] for c in cells}
    require(len(datasets) == 100 and all(type(d) is str for d in datasets),
            'complete 100-lineage inventory required')
    expected = {digest_identity([d, m, v, 11, s, z]) for d in datasets
                for m, v in CONFIGS for s in SEEDS for z in SIZES}
    actual = []
    groups = defaultdict(list)
    for c in cells:
        require(type(c['fit_seed']) is int and c['fit_seed'] == 11
                and type(c['sample_seed']) is int and c['sample_seed'] in SEEDS
                and type(c['size_multiplier']) is int and c['size_multiplier'] in SIZES,
                'frozen sample schedule differs')
        require(c['counts_as_dope_win'] is False
                and all(c[k] is None for k in ('mfs_v2', 'ptf_v1',
                                              'release_safe_l3', 'paired_superiority')),
                'research cell acquired a claim')
        require(type(c['charged_artifact_bytes']) is int and c['charged_artifact_bytes'] >= 0
                and type(c['artifact_within_l3_cap']) is bool
                and (not c['artifact_within_l3_cap'] or c['charged_artifact_bytes'] <= 10240),
                'invalid artifact charge')
        actual.append(digest_identity([c['dataset'], c['method'], c['configuration'],
                                       c['fit_seed'], c['sample_seed'], c['size_multiplier']]))
        groups[c['dataset'], c['method'], c['configuration'], c['size_multiplier']].append(c)
    require(len(actual) == len(set(actual)) == 6000 and set(actual) == expected,
            'complete matched frozen matrix required')
    summary, by_group = [], {}
    for key, values in sorted(groups.items()):
        values.sort(key=lambda c: c['sample_seed'])
        require(len({(c['charged_artifact_bytes'], c['artifact_within_l3_cap'])
                     for c in values}) == 1, 'sample group artifact charges differ')
        utility = {}
        for auditor in AUDITORS:
            scores = []
            for c in values:
                u = (c['utility'] or {}).get(auditor, {})
                value = (u.get('retention') if c['status'] == 'ok'
                         and u.get('informative') is True else None)
                require(value is None or (type(value) in (int, float) and math.isfinite(value)),
                        'invalid common retention')
                scores.append(value)
            complete = all(x is not None for x in scores)
            utility[auditor] = dict(complete_informative_sample_group=complete,
                                   sample_retention_values=scores,
                                   median_retention=median(scores) if complete else None)
        row = dict(dataset=key[0], method=key[1], configuration=key[2], size_multiplier=key[3],
                   fit_seed=11, sample_cells=3, statuses=dict(Counter(c['status'] for c in values)),
                   charged_artifact_bytes=values[0]['charged_artifact_bytes'],
                   artifact_within_l3_cap=values[0]['artifact_within_l3_cap'], utility=utility)
        summary.append(row)
        by_group[key] = row
    panels = []
    for method, configuration in CONFIGS:
        for size in SIZES:
            selected = [by_group[d, method, configuration, size] for d in sorted(datasets)]
            utility = {}
            for a in AUDITORS:
                scores = [g['utility'][a]['median_retention'] for g in selected
                          if g['utility'][a]['complete_informative_sample_group']]
                utility[a] = dict(complete_informative_lineages=len(scores),
                                  median_of_lineage_sample_medians=median(scores) if scores else None)
            panels.append(dict(method=method, configuration=configuration, size_multiplier=size,
                               lineages=100, utility=utility,
                               status_counts=dict(Counter(c['status'] for c in cells
                                   if (c['method'], c['configuration'], c['size_multiplier'])
                                   == (method, configuration, size)))))
    paired = []
    for profile in PROFILES:
        for method in METHODS:
            for variant in VARIANTS:
                for size in SIZES:
                    for a in AUDITORS:
                        rows = []
                        for d in sorted(datasets):
                            x = by_group[d, 'DOPE', profile, size]['utility'][a]['median_retention']
                            y = by_group[d, method, variant, size]['utility'][a]['median_retention']
                            if x is not None and y is not None:
                                rows.append((d, x, y))
                        paired.append(dict(dope_profile=profile, method=method, configuration=variant,
                            size_multiplier=size, auditor=a, complete_paired_lineages=len(rows),
                            dataset_ids=[r[0] for r in rows],
                            dope_matched_median=median(r[1] for r in rows) if rows else None,
                            baseline_matched_median=median(r[2] for r in rows) if rows else None,
                            median_dataset_difference=median(r[1] - r[2] for r in rows) if rows else None,
                            analysis='descriptive paired validation; one fit seed',
                            paired_superiority=None, counts_as_dope_win=False))
    return dict(summary=summary, configuration_panels=panels, paired_descriptive=paired,
                frozen_logical_cells=6000, datasets=100, global_family_selected=False,
                production_certified=False, official_tests_opened=False, mfs_v2=None,
                ptf_v1=None, release_safe_l3=None, paired_superiority=None, counts_as_dope_win=False)
