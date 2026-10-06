"""Describe existing matched validation utility, without selecting a DOPE family."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median

from research.benchmark.publish_checkpoint_index import bound_bytes, csv_text, encoded
from research.benchmark.publish_s3_matched import schema

RESULTS = Path(__file__).parent / 'results'
SOURCE_SHA = '51dc00cf85cc5160cf4f496a2357722f5c6116297b084ad3271bdbac475982a6'
PANELS = {
    'arf-matched-population-validation.json': 'c69ce66e79b234bf5d1f9d56938450ba253ae39a3f6655a6fd586f890b8992f7',
    'density-matched-population-validation.json': '5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a',
}
METHODS = {
    'DOPE': ('density-matched-population-validation.json', 'features12_steps2048'),
    'ARF': ('arf-matched-population-validation.json', 'native_selected'),
    'GaussianCopula': ('density-matched-population-validation.json', 'native_selected'),
    'Chow-Liu': ('density-matched-population-validation.json', 'native_selected'),
    'independent_marginals': ('density-matched-population-validation.json', 'native_selected'),
}
AUDITORS = ('catboost', 'linear', 'mlp')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def scalar(value):
    if value == '':
        return None
    result = float(value)
    require(math.isfinite(result), 'nonfinite metric')
    return result


def summarize(body):
    """One fit, three sample medians, then a common informative lineage median."""
    selected = {}
    for row in csv.DictReader(io.StringIO(body.decode())):
        method = row['method']
        if (row['source_panel'], row['configuration']) != METHODS.get(method):
            continue
        require(row['source_sha256'] == PANELS[row['source_panel']], 'source panel pin differs')
        require(row['fit_seed'] == '11' and row['size_multiplier'] in ('1', '4'), 'fit or size identity differs')
        require(row['sample_seeds'] == '101,211,307', 'sample seed schedule differs')
        require(row['complete_three_sample_group'] in ('True', 'False'), 'typed complete three-seed flag required')
        statuses = json.loads(row['status_counts'])
        require(isinstance(statuses, dict) and all(type(n) is int and n > 0 for n in statuses.values())
                and sum(statuses.values()) == 3, 'three-seed status count differs')
        complete = row['complete_three_sample_group'] == 'True'
        require(complete == (statuses == {'ok': 3}), 'complete three-seed evidence required')
        charge = int(row['charged_artifact_bytes'])
        require(str(charge) == row['charged_artifact_bytes'] and charge >= 0, 'invalid artifact charge')
        key = (method, row['dataset'], int(row['size_multiplier']))
        require(key not in selected, 'duplicate method-lineage-size identity')
        for auditor in AUDITORS:
            for loss in ('tstr_mse', 'trtr_mse'):
                value = scalar(row[auditor + '_' + loss])
                require((value is not None and value >= 0) if complete else value is None,
                        'missing measured loss or partial median in incomplete group')
            retention = scalar(row[auditor + '_retention'])
            require(complete or retention is None, 'partial retention in incomplete group')
        selected[key] = row
    datasets = {method: {d for m, d, size in selected if m == method and size == 1} for method in METHODS}
    require(all(len(ds) == 100 for ds in datasets.values()), 'complete 100-lineage source groups required')
    require(len(set(map(frozenset, datasets.values()))) == 1, 'method lineages differ')
    require(set(selected) == {(m, d, size) for m, ds in datasets.items() for d in ds for size in (1, 4)},
            'method sample-size coverage differs')
    coverage = {m: {str(size): dict(planned_groups=100,
        complete_three_sample_groups=sum(selected[(m, d, size)]['complete_three_sample_group'] == 'True' for d in datasets[m]),
        incomplete_datasets=sorted(d for d in datasets[m] if selected[(m, d, size)]['complete_three_sample_group'] != 'True'))
        for size in (1, 4)} for m in METHODS}
    summaries = []
    cohorts = {}
    for auditor in AUDITORS:
        for size in (1, 4):
            common = sorted(d for d in datasets['DOPE'] if all(
                scalar(selected[(m, d, size)][auditor + '_retention']) is not None for m in METHODS))
            require(common, 'no common informative lineage')
            cohort_sha = hashlib.sha256(encoded(common)).hexdigest()
            cohorts[auditor + '_size' + str(size)] = dict(datasets=common, sha256=cohort_sha)
            for dataset in common:
                controls = {scalar(selected[(m, dataset, size)][auditor + '_trtr_mse']) for m in METHODS}
                require(len(controls) == 1, 'matched real-vs-real controls differ')
            for method, (_, configuration) in METHODS.items():
                values = [scalar(selected[(method, d, size)][auditor + '_retention']) for d in common]
                dope = [scalar(selected[('DOPE', d, size)][auditor + '_retention']) for d in common]
                charges = [int(selected[(method, d, size)]['charged_artifact_bytes']) for d in common]
                summaries.append(dict(method=method, configuration=configuration, auditor=auditor,
                    fit_seed=11, size_multiplier=size, sample_seeds='101,211,307',
                    informative_matched_lineages=len(common), cohort_sha256=cohort_sha,
                    median_null_normalized_tstr_trtr_retention=median(values),
                    median_paired_dope_minus_method=median(x - y for x, y in zip(dope, values)),
                    median_charged_artifact_bytes=median(charges), maximum_charged_artifact_bytes=max(charges)))
    return summaries, cohorts, coverage


def build(source):
    body = bound_bytes(Path(source).absolute(), SOURCE_SHA)
    rows, cohorts, coverage = summarize(body)
    report = dict(format='matched-validation-descriptive-figure-v1', version=1,
        scope='Previously measured official-training-derived validation; unconstrained artifacts; one fit seed.',
        aggregation='Median of sample seeds 101/211/307 per lineage, then median over the common informative cohort.',
        dope_configuration='Fixed features12_steps2048 for description; no family selection from this table.',
        baseline_configuration='Native-selected in the frozen source panels; native KPI values are not compared here.',
        raw_mse_comparison_across_datasets=False, confidence_interval_measured=False,
        paired_significance_measured=False, complete_neural_comparison=False, full_five_fit_seed_coverage=False,
        scientific_execution=False, official_tests_opened=False, production_certified=False,
        native_selection_changed=False, native_values_ranked_across_methods=False,
        mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None,
        source_csv=dict(path='research/benchmark/results/checkpoint-figure-input-index/figure-kpis.csv',
                        sha256=SOURCE_SHA, bytes=len(body)),
        source_panels=PANELS, rows=rows, cohorts=cohorts, source_group_coverage=coverage,
        publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    require(bound_bytes(Path(source).absolute(), SOURCE_SHA) == body, 'source changed during projection')
    return report, csv_text(rows, tuple(rows[0]))


def render(report, output):
    """Pinned ordinary plotting; no new data, models, or auditors."""
    import matplotlib
    require(matplotlib.__version__ == '3.10.8', 'frozen matplotlib version required')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, ft2font, font_manager
    require(ft2font.__freetype_version__ == '2.6.1', 'frozen FreeType version required')
    font_sha = hashlib.sha256(Path(font_manager.findfont('DejaVu Sans')).read_bytes()).hexdigest()
    require(font_sha == '3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22',
            'frozen figure font required')
    matplotlib.rcParams.update({'svg.hashsalt': 'matched-validation-descriptive-v1', 'font.size': 9,
                               'font.family': 'DejaVu Sans'})
    labels = ['DOPE', 'ARF', 'Gaussian copula', 'Chow–Liu', 'Independent marginals']
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=True)
    for ax, auditor, title in zip(axes, AUDITORS, ['CatBoost', 'Linear', 'MLP']):
        rows = {r['method']: r for r in report['rows'] if r['auditor'] == auditor and r['size_multiplier'] == 4}
        values = [rows[m]['median_null_normalized_tstr_trtr_retention'] for m in METHODS]
        ax.scatter(range(5), values, c=['#205db0'] + ['#555555'] * 4, s=32, zorder=3)
        ax.axhline(1, color='#aaaaaa', lw=.7, ls='--')
        ax.set_title(f"{title}\n{rows['DOPE']['informative_matched_lineages']} matched informative lineages", fontsize=10)
        ax.set_xticks(range(5), labels, rotation=50, ha='right')
        ax.set_ylim(min(-.12, min(values) - .1), max(1.18, max(values) + .1))
        ax.grid(axis='y', color='#eeeeee'); ax.spines[['right', 'top']].set_visible(False)
        for index, value in enumerate(values):
            ax.annotate(f'{value:.3f}', (index, value), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    axes[0].set_ylabel('Median retention\n(null-normalized TSTR/TRTR)')
    fig.suptitle('Measured validation utility · 4n synthetic rows · one fit seed', fontsize=12)
    fig.text(.02, .015, 'Median of three sample seeds per lineage, then matched-lineage median. Native-selected baselines; fixed DOPE features12_steps2048.\nTraining-derived validation only; artifacts unconstrained. Full neural comparison and production certification remain incomplete.', fontsize=8)
    fig.tight_layout(rect=[0, .10, 1, .91])
    outputs = {}
    for suffix, metadata in [('svg', {'Date': None, 'Creator': 'Receipt-backed validation summary'}),
                             ('pdf', {'CreationDate': None, 'ModDate': None, 'Creator': 'Receipt-backed validation summary'})]:
        path = output / ('measured-validation-utility-4n.' + suffix)
        fig.savefig(path, metadata=metadata)
        body = path.read_bytes(); outputs[path.name] = dict(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
    plt.close(fig)
    return dict(matplotlib_version=matplotlib.__version__, freetype_version=ft2font.__freetype_version__,
                font_sha256=font_sha, source_csv_sha256=SOURCE_SHA,
                publisher_sha256=report['publisher_sha256'], outputs=outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=RESULTS / 'checkpoint-figure-input-index/figure-kpis.csv')
    parser.add_argument('--output-dir', type=Path, default=RESULTS / 'matched-validation-descriptive-figure')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    report, table = build(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'publication.json').write_bytes(encoded(report))
    (args.output_dir / 'publication.schema.json').write_bytes(encoded(schema(report)))
    (args.output_dir / 'figure-kpis.csv').write_bytes(table)
    if args.figures:
        receipt = render(report, args.output_dir)
        (args.output_dir / 'figures.receipt.json').write_bytes(encoded(receipt))
        (args.output_dir / 'figures.receipt.schema.json').write_bytes(encoded(schema(receipt)))


if __name__ == '__main__':
    main()
