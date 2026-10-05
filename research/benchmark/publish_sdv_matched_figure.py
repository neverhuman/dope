"""Export paired validation medians from the committed matched report only."""
import argparse
import os
from pathlib import Path

from research.benchmark import publish_sdv_matched as report

REPORT_SHA256 = '4dc367eb78159a0386882d23ae1f99b0aa2cc9f425d1aae499b6f3453222a44e'


def render(value, out):
    report.tables(value)
    os.environ['MPLCONFIGDIR'] = str(Path('target/matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'sdv-matched-validation-v1'
    matplotlib.rcParams['svg.fonttype'] = 'none'
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, size in zip(axes, (1, 4)):
        rows = [r for r in value['paired_descriptive'] if r['dope_profile'] == 'features12_steps2048'
                and r['auditor'] == 'catboost' and r['size_multiplier'] == size]
        for i, row in enumerate(rows):
            x, y = row['dope_matched_median'], row['baseline_matched_median']
            if x is None or y is None: continue
            ax.plot([y, x], [i, i], color='#a6a6a6', linewidth=2)
            ax.scatter([x], [i], color='#3567a8', label='Fixed DOPE 12/2048' if i == 0 else None, zorder=3)
            ax.scatter([y], [i], color='#cc7633', marker='s', label='Baseline' if i == 0 else None, zorder=3)
            ax.annotate(f"paired n={row['complete_paired_lineages']}", (max(x, y), i),
                        xytext=(6, 6), textcoords='offset points', fontsize=8)
        ax.set_yticks(range(len(rows)), [r['method'] + ' / ' + r['configuration'] for r in rows])
        ax.axvline(0, color='black', linewidth=.6); ax.set_title(f'{size}n synthetic rows')
        ax.set_xlabel('Median paired-lineage validation retention (CatBoost)')
        ax.margins(x=.3); ax.grid(axis='x', alpha=.2); ax.legend(loc='lower left', fontsize=8)
    figure.suptitle('S3 matched neural comparison: fixed DOPE profile and original native baselines')
    figure.text(.02, .018, '100 lineages accounted; available paired subset only; one fit seed / three sample seeds.\nNative KPI selects baselines. No method conclusions from scheduling cutoffs. Official tests sealed; gated scores null.', fontsize=8)
    figure.tight_layout(rect=(0, .09, 1, .95))
    for extension, metadata in (('svg', {'Date': None}),
                                ('pdf', {'CreationDate': None, 'ModDate': None})):
        figure.savefig(out / (report.NAME + '.' + extension), metadata=metadata)
    plt.close(figure)
    svg = out / (report.NAME + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    render(report.committed(args.results_dir / (report.NAME + '.json'), REPORT_SHA256), args.results_dir)


if __name__ == '__main__': main()
