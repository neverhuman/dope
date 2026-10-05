"""Render the descriptive discovery panel from its committed report only."""
import argparse
import os
from pathlib import Path

from research.benchmark import publish_dope_refinement_discovery as report

REPORT_SHA256 = 'e0c9cd641ff4975126c0ecd3942315299ba14835b9fdad25df5f10d693b47bc4'


def render(value, out):
    os.environ['MPLCONFIGDIR'] = str(Path('target/matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'dope-refinement-discovery-v1'
    matplotlib.rcParams['svg.fonttype'] = 'none'
    figure, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
    for ax, size in zip(axes, report.SIZES):
        rows = [r for r in value['summary'] if r['size_multiplier'] == size]
        for i, row in enumerate(rows):
            utility = row['utility']['catboost']; x = utility['median_retention']
            if x is None: continue
            color = '#3567a8' if row['method'] == 'DOPE' else '#cc7633' if row['configuration'] == 'native_selected' else '#919191'
            ax.scatter([x], [i], color=color, zorder=3)
            ax.annotate(f"N={utility['informative_complete_lineages']}/{row['planned_lineages']}",
                        (x, i), xytext=(6, 3), textcoords='offset points', fontsize=8)
        ax.set_yticks(range(len(rows)), [r['method'] + ' / ' + r['configuration'] for r in rows])
        ax.axvline(0, color='black', linewidth=.6); ax.set_title(f'{size}n synthetic rows')
        ax.set_xlabel('Median dataset validation retention (CatBoost)')
        ax.margins(x=.3); ax.grid(axis='x', alpha=.2)
    figure.suptitle('Completed bounded GPU refinement discovery: six stratified S3 lineages')
    figure.text(.02, .015, '23 successful fits / 1 charged artifact rejection. One fit seed; three sample seeds reduced within lineage.\nBaselines select by own likelihood KPI. Availability differs; descriptive discovery subset. Official tests sealed; gated scores null.', fontsize=8)
    figure.tight_layout(rect=(0, .07, 1, .95))
    for extension, metadata in (('svg', {'Date': None}), ('pdf', {'CreationDate': None, 'ModDate': None})):
        figure.savefig(out / (report.NAME + '.' + extension), metadata=metadata)
    plt.close(figure)
    svg = out / (report.NAME + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    render(report.bound(args.results_dir / (report.NAME + '.json'), REPORT_SHA256), args.results_dir)


if __name__ == '__main__': main()
