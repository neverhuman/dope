"""Regenerate SVG/PDF from the committed rights-safe progress report only."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics

from research.benchmark import publish_dope_refinement_progress as report

REPORT = '4b617c02d7bf93aa9bc7a35f34beb8ffb96f00f21bd96c2c28488251d7565e61'


def render(value, out):
    os.environ['MPLCONFIGDIR'] = str(Path('target/matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'dope-refinement-progress-v1'
    matplotlib.rcParams['svg.fonttype'] = 'none'
    groups = defaultdict(list)
    for c in value['cells'] + value['matched_reference_cells']:
        if c['size_multiplier'] == 4:
            groups[c['method'], c['configuration']].append(c['utility']['catboost']['retention'])
    labels, values, colors = [], [], []
    for (method, config), rows in sorted(groups.items()):
        labels.append(method + ' / ' + config); values.append(statistics.median(rows))
        colors.append('#3567a8' if method == 'DOPE' else '#cc7633' if config == 'native_selected' else '#919191')
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(labels, values, color=colors)
    ax.axvline(0, color='black', linewidth=.6)
    ax.set_xlabel('Validation retention, median of three sample seeds at 4n (CatBoost)')
    ax.set_title('One discovery lineage: closed GPU refinement progress batch')
    fig.text(.02, .015, '3 completed profiles / 24 planned fits. Baselines selected by own likelihood KPI. Official tests sealed; MFS/PTF/release/superiority null.', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out / (report.NAME + '.svg'), metadata={'Date': None})
    fig.savefig(out / (report.NAME + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    svg = out / (report.NAME + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()
    render(report.bound(args.results_dir / (report.NAME + '.json'), REPORT), args.results_dir)
    print(json.dumps({'figures_regenerated_from_committed_report': True, 'report_sha256': REPORT}))


if __name__ == '__main__': main()
