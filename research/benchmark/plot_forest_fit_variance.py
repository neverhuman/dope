"""Plot fit means and sample SD; no confidence-interval or population claim."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
panel = json.loads((root / 'panel.json').read_text())
assert panel['five_fit_default_cohort_complete'] and panel['common_sample_cells'] == panel['lineages'] * 30
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                           'svg.hashsalt': 'forest-two-lineage-fivefit-v1',
                           'axes.spines.top': False, 'axes.spines.right': False})
if panel['lineages'] > 2:
    matplotlib.rcParams['pdf.fonttype'] = 42
fig, axes = plt.subplots(1, 3, figsize=(max(10.4, panel['lineages'] * 2.1), 3.5))
datasets = sorted({r['dataset'] for r in panel['summary']})
metrics = [('marginal_error_mean', 'Marginal KS/TV error'),
           ('c2st_catboost_auc', 'CatBoost C2ST AUC'),
           ('catboost_retention', 'CatBoost utility retention')]
palette = ['#31688e', '#35b779', '#443983', '#21918c', '#7ad151', '#d9a441']
colors = [palette[i % len(palette)] for i in range(len(datasets))]
for ax, (metric, label) in zip(axes, metrics):
    for x, (dataset, color) in enumerate(zip(datasets, colors)):
        summary = next(r for r in panel['summary'] if r['dataset'] == dataset and r['row_multiplier'] == 4)
        stat = summary['metrics'][metric]
        fits = [r for r in panel['per_fit'] if r['dataset'] == dataset and r['row_multiplier'] == 4]
        fits.sort(key=lambda r: r['fit_seed'])
        if stat['measured_fits'] == 5:
            points = [r['metrics'][metric] for r in fits]
            ax.scatter([x - .16 + .08 * i for i in range(5)], points, color=color, s=25, zorder=3)
            ax.errorbar(x + .24, stat['mean'], yerr=stat['between_fit_sample_sd'],
                        fmt='s', color='#333333', capsize=4, markersize=4)
        else:
            ax.text(x, .48, 'Uninformative\n0/5 measured', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', color='#555555', fontsize=8)
    ax.set_xticks(list(range(len(datasets))), [d[:8] + '\u2026' for d in datasets])
    if len(datasets) > 2:
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.set_xlim(-.35, len(datasets) - .55)
    ax.set_ylabel(label)
    ax.set_xlabel('Frozen lineage ID')
    ax.grid(axis='y', alpha=.18)
scope = 'first two' if panel['lineages'] == 2 else str(panel['lineages'])
fig.suptitle('Forest-Flow author defaults: ' + scope + ' complete five-fit lineages, 4n', fontsize=11)
fig.text(.5, .01, 'Dots: mean of three samples within each fit. Squares: five-fit mean \u00b1 sample SD (not a CI). Validation only.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, .06, 1, .93))
fig.savefig(root / 'fit-variance.svg', metadata={'Date': None, 'Creator': 'Dope research benchmark'})
# Canonicalize generated path whitespace for projected publications while
# preserving the byte replay of earlier unprojected figures.
if panel['inputs'].get('cohort_projection') is not None:
    svg = root / 'fit-variance.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
fig.savefig(root / 'fit-variance.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'Dope research benchmark'})
fig.savefig(root / 'fit-variance-preview.png', dpi=140)
plt.close(fig)
