"""Plot fit means and sample SD; no confidence-interval or population claim."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
panel = json.loads((root / 'panel.json').read_text())
assert panel['five_fit_default_cohort_complete'] and panel['common_sample_cells'] == 60
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                           'svg.hashsalt': 'forest-two-lineage-fivefit-v1',
                           'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
datasets = sorted({r['dataset'] for r in panel['summary']})
metrics = [('marginal_error_mean', 'Marginal KS/TV error'),
           ('c2st_catboost_auc', 'CatBoost C2ST AUC'),
           ('catboost_retention', 'CatBoost utility retention')]
colors = ['#31688e', '#35b779']
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
    ax.set_xticks([0, 1], [d[:8] + '\u2026' for d in datasets])
    ax.set_xlim(-.35, 1.45)
    ax.set_ylabel(label)
    ax.set_xlabel('Frozen lineage ID')
    ax.grid(axis='y', alpha=.18)
fig.suptitle('Forest-Flow author defaults: first two complete five-fit lineages, 4n', fontsize=11)
fig.text(.5, .01, 'Dots: mean of three samples within each fit. Squares: five-fit mean \u00b1 sample SD (not a CI). Validation only.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, .06, 1, .93))
fig.savefig(root / 'fit-variance.svg', metadata={'Date': None, 'Creator': 'Dope research benchmark'})
fig.savefig(root / 'fit-variance.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'Dope research benchmark'})
fig.savefig(root / 'fit-variance-preview.png', dpi=140)
plt.close(fig)
