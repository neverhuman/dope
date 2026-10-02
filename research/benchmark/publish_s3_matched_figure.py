"""Render all declared S3 research configurations from the committed matched panel."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from .publish_s3_matched import CONFIGS, summarize

LABELS = ['DOPE\n12 / 512', 'DOPE\n12 / 2048', 'DOPE\n24 / 512', 'DOPE\n24 / 2048',
          'CTGAN\ndefault', 'CTGAN\nnative', 'TVAE\ndefault', 'TVAE\nnative']


def render(document, kind):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if (document['official_tests_opened'] is not False or document['production_certified'] is not False
            or any(document[k] is not None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority'))
            or summarize(document['cells'], document['dataset_ids']) != document['summary']):
        raise ValueError('matched panel scope or reconciliation changed')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.hashsalt': 'dope-s3-matched-v1', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        for ax, dataset in zip(axes.flat, document['dataset_ids']):
            rows = [r for r in document['summary'] if r['dataset'] == dataset]
            for size, offset, color in ((1, -.18, '#236e96'), (4, .18, '#b65436')):
                for index, (method, config) in enumerate(CONFIGS):
                    row = next(r for r in rows if r['method'] == method and r['configuration'] == config
                               and r['size_multiplier'] == size)
                    utility = row['utility']['catboost']
                    value = utility['median_retention']
                    if value is None:
                        ax.annotate('null', (index + offset, 0), rotation=90, color=color, fontsize=7)
                    else:
                        ax.bar(index + offset, value, width=.32, color=color,
                               label=f'{size}n' if index == 0 else None)
                        ax.errorbar(index + offset, value,
                                    yerr=[[value - min(utility['sample_retention_values'])],
                                          [max(utility['sample_retention_values']) - value]],
                                    fmt='none', ecolor='#20252a', capsize=2, linewidth=.8)
            ax.axhline(1, color='#676d74', linestyle='--', linewidth=.8)
            ax.axhline(0, color='#676d74', linewidth=.5)
            ax.set_xticks(range(len(CONFIGS)), LABELS, fontsize=8)
            ax.set_ylabel('Null-normalized retention')
            name = next(d['display_name'] for d in document['dataset_metadata'] if d['id'] == dataset)
            ax.set_title(name + '\n' + dataset, loc='left', fontweight='bold')
            ax.grid(axis='y', alpha=.16)
            ax.set_axisbelow(True)
            ax.legend(loc='best', ncol=2, fontsize=8)
        fig.suptitle('S3 matched discovery validation: DOPE / CTGAN / TVAE',
                     fontsize=15, fontweight='bold', y=.99)
        fig.text(.5, .96, 'CatBoost auditor · fit seed 11 · three sample seeds · bars: median · whiskers: sample min/max',
                 ha='center', fontsize=10)
        fig.text(.5, .035, 'DOPE labels: features / training steps. All declared profiles shown; no production family selected.',
                 ha='center', fontsize=9)
        fig.text(.5, .022, 'Null bars retain failed/low-signal cells. Baselines: native efficacy; unconstrained utility (bytes and all auditors in JSON/CSV).',
                 ha='center', fontsize=9)
        fig.text(.5, .009, 'Official-training-derived validation · official tests sealed · MFS-v2 / PTF-v1 / release-safe null',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .945), h_pad=2.6)
        out = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE benchmark'} if kind == 'svg' else {
            'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE benchmark'}
        fig.savefig(out, format=kind, metadata=metadata)
        plt.close(fig)
    data = out.getvalue()
    return b'\n'.join(line.rstrip() for line in data.splitlines()) + b'\n' if kind == 'svg' else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    document = json.loads(args.report.read_text())
    for kind in ('svg', 'pdf'):
        args.report.with_suffix('.' + kind).write_bytes(render(document, kind))


if __name__ == '__main__':
    main()
