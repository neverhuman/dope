"""Reproducible California single-fit common-outcome comparison figure."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path


CONFIGS = (
    ('ForestDiffusion', 'native_selected', 'Forest Flow\nGPU default/native'),
    ('CTGAN', 'default_and_tuned', 'CTGAN\ndefault/native'),
    ('TVAE', 'default', 'TVAE\ndefault'),
    ('TVAE', 'tuned', 'TVAE\nnative'),
    ('DOPE', 'q8_selected', 'DOPE\nq8'),
    ('DOPE', 'features12_steps512', 'DOPE\n12 features/512'),
    ('DOPE', 'features12_steps2048', 'DOPE\n12 features/2048'),
    ('DOPE', 'features24_steps512', 'DOPE\n24 features/512'),
    ('DOPE', 'features24_steps2048', 'DOPE\n24 features/2048'),
)


def render(document, kind):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if (document['official_tests_opened'] is not False or document['production_certified'] is not False
            or any(document[k] is not None for k in ('mfs_v2', 'ptf_v1', 'release_safe'))):
        raise ValueError('pilot seal or score changed')
    rows = [{**r, 'method': 'ForestDiffusion'} for r in document['summaries']]
    rows += document['matched_references']['summaries']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.hashsalt': 'dope-forest-native-v1', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 1, figsize=(12.6, 10.8))
        for ax, auditor in zip(axes, ('catboost', 'linear', 'mlp')):
            for size, offset, color in ((1, -.18, '#236e96'), (4, .18, '#b65436')):
                for index, (method, config, _) in enumerate(CONFIGS):
                    row = next(r for r in rows if r['dataset'] == 'California' and r['method'] == method
                               and r['configuration'] == config and r['size_multiplier'] == size and r['auditor'] == auditor)
                    value = row['median_retention']
                    if value is None:
                        ax.annotate('null', (index + offset, .02), rotation=90, color=color, fontsize=7)
                    else:
                        ax.bar(index + offset, value, width=.32, color=color,
                               label=f'{size}n' if index == 0 else None)
                        ax.errorbar(index + offset, value,
                                    yerr=[[value - row['min_retention']], [row['max_retention'] - value]],
                                    fmt='none', ecolor='#20252a', capsize=2, linewidth=.8)
            ax.axhline(1, color='#676d74', linestyle='--', linewidth=.8)
            ax.set_xticks(range(len(CONFIGS)), [c[2] for c in CONFIGS], fontsize=8)
            ax.set_ylabel('Null-normalized retention')
            ax.set_title(auditor.upper() if auditor == 'mlp' else auditor.capitalize(), loc='left', fontweight='bold')
            ax.grid(axis='y', alpha=.16)
            ax.set_axisbelow(True)
            ax.legend(loc='lower left', ncol=2, fontsize=8)
        fig.suptitle('California: ForestDiffusion and matched DOPE / CTGAN / TVAE validation',
                     fontsize=14, fontweight='bold', y=.99)
        fig.text(.5, .957, 'Fit seed 23 · three sample seeds · bars: sample median · whiskers: sample min/max',
                 ha='center', fontsize=10)
        fig.text(.5, .035, 'Unconstrained common quality: baselines exceed L3; DOPE artifacts fit L3. No release certification or superiority claim.',
                 ha='center', fontsize=8)
        fig.text(.5, .023, 'All DOPE profiles retained; no global candidate family selected. Forest default/native share one fit and samples.',
                 ha='center', fontsize=8)
        fig.text(.5, .010, 'Official-training-derived validation · official tests sealed · MFS-v2 / PTF-v1 / release-safe null · L3 cap 10,240 bytes',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .058, 1, .935), h_pad=2.1)
        out = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE benchmark'} if kind == 'svg' else {
            'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE benchmark'}
        fig.savefig(out, format=kind, metadata=metadata)
        plt.close(fig)
    data = out.getvalue()
    if kind == 'svg':
        data = b'\n'.join(line.rstrip() for line in data.splitlines()) + b'\n'
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    document = json.loads(args.report.read_text())
    for kind in ('svg', 'pdf'):
        args.report.with_suffix('.' + kind).write_bytes(render(document, kind))


if __name__ == '__main__':
    main()
