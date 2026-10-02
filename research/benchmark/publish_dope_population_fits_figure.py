"""Training byte-cap and cost figure; no utility or production score."""
from __future__ import annotations

import io
import json
from pathlib import Path

from .publish_dope_population_fits import PROFILES, require, summarize


def render(report, kind):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    require(report['official_tests_opened'] is False and report['production_certified'] is False
            and report['global_family_selected'] is False
            and all(report[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority'))
            and summarize(report['cells']) == report['summary'], 'fit figure scope changed')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'dope-population-fit-custody-v1', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        rows = report['summary']; xs = range(4)
        ok = [r['status_counts'].get('ok', 0) for r in rows]
        failed = [r['status_counts'].get('charged_artifact_cap', 0) for r in rows]
        axes[0].bar(xs, ok, color='#236e96', label='Charged bytes <=10,240')
        axes[0].bar(xs, failed, bottom=ok, color='#b65436', label='Byte-cap failure')
        for x, a, b in zip(xs, ok, failed): axes[0].text(x, a/2, str(a), color='white', ha='center')
        axes[0].set_ylim(0, 110); axes[0].set_ylabel('Closed dataset-specific fit cells')
        axes[0].legend(loc='lower center', fontsize=9)
        axes[0].set_title('All 100 S3 training-derived lineages')
        axes[1].bar(xs, [r['new_operation_seconds']/60 for r in rows], color='#236e96')
        axes[1].set_ylabel('New fit-operation minutes'); axes[1].set_title('Research cost (prior fit reuses charged separately)')
        for ax in axes:
            ax.set_xticks(list(xs), ['12 /512', '12 /2048', '24 /512', '24 /2048'])
            ax.set_xlabel('Feature budget /training steps'); ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
        fig.suptitle('DOPE GPU population training: complete 400-cell fit custody', fontweight='bold')
        fig.text(.5, .07, '354 new fits + 46 immutable prior successes; fit seed 11. Charges include generator and projection.', ha='center', fontsize=9)
        fig.text(.5, .035, 'Validation and privacy coverage incomplete; official tests sealed; MFS-v2/PTF-v1/release-safe null.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .94))
        out = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE benchmark'} if kind == 'svg' else {'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE benchmark'}
        fig.savefig(out, format=kind, metadata=metadata); plt.close(fig)
    data = out.getvalue()
    return b'\n'.join(line.rstrip() for line in data.splitlines())+b'\n' if kind == 'svg' else data


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1]); report = json.loads(path.read_text())
    for kind in ('svg', 'pdf'): path.with_suffix('.'+kind).write_bytes(render(report, kind))
