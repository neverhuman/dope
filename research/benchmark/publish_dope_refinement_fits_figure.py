"""Draw complete fit outcomes and costs from the rights-safe training ledger."""
import argparse
import io
import json
from pathlib import Path

from .publish_dope_refinement_fits import GATES, NAME, summarize
from .arf_runtime_guard import require


def render(report, kind):
    require(summarize(report['cells']) == report['summary']
        and all(report[k] is v for k, v in GATES.items()), 'figure matrix differs')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'dope-refinement-fits-v1', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); rows = report['summary']; xs = range(2)
        bottom = [0, 0]
        for status, label, color in (
            ('ok', 'OK including prior reuses', '#236e96'),
            ('transport_or_prelaunch_failure', 'Transport/prelaunch infrastructure', '#999999'),
            ('foreign_gpu_owner_appeared', 'Foreign owner rejection', '#e6b85c'),
            ('charged_artifact_cap', 'Charged byte-cap failure', '#b65436')):
            values = [r['status_counts'].get(status, 0) for r in rows]
            axes[0].bar(xs, values, bottom=bottom, color=color, label=label)
            bottom = [a + b for a, b in zip(bottom, values)]
        axes[0].set_ylim(0, 110); axes[0].set_ylabel('Closed dataset-specific fit cells')
        axes[0].set_title('Complete 100-lineage research matrix'); axes[0].legend(loc='lower center', fontsize=8)
        axes[1].bar(xs, [r['new_operation_seconds'] / 60 for r in rows], color='#236e96')
        axes[1].set_ylabel('New operation minutes'); axes[1].set_title('New fits and failed transport attempts')
        for ax in axes:
            ax.set_xticks(list(xs), ['12 /8192', '12 /width8 /8192']); ax.grid(axis='y', alpha=.16); ax.set_axisbelow(True)
        fig.suptitle('DOPE GPU refinement: complete 200-cell training ledger', fontweight='bold')
        fig.text(.5, .055, '176 new cells + 24 prior successes; fit seed 11. Model and projection bytes charged.', ha='center', fontsize=9)
        fig.text(.5, .02, 'Quality and privacy coverage pending; official tests sealed; MFS-v2/PTF-v1/release null.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .94)); out = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE benchmark'} if kind == 'svg' else {'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE benchmark'}
        fig.savefig(out, format=kind, metadata=metadata); plt.close(fig)
    data = out.getvalue()
    return b'\n'.join(line.rstrip() for line in data.splitlines()) + b'\n' if kind == 'svg' else data


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('report', type=Path); args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    for kind in ('svg', 'pdf'): args.report.with_suffix('.' + kind).write_bytes(render(report, kind))
    print(NAME)


if __name__ == '__main__': main()
