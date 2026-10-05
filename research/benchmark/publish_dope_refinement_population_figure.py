"""Draw descriptive dataset-paired differences with explicit cohort sizes."""
import argparse
import io
from pathlib import Path

from . import publish_dope_refinement_population as pub


def render(report, kind):
    pub.validate_report(report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'dope-refinement-population-common-v1', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, profile in zip(axes, pub.fits.PROFILES):
            rows = [r for r in report['paired_descriptive']
                    if r['configuration'] == profile and r['size_multiplier'] == 4 and r['auditor'] == 'catboost']
            pub.guard.require(len(rows) == 4, 'paired figure reference coverage differs')
            labels = []
            for y, row in enumerate(rows):
                name = {'DOPE': 'DOPE 12/2048', 'GaussianCopula': 'GaussianCopula/native',
                        'Chow-Liu': 'Chow-Liu/native (study)',
                        'independent_marginals': 'Independent marginals/native (study)'}[row['reference_method']]
                labels.append(name + f" [N={row['paired_complete_informative_lineages']}]")
                if row['median_paired_difference'] is not None:
                    ax.scatter(row['median_paired_difference'], y, color='#236e96', s=50)
            ax.axvline(0, color='#999999', linewidth=1); ax.set_yticks(range(4), labels)
            ax.set_title(profile); ax.set_xlabel('Median paired retention difference\n(DOPE − reference)')
            ax.grid(axis='x', alpha=.16); ax.invert_yaxis()
        fig.suptitle('Complete S3 refinement validation: descriptive 4n CatBoost pairs', fontweight='bold')
        fig.text(.5, .06, '100 planned lineages; complete informative groups only. Each pair can have a different cohort.', ha='center', fontsize=9)
        fig.text(.5, .025, 'One fit seed; sample-seed medians before dataset pairs. No superiority inference; MFS/PTF/release null.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, .93)); out = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE benchmark'} if kind == 'svg' else {'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE benchmark'}
        fig.savefig(out, format=kind, metadata=metadata); plt.close(fig)
    body = out.getvalue()
    return b'\n'.join(line.rstrip() for line in body.splitlines()) + b'\n' if kind == 'svg' else body


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('report', type=Path); args = parser.parse_args()
    report = pub.guard.decode(args.report.read_bytes())
    for kind in ('svg', 'pdf'): args.report.with_suffix('.' + kind).write_bytes(render(report, kind))
    print(args.report)


if __name__ == '__main__': main()
