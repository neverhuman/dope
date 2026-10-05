"""Deterministic figures from a complete, externally pinned research report."""
import argparse
import hashlib
import json
from pathlib import Path
import math
from .density_matched_aggregate import aggregate, CONFIGS, AUDITORS
from .publish_density_matched import load_report

LABELS = ['DOPE 12 / 512', 'DOPE 12 / 2048', 'DOPE 24 / 512', 'DOPE 24 / 2048',
          'Copula default', 'Copula native', 'Marginals default', 'Marginals native',
          'Chow-Liu default', 'Chow-Liu native']


def render(report_path, expected_sha256, output):
    report = load_report(report_path, expected_sha256)
    if (report['datasets'] != 100 or report['frozen_logical_cells'] != 6000
            or report['official_tests_opened'] is not False
            or report['production_certified'] is not False
            or report['global_family_selected'] is not False
            or any(report[k] is not None for k in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'paired_superiority'))):
        raise ValueError('complete descriptive validation report required')
    checked = aggregate(report['cells'])
    if any(report[k] != checked[k] for k in checked):
        raise ValueError('matched figure summaries do not replay')
    # Plotting starts after the report hash, coverage, flags and aggregate replay.
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({'svg.hashsalt': 'dope-density-matched-validation-v1',
                         'font.family': 'DejaVu Sans', 'font.size': 9})
    import matplotlib.pyplot as plt
    out = Path(output)
    out.mkdir(mode=0o700, exist_ok=False)
    artifacts, plotted = {}, []
    for size in (1, 4):
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for ax, auditor in zip(axes, AUDITORS):
            for index, config in enumerate(CONFIGS, 1):
                rows = [g for g in report['summary']
                        if (g['method'], g['configuration']) == config and g['size_multiplier'] == size]
                if len(rows) != 100: raise ValueError('figure lineage coverage differs')
                values = [g['utility'][auditor]['median_retention'] for g in rows
                          if g['utility'][auditor]['complete_informative_sample_group']]
                if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
                    raise ValueError('invalid figure retention')
                if values:
                    boxes = ax.boxplot([values], positions=[index], widths=0.55,
                                       patch_artist=True, manage_ticks=False,
                                       flierprops={'markersize': 2, 'alpha': 0.4})
                    color = '#7656b5' if config[0] == 'DOPE' else ('#387fa6' if config[1] == 'native_selected' else '#b3bcc4')
                    boxes['boxes'][0].set_facecolor(color)
                ax.text(index, 0.97, str(len(values)), ha='center', va='top',
                        transform=ax.get_xaxis_transform(), fontsize=8,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
                plotted.append(dict(method=config[0], configuration=config[1],
                                    size_multiplier=size, auditor=auditor,
                                    informative_lineages=len(values)))
            ax.axhline(1, color='#2a333b', linestyle='--', linewidth=0.8)
            ax.axhline(0, color='#8c969f', linewidth=0.6)
            ax.set_yscale('symlog', linthresh=0.5)
            lower, upper = ax.get_ylim()
            ax.set_yticks([v for v in (-10000, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, 10000)
                          if lower <= v <= upper])
            ax.set_ylabel(auditor + '\nretention (unclipped)')
            ax.grid(axis='y', alpha=0.2)
            ax.set_xlim(0.4, 10.6)
        axes[-1].set_xticks(range(1, 11), LABELS, rotation=30, ha='right')
        fig.suptitle(f'Matched S3 regression validation — {size}n synthetic rows', fontsize=13)
        fig.text(0.5, 0.035, '100 frozen lineages; one fit seed; three sample seeds aggregated within each lineage. Counts show informative lineages.\n'
                 'Box: median and interquartile range; whiskers: 1.5 IQR. Symmetric log scale. Dashed line: real-data utility retention = 1.\n'
                 'DOPE labels: features / training steps. Baselines selected only by their own likelihood objectives. All gated scores remain null.',
                 ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, 0.095, 1, 0.97))
        for extension in ('svg', 'pdf'):
            path = out / f'density-matched-retention-{size}n.{extension}'
            metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None}
            fig.savefig(path, format=extension, metadata=metadata)
            blob = path.read_bytes()
            if extension == 'svg':
                # Matplotlib emits trailing spaces in multiline path attributes.
                # Canonical whitespace keeps generated Git text reproducible.
                blob = b'\n'.join(line.rstrip() for line in blob.splitlines()) + b'\n'
                path.write_bytes(blob)
            artifacts[path.name] = {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
        plt.close(fig)
    manifest = dict(report_sha256=expected_sha256, datasets=100, logical_cells=6000,
        matplotlib_version=mpl.__version__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts=artifacts, plotted_counts=plotted, official_tests_opened=False,
        mfs_v2=None, ptf_v1=None, release_safe=None, superiority=None, production_certified=False)
    with (out / 'figures-manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = render(args.report, args.sha256, args.output)
    print(json.dumps({'figures': len(result['artifacts']), 'datasets': result['datasets']}))
