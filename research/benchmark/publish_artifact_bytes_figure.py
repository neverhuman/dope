"""Plot artifact bytes without implying generator quality or certification."""
from __future__ import annotations

import io
import json

from .publish_artifact_bytes import NAME, RESULTS, require


def render(report, kind):
    require(kind in ('svg', 'pdf') and report['candidate_count'] == 32
            and report['generator_validation_complete'] is False and report['production_certified'] is False
            and report['mfs_v2'] is None and report['ptf_v1'] is None and report['release_safe_l3'] is None,
            'byte figure acquired a gated claim')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped = {}
    for row in report['cells']:
        grouped.setdefault((row['dataset'], row['profile']), {})[row['compression_level']] = row
    require(len(grouped) == 16 and all(set(rows) == {1, 9} for rows in grouped.values()),
            'byte figure requires the complete probe grid')
    keys = sorted(grouped)
    with plt.rc_context({'svg.hashsalt': NAME, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'font.family': 'DejaVu Sans', 'font.size': 9}):
        fig, ax = plt.subplots(figsize=(10.4, 8.4))
        for offset, color, label, field, level in [
            (-0.23, '#b4bcc7', 'Original model + projection', 'original_charged_bytes', 1),
            (0.0, '#2f70b6', 'Lossless container, level 1', 'container_bytes', 1),
            (0.23, '#20a090', 'Lossless container, level 9', 'container_bytes', 9)]:
            ax.barh([i + offset for i in range(16)], [grouped[k][level][field] for k in keys],
                    height=0.21, color=color, label=label)
        ax.axvline(10240, color='#a92d30', linestyle='--', linewidth=1.2, label='10,240-byte cap')
        ax.set_yticks(range(16), [k[0][:8] + ' / ' + k[1].replace('features', 'f').replace('_steps', '/s') for k in keys])
        ax.invert_yaxis()
        ax.set_xlabel('All serialized artifact bytes (including container header and learned projection)')
        ax.set_title('Lossless byte feasibility for 16 previously over-cap DOPE fits')
        ax.set_axisbelow(True); ax.grid(axis='x', alpha=0.2)
        ax.legend(loc='upper right', fontsize=8)
        fig.text(0.02, 0.02, 'Byte inspection only. Wrapped generators unvalidated; MFS-v2/PTF-v1/release-safe null.', fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        output = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE research byte analysis'} if kind == 'svg' else {
            'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE research byte analysis'}
        fig.savefig(output, format=kind, metadata=metadata)
        plt.close(fig)
        data = output.getvalue()
        return b'\n'.join(line.rstrip() for line in data.splitlines()) + b'\n' if kind == 'svg' else data


if __name__ == '__main__':
    report = json.loads((RESULTS / (NAME + '.json')).read_text())
    for kind in ('svg', 'pdf'):
        (RESULTS / (NAME + '.' + kind)).write_bytes(render(report, kind))
