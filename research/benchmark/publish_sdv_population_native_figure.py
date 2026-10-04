"""Within-method native efficacy and coverage; never cross-method KPI ranks."""
import io

from . import publish_sdv_population_native as m


def render(report, kind):
    m.accounting.require(all(report[k] is v for k, v in m.GATES.items()),
                         'native figure acquired a gated claim')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
        'svg.fonttype': 'none', 'svg.hashsalt': 'sdv-population-native-v1', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, method in zip(axes, ('CTGAN', 'TVAE')):
            paired = [c for c in report['cells'] if c['method'] == method
                      and c['default_native_kpi'] is not None and c['selected_native_kpi'] is not None]
            for key, label, color in (('default_native_kpi', 'Author default', '#b65436'),
                                      ('selected_native_kpi', 'Native selected', '#236e96')):
                values = sorted(c[key]['value'] for c in paired)
                ax.step(values, [(i + 1) / 100 for i in range(len(values))], where='post',
                        color=color, label=f'{label} ({len(values)}/100 paired)')
            ax.set_title(method, loc='left', fontweight='bold')
            ax.set_xscale('symlog', linthresh=.5)
            ax.set_ylim(0, 1.04)
            ax.axvline(0, color='#676d74', linewidth=.6)
            ax.set_xlabel('Mean LR / MLP validation R² (symmetric log scale)')
            ax.set_ylabel('Fraction of all100 lineages with native KPI ≤ x')
            ax.grid(alpha=.15)
            ax.legend(fontsize=9)
        fig.suptitle('CTGAN / TVAE native selection: complete bounded research ledger', fontweight='bold')
        fig.text(.5, .065, 'Same paired cells within each method · one fit seed · negative R² retained · no cross-method KPI ranking', ha='center', fontsize=9)
        fig.text(.5, .025, 'Curve endpoints retain coverage gaps. Shared quality pending; official tests sealed; MFS-v2 / PTF-v1 null.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, .95))
        out = io.BytesIO()
        metadata = {'Date': None, 'Creator': 'DOPE benchmark'} if kind == 'svg' else {
            'CreationDate': None, 'ModDate': None, 'Creator': 'DOPE benchmark'}
        fig.savefig(out, format=kind, metadata=metadata)
        plt.close(fig)
    value = out.getvalue()
    return b'\n'.join(s.rstrip() for s in value.splitlines()) + b'\n' if kind == 'svg' else value
