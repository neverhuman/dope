"""All profiles and auditors; coverage denominator stays100, no score clipping."""
import io
import json
from pathlib import Path

from . import publish_dope_population_validation as m

def render(report,kind):
    m.require(report['official_tests_opened'] is False and report['production_certified'] is False
              and report['global_family_selected'] is False and report['counts_as_dope_win'] is False
              and all(report[k] is None for k in ('mfs_v2','ptf_v1','release_safe_l3','paired_superiority')),
              'figure acquired a gated claim')
    summary,panels=m.aggregate(report['cells'])
    m.require(summary==report['summary'] and panels==report['profile_panels'],'figure summary differs')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'svg.hashsalt':'dope-population-validation-v1','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(3,2,figsize=(13,11),sharey=True)
        palette=('#236e96','#b65436','#568447','#845c96')
        labels=('12 /512','12 /2048','24 /512','24 /2048')
        for row,auditor in enumerate(m.AUDITORS):
            for col,size in enumerate(m.SIZES):
                ax=axes[row,col]
                for profile,color,label in zip(m.PROFILES,palette,labels):
                    values=sorted(g['utility'][auditor]['median_retention'] for g in summary
                                  if g['profile']==profile and g['size_multiplier']==size
                                  and g['utility'][auditor]['complete_informative_sample_group'])
                    if values:ax.step(values,[(i+1)/100 for i in range(len(values))],where='post',
                                      color=color,label=f'{label} ({len(values)}/100)')
                    else:ax.plot([],[],color=color,label=f'{label} (0/100)')
                ax.set_xscale('symlog',linthresh=.5);ax.set_ylim(0,1.04)
                ax.axvline(1,color='#676d74',linestyle='--',linewidth=.8)
                ax.axvline(0,color='#676d74',linewidth=.5)
                ax.set_title(f'{auditor.capitalize()} auditor · {size}n',loc='left',fontweight='bold')
                ax.set_xlabel('Lineage median null-normalized retention (symmetric log scale)')
                ax.set_ylabel('Fraction of all100 lineages with median ≤ x')
                ax.grid(alpha=.16);ax.legend(loc='best',fontsize=8,ncol=2)
        fig.suptitle('DOPE all100 GPU research validation: all four frozen profiles',fontweight='bold',fontsize=14,y=.99)
        fig.text(.5,.962,'One fit seed · three sample seeds · training-derived validation · no global family selected',ha='center',fontsize=10)
        fig.text(.5,.038,'Legend: feature budget /training steps (complete informative lineages /100). Curve endpoints retain coverage gaps.',ha='center',fontsize=9)
        fig.text(.5,.021,'All finite positive and negative values are shown. Dashed line: retention1. No fit uncertainty or superiority claim.',ha='center',fontsize=9)
        fig.text(.5,.005,'Official tests sealed · MFS-v2 /PTF-v1 /release-safe null · byte-cap failures remain explicit in tables',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.945),h_pad=2)
        out=io.BytesIO();metadata={'Date':None,'Creator':'DOPE benchmark'} if kind=='svg' else {'CreationDate':None,'ModDate':None,'Creator':'DOPE benchmark'}
        fig.savefig(out,format=kind,metadata=metadata);plt.close(fig)
    value=out.getvalue()
    return b'\n'.join(s.rstrip() for s in value.splitlines())+b'\n' if kind=='svg' else value


if __name__ == '__main__':
    import sys
    path = Path(sys.argv[1]); report = json.loads(path.read_text())
    for kind in ('svg', 'pdf'): path.with_suffix('.'+kind).write_bytes(render(report,kind))
