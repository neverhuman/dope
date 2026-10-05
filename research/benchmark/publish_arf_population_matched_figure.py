"""Draw descriptive pairs against frozen ARF default and native selections."""
import argparse
import io
from pathlib import Path

from . import publish_arf_population_matched as pub


def render(report, kind):
    pub.validate_report(report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'svg.hashsalt':'dope-arf-population-matched-v1','pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,profile in zip(axes,pub.PROFILES):
            rows=[r for r in report['paired_descriptive'] if r['configuration']==profile
                  and r['size_multiplier']==4 and r['auditor']=='catboost']
            pub.guard.require(len(rows)==2,'ARF reference figure coverage differs')
            labels=[]
            for y,row in enumerate(rows):
                labels.append('ARF/'+row['reference_configuration']+f" [N={row['paired_complete_informative_lineages']}]")
                if row['median_paired_difference'] is not None:
                    ax.scatter(row['median_paired_difference'],y,color='#236e96',s=48)
            ax.axvline(0,color='#999999',linewidth=1);ax.set_yticks(range(2),labels)
            ax.set_title(profile);ax.set_xlabel('Median paired retention difference\n(DOPE − ARF)')
            ax.grid(axis='x',alpha=.16);ax.invert_yaxis()
        fig.suptitle('Matched S3 validation: original ARF vs DOPE, descriptive 4n CatBoost pairs',fontweight='bold')
        fig.text(.5,.06,'100 planned training-derived views; sample-seed medians before dataset pairs. Each pair may have a different cohort.',ha='center',fontsize=9)
        fig.text(.5,.025,'Native FORDE selection unchanged. One fit seed; unconstrained ARF bytes visible. MFS/PTF/release/superiority null.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.11,1,.93));out=io.BytesIO()
        metadata={'Date':None,'Creator':'DOPE benchmark'} if kind=='svg' else {'CreationDate':None,'ModDate':None,'Creator':'DOPE benchmark'}
        fig.savefig(out,format=kind,metadata=metadata);plt.close(fig)
    body=out.getvalue()
    return b'\n'.join(x.rstrip() for x in body.splitlines())+b'\n' if kind=='svg' else body


def main():
    parser=argparse.ArgumentParser();parser.add_argument('report',type=Path);args=parser.parse_args()
    report=pub.guard.decode(args.report.read_bytes())
    for kind in ('svg','pdf'):args.report.with_suffix('.'+kind).write_bytes(render(report,kind))


if __name__=='__main__':main()
