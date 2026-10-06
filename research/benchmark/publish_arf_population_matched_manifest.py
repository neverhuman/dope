"""Bind the complete rights-safe panel to immutable receipts and its renderer."""
import argparse
import json
from pathlib import Path

from . import publish_arf_population_matched as pub
from .publish_s3_matched import schema


def build(directory, report):
    pub.validate_report(report)
    files=[pub.NAME+'.'+s for s in ('json','schema.json','csv','md','svg','pdf')]
    sources=[Path(pub.__file__),Path(__file__),Path(__file__).with_name('publish_arf_population_matched_figure.py'),
             Path(pub.native.__file__),Path(pub.native.selection.__file__),Path(pub.guard.__file__),
             Path(pub.dope.__file__),Path(pub.dope.fits.__file__),
             Path(pub.dope.__file__).with_name('publish_dope_refinement_discovery.py')]
    return dict(format='dope-original-arf-matched-population-publication-manifest',version=1,
        source_locks=report['source_locks'],native_reference=report['native_selection_reference'],
        dope_reference=report['dope_reference'],cost=report['cost'],
        publisher_sources={p.name:pub.dope.fits.sha256(p) for p in sources},
        artifacts={n:dict(sha256=pub.dope.fits.sha256(directory/n),bytes=(directory/n).stat().st_size) for n in files},
        release_safe_l3_comparison_complete=False,**pub.GATES)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    report=pub.guard.decode((args.directory/(pub.NAME+'.json')).read_bytes())
    result=build(args.directory,report)
    for suffix,value in [('manifest.json',result),('manifest.schema.json',schema(result))]:
        (args.directory/(pub.NAME+'.'+suffix)).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
