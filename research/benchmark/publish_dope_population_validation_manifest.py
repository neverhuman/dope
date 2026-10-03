"""Bind the complete validation publication to externally supplied frozen custody."""
from __future__ import annotations

import json
from pathlib import Path

from .publish_dope_population_validation import NAME,RESULTS,aggregate,build as build_report,require,schema,tables
from .publish_dope_population_validation_figure import render
from .score import sha256


def build(receipt_sha,reconciliation_sha):
    report=json.loads((RESULTS/(NAME+'.json')).read_text())
    # Both anchors are supplied independently of mutable publication JSON.
    require(build_report(receipt_sha,reconciliation_sha)==report,'publication differs from frozen custody')
    summary,panels=aggregate(report['cells'])
    require(summary==report['summary'] and panels==report['profile_panels'],'publication summary differs')
    require(json.loads((RESULTS/(NAME+'.schema.json')).read_text())==schema(report),'publication schema differs')
    table,markdown=tables(report)
    require((RESULTS/(NAME+'.csv')).read_text()==table and (RESULTS/(NAME+'.md')).read_text()==markdown,'publication tables differ')
    for kind in ('svg','pdf'):require((RESULTS/(NAME+'.'+kind)).read_bytes()==render(report,kind),'publication figure differs')
    here=Path(__file__).parent
    sources={n:sha256(here/n) for n in ('publish_dope_population_validation.py','publish_dope_population_validation_figure.py',
        'publish_dope_population_validation_manifest.py','publish_dope_population_fits.py','publish_s3_forest.py',
        'publish_s3_matched.py','manifest.py','score.py')}
    artifacts={NAME+s:{'sha256':sha256(RESULTS/(NAME+s)),'bytes':(RESULTS/(NAME+s)).stat().st_size}
               for s in ('.json','.schema.json','.csv','.md','.svg','.pdf')}
    return {'format':'dope-s3-population-validation-publication-manifest','version':1,'scope':report['scope'],
            'artifacts':artifacts,'publisher_sources':sources,'source_locks':report['source_locks'],
            'immutable_references':report['immutable_references'],'datasets':100,'logical_sample_cells':2400,
            'physical_sample_batches':232,'logical_status_counts':report['logical_status_counts'],
            'all_frozen_cells_accounted':True,'campaign_complete':False,'global_family_selected':False,
            'official_tests_opened':False,'production_certified':False,'mfs_v2':None,'ptf_v1':None,
            'release_safe_l3':None,'paired_superiority':None,'counts_as_dope_win':False}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--reconciliation-sha256',required=True)
    args=parser.parse_args();manifest=build(args.receipt_sha256,args.reconciliation_sha256)
    path=RESULTS/(NAME+'.manifest.json')
    path.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n')
    path.with_suffix('.schema.json').write_text(json.dumps(schema(manifest),sort_keys=True,indent=2)+'\n')
    print(path)
