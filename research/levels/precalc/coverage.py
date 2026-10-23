"""Immutable, explicit whole-schedule coverage indexes; no raw data reads."""
import hashlib
import json
from pathlib import Path
from ..contracts import canonical, digest, require
from .worker import lookup, index_key
from .plan import validate_plan


def build(plan, roots, store):
    observed={}; repairs={}; attempts=[]; cells={c['key']:c for c in plan['cells']}
    for root in map(Path,roots):
        ownership=json.loads((root/'ownership.json').read_bytes())
        if ownership['plan_sha256']!=plan['sha256']:
            child=validate_plan(json.loads((root/'plan.json').read_bytes()))
            require(child['sha256']==ownership['plan_sha256']and child.get('parent_plan_sha256')==plan['sha256']
                    and child['catalog_sha256']==plan['catalog_sha256']and child['options']==plan['options']
                    and child['feature_recipe']==plan['feature_recipe']
                    and all(c['key']in cells and c==cells[c['key']]for c in child['cells']),
                    'precalc_coverage_unqualified_continuation')
        producer=json.loads((root/'producer.json').read_bytes())
        for path in sorted((root/'cells').glob('*/receipt.json')):
            receipt=json.loads(path.read_bytes());key=receipt['cell_key']
            require(key in cells,'precalc_coverage_unknown_cell')
            if receipt['state']=='repair_refused':
                attempts.append(dict(cell_key=key,receipt_sha256=digest(receipt),state='repair_refused'));continue
            if receipt['state']in ('complete','cached'):
                checked=lookup(cells[key],plan,receipt['producer'],store)
                require(checked is not None,'precalc_coverage_missing_publication')
                object_key=index_key(cells[key],plan,checked['key']);raw=store.get(object_key)
                record=dict(cell_key=key,state='ready',entry_key=object_key,
                            entry_sha256=hashlib.sha256(raw).hexdigest(),valid_features=checked['valid_features'],
                            producing_plan_sha256=json.loads(raw)['plan_sha256'])
            else:record=dict(cell_key=key,state=receipt['state'],receipt_sha256=digest(receipt),measurement_key=receipt['key'])
            if 'original_failure_receipt_sha256'in receipt:
                derivation=receipt['producer'].get('derivation',{})
                require(receipt['state']=='complete'and receipt['new_native_executions']==0
                        and receipt['new_optimizer_updates']==0
                        and derivation.get('original_receipt_sha256')==receipt['original_failure_receipt_sha256']
                        and derivation.get('adopted_report_sha256')==receipt['report_sha256']
                        and receipt['report_sha256']==json.loads(raw)['members']['report.json']
                        and key not in repairs,'precalc_coverage_repair_lineage')
                repairs[key]=(record,receipt);continue
            require(key not in observed or observed[key]==record,'precalc_coverage_duplicate_conflict');observed[key]=record
        for path in sorted((root/'failures').glob('*.json')):
            receipt=json.loads(path.read_bytes());key=receipt['cell_key']
            require(key in cells and key not in observed,'precalc_coverage_failure_conflict')
            observed[key]=dict(cell_key=key,state=receipt['state'],receipt_sha256=digest(receipt))
    for key,(record,receipt)in repairs.items():
        original=observed.get(key,{})
        require(original.get('state')in('failed','publication_failed')
                and original.get('receipt_sha256')==receipt['original_failure_receipt_sha256']
                and original.get('measurement_key')==receipt['original_failure_key'],
                'precalc_coverage_original_failure_required')
        record.update(original_failed_receipt_sha256=receipt['original_failure_receipt_sha256'],
                      original_failed_receipt_locator=receipt['original_failure_receipt_locator'],
                      zero_additional_native_executions=True)
        observed[key]=record
    records=[dict(scope=c['scope'],**observed.get(c['key'],dict(cell_key=c['key'],state='pending')))for c in plan['cells']]
    result=dict(format='dope-fit-cv-precalc-corpus-v1',plan_sha256=plan['sha256'],feature_recipe=plan['feature_recipe'],
        options=plan['options'],scheduled=len(records),records=records,
        counts={state:sum(r['state']==state for r in records)for state in sorted({r['state']for r in records})},
        training_admitted=False,official_test_access=False,repair_attempts=attempts)
    raw=canonical(result);checksum=hashlib.sha256(raw).hexdigest();key='corpora/'+checksum+'/index.json'
    store.put(key,raw)
    return dict(index_key=key,index_sha256=checksum,counts=result['counts'],scheduled=len(records),bytes=len(raw))
