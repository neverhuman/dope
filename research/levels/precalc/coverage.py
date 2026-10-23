"""Immutable, explicit whole-schedule coverage indexes; no raw data reads."""
import hashlib
import json
from pathlib import Path
from ..contracts import canonical, digest, require
from .worker import lookup, index_key


def build(plan, roots, store):
    observed={}; cells={c['key']:c for c in plan['cells']}
    for root in map(Path,roots):
        ownership=json.loads((root/'ownership.json').read_bytes())
        require(ownership['plan_sha256']==plan['sha256'],'precalc_coverage_run_plan')
        producer=json.loads((root/'producer.json').read_bytes())
        for path in sorted((root/'cells').glob('*/receipt.json')):
            receipt=json.loads(path.read_bytes());key=receipt['cell_key']
            require(key in cells,'precalc_coverage_unknown_cell')
            if receipt['state']in ('complete','cached'):
                checked=lookup(cells[key],plan,producer,store)
                require(checked is not None,'precalc_coverage_missing_publication')
                object_key=index_key(cells[key],plan,checked['key']);raw=store.get(object_key)
                record=dict(cell_key=key,state='ready',entry_key=object_key,
                            entry_sha256=hashlib.sha256(raw).hexdigest(),valid_features=checked['valid_features'])
            else:record=dict(cell_key=key,state=receipt['state'],receipt_sha256=digest(receipt))
            require(key not in observed or observed[key]==record,'precalc_coverage_duplicate_conflict');observed[key]=record
        for path in sorted((root/'failures').glob('*.json')):
            receipt=json.loads(path.read_bytes());key=receipt['cell_key']
            require(key in cells and key not in observed,'precalc_coverage_failure_conflict')
            observed[key]=dict(cell_key=key,state=receipt['state'],receipt_sha256=digest(receipt))
    records=[dict(scope=c['scope'],**observed.get(c['key'],dict(cell_key=c['key'],state='pending')))for c in plan['cells']]
    result=dict(format='dope-fit-cv-precalc-corpus-v1',plan_sha256=plan['sha256'],feature_recipe=plan['feature_recipe'],
        options=plan['options'],scheduled=len(records),records=records,
        counts={state:sum(r['state']==state for r in records)for state in sorted({r['state']for r in records})},
        training_admitted=False,official_test_access=False)
    raw=canonical(result);checksum=hashlib.sha256(raw).hexdigest();key='corpora/'+checksum+'/index.json'
    store.put(key,raw)
    return dict(index_key=key,index_sha256=checksum,counts=result['counts'],scheduled=len(records),bytes=len(raw))
