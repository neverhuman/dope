"""Content-addressed observed-score views; no source reads or model fitting."""
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
from ..contracts import canonical,digest,require,ResearchError
from ..landmark_features import FEATURE_RECIPE,OBSERVED_RECIPE,observed_view


def derive_corpus(store, source_key, source_sha256, *, workers=4, progress=None):
    require(type(workers)is int and 1<=workers<=16, 'precalc_view_worker_envelope')
    raw=store.get(source_key)
    require(raw is not None and hashlib.sha256(raw).hexdigest()==source_sha256, 'precalc_view_source_index_digest')
    source=json.loads(raw)
    require(source['format']=='dope-fit-cv-precalc-corpus-v1' and source['feature_recipe']==FEATURE_RECIPE
            and source['training_admitted']is False and source['official_test_access']is False
            and source['scheduled']==len(source['records'])
            and len({r['cell_key']for r in source['records']})==source['scheduled'], 'precalc_view_source_scope')
    def one(record):
        result=dict(scope=record['scope'],cell_key=record['cell_key'],state=record['state'],
                    source_record_sha256=digest(record),source_disposition=record['state'])
        if record['state']!='ready':return result
        entry_raw=store.get(record['entry_key'])
        require(entry_raw is not None and hashlib.sha256(entry_raw).hexdigest()==record['entry_sha256'],
                'precalc_view_source_entry_digest')
        entry=json.loads(entry_raw)
        require(entry['cell']['key']==record['cell_key'] and entry['cell']['scope']==record['scope']
                and entry['feature_recipe']==FEATURE_RECIPE and entry['training_admitted']is False
                and entry['official_test_access']is False
                and set(entry['members'])=={'manifest.json','report.json','features.json'}, 'precalc_view_source_entry_binding')
        bodies={}
        for name,checksum in entry['members'].items():
            body=store.get('objects/sha256/'+checksum+'.json')
            require(body is not None and hashlib.sha256(body).hexdigest()==checksum, 'precalc_view_source_member_digest')
            bodies[name]=body
        manifest=json.loads(bodies['manifest.json']);scope=manifest['scope']
        require(manifest['producer']==entry['producer'] and manifest['options']==source['options']
                and scope['physical_table_id']==record['scope']['physical_table_id']
                and scope['physical_rows_sha256']==record['scope']['training_rows_sha256'],
                'precalc_view_source_physical_binding')
        from .worker import measured_features
        values,mask=measured_features(json.loads(bodies['report.json']),scope['rows'],scope['task'],source['options'])
        require(json.loads(bodies['features.json'])==dict(feature_recipe=FEATURE_RECIPE,values=values,mask=mask),
                'precalc_view_source_features_binding')
        result.update(source_entry_key=record['entry_key'],source_entry_sha256=record['entry_sha256'],
                      source_report_sha256=entry['members']['report.json'])
        try:
            manifest,features=observed_view(bodies['manifest.json'],bodies['report.json'],
                entry['members']['manifest.json'],entry['producer']['engine_sha256'])
        except ResearchError as error:
            # A fully measured report can have no defined gains. Keep this explicit;
            # every other scope, digest, numeric or fold violation remains fatal.
            if str(error)!='cv_landmark_input_shape_or_empty':raise
            return dict(result,state='unavailable',reason='no_finite_observed_gains',valid_features=0)
        members={}
        for name,body in [('manifest.json',canonical(manifest)),('report.json',bodies['report.json']),
                          ('features.json',canonical(features))]:
            checksum=hashlib.sha256(body).hexdigest()
            if name!='report.json':store.put('objects/sha256/'+checksum+'.json',body)
            members[name]=checksum
        view=dict(format='dope-fit-cv-precalc-feature-view-v2',feature_recipe=OBSERVED_RECIPE,
                  cell=entry['cell'],members=members,source_entry_key=record['entry_key'],
                  source_entry_sha256=record['entry_sha256'],source_corpus_sha256=source_sha256,
                  training_admitted=False,official_test_access=False,new_native_executions=0,new_optimizer_updates=0)
        body=canonical(view);checksum=hashlib.sha256(body).hexdigest()
        key='feature-views/'+OBSERVED_RECIPE+'/entries/'+checksum+'.json';store.put(key,body)
        return dict(result,state='ready',entry_key=key,entry_sha256=checksum,valid_features=sum(features['mask']))
    records=[]
    with ThreadPoolExecutor(workers)as pool:
        for record in pool.map(one,source['records']):
            records.append(record)
            if progress is not None:progress(len(records),len(source['records']))
    result=dict(format='dope-fit-cv-precalc-corpus-view-v2',feature_recipe=OBSERVED_RECIPE,
                source_corpus_key=source_key,source_corpus_sha256=source_sha256,
                plan_sha256=source['plan_sha256'],options=source['options'],scheduled=source['scheduled'],
                records=records,counts={s:sum(r['state']==s for r in records)for s in sorted({r['state']for r in records})},
                training_admitted=False,official_test_access=False,new_native_executions=0,new_optimizer_updates=0)
    body=canonical(result);checksum=hashlib.sha256(body).hexdigest()
    key='feature-views/'+OBSERVED_RECIPE+'/corpora/'+checksum+'/index.json';store.put(key,body)
    return dict(index_key=key,index_sha256=checksum,scheduled=result['scheduled'],counts=result['counts'],bytes=len(body))


def materialize_view(store,key,expected_sha,output):
    """Download only a selected observed view; admission and row checks stay with training."""
    import os
    from pathlib import Path
    raw=store.get(key)
    require(raw is not None and hashlib.sha256(raw).hexdigest()==expected_sha,'precalc_selected_view_digest')
    view=json.loads(raw)
    require(view['format']=='dope-fit-cv-precalc-feature-view-v2'and view['feature_recipe']==OBSERVED_RECIPE
            and view['training_admitted']is False and view['official_test_access']is False
            and view['new_native_executions']==0 and view['new_optimizer_updates']==0
            and set(view['members'])=={'manifest.json','report.json','features.json'},'precalc_selected_view_binding')
    bodies={}
    for name,checksum in view['members'].items():
        body=store.get('objects/sha256/'+checksum+'.json')
        require(body is not None and hashlib.sha256(body).hexdigest()==checksum,'precalc_selected_view_member_digest');bodies[name]=body
    manifest=json.loads(bodies['manifest.json'])
    require(manifest['feature_recipe']==OBSERVED_RECIPE and manifest['format']=='dope-fit-cv-landmark-input-v2'
            and manifest['report_sha256']==view['members']['report.json'],'precalc_selected_view_manifest_binding')
    out=Path(output);out.mkdir(mode=0o700)
    for name,body in bodies.items():
        with(out/name).open('xb')as stream:os.chmod(out/name,0o600);stream.write(body)
    return dict(physical_table_id=manifest['scope']['physical_table_id'],directory=str(out.resolve()),
                manifest_sha256=view['members']['manifest.json'],engine_sha256=manifest['engine_sha256'])
