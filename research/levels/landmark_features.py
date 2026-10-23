"""Private, training-only cross-validated encoder inputs; never public code sidecars."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from .contracts import canonical, digest, require

FEATURE_RECIPE = 'fit-cv-gain23-mask-squash-v1'
OPTIONS = dict(panel='landmarks-native-v1', folds=5, ridge_rows=8192,
               small_rows=4096, knn_reference=2048, spectrum_features=64,
               work_budget=2000000000, seed=0)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def validate_values(features):
    require(isinstance(features,(tuple,list)) and len(features)==2, 'cv_landmark_inputs_required')
    values,mask=features
    require(len(values)==23 and len(mask)==23 and all(type(v)is bool for v in mask)
            and any(mask), 'cv_landmark_input_shape_or_empty')
    require(all(type(v)in(int,float) and math.isfinite(v) and abs(v)<=3.4028234663852886e38
                for v in values), 'cv_landmark_input_nonfinite')
    require(all(present or value==0 for value,present in zip(values,mask)), 'cv_landmark_missing_value_nonzero')
    return values,mask


def feature_tensor(features, device):
    import torch
    values,mask=validate_values(features)
    values=torch.as_tensor(values,dtype=torch.float32,device=device)
    present=torch.tensor(mask,dtype=torch.bool,device=device)
    # Fixed per-coordinate map: no dataset/global/development fitted transform.
    values=values/(1+values.abs())
    return torch.cat((values,present.to(torch.float32)))


def training_csv(data):
    """Materialize only fit rows; holdout values never influence this byte stream."""
    require(20<=len(data['rows'])<=200000 and len(data['row_ids'])==len(data['rows'])
            and 2<=len(data['kinds'])<=257 and len(data['rows'])*len(data['kinds'])<=2000000
            and all(len(row)==len(data['kinds']) for row in data['rows'])
            and all(k in ('numeric','categorical') for k in data['kinds']), 'cv_landmark_source_envelope')
    indices=sorted(data['roles']['fit'])
    require(all(type(i)is int and 0<=i<len(data['rows']) for i in indices), 'cv_landmark_fit_indices')
    forbidden=set(data['roles']['tune']+data['roles']['calibration']+data['roles']['confirmation'])
    require(len(indices)>=20 and len(set(indices))==len(indices)
            and not set(indices).intersection(forbidden), 'cv_landmark_training_role_overlap')
    require(data.get('split_policy','iid')=='iid', 'cv_landmark_group_time_not_qualified')
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow([f'c{i}' for i in range(len(data['kinds']))])
    writer.writerows(data['rows'][i] for i in indices)
    raw=stream.getvalue().encode()
    scope=dict(training_input_sha256=hashlib.sha256(raw).hexdigest(),rows=len(indices),
               physical_rows_sha256=digest([data['row_ids'][i] for i in indices]),
               column_kinds=data['kinds'],task=data['task'],allowed_roles=['fit'],split_policy='iid',
               physical_table_id=data['source_provenance']['physical_table_id'])
    return raw,scope


def validate_options(options):
    require(options == dict(OPTIONS,work_budget=options.get('work_budget'))
            and options['work_budget'] in (2000000000,20000000000,2000000000000), 'cv_landmark_options_unsupported')


def expected_query_rows(coordinate,rows,options):
    """IID folds are balanced; native code caps only each fold's own held-out rows."""
    folds=options['folds']
    cap=options['ridge_rows']if coordinate<=15 or coordinate==22 else(options['small_rows']+folds-1)//folds
    return sum(min(rows//folds+(k<rows%folds),cap)for k in range(folds))


def report_features(report, rows, task, *, options=None):
    options=OPTIONS if options is None else options
    validate_options(options)
    require(report['panel']==OPTIONS['panel'] and report['options']==options
            and report['kind']=='real' and report['privacy']=='private'
            and report['rows']==rows and report['task']==task
            and [c['id'] for c in report['cells']]==list(range(1,24)), 'cv_landmark_report_binding')
    values,mask=[],[]
    for cell in report['cells']:
        valid=cell['state']=='complete' and cell.get('gain')is not None
        if valid:
            require(len(cell['train_rows'])==5 and cell['query_rows']==expected_query_rows(cell['id'],rows,options)
                    and all(1<=n<rows for n in cell['train_rows']), 'cv_landmark_out_of_fold_required')
        values.append(float(cell['gain']) if valid else 0.);mask.append(valid)
    validate_values((values,mask))
    return values,mask


def read_input(directory, expected_manifest, expected_engine, *, data=None, csv_path=None, kinds=None, task=None):
    root=Path(directory)
    for name,limit in [('manifest.json',262144),('report.json',1048576)]:
        path=root/name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size<=limit,
                'cv_landmark_input_file_invalid')
    require(sha(root/'manifest.json')==expected_manifest, 'cv_landmark_manifest_digest')
    m=json.loads((root/'manifest.json').read_bytes())
    require(set(m)=={'format','feature_recipe','scope','options','engine_sha256','report_sha256','producer'}
            and m['format']=='dope-fit-cv-landmark-input-v1' and m['feature_recipe']==FEATURE_RECIPE
            and m['engine_sha256']==expected_engine
            and sha(root/'report.json')==m['report_sha256'], 'cv_landmark_input_binding')
    validate_options(m['options'])
    scope=m['scope']
    require(scope['allowed_roles']==['fit'] and scope['split_policy']=='iid', 'cv_landmark_fit_only_required')
    if data is not None:
        _,expected=training_csv(data)
        require(scope==expected,'cv_landmark_physical_scope_mismatch')
    else:
        require(csv_path is not None and sha(csv_path)==scope['training_input_sha256']
                and kinds==scope['column_kinds'] and task==scope['task'], 'cv_landmark_csv_binding')
    return report_features(json.loads((root/'report.json').read_bytes()),scope['rows'],scope['task'],options=m['options'])


def prepare_input(data, engine, expected_engine, output, *, wall_seconds=300, work_budget=20000000000):
    require(type(wall_seconds)is int and 1<=wall_seconds<=1800, 'cv_landmark_budget_unsupported')
    require(data['source_provenance']['kind']in('owned_prior','admitted_official_train'),
            'cv_landmark_source_not_admitted')
    engine=Path(engine).resolve();require(sha(engine)==expected_engine,'cv_landmark_engine_digest')
    options=dict(OPTIONS,work_budget=work_budget);validate_options(options)
    raw,scope=training_csv(data);out=Path(output);out.mkdir(mode=0o700)
    def write(name,value):
        with(out/name).open('xb')as stream:
            os.chmod(out/name,0o600);stream.write(value);stream.flush();os.fsync(stream.fileno())
    write('fit.csv',raw)
    plan=dict(feature_recipe=FEATURE_RECIPE,scope=scope,options=options,
              engine_sha256=expected_engine,wall_seconds=wall_seconds,threads=4)
    write('plan.json',canonical(plan))
    command=[str(engine),'--threads','4','landmarks','real',str((out/'fit.csv').resolve()),
             '--target',str(len(data['kinds'])-1),'--task',scope['task'],'--json']
    for name,value in options.items():command+=['--'+name.replace('_','-'),str(value)]
    try:
        with(out/'report.json').open('xb')as stream:
            os.chmod(out/'report.json',0o600)
            result=subprocess.run(command,stdout=stream,stderr=subprocess.DEVNULL,timeout=wall_seconds)
            stream.flush();os.fsync(stream.fileno())
        require(result.returncode==0,'cv_landmark_native_failed')
        report_features(json.loads((out/'report.json').read_bytes()),scope['rows'],scope['task'],options=options)
        manifest=dict(format='dope-fit-cv-landmark-input-v1',feature_recipe=FEATURE_RECIPE,scope=scope,
                      options=options,engine_sha256=expected_engine,report_sha256=sha(out/'report.json'),
                      producer=dict(source_sha256=sha(__file__),platform=platform.platform()))
        write('manifest.json',canonical(manifest))
    except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired)as error:
        write('failure.json',canonical(dict(state='failed',error_type=type(error).__name__)))
        raise
    return dict(physical_table_id=scope['physical_table_id'],directory=str(out.resolve()),
                manifest_sha256=sha(out/'manifest.json'),engine_sha256=expected_engine)


class InputCache:
    def __init__(self,index):
        require(index.get('format')=='dope-fit-cv-landmark-cache-v1'
                and index.get('feature_recipe')==FEATURE_RECIPE, 'cv_landmark_cache_recipe')
        self.entries={}
        for row in index['entries']:
            key=row['physical_table_id']
            require(key not in self.entries, 'cv_landmark_cache_duplicate_table')
            self.entries[key]=row
        self.identity=digest(index)

    def attach(self,data):
        data=dict(data);key=data['source_provenance']['physical_table_id']
        require(key in self.entries, 'cv_landmark_cache_entry_required')
        pin=self.entries[key]
        data['landmark_input']=read_input(pin['directory'],pin['manifest_sha256'],pin['engine_sha256'],data=data)
        data['source_provenance']=dict(data['source_provenance'],landmark_input_manifest_sha256=pin['manifest_sha256'],
                                       landmark_feature_sha256=digest(data['landmark_input']))
        return data
