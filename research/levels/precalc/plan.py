"""Finite official-TRAIN schedules, exact row scopes and immutable model-independent keys."""
import csv
import hashlib
import io
import json
import math
from functools import lru_cache
from pathlib import Path
from ..contracts import canonical, digest, require
from ..landmark_features import FEATURE_RECIPE, OPTIONS

CATALOG_SHA256='ab9fda8d2dea46067b70a42812e9d3c1d9dc6ba025780100df34377e81aa1120'
SPLITS=('fit-iid-65-v1',)+tuple(f'outer5-iid-fold{k}-v1' for k in range(5))


def physical_id(dataset):return digest(dict(namespace='jope-catalog-physical-v1',dataset_id=dataset))


@lru_cache(maxsize=1)
def iid_order(rows,physical,seed):
    return tuple(sorted(range(rows),key=lambda i:hashlib.sha256(f'{seed}:{physical}:{i}'.encode()).digest()))


def training_rows(rows,physical,split,seed):
    require(type(rows)is int and 20<=rows<=200000 and split in SPLITS, 'precalc_split_envelope')
    order=iid_order(rows,physical,seed)
    if split=='fit-iid-65-v1':chosen=order[:int(rows*.65)]
    else:
        fold=int(split[len('outer5-iid-fold')]);chosen=[i for rank,i in enumerate(order)if rank%5!=fold]
    require(len(chosen)>=20,'precalc_training_rows_insufficient')
    return sorted(chosen)


def make_plan(catalog,*,source_commit,split_seed=20261023,splits=SPLITS):
    require(len(source_commit)==40 and all(c in '0123456789abcdef'for c in source_commit),'precalc_source_commit')
    raw=Path(catalog).read_bytes();require(hashlib.sha256(raw).hexdigest()==CATALOG_SHA256,'precalc_catalog_digest')
    require(set(splits)<=set(SPLITS)and len(splits)==len(set(splits))and splits,'precalc_split_registry')
    cells=[];seen=set()
    for row in map(json.loads,raw.splitlines()):
        identity=row['dataset_hash'];require(identity not in seen,'precalc_duplicate_dataset');seen.add(identity)
        train=row['files']['train'];physical=physical_id(identity)
        task={'binary':'classification','multiclass':'classification','classification':'classification','regression':'regression'}.get(row['task'])
        for split in splits:
            reason=None
            if task is None:reason='unsupported_task'
            elif train['rows']>200000 or train['rows']*train['columns']>2000000 or train['bytes']>200000000:reason='source_envelope'
            elif not 2<=train['columns']<=257:reason='source_columns_unsupported'
            try:indices=training_rows(train['rows'],physical,split,split_seed)if reason is None else []
            except ValueError:indices=[];reason='training_partition_too_small'
            scope=dict(physical_table_id=physical,split=split,split_seed=split_seed,
                training_rows_sha256=digest([physical+f':{i}'for i in indices]),training_rows=len(indices),
                source_role='official_train',source_key='blobs/sha256/'+train['sha256'],train_sha256=train['sha256'],
                train_bytes=train['bytes'],rows=train['rows'],columns=train['columns'],has_header=train['has_header'],
                target=row['target'],task=task,dataset_id=identity,source_metadata_sha256=digest(row))
            cells.append(dict(key=digest(scope),scope=scope,state='unsupported'if reason else'pending',reason=reason))
    plan=dict(format='dope-fit-cv-precalc-plan-v1',catalog_sha256=CATALOG_SHA256,source_commit=source_commit,
              feature_recipe=FEATURE_RECIPE,options=dict(OPTIONS,work_budget=2000000000000),split_seed=split_seed,
              splits=list(splits),cells=cells,source_access='pinned_official_TRAIN_only',model_training=False,
              admission='rights/family/split-policy metadata retained unadmitted; this plan grants no training admission')
    plan['sha256']=digest(plan);return plan


def validate_plan(plan,catalog=None):
    p=dict(plan);expected=p.pop('sha256');require(expected==digest(p)and p['format']=='dope-fit-cv-precalc-plan-v1','precalc_plan_digest')
    require(p['catalog_sha256']==CATALOG_SHA256 and p['source_access']=='pinned_official_TRAIN_only'
            and p['model_training']is False,'precalc_source_scope')
    keys=set()
    for cell in p['cells']:
        scope=cell['scope'];require(cell['key']==digest(scope)and cell['key']not in keys,'precalc_cell_identity');keys.add(cell['key'])
        require(scope['source_role']=='official_train'and scope['source_key']=='blobs/sha256/'+scope['train_sha256'], 'precalc_non_TRAIN_object')
    if catalog is not None:
        raw=Path(catalog).read_bytes();require(hashlib.sha256(raw).hexdigest()==CATALOG_SHA256,'precalc_catalog_digest')
        entries={r['dataset_hash']:r for r in map(json.loads,raw.splitlines())}
        for c in p['cells']:
            s=c['scope'];row=entries[s['dataset_id']];train=row['files']['train']
            require(s['source_metadata_sha256']==digest(row)and s['train_sha256']==train['sha256']
                    and s['train_bytes']==train['bytes']and s['rows']==train['rows']
                    and s['columns']==train['columns']and s['has_header']==train['has_header']
                    and s['target']==row['target'],'precalc_catalog_source_binding')
    return plan


def fit_csv(source,scope):
    """Infer predictor types only inside the chosen training rows; target is explicitly declared."""
    raw=Path(source).read_bytes();require(len(raw)==scope['train_bytes']and hashlib.sha256(raw).hexdigest()==scope['train_sha256'],'precalc_source_digest')
    reader=csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline=''));count=scope['columns'];index=scope['target']['index']
    require(type(index)is int,'precalc_declared_target_required');index=index+count if index<0 else index
    require(0<=index<count,'precalc_target_index')
    if scope['has_header']:
        header=next(reader);require(len(header)==count and len(set(header))==count and header[index]==scope['target']['name'],'precalc_target_header')
    indices=training_rows(scope['rows'],scope['physical_table_id'],scope['split'],scope['split_seed']);selected=set(indices)
    order=[j for j in range(count)if j!=index]+[index];rows=[];n=0
    for i,row in enumerate(reader):
        require(len(row)==count,'precalc_row_width');n+=1
        if i in selected:rows.append([row[j]for j in order])
    require(n==scope['rows']and len(rows)==len(indices),'precalc_row_count')
    kinds=[]
    for j in range(count):
        values=[r[j]for r in rows if r[j].strip()not in('', '?','NA','NaN','null')]
        require(values,'precalc_empty_training_column')
        def numeric(v):
            try:return math.isfinite(float(v))
            except ValueError:return False
        kinds.append('numeric'if all(map(numeric,values))else'categorical')
    kinds[-1]='numeric'if scope['task']=='regression'else'categorical'
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow([f'c{i}'for i in range(count)]);writer.writerows(rows)
    body=stream.getvalue().encode()
    bound=dict(training_input_sha256=hashlib.sha256(body).hexdigest(),rows=len(rows),
        physical_rows_sha256=digest([scope['physical_table_id']+f':{i}'for i in indices]),column_kinds=kinds,
        task=scope['task'],allowed_roles=['fit'],split_policy='iid',physical_table_id=scope['physical_table_id'])
    require(bound['physical_rows_sha256']==scope['training_rows_sha256'],'precalc_row_identity_drift')
    return body,bound
