"""Explicit local training/export commands. No dataset discovery or automatic admission."""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
from .contracts import ModelConfig, TrainConfig, ResearchError, require


def source_fingerprint():
    root=Path(__file__).parent
    h=hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        h.update(path.name.encode());h.update(b'\0');h.update(path.read_bytes());h.update(b'\0')
    return h.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('train');run.add_argument('--config',type=Path,required=True)
    run.add_argument('--landmark-input-cache',type=Path)
    run.add_argument('--corpus',type=Path);run.add_argument('--prior-only',action='store_true')
    run.add_argument('--prior-recipe', choices=('owned-mechanisms-v1','owned-mechanisms-unique-v2'),
                     default='owned-mechanisms-v1',help='Explicit prior stream; v1 remains historical replay')
    run.add_argument('--out',type=Path,required=True);run.add_argument('--resume',type=Path)
    run.add_argument('--ordinary-base',type=Path);run.add_argument('--stop-after',type=int)
    run.add_argument('--phase',choices=['ordinary','ultra-residual'],default='ordinary')
    run.add_argument('--repair-reason');run.add_argument('--allow-busy-cpu',action='store_true',
        help='CPU threads remain bounded and nice10; never overrides GPU/memory/disk refusal')
    export=sub.add_parser('export');export.add_argument('--checkpoint',type=Path,required=True)
    export.add_argument('--out',type=Path,required=True)
    labels=sub.add_parser('landmarks', help='Create separate private fit-only IID profile labels')
    labels.add_argument('--dataset',type=Path,required=True)
    labels.add_argument('--engine',type=Path,required=True)
    labels.add_argument('--expected-engine-sha256',required=True)
    labels.add_argument('--out',type=Path,required=True)
    labels.add_argument('--wall-seconds',type=int,default=300)
    inputs=sub.add_parser('landmark-inputs',help='Private five-fold CV encoder features from training rows only')
    source=inputs.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset',type=Path)
    source.add_argument('--owned-family',type=int,choices=range(9))
    inputs.add_argument('--owned-seed',type=int,default=20261023)
    inputs.add_argument('--engine',type=Path,required=True)
    inputs.add_argument('--expected-engine-sha256',required=True)
    inputs.add_argument('--out',type=Path,required=True)
    inputs.add_argument('--wall-seconds',type=int,default=300)
    from .landmark_features import FEATURE_RECIPE,FEATURE_RECIPES
    inputs.add_argument('--feature-recipe',choices=FEATURE_RECIPES,default=FEATURE_RECIPE)
    derived=sub.add_parser('derive-landmark-inputs',help='Observed-score view of a saved private fit-CV report; no new fits')
    derived.add_argument('--input',type=Path,required=True)
    derived.add_argument('--expected-manifest-sha256',required=True)
    derived.add_argument('--expected-engine-sha256',required=True)
    derived.add_argument('--out',type=Path,required=True)
    enc=sub.add_parser('encode',help='Pinned Encoder629 features with optional/progressive landmarks')
    enc.add_argument('--csv',type=Path,required=True)
    enc.add_argument('--target',required=True,help='Target column name or 0-based index')
    enc.add_argument('--task',choices=('regression','binary','classification','multiclass'),required=True)
    enc.add_argument('--weights',type=Path,required=True)
    from .encoder import PINNED_MULTITASK_SAFETENSORS_SHA256
    enc.add_argument('--weights-sha256',default=PINNED_MULTITASK_SAFETENSORS_SHA256)
    enc.add_argument('--landmark-mode',choices=('none','fast','progressive','full'),default='full')
    enc.add_argument('--landmark-subsample',type=int,
                     help='Stratified training-row subsample for native CV (progressive default 300)')
    enc.add_argument('--bundle-json',type=Path,help='Cached LandmarkBundle v1 JSON (progressive start / full check)')
    enc.add_argument('--report-json',type=Path,help='Existing native landmarks report (full mode)')
    enc.add_argument('--engine',type=Path,help='Native dope binary for landmark CV (full/progressive)')
    enc.add_argument('--device',default='cpu')
    enc.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='encode':
        return run_encode(args)
    if args.command=='derive-landmark-inputs':
        from .landmark_features import derive_observed
        print(json.dumps(derive_observed(args.input,args.expected_manifest_sha256,args.expected_engine_sha256,args.out)))
        return 0
    if args.command=='landmark-inputs':
        from .landmark_features import prepare_input
        from .corpus import read_dataset
        from .contracts import digest
        from .resources import Guard,snapshot
        resource=snapshot(args.out.parent,cuda=False)
        require(resource['load'][0]<resource['cpu_available']*.85,'cpu_load_allocation_refused')
        Guard(args.out.parent,args.out.parent/'landmark-input-resources.jsonl').admit()
        if hasattr(os,'nice'):os.nice(10)
        if args.dataset is not None:
            data=read_dataset(json.loads(args.dataset.read_bytes()),include_landmarks=False)
        else:
            from .priors import table
            data=table(args.owned_family,args.owned_seed)
            data['source_provenance']=dict(kind='owned_prior',physical_table_id=digest(
                dict(family=args.owned_family,seed=args.owned_seed)))
        print(json.dumps(prepare_input(data,args.engine,args.expected_engine_sha256,args.out,
                                      wall_seconds=args.wall_seconds,recipe=args.feature_recipe)))
        return 0
    if args.command=='landmarks':
        from .corpus import prepare_landmarks
        from .resources import Guard,snapshot
        resource=snapshot(args.out.parent,cuda=False)
        require(resource['load'][0]<resource['cpu_available']*.85, 'cpu_load_allocation_refused')
        Guard(args.out.parent,args.out.parent/'landmark-resources.jsonl',cuda=False).admit()
        if hasattr(os,'nice'):os.nice(10)
        print(json.dumps(prepare_landmarks(json.loads(args.dataset.read_bytes()), args.engine,
                         args.expected_engine_sha256,args.out,wall_seconds=args.wall_seconds)))
        return 0
    if args.command=='export':
        from .packages import export_package
        manifest=export_package(args.checkpoint,args.out)
        print(json.dumps(dict(coordinate_space_id=manifest['coordinate_space_id'],certified=False)))
        return 0
    raw=json.loads(args.config.read_bytes())
    require(set(raw)=={'model','train'}, 'unknown_training_config')
    require(set(raw['model'])<={f.name for f in fields(ModelConfig)}
            and set(raw['train'])<={f.name for f in fields(TrainConfig)}, 'unknown_training_option')
    model,config=ModelConfig(**raw['model']).validate(),TrainConfig(**raw['train']).validate()
    require(args.prior_only != bool(args.corpus), 'choose_prior_only_or_admitted_corpus')
    require(args.out.parent.is_dir() and not args.out.exists(), 'new_run_parent_required')
    from .resources import Guard,snapshot
    resource=snapshot(args.out.parent,cuda=config.device=='cuda')
    require(args.allow_busy_cpu or resource['load'][0]<resource['cpu_available']*.85, 'cpu_load_allocation_refused')
    guard=Guard(args.out.parent,args.out/'resources.jsonl',cuda=config.device=='cuda');guard.admit()
    if hasattr(os,'nice'):os.nice(10)
    from .corpus import Corpus
    from .train import train
    corpus=Corpus(None if args.corpus is None else json.loads(args.corpus.read_bytes()),
                  seed=config.seed,prior_only=args.prior_only,prior_recipe=args.prior_recipe,
                  input_cache=None if args.landmark_input_cache is None else json.loads(args.landmark_input_cache.read_bytes()))
    _,terminal=train(model,config,corpus,args.out,source_sha256=source_fingerprint(),resume=args.resume,
        phase=args.phase,base_checkpoint=args.ordinary_base,stop_after=args.stop_after,
        resource_check=guard,repair_reason=args.repair_reason)
    print(json.dumps(terminal));return 0 if terminal['state']=='complete' else 2


def _read_csv_table(path, target):
    import csv
    with Path(path).open(newline='') as stream:
        rows=list(csv.reader(stream))
    require(len(rows)>=2,'encode_csv_empty')
    header,body=rows[0],rows[1:]
    idx=header.index(target) if target in header else int(target)
    require(0<=idx<len(header) and all(len(r)==len(header) for r in body),'encode_csv_shape')
    X=[[v for j,v in enumerate(r) if j!=idx] for r in body]
    y=[r[idx] for r in body]
    return X,y,idx


def run_encode(args):
    """``encode``: none/fast/full -> one JSON record; progressive -> NDJSON per update."""
    import time
    import numpy as np
    import torch
    from .encoder import DopeEncoder629
    from .multilevel_model import build_conditioning
    from .optional_landmarks import (DEFAULT_PROGRESSIVE_SUBSAMPLE, LandmarkBundle, native_group_runner,
                                     progressive_steps, report_task)
    from .scaling import FitOnlyTableTransform
    mode=args.landmark_mode
    task={'binary':'classification'}.get(args.task,args.task)
    X_raw,y_raw,target_idx=_read_csv_table(args.csv,args.target)
    transform=FitOnlyTableTransform.fit(X_raw,y_raw,task=task)
    X,y=transform.transform(X_raw,y_raw)
    y_train_raw=y_raw if task=='regression' else None
    encoder=DopeEncoder629.from_pretrained(args.weights,expected_sha256=args.weights_sha256,device=args.device)
    cached=None if args.bundle_json is None else LandmarkBundle.from_json(args.bundle_json.read_bytes())

    def emit(seq,bundle,native_report,started,subsample_n=None):
        out=build_conditioning(X,y,task=task,bundle=bundle,mode=mode,native_report=native_report,
                               level='high',subsample_n=subsample_n,y_train_raw=y_train_raw)
        dev=torch.device(args.device)
        with torch.no_grad():
            feats=encoder(torch.from_numpy(out['vec256'][None].copy()).to(dev),
                          torch.from_numpy(out['col_landmarks'][None].copy()).to(dev))
        raw=feats.detach().to('cpu').contiguous().numpy().astype('<f4').tobytes()
        shown=bundle if bundle is not None else LandmarkBundle.from_report(native_report,subsample_n=subsample_n)
        record=dict(seq=seq,status=shown.status,conditioning_digest=shown.conditioning_digest(),
                    features629_sha256=hashlib.sha256(raw).hexdigest(),
                    latency_ms=round((time.perf_counter()-started)*1000.0,3),
                    mode=mode,recipe=out['recipe'],effective_conditioning_digest=out['digest'])
        print(json.dumps(record,sort_keys=True) if args.json else
              f"seq={seq} status={record['status']} features629_sha256={record['features629_sha256']}",flush=True)

    started=time.perf_counter()
    if mode in ('none','fast'):
        emit(0,LandmarkBundle.zero() if cached is None else cached,None,started);return 0
    if mode=='full':
        if args.report_json is not None:
            report=json.loads(args.report_json.read_bytes())
        else:
            require(args.engine is not None,'encode_full_requires_report_or_engine')
            from .optional_landmarks import mlc_v3_training_options
            import subprocess
            cmd=[str(args.engine),'--threads','4','landmarks','real',str(args.csv),'--target',str(target_idx),
                 '--task',report_task(task),'--json']
            for name,value in mlc_v3_training_options().items():cmd+=['--'+name.replace('_','-'),str(value)]
            if args.landmark_subsample is not None:cmd+=['--subsample',str(args.landmark_subsample)]
            result=subprocess.run(cmd,capture_output=True,check=False)
            require(result.returncode==0,'encode_native_failed')
            report=json.loads(result.stdout)
        emit(0,cached,report,started,subsample_n=args.landmark_subsample);return 0
    require(args.engine is not None,'encode_progressive_requires_engine')
    n=DEFAULT_PROGRESSIVE_SUBSAMPLE if args.landmark_subsample is None else args.landmark_subsample
    runner=native_group_runner(args.engine,args.csv,target=target_idx,task=task,subsample_n=n)
    source=hashlib.sha256(Path(args.csv).read_bytes()+f'\0{target_idx}\0{n}'.encode()).hexdigest()
    if cached is not None and not cached.is_zero():
        require(cached.subsample_n==n and cached.source_digest==source,'encode_cached_bundle_scope')
    for seq,_,bundle,_ in progressive_steps(runner,subsample_n=n,source_digest=source,start=cached):
        emit(seq,bundle,None,started);started=time.perf_counter()
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ResearchError,OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
        # Third-party parser/backend diagnostics can contain source cells or credentials.
        print(json.dumps(dict(state='failed',error_type=type(error).__name__,
                              reason=ResearchError.reason,repair_hint=ResearchError.repair_hint,
                              docs=ResearchError.docs_url)))
        raise SystemExit(1)
