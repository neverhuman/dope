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
    args=parser.parse_args(argv)
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


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ResearchError,OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
        # Third-party parser/backend diagnostics can contain source cells or credentials.
        print(json.dumps(dict(state='failed',error_type=type(error).__name__,
                              reason=ResearchError.reason,repair_hint=ResearchError.repair_hint,
                              docs=ResearchError.docs_url)))
        raise SystemExit(1)
