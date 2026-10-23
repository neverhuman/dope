"""Explicit private fit-CV precalculation and publication commands; no model training."""
import argparse
import json
import os
import platform
from pathlib import Path
from ..contracts import canonical,require,digest
from .plan import make_plan,validate_plan
from .store import PrivateStore,LocalStore,TrainURLSource
from .worker import run,publish,private,lookup,producer


def client(profile):
    import boto3
    from botocore.config import Config
    return boto3.Session(profile_name=profile).client('s3',config=Config(retries={'total_max_attempts':1},
        connect_timeout=10,read_timeout=60,max_pool_connections=64))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan');plan.add_argument('--catalog',type=Path,required=True);plan.add_argument('--source-commit',required=True);plan.add_argument('--split-seed',type=int,default=20261023);plan.add_argument('--out',type=Path,required=True)
    job=sub.add_parser('run');job.add_argument('--plan',type=Path,required=True);job.add_argument('--binary',type=Path,required=True);job.add_argument('--expected-binary-sha256',required=True);job.add_argument('--catalog',type=Path,required=True);job.add_argument('--root',type=Path,required=True)
    job.add_argument('--workers',type=int,required=True);job.add_argument('--threads',type=int,default=4);job.add_argument('--timeout',type=int,default=1800)
    job.add_argument('--wall-seconds',type=int,default=43200);job.add_argument('--shard',type=int,default=0);job.add_argument('--shards',type=int,default=1)
    job.add_argument('--source-urls',type=Path);job.add_argument('--local-store',type=Path)
    upload=sub.add_parser('publish');upload.add_argument('--plan',type=Path,required=True);upload.add_argument('--root',type=Path,required=True)
    index=sub.add_parser('index');index.add_argument('--plan',type=Path,required=True);index.add_argument('--runs',type=Path,nargs='+',required=True)
    repair=sub.add_parser('recover');repair.add_argument('--source-run',type=Path,required=True);repair.add_argument('--root',type=Path,required=True)
    repair.add_argument('--catalog',type=Path,required=True);repair.add_argument('--binary',type=Path,required=True)
    repair.add_argument('--expected-binary-sha256',required=True);repair.add_argument('--source-commit',required=True);repair.add_argument('--local-store',type=Path)
    download=sub.add_parser('download');download.add_argument('--entry-key',required=True)
    download.add_argument('--expected-entry-sha256',required=True);download.add_argument('--out',type=Path,required=True)
    view=sub.add_parser('derive-observed-view');view.add_argument('--source-index-key',required=True)
    view.add_argument('--expected-source-index-sha256',required=True);view.add_argument('--workers',type=int,default=4)
    selected=sub.add_parser('download-view');selected.add_argument('--entry-key',required=True)
    selected.add_argument('--expected-entry-sha256',required=True);selected.add_argument('--out',type=Path,required=True)
    for parser in (job,upload,download,index,repair,view,selected):
        parser.add_argument('--bucket',default='veox-jopedime-datasets-use2');parser.add_argument('--profile',default='veox-jepson')
        parser.add_argument('--prefix',default='derived/dope/landmark-inputs/fit-cv-v1/precalc-20261023')
    args=p.parse_args();os.umask(0o077)
    if args.command=='plan':
        value=make_plan(args.catalog,source_commit=args.source_commit,split_seed=args.split_seed);private(args.out,value)
        print(json.dumps(dict(plan_sha256=value['sha256'],scheduled=len(value['cells']),splits=value['splits'],model_training=False)));return
    require(args.bucket=='veox-jopedime-datasets-use2','precalc_bucket_not_authorized')
    if args.command in('derive-observed-view','download-view'):
        from .views import derive_corpus,materialize_view
        store=PrivateStore(client(args.profile),args.bucket,args.prefix)
        if args.command=='derive-observed-view':
            from .worker import allocation_snapshot
            resources=allocation_snapshot(Path.cwd())
            require(1<=args.workers<=16 and resources['memory_available']>=12*2**30
                    and resources['disk_available']>=10*2**30
                    and resources['load'][0]<resources['cpu_available']-8,'precalc_view_resource_refused')
            os.nice(10)
            print(json.dumps(derive_corpus(store,args.source_index_key,args.expected_source_index_sha256,workers=args.workers)))
        else:print(json.dumps(materialize_view(store,args.entry_key,args.expected_entry_sha256,args.out)))
        return
    if args.command=='recover':
        from .recover import recover
        store=LocalStore(args.local_store)if args.local_store else PrivateStore(client(args.profile),args.bucket,args.prefix)
        result=recover(args.source_run,args.root,catalog=args.catalog,binary=args.binary,expected_engine=args.expected_binary_sha256,
                       source_commit=args.source_commit,store=store)
        print(json.dumps({k:result[k]for k in ('state','scheduled','recovered','refused','new_native_executions')}));return
    if args.command=='download':
        from .store import materialize
        print(json.dumps(materialize(PrivateStore(client(args.profile),args.bucket,args.prefix),args.entry_key,args.expected_entry_sha256,args.out)));return
    value=validate_plan(json.loads(args.plan.read_bytes()),args.catalog if args.command=='run'else None)
    if args.command=='index':
        from .coverage import build
        print(json.dumps(build(value,args.runs,PrivateStore(client(args.profile),args.bucket,args.prefix))));return
    if args.command=='run':
        require(producer(args.binary)['engine_sha256']==args.expected_binary_sha256,'precalc_binary_digest')
        c=None if args.source_urls else client(args.profile)
        source=TrainURLSource(json.loads(args.source_urls.read_bytes()),args.bucket,value)if args.source_urls else c
        store=LocalStore(args.local_store)if args.local_store else PrivateStore(c,args.bucket,args.prefix)
        require(not args.source_urls or args.local_store is not None,'precalc_delegated_source_requires_local_store')
        result=run(value,binary=args.binary,root=args.root,source_client=source,bucket=args.bucket,store=store,
            workers=args.workers,threads=args.threads,timeout=args.timeout,shard=args.shard,shards=args.shards,wall_seconds=args.wall_seconds)
        store.put('runs/'+value['sha256']+'/'+digest(platform.node())[:16]+'/'+args.root.name+'/terminal.json',canonical(result))
        print(json.dumps({k:result[k]for k in ('state','scheduled','terminal','pending')}));return
    store=PrivateStore(client(args.profile),args.bucket,args.prefix);cells={c['key']:c for c in value['cells']};counts={}
    for path in sorted(args.root.glob('cells/*/receipt.json')):
        receipt=json.loads(path.read_bytes());cell=cells[receipt['cell_key']]
        if receipt['state']=='complete':publish(cell,value,receipt,path.parent,store)
        store.put('receipts/'+receipt['key']+'/'+args.root.name+'.json',canonical(receipt));counts[receipt['state']]=counts.get(receipt['state'],0)+1
    print(json.dumps(dict(published_receipts=counts,model_training=False)))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ImportError)as error:
        # Network/native/input diagnostics may contain private values or delegated URLs.
        print(json.dumps(dict(state='failed',error_type=type(error).__name__,repair_hint='Preserve the attempt; inspect private receipts and repair the declared allocation/input before a new run.')))
        raise SystemExit(1)
