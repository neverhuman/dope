"""Bounded low-priority CV workers with exact scopes and no training admission."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from ..contracts import canonical,digest,require
from ..landmark_features import FEATURE_RECIPE,report_features
from .plan import fit_csv,validate_plan


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('xb')as out:
        os.chmod(path,0o600);out.write(canonical(value));out.flush();os.fsync(out.fileno())


def producer(binary):
    cpu=''
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[1].strip()for line in Path('/proc/cpuinfo').read_text().splitlines()if line.startswith('model name')),'')
    return dict(engine_sha256=sha(binary),platform=platform.platform(),machine=platform.machine(),cpu_model=cpu,
                source_sha256=digest({p.name:sha(p)for p in Path(__file__).parent.glob('*.py')}),
                source_admission='unadmitted; consumer must separately resolve rights/family/physical split policy')


def cache_key(cell,plan,who):
    return digest(dict(cell=cell['key'],feature_recipe=FEATURE_RECIPE,options=plan['options'],producer=who))


def index_key(cell,plan,key):
    s=cell['scope']
    return ('datasets/'+s['dataset_id']+'/physical-'+s['physical_table_id']+'/splits/'+s['split']
            +'/seed-'+str(s['split_seed'])+'/train-'+s['training_rows_sha256']
            +'/cv-'+digest(plan['options'])[:16]+'/producer-'+key+'.json')


def measured_features(report,rows,task,options):
    try:return report_features(report,rows,task,options=options)
    except ValueError as error:
        # A complete private measurement can have no usable gains (e.g. a constant
        # target). Retain every number/state; current encoder admission stays separate.
        require(str(error)=='cv_landmark_input_shape_or_empty'
                and report['rows']==rows and report['task']==task and report['options']==options
                and all(c['state']in('uninformative','unsupported','over_budget','failed')
                        and c.get('gain')is None for c in report['cells']), 'precalc_invalid_empty_measurement')
        return [0.]*23,[False]*23


def lookup(cell,plan,who,store):
    key=cache_key(cell,plan,who);raw=store.get(index_key(cell,plan,key))
    if raw is None:return None
    index=json.loads(raw)
    require(index['format']=='dope-fit-cv-precalc-entry-v1'and index['cell']==cell
            and index['producer']==who and index['feature_recipe']==FEATURE_RECIPE
            and index['training_admitted']is False and index['official_test_access']is False,'precalc_cached_entry_binding')
    members={}
    for name,checksum in index['members'].items():
        require(name in ('manifest.json','report.json','features.json'),'precalc_cached_member')
        body=store.get('objects/sha256/'+checksum+'.json')
        require(body is not None and hashlib.sha256(body).hexdigest()==checksum,'precalc_cached_object_digest');members[name]=body
    require(set(members)=={'manifest.json','report.json','features.json'},'precalc_cached_inventory')
    manifest=json.loads(members['manifest.json']);scope=manifest['scope']
    require(manifest['feature_recipe']==FEATURE_RECIPE and manifest['producer']==who
            and manifest['engine_sha256']==who['engine_sha256']and manifest['options']==plan['options']
            and manifest['report_sha256']==hashlib.sha256(members['report.json']).hexdigest()
            and scope['physical_table_id']==cell['scope']['physical_table_id']
            and scope['physical_rows_sha256']==cell['scope']['training_rows_sha256'],'precalc_cached_manifest_binding')
    values,mask=measured_features(json.loads(members['report.json']),scope['rows'],scope['task'],plan['options'])
    require(json.loads(members['features.json'])==dict(feature_recipe=FEATURE_RECIPE,values=values,mask=mask),'precalc_cached_features_binding')
    return dict(key=key,cell_key=cell['key'],dataset_id=cell['scope']['dataset_id'],split=cell['scope']['split'],
        state='cached',valid_features=sum(mask),members=index['members'],reused_from_plan=index['plan_sha256'],
        model_training=False,official_test_access=False)


def run_cell(cell,plan,*,binary,source,root,threads=4,timeout=1800,rss_bytes=2*2**30,store=None,who=None):
    who=producer(binary)if who is None else who;key=cache_key(cell,plan,who);out=Path(root)/'cells'/key
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    with(out/'lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        terminal=out/'receipt.json'
        if terminal.exists():return json.loads(terminal.read_bytes())
        require(not(out/'started.json').exists(),'precalc_interrupted_attempt_requires_review')
        private(out/'started.json',dict(pid=os.getpid(),key=key,host=platform.node(),unix=time.time()))
        started=time.monotonic();receipt=dict(key=key,cell_key=cell['key'],dataset_id=cell['scope']['dataset_id'],
            split=cell['scope']['split'],producer=who,model_training=False,official_test_access=False,
            state='failed',threads=threads,wall_limit=timeout,rss_limit=rss_bytes)
        try:
            if cell['state']=='unsupported':receipt.update(state='unsupported',reason=cell['reason'])
            else:
                raw,scope=fit_csv(source,cell['scope']);csv=out/'fit.csv'
                with csv.open('xb')as stream:os.chmod(csv,0o600);stream.write(raw)
                require(sha(binary)==who['engine_sha256'],'precalc_binary_changed')
                args=[str(binary),'--threads',str(threads),'landmarks','real',str(csv),'--target',str(cell['scope']['columns']-1),
                      '--task',scope['task'],'--json']
                for name,value in plan['options'].items():args+=['--'+name.replace('_','-'),str(value)]
                env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',RAYON_NUM_THREADS=str(threads),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with(out/'report.json').open('xb')as report,(out/'native.private.log').open('xb')as log:
                    os.chmod(report.name,0o600);os.chmod(log.name,0o600)
                    p=subprocess.Popen(args,stdout=report,stderr=log,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
                    private(out/'native-owner.json',dict(pid=p.pid,host=platform.node(),key=key,binary_sha256=who['engine_sha256']))
                    end=time.monotonic()+timeout;peak=0;disposition=None
                    while p.poll()is None:
                        if time.monotonic()>=end:disposition='native_wall_limit'
                        observed=subprocess.run(['/bin/ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True,timeout=3)
                        if observed.returncode==0 and observed.stdout.strip().isdigit():
                            peak=max(peak,int(observed.stdout.strip())*1024)
                            if peak>rss_bytes:disposition='native_rss_limit'
                        elif p.poll()is None:disposition='native_memory_observation_failed'
                        if disposition:
                            # Only this freshly created, recorded native process is stopped.
                            try:p.kill()
                            except ProcessLookupError:pass
                            p.wait();break
                        time.sleep(.25)
                receipt.update(native_exit=p.returncode,sampled_rss_peak=peak)
                if disposition is not None:
                    receipt.update(state='overbudget'if disposition in ('native_wall_limit','native_rss_limit')else'failed',reason=disposition)
                    raise ValueError('precalc_native_resource_limit')
                require(p.returncode==0,'precalc_native_exit')
                report=json.loads((out/'report.json').read_bytes())
                values,mask=measured_features(report,scope['rows'],scope['task'],plan['options'])
                manifest=dict(format='dope-fit-cv-landmark-input-v1',feature_recipe=FEATURE_RECIPE,scope=scope,
                    options=plan['options'],engine_sha256=who['engine_sha256'],report_sha256=sha(out/'report.json'),producer=who)
                private(out/'manifest.json',manifest);private(out/'features.json',dict(feature_recipe=FEATURE_RECIPE,values=values,mask=mask))
                receipt.update(state='complete',valid_features=sum(mask),encoder_input_usable=any(mask),
                    native_cell_states={state:sum(c['state']==state for c in report['cells'])for state in ('complete','uninformative','unsupported','over_budget','failed')},training_rows=scope['rows'],
                    training_rows_sha256=scope['physical_rows_sha256'],manifest_sha256=sha(out/'manifest.json'),
                    report_sha256=sha(out/'report.json'),feature_sha256=sha(out/'features.json'))
                if store is not None:publish(cell,plan,receipt,out,store)
        except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError)as error:
            if receipt['state']=='complete':receipt['state']='publication_failed'
            receipt.update(error_type=type(error).__name__);receipt.setdefault('reason','precalc_input_native_or_publication_refused')
        receipt['elapsed_seconds']=time.monotonic()-started
        if(out/'native.private.log').exists():receipt['native_log_sha256']=sha(out/'native.private.log')
        private(terminal,receipt)
        if store is not None:
            store.put('receipts/'+key+'/'+Path(root).name+'.json',canonical(receipt))
            if receipt['state']not in ('complete','cached'):store.put('dispositions/'+key+'.json',canonical(receipt))
        return receipt


def publish(cell,plan,receipt,out,store):
    require(receipt['state']=='complete','precalc_publish_requires_complete')
    members={}
    for name in ('manifest.json','report.json','features.json'):
        raw=(out/name).read_bytes();checksum=hashlib.sha256(raw).hexdigest()
        store.put('objects/sha256/'+checksum+'.json',raw);members[name]=checksum
    index=dict(format='dope-fit-cv-precalc-entry-v1',cell=cell,producer=receipt['producer'],members=members,
               plan_sha256=plan['sha256'],feature_recipe=FEATURE_RECIPE,training_admitted=False,official_test_access=False)
    store.put(index_key(cell,plan,receipt['key']),canonical(index))
    return index


def allocation_snapshot(root):
    from ..resources import snapshot
    v=snapshot(root)
    if platform.system()=='Darwin':
        import re
        output=subprocess.check_output(['vm_stat'],text=True)
        page=int(re.search(r'page size of (\d+) bytes',output).group(1))
        counters={name:int(value.replace('.',''))for name,value in re.findall(r'^(Pages [^:]+):\s+(\d+\.)',output,re.M)}
        v['memory_available']=page*sum(counters.get(name,0)for name in ('Pages free','Pages inactive','Pages speculative'))
    require(v['memory_available']is not None,'precalc_memory_observation_required')
    return v


def run(plan,*,binary,root,source_client,bucket,store,workers,threads=4,timeout=1800,shard=0,shards=1,wall_seconds=43200):
    validate_plan(plan);require(1<=workers<=32 and 1<=threads<=8 and 0<=shard<shards<=16
            and 1<=timeout<=1800 and timeout+30<wall_seconds<=43200,'precalc_allocation')
    root=Path(root);root.mkdir(mode=0o700,parents=True);os.umask(0o077);os.nice(10);who=dict(producer(binary),threads=threads)
    admission=allocation_snapshot(root)
    require(admission['memory_available']is None or admission['memory_available']>=8*2**30+workers*4*2**30,
            'precalc_aggregate_memory_allocation_refused')
    require(workers*threads<=max(0,admission['cpu_available']-8-int(admission['load'][0])),
            'precalc_spare_cpu_allocation_refused')
    private(root/'admission.json',dict(resources=admission,processes=subprocess.check_output(
        ['ps','-eo','user,pid,ppid,ni,pcpu,pmem,comm']).decode()))
    require(not(root/'ownership.json').exists(),'precalc_duplicate_run_refused')
    private(root/'ownership.json',dict(pid=os.getpid(),host=platform.node(),plan_sha256=plan['sha256']))
    private(root/'plan.json',plan);private(root/'producer.json',who);started=time.monotonic();assigned=[c for c in plan['cells']if int(digest(c['scope']['dataset_id'])[:8],16)%shards==shard];records=[]
    from .store import train_source
    def one(cell):
        existing=root/'cells'/cache_key(cell,plan,who)/'receipt.json'
        if existing.exists():
            receipt=json.loads(existing.read_bytes())
            if receipt['state']=='complete':require(lookup(cell,plan,who,store)is not None,'precalc_completed_cache_missing')
            return receipt
        prior=store.get('dispositions/'+cache_key(cell,plan,who)+'.json')
        if prior is not None:return json.loads(prior)
        cached=lookup(cell,plan,who,store)
        if cached is not None:
            private(root/'cells'/cached['key']/'receipt.json',dict(cached,producer=who));return cached
        source=None
        if cell['state']=='pending':source=train_source(source_client,bucket,cell,root/'source-cache')
        return run_cell(cell,plan,binary=binary,source=source,root=root,threads=threads,timeout=timeout,store=store,who=who)
    # The caller freezes workers from current resource/ownership evidence. Replenishment
    # checks RAM/disk and leaves a conservative foreground CPU reserve each time.
    def admit(active):
        v=allocation_snapshot(root)
        return (time.monotonic()-started+timeout+30<wall_seconds and v['disk_available']>10*2**30
                and (v['memory_available']is None or v['memory_available']>8*2**30)
                and v['load'][0]<v['cpu_available']-8+active*threads)
    summary=dict(format='dope-fit-cv-precalc-progress-v1',plan_sha256=plan['sha256'],scheduled=len(assigned),terminal=0,pending=len(assigned),records=[],workers=workers,threads_per_worker=threads,model_training=False,official_test_access=False)
    with(root/'run.lock').open('a')as lock,concurrent.futures.ThreadPoolExecutor(max_workers=workers)as pool:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pending=iter(assigned);active={}
        def submit():
            if not admit(len(active)):return False
            cell=next(pending,None)
            if cell is None:return False
            active[pool.submit(one,cell)]=cell;return True
        for _ in range(workers):submit()
        while active:
            done,_=concurrent.futures.wait(active,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                cell=active.pop(future)
                try:records.append(future.result())
                except Exception as error:
                    failure=dict(cell_key=cell['key'],key=cache_key(cell,plan,who),state='interrupted',
                                 error_type=type(error).__name__,model_training=False,official_test_access=False)
                    private(root/'failures'/(cell['key']+'.json'),failure)
                    store.put('dispositions/'+failure['key']+'.json',canonical(failure));records.append(failure)
                submit()
            summary=dict(format='dope-fit-cv-precalc-progress-v1',plan_sha256=plan['sha256'],scheduled=len(assigned),
                         terminal=len(records),pending=len(assigned)-len(records),records=records,workers=workers,
                         threads_per_worker=threads,model_training=False,official_test_access=False)
            tmp=root/'progress.staging.json';tmp.write_bytes(canonical(summary));os.replace(tmp,root/'progress.json')
    summary.update(state='complete'if len(records)==len(assigned)else'resource_or_budget_paused')
    private(root/'terminal.json',summary);return summary
