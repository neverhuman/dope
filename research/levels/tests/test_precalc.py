"""Private split cache safety/immutability/reuse; no scientific quality claims."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research.levels.contracts import canonical,digest,ResearchError
from research.levels.precalc.plan import make_plan,validate_plan,training_rows,fit_csv,physical_id,SPLITS
from research.levels.precalc.store import PrivateStore,LocalStore,train_source,TrainURLSource
from research.levels.precalc.worker import lookup,publish,cache_key,measured_features
from research.levels.landmark_features import FEATURE_RECIPE,OPTIONS


def catalog_fixture(root):
    raw=('target,x\n'+''.join(f'{i*2},{i}\n'for i in range(100))).encode();source=root/'source.csv';source.write_bytes(raw)
    row=dict(dataset_hash='a'*16,files=dict(train=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),rows=100,columns=2,has_header=True),test=dict(sha256='SECRET_TEST_KEY')),
             task='regression',target=dict(index=0,name='target'))
    path=root/'catalog.jsonl';path.write_bytes(canonical(row)+b'\n')
    with patch('research.levels.precalc.plan.CATALOG_SHA256',hashlib.sha256(path.read_bytes()).hexdigest()):
        plan=make_plan(path,source_commit='1'*40)
        validate_plan(plan)
    return source,plan


class PrecalcTests(unittest.TestCase):
    def test_registry_covers_every_training_partition_and_target_is_explicit(self):
        with tempfile.TemporaryDirectory()as d:
            source,plan=catalog_fixture(Path(d));self.assertEqual(len(plan['cells']),6)
            self.assertEqual({c['scope']['split']for c in plan['cells']},set(SPLITS))
            for cell in plan['cells']:
                raw,scope=fit_csv(source,cell['scope']);rows=list(csv.reader(io.StringIO(raw.decode())))
                self.assertEqual(rows[0],['c0','c1']);self.assertEqual(len(rows)-1,scope['rows'])
                for x,y in rows[1:]:self.assertEqual(float(y),2*float(x))
            physical=physical_id('a'*16)
            outer=[set(training_rows(100,physical,split,20261023))for split in SPLITS[1:]]
            for i in range(100):self.assertEqual(sum(i in rows for rows in outer),4)

    def test_outer_heldout_values_cannot_change_its_landmark_input(self):
        with tempfile.TemporaryDirectory()as d:
            source,plan=catalog_fixture(Path(d));cell=plan['cells'][2];before,scope=fit_csv(source,cell['scope'])
            selected=set(training_rows(100,scope['physical_table_id'],cell['scope']['split'],20261023))
            text=source.read_text().splitlines();poisoned=[text[0]]+[row if i in selected else 'SECRET_TARGET,SECRET_FEATURE'for i,row in enumerate(text[1:])]
            source.write_text('\n'.join(poisoned)+'\n');changed=copy.deepcopy(cell['scope']);changed['train_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();changed['train_bytes']=source.stat().st_size
            after,other=fit_csv(source,changed);self.assertEqual(before,after);self.assertEqual(scope,other)
            self.assertNotIn(b'SECRET',after)

    def test_s3_writes_are_private_conditional_and_never_overwrite(self):
        class Error(RuntimeError):
            def __init__(self):self.response={'Error':{'Code':'PreconditionFailed'}}
        class Client:
            def __init__(self):self.objects={};self.calls=[]
            def put_object(self,**k):
                self.calls.append(k);key=k['Key']
                if key in self.objects:raise Error()
                self.objects[key]=k['Body']
            def get_object(self,**k):return dict(Body=io.BytesIO(self.objects[k['Key']]))
        c=Client();s=PrivateStore(c,'private','derived/cache');s.put('objects/one.json',b'{"x":1}');s.put('objects/one.json',b'{"x":1}')
        with self.assertRaises(ResearchError):s.put('objects/one.json',b'{"x":2}')
        for call in c.calls:
            self.assertEqual(call['IfNoneMatch'],'*');self.assertEqual(call['ServerSideEncryption'],'AES256')
            self.assertEqual(call['Metadata']['privacy'],'private-measurement')
        with self.assertRaises(ResearchError):s.put('../source.csv',b'{}')

    def test_complete_entry_reuses_all_member_hashes_and_detects_corruption(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source,plan=catalog_fixture(root);cell=plan['cells'][0];raw,scope=fit_csv(source,cell['scope'])
            who=dict(engine_sha256='b'*64,platform='fixture',source_sha256='c'*64)
            options=plan['options'];n=scope['rows'];report=dict(panel=options['panel'],options=options,kind='real',privacy='private',task='regression',rows=n,
                cells=[dict(id=i+1,state='complete',gain=.1,train_rows=[n-n//5-1]*5,query_rows=n)for i in range(23)])
            out=root/'artifacts';out.mkdir();report_raw=canonical(report)
            manifest=dict(format='dope-fit-cv-landmark-input-v1',feature_recipe=FEATURE_RECIPE,scope=scope,options=options,engine_sha256=who['engine_sha256'],report_sha256=hashlib.sha256(report_raw).hexdigest(),producer=who)
            for name,value in [('report.json',report),('manifest.json',manifest),('features.json',dict(feature_recipe=FEATURE_RECIPE,values=[.1]*23,mask=[True]*23))]:(out/name).write_bytes(canonical(value))
            receipt=dict(key=cache_key(cell,plan,who),state='complete',producer=who);store=LocalStore(root/'store');publish(cell,plan,receipt,out,store)
            reused=lookup(cell,plan,who,store);self.assertEqual(reused['state'],'cached');self.assertEqual(reused['valid_features'],23)
            checksum=hashlib.sha256(report_raw).hexdigest();(store.root/('objects/sha256/'+checksum+'.json')).write_bytes(b'{}')
            with self.assertRaisesRegex(ResearchError,'cached_object_digest'):lookup(cell,plan,who,store)

    def test_no_source_test_or_unsigned_url_access_and_empty_measurements_remain_explicit(self):
        with tempfile.TemporaryDirectory()as d:
            _,plan=catalog_fixture(Path(d));client=TrainURLSource({},'private',plan)
            with self.assertRaises(ResearchError):client.get_object(Bucket='private',Key='blobs/sha256/SECRET_TEST_KEY')
            with self.assertRaises(ResearchError):TrainURLSource({'blobs/sha256/TEST':'https://invalid'},'private',plan)
        options=dict(OPTIONS,work_budget=20000000000)
        report=dict(options=options,panel=options['panel'],kind='real',privacy='private',rows=80,task='regression',
                    cells=[dict(id=i+1,state='uninformative',gain=None)for i in range(23)])
        self.assertEqual(measured_features(report,80,'regression',options),([0.]*23,[False]*23))

    def test_bounded_high_budget_is_an_explicit_recipe_and_nonfinite_storage_is_refused(self):
        from research.levels.landmark_features import validate_options
        validate_options(dict(OPTIONS,work_budget=2000000000000))
        with self.assertRaises(ResearchError):validate_options(dict(OPTIONS,work_budget=999))
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):LocalStore(d).put('invalid.json',b'{"x":NaN}')

    def test_materialization_and_coverage_never_fetch_source_and_keep_pending_denominators(self):
        from research.levels.precalc.coverage import build
        from research.levels.precalc.store import materialize
        from research.levels.precalc.worker import index_key
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source,plan=catalog_fixture(root);cell=plan['cells'][0];_,scope=fit_csv(source,cell['scope'])
            who=dict(engine_sha256='b'*64,platform='fixture',source_sha256='c'*64)
            n=scope['rows'];report=dict(panel=OPTIONS['panel'],options=plan['options'],kind='real',privacy='private',task='regression',rows=n,
                cells=[dict(id=i+1,state='complete',gain=.2,train_rows=[n-n//5-1]*5,query_rows=n)for i in range(23)])
            run=root/'run';out=run/'cells'/'one';out.mkdir(parents=True)
            for name,value in [('report.json',report),('features.json',dict(feature_recipe=FEATURE_RECIPE,values=[.2]*23,mask=[True]*23)),
                ('manifest.json',dict(format='dope-fit-cv-landmark-input-v1',feature_recipe=FEATURE_RECIPE,scope=scope,options=plan['options'],engine_sha256=who['engine_sha256'],report_sha256=digest(report),producer=who))]:
                (out/name).write_bytes(canonical(value))
            receipt=dict(cell_key=cell['key'],key=cache_key(cell,plan,who),state='complete',producer=who)
            (out/'receipt.json').write_bytes(canonical(receipt));(run/'ownership.json').write_bytes(canonical(dict(plan_sha256=plan['sha256'])));(run/'producer.json').write_bytes(canonical(who))
            store=LocalStore(root/'store');publish(cell,plan,receipt,out,store)
            key=index_key(cell,plan,receipt['key']);raw=store.get(key)
            entry=materialize(store,key,hashlib.sha256(raw).hexdigest(),root/'download')
            self.assertEqual(entry['manifest_sha256'],hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest())
            result=build(plan,[run],store);self.assertEqual(result['counts'],dict(pending=5,ready=1));self.assertEqual(result['scheduled'],6)
            self.assertFalse((root/'download'/'fit.csv').exists())
            # A new transport/host producer may reuse a separately pinned old producer;
            # both the raw-data API and native execution are forbidden in this proof.
            from research.levels.precalc.worker import run
            planned=copy.deepcopy(plan);planned['cells']=[cell]
            planned['reuse_entries']={cell['key']:dict(entry_key=key,entry_sha256=hashlib.sha256(raw).hexdigest())}
            planned.pop('sha256');planned['sha256']=digest(planned)
            binary=root/'binary';binary.write_bytes(b'not executable; must never run')
            class NoSource:
                def get_object(self,**kw):raise AssertionError('raw source read during reuse')
            allocation=dict(memory_available=100*2**30,disk_available=100*2**30,cpu_available=128,load=[0,0,0])
            with patch('research.levels.precalc.plan.CATALOG_SHA256',planned['catalog_sha256']),patch('research.levels.precalc.worker.os.nice'),patch('research.levels.precalc.worker.allocation_snapshot',return_value=allocation),patch('research.levels.precalc.worker.run_cell',side_effect=AssertionError('native execution during reuse')):
                reused=run(planned,binary=binary,root=root/'reused',source_client=NoSource(),bucket='private',store=store,workers=1,threads=1,timeout=1,wall_seconds=35)
            self.assertEqual(reused['records'][0]['state'],'cached');self.assertEqual(reused['terminal'],1)
            self.assertEqual(build(planned,[root/'reused'],store)['counts'],dict(ready=1))

