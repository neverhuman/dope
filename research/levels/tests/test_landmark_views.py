"""Private corpus-wide feature views preserve every source disposition and byte."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research.levels.contracts import canonical,digest,ResearchError
from research.levels.landmark_features import FEATURE_RECIPE,OBSERVED_RECIPE,read_input
from research.levels.precalc.coverage import build
from research.levels.precalc.plan import fit_csv
from research.levels.precalc.store import LocalStore
from research.levels.precalc.views import derive_corpus,materialize_view
from research.levels.precalc.worker import publish,cache_key
from research.levels.tests.test_precalc import catalog_fixture


class LandmarkViewTests(unittest.TestCase):
    def test_full_derived_view_is_immutable_complete_and_never_reads_source_or_fits(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source,plan=catalog_fixture(root);cell=plan['cells'][0];csv,scope=fit_csv(source,cell['scope'])
            who=dict(engine_sha256='b'*64,platform='fixture',source_sha256='c'*64)
            n=scope['rows'];report=dict(panel=plan['options']['panel'],options=plan['options'],kind='real',privacy='private',rows=n,task='regression',
                cells=[dict(id=i,state='uninformative',gain=-.2 if i%2 else 0.,train_rows=[n-n//5-1]*5,query_rows=n)for i in range(1,24)])
            report['cells'][-1].update(state='unsupported',gain=None)
            manifest=dict(format='dope-fit-cv-landmark-input-v1',feature_recipe=FEATURE_RECIPE,scope=scope,
                options=plan['options'],engine_sha256=who['engine_sha256'],report_sha256=digest(report),producer=who)
            run=root/'run';out=run/'cells'/'one';out.mkdir(parents=True)
            for name,value in [('report.json',report),('manifest.json',manifest),
                               ('features.json',dict(feature_recipe=FEATURE_RECIPE,values=[0.]*23,mask=[False]*23))]:
                (out/name).write_bytes(canonical(value))
            receipt=dict(cell_key=cell['key'],key=cache_key(cell,plan,who),state='complete',producer=who)
            for path,value in [(out/'receipt.json',receipt),(run/'ownership.json',dict(plan_sha256=plan['sha256'])),
                               (run/'producer.json',who)]:path.write_bytes(canonical(value))
            class DerivedOnly(LocalStore):
                def get(self,key):
                    if key.startswith('blobs/'):raise AssertionError('raw source access')
                    return super().get(key)
            store=DerivedOnly(root/'store');publish(cell,plan,receipt,out,store);indexed=build(plan,[run],store)
            original={str(p.relative_to(store.root)):p.read_bytes()for p in store.root.rglob('*.json')}
            with patch('subprocess.run',side_effect=AssertionError('no fit process')):
                result=derive_corpus(store,indexed['index_key'],indexed['index_sha256'],workers=2)
                self.assertEqual(result,derive_corpus(store,indexed['index_key'],indexed['index_sha256'],workers=1))
            self.assertEqual(result['counts'],dict(pending=5,ready=1));self.assertEqual(result['scheduled'],6)
            index=json.loads(store.get(result['index_key']));self.assertFalse(index['training_admitted'])
            self.assertEqual(index['new_native_executions'],0);self.assertEqual(index['new_optimizer_updates'],0)
            ready=next(r for r in index['records']if r['state']=='ready');self.assertEqual(ready['valid_features'],22)
            entry=materialize_view(store,ready['entry_key'],ready['entry_sha256'],root/'download')
            fit=root/'fit.csv';fit.write_bytes(csv)
            values,mask=read_input(root/'download',entry['manifest_sha256'],who['engine_sha256'],
                                  csv_path=fit,kinds=scope['column_kinds'],task='regression',recipe=OBSERVED_RECIPE)
            self.assertEqual(values[:2],[-.2,0.]);self.assertEqual(mask[:2],[True,True]);self.assertFalse(mask[-1])
            for name,raw in original.items():self.assertEqual((store.root/name).read_bytes(),raw)
            with self.assertRaisesRegex(ResearchError,'source_index_digest'):
                derive_corpus(store,indexed['index_key'],'a'*64)
            with self.assertRaises(ResearchError):materialize_view(store,ready['entry_key'],'a'*64,root/'bad')
            # Corruption is a refused derivation, never an unavailable/omitted cell.
            (store.root/('objects/sha256/'+digest(report)+'.json')).write_bytes(b'{}')
            with self.assertRaisesRegex(ResearchError,'source_member_digest'):
                derive_corpus(store,indexed['index_key'],indexed['index_sha256'])
