"""Measured weak/negative CV scores stay present only under the explicit successor."""
import copy
from dataclasses import replace
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from research.levels.contracts import (ResearchError,canonical,LANDMARK_INPUT_GRAPH,
    LANDMARK_OBSERVED_GRAPH,LANDMARK_OBSERVED_MASKED_GRAPH)
from research.levels.corpus import Corpus
from research.levels.landmark_features import (FEATURE_RECIPE,OBSERVED_RECIPE,OPTIONS,
    InputCache,derive_observed,read_input,report_features,feature_tensor,prepare_input)
from research.levels.tests.test_landmark_inputs import write_cache
from research.levels.tests.test_v2 import graph_config
from research.levels.tests.test_models import fixture
from research.levels.tests.test_training import config
from research.levels.model import SharedModel,tensors
from research.levels.train import train


def report():
    return dict(panel=OPTIONS['panel'],options=OPTIONS,kind='real',privacy='private',rows=100,
        task='classification',cells=[dict(id=i,state='complete',gain=.2,query_rows=100,train_rows=[80]*5)
                                    for i in range(1,24)])


class ObservedLandmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_explicit_recipe_keeps_negative_zero_and_weak_scores_without_reinterpreting_v1(self):
        r=report()
        for i,gain in enumerate((-.5,0.,.005,None)):
            r['cells'][i].update(state='uninformative',gain=gain)
        r['cells'][4].update(state='failed',gain=.8)
        historical=report_features(r,100,'classification')
        values,mask=report_features(r,100,'classification',recipe=OBSERVED_RECIPE)
        self.assertEqual(historical[0][:5],[0.]*5);self.assertEqual(historical[1][:5],[False]*5)
        self.assertEqual(values[:5],[-.5,0.,.005,0.,0.]);self.assertEqual(mask[:5],[True,True,True,False,False])
        x=feature_tensor((values,mask),'cpu')
        self.assertAlmostEqual(float(x[0]),-1/3,places=6);self.assertEqual(float(x[1]),0.)
        self.assertEqual(x[23:28].tolist(),[1.,1.,1.,0.,0.])
        self.assertEqual(report_features(r,100,'classification'),historical)

    def test_observed_scores_still_require_finite_training_only_five_fold_witnesses(self):
        r=report();r['cells'][0].update(state='uninformative',gain=-1.)
        for field,value in [('gain',float('nan')),('gain',float('inf')),('gain',1e100),
                            ('gain',True),('query_rows',0),('query_rows',101),('train_rows',[True]*5),
                            ('train_rows',[100]*5),('state','invented')]:
            changed=copy.deepcopy(r);changed['cells'][0][field]=value
            with self.assertRaises(ResearchError):report_features(changed,100,'classification',recipe=OBSERVED_RECIPE)
        for c in r['cells']:c.update(state='uninformative',gain=0.)
        self.assertEqual(report_features(r,100,'classification',recipe=OBSERVED_RECIPE),([0.]*23,[True]*23))
        with self.assertRaisesRegex(ResearchError,'shape_or_empty'):report_features(r,100,'classification')
        for c in r['cells']:c['gain']=None
        with self.assertRaisesRegex(ResearchError,'shape_or_empty'):
            report_features(r,100,'classification',recipe=OBSERVED_RECIPE)

    def test_additive_derivation_never_fits_or_opens_source_and_binds_the_exact_saved_report(self):
        data=Corpus(prior_only=True).get('train',0)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'measurement';source.mkdir();pin=write_cache(source,data)
            r=json.loads((source/'report.json').read_bytes());r['cells'][0].update(state='uninformative',gain=-.25)
            (source/'report.json').write_bytes(canonical(r));m=json.loads((source/'manifest.json').read_bytes())
            m['report_sha256']=hashlib.sha256(canonical(r)).hexdigest();(source/'manifest.json').write_bytes(canonical(m))
            original={p.name:p.read_bytes()for p in source.iterdir()};sha=hashlib.sha256(canonical(m)).hexdigest()
            with patch('subprocess.run',side_effect=AssertionError('no fits in derivation')):
                out=derive_observed(source,sha,'a'*64,root/'observed')
            self.assertEqual({p.name:p.read_bytes()for p in source.iterdir()},original)
            self.assertEqual((root/'observed/report.json').read_bytes(),original['report.json'])
            features=read_input(root/'observed',out['manifest_sha256'],'a'*64,data=data,recipe=OBSERVED_RECIPE)
            self.assertEqual(features[0][0],-.25);self.assertTrue(features[1][0])
            cache=InputCache(dict(format='dope-fit-cv-landmark-cache-v1',feature_recipe=OBSERVED_RECIPE,entries=[out]))
            self.assertEqual(cache.attach(data)['landmark_input'],features)
            with self.assertRaises(ResearchError):read_input(root/'observed',out['manifest_sha256'],'a'*64,data=data)
            with self.assertRaises(ResearchError):derive_observed(source,'b'*64,'a'*64,root/'bad')
            self.assertFalse((root/'bad').exists())
            derived=json.loads((root/'observed/manifest.json').read_bytes())['producer']['derivation']
            self.assertEqual(derived['source_manifest_sha256'],sha);self.assertEqual(derived['new_native_executions'],0)

    def test_new_graphs_have_identical_capacity_and_initialization_and_isolate_distribution(self):
        torch.manual_seed(17);a=SharedModel(graph_config(LANDMARK_OBSERVED_GRAPH))
        torch.manual_seed(17);b=SharedModel(graph_config(LANDMARK_OBSERVED_MASKED_GRAPH))
        self.assertEqual(a.config.landmark_recipe(),OBSERVED_RECIPE)
        self.assertEqual(graph_config(LANDMARK_INPUT_GRAPH).landmark_recipe(),FEATURE_RECIPE)
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        c,_,_=fixture();v,o=tensors(c,'cpu');negative=([-.5]+[0.]*22,[True]*23);zero=([0.]*23,[True]*23)
        with torch.no_grad():
            code=a.encoder(v,o,c.schema,through='ultra',landmark_features=negative)
            other=a.encoder(v,o,c.schema,through='ultra',landmark_features=zero)
            self.assertFalse(torch.equal(code,other));offset=0
            for width in (1024,3072,4096,2048):
                self.assertTrue(torch.equal(code[offset:offset+width//2],other[offset:offset+width//2]));offset+=width
            self.assertTrue(torch.equal(b.encoder(v,o,c.schema,landmark_features=negative),b.encoder(v,o,c.schema,landmark_features=zero)))
            self.assertTrue(torch.equal(a.encoder(v,o,c.schema,target_included=False,landmark_features=negative),
                                        a.encoder(v,o,c.schema,target_included=False,landmark_features=zero)))

    def test_training_rejects_cross_recipe_cache_and_checkpoint(self):
        class FixtureCorpus(Corpus):
            def __init__(self,recipe):
                super().__init__(prior_only=True)
                class Cache:
                    feature_recipe=recipe
                    def attach(self,data):return data
                self.input_cache=Cache()
            def get(self,*args):
                data=super().get(*args);data['landmark_input']=([-.1]+[0.]*22,[True]*23);return data
        cfg=replace(config(),steps=2,validate_every=1,checkpoint_every=1,transfer_weight=0,driver_weight=0)
        model=graph_config(LANDMARK_OBSERVED_GRAPH)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ResearchError,'model_cache_recipe'):
                train(model,cfg,FixtureCorpus(FEATURE_RECIPE),root/'wrong',source_sha256='a'*64)
            a,whole=train(model,cfg,FixtureCorpus(OBSERVED_RECIPE),root/'whole',source_sha256='a'*64)
            _,part=train(model,cfg,FixtureCorpus(OBSERVED_RECIPE),root/'part',source_sha256='a'*64,stop_after=1)
            b,_=train(model,cfg,FixtureCorpus(OBSERVED_RECIPE),root/'resumed',source_sha256='a'*64,resume=part['last_checkpoint'])
            for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
            with self.assertRaises(ResearchError):
                train(graph_config(LANDMARK_INPUT_GRAPH),cfg,FixtureCorpus(FEATURE_RECIPE),root/'cross',
                      source_sha256='a'*64,resume=part['last_checkpoint'])

    def test_fresh_owned_producer_explicitly_selects_observed_recipe_and_keeps_v1_default(self):
        from types import SimpleNamespace
        data=Corpus(prior_only=True).get('train',0)
        n=len(data['roles']['fit'])
        def native(command,*,stdout,**kwargs):
            # Inert producer fixture; no scientific/native measurement claim.
            options=dict(OPTIONS,work_budget=20000000000)
            value=dict(panel=OPTIONS['panel'],options=options,kind='real',privacy='private',
                rows=n,task=data['task'],cells=[dict(id=i,state='uninformative',gain=-.2 if i%2 else 0.,
                    query_rows=n,train_rows=[n-n//5-1]*5)for i in range(1,24)])
            stdout.write(canonical(value));return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);binary=root/'inert-native';binary.write_bytes(b'not executable')
            expected=hashlib.sha256(binary.read_bytes()).hexdigest()
            with patch('research.levels.landmark_features.subprocess.run',side_effect=native),\
                    patch('research.levels.landmark_features.platform.platform',return_value='synthetic-unit-platform'):
                with self.assertRaisesRegex(ResearchError,'shape_or_empty'):
                    prepare_input(data,binary,expected,root/'v1')
                self.assertEqual(json.loads((root/'v1/plan.json').read_bytes())['feature_recipe'],FEATURE_RECIPE)
                self.assertTrue((root/'v1/failure.json').is_file());self.assertFalse((root/'v1/manifest.json').exists())
                pin=prepare_input(data,binary,expected,root/'v2',recipe=OBSERVED_RECIPE)
            manifest=json.loads((root/'v2/manifest.json').read_bytes())
            self.assertEqual(manifest['format'],'dope-fit-cv-landmark-input-v2')
            self.assertEqual(manifest['feature_recipe'],OBSERVED_RECIPE)
            values,mask=read_input(root/'v2',pin['manifest_sha256'],expected,data=data,recipe=OBSERVED_RECIPE)
            self.assertEqual(values[:2],[-.2,0.]);self.assertEqual(mask,[True]*23)
