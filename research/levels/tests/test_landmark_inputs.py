"""Training-only CV input binding, influence, historical isolation and exact replay."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from research.levels.contracts import LANDMARK_INPUT_GRAPH, LANDMARK_MASKED_GRAPH, canonical, digest, ResearchError
from research.levels.corpus import Corpus
from research.levels.landmark_features import (FEATURE_RECIPE, OPTIONS, feature_tensor, training_csv,
                                             report_features, read_input, InputCache)
from research.levels.losses import objective
from research.levels.model import SharedModel,tensors
from research.levels.packages import encode_table,export_package,export_encoding,load_decoder
from research.levels.tests.test_models import fixture
from research.levels.tests.test_v2 import graph_config
from research.levels.tests.test_training import config
from research.levels.train import train


def inputs():return ([.1*(i+1) for i in range(23)], [True]*23)


def write_cache(root, data):
    """Synthetic report fixture for binding tests; never native-quality evidence."""
    raw,scope=training_csv(data);(root/'fit.csv').write_bytes(raw)
    n=scope['rows'];report=dict(panel=OPTIONS['panel'],options=OPTIONS,kind='real',privacy='private',
        task=scope['task'],rows=n,cells=[dict(id=i+1,state='complete',gain=.1*(i+1),
        train_rows=[n-n//5-1]*5,query_rows=n)for i in range(23)])
    (root/'report.json').write_bytes(canonical(report))
    manifest=dict(format='dope-fit-cv-landmark-input-v1',feature_recipe=FEATURE_RECIPE,scope=scope,
        options=OPTIONS,engine_sha256='a'*64,report_sha256=hashlib.sha256(canonical(report)).hexdigest(),producer={})
    (root/'manifest.json').write_bytes(canonical(manifest))
    return dict(physical_table_id=scope['physical_table_id'],directory=str(root),
                manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),engine_sha256='a'*64)


class LandmarkInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_features_condition_only_task_blocks_and_keep_exact_prefixes(self):
        m=SharedModel(graph_config(LANDMARK_INPUT_GRAPH));c,_,_=fixture();v,o=tensors(c,'cpu');f=inputs()
        with torch.no_grad():
            a=m.encoder(v,o,c.schema,through='ultra',landmark_features=f)
            g=([x+2 for x in f[0]],f[1]);b=m.encoder(v,o,c.schema,through='ultra',landmark_features=g)
            self.assertTrue(torch.equal(a[:1024],m.encoder(v,o,c.schema,through='low',landmark_features=f)))
            offset=0
            for width in (1024,3072,4096,2048):
                self.assertTrue(torch.equal(a[offset:offset+width//2],b[offset:offset+width//2]))
                self.assertFalse(torch.equal(a[offset+width//2:offset+width],b[offset+width//2:offset+width]))
                offset+=width
            x=m.encoder(v,o,c.schema,target_included=False,through='ultra')
            y=m.encoder(v,o,c.schema,target_included=False,through='ultra',landmark_features=g)
            self.assertTrue(torch.equal(x,y))
        with self.assertRaisesRegex(ResearchError,'cv_landmark_inputs_required'):m.encoder(v,o,c.schema)

    def test_control_shapes_init_equal_and_masked_graph_ignores_values(self):
        torch.manual_seed(42);m=SharedModel(graph_config(LANDMARK_INPUT_GRAPH))
        torch.manual_seed(42);control=SharedModel(graph_config(LANDMARK_MASKED_GRAPH))
        for key,value in m.state_dict().items():self.assertTrue(torch.equal(value,control.state_dict()[key]),key)
        c,_,_=fixture();v,o=tensors(c,'cpu')
        with torch.no_grad():
            a=control.encoder(v,o,c.schema,landmark_features=inputs())
            b=control.encoder(v,o,c.schema,landmark_features=([0.]*23,[True]*23))
            self.assertTrue(torch.equal(a,b))
        old=SharedModel(graph_config())
        with self.assertRaisesRegex(ResearchError,'landmark_input_graph_required'):
            old.encoder(v,o,c.schema,landmark_features=inputs())

    def test_missing_nonfinite_and_empty_refuse_and_real_zero_is_present(self):
        for f in [None,([0.]*23,[False]*23),([float('nan')]+[0.]*22,[True]*23),
                  ([1.]+[0.]*22,[False]+[True]*22),([0.]*23,[1]*23)]:
            with self.assertRaises(ResearchError):feature_tensor(f,'cpu')
        zero=feature_tensor(([0.]*23,[True]*23),'cpu')
        self.assertTrue(torch.equal(zero[:23],torch.zeros(23)));self.assertEqual(int(zero[23:].sum()),23)

    def test_heldout_values_never_enter_training_csv_or_feature_scope(self):
        data=Corpus(prior_only=True).get('train',0);raw,scope=training_csv(data);other=copy.deepcopy(data)
        for role in ('tune','calibration','confirmation'):
            for i in other['roles'][role]:other['rows'][i]=['SECRET']*len(other['kinds'])
        self.assertEqual((raw,scope),training_csv(other));self.assertNotIn(b'SECRET',raw)
        other['roles']['fit'].append(other['roles']['tune'][0])
        with self.assertRaisesRegex(ResearchError,'training_role_overlap'):training_csv(other)
        for policy in ('group','time'):
            other=copy.deepcopy(data);other['split_policy']=policy
            with self.assertRaisesRegex(ResearchError,'group_time_not_qualified'):training_csv(other)

    def test_cache_binds_recipe_rows_reports_and_out_of_fold_counts(self):
        data=Corpus(prior_only=True).get('train',0)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);pin=write_cache(root,data)
            cache=InputCache(dict(format='dope-fit-cv-landmark-cache-v1',feature_recipe=FEATURE_RECIPE,entries=[pin]))
            attached=cache.attach(data);self.assertEqual(attached['landmark_input'],inputs())
            changed=copy.deepcopy(data);changed['rows'][data['roles']['fit'][0]][0]=9876
            with self.assertRaisesRegex(ResearchError,'physical_scope_mismatch'):cache.attach(changed)
            with self.assertRaisesRegex(ResearchError,'manifest_digest'):read_input(root,'b'*64,'a'*64,data=data)
            report=json.loads((root/'report.json').read_bytes());report['cells'][0]['query_rows']=0
            with self.assertRaisesRegex(ResearchError,'out_of_fold'):report_features(report,len(data['roles']['fit']),data['task'])
            (root/'report.json').write_bytes(canonical(report))
            with self.assertRaisesRegex(ResearchError,'input_binding'):cache.attach(data)

    def test_training_uses_inputs_not_auxiliary_targets_and_resume_binds_cache(self):
        class FixtureCorpus(Corpus):
            # Public synthetic feature fixture; no claim that a native panel ran in this unit test.
            def __init__(self):
                super().__init__(prior_only=True)
                class FixtureCache:
                    feature_recipe=FEATURE_RECIPE
                    def attach(self,data):return data
                self.input_cache=FixtureCache()
            def get(self,*a):
                d=super().get(*a);d['landmark_input']=inputs();return d
        cfg=replace(config(),steps=2,validate_every=1,checkpoint_every=1,transfer_weight=0,driver_weight=0)
        model_cfg=graph_config(LANDMARK_INPUT_GRAPH)
        with tempfile.TemporaryDirectory()as directory:
            r=Path(directory);corpus=FixtureCorpus()
            whole,terminal=train(model_cfg,cfg,corpus,r/'whole',source_sha256='a'*64)
            _,part=train(model_cfg,cfg,corpus,r/'part',source_sha256='a'*64,stop_after=1)
            resumed,_=train(model_cfg,cfg,corpus,r/'resumed',source_sha256='a'*64,resume=part['last_checkpoint'])
            for key,value in whole.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[key]),key)
            changed=FixtureCorpus();changed.identity='b'*64
            with self.assertRaises(ResearchError):
                train(model_cfg,cfg,changed,r/'changed-cache',source_sha256='a'*64,resume=part['last_checkpoint'])
            manifest=export_package(terminal['last_checkpoint'],r/'model');c,q,_=fixture()
            enc=encode_table(whole,manifest,c,landmark_features=inputs());raw=export_encoding(enc,'low')
            self.assertNotIn(b'landmark_input',raw.bytes);self.assertNotIn(b'train_rows',raw.bytes)
            (r/'model/encoder.safetensors').unlink();load_decoder(r/'model')
            with self.assertRaisesRegex(ResearchError,'cv_landmark_input_cache'):
                train(model_cfg,cfg,Corpus(prior_only=True),r/'unlabeled',source_sha256='a'*64)
            whole.zero_grad(set_to_none=True)
            loss,raw_objective=objective(whole,c,q,inputs(),cfg,0)
            loss.backward()
            self.assertGreater(float(whole.encoder.landmark_projection.weight.grad.abs().sum()),0.)
            self.assertEqual(raw_objective['low']['landmark_input_slots'],23)
            self.assertEqual(raw_objective['low']['profile'],0.)
            with self.assertRaisesRegex(ResearchError,'not_profile_targets'):
                objective(whole,c,q,inputs(),replace(cfg,profile_weight=.1),0)

    def test_query_evidence_matches_declared_native_caps_inside_training_rows(self):
        from research.levels.landmark_features import expected_query_rows
        n=50001
        report=dict(panel=OPTIONS['panel'],options=OPTIONS,kind='real',privacy='private',task='regression',rows=n,
            cells=[dict(id=i,state='complete',gain=.1,train_rows=[1000]*5,
                        query_rows=40960 if i<=15 or i==22 else 4100)for i in range(1,24)])
        self.assertEqual(report_features(report,n,'regression'),([.1]*23,[True]*23))
        for coordinate in (1,16,22,23):
            self.assertEqual(expected_query_rows(coordinate,103,OPTIONS),103)
        for coordinate in (1,16,22,23):
            bad=copy.deepcopy(report);bad['cells'][coordinate-1]['query_rows']=n
            with self.assertRaisesRegex(ResearchError,'out_of_fold'):report_features(bad,n,'regression')
        bad=copy.deepcopy(report);bad['cells'][0]['train_rows']=[n]*5
        with self.assertRaisesRegex(ResearchError,'out_of_fold'):report_features(bad,n,'regression')
