"""Training, surrogate gradients and frozen-base residual regressions."""
from dataclasses import replace
import tempfile
import json
import hashlib
import unittest
from pathlib import Path
import torch
from research.levels.contracts import TrainConfig,ResearchError
from research.levels.corpus import Corpus
from research.levels.losses import objective
from research.levels.model import SharedModel
from research.levels.packages import export_package,load_package,load_decoder
from research.levels.train import train
from research.levels.tests.test_models import tiny,fixture


def config():
    return TrainConfig(steps=4,accumulation=1,validate_every=2,checkpoint_every=2,cpu_threads=1,
                       wall_seconds=120,consistency_weight=0,profile_weight=0)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_prior_development_mechanisms_do_not_alias_training_families(self):
        corpus=Corpus(prior_only=True)
        train_families={corpus.get('train',i)['family_id'] for i in range(30)}
        dev_families={corpus.get('development',i)['family_id'] for i in range(30)}
        self.assertFalse(train_families.intersection(dev_families))
        self.assertNotIn('owned-prior:9',train_families|dev_families)

    def test_versioned_prior_stream_counts_physical_tables_and_keeps_development_fixed(self):
        from research.levels.contracts import digest
        legacy=Corpus(prior_only=True)
        unique=Corpus(prior_only=True,prior_recipe='owned-mechanisms-unique-v2')
        self.assertEqual(legacy.identity,digest(dict(manifest=None,prior_recipe='owned-mechanisms-v1',
                                                   prior_only=True,seed=20261023)))
        self.assertNotEqual(legacy.identity,unique.identity)
        old=[legacy.get('train',i) for i in range(28)]
        new=[unique.get('train',i) for i in range(28)]
        self.assertEqual(len({r['source_provenance']['physical_table_id'] for r in old}),14)
        self.assertEqual(len({r['source_provenance']['physical_table_id'] for r in new}),28)
        for i in range(14):self.assertEqual(old[2*i]['row_ids'],old[2*i+1]['row_ids'])
        self.assertEqual({r['family_id'] for r in new},{f'owned-prior:{i}' for i in range(7)})
        for family in range(7):self.assertEqual(sum(r['family_id']==f'owned-prior:{family}' for r in new),4)
        for i in range(12):
            a,b=legacy.get('development',i),unique.get('development',i)
            self.assertEqual(a['row_ids'],b['row_ids'])
            self.assertEqual(a['rows'],b['rows'])
            self.assertFalse(set(a['row_ids']).intersection(r for source in new for r in source['row_ids']))
        # Mixed real/prior mode already consumes prior draws at even cursors; its stream stays unchanged.
        mixed=Corpus(dict(format='dope-meta-corpus-v1',datasets=[dict(family_id=f'owned:{i}',dataset_id=str(i),
            meta_role=role,train_sha256=str(i)*64,source_role='official_train',rights=dict(admitted=True,
            license_id='CC0-1.0'),target_index=1,split_policy='iid') for i,role in [(1,'train'),(2,'development')]]),
            prior_recipe='owned-mechanisms-unique-v2')
        for i in range(0,28,2):self.assertEqual(mixed.get('train',i)['row_ids'],legacy.get('train',i)['row_ids'])
        with self.assertRaisesRegex(ResearchError,'unknown_prior_recipe'):Corpus(prior_only=True,prior_recipe='unknown')
        for cursor in (-1,True,1.5):
            with self.assertRaisesRegex(ResearchError,'invalid_corpus_cursor'):unique.get('train',cursor)
        with self.assertRaisesRegex(ResearchError,'final_access_forbidden'):unique.get('final',0)

    def test_unique_prior_resume_keeps_provenance_and_refuses_cross_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=replace(config(),transfer_weight=0,driver_weight=0)
            def corpus():return Corpus(prior_only=True,prior_recipe='owned-mechanisms-unique-v2')
            whole,full=train(tiny(),cfg,corpus(),root/'whole',source_sha256='a'*64)
            _,part=train(tiny(),cfg,corpus(),root/'part',source_sha256='a'*64,stop_after=2)
            resumed,terminal=train(tiny(),cfg,corpus(),root/'resumed',source_sha256='a'*64,resume=part['last_checkpoint'])
            self.assertEqual(terminal['optimizer_updates'],4)
            for name,value in whole.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[name]),name)
            def rows(name):return [json.loads(line) for line in (root/name/'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(rows('whole'),rows('part')+rows('resumed'))
            self.assertEqual(len({r['source_provenance']['physical_table_id'] for r in rows('whole')}),4)
            with self.assertRaisesRegex(ResearchError,'resume_recipe_or_runtime_mismatch'):
                train(tiny(),cfg,Corpus(prior_only=True),root/'wrong',source_sha256='a'*64,resume=part['last_checkpoint'])
            with self.assertRaisesRegex(ResearchError,'resume_recipe_or_runtime_mismatch'):
                train(tiny(),replace(cfg,neural_unrolls=48),corpus(),root/'wrong-unroll',
                      source_sha256='a'*64,resume=part['last_checkpoint'])

    def test_transfer_and_driver_surrogates_have_decoder_and_encoder_gradients(self):
        model=SharedModel(tiny());context,query,_=fixture()
        loss,raw=objective(model,context,query,([0.]*23,[False]*23),config(),101)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None),0)
        self.assertEqual(set(raw),{'low','medium','high'})

    def test_bounded_neural_unrolls_keep_default_replay_and_long_gradient_path(self):
        from research.levels.contracts import digest
        context,query,_=fixture();model=SharedModel(tiny())
        def measured(cfg):
            torch.manual_seed(19283)
            return objective(model,context,query,([0.]*23,[False]*23),cfg,101)
        default,records=measured(config())
        explicit,other=measured(replace(config(),neural_unrolls=3))
        self.assertTrue(torch.equal(default,explicit));self.assertEqual(records,other)
        longer=replace(config(),neural_unrolls=48)
        value,record=measured(longer);value.backward()
        self.assertFalse(torch.equal(default,value))
        self.assertTrue(all(v['neural_unrolls']==48 for v in record.values()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        for part in (model.encoder,model.decoder):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None),0)
        self.assertNotEqual(digest(vars(config())),digest(vars(longer)))
        for bad in (0,65,True,3.5):
            with self.assertRaises(ResearchError):replace(config(),neural_unrolls=bad).validate()

    def test_long_neural_fit_differentiates_through_synthetic_rows_and_targets(self):
        from research.levels.proxies import features,labels,neural_fit,loss
        from research.levels.model import tensors
        context,query,_=fixture();cv,co=tensors(context,'cpu');qv,qo=tensors(query,'cpu')
        xs,_=features(cv,co,context.schema);xs.requires_grad_()
        ys=labels(cv,context.schema).clone().requires_grad_()
        xq,_=features(qv,qo,context.schema);yq=labels(qv,context.schema)
        a,b=neural_fit(xs,ys,context.schema,differentiable=True,iterations=48)
        value=loss(torch.tanh(xq@a)@b,yq,context.schema)
        gradients=torch.autograd.grad(value,(xs,ys))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()),0)

    def test_ultra_refuses_random_base_then_freezes_all_lower_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=replace(config(),transfer_weight=0,driver_weight=0)
            with self.assertRaisesRegex(ResearchError,'ultra_requires'):
                train(tiny(),cfg,Corpus(prior_only=True),root/'bad',source_sha256='a'*64,phase='ultra-residual')
            ordinary,part=train(tiny(),cfg,Corpus(prior_only=True),root/'ordinary',source_sha256='a'*64,stop_after=1)
            latest=json.loads((root/'ordinary/checkpoints/latest.json').read_bytes())
            checkpoint=root/'ordinary/checkpoints'/latest['checkpoint']
            self.assertEqual(checkpoint.name,'step-000000001')
            self.assertTrue(checkpoint.is_dir())
            self.assertEqual(hashlib.sha256((checkpoint/'manifest.json').read_bytes()).hexdigest(),latest['manifest_sha256'])
            residual,result=train(tiny(),cfg,Corpus(prior_only=True),root/'residual',source_sha256='a'*64,
                                  phase='ultra-residual',base_checkpoint=part['last_checkpoint'],stop_after=1)
            changed=[]
            for name,value in ordinary.state_dict().items():
                if not torch.equal(value,residual.state_dict()[name]):changed.append(name)
            self.assertTrue(changed)
            self.assertTrue(all(any(name.startswith(prefix) for prefix in
                ('encoder.distribution_heads.3.','encoder.task_heads.3.','decoder.conditions.3.','profiles.ultra.'))
                                for name in changed),changed)
            self.assertEqual(result['optimizer_updates'],1)
            export_package(result['last_checkpoint'],root/'model')
            loaded,_=load_package(root/'model')
            for name,value in residual.encoder.state_dict().items():
                self.assertTrue(torch.equal(value,loaded.encoder.state_dict()[name]))
            (root/'model/encoder.safetensors').unlink()
            decoder,_=load_decoder(root/'model')
            self.assertEqual(set(decoder.state_dict()),set(residual.decoder.state_dict()))

    def test_resource_pause_discards_partial_update_and_resume_replays_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=replace(config(),accumulation=2,transfer_weight=0,driver_weight=0)
            checks=[]
            def guard():checks.append(True);return len(checks)<3
            _,part=train(tiny(),cfg,Corpus(prior_only=True),root/'part',source_sha256='a'*64,resource_check=guard)
            self.assertEqual(part['state'],'resource_pause');self.assertEqual(part['optimizer_updates'],0)
            resumed,_=train(tiny(),cfg,Corpus(prior_only=True),root/'resumed',source_sha256='a'*64,
                            resume=part['last_checkpoint'],stop_after=1)
            whole,_=train(tiny(),cfg,Corpus(prior_only=True),root/'whole',source_sha256='a'*64,stop_after=1)
            for name,value in whole.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[name]),name)
