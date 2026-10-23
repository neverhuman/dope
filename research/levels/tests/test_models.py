from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from research.levels.contracts import ModelConfig, TrainConfig, ResearchError, LEVELS
from research.levels.corpus import Corpus
from research.levels.data import episode, fit_transform, role_indices, transform
from research.levels.model import SharedModel,tensors
from research.levels.packages import LocalEncoding,export_encoding,read_code
from research.levels.priors import table
from research.levels.train import train


def tiny():
    return ModelConfig(width=8,heads=2,feature_blocks=1,inducing_blocks=1,inducing_slots=2,
                       summary_slots=2,target_blocks=1,decoder_blocks=1,condition_slots=1,
                       max_context=16,max_query=8,max_features=8,max_categories=8,cell_cap=128)


def fixture(config=None):
    config=tiny() if config is None else config
    source=table(0,123)
    return episode(**{k:source[k] for k in ('rows','kinds','slots','row_ids','task','roles')},
                   config=config,cursor=0,seed=12)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_categorical_sampler_gradually_reveals_and_never_redraws_visible_cells(self):
        from research.levels.decoder import sample
        class ProbeDecoder:
            config=tiny()
            def __init__(self):self.inputs=[]
            def __call__(self,code,schema,values,observed,time):
                self.inputs.append((values.clone(),observed.clone()))
                logits=torch.full((*values.shape,4),-1e9)
                logits[:,:,2 if float(time)>.5 else 3]=100
                return torch.zeros_like(values),logits,torch.full_like(values,-100)
        schema=dict(columns=[dict(kind='numeric'),dict(kind='categorical')],target_index=1)
        probe=ProbeDecoder()
        values,observed=sample(probe,torch.zeros(256),schema,200,7,steps=8)
        self.assertEqual(set(values[:,1].tolist()),{2.,3.})
        self.assertTrue(observed[:,1].all())
        self.assertTrue((probe.inputs[0][0][:,1]==0).all())
        self.assertTrue((probe.inputs[0][1][:,1]==False).all())
        for (previous,_),(current,present) in zip(probe.inputs,probe.inputs[1:]):
            revealed=previous[:,1]!=0
            self.assertTrue(torch.equal(previous[revealed,1],current[revealed,1]))
            self.assertTrue((present[:,1]==(current[:,1]!=0)).all())
        replay,_=sample(ProbeDecoder(),torch.zeros(256),schema,200,7,steps=8)
        self.assertTrue(torch.equal(values,replay))
        redraw,_=sample(ProbeDecoder(),torch.zeros(256),schema,200,7,steps=8,categorical_schedule='redraw-v1')
        self.assertTrue((redraw[:,1]==3).all())
        with self.assertRaises(ResearchError):
            sample(ProbeDecoder(),torch.zeros(256),schema,2,7,steps=8,categorical_schedule='unknown')

    def test_holdout_mutations_do_not_change_fit_transforms_or_context(self):
        source=table(0,123)
        args={k:source[k] for k in ('rows','kinds','slots','row_ids','task','roles')}
        first=episode(**args,config=tiny(),cursor=0,seed=12)
        for i in source['roles']['confirmation']+source['roles']['calibration']:
            source['rows'][i]=[91827]*len(source['kinds'])
        second=episode(**args,config=tiny(),cursor=0,seed=12)
        np.testing.assert_array_equal(first[0].values,second[0].values)
        self.assertEqual(first[0].schema,second[0].schema)
        self.assertEqual(first[2],second[2])

    def test_target_free_distribution_prefix_and_row_permutation(self):
        model=SharedModel(tiny()).eval();context,_,_=fixture()
        values,observed=tensors(context,'cpu')
        with torch.no_grad():
            high=model.encoder(values,observed,context.schema)
            low=model.encoder(values,observed,context.schema,through='low')
            self.assertTrue(torch.equal(high[:256],low))
            changed=values.clone();changed[:,-1]=changed[:,-1].roll(1)
            other=model.encoder(changed,observed,context.schema)
            offset=0
            for block in (256,768,3072):
                self.assertTrue(torch.equal(high[offset:offset+block//2],other[offset:offset+block//2]))
                offset+=block
            self.assertFalse(torch.allclose(high,other))
            row=model.encoder(values.flip(0),observed.flip(0),context.schema)
            torch.testing.assert_close(high,row,atol=2e-5,rtol=2e-5)

    def test_feature_binding_retains_which_identical_marginal_drives_target(self):
        model=SharedModel(tiny()).eval()
        values=torch.tensor([[i,j,i] for i,j in [(-1.,1.),(1.,-1.),(-1.,-1.),(1.,1.)]])
        observed=torch.ones_like(values,dtype=torch.bool)
        columns=[dict(slot=s,kind='numeric',center=0.,scale=1.,categories=0) for s in (1,2,0)]
        schema=dict(columns=columns,task='regression',target_index=2)
        first=model.encoder(values,observed,schema)
        changed=values.clone();changed[:,-1]=values[:,1]
        second=model.encoder(changed,observed,schema)
        self.assertFalse(torch.allclose(first,second,atol=1e-6))

    def test_selected_export_contains_no_private_high_coordinates_and_rejects_spaces(self):
        context,_,_=fixture()
        manifest=dict(trained_levels=['low','medium','high'],coordinate_space_id='a'*64,evidence='synthetic_development_only')
        code=torch.zeros(4096);code[256:]=5123
        local=LocalEncoding(code,context.schema,manifest,True)
        exported=export_encoding(local,'low')
        parsed,metadata=read_code(exported.bytes,manifest)
        self.assertEqual(parsed.numel(),256);self.assertEqual(float(parsed.sum()),0)
        self.assertEqual(metadata['exported_levels'],['low'])
        with self.assertRaises(ResearchError):read_code(exported.bytes,dict(manifest,coordinate_space_id='b'*64))
        with self.assertRaises(ResearchError):export_encoding(local,'ultra')
        with self.assertRaises(ResearchError):export_encoding(local,'low',max_bytes=100)

    def test_unseen_target_class_refuses_without_dropping_rows(self):
        columns=fit_transform([[1,'a'],[2,'a'],[3,'b'],[4,'b']],['numeric','categorical'],[1,0],'classification')
        with self.assertRaisesRegex(ResearchError,'unsupported_target_class'):
            transform([[5,'unseen']],columns,['query-row'],'classification')

    def test_group_and_time_blocks_never_cross_roles(self):
        ids=[str(i) for i in range(160)];groups=[i//10 for i in range(160)]
        for policy in ('group','time'):
            roles=role_indices(ids,0,policy=policy,groups=groups,times=groups)
            units=[{groups[i] for i in indices} for indices in roles.values()]
            for i,unit in enumerate(units):
                for other in units[i+1:]:self.assertFalse(unit.intersection(other))
            if policy=='time':self.assertLess(max(roles['fit']),min(roles['tune']))

    def test_checkpoint_resume_reproduces_the_next_optimizer_updates(self):
        config=TrainConfig(steps=4,accumulation=1,validate_every=2,checkpoint_every=2,
                           transfer_weight=0,driver_weight=0,profile_weight=0,consistency_weight=0,
                           cpu_threads=1,wall_seconds=120)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            whole,full=train(tiny(),config,Corpus(prior_only=True),root/'whole',source_sha256='a'*64)
            _,part=train(tiny(),config,Corpus(prior_only=True),root/'part',source_sha256='a'*64,stop_after=2)
            resumed,result=train(tiny(),config,Corpus(prior_only=True),root/'resumed',source_sha256='a'*64,
                                  resume=part['last_checkpoint'])
            self.assertEqual(full['optimizer_updates'],4);self.assertEqual(result['new_updates'],2)
            for name,value in whole.state_dict().items():self.assertTrue(torch.equal(value,resumed.state_dict()[name]),name)
