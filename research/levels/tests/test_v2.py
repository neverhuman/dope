"""Versioned capacity, slot information and frozen residual qualification fixtures."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
from research.levels.contracts import (LAYOUT, LAYOUT_V2, SLOT_GRAPH, MATCHED_GRAPH,
    POOLED_GRAPH, LEVELS, ResearchError, level_widths)
from research.levels.corpus import Corpus
from research.levels.model import SharedModel, tensors
from research.levels.packages import export_package, encode_table, export_encoding, read_code, load_decoder
from research.levels.train import train
from research.levels.tests.test_models import tiny, fixture
from research.levels.tests.test_training import config


def graph_config(graph=SLOT_GRAPH):
    return replace(tiny(),layout=LAYOUT_V2,graph=graph,summary_slots=4)


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_layout_pairs_refuse_cross_version_and_old_defaults_stay_exact(self):
        self.assertEqual(tiny().layout,LAYOUT)
        self.assertEqual(level_widths(LAYOUT),LEVELS)
        self.assertEqual(list(level_widths(LAYOUT_V2).values()),[1024,4096,8192,10240])
        for graph in (SLOT_GRAPH,MATCHED_GRAPH,POOLED_GRAPH):graph_config(graph).validate()
        for cfg in [replace(tiny(),layout=LAYOUT_V2),replace(tiny(),graph=SLOT_GRAPH),
                    replace(graph_config(),summary_slots=1),replace(graph_config(),summary_slots=33)]:
            with self.assertRaises(ResearchError):cfg.validate()

    def test_slot_readout_retains_ordered_information_that_pooled_control_removes(self):
        torch.manual_seed(31);slot=SharedModel(graph_config())
        torch.manual_seed(31);pooled=SharedModel(graph_config(MATCHED_GRAPH))
        self.assertEqual({n:p.shape for n,p in slot.named_parameters()},
                         {n:p.shape for n,p in pooled.named_parameters()})
        for n,v in slot.state_dict().items():self.assertTrue(torch.equal(v,pooled.state_dict()[n]),n)
        summary=torch.tensor([[1.,2.,4.,8.,0.,-1.,3.,9.],[4.,1.,-2.,3.,6.,7.,0.,-4.]])
        def read(model,rows):return model.encoder.distribution_heads[0](
            model.encoder.readout(rows,model.encoder.distribution_norm))
        self.assertFalse(torch.equal(read(slot,summary),read(slot,summary.flip(0))))
        self.assertTrue(torch.equal(read(pooled,summary),read(pooled,summary.flip(0))))

    def test_selected_prefix_target_isolation_and_upper_query_poisoning(self):
        model=SharedModel(graph_config());context,query,_=fixture();v,o=tensors(context,'cpu')
        with torch.no_grad():
            full=model.encoder(v,o,context.schema,through='ultra')
            low=model.encoder(v,o,context.schema,through='low')
            self.assertTrue(torch.equal(low,full[:1024]))
            changed=v.clone();changed[:,-1]*=-3
            other=model.encoder(changed,o,context.schema,through='ultra')
            start=0
            for width in (1024,3072,4096,2048):
                self.assertTrue(torch.equal(full[start:start+width//2],other[start:start+width//2]))
                start+=width
            model.encoder.residual_summaries.add_(10000)
            for head in [model.encoder.distribution_heads[3],model.encoder.task_heads[3]]:head.weight.fill_(500)
            self.assertTrue(torch.equal(low,model.encoder(v,o,context.schema,through='low')))

    def test_v2_resume_and_ultra_only_changes_residual_queries_and_heads(self):
        cfg=replace(config(),steps=2,validate_every=1,checkpoint_every=1,transfer_weight=0,driver_weight=0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);corpus=Corpus(prior_only=True)
            whole,_=train(graph_config(),cfg,corpus,root/'whole',source_sha256='a'*64)
            _,part=train(graph_config(),cfg,corpus,root/'part',source_sha256='a'*64,stop_after=1)
            resumed,_=train(graph_config(),cfg,corpus,root/'resumed',source_sha256='a'*64,resume=part['last_checkpoint'])
            for n,v in whole.state_dict().items():self.assertTrue(torch.equal(v,resumed.state_dict()[n]),n)
            residual,terminal=train(graph_config(),cfg,corpus,root/'residual',source_sha256='a'*64,
                                    phase='ultra-residual',base_checkpoint=part['last_checkpoint'],stop_after=1)
            # Compare the exact ordinary base, not the separately completed two-update model.
            from safetensors.torch import load_file
            base=load_file(str(Path(part['last_checkpoint'])/'state.safetensors'))
            changed=[]
            allowed=('encoder.residual_summaries','encoder.distribution_heads.3.','encoder.task_heads.3.',
                     'decoder.conditions.3.','profiles.ultra.')
            for n,v in residual.state_dict().items():
                if not torch.equal(v,base['model.'+n]):changed.append(n)
                if not n.startswith(allowed):self.assertTrue(torch.equal(v,base['model.'+n]),n)
            self.assertIn('encoder.residual_summaries',changed)
            manifest=export_package(terminal['last_checkpoint'],root/'model')
            context,_,_=fixture();encoded=encode_table(residual,manifest,context,through='ultra')
            for level,width in level_widths(LAYOUT_V2).items():
                raw=export_encoding(encoded,level)
                code,meta=read_code(raw.bytes,manifest)
                self.assertEqual(code.numel(),width);self.assertEqual(meta['layout'],LAYOUT_V2)
                self.assertEqual(meta['views'][0]['length'],width)
            wrong=dict(manifest,layout=LAYOUT)
            with self.assertRaises(ResearchError):read_code(raw.bytes,wrong)
            (root/'model/encoder.safetensors').unlink()
            decoder,_=load_decoder(root/'model')
            from research.levels.decoder import sample
            for level,width in level_widths(LAYOUT_V2).items():
                values,observed=sample(decoder,encoded.code[:width],encoded.schema,4,11,steps=8)
                self.assertTrue(torch.isfinite(values).all());self.assertTrue(observed[:,-1].all())
