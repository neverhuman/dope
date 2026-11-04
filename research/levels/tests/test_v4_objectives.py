"""Tiny public CPU conformance cases, not a training/quality campaign."""
import dataclasses
import hashlib
import json
from pathlib import Path
import math
import unittest

import torch
import torch.nn.functional as F

from research.levels.v4_model import (GATE_KEYS, NativeLandmarks, V4Error, V4Model, basis, ridge)
from research.levels.v4_objectives import LossWeights, sample_differentiable, synthesize_then_learn_loss
from research.levels.v4_encode import export_packet
from research.levels.v4_wire_reference import basis as reference_basis, decode as reference_decode, sample_with_draws


def schema(task="regression", p=2):
    return {"schema":"dope.nc1-public/v1","recipe":"nc1-minmax-onehot-v1","columns":p,
            "groups":[{"id":i,"kind":"numeric","start":i,"width":1,"has_missing":False} for i in range(p)],
            "target":task,"classes":0 if task=="regression" else (2 if task=="binary" else 3)}


class V4ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.model=V4Model().double()
        t=torch.linspace(.02,.98,24,dtype=torch.float64)
        self.x=torch.stack([t,torch.remainder(t*2.3,.97)],dim=1)
        self.y=.2+.4*t+.1*torch.sin(2*math.pi*self.x[:,1])
        self.native=NativeLandmarks.none("a"*64,dtype=torch.float64)
        self.weights=LossWeights(tstr=1.,driver=.2,distribution=.1,native=.1)

    def test_tstr_uses_synthetic_rows_and_all_six_gate_gradients(self):
        state=self.model.encode(self.x,self.y,schema(),self.native)
        draws=torch.rand((32,5),dtype=torch.float64)
        result=synthesize_then_learn_loss(self.model,state,self.x.flip(0),self.y.flip(0),draws,weights=self.weights)
        gradients=torch.autograd.grad(result["total"],tuple(state.gates.values()),retain_graph=True)
        for name,gradient in zip(GATE_KEYS,gradients):
            self.assertTrue(torch.isfinite(gradient),name)
            self.assertGreater(abs(gradient.item()),1e-14,name)
        result["total"].backward()
        for head in [self.model.column_net[0],self.model.group_net[0],self.model.descriptor_net[0],
                     self.model.task_head,self.model.dist_head,self.model.gate_head,self.model.coefficient_head[-1],
                     self.model.pair_head[-1],self.model.row_head,self.model.intercept_head,self.model.marginal_head,self.model.noise_head]:
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(head.weight.grad.abs().sum().item(),1e-14)
        changed=synthesize_then_learn_loss(self.model,state,self.x.flip(0),self.y.flip(0),(draws*.7+.1),weights=self.weights)
        self.assertNotEqual(result["tstr"].item(),changed["tstr"].item())
        self.assertTrue(result["driver_informative"])
        self.assertEqual(state.embedding512.shape,(512,))

    def test_fixed_draws_match_served_sampler_and_independent_basis(self):
        mixed={"schema":"dope.nc1-public/v1","recipe":"nc1-minmax-onehot-v1","columns":5,
               "groups":[{"id":0,"kind":"numeric","start":0,"width":2,"has_missing":True},
                         {"id":1,"kind":"nominal","start":2,"width":2,"has_missing":False},
                         {"id":2,"kind":"boolean","start":4,"width":1,"has_missing":False}],"target":"regression","classes":0}
        x=torch.stack([self.x[:,0],(torch.arange(24)%5==0).double(),(torch.arange(24)%2==0).double(),(torch.arange(24)%2==1).double(),(torch.arange(24)%3==0).double()],dim=1)
        state=self.model.encode(x,self.y,mixed,self.native)
        draws=torch.rand((12,7),dtype=torch.float64)
        generated=sample_differentiable(state,draws)
        packet,wire=export_packet(self.model,state,level="16k")
        expected=sample_with_draws(packet,draws.tolist())
        torch.testing.assert_close(generated["x"],torch.tensor(expected["x"],dtype=torch.float64),rtol=0,atol=1e-12)
        torch.testing.assert_close(generated["y"],torch.tensor(expected["y"],dtype=torch.float64),rtol=0,atol=1e-11)
        self.assertTrue(bool((generated["x"][:,2]+generated["x"][:,3]==1).all()))
        independent=reference_basis(generated["x"][0].tolist(),packet["gates"])
        torch.testing.assert_close(basis(generated["x"][:1],state.gates)[0],torch.tensor(independent,dtype=torch.float64),rtol=0,atol=1e-12)
        self.assertEqual(set(packet["gates"]),set(GATE_KEYS));self.assertLessEqual(len(wire),16384)
        self.assertEqual(packet["identity"]["weights_sha256"],self.model.weights_sha256())

    def test_multiclass_head_has_class_specific_useful_gradient(self):
        y=(torch.arange(24)%3).double()
        state=self.model.encode(self.x,y,schema("multiclass"),self.native)
        self.assertEqual(state.target["coefficients"].shape,(3,10))
        self.assertFalse(torch.equal(state.target["coefficients"][0],state.target["coefficients"][1]))
        draws=torch.rand((30,5),dtype=torch.float64)
        result=synthesize_then_learn_loss(self.model,state,self.x,y,draws,weights=self.weights)
        result["total"].backward()
        self.assertGreater(self.model.intercept_head.weight.grad.abs().sum().item(),1e-14)
        self.assertGreater(self.model.class_embedding.weight.grad.abs().sum().item(),1e-14)
        packet,_=export_packet(self.model,state,level="16k")
        expected=sample_with_draws(packet,draws.tolist())
        torch.testing.assert_close(result["synthetic"]["y"],torch.tensor(expected["y"],dtype=torch.float64),rtol=0,atol=0)
        before=state.target["coefficients"].detach().clone()
        probabilities=torch.softmax(basis(self.x,state.gates)@before.T+state.target["intercepts"].detach(),dim=1).detach()
        torch.optim.SGD(self.model.parameters(),lr=1e-3).step()
        updated=self.model.encode(self.x,y,schema("multiclass"),self.native)
        self.assertFalse(torch.equal(before,updated.target["coefficients"]))
        next_probabilities=torch.softmax(basis(self.x,updated.gates)@updated.target["coefficients"].T+updated.target["intercepts"],dim=1)
        self.assertFalse(torch.equal(probabilities,next_probabilities))
        scalar=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
        logits=torch.randn(6,3,dtype=torch.float64)+scalar
        F.cross_entropy(logits,torch.arange(6)%3).backward()
        self.assertLess(abs(scalar.grad.item()),1e-14)  # released-v3 bug negative control

    def test_nc1_schema_and_exclusive_group_fit_semantics_match_native_contract(self):
        binary_multiclass=schema("multiclass");binary_multiclass["classes"]=2
        state=self.model.encode(self.x,(torch.arange(len(self.x))%2).double(),binary_multiclass,self.native)
        self.assertEqual(state.target["coefficients"].shape,(2,10))
        _,wire=export_packet(self.model,state,level="16k")
        self.assertEqual(reference_decode(wire)["schema"]["classes"],2)
        for bad in [dict(schema(),groups=[{"id":7,"kind":"numeric","start":0,"width":2,"has_missing":True}]),
                    dict(schema(),columns=1,groups=[{"id":0,"kind":"nominal","start":0,"width":1,"has_missing":True}])]:
            with self.assertRaisesRegex(V4Error,"^invalid_v4_group$"):
                self.model.encode(self.x[:,:bad["columns"]],self.y,bad,self.native)
        nominal=dict(schema(),groups=[{"id":0,"kind":"nominal","start":0,"width":2,"has_missing":False}])
        with self.assertRaisesRegex(V4Error,"^invalid_v4_onehot_group$"):
            self.model.encode(torch.full_like(self.x,.5),self.y,nominal,self.native)
        missing=dict(schema(),groups=[{"id":0,"kind":"numeric","start":0,"width":2,"has_missing":True}])
        with self.assertRaisesRegex(V4Error,"^invalid_v4_missing_indicator$"):
            self.model.encode(self.x,self.y,missing,self.native)
        exclusive=F.one_hot(torch.arange(len(self.x))%2,2).double()
        valid=self.model.encode(exclusive,self.y,nominal,self.native)
        loss=synthesize_then_learn_loss(self.model,valid,exclusive,self.y,torch.rand(24,3,dtype=torch.float64),weights=self.weights)
        loss["total"].backward()
        self.assertIsNotNone(self.model.support_head[-1].weight.grad)
        self.assertGreater(self.model.support_head[-1].weight.grad.abs().sum().item(),1e-14)

    def test_tracked_producer_fixture_manifest_binds_sources_and_artifacts(self):
        root=Path(__file__).resolve().parents[3]
        folder=root/"engine/dope-core/tests/fixtures/v4_producer"
        manifest=json.loads((folder/"manifest.json").read_text())
        self.assertEqual(manifest["kind"],"public_synthetic_conformance_only")
        self.assertEqual(manifest["optimizer_updates"],0)
        self.assertEqual(manifest["real_data_reads"],0)
        self.assertEqual(set(manifest["source_sha256"]),{"v4_model.py","v4_objectives.py","v4_encode.py","v4_wire_reference.py"})
        for name,expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((root/"research/levels"/name).read_bytes()).hexdigest(),expected,"producer source drift; regenerate into a new directory with retained evidence")
        self.assertEqual(hashlib.sha256((root/"research/levels/tools/export_v4_conformance.py").read_bytes()).hexdigest(),manifest["generator_sha256"])
        for artifact in manifest["files"]:
            self.assertEqual(Path(artifact["file"]).name,artifact["file"])
            data=(folder/artifact["file"]).read_bytes()
            self.assertEqual(len(data),artifact["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(),artifact["sha256"])

    def test_amp_keeps_gram_and_solve_in_fp32(self):
        features=self.x.float();target=self.y.float()[:,None]
        expected=ridge(features,target,.02)
        with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
            actual=ridge(features,target,.02)
        self.assertEqual(actual.dtype,torch.float32)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_padding_cannot_affect_encoding_and_column_65_is_not_truncated(self):
        baseline=self.model.encode(self.x,self.y,schema(),self.native)
        x=torch.full((1,27,7),float("nan"),dtype=torch.float64);x[0,:24,:2]=self.x
        y=torch.full((1,27),float("nan"),dtype=torch.float64);y[0,:24]=self.y
        mask=torch.zeros((1,27),dtype=torch.bool);mask[0,:24]=True
        batched=self.model.encode_batch(x,y,mask,[schema()],[self.native])[0]
        torch.testing.assert_close(baseline.embedding512,batched.embedding512,rtol=0,atol=0)
        wide=torch.rand((6,65),dtype=torch.float64)
        a=self.model.encode(wide,torch.arange(6,dtype=torch.float64)/6,schema(p=65),self.native)
        changed=wide.clone();changed[:,-1]=0
        b=self.model.encode(changed,torch.arange(6,dtype=torch.float64)/6,schema(p=65),self.native)
        self.assertFalse(torch.equal(a.embedding512,b.embedding512))
        self.assertEqual(a.target["coefficients"].shape,(325,))
        with self.assertRaisesRegex(V4Error,"^v4_packet_over_budget$"):export_packet(self.model,a,level="2k")
        with self.assertRaises(V4Error):self.model.encode(wide,self.y[:6],schema(p=257),self.native)

    def test_native_provenance_dropout_and_outer_role_refusals(self):
        report=NativeLandmarks("landmarks-native-v1","a"*64,torch.linspace(0,1,23,dtype=torch.float64),torch.ones(23,dtype=torch.bool),"b"*64)
        state=self.model.encode(self.x,self.y,schema(),report)
        keep=torch.ones(23,dtype=torch.bool);keep[0]=False
        dropped=self.model.encode(self.x,self.y,schema(),report,keep_mask=keep)
        self.assertNotEqual(state.conditioning_sha256,dropped.conditioning_sha256)
        self.assertEqual(dropped.native_observed.sum().item(),23)  # supervision distinct from input mask
        loss=synthesize_then_learn_loss(self.model,dropped,self.x,self.y,torch.rand(12,5,dtype=torch.float64),weights=self.weights)
        self.assertGreater(loss["native"].item(),0)
        loss["total"].backward()
        self.assertGreater(self.model.native_head.weight.grad.abs().sum().item(),1e-14)
        with self.assertRaises(V4Error):self.model.encode(self.x,self.y,schema(),dataclasses.replace(report,recipe="heuristic-proxy"))
        for role in ["outer_test","sealed_meta_test","tune"]:
            with self.assertRaisesRegex(V4Error,"^outer_query_role_forbidden$"):
                synthesize_then_learn_loss(self.model,state,self.x,self.y,torch.rand(12,5,dtype=torch.float64),weights=self.weights,query_role=role)
        with self.assertRaises(V4Error):self.model.encode(self.x,self.y,schema(),None)
        with self.assertRaises(V4Error):sample_differentiable(state,torch.ones(12,5,dtype=torch.float64))

    def test_export_refuses_private_or_unknown_graph_fields(self):
        state=self.model.encode(self.x,self.y,schema(),self.native)
        state.groups[0]["CANARY_PRIVATE_HEADER"]="PRIVATE_VALUE"
        with self.assertRaisesRegex(V4Error,"^invalid_v4_group_export$"):
            export_packet(self.model,state,level="16k")

    def test_export_refuses_state_encoded_before_parameter_update(self):
        state=self.model.encode(self.x,self.y,schema(),self.native)
        with torch.no_grad():self.model.gate_head.bias.add_(.01)
        with self.assertRaisesRegex(V4Error,"^stale_v4_encoded_state$"):
            export_packet(self.model,state,level="16k")
        fresh=self.model.encode(self.x,self.y,schema(),self.native)
        self.assertEqual(export_packet(self.model,fresh,level="16k")[0]["identity"]["weights_sha256"],self.model.weights_sha256())

    def test_query_poisoning_changes_loss_only_and_regression_tails_are_not_clipped(self):
        state=self.model.encode(self.x,self.y,schema(),self.native)
        before={k:v.detach().clone() for k,v in state.gates.items()}
        draws=torch.rand(12,5,dtype=torch.float64)
        small=synthesize_then_learn_loss(self.model,state,self.x,self.y,draws,weights=self.weights)
        large=synthesize_then_learn_loss(self.model,state,self.x+10,self.y+100,draws,weights=self.weights)
        self.assertGreater(large["tstr"].item(),small["tstr"].item())
        for k in GATE_KEYS:torch.testing.assert_close(state.gates[k],before[k],rtol=0,atol=0)
        self.assertNotIn("PRIVATE",json.dumps(export_packet(self.model,state,level="16k")[0]))


if __name__=="__main__":unittest.main()
