"""Actual D08/D09 CPU adapter, native-mask and exact resume conformance."""
import dataclasses
import hashlib
import json
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from research.levels.v4_model import NativeLandmarks,V4Model
from research.levels.v4_objectives import LossWeights
from research.levels.v4_training_adapter import V4ObjectiveAdapter
from research.levels.v4_trainer import (BatchCell,IdentityMismatch,RoleViolation,TrainerError,TrainingIdentity,V4Trainer,V4TrainerConfig,collate_cells)

def cells():
    schema={"schema":"dope.nc1-public/v1","recipe":"nc1-minmax-onehot-v1","columns":2,
        "groups":[{"id":i,"kind":"numeric","start":i,"width":1,"has_missing":False} for i in range(2)],"target":"regression","classes":0}
    result=[]
    for i in range(3):
        x=torch.linspace(.01,.99,16).reshape(8,2);y=.2+.3*x[:,0]+.1*torch.sin(6.28*x[:,1])
        q=x[:6].flip(0).clone();qy=y[:6].flip(0).clone()
        scope=hashlib.sha256(f"public-fixture-fit-scope-{i}".encode()).hexdigest()
        result.append(BatchCell(f"cell-{i}",x,y,q,qy,schema,NativeLandmarks.none(scope),
            tuple(f"{i}-fit-{n}" for n in range(8)),tuple(f"{i}-query-{n}" for n in range(6)),
            keep_mask=torch.zeros(23,dtype=torch.bool),family_id=f"family-{i}",fit_scope_sha256=scope))
    return result

class RealModelTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old_threads)
    def make(self,seed=17,source=None):
        torch.manual_seed(seed);model=V4Model();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        callback=V4ObjectiveAdapter(LossWeights(1.,.2,.1,.1),synthetic_rows=12)
        identity=TrainingIdentity("a"*64,"b"*64,"c"*64,"d"*64,callback.identity_sha256(),"e"*64)
        trainer=V4Trainer(model,optimizer,cells() if source is None else source,callback,identity,V4TrainerConfig(batch_size=1,seed=7,report_interval=1))
        return trainer
    def test_real_model_two_steps_checkpoint_resume_and_native23_mask(self):
        trainer=self.make();before={k:v.clone() for k,v in trainer.model.state_dict().items()}
        batch=collate_cells([trainer.cells[0]],trainer.config);self.assertEqual(batch.keep_masks.shape,(1,23))
        trainer.fit_step();self.assertEqual(trainer.update_count,1)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in trainer.model.state_dict().items()))
        with tempfile.TemporaryDirectory() as out:
            trainer.save_checkpoint(out);trainer.fit_step()
            expected={k:v.clone() for k,v in trainer.model.state_dict().items()};random_after=torch.rand(4)
            restored=self.make(seed=999);restored.load_checkpoint(out);restored.fit_step()
            self.assertEqual(restored.update_count,2);self.assertEqual(restored.cursor,trainer.cursor)
            for k,v in expected.items():torch.testing.assert_close(v,restored.model.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(random_after,torch.rand(4),rtol=0,atol=0)
    def test_same_cell_id_data_schema_membership_weight_changes_refuse_resume(self):
        trainer=self.make();trainer.fit_step()
        with tempfile.TemporaryDirectory() as out:
            trainer.save_checkpoint(out)
            for field in ("data","schema","membership","weight"):
                changed=cells()
                if field=="data":changed[0]=dataclasses.replace(changed[0],x_query=changed[0].x_query+.01)
                elif field=="schema":changed[0]=dataclasses.replace(changed[0],schema={**changed[0].schema,"classes":1})
                elif field=="membership":changed[0]=dataclasses.replace(changed[0],query_row_ids=tuple("changed-"+v for v in changed[0].query_row_ids))
                else:changed[0]=dataclasses.replace(changed[0],weight=2.)
                restored=self.make(seed=123,source=changed)
                with self.assertRaises(IdentityMismatch):restored.load_checkpoint(out)
        cell=trainer.cells[0];cell.x_fit[0,0]=.99
        trainer.permutation=[(cell.cell_id,)];trainer.cursor=0
        with self.assertRaises(IdentityMismatch):trainer.fit_step()
    def test_invalid_optimizer_checkpoint_does_not_partially_restore_model(self):
        source=self.make();source.fit_step()
        with tempfile.TemporaryDirectory() as out:
            source.save_checkpoint(out)
            pointer=json.loads((Path(out)/"CURRENT").read_text())
            state_path=Path(out)/pointer["state"]
            state=torch.load(state_path,weights_only=True)
            state["optimizer"]["param_groups"]=[]
            torch.save(state,state_path)
            target=self.make(seed=123)
            before={k:v.clone() for k,v in target.model.state_dict().items()}
            rng=torch.get_rng_state().clone()
            with self.assertRaises(TrainerError):target.load_checkpoint(out)
            for k,v in before.items():torch.testing.assert_close(v,target.model.state_dict()[k],rtol=0,atol=0)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            self.assertEqual(target.update_count,0)
    def test_native23_and_role_violations_are_distinct_from_padding_masks(self):
        source=cells();bad=dataclasses.replace(source[0],keep_mask=torch.ones_like(source[0].x_fit,dtype=torch.bool))
        with self.assertRaises(RoleViolation):collate_cells([bad],V4TrainerConfig())
        bad=dataclasses.replace(source[0],keep_mask=torch.ones(23,dtype=torch.bool))
        with self.assertRaises(RoleViolation):collate_cells([bad],V4TrainerConfig())
        with self.assertRaises(RoleViolation):dataclasses.replace(source[0],query_role="outer_test").validate()

    def test_batch2_weighted_loss_gradients_match_cellwise_and_padding_poison_is_ignored(self):
        source=cells()
        narrow_schema={"schema":"dope.nc1-public/v1","recipe":"nc1-minmax-onehot-v1","columns":1,
            "groups":[{"id":0,"kind":"numeric","start":0,"width":1,"has_missing":False}],"target":"regression","classes":0}
        narrow=dataclasses.replace(source[1],x_fit=source[1].x_fit[:5,:1].clone(),y_fit=source[1].y_fit[:5].clone(),
            x_query=source[1].x_query[:2,:1].clone(),y_query=source[1].y_query[:2].clone(),schema=narrow_schema,
            fit_row_ids=source[1].fit_row_ids[:5],query_row_ids=source[1].query_row_ids[:2],weight=3.)
        pair=[dataclasses.replace(source[0],weight=1.),narrow]
        config=V4TrainerConfig(batch_size=2,row_bucket=4,column_bucket=2,seed=11)
        callback=V4ObjectiveAdapter(LossWeights(1.,.2,.1,.1),synthetic_rows=8)
        torch.manual_seed(117)
        model=V4Model()
        batch=collate_cells(pair,config)
        self.assertEqual(tuple(batch.x_fit.shape),(2,8,2))
        self.assertEqual(batch.fit_row_mask.sum(1).tolist(),[8,5])
        self.assertEqual(batch.query_row_mask.sum(1).tolist(),[6,2])
        self.assertEqual(batch.fit_column_mask.sum(1).tolist(),[2,1])
        base={k:v.detach().clone() for k,v in model.state_dict().items()}
        batch_rng=torch.get_rng_state().clone()
        actual=callback(model,batch)["loss"]
        actual.backward()
        actual_grads=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]

        reference=V4Model();reference.load_state_dict(base)
        total_weight=sum(cell.weight for cell in pair)
        ref_loss=0.
        ref_grads=[torch.zeros_like(p) for p in reference.parameters()]
        rng_cursor=batch_rng.clone()
        for cell in pair:
            reference.zero_grad(set_to_none=True)
            torch.set_rng_state(rng_cursor)
            one=collate_cells([cell],config)
            one_loss=callback(reference,one)["loss"]
            contribution=cell.weight/total_weight
            ref_loss += float(one_loss.detach())*contribution
            (one_loss*contribution).backward()
            for index,p in enumerate(reference.parameters()):
                if p.grad is not None:ref_grads[index] += p.grad.detach()
            rng_cursor=torch.get_rng_state().clone()
        self.assertTrue(any(grad.abs().sum()>0 for grad in ref_grads),"fixture produced no useful parameter gradient")
        torch.testing.assert_close(actual.detach(),torch.tensor(ref_loss),rtol=1e-6,atol=1e-7)
        for got,want in zip(actual_grads,ref_grads):
            if want.abs().sum()>0:
                self.assertIsNotNone(got)
                torch.testing.assert_close(got,want,rtol=1e-5,atol=1e-7)

        poisoned=copy.deepcopy(batch)
        poisoned.x_fit[1,5:,:]=float("nan");poisoned.y_fit[1,5:]=float("nan")
        poisoned.x_query[1,2:,:]=float("nan");poisoned.y_query[1,2:]=float("nan")
        poisoned.x_fit[1,:5,1:]=float("nan");poisoned.x_query[1,:2,1:]=float("nan")
        poison_model=V4Model();poison_model.load_state_dict(base)
        torch.set_rng_state(batch_rng)
        poison_loss=callback(poison_model,poisoned)["loss"]
        poison_loss.backward()
        torch.testing.assert_close(poison_loss.detach(),actual.detach(),rtol=0,atol=0)
        for got,p in zip(actual_grads,poison_model.parameters()):
            if got is not None:
                torch.testing.assert_close(p.grad,got,rtol=0,atol=0)

if __name__=="__main__":unittest.main()
