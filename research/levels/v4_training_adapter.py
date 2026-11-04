"""Explicit frozen callback connecting bounded D08 training to D09 math."""
from dataclasses import asdict, dataclass
import hashlib
import json
import torch
from .v4_model import NativeLandmarks, pin, require
from .v4_objectives import LossWeights, synthesize_then_learn_loss

@dataclass(frozen=True)
class V4ObjectiveAdapter:
    weights: LossWeights
    synthetic_rows: int

    def identity_sha256(self):
        self.weights.validate()
        require(type(self.synthetic_rows) is int and 2 <= self.synthetic_rows <= 100000,"invalid_v4_synthetic_rows")
        return hashlib.sha256(json.dumps({"adapter":"dope.v4-training-adapter/v1","weights":asdict(self.weights),
            "synthetic_rows":self.synthetic_rows,"query_role":"gradient_query"},sort_keys=True,separators=(",",":")).encode()).hexdigest()

    def __call__(self,model,batch):
        self.identity_sha256()
        for scope,report in zip(batch.fit_scopes,batch.native_reports):
            pin(scope)
            require(isinstance(report,NativeLandmarks) and report.fit_scope_sha256==scope,"native_fit_scope_mismatch")
        states=model.encode_batch(batch.x_fit,batch.y_fit,batch.fit_row_mask,batch.schemas,batch.native_reports,keep_masks=batch.keep_masks)
        totals=[];components={key:[] for key in ("tstr","driver","distribution","native")}
        for i,state in enumerate(states):
            p=state.schema["columns"]
            x=batch.x_query[i,batch.query_row_mask[i],:p];y=batch.y_query[i,batch.query_row_mask[i]]
            draws=torch.rand((self.synthetic_rows,2*len(state.groups)+1),dtype=state.x_fit.dtype,device=state.x_fit.device)
            result=synthesize_then_learn_loss(model,state,x,y,draws,weights=self.weights,query_role="gradient_query")
            totals.append(result["total"])
            for key in components:components[key].append(result[key].detach())
        weights=batch.weights.to(totals[0])
        loss=(torch.stack(totals)*weights).sum()/weights.sum()
        return {"loss":loss,**{k:torch.stack(v).mean() for k,v in components.items()}}
