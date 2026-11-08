"""Differentiable synthesize -> learner -> disjoint real-query objective.

Hard categorical/missing draws match DML4 in the forward pass. Gradients use a
named CDF straight-through estimator; it is not an unbiased wire derivative.
Scientific loss weights are explicit caller inputs, not inferred from outcomes.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .v4_model import V4Error, basis, require, ridge


@dataclass(frozen=True)
class LossWeights:
    tstr: float
    driver: float
    distribution: float
    native: float
    temperature: float = .1

    def validate(self):
        require(all(math.isfinite(v) and v >= 0 for v in (self.tstr,self.driver,self.distribution,self.native))
                and self.tstr > 0 and math.isfinite(self.temperature) and self.temperature > 0,
                "invalid_v4_loss_weights")


def _categorical(probabilities, uniform, temperature):
    probabilities = probabilities / probabilities.sum()
    cdf = probabilities.cumsum(0)
    index = torch.searchsorted(cdf.detach().contiguous(),uniform.contiguous(),right=True).clamp_max(len(probabilities)-1)
    hard = F.one_hot(index,len(probabilities)).to(probabilities)
    midpoint = cdf - .5*probabilities
    soft = torch.softmax(-((uniform[:,None]-midpoint[None])/temperature).square(),dim=1)
    # Parentheses preserve exact hard forward values while retaining gradients.
    return hard + (soft-soft.detach())


def sample_differentiable(state, draws, *, temperature=.1):
    groups, schema = state.groups, state.schema
    require(draws.dtype == state.x_fit.dtype and draws.device == state.x_fit.device, "v4_draw_dtype_mismatch")
    require(draws.ndim == 2 and 1 <= len(draws) <= 100000 and draws.shape[1] == 2*len(groups)+1
            and len(draws)*schema["columns"] <= 16000000, "invalid_v4_draw_shape")
    require(bool(torch.isfinite(draws).all()) and bool(((draws >= 0)&(draws < 1)).all()), "invalid_v4_draws")
    require(math.isfinite(temperature) and temperature > 0, "invalid_v4_temperature")
    rows=[]
    for index,(spec,model) in enumerate(zip(schema["groups"],groups)):
        u,missing_u=draws[:,2*index],draws[:,2*index+1]
        if model["kind"] == "categorical":
            rows.append(_categorical(model["probabilities"],u,temperature));continue
        probability=model["missing_probability"]
        hard=(missing_u < probability.detach()).to(draws)
        soft=torch.sigmoid((probability-missing_u)/temperature)
        missing=hard+(soft-soft.detach())
        if model["kind"] == "discrete":
            onehot=_categorical(model["probabilities"],u,temperature)
            value=onehot @ model["values"]
        else:
            knots=model["knots"]
            position=u*(len(knots)-1)
            low=position.floor().long().clamp_max(len(knots)-2)
            value=knots[low]+(position-low)*(knots[low+1]-knots[low])
        value=missing*model["imputation"]+(1-missing)*value
        rows.append(value[:,None])
        if spec["has_missing"]:rows.append(missing[:,None])
    x=torch.cat(rows,dim=1);phi=basis(x,state.gates);target=state.target
    if target["task"] == "regression":
        y=target["intercept"]+phi @ target["coefficients"]+target["noise_scale"]*(2*draws[:,-1]-1)*math.sqrt(3)
        probabilities=None
    else:
        logits=phi @ target["coefficients"].T+target["intercepts"]
        p=torch.softmax(logits,dim=1)
        # Per-row CDF has a different learned class distribution.
        cdf=p.cumsum(1);index=(draws[:,-1,None] >= cdf.detach()).sum(1).clamp_max(p.shape[1]-1)
        hard=F.one_hot(index,p.shape[1]).to(p)
        midpoint=cdf-.5*p
        soft=torch.softmax(-((draws[:,-1,None]-midpoint)/temperature).square(),dim=1)
        probabilities=hard+(soft-soft.detach())
        y=probabilities @ torch.arange(p.shape[1],dtype=x.dtype,device=x.device)
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()), "nonfinite_v4_synthesis")
    return {"x":x,"y":y,"target_probabilities":probabilities}


def _learner_features(x):
    return torch.cat([x,x.square(),torch.sin(2*math.pi*x)],dim=1)


def _fit_learner(x,y,probabilities,classes,alpha):
    if classes:
        targets=probabilities if probabilities is not None else F.one_hot(y.long(),classes).to(x)
        prior=targets.mean(0).clamp_min(1e-6)
        weights=ridge(_learner_features(x),targets-prior,alpha)
        return weights,torch.log(prior)
    return ridge(_learner_features(x),y[:,None],alpha),None


def _predict(learner,x):
    weights,prior=learner
    features=_learner_features(x).to(weights)
    design=torch.cat([torch.ones((len(x),1),dtype=x.dtype,device=x.device),features],dim=1)
    output=design @ weights
    return 3*output+prior if prior is not None else output.squeeze(-1)


def _loss(prediction,target,classes):
    return F.cross_entropy(prediction,target.long()) if classes else F.mse_loss(prediction,target)


def synthesize_then_learn_loss(model,state,x_query,y_query,draws,*,weights,query_role="gradient_query"):
    require(query_role == "gradient_query", "outer_query_role_forbidden")
    require(isinstance(weights,LossWeights), "v4_loss_weights_required");weights.validate()
    require(x_query.ndim == 2 and len(x_query) >= 2 and x_query.shape[1] == state.schema["columns"]
            and y_query.shape == (len(x_query),), "invalid_v4_query_shape")
    require(bool(torch.isfinite(x_query).all() and torch.isfinite(y_query).all()), "invalid_v4_query_values")
    classes=state.schema["classes"]
    if classes:require(bool(((y_query==y_query.round())&(y_query>=0)&(y_query<classes)).all()), "invalid_v4_query_classes")
    synthetic=sample_differentiable(state,draws,temperature=weights.temperature)
    learned=_fit_learner(synthetic["x"],synthetic["y"],synthetic["target_probabilities"],classes,model.ridge_alpha)
    tstr=_loss(_predict(learned,x_query),y_query,classes)
    real=_fit_learner(state.x_fit,state.y_fit,None,classes,model.ridge_alpha)
    base_real=_loss(_predict(real,x_query),y_query,classes)
    base_synthetic=tstr
    real_importance=[];synthetic_importance=[]
    for group in state.schema["groups"]:
        changed=x_query.clone();a,w=group["start"],group["width"]
        # One deterministic row permutation is shared by every member of group.
        changed[:,a:a+w]=x_query[:,a:a+w].roll(1,dims=0)
        real_importance.append((_loss(_predict(real,changed),y_query,classes)-base_real).clamp_min(0).detach())
        synthetic_importance.append((_loss(_predict(learned,changed),y_query,classes)-base_synthetic).clamp_min(0))
    ri,si=torch.stack(real_importance),torch.stack(synthetic_importance)
    informative=bool(ri.sum()>1e-12)
    driver=F.l1_loss(si/(si.sum()+1e-8),ri/(ri.sum()+1e-8)) if informative else si.sum()*0
    distribution=F.mse_loss(synthetic["x"].mean(0),state.x_fit.mean(0)) + F.mse_loss(synthetic["x"].var(0,unbiased=False),state.x_fit.var(0,unbiased=False))
    native=F.mse_loss(state.native_prediction[state.native_observed],state.native_values[state.native_observed]) if bool(state.native_observed.any()) else state.native_prediction.sum()*0
    total=weights.tstr*tstr+weights.driver*driver+weights.distribution*distribution+weights.native*native
    require(bool(torch.isfinite(total)), "nonfinite_v4_objective")
    return {"total":total,"tstr":tstr,"driver":driver,"distribution":distribution,"native":native,
            "driver_informative":informative,"synthetic":synthetic}
