"""NC1-native research producer for the frozen DML4 reference graph.

This is an additive unqualified model, not released-v3 input compatibility.
Marginals are independent by the served contract; every target basis gate and
coefficient is exported. No hidden neural residual remains in the sampler.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

GATE_KEYS = ("linear", "asinh", "periodic", "hinge", "pair", "row_mean")
KINDS = ("numeric", "integer", "ordinal", "boolean", "nominal")


class V4Error(ValueError):
    """Source-free producer failure code."""


def require(ok, code):
    if not ok:
        raise V4Error(code)


def pin(value):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid_v4_pin")
    return value


def canonical_schema(schema):
    require(type(schema) is dict and set(schema) == {"schema", "recipe", "columns", "groups", "target", "classes"}, "invalid_v4_schema")
    require(schema["schema"] == "dope.nc1-public/v1" and schema["recipe"] == "nc1-minmax-onehot-v1", "invalid_v4_schema")
    p = schema["columns"]
    require(type(p) is int and 1 <= p <= 256 and type(schema["groups"]) is list and schema["groups"], "invalid_v4_shape")
    require(schema["target"] in ("regression", "binary", "multiclass"), "invalid_v4_task")
    k = schema["classes"]
    require(type(k) is int and ((schema["target"] == "regression" and k == 0) or
            (schema["target"] == "binary" and k == 2) or (schema["target"] == "multiclass" and 3 <= k <= 256)), "invalid_v4_classes")
    groups, cursor, ids = [], 0, set()
    for g in schema["groups"]:
        require(type(g) is dict and set(g) == {"id", "kind", "start", "width", "has_missing"}, "invalid_v4_group")
        require(type(g["id"]) is int and g["id"] >= 0 and g["id"] not in ids and g["kind"] in KINDS and type(g["has_missing"]) is bool, "invalid_v4_group")
        require(type(g["start"]) is int and g["start"] == cursor and type(g["width"]) is int and g["width"] > 0, "invalid_v4_group")
        require(g["kind"] == "nominal" or g["width"] == 1 + int(g["has_missing"]), "invalid_v4_group")
        ids.add(g["id"]); cursor += g["width"]
        groups.append({key: g[key] for key in ("id", "kind", "start", "width", "has_missing")})
    require(cursor == p, "invalid_v4_group")
    return {"schema": schema["schema"], "recipe": schema["recipe"], "columns": p,
            "groups": groups, "target": schema["target"], "classes": k}


def schema_sha256(schema):
    return hashlib.sha256(json.dumps(canonical_schema(schema), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class NativeLandmarks:
    recipe: str
    fit_scope_sha256: str
    values: torch.Tensor
    observed: torch.Tensor
    source_report_sha256: str

    @classmethod
    def none(cls, fit_scope_sha256, *, dtype=torch.float32, device=None):
        return cls("none", pin(fit_scope_sha256), torch.zeros(23, dtype=dtype, device=device),
                   torch.zeros(23, dtype=torch.bool, device=device), "0" * 64)

    def validate(self, required_recipe):
        pin(self.fit_scope_sha256); pin(self.source_report_sha256)
        require(self.values.shape == (23,) and self.observed.shape == (23,) and self.observed.dtype == torch.bool,
                "invalid_native_landmark_shape")
        require(not self.values.requires_grad and bool(torch.isfinite(self.values).all()), "invalid_native_landmark_values")
        if self.recipe == "none":
            require(not bool(self.observed.any()) and bool((self.values == 0).all()) and self.source_report_sha256 == "0" * 64,
                    "invalid_absent_landmarks")
        else:
            require(self.recipe == required_recipe and self.source_report_sha256 != "0" * 64, "native_landmark_recipe_mismatch")


@dataclass
class EncodedState:
    schema: dict
    x_fit: torch.Tensor
    y_fit: torch.Tensor
    embedding512: torch.Tensor
    column_tokens: torch.Tensor
    gates: dict[str, torch.Tensor]
    groups: list[dict]
    target: dict
    native_prediction: torch.Tensor
    native_values: torch.Tensor
    native_observed: torch.Tensor
    fit_scope_sha256: str
    conditioning_sha256: str
    producer_versions: tuple


def basis(x, gates):
    require(x.ndim == 2 and 1 <= x.shape[1] <= 256 and set(gates) == set(GATE_KEYS), "invalid_v4_basis_shape")
    p = x.shape[1]
    out = torch.cat([gates["linear"] * x, gates["asinh"] * torch.asinh(x),
        gates["periodic"] * torch.sin(2 * math.pi * x), gates["hinge"] * F.relu(x - .5),
        gates["pair"] * x[:, :-1] * x[:, 1:], gates["row_mean"] * x.mean(dim=1, keepdim=True)], dim=1)
    require(out.shape[1] == 5*p and bool(torch.isfinite(out).all()), "nonfinite_v4_basis")
    return out


def ridge(features, targets, alpha):
    # Autograd-preserving FP32 solves under AMP; no half-precision Gram solve.
    if features.dtype in (torch.float16,torch.bfloat16):
        features,targets=features.float(),targets.float()
    design = torch.cat([torch.ones((len(features), 1), dtype=features.dtype, device=features.device), features], dim=1)
    eye = torch.eye(design.shape[1], dtype=features.dtype, device=features.device)
    eye[0, 0] = 1e-6
    try:
        result = torch.linalg.solve(design.T @ design / len(design) + alpha * eye,
                                    design.T @ targets / len(design))
    except RuntimeError:
        raise V4Error("v4_ridge_solve_failed") from None
    require(bool(torch.isfinite(result).all()), "nonfinite_v4_ridge")
    return result


class V4Model(nn.Module):
    def __init__(self, *, native_recipe="landmarks-native-v1", ridge_alpha=.02):
        super().__init__()
        require(native_recipe == "landmarks-native-v1" and math.isfinite(ridge_alpha) and ridge_alpha > 0,
                "invalid_v4_config")
        self.native_recipe, self.ridge_alpha = native_recipe, ridge_alpha
        self.column_net = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.group_net = nn.Sequential(nn.Linear(128, 128), nn.SiLU())
        self.descriptor_net = nn.Sequential(nn.Linear(180, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.task_head, self.dist_head = nn.Linear(256, 128), nn.Linear(256, 128)
        self.gate_head, self.native_head = nn.Linear(256, 6), nn.Linear(256, 23)
        self.class_embedding = nn.Embedding(256, 32)
        self.coefficient_head = nn.Sequential(nn.Linear(288, 64), nn.SiLU(), nn.Linear(64, 4))
        self.pair_head = nn.Sequential(nn.Linear(416, 64), nn.SiLU(), nn.Linear(64, 1))
        self.row_head, self.intercept_head = nn.Linear(160, 1), nn.Linear(160, 1)
        self.marginal_head = nn.Linear(256, 2)
        self.support_head = nn.Sequential(nn.Linear(258, 32), nn.SiLU(), nn.Linear(32, 1))
        self.noise_head = nn.Linear(128, 1)

    def recipe(self):
        return {"contract": "dope.v4-producer/v1", "basis": "dope.dml4-reference/v1", "max_columns": 256,
                "embedding": [256,128,128], "native_recipe": self.native_recipe,
                "ridge_alpha": self.ridge_alpha, "discrete_gradient": "cdf-straight-through-v1",
                "column_statistics": ["mean","std","min","max","target_covariance","missing_rate","group_width","kind_code"]}

    def recipe_sha256(self):
        return hashlib.sha256(json.dumps(self.recipe(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def parameter_versions(self):
        return tuple((name,str(value.dtype),str(value.device),tuple(value.shape),value._version)
                     for name,value in sorted(self.state_dict(keep_vars=True).items()))

    def weights_sha256(self):
        h = hashlib.sha256(self.recipe_sha256().encode())
        for name, value in sorted(self.state_dict().items()):
            h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def encode(self, x_fit, y_fit, schema, native_report, *, keep_mask=None):
        schema = canonical_schema(schema); p = schema["columns"]
        require(x_fit.ndim == 2 and len(x_fit) >= 2 and x_fit.shape[1] == p and y_fit.shape == (len(x_fit),), "invalid_v4_fit_shape")
        require(bool(torch.isfinite(x_fit).all() and torch.isfinite(y_fit).all()) and bool(((x_fit >= 0) & (x_fit <= 1)).all()), "invalid_v4_fit_values")
        require(isinstance(native_report, NativeLandmarks), "native_landmark_provenance_required")
        native_report.validate(self.native_recipe)
        if schema["target"] == "regression":
            require(bool(((y_fit >= 0) & (y_fit <= 1)).all()), "invalid_v4_fit_target_range")
        else:
            require(bool(((y_fit == y_fit.round()) & (y_fit >= 0) & (y_fit < schema["classes"])).all()), "invalid_v4_fit_classes")
        observed = native_report.observed.to(x_fit.device).clone()
        if keep_mask is not None:
            require(keep_mask.shape == (23,) and keep_mask.dtype == torch.bool and not bool((keep_mask.to(observed.device) & ~observed).any()), "invalid_native_dropout_mask")
            observed = observed & keep_mask.to(observed.device)
        values = native_report.values.to(x_fit).detach().clone()
        active = torch.where(observed, values, torch.zeros_like(values))
        centered_y = y_fit - y_fit.mean()
        means, stds = x_fit.mean(0), x_fit.std(0, unbiased=False)
        covariance = ((x_fit - means) * centered_y[:, None]).mean(0)
        group_widths, missing, kind_codes = x_fit.new_zeros(p), x_fit.new_zeros(p), x_fit.new_zeros(p)
        for g in schema["groups"]:
            a, w = g["start"], g["width"]
            group_widths[a:a+w] = w / 256
            kind_codes[a:a+w] = KINDS.index(g["kind"]) / (len(KINDS)-1)
            if g["has_missing"]:
                missing[a:a+w] = x_fit[:, a+w-1].mean()
            if g["kind"] == "nominal":
                require(bool((x_fit[:, a:a+w].sum(1) - 1).abs().le(1e-6).all()), "invalid_v4_onehot_group")
        stats = torch.stack([means, stds, x_fit.min(0).values, x_fit.max(0).values,
                             covariance, missing, group_widths, kind_codes], dim=1)
        columns = self.column_net(stats)
        tokens, pooled = [], []
        for g in schema["groups"]:
            c = columns[g["start"]:g["start"]+g["width"]]
            t = self.group_net(torch.cat([c, c.mean(0, keepdim=True).expand_as(c)], dim=1))
            tokens.append(t); pooled.append(t.mean(0))
        column_tokens = torch.cat(tokens, dim=0)
        task_index = {"regression":0,"binary":1,"multiclass":2}[schema["target"]]
        task = F.one_hot(torch.tensor(task_index, device=x_fit.device), 3).to(x_fit)
        shape = x_fit.new_tensor([math.log1p(len(x_fit))/16, p/256, schema["classes"]/256])
        desc = self.descriptor_net(torch.cat([torch.stack(pooled).mean(0), active, observed.to(x_fit), task, shape]))
        task_latent, dist_latent = self.task_head(desc), self.dist_head(desc)
        embedding = torch.cat([desc, task_latent, dist_latent])
        gates = dict(zip(GATE_KEYS, torch.sigmoid(self.gate_head(desc)) * 4))
        groups = []
        for g in schema["groups"]:
            a, w = g["start"], g["width"]
            context = torch.cat([column_tokens[a:a+w].mean(0), dist_latent])
            warp, miss_shift = self.marginal_head(context)
            if g["kind"] == "nominal":
                base = x_fit[:, a:a+w].mean(0)
                logits = []
                for j in range(w):
                    support = x_fit.new_tensor([j/max(w-1,1), (j/max(w-1,1))**2])
                    logits.append(self.support_head(torch.cat([column_tokens[a+j],dist_latent,support])).squeeze())
                groups.append({"kind":"categorical", "probabilities": torch.softmax(torch.log(base.clamp_min(1e-6)) + .2*torch.stack(logits), 0)})
                continue
            valid = x_fit[:, a+w-1] < .5 if g["has_missing"] else torch.ones(len(x_fit),dtype=torch.bool,device=x_fit.device)
            values_fit = x_fit[valid, a]
            if not len(values_fit): values_fit = x_fit.new_tensor([.5 if g["kind"] != "boolean" else 0.])
            base_missing = (~valid).to(x_fit).mean()
            missing_prob = torch.sigmoid(torch.logit(base_missing.clamp(1e-6,1-1e-6)) + .2*miss_shift) if g["has_missing"] else base_missing*0
            imputation = values_fit.median()
            if g["kind"] == "numeric":
                knots = torch.quantile(values_fit, torch.linspace(0,1,9,device=x_fit.device,dtype=x_fit.dtype))
                knots = torch.sigmoid(torch.logit(knots.clamp(1e-6,1-1e-6)) + .2*warp)
                groups.append({"kind":"numeric", "knots":knots,"missing_probability":missing_prob,"imputation":imputation})
            else:
                support, counts = torch.unique(values_fit.detach(), sorted=True, return_counts=True)
                require(len(support) <= 256 and (g["kind"] != "boolean" or bool(((support == 0)|(support == 1)).all())), "invalid_v4_discrete_support")
                context_all = context.expand(len(support), -1)
                offsets = self.support_head(torch.cat([context_all,support[:,None],support[:,None].square()],dim=1)).squeeze(-1)
                probabilities = torch.softmax(torch.log(counts.to(x_fit)) + .2*offsets,0)
                groups.append({"kind":"discrete","values":support,"probabilities":probabilities,"missing_probability":missing_prob,"imputation":imputation})
        k = schema["classes"] or 1
        class_context = torch.cat([task_latent.expand(k,-1),self.class_embedding(torch.arange(k,device=x_fit.device))],dim=1)
        per_column = torch.cat([column_tokens[None].expand(k,-1,-1),class_context[:,None].expand(-1,p,-1)],dim=2)
        blocks = self.coefficient_head(per_column).permute(0,2,1).reshape(k,4*p)
        if p > 1:
            pair_input = torch.cat([column_tokens[:-1],column_tokens[1:]],dim=1)[None].expand(k,-1,-1)
            pair = self.pair_head(torch.cat([pair_input,class_context[:,None].expand(-1,p-1,-1)],dim=2)).squeeze(-1)
        else: pair = x_fit.new_empty((k,0))
        residual = torch.cat([blocks,pair,self.row_head(class_context)],dim=1)
        phi = basis(x_fit,gates)
        if schema["target"] == "regression":
            fitted = ridge(phi,y_fit[:,None],self.ridge_alpha).squeeze(-1)
            target = {"task":"regression","intercept":fitted[0]+.05*self.intercept_head(class_context)[0,0],
                      "coefficients":fitted[1:]+.05*residual[0],"noise_scale":torch.sigmoid(self.noise_head(dist_latent)).squeeze()*.15}
        else:
            onehot = F.one_hot(y_fit.long(),k).to(x_fit)
            prior = onehot.mean(0).clamp_min(1e-6)
            fitted = ridge(phi,onehot-prior,self.ridge_alpha)
            target = {"task":"classification","intercepts":torch.log(prior)+3*fitted[0]+.05*self.intercept_head(class_context).squeeze(-1),
                      "coefficients":3*fitted[1:].T+.05*residual}
        h = hashlib.sha256()
        for tensor in (active,observed): h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        h.update(native_report.source_report_sha256.encode());h.update(schema_sha256(schema).encode())
        return EncodedState(schema,x_fit,y_fit,embedding,column_tokens,gates,groups,target,self.native_head(desc),values,
                            native_report.observed.to(x_fit.device).clone(),native_report.fit_scope_sha256,h.hexdigest(),self.parameter_versions())

    def encode_batch(self, x_fit, y_fit, row_mask, schemas, native_reports, *, keep_masks=None):
        require(x_fit.ndim == 3 and y_fit.shape == x_fit.shape[:2] and row_mask.shape == y_fit.shape
                and row_mask.dtype == torch.bool and len(schemas) == len(x_fit) == len(native_reports), "invalid_v4_batch")
        if keep_masks is not None: require(len(keep_masks) == len(x_fit), "invalid_native_dropout_mask")
        out = []
        for i,schema in enumerate(schemas):
            p = canonical_schema(schema)["columns"]
            require(p <= x_fit.shape[2], "invalid_v4_batch")
            # Padded values (including NaN) are never read by any moment/solve.
            out.append(self.encode(x_fit[i,row_mask[i],:p],y_fit[i,row_mask[i]],schema,native_reports[i],
                                   keep_mask=None if keep_masks is None else keep_masks[i]))
        return out

    forward = encode_batch
