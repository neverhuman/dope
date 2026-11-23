"""Export all served DML4 parameters; no residual model/source dependency."""
from __future__ import annotations
import hashlib
import json
import math
import struct
import zlib

import torch

from .v4_model import GATE_KEYS, canonical_schema, pin, require, schema_sha256
from .v4_wire_reference import decode


def _public(value):
    if isinstance(value,torch.Tensor):
        require(bool(torch.isfinite(value).all()),"nonfinite_v4_export")
        return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:_public(v) for k,v in value.items()}
    if isinstance(value,list):return [_public(v) for v in value]
    return value



def _validate_graph(packet):
    unit=lambda v:type(v) in (int,float) and math.isfinite(v) and 0<=v<=1
    parameter=lambda v:type(v) in (int,float) and math.isfinite(v) and abs(v)<=1e6
    for spec,group in zip(packet["schema"]["groups"],packet["groups"]):
        kind=group.get("kind")
        keys={"numeric":{"kind","knots","missing_probability","imputation"},
              "discrete":{"kind","values","probabilities","missing_probability","imputation"},
              "categorical":{"kind","probabilities"}}
        require(kind in keys and set(group)==keys[kind],"invalid_v4_group_export")
        if kind=="categorical":
            require(spec["kind"]=="nominal" and len(group["probabilities"])==spec["width"],"invalid_v4_group_export")
        else:
            require(unit(group["imputation"]) and unit(group["missing_probability"])
                    and (spec["has_missing"] or group["missing_probability"]==0),"invalid_v4_group_export")
        if kind=="numeric":
            q=group["knots"]
            require(spec["kind"]=="numeric" and 2<=len(q)<=65 and all(unit(v) for v in q)
                    and all(a<=b for a,b in zip(q,q[1:])),"invalid_v4_group_export")
        if kind=="discrete":
            values=group["values"]
            require(spec["kind"] in ("boolean","integer","ordinal") and 1<=len(values)<=256
                    and len(values)==len(group["probabilities"]) and all(unit(v) for v in values)
                    and all(a<b for a,b in zip(values,values[1:])),"invalid_v4_group_export")
            if spec["kind"]=="boolean":require(all(v in (0,1) for v in values) and group["imputation"] in (0,1),"invalid_v4_group_export")
        if kind in ("discrete","categorical"):
            probabilities=group["probabilities"]
            require(probabilities and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in probabilities)
                    and math.isfinite(sum(probabilities)) and sum(probabilities)>0,"invalid_v4_group_export")
    require(len(packet["groups"])==len(packet["schema"]["groups"]),"invalid_v4_group_export")
    target=packet["target"];width=5*packet["schema"]["columns"]
    if packet["schema"]["target"]=="regression":
        require(set(target)=={"task","intercept","coefficients","noise_scale"} and target["task"]=="regression"
                and parameter(target["intercept"]) and len(target["coefficients"])==width
                and all(parameter(v) for v in target["coefficients"]) and 0<=target["noise_scale"]<=1e6,"invalid_v4_target_export")
    else:
        k=packet["schema"]["classes"]
        require(set(target)=={"task","intercepts","coefficients"} and target["task"]=="classification"
                and len(target["intercepts"])==len(target["coefficients"])==k and all(parameter(v) for v in target["intercepts"])
                and all(len(row)==width and all(parameter(v) for v in row) for row in target["coefficients"]),"invalid_v4_target_export")


def export_packet(model,state,*,level):
    require(level in ("2k","16k","64k"),"invalid_v4_level")
    require(state.producer_versions == model.parameter_versions(), "stale_v4_encoded_state")
    schema=canonical_schema(state.schema)
    packet={"contract":"dope.dml4-reference/v1","level":level,"schema":schema,
        "schema_sha256":schema_sha256(schema),
        "identity":{"weights_sha256":model.weights_sha256(),"encoder_recipe_sha256":model.recipe_sha256(),
                    "conditioning_sha256":pin(state.conditioning_sha256),"fit_scope_sha256":pin(state.fit_scope_sha256)},
        "gates":{k:_public(state.gates[k]) for k in GATE_KEYS},"groups":_public(state.groups),"target":_public(state.target)}
    require(all(math.isfinite(v) and 0<=v<=4 for v in packet["gates"].values()),"invalid_v4_gate")
    _validate_graph(packet)
    body=json.dumps(packet,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    wire=b"DML4\x01"+struct.pack("<II",len(body),zlib.crc32(body))+body
    require(len(wire)<={"2k":2048,"16k":16384,"64k":65536}[level],"v4_packet_over_budget")
    # Frozen reference envelope validation; Rust remains product semantic authority.
    require(decode(wire)==packet,"v4_export_decode_mismatch")
    return packet,wire
