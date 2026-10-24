"""Tensor-only local model packages and exact selected-code exports."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch
from safetensors.torch import load_file, save, load
from .checkpoints import private_json
from .contracts import LAYOUT, level_widths, PREPROCESSING, ModelConfig, canonical, digest, levels, require, validate_schema
from .corpus import file_sha
from .model import SharedModel, tensors
from .table_encoder import quantized_code


def model_layout(model):
    layout = model.get('layout', LAYOUT)
    level_widths(layout)
    if 'graph' in model:
        require(ModelConfig(**model['graph']).validate().layout == layout, 'model_layout_binding')
    else:
        require(layout == LAYOUT, 'versioned_model_graph_required')
    return layout


def atomic_bytes(path, raw):
    path=Path(path)
    require(path.parent.is_dir(), 'output_parent_missing')
    staging=path.with_name('.'+uuid.uuid4().hex+'.partial')
    descriptor=os.open(staging,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        require(staging.read_bytes()==raw,'export_readback_failed')
        os.link(staging,path)
    finally:staging.unlink(missing_ok=True)


def export_package(checkpoint, output):
    checkpoint,output=Path(checkpoint),Path(output)
    metadata=json.loads((checkpoint/'manifest.json').read_bytes())
    require(metadata['format']=='dope-research-checkpoint-v1'
            and metadata['state']['optimizer_updates']>0, 'trained_checkpoint_required')
    require(file_sha(checkpoint/'state.safetensors')==metadata['tensor_sha256'],'checkpoint_digest_mismatch')
    require(not output.exists(),'package_output_must_be_new')
    weights=load_file(str(checkpoint/'state.safetensors'),device='cpu')
    staging=output.with_name('.'+output.name+'-'+uuid.uuid4().hex)
    staging.mkdir(mode=0o700,parents=True)
    files={}
    for component in ('encoder','decoder'):
        prefix='model.'+component+'.'
        tensors={key[len(prefix):]:value.contiguous() for key,value in weights.items() if key.startswith(prefix)}
        raw=save(tensors)
        name=component+'.safetensors'
        atomic_bytes(staging/name,raw)
        files[component]=dict(file=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                              parameters=sum(t.numel() for t in tensors.values()))
    manifest=dict(format='dope-shared-model-v1',graph=metadata['recipe']['model'],
        preprocessing=PREPROCESSING,layout=metadata['recipe']['model']['layout'],weights=files,
        trained_levels=metadata['state']['trained_levels'],dtype='F32',
        source_sha256=metadata['recipe']['source_sha256'],
        training_manifest_sha256=digest(metadata['recipe']),
        evidence=metadata['recipe']['evidence'],certified=False,
        output_schema='positional_numeric_and_category_ids',
        supported_backends=['cpu_reference'],quantization='fixed-fp16-v1')
    manifest['coordinate_space_id']=digest(manifest)
    private_json(staging/'model.json',manifest)
    os.rename(staging,output)
    return manifest


def package_manifest(path):
    path=Path(path)
    require(not (path/'model.json').is_symlink() and (path/'model.json').stat().st_size<=262144,
            'model_manifest_too_large_or_symlink')
    manifest=json.loads((path/'model.json').read_bytes())
    supplied=dict(manifest);identity=supplied.pop('coordinate_space_id')
    require(set(manifest)=={'format','graph','preprocessing','layout','weights','trained_levels','dtype',
            'source_sha256','training_manifest_sha256','evidence','certified','output_schema',
            'supported_backends','quantization','coordinate_space_id'} and identity==digest(supplied)
            and manifest['format']=='dope-shared-model-v1' and manifest['layout']==manifest['graph']['layout']
            and manifest['preprocessing']==PREPROCESSING and manifest['quantization']=='fixed-fp16-v1'
            and manifest['dtype']=='F32' and manifest['certified']is False
            and manifest['supported_backends']==['cpu_reference']
            and manifest['output_schema']=='positional_numeric_and_category_ids'
            and manifest['evidence']in('synthetic_development_only','unqualified_development')
            and manifest['trained_levels']in (['low','medium','high'],['low','medium','high','ultra'])
            and set(manifest['weights'])=={'encoder','decoder'}, 'unknown_model_package')
    config=ModelConfig(**manifest['graph']).validate()
    with torch.device('meta'):estimate=SharedModel(config)
    require(sum(p.numel() for p in estimate.parameters())<=100000000,'model_parameter_envelope')
    return config,manifest


def component(path,name,config,manifest,device):
    from .table_encoder import TableEncoder
    from .decoder import Decoder
    record=manifest['weights'][name]
    require(set(record)=={'file','bytes','sha256','parameters'} and record['file']==name+'.safetensors',
            'invalid_weight_locator')
    file=Path(path)/record['file']
    require(not file.is_symlink() and file.stat().st_size==record['bytes']<=536870912
            and file_sha(file)==record['sha256'],'model_weight_digest_mismatch')
    weights=load_file(str(file),device=device)
    require(all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in weights.values()),
            'model_weight_dtype_or_nonfinite')
    module=(TableEncoder(config) if name=='encoder' else Decoder(config)).to(device).eval()
    module.load_state_dict(weights,strict=True)
    require(sum(t.numel() for t in weights.values())==record['parameters'],'model_parameter_inventory')
    return module


def load_decoder(path,device='cpu'):
    config,manifest=package_manifest(path)
    return component(path,'decoder',config,manifest,device),manifest


def load_package(path,device='cpu'):
    config,manifest=package_manifest(path)
    model=SharedModel(config).to(device).eval()
    model.encoder=component(path,'encoder',config,manifest,device)
    model.decoder=component(path,'decoder',config,manifest,device)
    return model,manifest


@dataclass(frozen=True,repr=False)
class LocalEncoding:
    code: torch.Tensor=field(repr=False)
    schema: dict=field(repr=False)
    model: dict=field(repr=False)
    target_included: bool


@dataclass(frozen=True)
class ExportedCode:
    bytes: bytes=field(repr=False)
    manifest: dict


@torch.no_grad()
def encode_table(model, manifest, context, *, target_included=True, through='high', landmark_features=None):
    require(ModelConfig(**manifest['graph']).validate() == model.config
            and model_layout(manifest) == model.config.layout, 'encoder_model_binding')
    require(through in manifest['trained_levels'],'level_not_trained')
    values,observed=tensors(context,next(model.parameters()).device)
    code=quantized_code(model.encoder(values,observed,context.schema,target_included=target_included,through=through,landmark_features=landmark_features))
    schema=context.schema
    if not target_included:
        schema=dict(schema,columns=schema['columns'][:-1],target_index=None,task=None)
    return LocalEncoding(code.cpu(),schema,manifest,target_included)


def export_encoding(local, selected='all', *, max_bytes=65536):
    selected=levels(selected)
    widths=level_widths(model_layout(local.model))
    width=max(widths[level] for level in selected)
    require(width<=local.code.numel() and all(level in local.model['trained_levels'] for level in selected),
            'requested_view_unavailable')
    validate_schema(local.schema)
    require(type(local.target_included)is bool and local.target_included==(local.schema['target_index']is not None)
            and local.model['evidence']in('synthetic_development_only','unqualified_development'),
            'export_metadata_invalid')
    # Materialize only the admitted buffer before serialization, hashing or logging.
    code=local.code[:width].detach().clone().to(torch.float16).contiguous()
    require(torch.isfinite(code).all().item(),'export_code_nonfinite')
    if 'graph' in local.model and ModelConfig(**local.model['graph']).bounded_code():
        require((code.abs()<=1).all().item(), 'bounded_code_range')
    manifest=dict(format='dope-table-code-v1',representation_kind='table_code',layout=model_layout(local.model),
        preprocessing=PREPROCESSING,coordinate_space_id=local.model['coordinate_space_id'],
        exported_levels=selected,target_included=local.target_included,
        dtype='F16',schema=local.schema,views=[dict(level=level,offset=0,length=widths[level]) for level in selected],
        normalization='raw_generation_code',evidence=local.model['evidence'],certified=False)
    raw=save({'code':code},metadata={'manifest':canonical(manifest).decode()})
    require(type(max_bytes)is int and len(raw)<=max_bytes,'code_payload_budget_exceeded')
    return ExportedCode(raw,manifest)


def read_code(raw, model_manifest):
    layout=model_layout(model_manifest)
    # Byte-buffer metadata is read with the format's bounded JSON prefix, without file access.
    require(8<=len(raw)<=1048576,'code_size_invalid')
    header_length=int.from_bytes(raw[:8],'little')
    require(2<=header_length<=262144 and 8+header_length<=len(raw),'code_header_invalid')
    header=json.loads(raw[8:8+header_length])
    require(set(header)=={'__metadata__','code'} and set(header['__metadata__'])=={'manifest'},'unexpected_code_tensors')
    manifest=json.loads(header['__metadata__']['manifest'])
    require(manifest['format']=='dope-table-code-v1' and manifest['layout']==layout
            and manifest['preprocessing']==PREPROCESSING
            and manifest['coordinate_space_id']==model_manifest['coordinate_space_id']
            and manifest['representation_kind']=='table_code' and manifest['dtype']=='F16'
            and manifest['normalization']=='raw_generation_code' and manifest['certified']is False
            and manifest['evidence']==model_manifest['evidence'], 'code_space_mismatch')
    require(set(manifest)=={'format','representation_kind','layout','preprocessing','coordinate_space_id',
            'exported_levels','target_included','dtype','schema','views','normalization','evidence','certified'},
            'unexpected_code_metadata')
    validate_schema(manifest['schema'],ModelConfig(**model_manifest['graph']) if 'graph'in model_manifest else None)
    require(type(manifest['target_included'])is bool
            and manifest['target_included']==(manifest['schema']['target_index']is not None),'code_target_disclosure_mismatch')
    widths=level_widths(layout)
    selected=levels(','.join(manifest['exported_levels']))
    require(manifest['exported_levels']==selected and all(l in model_manifest['trained_levels'] for l in selected),
            'code_views_invalid')
    width=max(widths[level] for level in selected)
    require(manifest['views']==[dict(level=level,offset=0,length=widths[level]) for level in selected]
            and header['code']['shape']==[width] and header['code']['dtype']=='F16'
            and header['code']['data_offsets']==[0,width*2]
            and len(raw)==8+header_length+width*2,'code_inventory_mismatch')
    code=load(raw)['code']
    require(torch.isfinite(code).all().item(),'nonfinite_code')
    if 'graph' in model_manifest and ModelConfig(**model_manifest['graph']).bounded_code():
        require((code.abs()<=1).all().item(), 'bounded_code_range')
    return code.float(),manifest
