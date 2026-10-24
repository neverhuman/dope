"""Generate tiny synthetic operator-parity fixtures, not a production model."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from .checkpoints import private_json, save_checkpoint
from .cli import source_fingerprint
from .contracts import ModelConfig, TrainConfig, digest, LAYOUT_V2, V2_GRAPHS, LANDMARK_GRAPHS, level_widths
from .data import episode
from .losses import objective
from .model import SharedModel,tensors
from .packages import encode_table,export_encoding,export_package
from .priors import table
from .train import make_optimizer


def generate(output, *, graph="mixed-code-diffusion-v1"):
    output=Path(output)
    config=ModelConfig(width=4,heads=1,feature_blocks=1,inducing_blocks=1,inducing_slots=1,
        summary_slots=1,target_blocks=1,decoder_blocks=1,condition_slots=1,max_features=8,
        max_context=16,max_query=8,max_categories=8,cell_cap=128)
    if graph != 'mixed-code-diffusion-v1':
        config=replace(config,layout=LAYOUT_V2,graph=graph,summary_slots=4).validate()
    train_config=TrainConfig(steps=1,accumulation=1,validate_every=1,checkpoint_every=1,
        transfer_weight=0,driver_weight=0,profile_weight=0,consistency_weight=0,cpu_threads=1)
    torch.set_num_threads(1);torch.manual_seed(20261023)
    source=table(0,123)
    context,query,_=episode(**{k:source[k] for k in ('rows','kinds','slots','row_ids','task','roles')},
                            config=config,cursor=0,seed=12)
    model=SharedModel(config)
    landmark_features=([.1*(i+1) for i in range(23)],[True]*23) if config.landmark_input() else None
    if config.landmark_recipe()=='fit-cv-gain23-observed-mask-squash-v2':
        landmark_features=([-1.,0.,.005]+[.1]*(23-3),[True]*23)
    profile=landmark_features if landmark_features is not None else ([0.]*23,[False]*23)
    input_args=dict(landmark_features=landmark_features) if landmark_features is not None else {}
    optimizer=make_optimizer(model,train_config)
    loss,_=objective(model,context,query,profile,train_config,0)
    loss.backward();optimizer.step();optimizer.zero_grad()
    if graph != 'mixed-code-diffusion-v1':
        model.phase('ultra-residual');optimizer=make_optimizer(model,train_config)
        loss,_=objective(model,context,query,profile,train_config,0,phase='ultra-residual')
        loss.backward();optimizer.step();optimizer.zero_grad()
    recipe=dict(model=config.manifest(),source_sha256=source_fingerprint(),
                evidence='synthetic_development_only')
    checkpoint=save_checkpoint(output/'private-checkpoint',model,optimizer,recipe=recipe,
        state=dict(step=1 if graph=='mixed-code-diffusion-v1' else 2,
                   optimizer_updates=1 if graph=='mixed-code-diffusion-v1' else 2,trained_levels=['low','medium','high'] if graph=='mixed-code-diffusion-v1'
                   else ['low','medium','high','ultra']))
    manifest=export_package(checkpoint,output/'model')
    values,observed=tensors(context,'cpu')
    local=encode_table(model,manifest,context,**input_args)
    target_free=encode_table(model,manifest,context,target_included=False,**input_args)
    raw=export_encoding(local,'low').bytes
    (output/'low.code').write_bytes(raw)
    torch.manual_seed(91827)
    noisy=torch.randn((3,len(context.columns)))
    for j,column in enumerate(context.columns):
        if column.kind=='categorical':noisy[:,j]=0
    present=torch.tensor([[1.]*len(context.columns)]*3,dtype=torch.bool)
    with torch.no_grad():
        numeric,categorical,missing=model.decoder(local.code[:level_widths(config.layout)['low']],context.schema,noisy,present,.37)
    expected=dict(values=values,observed=observed.float(),noisy=noisy,noisy_observed=present.float(),
        code_high=local.code,code_target_free=target_free.code,
        numeric=numeric,categorical=categorical,missing=missing)
    if landmark_features is not None:
        from .landmark_features import feature_tensor
        expected['landmark_inputs']=feature_tensor(landmark_features,'cpu')
    if graph != 'mixed-code-diffusion-v1':
        full=encode_table(model,manifest,context,through='ultra',**input_args)
        for level,width in level_widths(config.layout).items():
            (output/(level+'.code')).write_bytes(export_encoding(full,level).bytes)
            with torch.no_grad():
                n,c,m=model.decoder(full.code[:width],context.schema,noisy,present,.37)
            expected.update({f'code_{level}':full.code[:width],f'numeric_{level}':n,
                             f'categorical_{level}':c,f'missing_{level}':m})
    save_file({k:v.detach().clone().contiguous() for k,v in expected.items()},str(output/'parity.safetensors'))
    private_json(output/'fixture.json',dict(format='dope-learned-parity-fixture-v1',schema=context.schema,
        time=.37,generator='research/levels/fixtures.py',seed=20261023,
        purpose='tiny synthetic graph conformance, not a trained production model',
        code_absolute_tolerance=.002,operator_absolute_tolerance=2e-4,
        runtime=dict(torch=torch.__version__)))
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--graph',choices=['mixed-code-diffusion-v1',*V2_GRAPHS,*LANDMARK_GRAPHS],
                        default='mixed-code-diffusion-v1')
    args=parser.parse_args();generate(args.out,graph=args.graph)
