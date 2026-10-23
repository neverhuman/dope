"""Generate tiny synthetic operator-parity fixtures, not a production model."""
from dataclasses import replace
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from .checkpoints import private_json, save_checkpoint
from .cli import source_fingerprint
from .contracts import ModelConfig, TrainConfig, digest
from .data import episode
from .losses import objective
from .model import SharedModel,tensors
from .packages import encode_table,export_encoding,export_package
from .priors import table
from .train import make_optimizer


def generate(output):
    output=Path(output)
    config=ModelConfig(width=4,heads=1,feature_blocks=1,inducing_blocks=1,inducing_slots=1,
        summary_slots=1,target_blocks=1,decoder_blocks=1,condition_slots=1,max_features=8,
        max_context=16,max_query=8,max_categories=8,cell_cap=128)
    train_config=TrainConfig(steps=1,accumulation=1,validate_every=1,checkpoint_every=1,
        transfer_weight=0,driver_weight=0,profile_weight=0,consistency_weight=0,cpu_threads=1)
    torch.set_num_threads(1);torch.manual_seed(20261023)
    source=table(0,123)
    context,query,_=episode(**{k:source[k] for k in ('rows','kinds','slots','row_ids','task','roles')},
                            config=config,cursor=0,seed=12)
    model=SharedModel(config)
    optimizer=make_optimizer(model,train_config)
    loss,_=objective(model,context,query,([0.]*23,[False]*23),train_config,0)
    loss.backward();optimizer.step();optimizer.zero_grad()
    recipe=dict(model=config.manifest(),source_sha256=source_fingerprint(),
                evidence='synthetic_development_only')
    checkpoint=save_checkpoint(output/'private-checkpoint',model,optimizer,recipe=recipe,
        state=dict(step=1,optimizer_updates=1,trained_levels=['low','medium','high']))
    manifest=export_package(checkpoint,output/'model')
    values,observed=tensors(context,'cpu')
    local=encode_table(model,manifest,context)
    target_free=encode_table(model,manifest,context,target_included=False)
    raw=export_encoding(local,'low').bytes
    (output/'low.code').write_bytes(raw)
    torch.manual_seed(91827)
    noisy=torch.randn((3,len(context.columns)))
    for j,column in enumerate(context.columns):
        if column.kind=='categorical':noisy[:,j]=0
    present=torch.tensor([[1.]*len(context.columns)]*3,dtype=torch.bool)
    with torch.no_grad():
        numeric,categorical,missing=model.decoder(local.code[:256],context.schema,noisy,present,.37)
    expected=dict(values=values,observed=observed.float(),noisy=noisy,noisy_observed=present.float(),
        code_high=local.code,code_target_free=target_free.code,
        numeric=numeric,categorical=categorical,missing=missing)
    save_file({k:v.detach().contiguous() for k,v in expected.items()},str(output/'parity.safetensors'))
    private_json(output/'fixture.json',dict(format='dope-learned-parity-fixture-v1',schema=context.schema,
        time=.37,generator='research/levels/fixtures.py',seed=20261023,
        purpose='tiny synthetic graph conformance, not a trained production model',
        code_absolute_tolerance=.002,operator_absolute_tolerance=2e-4,
        runtime=dict(torch=torch.__version__)))
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    generate(parser.parse_args().out)
