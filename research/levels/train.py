"""Bounded, resumable shared-model optimization with fixed development episodes."""
from dataclasses import asdict
import json
import hashlib
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .checkpoints import load_checkpoint, private_json, save_checkpoint, warm_start
from .contracts import ResearchError, canonical, digest, require
from .data import episode
from .losses import objective
from .model import SharedModel


def make_optimizer(model, config):
    decay, plain = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        (plain if parameter.ndim < 2 or name.endswith('.bias') else decay).append(parameter)
    return torch.optim.AdamW([dict(params=decay,weight_decay=config.weight_decay),
                             dict(params=plain,weight_decay=0.)], lr=config.learning_rate,
                            betas=(.9,.95),eps=1e-8)


def learning_rate(config, step):
    warmup = max(1,round(config.steps*config.warmup_fraction))
    if step < warmup: return config.learning_rate*(step+1)/warmup
    progress = (step-warmup)/max(1,config.steps-warmup-1)
    return config.min_learning_rate+(config.learning_rate-config.min_learning_rate)*(1+math.cos(math.pi*progress))/2


def prepared(corpus, role, cursor, model_config):
    source = corpus.get(role,cursor)
    context, query, identity = episode(**{k:source[k] for k in
        ('rows','kinds','slots','row_ids','task','roles')}, config=model_config,
        cursor=cursor,seed=corpus.seed)
    return context,query,identity,source['landmarks']


def validation(model, corpus, config, step, *, phase='ordinary'):
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    records = []
    model.eval()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed+719)
        with torch.no_grad():
            for cursor in range(6):
                context,query,_,profile = prepared(corpus,'development',cursor,model.config)
                _, raw = objective(model,context,query,profile,config,step,phase=phase,differentiable=False)
                records.append(raw)
    model.train()
    return {level:{key:sum(r[level][key] for r in records)/len(records)
                   for key in ('total','generation','numeric','categorical','missing','transfer_proxy',
                               'driver_proxy','profile','consistency')}
            for level in records[0]}


def append(path, value):
    with Path(path).open('ab') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())


def train(model_config, config, corpus, out, *, source_sha256, resume=None,
          stop_after=None, phase='ordinary', resource_check=None, repair_reason=None, base_checkpoint=None):
    started=time.monotonic()
    model_config.validate();config.validate()
    require(phase != 'ultra-residual' or resume or base_checkpoint, 'ultra_requires_trained_ordinary_base')
    require(not (resume and base_checkpoint), 'resume_and_warm_start_are_exclusive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out = Path(out)
    require(not out.exists(), 'run_output_must_be_new')
    require(config.device != 'cuda' or torch.cuda.is_available(), 'cuda_unavailable')
    require(config.precision != 'bf16' or (config.device == 'cuda' and torch.cuda.is_bf16_supported()),
            'bf16_backend_unsupported')
    out.mkdir(mode=0o700,parents=True)
    os.umask(0o077);torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    if config.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(min(1.,16*2**30/torch.cuda.get_device_properties(0).total_memory))
        torch.backends.cuda.matmul.allow_tf32=False
    with torch.device('meta'):estimate=SharedModel(model_config)
    require(sum(p.numel() for p in estimate.parameters())<=100000000, 'model_parameter_envelope')
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    model = SharedModel(model_config).to(config.device).phase(phase)
    optimizer = make_optimizer(model,config)
    recipe = dict(format='dope-shared-training-v1', model=model_config.manifest(),
                  train=asdict(config),corpus_sha256=corpus.identity,source_sha256=source_sha256,
                  phase=phase, base_checkpoint_sha256=(json.loads((Path(resume)/'manifest.json').read_bytes())['recipe'].get('base_checkpoint_sha256') if resume else None) if base_checkpoint is None else hashlib.sha256((Path(base_checkpoint)/'manifest.json').read_bytes()).hexdigest(), evidence='synthetic_development_only' if corpus.prior_only else 'unqualified_development')
    private_json(out/'recipe.json',recipe)
    state = dict(step=0,cursor=0,episodes=0,processed_cells=0,wall_seconds_spent=0.,
                 optimizer_updates=0,phase=phase,trained_levels=[])
    if base_checkpoint:
        warm_start(base_checkpoint,model,recipe=recipe)
        state['trained_levels']=['low','medium','high']
    if resume:
        state=load_checkpoint(resume,model,optimizer,recipe=recipe)
        parent=Path(resume).parent.parent/'terminal.json'
        if parent.exists():
            terminal=json.loads(parent.read_bytes())
            require(terminal['state'] != 'failed' or bool(repair_reason), 'failed_resume_requires_repair')
            state['wall_seconds_spent']=max(state['wall_seconds_spent'],terminal['wall_seconds_spent'])
        private_json(out/'resume.json',dict(parent=str(Path(resume).resolve()),repair_reason=repair_reason))
    initial_step=state['step'];spent=state['wall_seconds_spent']
    stop_after=config.steps if stop_after is None else min(config.steps,stop_after)
    status='complete';last_saved=None
    try:
        initial=validation(model,corpus,config,state['step'],phase=phase)
        append(out/'metrics.jsonl',dict(step=state['step'],kind='validation',levels=initial,
                                       optimizer_updates=state['optimizer_updates']))
        while state['step'] < stop_after:
            if spent+time.monotonic()-started >= config.wall_seconds:
                status='overbudget';break
            if resource_check and not resource_check():
                status='resource_pause';break
            optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(config,state['step'])
            for group in optimizer.param_groups:group['lr']=lr
            per_level=[];successful=0;attempts=0
            boundary=state.copy()
            cpu_rng=torch.get_rng_state();python_rng=random.getstate();numpy_rng=np.random.get_state()
            cuda_rng=torch.cuda.get_rng_state() if config.device=='cuda' else None
            interrupted=False
            while successful < config.accumulation:
                if spent+time.monotonic()-started >= config.wall_seconds or (resource_check and not resource_check()):
                    status='overbudget' if spent+time.monotonic()-started >= config.wall_seconds else 'resource_pause'
                    append(out/'episodes.jsonl',dict(state='partial_update_discarded',step=state['step'],
                        completed_episodes=successful,attempts=attempts,
                        processed_cells=state['processed_cells']-boundary['processed_cells']))
                    state=boundary;torch.set_rng_state(cpu_rng);random.setstate(python_rng);np.random.set_state(numpy_rng)
                    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
                    optimizer.zero_grad(set_to_none=True);interrupted=True;break
                cursor=state['cursor'];state['cursor']+=1;attempts+=1
                require(attempts <= config.accumulation*4,'too_many_unsupported_episodes')
                try:context,query,identity,profile=prepared(corpus,'train',cursor,model_config)
                except ResearchError as error:
                    append(out/'episodes.jsonl',dict(cursor=cursor,state='unsupported',reason=str(error)))
                    continue
                with torch.autocast(device_type=config.device,dtype=torch.bfloat16,
                                    enabled=config.precision=='bf16'):
                    loss,raw=objective(model,context,query,profile,config,state['step'],phase=phase)
                (loss/config.accumulation).backward()
                per_level.append(raw);successful+=1;state['episodes']+=1
                state['processed_cells']+=(len(context.values)+len(query.values))*len(context.columns)
                append(out/'episodes.jsonl',dict(cursor=cursor,state='complete',identity=identity,
                    context_rows=len(context.values),query_rows=len(query.values),columns=len(context.columns),
                    proxy_rows=min(config.proxy_rows,len(context.values)),
                    encoder_passes=1+sum(config.consistency_weight>0 and len(context.values)>=8 for _ in raw),
                    decoder_passes=len(raw)*(1+int(any(r['proxy_ramp']>0 for r in raw.values())
                        and bool(config.transfer_weight or config.driver_weight))),phase=phase))
            if interrupted:break
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip,error_if_nonfinite=True)
            optimizer.step();state['step']+=1;state['optimizer_updates']+=1
            state['trained_levels']=['low','medium','high'] if phase=='ordinary' else ['low','medium','high','ultra']
            state['wall_seconds_spent']=spent+time.monotonic()-started
            training={level:{key:sum(row[level][key] for row in per_level)/len(per_level)
                             for key in ('total','generation','numeric','categorical','missing','transfer_proxy',
                                         'driver_proxy','profile','consistency')}
                      for level in per_level[0]}
            event=dict(step=state['step'],kind='train',levels=training,gradient_norm=float(gradient),
                       learning_rate=lr,optimizer_updates=state['optimizer_updates'],episodes=state['episodes'],
                       processed_cells=state['processed_cells'],wall_seconds=state['wall_seconds_spent'],
                       cuda_peak_bytes=torch.cuda.max_memory_allocated() if config.device=='cuda' else None)
            append(out/'metrics.jsonl',event)
            if state['step']%config.validate_every==0:
                result=validation(model,corpus,config,state['step'],phase=phase)
                append(out/'metrics.jsonl',dict(step=state['step'],kind='validation',levels=result,
                    optimizer_updates=state['optimizer_updates']))
            if state['step']%config.checkpoint_every==0:
                state['wall_seconds_spent']=spent+time.monotonic()-started
                last_saved=save_checkpoint(out/'checkpoints',model,optimizer,recipe=recipe,state=state.copy())
        if last_saved is None or last_saved.name != f"step-{state['step']:09d}":
            state['wall_seconds_spent']=spent+time.monotonic()-started
            last_saved=save_checkpoint(out/'checkpoints',model,optimizer,recipe=recipe,state=state.copy())
    except (ResearchError,OSError,RuntimeError,ValueError) as error:
        status='failed'
        private_json(out/'failure.json',dict(error_type=type(error).__name__,hint=ResearchError.repair_hint,
                                            docs=ResearchError.docs_url,last_completed_step=state['step']))
        raise
    finally:
        terminal=dict(state=status,step=state['step'],optimizer_updates=state['optimizer_updates'],
            new_updates=state['step']-initial_step,episodes=state['episodes'],processed_cells=state['processed_cells'],
            wall_seconds_spent=spent+time.monotonic()-started,
            last_checkpoint=str(last_saved) if last_saved is not None else None,recipe_sha256=digest(recipe))
        private_json(out/'terminal.json',terminal)
    return model,terminal
