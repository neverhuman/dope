"""Atomic, tensor-only checkpoints. No pickle, code objects or optimizer deserialization."""
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from .contracts import canonical, require
from .corpus import file_sha


def private_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name("."+uuid.uuid4().hex+".partial")
    fd = os.open(tmp, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        if replace: os.replace(tmp,path)
        else: os.link(tmp,path)
    finally: tmp.unlink(missing_ok=True)


def save_checkpoint(root, model, optimizer, *, recipe, state):
    root = Path(root)
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    checkpoint_name = f"step-{state['step']:09d}"
    destination = root/checkpoint_name
    require(not destination.exists(), "checkpoint_already_exists")
    staging = root/("."+checkpoint_name+"-"+uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    tensors = {"model."+n:t.detach().cpu().contiguous() for n,t in model.state_dict().items()}
    names = {parameter:name for name,parameter in model.named_parameters()}
    optimizer_metadata = {}
    for parameter, entry in optimizer.state.items():
        name = names[parameter]
        for key, value in entry.items():
            require(isinstance(value,torch.Tensor), "unknown_optimizer_state")
            tensors["optimizer."+name+"."+key] = value.detach().cpu().contiguous()
        optimizer_metadata[name] = sorted(entry)
    groups = []
    for group in optimizer.param_groups:
        groups.append({key:([names[p] for p in value] if key == "params" else value)
                       for key,value in group.items()})
    tensors['rng.cpu'] = torch.get_rng_state()
    device = next(model.parameters()).device
    if device.type == 'cuda': tensors['rng.cuda.0'] = torch.cuda.get_rng_state(device)
    numpy = np.random.get_state()
    metadata = dict(format="dope-research-checkpoint-v1", recipe=recipe, state=state,
                    optimizer_entries=optimizer_metadata, optimizer_groups=groups,
                    python_rng=random.getstate(), numpy_rng=[numpy[0],numpy[1].tolist(),*numpy[2:]],
                    torch_version=torch.__version__, cuda_version=torch.version.cuda, backend=device.type)
    try:
        save_file(tensors,str(staging/'state.safetensors'))
        os.chmod(staging/'state.safetensors',0o600)
        metadata['tensor_sha256'] = file_sha(staging/'state.safetensors')
        private_json(staging/'manifest.json',metadata)
        for path in staging.iterdir():
            with path.open('rb') as stream: os.fsync(stream.fileno())
        os.rename(staging,destination)
        fd = os.open(root,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        private_json(root/'latest.json',dict(checkpoint=checkpoint_name,manifest_sha256=file_sha(destination/'manifest.json')),
                     replace=True)
    except (OSError,ValueError,RuntimeError):
        # Keep a failed partial directory for diagnosis; never publish it through latest.
        raise
    return destination


def load_checkpoint(path, model, optimizer, *, recipe):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "invalid_checkpoint_path")
    raw = (path/'manifest.json').read_bytes()
    require(len(raw) <= 4000000, "checkpoint_manifest_too_large")
    metadata = json.loads(raw)
    require(metadata['format'] == 'dope-research-checkpoint-v1'
            and metadata['recipe'] == recipe and metadata['torch_version'] == torch.__version__
            and metadata['cuda_version'] == torch.version.cuda
            and metadata['backend'] == next(model.parameters()).device.type, "resume_recipe_or_runtime_mismatch")
    require(not (path/'state.safetensors').is_symlink()
            and file_sha(path/'state.safetensors') == metadata['tensor_sha256'], "checkpoint_digest_mismatch")
    tensors = load_file(str(path/'state.safetensors'),device='cpu')
    weights = {key[6:]:value for key,value in tensors.items() if key.startswith('model.')}
    model.load_state_dict(weights,strict=True)
    parameters = dict(model.named_parameters())
    current_groups = [[name for name,p in model.named_parameters() if any(p is q for q in group['params'])]
                      for group in optimizer.param_groups]
    for current,group in zip(current_groups,metadata['optimizer_groups']):
        require(set(current) == set(group['params']), "resume_optimizer_group_mismatch")
    require(len(current_groups) == len(metadata['optimizer_groups']), "resume_optimizer_group_count")
    for group,saved in zip(optimizer.param_groups,metadata['optimizer_groups']):
        group.update({key:value for key,value in saved.items() if key != 'params'})
    optimizer.state.clear()
    used = {'model.'+name for name in weights} | {'rng.cpu'}
    for name,entries in metadata['optimizer_entries'].items():
        require(name in parameters and set(entries) == {'step','exp_avg','exp_avg_sq'}, "resume_optimizer_schema")
        parameter = parameters[name]
        optimizer.state[parameter] = {}
        for key in entries:
            tensor_name = 'optimizer.'+name+'.'+key
            value = tensors[tensor_name]
            require(key == 'step' or value.shape == parameter.shape, "resume_optimizer_shape")
            optimizer.state[parameter][key] = value.to(parameter.device) if key != 'step' else value.clone()
            used.add(tensor_name)
    def tuples(value): return tuple(tuples(v) for v in value) if isinstance(value,list) else value
    random.setstate(tuples(metadata['python_rng']))
    numpy = metadata['numpy_rng'];np.random.set_state((numpy[0],np.asarray(numpy[1],dtype=np.uint32),*numpy[2:]))
    torch.set_rng_state(tensors['rng.cpu'])
    cuda = sorted(k for k in tensors if k.startswith('rng.cuda.'))
    require(len(cuda) == (1 if next(model.parameters()).device.type == 'cuda' else 0), "resume_device_count_mismatch")
    if cuda: torch.cuda.set_rng_state(tensors[cuda[0]], next(model.parameters()).device)
    used.update(cuda)
    require(set(tensors) == used, "unexpected_checkpoint_tensors")
    return metadata['state']


def warm_start(path, model, *, recipe):
    """Start the residual stage from verified ordinary weights, with a new optimizer/RNG."""
    path=Path(path)
    require(path.is_dir() and not path.is_symlink(), 'invalid_warm_start')
    metadata=json.loads((path/'manifest.json').read_bytes())
    require(metadata['format']=='dope-research-checkpoint-v1'
            and metadata['recipe']['phase']=='ordinary'
            and metadata['state']['optimizer_updates']>0
            and metadata['state']['trained_levels']==['low','medium','high']
            and all(metadata['recipe'][k]==recipe[k] for k in
                    ('model','corpus_sha256','source_sha256','evidence')), 'warm_start_recipe_mismatch')
    require(not (path/'state.safetensors').is_symlink()
            and file_sha(path/'state.safetensors')==metadata['tensor_sha256'], 'warm_start_digest_mismatch')
    weights=load_file(str(path/'state.safetensors'),device='cpu')
    model.load_state_dict({k[6:]:v for k,v in weights.items() if k.startswith('model.')},strict=True)
