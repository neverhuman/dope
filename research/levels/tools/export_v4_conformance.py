"""Generate immutable public synthetic v4 producer/sampler conformance fixtures.

No optimizer or real source is used. Existing output directories are refused.
Run as `python3 -B -m research.levels.tools.export_v4_conformance --out NEW_DIR`.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..v4_model import NativeLandmarks,V4Model
from ..v4_objectives import sample_differentiable
from ..v4_encode import export_packet


def generate(out):
    out=Path(out)
    if out.exists():raise ValueError("conformance_output_exists")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model=V4Model().double()
    t=torch.linspace(.02,.98,24,dtype=torch.float64)
    x=torch.stack([t,torch.remainder(t*2.3,.97)],dim=1)
    schema={"schema":"dope.nc1-public/v1","recipe":"nc1-minmax-onehot-v1","columns":2,
        "groups":[{"id":i,"kind":"numeric","start":i,"width":1,"has_missing":False} for i in range(2)],"target":"regression","classes":0}
    draws=torch.tensor([[.11,.21,.31,.41,.51],[.72,.13,.44,.25,.96],[.23,.87,.65,.43,.21]],dtype=torch.float64)
    inventory=[]
    for task in ("regression","multiclass"):
        view={**schema,"target":task,"classes":0 if task=="regression" else 3}
        y=.2+.4*t+.1*torch.sin(2*torch.pi*x[:,1]) if task=="regression" else (torch.arange(24)%3).double()
        state=model.encode(x,y,view,NativeLandmarks.none("a"*64,dtype=torch.float64))
        packet,wire=export_packet(model,state,level="16k")
        sampled=sample_differentiable(state,draws)
        files={task+".dml4":wire,task+".packet.json":json.dumps(packet,indent=2).encode()+b"\n",
               task+".draws.json":json.dumps(draws.tolist(),indent=2).encode()+b"\n",
               task+".samples.json":json.dumps({"x":sampled["x"].detach().tolist(),"y":sampled["y"].detach().tolist()},indent=2).encode()+b"\n"}
        for name,data in files.items():
            (out/name).write_bytes(data)
            inventory.append({"file":name,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
    (out/"manifest.json").write_text(json.dumps({"kind":"public_synthetic_conformance_only","producer_recipe_sha256":model.recipe_sha256(),
        "weights_sha256":model.weights_sha256(),"torch_version":torch.__version__,"optimizer_updates":0,
        "native_landmarks":"explicit_none","real_data_reads":0,
        "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                          for name in ("v4_model.py","v4_objectives.py","v4_encode.py","v4_wire_reference.py")},
        "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"files":inventory},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",required=True)
    generate(parser.parse_args().out)
