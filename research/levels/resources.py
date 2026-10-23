"""Read-only local allocation checks; never stop another process or alter host services."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from .contracts import require


def snapshot(root, *, cuda=False):
    memory={}
    if Path('/proc/meminfo').exists():
        memory={line.split(':')[0]:int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines()}
    result=dict(unix=time.time(),pid=os.getpid(),host=os.uname().nodename,os=os.uname().sysname,
                cpu_available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count(),
                load=os.getloadavg(),memory_available=memory.get('MemAvailable'),
                disk_available=shutil.disk_usage(root).free,gpus=[],gpu_processes=[])
    if cuda:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        result['gpus']=[line.split(', ') for line in gpu.stdout.splitlines() if line]
        result['gpu_processes']=[line.split(', ') for line in processes.stdout.splitlines() if line]
    return result


def admissible(value, *, cuda=False):
    common=value['disk_available']>=10*2**30 and (value['memory_available']is None or value['memory_available']>=8*2**30)
    if not cuda:return common
    # This launcher owns one visible GPU. Multi-GPU/DDP needs a separate admission protocol.
    gpus=value['gpus'];visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    if visible not in ('0',''):return False
    return (common and bool(gpus) and float(gpus[0][2])>=20*1024
            and not any(p[0]==gpus[0][1] and int(p[1])!=os.getpid() for p in value['gpu_processes']))


class Guard:
    def __init__(self, root, output, *, cuda=False):
        self.root,self.output,self.cuda=Path(root),Path(output),cuda
        self.next_check=0.;self.last=None

    def __call__(self):
        if time.monotonic()<self.next_check:return self.last
        value=snapshot(self.root,cuda=self.cuda)
        self.last=admissible(value,cuda=self.cuda);self.next_check=time.monotonic()+10
        if self.output.parent.exists():
            with self.output.open('a') as stream:stream.write(json.dumps(dict(value,admitted=self.last))+'\n')
        return self.last

    def admit(self):
        require(self(), 'competing_gpu_or_resource_allocation_refused')
