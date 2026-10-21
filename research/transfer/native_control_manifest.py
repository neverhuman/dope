"""Root invokes after actual build; pins paid static ELF plus compiler provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--build-receipt',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];package=root/"ops/native_control"
    sys.path.insert(0,str(root))
    from research.transfer.tdl import control_boundary as b
    from research.transfer.tdl.control_native import ABI,SCOPE,SOURCE_NAMES,verify_runtime
    executable=Path(args.binary).resolve(strict=True)
    pins={name:b.digest(b.regular_bytes(package/('src/'+name if name.endswith('.rs') else name))) for name in SOURCE_NAMES}
    runtime={'scope':SCOPE,'abi':ABI,'executable':str(executable),'executable_sha256':b.digest(b.regular_bytes(executable)),
        'source_dir':str(package),'source_pins':pins,'closure_pins':{str(executable):b.digest(b.regular_bytes(executable))},
        'build_receipt':str(Path(args.build_receipt).resolve(strict=True)),
        'build_receipt_sha256':b.digest(b.regular_bytes(Path(args.build_receipt).resolve(strict=True)))}
    raw=verify_runtime(runtime)
    result={'runtime':runtime,'actual_paid_elf_bytes':len(raw),'dynamic_dependencies':[],
        'budget_evaluation':'artifact + complete ELF + canonical JSON request +104-byte wire; evaluate per draw',
        'scientific_admission':False,'size_under_32768_not_assumed':True}
    target=Path(args.out).resolve()
    b.private_file(target,b.canonical(result))

if __name__=='__main__':
    try:main()
    except (OSError, ValueError, KeyError, TypeError):raise SystemExit('native_control_manifest_refused') from None
