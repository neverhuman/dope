"""Root-only admitted offline build with exact before/after source evidence.

This program was authored, not run, by the source task. It neither grants its
own resources nor runs tests. Use the existing Root finite CPU guardian.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/"ops/native_control"
SOURCE_NAMES=('main.rs','sys.rs','decode.rs','sample.rs','hash.rs','host_math.rs','inflate.rs','loader.rs','Cargo.toml','Cargo.lock','linker.ld')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def snapshot():
    return {n:sha(PACKAGE/('src/'+n if n.endswith('.rs') else n)) for n in SOURCE_NAMES}

def write(path,doc):
    data=json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as out:
        out.write(data);out.flush();os.fsync(out.fileno())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(mode=0o700)
    before=snapshot()
    lock=tomllib.loads((PACKAGE/'Cargo.lock').read_text())
    cache=Path(os.environ.get('CARGO_HOME',str(Path.home()/'.cargo')))/'registry/cache'
    archives={}
    for item in lock['package']:
        if 'checksum' not in item:continue
        found=list(cache.glob('*/'+item['name']+'-'+item['version']+'.crate'))
        if len(found)!=1 or sha(found[0])!=item['checksum']:raise ValueError('offline_archive_missing_or_changed')
        archives[item['name']]={'version':item['version'],'path':str(found[0]),'sha256':sha(found[0])}
    tools={}
    for name in ('rustc','cargo'):
        tool=subprocess.check_output(['rustup','which',name],text=True).strip()
        tools[name]={'path':tool,'sha256':sha(tool),'version':subprocess.check_output([tool,'--version'],text=True).strip()}
    math_archive=Path('/usr/lib/x86_64-linux-gnu/libm-2.39.a')
    archives['host_glibc_math']={'path':str(math_archive),'sha256':sha(math_archive)}
    argv=['/bin/sh',str(PACKAGE/'build-offline.sh')]
    start={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_pins':before,
           'build_script_sha256':sha(PACKAGE/'build-offline.sh'),'argv':argv,'tools':tools,'offline_archives':archives}
    write(out/'START.json',start)
    with open(out/'stdout.private','xb') as stdout,open(out/'stderr.private','xb') as stderr:
        os.chmod(stdout.name,0o600);os.chmod(stderr.name,0o600)
        env=dict(os.environ)
        for key in ('RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','CARGO_ENCODED_RUSTFLAGS'):
            env.pop(key,None)
        env.update(RUSTC=tools['rustc']['path'],CARGO=tools['cargo']['path'])
        result=subprocess.run(argv,cwd=ROOT,stdout=stdout,stderr=stderr,check=False,env=env)
    after=snapshot();binary=ROOT/'target/x86_64-unknown-linux-gnu/release/dope-native-control-runtime'
    status='built_source_unchanged' if result.returncode==0 and before==after and binary.is_file() and start['build_script_sha256']==sha(PACKAGE/'build-offline.sh') and all(sha(t['path'])==t['sha256'] for t in tools.values()) else 'refused'
    receipt={'status':status,'returncode':result.returncode,'source_before':before,'source_after':after,
        'start_sha256':sha(out/'START.json'),'executable':str(binary),
        'executable_sha256':sha(binary) if status=='built_source_unchanged' else None,
        'executable_bytes':binary.stat().st_size if status=='built_source_unchanged' else None,
        'stdout_sha256':sha(out/'stdout.private'),'stderr_sha256':sha(out/'stderr.private'),
        'build_script_sha256':sha(PACKAGE/'build-offline.sh'),'tests_ran':False,'size_gate_passed':False,
        'inner_executable_sha256':sha(ROOT/'target/x86_64-unknown-linux-gnu/release/dope-native-control') if status=='built_source_unchanged' else None,
        'encoded_inner_sha256':sha(ROOT/'target/native-control-packed-runtime.tdl') if status=='built_source_unchanged' else None,
        'scientific_admission':False}
    write(out/'RESULT.json',receipt)
    if status!='built_source_unchanged':raise ValueError('native_build_refused')

if __name__=='__main__':
    try:main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):raise SystemExit('native_control_build_refused') from None
