"""ABI2 standalone static native sampler: no CPython/DSO in the child closure.

The trusted parent still compiles and authenticates provenance. This module pins
and seals the executable, pays every native runtime/invocation byte, and projects
the retained raw native attestation into the historical typed receipt vocabulary.
"""
import fcntl
import json
import os
from pathlib import Path
import platform
import struct
import time

ABI = 2
SCOPE = 'standalone-native-control-v2'
ISOLATION = 'standalone_native_landlock_seccomp_tsync_v2'
WIRE = struct.Struct('<8sIIQQQ32s32s')
ATTEST = struct.Struct('<8sQQQIIQ32s32s32s')
SOURCE_NAMES = ('main.rs','sys.rs','decode.rs','sample.rs','hash.rs','host_math.rs','inflate.rs','loader.rs','Cargo.toml','Cargo.lock','linker.ld')


def _boundary():
    from . import control_boundary
    return control_boundary


def static_elf(raw):
    b = _boundary()
    b.require(len(raw)>=64 and raw[:7]==b'\x7fELF\x02\x01\x01')
    kind,machine=struct.unpack_from('<HH',raw,16)
    b.require(kind==2 and machine==62 and b'DOPECTRL_ABI2_V1\0' in raw)
    offset=struct.unpack_from('<Q',raw,32)[0]
    size,count=struct.unpack_from('<HH',raw,54)
    b.require(size==56 and 0<count<=32 and offset+size*count<=len(raw))
    loads=[]
    for i in range(count):
        ptype,flags,at,vaddr,_,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',raw,offset+i*size)
        # PT_INTERP and PT_DYNAMIC both refused. No undeclared shared libraries,
        # static PIE relocation loader, Python, or product DSO can be hidden.
        b.require(ptype not in (2,3) and at+filesz<=len(raw) and filesz<=memsz)
        if ptype==1:
            b.require(not(flags&1 and flags&2));loads.append((vaddr,memsz))
    b.require(bool(loads) and sum(n for _,n in loads)<=64<<20)
    return True


def verify_runtime(runtime, *, read_sources=True):
    b=_boundary(); r=runtime
    b.require(platform.system()=='Linux' and platform.machine()=='x86_64')
    b.require(set(r)=={'scope','abi','executable','executable_sha256','source_dir','source_pins','closure_pins','build_receipt','build_receipt_sha256'}
        and r['scope']==SCOPE and type(r['abi']) is int and r['abi']==ABI)
    b.require(set(r['source_pins'])==set(SOURCE_NAMES))
    for expected in r['source_pins'].values():
        b.require(type(expected)is str and len(expected)==64 and all(x in '0123456789abcdef' for x in expected))
    # A genuinely static ELF is the complete file-backed sampler runtime. Rust
    # sources are provenance, not interpreted runtime files. Their code is paid
    # in the whole ELF. No file used by the child is omitted from this inventory.
    b.require(r['closure_pins']=={r['executable']:r['executable_sha256']})
    executable=b.regular_bytes(r['executable']);b.require(b.digest(executable)==r['executable_sha256'])
    static_elf(executable)
    build_raw=b.regular_bytes(r['build_receipt'],b.MAX_ARTIFACT)
    b.require(b.digest(build_raw)==r['build_receipt_sha256'])
    build=json.loads(build_raw)
    b.require(build['status']=='built_source_unchanged' and build['returncode']==0 and
        build['source_before']==build['source_after']==r['source_pins'] and
        build['executable_sha256']==r['executable_sha256'] and build['executable_bytes']==len(executable))
    if read_sources:
        for name,expected in r['source_pins'].items():
            relative='src/'+name if name.endswith('.rs') else name
            b.require(b.digest(b.regular_bytes(Path(r['source_dir'])/relative))==expected)
    return executable


def request(runtime, artifact_sha256, artifact_bytes, method, rows, seed, files):
    b=_boundary()
    b.require(type(rows)is int and 0<rows<=b.MAX_VALUES and type(seed)is int and 0<=seed<1<<64
              and method in ('compressed_real_rows','gaussian_copula'))
    closure=[{'sha256':b.digest(raw),'bytes':len(raw)} for _,raw in sorted(files)]
    b.require(dict((path,b.digest(raw)) for path,raw in files)==runtime['closure_pins'])
    return {'method':method,'rows':rows,'seed':seed,'abi':ABI,'artifact_sha256':artifact_sha256,
        'artifact_bytes':artifact_bytes,'library_sha256':runtime['executable_sha256'],
        'source_sha256':b.digest(b.canonical(runtime['source_pins'])),'source_pins':runtime['source_pins'],
        'runtime_closure_sha256':b.digest(b.canonical(closure)),
        'runtime_bytes':sum(v['bytes'] for v in closure),'closure_pins':runtime['closure_pins']}


def invocation_bytes(req):
    return len(_boundary().canonical(req))+WIRE.size


def wire(req):
    b=_boundary()
    return WIRE.pack(b'DNC2REQ\0',1 if req['method']=='compressed_real_rows' else 2,0,
        req['rows'],req['seed'],req['artifact_bytes'],bytes.fromhex(req['artifact_sha256']),
        bytes.fromhex(b.digest(b.canonical(req))))


def attest(raw, native_receipt, req):
    b=_boundary();b.require(type(native_receipt)is bytes and len(native_receipt)==ATTEST.size)
    magic,rows,columns,seed,method,denied,length,artifact,request_sha,sample_sha=ATTEST.unpack(native_receipt)
    b.require(magic==b'DNC2OUT\0' and rows==req['rows'] and seed==req['seed'] and
        method=={'compressed_real_rows':1,'gaussian_copula':2}[req['method']] and denied==0x3f)
    b.require(1<=columns<=64 and rows*columns<=b.MAX_VALUES and length==len(raw)==rows*columns*8)
    b.require(artifact.hex()==req['artifact_sha256'] and request_sha.hex()==b.digest(b.canonical(req))
              and sample_sha.hex()==b.digest(raw))
    return columns


def sample(artifact, artifact_sha256, method, rows, seed, *, trusted_runtime, workdir,
           wall_seconds, byte_budget, capture_result=None, capture_started=None):
    b=_boundary()
    b.require(type(rows)is int and 0<rows<=b.MAX_VALUES and type(seed)is int and 0<=seed<1<<64
              and type(method)is str and method in ('compressed_real_rows','gaussian_copula'))
    b.require(type(wall_seconds)in(int,float) and 0<wall_seconds<float('inf') and
        type(byte_budget)is int and 0<byte_budget<=1<<30)
    started_at=time.monotonic();deadline=started_at+wall_seconds
    b.require((capture_result is None or callable(capture_result)) and
              (capture_started is None or callable(capture_started)))
    binary=verify_runtime(trusted_runtime)
    artifact_raw=b.regular_bytes(artifact,b.MAX_ARTIFACT)
    b.require(b.digest(artifact_raw)==artifact_sha256)
    files=((trusted_runtime['executable'],binary),)
    req=request(trusted_runtime,artifact_sha256,len(artifact_raw),method,rows,seed,files)
    charge=req['artifact_bytes']+req['runtime_bytes']+invocation_bytes(req)
    # Check before workdir creation, memfd executable staging or Popen.
    b.require(charge<=byte_budget)
    work=Path(workdir)
    b.require(work.is_absolute() and '..' not in work.parts and not work.exists()
              and work.parent.resolve(strict=True)==work.parent)
    work.mkdir(mode=0o700);bundle=work/'bundle';runtime=work/'runtime'
    bundle.mkdir(mode=0o700);runtime.mkdir(mode=0o700)
    b.private_file(bundle/'control.tdl',artifact_raw)
    b.private_file(bundle/'request.bin',wire(req));b.private_file(runtime/'request.json',b.canonical(req))
    def started(pid):
        b.private_file(runtime/'native.started.json',b.canonical({'pid':pid,'abi':ABI}))
        if capture_started is not None: capture_started(pid)
    def interrupted(code,raw,stderr):
        b.private_file(runtime/'native.stderr.bin',stderr)
        b.private_file(runtime/'native.stdout.bin',raw)
        b.private_file(runtime/'native.exit.json',b.canonical({'returncode':code,'interrupted':True}))
        b.private_file(runtime/'native.cost.json',b.canonical({
            'elapsed_wall_ms':max(1,int((time.monotonic()-started_at)*1000+1)),
            'process_peak_memory_bytes':None,'reason':'bounded_child_interrupted','retry':False}))
        if capture_result is not None: capture_result(code,raw,stderr)
    fd=os.memfd_create('dope-native-control',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        position=0
        while position<len(binary):
            n=os.write(fd,binary[position:]);b.require(n>0);position+=n
        os.fchmod(fd,0o500)
        seals=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
        fcntl.fcntl(fd,fcntl.F_ADD_SEALS,seals)
        b.require(fcntl.fcntl(fd,fcntl.F_GET_SEALS)&seals==seals)
        remaining=deadline-time.monotonic();b.require(remaining>0)
        code,raw,native=b.bounded_child(['/proc/self/fd/'+str(fd)],bundle,remaining,min(b.MAX_VALUES,rows*64)*8,
            pass_fds=(fd,),on_started=started,on_interrupted=interrupted)
    finally:
        os.close(fd)
    # Keep the ACTUAL binary child's stdout, binary stderr, exit and request,
    # including failures. Never replace its attestation with a JSON success flag.
    b.private_file(runtime/'native.stderr.bin',native)
    b.private_file(runtime/'native.stdout.bin',raw)
    b.private_file(runtime/'native.exit.json',b.canonical({'returncode':code}))
    b.private_file(runtime/'native.cost.json',b.canonical({
        'elapsed_wall_ms':max(1,int((time.monotonic()-started_at)*1000+1)),
        'process_peak_memory_bytes':None,'sampling_closure_bytes':charge,'retry':False}))
    if code!=0:
        if capture_result is not None: capture_result(code,raw,native)
        b.require(False)
    columns=attest(raw,native,req)
    receipt={k:req[k] for k in ('method','rows','seed','abi','artifact_sha256','artifact_bytes',
        'library_sha256','source_sha256','runtime_closure_sha256','runtime_bytes')}
    receipt.update(state='sampled',format='f64le',layout='row_major',columns=columns,
        column_order='features_then_target',sample_bytes=len(raw),sample_sha256=b.digest(raw),
        isolation=ISOLATION,invocation_bytes=invocation_bytes(req),sampling_closure_bytes=charge,
        pipeline_complete=False,scientific_admission=False,citable=False,native_attestation_hex=native.hex())
    public=b.canonical(receipt)+b'\n'
    if capture_result is not None: capture_result(code,raw,public)
    b.require(b.digest(b.regular_bytes(trusted_runtime['executable']))==trusted_runtime['executable_sha256'])
    return b.validate_output(code,raw,public,req)
