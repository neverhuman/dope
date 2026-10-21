#!/bin/sh
set -eu
case "$(uname -s):$(uname -m)" in Linux:x86_64) ;; *) echo native_control_platform_unsupported >&2; exit 3 ;; esac
package=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH= cd -- "$package/../.." && pwd)
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
base="-C force-unwind-tables=no -C relocation-model=static -C target-feature=-crt-static -C link-arg=-nostartfiles -C link-arg=-static -C link-arg=-no-pie -C link-arg=-Wl,--build-id=none -C link-arg=-Wl,--gc-sections -C link-arg=-Wl,--no-eh-frame-hdr -C link-arg=-Wl,-z,noseparate-code -C link-arg=-Wl,-z,norelro -C link-arg=-Wl,-T,$package/linker.ld"
export RUSTFLAGS="$base -C link-arg=-Wl,--defsym=__ieee754_exp=__ieee754_exp_fma -C link-arg=/usr/lib/x86_64-linux-gnu/libm-2.39.a"
"${CARGO:-cargo}" build --offline --locked --release --target x86_64-unknown-linux-gnu --manifest-path "$package/Cargo.toml" --target-dir "$repo/target" --bin dope-native-control
export DOPE_NATIVE_CONTROL_IMAGE="$repo/target/native-control-packed-runtime.tdl"
python3 -B - "$repo" "$DOPE_NATIVE_CONTROL_IMAGE" <<'PACK'
import gzip,sys,struct,os
from pathlib import Path
source=Path(sys.argv[1])/'target/x86_64-unknown-linux-gnu/release/dope-native-control'
raw=source.read_bytes()
if not 64 <= len(raw) <= 65536: raise SystemExit('native_control_inner_size_refused')
payload=bytearray(gzip.compress(raw,compresslevel=9,mtime=0));payload[9]=255
name=b'literal_payload';frame=b'TDL1'+struct.pack('<II',1,len(name))+name+struct.pack('<Q',len(payload))+payload
p=Path(sys.argv[2]);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.partial')
with tmp.open('xb') as f:f.write(frame);f.flush();os.fsync(f.fileno())
os.replace(tmp,p)
PACK
export RUSTFLAGS="$base -C link-arg=-Wl,-Ttext-segment=0x10000000"
exec "${CARGO:-cargo}" build --offline --locked --release --target x86_64-unknown-linux-gnu --manifest-path "$package/Cargo.toml" --target-dir "$repo/target" --bin dope-native-control-runtime
