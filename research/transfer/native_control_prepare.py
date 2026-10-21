"""Build and pin the actual native test runtime once before transfer discovery."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def main():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        # The fixture suite explicitly labels this unsupported native route.
        return
    parent = ROOT / 'target/native-control-preparation'
    parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='build-', dir=parent))
    subprocess.run([sys.executable, '-B', str(ROOT/'research/transfer/native_control_build.py'),
                    '--out', str(attempt/'build')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, '-B', str(ROOT/'research/transfer/native_control_manifest.py'),
                    '--binary', str(ROOT/'target/x86_64-unknown-linux-gnu/release/dope-native-control-runtime'),
                    '--build-receipt', str(attempt/'build/RESULT.json'),
                    '--out', str(attempt/'runtime.json')], check=True, cwd=ROOT)
    raw = (attempt/'runtime.json').read_bytes()
    # Failed builds preserve evidence and never publish an old source manifest.
    json.loads(raw)
    pointer = ROOT/'target/native-control-runtime.json'
    staging = attempt/'verified-runtime.json'
    with staging.open('xb') as out:
        os.chmod(staging, 0o600)
        out.write(raw); out.flush(); os.fsync(out.fileno())
    os.replace(staging, pointer)

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError):
        raise SystemExit('native_control_preparation_refused') from None
