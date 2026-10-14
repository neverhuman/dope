#!/usr/bin/env python3
"""Freeze existing local legal inputs for the locked default Linux native package.

No fetch, acceptance or dependency update is performed. The normal/build graph
is a conservative package notice inventory, not a signed release qualification.
"""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
NOTICE = ROOT / "scripts/native-third-party.NOTICE"
PINS = ROOT / "scripts/native-legal.sha256"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def render():
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    locked = {(row["name"], row["version"]): row for row in lock["package"]}
    tree = subprocess.check_output([
        "cargo", "tree", "--locked", "--offline", "--target", "x86_64-unknown-linux-gnu",
        "-e", "normal,build", "--prefix", "none", "--format", "{p}",
    ], cwd=ROOT, stderr=subprocess.DEVNULL, text=True)
    packages = sorted(set(re.findall(r"^([\w-]+) v([^\s]+)", tree, re.M))
                      - {("dope-kernel", tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"])})
    cache = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))) / "registry/src"
    payload = [b"DOPE locked default Linux x86-64 native dependency notices\n",
               b"Inventory: normal/build Cargo packages and installed Rust library notices.\n",
               b"No GPU/research adapters or separately provisioned OS runtime are distributed here.\n",
               b"License texts retain upstream bytes; this inventory does not qualify a release.\n",
               f"Cargo.lock sha256 {digest((ROOT / 'Cargo.lock').read_bytes())}\n".encode()]
    for name, version in packages:
        if (name, version) not in locked:
            raise ValueError("native_notice_lock_mismatch")
        roots = list(cache.glob("*/" + name + "-" + version))
        if len(roots) != 1:
            raise ValueError("native_notice_cache_unavailable")
        source = roots[0]
        package_archive = cache.parent / "cache" / source.parent.name / f"{name}-{version}.crate"
        if digest(package_archive.read_bytes()) != locked[name, version].get("checksum"):
            raise ValueError("native_notice_registry_package_mismatch")
        legal = sorted(path for path in source.rglob("*") if path.is_file()
                       and not path.is_symlink()
                       and path.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT")))
        if not legal:
            raise ValueError("native_notice_license_source_unavailable")
        payload.append(f"\n===== {name} {version} =====\n".encode())
        with tarfile.open(package_archive, "r:gz") as archive:
            for path in legal:
                content = path.read_bytes()
                member = archive.getmember(f"{name}-{version}/{path.relative_to(source)}")
                if not member.isfile():
                    raise ValueError("native_notice_registry_license_mismatch")
                with archive.extractfile(member) as stream:
                    if stream.read() != content:
                        raise ValueError("native_notice_registry_license_mismatch")
                payload += [f"--- {path.relative_to(source)} sha256 {digest(content)} ---\n".encode(), content, b"\n"]
    sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip())
    copyright_file = sysroot / "share/doc/rust/COPYRIGHT-library.html"
    if not copyright_file.is_file():
        raise ValueError("native_notice_rust_library_source_unavailable")
    library_legal = [copyright_file] + sorted((sysroot / "share/doc/rust/licenses").glob("*.txt"))
    payload.append(b"\n===== Rust library upstream notice and license inventory =====\n")
    for path in library_legal:
        content = path.read_bytes()
        payload += [f"--- {path.relative_to(sysroot / 'share/doc/rust')} sha256 {digest(content)} ---\n".encode(), content, b"\n"]
    notice = b"".join(payload)
    pins = "".join(f"{digest(content)}  {name}\n" for name, content in [
        ("LICENSE", (ROOT / "LICENSE").read_bytes()),
        ("scripts/native-third-party.NOTICE", notice),
        ("Cargo.lock", (ROOT / "Cargo.lock").read_bytes()),
        ("rust-toolchain.toml", (ROOT / "rust-toolchain.toml").read_bytes()),
    ]).encode()
    return notice, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        notice, pins = render()
        if args.check:
            if NOTICE.read_bytes() != notice or PINS.read_bytes() != pins:
                raise ValueError("native_notice_frozen_inputs_drift")
        else:
            NOTICE.write_bytes(notice)
            PINS.write_bytes(pins)
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError):
        sys.exit("native_notice_inputs_unavailable_or_drift")


if __name__ == "__main__":
    main()
