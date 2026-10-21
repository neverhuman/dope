"""Explicit fixed-runtime inventories for Linux conformance workers."""
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

from .common import MetricError, require


def native_runtime_paths(executables):
    """Inventory trusted local executables/DSOs and their dynamic dependencies."""
    paths = {Path(path).resolve(strict=True) for path in executables}
    try:
        linked = subprocess.run(["ldd", *map(str, sorted(paths))], capture_output=True, text=True,
                                env={"LC_ALL": "C", "PATH": os.defpath}, check=True)
        require("not found" not in linked.stdout, "runtime_inventory_unavailable")
        for line in linked.stdout.splitlines():
            match = re.search(r"(?:=>\s+|^\s*)(/[^\s]+)\s+\(", line)
            if match:
                paths.add(Path(match[1]).resolve(strict=True))
        cache = Path("/etc/ld.so.cache")
        if cache.is_file():
            paths.add(cache.resolve())
        return paths
    except (OSError, subprocess.CalledProcessError):
        raise MetricError("runtime_inventory_unavailable") from None


def python_runtime_paths(*, include_bytecode=True):
    # Only an explicitly source-only child may omit ambient bytecode. -B alone
    # prevents writes but still reads existing caches; the caller must also
    # force pycache_prefix to a non-directory before interpreter startup.
    # A conservative paid stdlib inventory permits lazy imports and native
    # extension loads without treating the interpreter as an uncharged oracle.
    require(type(include_bytecode) is bool, "invalid_runtime_inventory_mode")
    stdlib = Path(sysconfig.get_path("stdlib"))
    files = set()
    for directory, children, names in os.walk(stdlib):
        children[:] = [name for name in children if name not in {"site-packages", "dist-packages"}
                       and (include_bytecode or name != "__pycache__")]
        files.update(path.resolve() for name in names if (path := Path(directory) / name).is_file()
                     and (include_bytecode or path.suffix not in {".pyc", ".pyo"}))
    runtime = Path(__file__).resolve().parent
    files.update(runtime / name for name in ("literal_worker.py", "common.py"))
    cache = runtime / "__pycache__"
    if include_bytecode and cache.is_dir():
        files.update(path.resolve() for path in cache.iterdir()
                     if path.name.startswith(("literal_worker.", "common.")) and path.is_file())
    files.update(native_runtime_paths([sys.executable, *(path for path in files if path.suffix == ".so")]))
    if Path("/etc/localtime").is_file():
        files.add(Path("/etc/localtime").resolve())
    executable = Path(sys.executable).absolute()
    for config in (executable.parent / "pyvenv.cfg", executable.parent.parent / "pyvenv.cfg"):
        if config.is_file():
            files.add(config.resolve())
    return files


def closure_inventory(paths):
    """Return charged bytes and opaque IDs; machine paths stay local."""
    entries, inventory = [], {}
    try:
        for index, path in enumerate(sorted({Path(path).resolve(strict=True) for path in paths})):
            name = f"runtime_{index:05d}"
            entries.append({"name": name, "hex": path.read_bytes().hex(),
                            "scope": "runtime", "dataset_specific": False})
            inventory[name] = path
    except OSError:
        raise MetricError("runtime_inventory_unavailable") from None
    return entries, inventory
