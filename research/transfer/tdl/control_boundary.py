"""Pinned trusted-runtime control sampling; identity is not an operation grant."""
import ctypes
import fcntl
import hashlib
import json
import math
import struct
import os
from pathlib import Path
import platform
import selectors
import stat
import subprocess
import time

ABI = 1
SCOPE = "dedicated-control-worker-v1"
MAX_FILE = 512 << 20
MAX_ARTIFACT = 16 << 20
MAX_VALUES = 1_000_000
SOURCES = ("control_boundary.py", "control_isolated_worker.py", "literal_worker.py", "common.py")


class BoundaryError(ValueError):
    """Only static failure codes are public."""


def require(ok):
    if not ok:
        raise BoundaryError("control_boundary_refused")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def regular_bytes(path, limit=MAX_FILE):
    """Walk every component without symlinks; nonblocking final read and race check."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts)
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit)
            chunks, size = [], 0
            while size <= limit:
                chunk = os.read(fd, min(1 << 20, limit + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            after = os.fstat(fd)
            require(size == before.st_size <= limit and
                    (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns))
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def load_boundary(path, expected_sha256, expected_abi):
    """Child only. dlopen an immutable sealed copy, never an artifact-selected DSO."""
    require(platform.system() == "Linux" and platform.machine() == "x86_64" and type(expected_abi) is int and expected_abi == ABI)
    raw = regular_bytes(path)
    require(bool(raw) and digest(raw) == expected_sha256)
    fd = os.memfd_create("dope-control-runtime", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
            stream.flush()
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        require(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals == seals)
        library = ctypes.CDLL(f"/proc/self/fd/{fd}", mode=os.RTLD_LOCAL | os.RTLD_NOW)
        for name in ("version", "confine_current_bundle_directory", "seal_sampling_process"):
            symbol = "dope_control_isolation_version" if name == "version" else "dope_control_" + name
            function = getattr(library, symbol)
            function.argtypes = []
            function.restype = ctypes.c_uint32 if name == "version" else ctypes.c_int
            setattr(library, name, function)
        require(library.version() == expected_abi)
        return library
    finally:
        os.close(fd)


def private_file(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def bounded_child(argv, cwd, wall_seconds, output_limit):
    """Bound private pipes and terminate only this newly created child session."""
    with subprocess.Popen(argv, cwd=cwd, env={"LC_ALL": "C", "PYTHONHASHSEED": "0"},
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True, close_fds=True) as child:
        streams = {child.stdout: bytearray(), child.stderr: bytearray()}
        try:
            with selectors.DefaultSelector() as poll:
                for stream in streams:
                    poll.register(stream, selectors.EVENT_READ)
                deadline = time.monotonic() + wall_seconds
                while poll.get_map():
                    remaining = deadline - time.monotonic()
                    require(remaining > 0)
                    for key, _ in poll.select(min(remaining, 0.1)):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            poll.unregister(key.fileobj)
                        else:
                            streams[key.fileobj].extend(chunk)
                            cap = output_limit if key.fileobj is child.stdout else 65536
                            require(len(streams[key.fileobj]) <= cap)
                code = child.wait(timeout=max(0.001, deadline - time.monotonic()))
            return code, bytes(streams[child.stdout]), bytes(streams[child.stderr])
        finally:
            if child.poll() is None:
                import signal
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def validate_output(code, raw, public, request):
    require(code == 0)
    receipt = json.loads(public)
    for key in ("method", "rows", "seed", "artifact_sha256", "library_sha256", "abi",
                "source_sha256", "runtime_closure_sha256", "artifact_bytes", "runtime_bytes"):
        require(type(receipt[key]) is type(request[key]) and receipt[key] == request[key])
    columns = receipt["columns"]
    require(type(columns) is int and 1 <= columns <= 64 and request["rows"] * columns <= MAX_VALUES)
    require(len(raw) == request["rows"] * columns * 8 == receipt["sample_bytes"])
    require(all(math.isfinite(value) for (value,) in struct.iter_unpack("<d", raw)))
    require(receipt["scientific_admission"] is False and receipt["citable"] is False)
    require(receipt["sample_sha256"] == digest(raw) and receipt["state"] == "sampled"
            and receipt["format"] == "f64le" and receipt["layout"] == "row_major"
            and receipt["column_order"] == "features_then_target"
            and receipt["isolation"] == "existing_rust_landlock_seccomp_tsync"
            and receipt["pipeline_complete"] is False)
    require(receipt["invocation_bytes"] == len(canonical(request)))
    require(receipt["sampling_closure_bytes"] == request["artifact_bytes"] +
            request["runtime_bytes"] + receipt["invocation_bytes"])
    return raw, receipt


def sample_control(artifact, artifact_sha256, method, rows, seed, *, trusted_runtime,
                   workdir, wall_seconds):
    """Caller grants trusted-runtime scope separately; never derive it from artifact bytes.

    trusted_runtime: scope, abi, library, library_sha256, python, source_dir,
    source_pins (four fixed filenames), closure_pins (absolute canonical file paths).
    workdir must be a fresh private child directory. External job admission owns RAM/CPU.
    """
    try:
        require(trusted_runtime["scope"] == SCOPE and type(trusted_runtime["abi"]) is int
                and trusted_runtime["abi"] == ABI)
        require(method in ("gaussian_copula", "compressed_real_rows"))
        require(type(rows) is int and 0 < rows <= MAX_VALUES // 64)
        require(type(seed) is int and 0 <= seed < 1 << 64)
        require(type(wall_seconds) in (int, float) and 0 < wall_seconds < float("inf"))
        rt = trusted_runtime
        require(set(rt["source_pins"]) == set(SOURCES))
        library = regular_bytes(rt["library"])
        require(bool(library) and digest(library) == rt["library_sha256"])
        source = {name: regular_bytes(Path(rt["source_dir"]) / name, MAX_ARTIFACT) for name in SOURCES}
        require(all(digest(raw) == rt["source_pins"][name] for name, raw in source.items()))
        pins = rt["closure_pins"]
        require(pins[rt["library"]] == rt["library_sha256"] and rt["python"] in pins)
        for name in SOURCES:
            require(pins[str(Path(rt["source_dir"]) / name)] == rt["source_pins"][name])
        closure = []
        for path, expected in sorted(pins.items()):
            raw = regular_bytes(path)
            require(digest(raw) == expected)
            closure.append({"sha256": expected, "bytes": len(raw)})
        raw_artifact = regular_bytes(artifact, MAX_ARTIFACT)
        require(bool(raw_artifact) and digest(raw_artifact) == artifact_sha256)
        work = Path(workdir)
        require(work.is_absolute() and ".." not in work.parts and not work.exists()
                and work.parent.resolve(strict=True) == work.parent)
        work.mkdir(mode=0o700)
        bundle, runtime = work / "bundle", work / "runtime"
        bundle.mkdir(mode=0o700)
        runtime.mkdir(mode=0o700)
        private_file(bundle / "control.tdl", raw_artifact)
        private_file(runtime / "boundary.so", library)
        for name, raw in source.items():
            private_file(runtime / name, raw)
        request = {"method": method, "rows": rows, "seed": seed, "abi": ABI,
                   "artifact_sha256": artifact_sha256, "artifact_bytes": len(raw_artifact),
                   "library_sha256": rt["library_sha256"], "source_sha256": digest(canonical(rt["source_pins"])),
                   "source_pins": rt["source_pins"], "runtime_closure_sha256": digest(canonical(closure)),
                   "runtime_bytes": sum(entry["bytes"] for entry in closure), "closure_pins": pins}
        private_file(runtime / "request.json", canonical(request))
        result = bounded_child([rt["python"], "-I", "-S", "-B", str(runtime / "control_isolated_worker.py")],
                               bundle, wall_seconds, rows * 64 * 8)
        # Detect persistent changes around the child; trusted runtime is not a hostile-writer sandbox.
        require(all(digest(regular_bytes(p)) == h for p, h in pins.items()))
        require(all(digest(regular_bytes(runtime / n)) == rt["source_pins"][n] for n in SOURCES))
        return validate_output(*result, request)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.SubprocessError):
        raise BoundaryError("control_boundary_refused") from None
