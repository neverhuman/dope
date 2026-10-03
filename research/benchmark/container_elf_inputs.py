"""Inspect digest-bound ELF declarations; never resolve or load a library."""
import hashlib
import re
import struct


def require(condition):
    if not condition:
        raise ValueError('compressed ELF input inspection rejected')


def file_offset(segments, address, size):
    candidates = [offset + address - start for kind, offset, start, count in segments
                  if kind == 1 and start <= address and address + size <= start + count]
    require(len(candidates) == 1)
    return candidates[0]


def inspect_elf_inputs(blob, expected_sha256):
    """Only ELF64 little-endian x86-64 program headers are supported.

    Declaration extraction is not ELF validity, loader resolution, executable
    closure, filesystem custody or research job admission evidence.
    """
    require(type(blob) is bytes and type(expected_sha256) is str)
    require(re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None)
    require(hashlib.sha256(blob).hexdigest() == expected_sha256)
    try:
        require(len(blob) >= 64)
        header = struct.unpack_from('<16sHHIQQQIHHHHHH', blob)
        require(header[0][:7] == b'\x7fELF\x02\x01\x01')
        require(header[1] in (2, 3) and header[2] == 62 and header[3] == 1)
        require(header[8] == 64 and header[9] == 56 and 0 < header[10] < 1024)
        phoff, phnum = header[5], header[10]
        require(phoff >= 64 and phoff + phnum * 56 <= len(blob))
        segments = []
        for index in range(phnum):
            kind, flags, offset, address, _, size, memory, alignment = struct.unpack_from(
                '<IIQQQQQQ', blob, phoff + index * 56)
            require(offset + size <= len(blob) and address + memory <= (1 << 64))
            require(kind != 1 or size <= memory)
            segments.append((kind, offset, address, size))
        interpreters = [s for s in segments if s[0] == 3]
        dynamics = [s for s in segments if s[0] == 2]
        require(len(interpreters) <= 1 and len(dynamics) <= 1)
        interpreter = None
        if interpreters:
            _, offset, _, size = interpreters[0]
            raw = blob[offset:offset + size]
            require(raw.startswith(b'/') and raw.endswith(b'\0') and b'\0' not in raw[:-1])
            interpreter = raw[:-1].decode('utf-8')
        entries = []
        if dynamics:
            _, offset, address, size = dynamics[0]
            require(0 < size <= 4096 * 16 and size % 16 == 0)
            require(file_offset(segments, address, size) == offset)
            raw_entries = [struct.unpack_from('<qQ', blob, i)
                           for i in range(offset, offset + size, 16)]
            require(any(tag == 0 for tag, _ in raw_entries))
            end = next(i for i, (tag, _) in enumerate(raw_entries) if tag == 0)
            require(all(entry == (0, 0) for entry in raw_entries[end:]))
            entries = raw_entries[:end]
        tags = {1: 'needed', 15: 'rpath', 29: 'runpath', 0x6ffffefb: 'depaudit',
                0x6ffffefc: 'audit', 0x7ffffffd: 'auxiliary', 0x7fffffff: 'filter'}
        declarations = {name: [] for name in tags.values()}
        if any(tag in tags for tag, _ in entries):
            addresses = [v for tag, v in entries if tag == 5]
            sizes = [v for tag, v in entries if tag == 10]
            require(len(addresses) == len(sizes) == 1 and sizes[0] > 0)
            address, size = addresses[0], sizes[0]
            offset = file_offset(segments, address, size)
            strings = blob[offset:offset + size]
            require(len(strings) == size)
            for tag, index in entries:
                if tag in tags:
                    require(index < len(strings))
                    end = strings.find(b'\0', index)
                    require(end >= 0)
                    declarations[tags[tag]].append(strings[index:end].decode('utf-8'))
        return {'format': 'dope-compressed-elf-input-declarations', 'version': 1,
                'elf_sha256': expected_sha256, 'interpreter': interpreter,
                'declarations': declarations, 'loader_resolution_verified': False,
                'full_runtime_closure_verified': False, 'execution_admitted': False,
                'candidate_processes_started': 0, 'official_tests_opened': False,
                'mfs_v2': None, 'ptf_v1': None, 'release_safe': None, 'superiority': None}
    except (ValueError, TypeError, struct.error):
        raise ValueError('compressed ELF input inspection rejected') from None
