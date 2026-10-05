"""Run frozen research source buffers after an externally pinned startup check."""
import hashlib
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    require(type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value),
            'external research digest required')


def regular(path):
    require(path.is_absolute() and path == path.resolve(strict=True)
            and not any(p.is_symlink() for p in (path, *path.parents))
            and stat.S_ISREG(path.lstat().st_mode), 'research source path invalid')


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate research declaration')
        result[key] = value
    return result


def decode(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_float=finite_float,
                      parse_constant=lambda _: require(False, 'nonfinite research declaration'))


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), 'nonfinite research declaration')
    return result


def checked_buffers(source, expected):
    """Return the exact hashed buffers; do not import or execute a source yet."""
    digest(expected)
    source = Path(source)
    regular(source)
    root = source.parents[1]
    require(source.parent.name == 'source' and 'evaluator' not in root.parts,
            'research source tree invalid')
    round_path = root / 'round.lock.json'
    regular(round_path)
    data = round_path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'research round changed')
    lock = decode(data)
    execution_path = root / 'execution.lock.json'
    regular(execution_path)
    require(decode(execution_path.read_bytes())['round_sha256'] == expected,
            'research execution identity differs')
    require(lock['official_tests_opened'] is False and lock['full_campaign_admitted'] is False
            and all(lock[k] is None for k in ('mfs_v2', 'ptf_v1', 'release_safe', 'superiority')),
            'research startup claim differs')
    files = lock['source_files']
    require(type(files) is dict and str(source) in files and str(source.with_name('common.py')) in files
            and {str(p) for p in source.parent.iterdir()} == set(files),
            'research source membership differs')
    buffers = {}
    for name, pin in files.items():
        digest(pin)
        path = Path(name)
        require(path.parent == source.parent, 'research source outside flat inventory')
        regular(path)
        code = path.read_bytes()
        require(hashlib.sha256(code).hexdigest() == pin, 'research source changed')
        buffers[name] = code
    return buffers


class CheckedSourceLoader(importlib.machinery.SourceFileLoader):
    """Use only the owned code object; never fall back to source or bytecode files."""
    def __init__(self, name, path, body):
        super().__init__(name, str(path))
        require(type(body) is bytes, 'checked research source bytes required')
        self.code = compile(body, str(path), 'exec', dont_inherit=True, optimize=0)

    def get_code(self, fullname):
        require(type(fullname) is str and fullname == self.name,
                'checked research module identity differs')
        return self.code

    def module(self):
        spec = importlib.util.spec_from_file_location(self.name, self.path, loader=self)
        return importlib.util.module_from_spec(spec)


def run(source, expected, arguments):
    buffers = checked_buffers(source, expected)
    source = Path(source)
    common_path = source.with_name('common.py')
    # Compile both selected modules before either initializer can run.
    common_loader = CheckedSourceLoader('common', common_path, buffers[str(common_path)])
    entry_loader = CheckedSourceLoader('__main__', source, buffers[str(source)])
    common = common_loader.module()
    common_loader.exec_module(common)
    sys.modules['common'] = common
    common.check(expected)
    sys.argv = [str(source), *arguments]
    entry_loader.exec_module(entry_loader.module())


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
