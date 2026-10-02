"""Compare cached executable code with pinned source without executing either."""

import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
import sys
import types


def canonical(value):
    if isinstance(value, types.CodeType):
        return {name: canonical(getattr(value, name)) for name in dir(value)
                if name.startswith('co_') and not callable(getattr(value, name))}
    if isinstance(value, tuple):
        return ('tuple', tuple(canonical(item) for item in value))
    if isinstance(value, frozenset):
        return ('frozenset', frozenset(canonical(item) for item in value))
    if type(value) in (type(None), bool, int, float, complex, str, bytes, type(Ellipsis)):
        return (type(value).__name__, value)
    raise ValueError('unsupported cache constant')


def verify(entries):
    results = []
    for row in entries:
        path = Path(row['path'])
        if path.suffix != '.pyc' or path.is_symlink():
            raise ValueError('unsupported executable cache')
        source = Path(importlib.util.source_from_cache(str(path)))
        if source.is_symlink() or str(source) != row['source_path']:
            raise ValueError('cache source identity changed')
        data, original = path.read_bytes(), source.read_bytes()
        if (hashlib.sha256(data).hexdigest() != row['sha256']
                or hashlib.sha256(original).hexdigest() != row['source_sha256']
                or data[:4] != importlib.util.MAGIC_NUMBER or len(data) < 16):
            raise ValueError('cache or pinned source changed')
        try:
            cached = marshal.loads(data[16:])
            if not isinstance(cached, types.CodeType):
                raise ValueError('cache is not executable code')
            optimization = int(path.stem.rsplit('.opt-', 1)[1]) if '.opt-' in path.stem else 0
            if optimization not in (0, 1, 2):
                raise ValueError('unsupported cache optimization')
            compiled = compile(original, cached.co_filename, 'exec', dont_inherit=True,
                               optimize=optimization)
            if canonical(cached) != canonical(compiled):
                raise ValueError('cache differs from pinned source')
        except (EOFError, TypeError, SyntaxError, OverflowError, ValueError):
            raise ValueError('cache differs from pinned source') from None
        results.append({'path': str(path), 'sha256': row['sha256'],
                        'source_path': str(source), 'source_sha256': row['source_sha256'],
                        'code_equivalent': True})
    return results


if __name__ == '__main__':
    try:
        result = {'status': 'ok', 'caches': verify(json.load(sys.stdin))}
    except (KeyError, OSError, TypeError, ValueError):
        result = {'status': 'rejected', 'reason': 'executable cache verification failed'}
    print(json.dumps(result, sort_keys=True))
    sys.exit(0 if result['status'] == 'ok' else 1)
