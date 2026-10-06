"""Check frozen reusable TabSyn providers before third-party initialization.

A separate code manifest binds this lane's controls without changing the
previously inventoried Python, packages, author source, or control directory.
The caller supplies both manifest digests from an external launch declaration.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    require(type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'TabSyn digest differs')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe(path):
    p = Path(path)
    require(p.is_absolute() and p == p.resolve(strict=True)
            and not any(q.is_symlink() for q in (p, *p.parents)), 'TabSyn path is not owned')
    return p


def bound(path, expected):
    digest(expected)
    p = safe(path)
    require(stat.S_ISREG(p.stat().st_mode), 'TabSyn manifest is not regular')
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'TabSyn manifest changed')
    return json.loads(raw)


def verify_inventory(inv):
    root = safe(inv['root'])
    require(root.is_dir(), 'TabSyn provider root differs')
    files, aliases, directories = set(), {}, set()
    for p in root.rglob('*'):
        name = str(p.relative_to(root))
        mode = p.lstat().st_mode
        if stat.S_ISLNK(mode):
            aliases[name] = dict(target=os.readlink(p), resolved=str(p.resolve(strict=True)))
        elif stat.S_ISDIR(mode):
            directories.add(name)
        elif stat.S_ISREG(mode):
            files.add(name)
        else:
            require(False, 'TabSyn special provider differs')
    require(files == set(inv['files']) and aliases == inv['aliases']
            and directories == set(inv['directories']), 'TabSyn provider inventory changed')
    for name, row in inv['files'].items():
        digest(row['sha256'])
        p = root/name
        require(p.is_relative_to(root) and safe(p) == p and type(row['bytes']) is int
                and row['bytes'] >= 0 and p.stat().st_size == row['bytes']
                and sha(p) == row['sha256'], 'TabSyn provider changed')


def verify(path, expected, code_path, code_expected):
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode,
            'TabSyn isolated interpreter required')
    lock = bound(path, expected)
    require(lock['format'] == 'dope-tabsyn-owned-runtime' and type(lock['version']) is int
            and lock['version'] == 1 and lock['official_tests_opened'] is False,
            'TabSyn runtime declaration differs')
    base = safe(lock['base'])
    require(safe(path) == base/'runtime.lock.json'
            and os.environ.get('LD_LIBRARY_PATH') == lock['LD_LIBRARY_PATH']
            and not os.environ.get('LD_PRELOAD'), 'TabSyn loader environment differs')
    python = safe(lock['python'])
    require(Path(sys.executable).resolve(strict=True) == python
            and sha(python) == lock['python_sha256'], 'TabSyn interpreter changed')
    prefix = python.parents[1]
    require(lock['stdlib'] == str(prefix/'lib/python3.10')
            and lock['dynload'] == str(prefix/'lib/python3.10/lib-dynload')
            and not os.path.lexists(prefix/'lib/python310.zip'), 'TabSyn interpreter path differs')
    cache = safe(sys.pycache_prefix)
    require(cache.is_dir() and not list(cache.rglob('*')), 'TabSyn empty bytecode cache required')
    require({i['root'] for i in lock['roots']} ==
            {str(base/n) for n in ['python', 'venv', 'source', 'control']},
            'TabSyn runtime roots differ')
    for inv in lock['roots']:
        verify_inventory(inv)
    for name, row in lock['system_files'].items():
        p = safe(name)
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
                'TabSyn system provider changed')
    code = bound(code_path, code_expected)
    require(code['format'] == 'dope-tabsyn-adapter-controls' and type(code['version']) is int
            and code['version'] == 1 and code['runtime_sha256'] == expected
            and code['source_commit'] == 'cb5ac0f74ec36ee88e7a974a393dfbef50d42da7'
            and code['execution_admitted'] is False, 'TabSyn control declaration differs')
    verify_inventory(code['inventory'])
    control = safe(code['inventory']['root'])
    own = safe(__file__)
    require(own == control/'research/benchmark/tabsyn_runtime_guard.py'
            and code['inventory']['files'][str(own.relative_to(control))]['sha256'] == sha(own),
            'TabSyn executing guard differs')
    adapter = sys.modules.get('research.benchmark.tabsyn_adapter')
    if adapter is not None:
        require(safe(adapter.__file__) == control/'research/benchmark/tabsyn_adapter.py',
                'TabSyn executing adapter differs')
    source = next(i for i in lock['roots'] if i['root'] == lock['source'])
    require({name: r['sha256'] for name, r in source['files'].items()} == code['author_files'],
            'TabSyn author source differs')
    require(lock['source'] == str(base/'source') and lock['control'] == str(base/'control')
            and lock['package_site'] == str(base/'venv/lib/python3.10/site-packages'),
            'TabSyn import paths differ')
    sys.path[:] = [str(control), lock['source'], lock['package_site'], lock['stdlib'], lock['dynload']]
    return lock
