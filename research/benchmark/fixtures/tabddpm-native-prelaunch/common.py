"""Standard-library custody and capacity helpers for a bounded research round."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
HOSTS = ('xbabe1', 'xbabe2', 'xbabe3')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def once(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def safe(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve().is_relative_to(BASE)
    assert 'evaluator' not in path.parts and 'evaluator' not in path.resolve().parts
    assert path.name != 'test.csv' and not path.is_symlink()
    assert not any(p.is_symlink() for p in path.parents)
    return path

def check(expected_round=None):
    execution = read(safe(ROOT/'execution.lock.json'))
    h = sha(safe(ROOT/'round.lock.json'))
    assert h == execution['round_sha256'] and (expected_round is None or h == expected_round)
    lock = read(ROOT/'round.lock.json')
    assert lock['official_tests_opened'] is False and lock['mfs_v2'] is None and lock['ptf_v1'] is None
    for path, expected in lock['source_files'].items(): assert sha(safe(path)) == expected
    assert sha(safe(ROOT/'runtime-inventory.lock.json')) == lock['runtime_inventory_sha256']
    assert sha(lock['ssh_binary']) == lock['ssh_binary_sha256']
    assert sha(safe(lock['known_hosts'])) == lock['known_hosts_sha256']
    assert sha(lock['du_binary']) == lock['du_binary_sha256']
    return lock

def verify_runtime(lock):
    inventory = read(ROOT/'runtime-inventory.lock.json')
    aliases = {r['path']: r for r in inventory['aliases']}
    assert len(aliases) == len(inventory['aliases'])
    roots = [Path(r) for r in inventory['roots']]
    seen = set()
    for root in roots:
        safe(root)
        for p in root.rglob('*'):
            if p.is_symlink():
                r = aliases.get(str(p)); assert r is not None
                assert str(p.readlink()) == r['symlink_target'] and str(p.resolve(strict=True)) == r['resolved_path']
                seen.add(str(p))
    assert seen == set(aliases)
    files = inventory['files']
    for name, row in files.items():
        p = Path(name)
        if p.is_symlink():
            assert name in aliases
        else: safe(p)
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    for key, mode in inventory['inventory_modes'].items():
        root = Path(key)
        relevant = lambda p: mode == 'all' or p.suffix in ('.py','.pyc','.pyo','.so') or '.so.' in p.name
        actual = {str(p) for p in root.rglob('*') if p.is_file() and relevant(p)}
        expected = {n for n in files if Path(n).is_relative_to(root) and relevant(Path(n))}
        assert actual == expected
    return inventory

def check_worker(job):
    worker = safe(job['worker']['path'])
    assert not (worker/'test.csv').exists()
    assert set(job['worker']['files']) == {'worker-manifest.json','projection.json','train.csv','validation.csv'}
    for name, h in job['worker']['files'].items(): assert sha(safe(worker/name)) == h

def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def ssh(lock, host, argv, **kwargs):
    assert host in HOSTS and host != 'xbabe2'
    return subprocess.run([lock['ssh_binary'], '-o','BatchMode=yes','-o','ConnectTimeout=8',
        '-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+host,
        '-o','UserKnownHostsFile='+lock['known_hosts'],host,*argv], **kwargs)

def owner_snapshot(lock):
    """Include non-Python parents when grouping verified study descendants."""
    known = lock['recognized_owner_sources']; processes = {}; candidates = {}; study = set(); unknown = set()
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid = int(p.parent.name); args = [a.decode() for a in p.read_bytes().split(b'\0') if a]
            parent = int(next(x.split()[1] for x in (p.parent/'status').read_text().splitlines() if x.startswith('PPid:')))
            uid = p.parent.stat().st_uid
        except (OSError, ValueError, UnicodeError, StopIteration): continue
        processes[pid] = parent
        if not args or 'python' not in args[0] or pid == os.getpid(): continue
        if not (any(str(BASE) in a and a.endswith('.py') for a in args[1:]) or any(a.startswith('research.benchmark.') for a in args[1:])): continue
        study.add(pid); matches = [n for n in known if n in args]
        if len(matches) == 1:
            n = matches[0]
            try: valid = uid == os.getuid() and sha(n) == known[n]['sha256']
            except OSError: valid = False
            if valid: candidates[pid] = known[n]['role']
    roots = {}
    for pid, role in candidates.items():
        ancestor = processes.get(pid); seen = {pid}
        while ancestor in processes and ancestor not in seen:
            if ancestor in candidates: break
            seen.add(ancestor); ancestor = processes.get(ancestor)
        else: roots[pid] = role; continue
        if ancestor not in candidates: roots[pid] = role
    for pid in study:
        ancestor = pid; seen = set()
        while ancestor in processes and ancestor not in seen and ancestor not in roots:
            seen.add(ancestor); ancestor = processes[ancestor]
        if ancestor not in roots: unknown.add(pid)
    inventory = load_file('frozen_inventory',ROOT/'source/inventory_hosts.py').local()
    assert inventory['host'].split('.')[0] in HOSTS
    return {'host':socket.gethostname().split('.')[0],'roots':roots,'unknown_study_pids':sorted(unknown),
            'inventory':inventory,'utc':datetime.now(timezone.utc).isoformat()}

if __name__ == '__main__':
    lock = check(sys.argv[1]); print(json.dumps(owner_snapshot(lock)))
