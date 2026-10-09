#!/usr/bin/env python3
"""Run the declared Lane B paper-reduction graph; never discover experiments."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / 'research/benchmark/review_fixes/paper-build.json'
LOCK = REPO / 'ops/ci/paper-review-inputs.json'
LOCK_SHA256 = 'b89c850dc7f009610b594f0fd3d39fc8fa459b94d00016c0619b75d9fd2bed68'


def authenticated(path, digest, size=None):
    if path.is_symlink():
        raise ValueError('symlinked review paper input')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest or (size is not None and len(raw) != size):
        raise ValueError('review paper input digest or length mismatch')
    return raw


def authenticated_graph():
    lock = json.loads(authenticated(LOCK, LOCK_SHA256))
    if lock.get('format') != 'dope-paper-review-inputs-v1':
        raise ValueError('invalid review paper input lock')
    files = lock['files']
    graph_name = GRAPH.relative_to(REPO).as_posix()
    if graph_name not in files:
        raise ValueError('review graph has no input pin')
    raw_graph = None
    for name, pin in files.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.as_posix() != name
                or relative.suffix != '.json'
                or not name.startswith(('research/benchmark/results/', 'research/benchmark/review_fixes/'))):
            raise ValueError('invalid review paper input path')
        path = REPO / relative
        if path.resolve() != REPO.resolve() / relative:
            raise ValueError('redirected review paper input')
        if type(pin['bytes']) is not int or pin['bytes'] <= 0:
            raise ValueError('invalid review paper input length')
        raw = authenticated(path, pin['sha256'], pin['bytes'])
        if name == graph_name:
            raw_graph = raw
    # Every public dependency is authenticated before decoding the build graph.
    graph = json.loads(raw_graph)
    for entry in graph['scripts']:
        args = entry.get('args', [])
        if len(args) != 2 or args[0] != '--from-panel' or args[1] not in files:
            raise ValueError('review renderer requires one independently pinned public panel')
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not GRAPH.is_file():
        if list((REPO / 'docs/whitepaper/generated').glob('review-*')):
            raise ValueError('review outputs require a declared paper-build graph')
        print('review paper graph has no committed outputs yet')
        return
    graph = authenticated_graph()
    if graph.get('format') != 'dope-review-paper-build-v1':
        raise ValueError('invalid review paper graph')
    declared = set()
    for entry in graph['scripts']:
        script = Path(entry['path'])
        if (script.parent.as_posix() != 'docs/whitepaper/scripts'
                or not script.name.startswith('review_') or script.suffix != '.py'):
            raise ValueError('invalid review reduction script')
        command = [sys.executable, str(REPO / script), *entry.get('args', [])]
        if args.check:
            command.extend(entry['check_args'])
        subprocess.run(command, cwd=REPO, check=True)
        for relative in entry['outputs']:
            path = Path(relative)
            if (path.parent.as_posix() not in ('docs/whitepaper/generated', 'docs/whitepaper/figures')
                    or not path.name.startswith('review-')):
                raise ValueError('invalid review output path')
            if not (REPO / path).is_file():
                raise ValueError('missing declared review output')
            declared.add(path.as_posix())
    actual = {p.relative_to(REPO).as_posix() for directory in ('generated', 'figures')
              for p in (REPO / 'docs/whitepaper' / directory).glob('review-*') if p.is_file()}
    if actual != declared:
        raise ValueError('review build graph does not cover every committed output')


if __name__ == '__main__':
    main()
