#!/usr/bin/env python3
"""Run the declared Lane B paper-reduction graph; never discover experiments."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / 'research/benchmark/review_fixes/paper-build.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not GRAPH.is_file():
        if list((REPO / 'docs/whitepaper/generated').glob('review-*')):
            raise ValueError('review outputs require a declared paper-build graph')
        print('review paper graph has no committed outputs yet')
        return
    graph = json.loads(GRAPH.read_text())
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
