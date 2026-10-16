#!/usr/bin/env python3
"""Check historical evidence bytes before CI can rewrite or re-pin them.

frozen-evidence.json records SHA256 hashes from the last pre-rebuild commit.
Its files map is immutable. Exceptions go in allowlist as exact repository paths
with {"sha256": "<approved replacement hash>", "reason": "<review rationale>"}.
There is deliberately no command to regenerate pins from the working tree.
Only Git-tracked or unignored new files are inventoried; Python caches are not
evidence. No history or network access is needed, including in shallow CI clones.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('frozen-evidence.json')
BASELINE = 'b8340ec41799ccfe18d7aeee25bc6a23c2e0a093'
RECORDED_HASHES_SHA256 = '99c09343219da7cd7175fe4ccc567e67ba6db6da15ee00f22c9824577992a82b'


def protected(name):
    return name.startswith((
        'research/benchmark/results/',
        'docs/whitepaper/generated/',
        'research/benchmark/review_fixes/predeclare',
    )) or name in ('research/benchmark/methods.lock.json', 'ops/ci/paper-review-inputs.json')


def valid_path(name):
    return (isinstance(name, str) and name == PurePosixPath(name).as_posix()
            and not PurePosixPath(name).is_absolute()
            and '..' not in PurePosixPath(name).parts
            and not any(char in name for char in '*?[]\\') and protected(name))


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def replacement_hashes(allowlist):
    if not isinstance(allowlist, dict):
        raise ValueError('allowlist must map exact paths to replacement hashes and reasons')
    result = {}
    for name, entry in allowlist.items():
        if (not valid_path(name) or not isinstance(entry, dict)
                or set(entry) != {'sha256', 'reason'} or not valid_hash(entry['sha256'])
                or not isinstance(entry['reason'], str) or not entry['reason'].strip()):
            raise ValueError(f'invalid frozen-evidence allowlist entry: {name}')
        result[name] = entry['sha256']
    return result


def load_manifest(path):
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or set(data) != {'baseline_commit', 'files', 'allowlist'}
            or data['baseline_commit'] != BASELINE or not isinstance(data['files'], dict)):
        raise ValueError('invalid frozen-evidence manifest or baseline commit')
    recorded = json.dumps(data['files'], sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(recorded).hexdigest() != RECORDED_HASHES_SHA256:
        raise ValueError('recorded frozen hashes changed; use an explicit allowlist exception')
    replacement_hashes(data['allowlist'])
    return data


def inventory(root):
    top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True)
    if Path(top.strip()).resolve() != root:
        raise ValueError('--root must be the Git repository root')
    output = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    return {name for raw in output.split(b'\0') if raw
            if protected(name := raw.decode('utf-8'))}


def check_tree(root, recorded, allowlist):
    expected = dict(recorded, **replacement_hashes(allowlist))
    failures = []
    for name in sorted(inventory(root) | expected.keys()):
        if name not in expected:
            failures.append(f'{name}: unrecorded frozen-evidence file; requires an allowlist entry')
            continue
        path = root / name
        if not valid_path(name) or path.resolve() != path or not path.is_file():
            failures.append(f'{name}: frozen file missing, not regular, or reached through a symlink')
            continue
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected[name]:
            failures.append(f'{name}: frozen SHA256 drift (expected {expected[name]}, got {digest})')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        data = load_manifest(MANIFEST)
        failures = check_tree(args.root.resolve(), data['files'], data['allowlist'])
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Frozen evidence check failed: {error}', file=sys.stderr)
        return 1
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        print(f'Frozen evidence check failed: {len(failures)} file(s). '
              'Restore recorded bytes or add a reviewed exact-path/hash/reason exception.', file=sys.stderr)
        return 1
    print(f'Frozen evidence verified: {len(data["files"])} recorded files; '
          f'{len(data["allowlist"])} explicit exceptions (baseline {BASELINE}).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
