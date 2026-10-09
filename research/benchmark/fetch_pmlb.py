"""Credential-free, SHA-pinned PMLB source downloads and historical split identities.

The upstream source is not the historical staged coreset. No split is selected,
model fit, or official test scored by this tool. See the supplement's public kit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / 'ops/ci/paper-inputs/pmlb-sources.json'
MANIFEST_SHA256 = '0d380a0a8605417edddcd47718efccd9d47d66cee3bbd628dbc64009ee8e8407'


def source_manifest() -> dict:
    raw = MANIFEST.read_bytes()
    if MANIFEST.is_symlink() or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('public source manifest digest mismatch')
    document = json.loads(raw)
    rows = document['entries']
    if (document['format'] != 'dope-public-pmlb-sources-v1' or len(rows) != 100
            or len({row['dataset_id'] for row in rows}) != len(rows)
            or len({row['pmlb_id'] for row in rows}) != len(rows)):
        raise ValueError('public source manifest matrix differs')
    return document


def _verify(path: Path, source: dict) -> None:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    if path.stat().st_size != source['bytes'] or digest.hexdigest() != source['sha256']:
        raise ValueError('public source bytes or digest mismatch')


def fetch(dataset_id: str, output_root: Path) -> dict:
    manifest = source_manifest()
    matches = [row for row in manifest['entries'] if dataset_id in (row['dataset_id'], row['pmlb_id'])]
    if len(matches) != 1:
        raise ValueError('dataset absent from public source manifest')
    source = matches[0]
    if output_root.resolve().is_relative_to(Path('/tmp')):
        raise ValueError('public source output must persist outside /tmp')
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / (source['pmlb_id'] + '.tsv.gz')
    if path.is_symlink():
        raise ValueError('symlinked public source destination')
    if path.exists():
        _verify(path, source)
    else:
        partial = path.with_suffix(path.suffix + '.part')
        request = urllib.request.Request(source['url'], headers={'User-Agent': 'dope-public-data-kit'})
        # Exclusive create avoids replacing another download or partial evidence.
        with partial.open('xb') as handle, urllib.request.urlopen(request, timeout=60) as response:
            remaining = source['bytes']
            while remaining:
                block = response.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError('incomplete public source download')
                handle.write(block)
                remaining -= len(block)
            if response.read(1):
                raise ValueError('oversized public source download')
        _verify(partial, source)
        if path.exists():
            raise ValueError('public source destination appeared during download')
        partial.rename(path)
    return {'format': 'dope-public-pmlb-fetch-v1', 'dataset_id': source['dataset_id'],
            'pmlb_id': source['pmlb_id'], 'revision': manifest['revision'],
            'sha256': source['sha256'], 'bytes': source['bytes'],
            'historical_split': source['historical_split'],
            'historical_projected_sha256': source['historical_projected_sha256'],
            'exact_historical_partitions_reconstructible': False, 'path': str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_id', nargs='?')
    parser.add_argument('output_root', nargs='?', type=Path)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(source_manifest(), sort_keys=True, indent=2))
    else:
        if not args.dataset_id or args.output_root is None:
            parser.error('dataset_id and output_root are required unless --list is used')
        print(json.dumps(fetch(args.dataset_id, args.output_root), sort_keys=True))


if __name__ == '__main__':
    main()
