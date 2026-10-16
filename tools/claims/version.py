#!/usr/bin/env python3
"""Synchronize the citation version with the Cargo package version."""
import argparse
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def failures(root: Path = ROOT) -> list[str]:
    cargo = tomllib.loads((root / 'Cargo.toml').read_text())['package']['version']
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    errors = []
    if project['project'].get('name') != 'dope' or 'version' not in project['project'].get('dynamic', []):
        errors.append('Python package must be dope with a dynamic Cargo version')
    if project['tool']['setuptools']['dynamic']['version'] != {'attr': 'dope_package_version.package_version.VERSION'}:
        errors.append('Python version source differs from Cargo reader')
    citation = re.search(r'^version:\s*(.+)$', (root / 'CITATION.cff').read_text(), re.M)
    if citation is None or citation.group(1).strip('"\'') != cargo:
        errors.append('CITATION.cff version differs from Cargo.toml; run tools/claims/version.py --write')
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        version = tomllib.loads((ROOT / 'Cargo.toml').read_text())['package']['version']
        path = ROOT / 'CITATION.cff'
        path.write_text(re.sub(r'^version:.*$', 'version: ' + version, path.read_text(), flags=re.M))
    errors = failures()
    print('\n'.join(errors) if errors else 'Cargo, Python dynamic metadata and citation versions agree')
    raise SystemExit(bool(errors))
