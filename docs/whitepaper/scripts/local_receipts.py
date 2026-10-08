#!/usr/bin/env python3
"""Regenerate original-receipt extracts using committed metadata only."""
import argparse
import json
from pathlib import Path
from paper_receipts import reduce_originals

OUT = Path(__file__).resolve().parents[1] / 'generated'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    extracts, _ = reduce_originals()
    for name, payload in extracts.items():
        path = OUT / name
        text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        if args.check:
            if not path.is_file() or path.read_text() != text:
                raise ValueError('original receipt extract drift: ' + name)
        else:
            path.write_text(text)
    print('original receipt extracts match committed metadata' if args.check else 'original receipt extracts regenerated from committed metadata')


if __name__ == '__main__':
    main()
