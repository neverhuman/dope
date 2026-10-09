"""Public manifest coverage, byte verification, and explicit AWS selection."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.benchmark import fetch_pmlb, fetch_jope


class PublicFetchTests(unittest.TestCase):
    def test_public_manifest_matches_every_prepared_lineage_and_split(self):
        import json
        manifest = fetch_pmlb.source_manifest()
        lock = json.loads((fetch_pmlb.REPO / 'research/benchmark/results/s3-data.lock.json').read_text())
        prepared = {row['id']: row for row in lock['entries'] if row.get('split')}
        self.assertEqual({row['dataset_id'] for row in manifest['entries']}, set(prepared))
        for row in manifest['entries']:
            self.assertEqual(row['historical_split'], prepared[row['dataset_id']]['split'])
            self.assertIn(manifest['revision'], row['url'])
            self.assertEqual(len(row['sha256']), 64)
        self.assertFalse(manifest['exact_historical_partitions_reconstructible'])

    def test_mock_download_authenticates_and_reuses_without_credentials(self):
        payload = b'public fixture source bytes'
        row = {'dataset_id': 'fixture', 'pmlb_id': 'fixture', 'bytes': len(payload),
               'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://example.org/source',
               'historical_split': {}, 'historical_projected_sha256': {}}
        manifest = {'entries': [row], 'revision': 'frozen'}
        with tempfile.TemporaryDirectory(dir=fetch_pmlb.REPO / 'target') as directory:
            with patch.object(fetch_pmlb, 'source_manifest', return_value=manifest), \
                    patch.object(fetch_pmlb.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as request:
                receipt = fetch_pmlb.fetch('fixture', Path(directory))
                self.assertEqual(receipt['sha256'], row['sha256'])
                self.assertEqual(Path(receipt['path']).read_bytes(), payload)
                fetch_pmlb.fetch('fixture', Path(directory))
                self.assertEqual(request.call_count, 1)
                Path(receipt['path']).write_bytes(b'drift')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    fetch_pmlb.fetch('fixture', Path(directory))

    def test_default_aws_command_omits_profile_and_explicit_profile_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=fetch_pmlb.REPO / 'target') as directory:
            with patch.object(fetch_jope.subprocess, 'run') as run:
                for profile in (None, 'user-supplied'):
                    fetch_jope.fetch_object('fixture', Path(directory) / 'data.csv', profile)
                    command = run.call_args.args[0]
                    self.assertEqual('--profile' in command, profile is not None)


if __name__ == '__main__':
    unittest.main()
