"""Hermetic publication controls: opaque owned bytes, no ML or real data."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

from research.benchmark import publish_artifact_bytes as m


class BytePublication(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).resolve().parents[3] / 'target'
        target.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'probe'
        self.root.mkdir()
        self.parent = self.base / 'dope-s3-population-research-v2'
        self.parent.mkdir()
        shutil.copyfile(m.codec.__file__, self.root / 'codec.py')
        jobs, fits, inputs, rows, refs = [], [], [], [], {}
        for index in range(16):
            artifact = self.parent / ('model' + str(index) + '.dpk')
            artifact.write_bytes(b'opaque model bytes' * 370)
            worker = self.base / ('worker' + str(index))
            worker.mkdir()
            projection = worker / 'projection.json'
            projection.write_bytes(b'opaque projection bytes' * 420)
            job = {'dataset': 'fixture' + str(index // 4), 'research_profile': 'profile' + str(index % 4),
                'worker': {'path': str(worker), 'files': {'projection.json': m.sha256(projection)}}}
            key = m.digest(job); jobs.append(job)
            refs[str(artifact)] = m.sha256(artifact); refs[str(projection)] = m.sha256(projection)
            size = artifact.stat().st_size + projection.stat().st_size
            fits.append({'job_sha256': key, 'dataset': job['dataset'], 'profile': job['research_profile'],
                'status': 'charged_artifact_cap', 'model_path': str(artifact),
                'model_sha256': m.sha256(artifact), 'charged_artifact_bytes': size})
            item = {'parent_job_sha256': key, 'dataset': job['dataset'], 'profile': job['research_profile'],
                'model': str(artifact), 'model_sha256': m.sha256(artifact),
                'projection': str(projection), 'projection_sha256': m.sha256(projection),
                'original_charged_bytes': size}
            inputs.append(item)
            for level in (1, 9):
                output = self.root / (key + '.level' + str(level) + '.artifact')
                blob = m.codec.encode(artifact.read_bytes(), projection.read_bytes(), level)
                output.write_bytes(blob)
                row = {**item, 'compression_level': level, 'status': 'ok',
                    'raw_model_bytes': artifact.stat().st_size, 'raw_projection_bytes': projection.stat().st_size,
                    'artifact_path': str(output), 'artifact_sha256': m.sha256(output),
                    'container_bytes': len(blob), 'container_header_bytes': m.codec.HEADER.size,
                    'all_container_bytes_charged': True, 'byte_cap_satisfied': len(blob) <= 10240,
                    'exact_original_bytes_recovered': True, 'encoding_replay_exact': True,
                    'generator_admissible': False, 'generator_validation_complete': False,
                    'release_safe_l3': None, 'new_gpu_fits': 0, 'elapsed_seconds': 0.001,
                    'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}
                self.write(output.with_suffix('.receipt.json'), row); rows.append(row)
        self.write(self.parent / 'round.lock.json', {'jobs': jobs})
        self.parent_anchors = {'round.lock.json': m.sha256(self.parent / 'round.lock.json')}
        flags = {'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None}
        self.write(self.parent / 'reconciliation-v1.json', {'fit_cells': fits,
            'round_sha256': self.parent_anchors['round.lock.json'], **flags})
        self.parent_anchors['reconciliation-v1.json'] = m.sha256(self.parent / 'reconciliation-v1.json')
        self.write(self.parent / 'receipt-lock-v1.json', {'refs': refs,
            'reconciliation_sha256': self.parent_anchors['reconciliation-v1.json'], **flags})
        self.parent_anchors['receipt-lock-v1.json'] = m.sha256(self.parent / 'receipt-lock-v1.json')
        self.grid = {'inputs': inputs, 'parent_round_sha256': self.parent_anchors['round.lock.json'],
            'parent_receipt_lock_sha256': self.parent_anchors['receipt-lock-v1.json'],
            'candidate_compression_levels': [1, 9], 'candidate_count': 32, 'artifact_cap_bytes': 10240,
            'whole_container_bytes_charged': True, 'maximum_threads': 1, 'maximum_elapsed_seconds': 60,
            'maximum_decoded_bytes_per_candidate': 65536, 'generator_validation_admitted': False,
            'new_gpu_fits': 0, 'codec_source_sha256': m.sha256(self.root / 'codec.py'), **flags}
        self.write(self.root / 'analysis-grid.lock.json', self.grid)
        self.analysis = {'candidates': rows, 'analysis_grid_sha256': m.sha256(self.root / 'analysis-grid.lock.json'),
            'codec_sha256': self.grid['codec_source_sha256'], 'candidate_count': 32,
            'all_candidates_accounted': True, 'byte_cap_satisfied_candidates': 32,
            'generator_validation_admitted': False, 'raw_parent_scores_rewritten': False,
            'production_certified': False, 'counts_as_dope_win': False, 'release_safe_l3': None,
            'new_gpu_fits': 0, 'elapsed_seconds': 0.1, 'host': 'xbabe2', **flags}
        self.refresh()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True))

    def refresh(self):
        self.write(self.root / 'analysis.json', self.analysis)
        receipt = {'refs': {str(p): m.sha256(p) for p in self.root.iterdir()
                           if p.is_file() and p.name != 'receipt-lock-v1.json'},
            'analysis_grid_sha256': m.sha256(self.root / 'analysis-grid.lock.json'),
            'analysis_sha256': m.sha256(self.root / 'analysis.json'),
            'official_tests_opened': False, 'mfs_v2': None, 'ptf_v1': None,
            'generator_validation_admitted': False, 'production_certified': False}
        self.write(self.root / 'receipt-lock-v1.json', receipt)
        self.receipt_sha = m.sha256(self.root / 'receipt-lock-v1.json')

    def call(self):
        with patch.object(m, 'ROOT', self.root), patch.object(m, 'RECEIPT_SHA', self.receipt_sha), \
             patch.object(m, 'GRID_SHA', m.sha256(self.root / 'analysis-grid.lock.json')), \
             patch.object(m, 'PARENT_ANCHORS', self.parent_anchors):
            return m.build()

    def test_complete_grid_preserves_null_generator_claims_and_full_bytes(self):
        report = self.call()
        self.assertEqual(report['candidate_count'], 32)
        self.assertEqual(report['parent_fit_cells'], 16)
        self.assertFalse(report['matrix_admission'])
        self.assertFalse(report['generator_validation_complete'])
        self.assertFalse(report['product_wire_format_supported'])
        self.assertIsNone(report['mfs_v2'])
        self.assertIn(f'{m.codec.HEADER.size}-byte header', m.tables(report)[1])

    def test_receipt_rewrite_rejected_before_any_metric_read(self):
        (self.root / 'receipt-lock-v1.json').write_text('{}')
        with patch.object(m, 'read', side_effect=AssertionError('metric read before receipt proof')):
            with self.assertRaises(ValueError): self.call()

    def test_added_directory_alias_rejected(self):
        (self.root / 'alias').symlink_to(self.parent, target_is_directory=True)
        with self.assertRaises(ValueError): self.call()

    def test_mutable_analysis_rejected(self):
        (self.root / 'analysis.json').write_text('{}')
        with self.assertRaises(ValueError): self.call()

    def test_codec_source_drift_rejected(self):
        (self.root / 'codec.py').write_text('changed source')
        with self.assertRaises(ValueError): self.call()

    def test_missing_grid_cell_rejected_even_with_new_fixture_anchor(self):
        self.analysis['candidates'].pop(); self.refresh()
        with self.assertRaises(ValueError): self.call()

    def test_duplicate_grid_cell_rejected_even_with_new_fixture_anchor(self):
        self.analysis['candidates'].append(self.analysis['candidates'][0]); self.refresh()
        with self.assertRaises(ValueError): self.call()

    def test_byte_undercharge_rejected_even_with_new_fixture_anchor(self):
        row = self.analysis['candidates'][0]; row['container_bytes'] -= 1
        self.write(Path(row['artifact_path']).with_suffix('.receipt.json'), row); self.refresh()
        with self.assertRaises(ValueError): self.call()

    def test_gated_claim_rejected_even_with_new_fixture_anchor(self):
        self.analysis['production_certified'] = True; self.refresh()
        with self.assertRaises(ValueError): self.call()

    def test_artifact_corruption_rejected(self):
        Path(self.analysis['candidates'][0]['artifact_path']).write_bytes(b'changed artifact')
        with self.assertRaises(ValueError): self.call()

    def test_owned_parent_artifact_drift_rejected(self):
        Path(self.grid['inputs'][0]['model']).write_bytes(b'changed model')
        with self.assertRaises(ValueError): self.call()


if __name__ == '__main__': unittest.main()
