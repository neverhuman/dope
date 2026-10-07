"""Metadata-only first12 publication controls; no private operation/data reads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from research.benchmark import publish_common_validation_batch as publisher
from research.benchmark import common_validation_receipt_verifier as verifier

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / 'results/common-A952-validation-v1'
TEMP_ROOT = HERE.parents[1] / 'target'


def reference(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class PublicationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = types.ModuleType('common_A952_public_renderer')
        cls.renderer.__file__ = str(RESULTS / 'render.py')
        exec(compile((RESULTS / 'render.py').read_bytes(), cls.renderer.__file__, 'exec'), cls.renderer.__dict__)
        cls.public = cls.renderer.checked_public(RESULTS / 'first12.completed.json')

    def setUp(self):
        publisher.READ_REFS.clear()

    def test_included_public_leaves_regenerate_exactly(self):
        for name, raw in self.renderer.render(self.public).items():
            self.assertEqual((RESULTS / name).read_bytes(), raw, name)

    def test_receipt_hash_rejects_before_json_decode(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            path = Path(directory) / 'generated.json'; path.write_bytes(b'not JSON')
            item = reference(path); item['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'metadata_pin_changed'):
                publisher.checked(item)

    def test_reference_length_requires_builtin_int(self):
        item = dict(path='/generated/not-opened.json', bytes=True, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'reference_identity_changed'):
            publisher.checked_bytes(item)

    def test_source_pin_rejects_before_source_execution(self):
        item = dict(path='/generated/not-opened.py', bytes=0, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'unreviewed_receipt_verifier'):
            publisher.adapter_from_ref(item)

    def test_authenticated_adapter_is_provenance_not_executable_code(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            path = Path(directory) / 'generated_adapter.py'
            path.write_text('raise AssertionError("source must not execute")\n')
            item = reference(path)
            with patch.object(publisher, 'ADAPTER', item['sha256']):
                self.assertIs(publisher.adapter_from_ref(item), verifier)
            self.assertEqual(publisher.READ_REFS[str(path)], item)

    def test_public_input_hash_rejects_before_json_decode(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            path = Path(directory) / 'generated.json'; path.write_bytes(b'not JSON')
            with self.assertRaisesRegex(ValueError, 'public_input_receipt_changed'):
                self.renderer.checked_public(path)

    def test_gated_scores_remain_null(self):
        for key in ('mfs_v2', 'ptf_v1', 'release_safe_l3', 'superiority'):
            with self.subTest(key=key):
                value = copy.deepcopy(self.public); value[key] = 0
                with self.assertRaisesRegex(ValueError, 'gated_or_unmeasured_claim'):
                    self.renderer.check_contract(value)

    def test_nested_gated_scores_remain_null(self):
        value = copy.deepcopy(self.public); value['cells'][0]['diagnostics']['mfs_v2'] = 0
        with self.assertRaisesRegex(ValueError, 'gated_or_unmeasured_claim'):
            self.renderer.check_contract(value)

    def test_duplicate_physical_cell_is_not_alias(self):
        value = copy.deepcopy(self.public)
        value['cells'][1]['physical_job_sha256'] = value['cells'][0]['physical_job_sha256']
        with self.assertRaisesRegex(ValueError, 'physical_cell_repeated'):
            self.renderer.check_contract(value)

    def test_logical_aliases_do_not_multiply_physical_cost(self):
        value = copy.deepcopy(self.public); value['closed_batch_operation_wall_seconds_sum'] *= 14 / 12
        with self.assertRaisesRegex(ValueError, 'physical_cost_sum_changed'):
            self.renderer.check_contract(value)

    def test_campaign_and_core_costs_stay_unmeasured(self):
        value = copy.deepcopy(self.public); value['cells'][0]['cost']['core_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'gated_or_unmeasured_claim'):
            self.renderer.check_contract(value)

    def test_complete_batch_does_not_complete_cohort(self):
        value = copy.deepcopy(self.public); value['cohort_complete'] = True
        with self.assertRaisesRegex(ValueError, 'public_coverage_changed'):
            self.renderer.check_contract(value)


class StaticReceiptVerifierContract(unittest.TestCase):
    def setUp(self):
        publisher.READ_REFS.clear()
        self.directory = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.job = {'generated_metadata_only': True}
        self.key = publisher.digest(self.job)
        self.lock = {'jobs': {self.key: self.job}, 'kind_jobs': {}}
        self.baseline = {'source_files': {'worker.py': 'a' * 64}, 'source_root': '/generated/source',
                         'runtime_ref': {'path': '/generated/runtime.json', 'bytes': 1, 'sha256': 'b' * 64},
                         'preparation_root': '/generated/preparation'}
        self.baseline_ref = self.write('baseline.json', self.baseline)
        self.round_ref = self.write('round.json', self.lock)
        self.request = {'job': self.job, 'job_sha256': self.key}
        self.request_ref = self.write('request.json', self.request)
        self.config = dict(self.baseline, action='metric-one', request_ref=self.request_ref,
                           original_round_ref=self.round_ref)
        self.config_ref = self.write('config.json', self.config)
        self.operation = {'config_ref': self.config_ref, 'action': 'metric-one',
                          'status': 'actual_exit_recorded', 'actual_Popen_exit': 0,
                          'scope_result_eligible': True, 'reason': None,
                          'remaining_captured_identities': [], 'unknown_operation_group_identities': []}
        self.stdout = {'status': 'ok', 'job_sha256': self.key, 'official_tests_opened': False}
        self.operation_ref = self.write('operation.json', self.operation)
        self.stdout_ref = self.write('stdout.json', self.stdout)
        self.record = {'config_ref': self.config_ref, 'operation_ref': self.operation_ref,
                       'stdout_ref': self.stdout_ref}
        self.patcher = patch.object(verifier, 'BASE_CONFIG', self.baseline_ref)
        self.patcher.start(); self.addCleanup(self.patcher.stop)

    def write(self, name, value):
        path = self.root / name; path.write_bytes(publisher.canonical(value) + b'\n')
        return reference(path)

    def positive(self):
        return verifier.positive(self.lock, self.record, publisher.checked)

    def test_complete_generated_metadata_tracks_all_transitive_receipts(self):
        self.assertEqual(self.positive(), ('metric-one', self.key, self.stdout))
        expected = {str(self.root / name) for name in ('baseline.json', 'round.json', 'request.json',
                                                     'config.json', 'operation.json', 'stdout.json')}
        self.assertEqual(set(publisher.READ_REFS), expected)

    def test_transitive_drift_rejects_before_json_decode(self):
        (self.root / 'request.json').write_bytes(b'not JSON')
        with self.assertRaisesRegex(ValueError, 'metadata_pin_changed'):
            self.positive()

    def test_transitive_drift_after_validation_rejects_at_final_rehash(self):
        self.positive(); (self.root / 'baseline.json').write_bytes(b'not JSON')
        with self.assertRaisesRegex(ValueError, 'metadata_pin_changed'):
            for item in list(publisher.READ_REFS.values()):
                publisher.checked_bytes(item)

    def test_source_runtime_and_round_drift_reject(self):
        for field, value in (('source_files', {}), ('runtime_ref', {}), ('source_root', '/changed')):
            with self.subTest(field=field):
                publisher.READ_REFS.clear()
                config = dict(self.config); config[field] = value
                pin = self.write('changed-config.json', config)
                record = dict(self.record, config_ref=pin)
                with self.assertRaisesRegex(ValueError, 'positive_frozen_science_changed'):
                    verifier.positive(self.lock, record, publisher.checked)
        publisher.READ_REFS.clear()
        changed = self.write('changed-round.json', {'jobs': {}, 'kind_jobs': {}})
        config = dict(self.config, original_round_ref=changed)
        record = dict(self.record, config_ref=self.write('changed-config.json', config))
        with self.assertRaisesRegex(ValueError, 'positive_frozen_science_changed'):
            verifier.positive(self.lock, record, publisher.checked)

    def test_job_join_drift_rejects(self):
        for request in ({'job': {'changed': True}, 'job_sha256': self.key},
                        {'job': self.job, 'job_sha256': '0' * 64}):
            publisher.READ_REFS.clear()
            config = dict(self.config, request_ref=self.write('changed-request.json', request))
            record = dict(self.record, config_ref=self.write('changed-config.json', config))
            with self.assertRaisesRegex(ValueError, 'positive_job_changed'):
                verifier.positive(self.lock, record, publisher.checked)

    def test_nonzero_ineligible_or_unclosed_operation_is_not_positive(self):
        for field, value in (('actual_Popen_exit', 2), ('scope_result_eligible', False),
                             ('remaining_captured_identities', [[1, 2, 3]]),
                             ('unknown_operation_group_identities', [[1, 2, 3]]),
                             ('reason', 'unavailable')):
            with self.subTest(field=field):
                publisher.READ_REFS.clear(); operation = dict(self.operation); operation[field] = value
                record = dict(self.record, operation_ref=self.write('changed-operation.json', operation))
                with self.assertRaisesRegex(ValueError, 'positive_actual_custody_missing'):
                    verifier.positive(self.lock, record, publisher.checked)

    def test_unavailable_or_test_access_is_not_positive(self):
        for field, value in (('status', 'unavailable'), ('official_tests_opened', True)):
            publisher.READ_REFS.clear(); stdout = dict(self.stdout); stdout[field] = value
            record = dict(self.record, stdout_ref=self.write('changed-stdout.json', stdout))
            with self.assertRaisesRegex(ValueError, 'positive_worker_result_missing'):
                verifier.positive(self.lock, record, publisher.checked)

    def test_exact_TS_count_and_frozen_caps(self):
        jobs = {publisher.digest({'index': index}): {'index': index} for index in range(143)}
        lock = {'scope': 'tabsyn_closed143_common_validation_v2', 'matrix': None, 'kind_jobs': {},
                'jobs': jobs, 'historical_parent_exit': None, 'deadline_seconds_per_job': 600,
                'ram_cap_bytes': 16 * 1024 ** 3, 'max_cores': 16, 'gpu_allowed': False}
        verifier.validate_jobs(lock)
        for field, value, reason in (('jobs', dict(list(jobs.items())[:142]), 'frozen_job_counts_changed'),
                                    ('deadline_seconds_per_job', 601, 'frozen_caps_changed'),
                                    ('gpu_allowed', True, 'frozen_caps_changed')):
            with self.subTest(field=field):
                changed = dict(lock); changed[field] = value
                with self.assertRaisesRegex(ValueError, reason):
                    verifier.validate_jobs(changed)
        changed = copy.deepcopy(lock); next(iter(changed['jobs'].values()))['index'] = -1
        with self.assertRaisesRegex(ValueError, 'canonical_job_changed'):
            verifier.validate_jobs(changed)


if __name__ == '__main__':
    unittest.main()
