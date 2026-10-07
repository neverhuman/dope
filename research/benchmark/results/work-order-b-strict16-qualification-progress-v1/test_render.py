#!/usr/bin/env python3
"""Generated scalar controls; no private inputs, hosts, or numerical dependencies."""
import copy
import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

RENDER_PIN = (18878, 'f82d74fd64d2ceb4d78adbb15af7f378080b5f9d4f5a929c890d72f0a4428e2d')
HERE = Path(__file__).resolve().parent
RAW = (HERE / 'render.py').read_bytes()
if (len(RAW), hashlib.sha256(RAW).hexdigest()) != RENDER_PIN:
    raise ValueError('renderer_source_pin_changed')
MODULE_NAME = 'strict16_public_renderer'


class PinnedRendererLoader(importlib.machinery.SourceFileLoader):
    """Load the fixed source buffer, bypassing every bytecode-cache path."""

    def get_code(self, fullname):
        if type(fullname) is not str or fullname != MODULE_NAME:
            raise ValueError('renderer_module_identity_changed')
        raw = (HERE / 'render.py').read_bytes()
        if (len(raw), hashlib.sha256(raw).hexdigest()) != RENDER_PIN:
            raise ValueError('renderer_source_pin_changed')
        return self.source_to_code(raw, str(HERE / 'render.py'))


LOADER = PinnedRendererLoader(MODULE_NAME, str(HERE / 'render.py'))
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, LOADER.path, loader=LOADER)
M = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(M)
DATA = M.checked(HERE, 'delta.json')
SCHEMA = M.checked(HERE, 'delta.schema.json')
PROOF = M.checked(HERE, 'source-proof.json')
PROOF_SCHEMA = M.checked(HERE, 'source-proof.schema.json')


class PublicSnapshotControls(unittest.TestCase):
    def rejected(self, mutate):
        d, p = copy.deepcopy(DATA), copy.deepcopy(PROOF)
        mutate(d, p)
        with self.assertRaises((ValueError, KeyError)):
            M.validate(d, SCHEMA)
            M.validate(p, PROOF_SCHEMA)
            M.contract(d, p)

    def test_01_valid_and_exact_regeneration(self):
        M.validate(DATA, SCHEMA)
        M.validate(PROOF, PROOF_SCHEMA)
        M.contract(DATA, PROOF)
        for name, body in M.outputs(DATA).items():
            self.assertEqual((HERE / name).read_text(encoding='utf-8'), body)

    def test_02_qualification_not_native_charge(self):
        self.rejected(lambda d, p: d['cost'].__setitem__('cumulative_new_native_and_prefit_paid_parent_seconds', 1261.6301180948503))

    def test_03_nested_clock_nonadditive(self):
        self.rejected(lambda d, p: d['cost'].__setitem__('nested_elapsed_added_to_parent', True))

    def test_04_infrastructure_paid_history_preserved(self):
        self.rejected(lambda d, p: d['cost'].__setitem__('historical_prefit_infrastructure_parent_seconds', 0))

    def test_05_infrastructure_not_extra_logical_slot(self):
        self.rejected(lambda d, p: d['progress'].__setitem__('infra_attempt_adds_an_extra_logical_slot', True))

    def test_06_fixed_pending_partition(self):
        self.rejected(lambda d, p: d['progress'].__setitem__('pending_logical_slots_in_snapshot', 389))

    def test_07_model_parity_join(self):
        self.rejected(lambda d, p: d['qualification']['parity'].__setitem__('model_sha256', '0' * 64))

    def test_08_projection_parity_join(self):
        self.rejected(lambda d, p: d['qualification']['parity'].__setitem__('projection_sha256', '0' * 64))

    def test_09_inspection_parity_join(self):
        self.rejected(lambda d, p: d['qualification']['parity'].__setitem__('inspection_sha256', '0' * 64))

    def test_10_observed_resident_not_true_peak(self):
        self.rejected(lambda d, p: d['cost'].__setitem__('true_peak_VRAM_bytes', 430 * 1048576))

    def test_11_foreign_mutex_unknown(self):
        self.rejected(lambda d, p: d['scope'].__setitem__('foreign_DLTrain_campaign_mutex_participation', True))

    def test_12_private_payload_key_rejected(self):
        self.rejected(lambda d, p: d.__setitem__('rows', [[1, 2]]))

    def test_13_operation_alias_dedup(self):
        self.rejected(lambda d, p: d['operations'].__setitem__(1, copy.deepcopy(d['operations'][0])))

    def test_14_parent_receipt_join(self):
        self.rejected(lambda d, p: d['operations'][2]['parent_ref'].__setitem__('sha256', '0' * 64))

    def test_15_method_completion_null(self):
        self.rejected(lambda d, p: d['scope'].__setitem__('method_complete', True))

    def test_16_native8_not_recharged(self):
        self.rejected(lambda d, p: d['cost'].__setitem__('native8_committed_slice_recharged', True))

    def test_17_nonfinite_clock_rejected(self):
        self.rejected(lambda d, p: d['operations'][2].__setitem__('parent_elapsed_seconds', float('nan')))

    def test_18_parent_exit_actual(self):
        self.rejected(lambda d, p: d['operations'][2].__setitem__('actual_parent_wait_exit', 1))

    def test_19_renderer_drift_rejected_before_initialization(self):
        with mock.patch.object(Path, 'read_bytes', return_value=b'raise AssertionError()'), \
             mock.patch.object(LOADER, 'source_to_code') as compiler:
            with self.assertRaisesRegex(ValueError, 'renderer_source_pin_changed'):
                LOADER.exec_module(importlib.util.module_from_spec(SPEC))
            compiler.assert_not_called()

    def test_20_bytecode_cache_loader_never_used(self):
        with mock.patch.object(importlib.machinery.SourceFileLoader, 'get_code',
                               side_effect=AssertionError('bytecode_loader_called')):
            self.assertIsNotNone(LOADER.get_code(MODULE_NAME))

    def test_21_module_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'renderer_module_identity_changed'):
            LOADER.get_code('another_renderer')


if __name__ == '__main__':
    unittest.main()
