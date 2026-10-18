"""Deterministic local fixtures only; no cohort or protected partitions."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from research.transfer.evidence import EvidenceError, validate_behavior_contract

ROOT = Path(__file__).resolve().parents[3]


class BehaviorContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((ROOT / 'production/behavior-contract-v1.json').read_bytes())

    def test_draft_keeps_confirmation_unavailable_and_history_unchanged(self):
        validate_behavior_contract(self.contract)
        self.assertFalse(self.contract['protocol_frozen'])
        self.assertFalse(self.contract['confirmation_allowed'])
        self.assertFalse(self.contract['endpoints']['score_implies_release'])
        for path, expected in self.contract['historical_contract_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_unknown_margins_checkpoint_and_rights_prevent_freeze(self):
        resolved = copy.deepcopy(self.contract)
        resolved.update(status='frozen', protocol_frozen=True, confirmation_allowed=True)
        resolved['freeze_requirements'] = {k: 'a' * 64 for k in resolved['freeze_requirements']}
        resolved['margins'] = {k: .1 for k in resolved['margins']}
        resolved['budgets'] = {k: 1 for k in resolved['budgets']}
        for k in ('table_schema_hash', 'task_manifest_hash', 'population_history_hash'):
            resolved['workload'][k] = 'a' * 64
        resolved['learners']['held_out_families'] = ['held_out']
        resolved['disclosure'].update(protected_unit='row', auxiliary_information_hash='a' * 64,
            repeated_release_policy_hash='a' * 64, privacy_route='scoped', legal_route='scoped', purpose_grants=['customer_modeling'])
        validate_behavior_contract(resolved)
        cases = [('margins', 'noninferiority'), ('margins', 'catastrophic_harm'),
                 ('freeze_requirements', 'checkpoint_manifest'), ('freeze_requirements', 'rights_manifest')]
        for section, key in cases:
            with self.subTest(key=key):
                broken = copy.deepcopy(resolved)
                broken[section][key] = None
                with self.assertRaises(EvidenceError):
                    validate_behavior_contract(broken)
                broken_tbd = copy.deepcopy(resolved)
                broken_tbd[section][key] = 'TBD-D02'
                with self.assertRaises(EvidenceError):
                    validate_behavior_contract(broken_tbd)
        tbds = {k: v for k, v in self.contract['freeze_requirements'].items()
                if isinstance(v, str) and v.startswith('TBD-')}
        self.assertEqual(len(tbds), 13)

    def test_score_and_prerequisite_booleans_cannot_grant_authority(self):
        for field in ('active_production_tier', 'confirmation_allowed'):
            broken = copy.deepcopy(self.contract)
            broken[field] = True
            with self.assertRaises(EvidenceError):
                validate_behavior_contract(broken)
        broken = copy.deepcopy(self.contract)
        broken['endpoints']['score_implies_release'] = True
        with self.assertRaises(EvidenceError):
            validate_behavior_contract(broken)



# Everything below is generated under the lane's target/; fixtures contain only
# synthetic constants. Failed-test artifacts are retained for repair evidence.
import os
import tempfile
from research.transfer.evidence import (
    EvaluationIdentity, UnverifiedReceipt, VerificationPolicy,
    VerifiedEvaluationReceipt, canonical_json, parse_json, digest,
    scientific_cache_key, fit_replicate_key, verify_receipt,
    commit_completed_receipt, load_completed_receipt,
)


class ReceiptFixture:
    def __init__(self, confirmation=False):
        parent = ROOT / 'target/vnext/evidence-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(dir=parent))
        self.nodes = []
        schema = json.loads((ROOT / 'schemas/transfer-run-v1.schema.json').read_text())
        identity = {}
        for key in schema['$defs']['EvaluationIdentity']['properties']:
            if key.endswith('_hash') and key not in ('protocol_hash', 'fit_hash', 'artifact_inventory_hash', 'split_hash'):
                identity[key] = self.node(key[:-5], key.encode())
        identity.update(partition_role='confirmation' if confirmation else 'selection',
            global_seed=7, generator_fit_seed=11, sample_seed=13, learner_seed=17,
            fit_rows=3, requested_rows=4, produced_rows=4, consumed_rows=4, query_rows=3,
            context_rows=4, ensemble_count=1, precision='float64', determinism='deterministic')
        identity['split_hash'] = self.node('split', canonical_json({'format': 'dope-split-manifest', 'version': 1, **{k: identity[k] for k in ('data_hash', 'query_hash', 'partition_role', 'fit_rows', 'query_rows')}}), [identity['data_hash'], identity['query_hash']])
        protocol = json.loads((ROOT / 'production/behavior-contract-v1.json').read_text())
        commitments = []
        if confirmation:
            policy_hash = self.node('policy', b'fixture-policy-without-real-authorization')
            protocol.update(status='frozen', protocol_frozen=True, confirmation_allowed=True)
            protocol['freeze_requirements'] = {k: policy_hash for k in protocol['freeze_requirements']}
            protocol['margins'] = {k: .1 for k in protocol['margins']}
            protocol['budgets'] = {k: 1 for k in protocol['budgets']}
            for k, identity_field in (('table_schema_hash', 'schema_hash'), ('task_manifest_hash', 'task_hash'), ('population_history_hash', 'source_family_hash')):
                protocol['workload'][k] = identity[identity_field]
            protocol['learners']['held_out_families'] = ['held_out']
            protocol['disclosure'].update(protected_unit='row', auxiliary_information_hash=policy_hash,
                repeated_release_policy_hash=policy_hash, privacy_route='scoped', legal_route='scoped', purpose_grants=['customer_modeling'])
            commitments = [policy_hash, identity['schema_hash'], identity['task_hash'], identity['source_family_hash']]
        identity['protocol_hash'] = self.node('protocol', canonical_json(protocol) if confirmation else (ROOT / 'production/behavior-contract-v1.json').read_bytes(), commitments)
        state_hash = self.node('fit_state', b'fixture-fit-state')
        fields = ('data_hash', 'split_hash', 'generator_hash', 'configuration_hash',
                  'feature_transform_hash', 'target_transform_hash', 'generator_fit_seed', 'global_seed')
        fit = {'format': 'dope-generator-fit', 'version': 1, 'state_hash': state_hash,
               **{k: identity[k] for k in fields}}
        identity['fit_hash'] = self.node('fit', canonical_json(fit), [state_hash, *(identity[k] for k in fields if k.endswith('_hash'))])
        artifact = self.node('artifact', b'fixture-sampling-artifact')
        identity['artifact_inventory_hash'] = self.node('artifact_inventory', canonical_json(
            {'format': 'dope-artifact-inventory', 'version': 1, 'files': [artifact]}), [artifact])
        paired = {'format': 'dope-paired-losses', 'version': 1, 'query_hash': identity['query_hash'],
                  'metric_hash': identity['metric_hash'], 'loss': 'mse', 'loss_space': 'original_target_units',
                  'loss_unit': 'target_units_squared', 'real': [0, 1, 2], 'synthetic': [1, 2, 3], 'null': [2, 3, 4]}
        paired_hash = self.node('paired_evidence', canonical_json(paired), [identity['query_hash'], identity['metric_hash']])
        executor = self.node('executor', b'fixture-executor')
        freeze_hash = None
        if confirmation:
            selection = self.node('selection_receipt', b'fixture-selection-receipt')
            access = self.node('access_ledger', b'fixture-access-ledger')
            freeze = {'format': 'dope-freeze', 'version': 1, **{k: identity[k] for k in ('protocol_hash', 'artifact_inventory_hash', 'configuration_hash', 'split_hash', 'evaluation_code_hash')},
                'selection_receipt_hash': selection, 'executor_hash': executor, 'access_ledger_hash': access,
                'cell_hash': scientific_cache_key(EvaluationIdentity(identity)), 'parent_freeze_hash': None,
                'protocol_frozen': True, 'confirmation_allowed': True}
            freeze_hash = self.node('freeze', canonical_json(freeze), [selection, access])
        self.receipt = {'format': 'dope-transfer-run', 'version': 1, 'identity': identity,
            'executor_hash': executor, 'freeze_hash': freeze_hash, 'dependencies': self.nodes,
            'sampling_payload_bytes': len(b'fixture-sampling-artifact'), 'paired_evidence_hash': paired_hash,
            'loss': 'mse', 'loss_space': 'original_target_units', 'loss_unit': 'target_units_squared',
            'losses': {'real': 1.0, 'synthetic': 2.0, 'null': 3.0}, 'execution': 'complete',
            'attempt': {'attempt_id': 'a' * 64, 'parent_attempt_id': None, 'exit_code': 0,
                        'wall_time_ms': 1, 'cpu_time_ms': 0, 'peak_memory_bytes': 1},
            'projection': 'private', 'release': 'restricted_local', 'transfer': 'unmeasured',
            'empirical_privacy': 'unmeasured', 'formal_mechanism': 'none', 'legal_route': 'unreviewed'}

    def node(self, kind, data, requires=()):
        key = hashlib.sha256(data).hexdigest()
        path = str(len(self.nodes)) + '.blob'
        (self.root / path).write_bytes(data)
        self.nodes.append({'sha256': key, 'path': path, 'bytes': len(data), 'kind': kind, 'requires': list(requires)})
        return key

    def policy(self, receipt=None):
        r = self.receipt if receipt is None else receipt
        return VerificationPolicy(EvaluationIdentity(r['identity']), r['executor_hash'], digest(r), r['freeze_hash'])

    def verify(self, receipt=None, policy=None):
        r = self.receipt if receipt is None else receipt
        return verify_receipt(UnverifiedReceipt(canonical_json(r)), self.root, self.policy(r) if policy is None else policy)

    def replace_manifest(self, key, value):
        n = next(n for n in self.nodes if n['sha256'] == key)
        data = canonical_json(value)
        (self.root / n['path']).write_bytes(data)
        new_key = hashlib.sha256(data).hexdigest()
        n.update(sha256=new_key, bytes=len(data))
        return new_key


class CanonicalTests(unittest.TestCase):
    def test_golden_finite_json_encoding(self):
        self.assertEqual(canonical_json({'z': -.0, 'b': .3, 'a': [True, None, 'é']}),
                         b'{"a":[true,null,"\xc3\xa9"],"b":2.99999999999999989e-1,"z":-0.00000000000000000e0}')
        self.assertEqual(digest({'a': 1, 'b': 2}), '43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777')

    def test_duplicate_nonfinite_oversize_and_surrogate_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":{"a":1,"a":2}}', b'NaN', b'Infinity', b'-Infinity', b'1e999', b'9007199254740992', b'"\\ud800"'):
            with self.subTest(raw=raw), self.assertRaises(EvidenceError):
                UnverifiedReceipt(raw)

    def test_unknown_versions_wrong_scalars_and_extras_rejected(self):
        f = ReceiptFixture()
        mutations = [('version', True), ('version', 2), ('version', 1.0), ('sampling_payload_bytes', True), ('sampling_payload_bytes', 0), ('losses', {'real': True, 'synthetic': 2, 'null': 3}), ('artifact_sampling_verified', True)]
        for key, value in mutations:
            r = copy.deepcopy(f.receipt)
            r[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(EvidenceError):
                f.verify(r)


class ReceiptTests(unittest.TestCase):
    def test_completed_verification_and_defensive_snapshot(self):
        f = ReceiptFixture()
        v = f.verify()
        self.assertIsInstance(v, VerifiedEvaluationReceipt)
        view = v.value
        view['losses']['synthetic'] = 0
        self.assertEqual(v.value['losses']['synthetic'], 2.0)
        with self.assertRaises(EvidenceError):
            VerifiedEvaluationReceipt(canonical_json(f.receipt))

    def test_full_cache_identity_changes(self):
        f = ReceiptFixture()
        identity = EvaluationIdentity(f.receipt['identity'])
        key = scientific_cache_key(identity)
        self.assertEqual(key, '4bfc81f998d7d6a907b60e44948a4f211c5aa84f2ed865f731fb7fb253e8952e')
        self.assertEqual(fit_replicate_key(identity), '6774caa4afc0fcad64da7139d288b94a849ecc786c7a0cfa45112e36175da5a1')
        for field in identity.value:
            changed = identity.value
            v = changed[field]
            changed[field] = ('f' * 64 if field.endswith('_hash') else
                              v + 1 if type(v) is int else
                              'exploratory' if field == 'partition_role' else
                              'float32' if field == 'precision' else 'seeded_best_effort')
            with self.subTest(field=field):
                self.assertNotEqual(scientific_cache_key(EvaluationIdentity(changed)), key)

    def test_attempt_and_path_aliases_keep_fit_replicate_identity(self):
        f = ReceiptFixture()
        before = f.verify()
        f.receipt['attempt']['attempt_id'] = 'b' * 64
        for n in f.nodes:
            new = 'alias-' + n['path']
            os.link(f.root / n['path'], f.root / new)
            n['path'] = new
        after = f.verify()
        self.assertNotEqual(before.receipt_hash, after.receipt_hash)
        self.assertEqual(scientific_cache_key(before.identity), scientific_cache_key(after.identity))
        self.assertEqual(fit_replicate_key(before.identity), fit_replicate_key(after.identity))
        changed = after.identity.value
        changed.update(sample_seed=100, learner_seed=101, sample_hash='f' * 64, fit_hash='e' * 64)
        self.assertEqual(fit_replicate_key(before.identity), fit_replicate_key(EvaluationIdentity(changed)))
        changed['generator_fit_seed'] += 1
        self.assertNotEqual(fit_replicate_key(before.identity), fit_replicate_key(EvaluationIdentity(changed)))

    def test_independent_executor_receipt_identity_and_freeze_pins(self):
        f = ReceiptFixture()
        for kwargs in ({'receipt_hash': 'a' * 64}, {'executor_hash': 'a' * 64}, {'freeze_hash': 'a' * 64}, {'identity': EvaluationIdentity({**f.receipt['identity'], 'metric_hash': 'a' * 64})}):
            policy = f.policy()
            for k, v in kwargs.items():
                setattr(policy, k, v)
            with self.subTest(kwargs=kwargs), self.assertRaises(EvidenceError):
                f.verify(policy=policy)

    def test_corruption_size_kind_missing_cycle_and_alias(self):
        for mutation in ('corrupt', 'bytes', 'kind', 'missing', 'cycle', 'alias', 'unbound'):
            f = ReceiptFixture()
            node = f.nodes[0]
            if mutation == 'corrupt': (f.root / node['path']).write_bytes(b'corrupt')
            elif mutation == 'bytes': node['bytes'] += 1
            elif mutation == 'kind': node['kind'] = 'checkpoint'
            elif mutation == 'missing': f.nodes.pop(0)
            elif mutation == 'cycle': node['requires'] = [node['sha256']]
            elif mutation == 'alias': f.nodes.append(copy.deepcopy(node))
            else: f.node('policy', b'unreferenced-dependency')
            with self.subTest(mutation=mutation), self.assertRaises(EvidenceError): f.verify()

    def test_inconsistent_bytes_rows_units_and_paired_means(self):
        cases = [('sampling_payload_bytes', 999), ('loss_unit', 'target_units'), ('losses', {'real': 1, 'synthetic': 0, 'null': 3}), ('projection', 'sanitized_public'), ('release', 'eligible')]
        for key, value in cases:
            f = ReceiptFixture()
            f.receipt[key] = value
            with self.subTest(key=key), self.assertRaises(EvidenceError): f.verify()
        for field in ('produced_rows', 'consumed_rows', 'query_rows', 'context_rows'):
            f = ReceiptFixture()
            f.receipt['identity'][field] = 999
            with self.subTest(field=field), self.assertRaises(EvidenceError): f.verify()

    def test_booleans_cannot_fabricate_fit_or_paired_losses(self):
        for kind, field in (('fit', 'generator_fit_seed'), ('paired_evidence', 'real')):
            f = ReceiptFixture()
            node = next(n for n in f.nodes if n['kind'] == kind)
            value = parse_json((f.root / node['path']).read_bytes())
            value[field] = True if kind == 'fit' else [True, 1, 2]
            new_key = f.replace_manifest(node['sha256'], value)
            if kind == 'fit': f.receipt['identity']['fit_hash'] = new_key
            else: f.receipt['paired_evidence_hash'] = new_key
            with self.assertRaises(EvidenceError): f.verify()

    def test_exact_zero_loss_is_valid(self):
        f = ReceiptFixture()
        key = f.receipt['paired_evidence_hash']
        node = next(n for n in f.nodes if n['sha256'] == key)
        value = parse_json((f.root / node['path']).read_bytes())
        for k in ('real', 'synthetic', 'null'): value[k] = [0, 0, 0]
        f.receipt['paired_evidence_hash'] = f.replace_manifest(key, value)
        f.receipt['losses'] = {'real': 0, 'synthetic': 0, 'null': 0}
        f.verify()

    def test_unsafe_paths_symlinks_and_error_redaction(self):
        for unsafe in ('../private-source-value', '/private-source-value', './private-source-value', 'private-source-value\\secret'):
            f = ReceiptFixture()
            f.nodes[0]['path'] = unsafe
            with self.assertRaises(EvidenceError) as cm: f.verify()
            self.assertNotIn('private-source-value', str(cm.exception))
        for directory in (False, True):
            f = ReceiptFixture()
            n = f.nodes[0]
            if directory:
                (f.root / 'link').symlink_to(f.root, target_is_directory=True)
                n['path'] = 'link/' + n['path']
            else:
                (f.root / 'link').symlink_to(f.root / n['path'])
                n['path'] = 'link'
            with self.assertRaises(EvidenceError): f.verify()

    def test_confirmation_requires_bound_frozen_protocol(self):
        f = ReceiptFixture(confirmation=True)
        f.verify()
        frozen_key = f.receipt['freeze_hash']
        n = next(n for n in f.nodes if n['sha256'] == frozen_key)
        freeze = parse_json((f.root / n['path']).read_bytes())
        freeze['cell_hash'] = 'a' * 64
        f.receipt['freeze_hash'] = f.replace_manifest(frozen_key, freeze)
        with self.assertRaises(EvidenceError): f.verify()
        f = ReceiptFixture()
        f.receipt['identity']['partition_role'] = 'confirmation'
        with self.assertRaises(EvidenceError): f.verify()

    def test_failed_dispositions_never_become_verified_success(self):
        for status in ('unsupported', 'timeout', 'oom', 'invalid_output', 'byte_infeasible', 'protocol_violation', 'unresolved'):
            f = ReceiptFixture()
            f.receipt.update(execution=status, losses=None, paired_evidence_hash=None)
            with self.subTest(status=status), self.assertRaises(EvidenceError): f.verify()


class CacheTests(unittest.TestCase):
    def test_atomic_completion_roundtrip_and_dependency_reverification(self):
        f = ReceiptFixture()
        cache = f.root / 'cache'
        v = f.verify()
        path = commit_completed_receipt(cache, v)
        self.assertEqual(path.name, scientific_cache_key(v.identity) + '.json')
        self.assertEqual(path.stat().st_mode & 0o077, 0)
        self.assertEqual(cache.stat().st_mode & 0o077, 0)
        self.assertEqual(load_completed_receipt(cache, f.root, f.policy()).receipt_hash, v.receipt_hash)
        (f.root / f.nodes[0]['path']).write_bytes(b'changed-after-caching')
        with self.assertRaises(EvidenceError): load_completed_receipt(cache, f.root, f.policy())

    def test_partial_corrupt_and_forged_completion_marker_fail(self):
        for state in ('partial_only', 'truncated', 'false_marker', 'wrong_digest'):
            f = ReceiptFixture()
            cache = f.root / 'cache'
            path = commit_completed_receipt(cache, f.verify())
            if state == 'partial_only': path.rename(cache / '.staging-interrupted')
            elif state == 'truncated': path.write_bytes(b'{')
            else:
                envelope = parse_json(path.read_bytes())
                if state == 'false_marker': envelope['state'] = 'running'
                else: envelope['receipt_hash'] = 'a' * 64
                path.write_bytes(canonical_json(envelope))
            with self.subTest(state=state), self.assertRaises(EvidenceError): load_completed_receipt(cache, f.root, f.policy())

    def test_final_cache_cannot_be_reused_by_selection_policy(self):
        f = ReceiptFixture(confirmation=True)
        cache = f.root / 'cache'
        commit_completed_receipt(cache, f.verify())
        policy = f.policy()
        identity = policy.identity.value
        identity['partition_role'] = 'selection'
        policy.identity = EvaluationIdentity(identity)
        with self.assertRaises(EvidenceError): load_completed_receipt(cache, f.root, policy)


if __name__ == '__main__':
    unittest.main()
