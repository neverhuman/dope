"""Opaque controls for frozen container provenance and complete byte charges."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch
import zlib

from research.benchmark import research_container as codec
from research.benchmark import research_container_members as m


def sha(body):
    return hashlib.sha256(body).hexdigest()


class MemberControls(unittest.TestCase):
    def setUp(self):
        self.model = b'opaque model' * 200
        self.projection = b'opaque projection' * 100
        self.blob = codec.encode(self.model, self.projection, 9)
        self.expected = dict(artifact_sha256=sha(self.blob),
                             artifact_bytes=len(self.blob), model_sha256=sha(self.model),
                             projection_sha256=sha(self.projection))

    def test_complete_charge_includes_header_and_both_members(self):
        for level in (1, 9):
            blob = codec.encode(self.model, self.projection, level)
            expected = dict(self.expected, artifact_sha256=sha(blob), artifact_bytes=len(blob))
            result = m.decode_verified_members(blob, **expected)
            self.assertEqual((result.model, result.projection), (self.model, self.projection))
            self.assertEqual(result.artifact_bytes, codec.HEADER.size + len(zlib.compress(self.model + self.projection, level)))
            self.assertEqual(result.artifact_sha256, sha(blob))
            self.assertEqual(m.decode_verified_members(blob, **expected), result)
            with self.assertRaises(FrozenInstanceError):
                result.model = b'changed'
            self.assertIs(type(result.model), bytes)
            self.assertIs(type(result.projection), bytes)

    def test_cap_matches_unchanged_product_contract(self):
        root = Path(__file__).resolve().parents[3]
        production = json.loads((root / 'production/kpi-contract.json').read_text())
        study = json.loads((root / 'research/benchmark/contract.json').read_text())
        self.assertEqual(m.ARTIFACT_CAP, production['tier_byte_limits']['l3'])
        self.assertEqual(m.ARTIFACT_CAP, study['tier_byte_limits']['l3'])

    def test_identity_or_charge_failure_rejects_before_decode(self):
        cases = [dict(artifact_sha256='0' * 64),
                 dict(artifact_bytes=len(self.blob) - codec.HEADER.size),
                 dict(artifact_bytes=len(self.blob) + 1),
                 dict(artifact_bytes=True), dict(artifact_bytes=float(len(self.blob))),
                 dict(artifact_bytes=m.ARTIFACT_CAP + 1), dict(artifact_bytes=0)]
        for change in cases:
            with self.subTest(change=change), patch.object(codec, 'decode') as decode:
                with self.assertRaises(ValueError):
                    m.decode_verified_members(self.blob, **dict(self.expected, **change))
                decode.assert_not_called()

    def test_invalid_external_member_digest_rejects_before_decode(self):
        for name in ('artifact_sha256', 'model_sha256', 'projection_sha256'):
            for value in (None, 1, '0' * 63, 'A' * 64, 'x' * 64):
                with self.subTest(name=name, value=value), patch.object(codec, 'decode') as decode:
                    with self.assertRaises(ValueError):
                        m.decode_verified_members(self.blob, **dict(self.expected, **{name: value}))
                    decode.assert_not_called()

    def test_digest_subclasses_cannot_override_identity_comparisons(self):
        class Digest(str):
            def __ne__(self, other):
                return False

        for name in ('artifact_sha256', 'model_sha256', 'projection_sha256'):
            with self.subTest(name=name), patch.object(codec, 'decode') as decode:
                with self.assertRaisesRegex(ValueError, '^research container frozen digest invalid$'):
                    m.decode_verified_members(self.blob, **dict(self.expected, **{name: Digest('0' * 64)}))
                decode.assert_not_called()

    def test_external_member_identities_cannot_be_replaced_by_header(self):
        for model, projection in [(self.model + b'change', self.projection),
                                  (self.model, self.projection + b'change')]:
            forged = codec.encode(model, projection, 9)
            expected = dict(self.expected, artifact_sha256=sha(forged), artifact_bytes=len(forged))
            with self.assertRaisesRegex(ValueError, '^research container member identity changed$'):
                m.decode_verified_members(forged, **expected)

    def test_mutable_blob_is_rejected_before_decode(self):
        for body in (bytearray(self.blob), memoryview(self.blob)):
            with patch.object(codec, 'decode') as decode:
                with self.assertRaises(ValueError):
                    m.decode_verified_members(body, **self.expected)
                decode.assert_not_called()

    def test_encoded_identity_cannot_be_refreshed_to_accept_bad_frames(self):
        changed_header = bytearray(self.blob)
        struct.pack_into('<I', changed_header, 8, len(self.model) + 1)
        for body in (self.blob + b'trailing', self.blob[:-1], bytes(changed_header),
                     self.blob + zlib.compress(b'another stream')):
            expected = dict(self.expected, artifact_sha256=sha(body), artifact_bytes=len(body))
            with self.subTest(size=len(body)), self.assertRaises(ValueError):
                m.decode_verified_members(body, **expected)

    def test_expansion_bound_still_applies_to_verified_encoded_bytes(self):
        blob = codec.HEADER.pack(codec.MAGIC, 1, 1, codec.digest(b'a'), codec.digest(b'b'))
        blob += zlib.compress(b'a' * (codec.DECODED_LIMIT * 4), 9)
        expected = dict(artifact_sha256=sha(blob), artifact_bytes=len(blob),
                        model_sha256=sha(b'a'), projection_sha256=sha(b'b'))
        with self.assertRaises(ValueError):
            m.decode_verified_members(blob, **expected)

    def test_errors_do_not_echo_private_members_or_digests(self):
        try:
            m.decode_verified_members(self.blob, **dict(self.expected, projection_sha256='0' * 64))
        except ValueError as error:
            self.assertEqual(str(error), 'research container member identity changed')
            self.assertNotIn(self.projection.decode(), str(error))
            self.assertNotIn(self.expected['projection_sha256'], str(error))
        else:
            self.fail('changed external projection identity accepted')


if __name__ == '__main__':
    unittest.main(verbosity=2)
