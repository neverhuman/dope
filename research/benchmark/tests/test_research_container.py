"""Bounded decoding controls using opaque bytes only, with no model execution."""
import struct
import unittest
import zlib

from research.benchmark import research_container as m


class ContainerControls(unittest.TestCase):
    def setUp(self):
        self.model = b'opaque model bytes' * 127
        self.projection = b'opaque learned projection bytes' * 173
        self.blob = m.encode(self.model, self.projection, 9)

    def rejects(self, body):
        with self.assertRaises(ValueError): m.decode(body)

    def test_exact_roundtrip_and_encoding_replay(self):
        for level in [1, 9]:
            blob = m.encode(self.model, self.projection, level)
            self.assertEqual(m.decode(blob), (self.model, self.projection))
            self.assertEqual(blob, m.encode(self.model, self.projection, level))
            self.assertEqual(len(blob), m.HEADER.size + len(zlib.compress(self.model + self.projection, level)))

    def test_truncated_headers_rejected(self):
        for length in [0, 1, m.HEADER.size - 1]: self.rejects(self.blob[:length])

    def test_wrong_magic_rejected(self):
        self.rejects(b'WRONGHDR' + self.blob[8:])

    def test_header_digest_change_rejected(self):
        body = bytearray(self.blob);body[16] ^= 1;self.rejects(bytes(body))

    def test_uncompressed_length_change_rejected(self):
        body = bytearray(self.blob);struct.pack_into('<I', body, 8, len(self.model) + 1)
        self.rejects(bytes(body))

    def test_zero_length_member_rejected(self):
        body = bytearray(self.blob);struct.pack_into('<I', body, 8, 0)
        self.rejects(bytes(body))

    def test_declared_large_decoded_member_rejected(self):
        body = bytearray(self.blob);struct.pack_into('<I', body, 8, m.DECODED_LIMIT + 1)
        self.rejects(bytes(body))

    def test_trailing_bytes_rejected(self):
        self.rejects(self.blob + b'trailing')

    def test_second_stream_rejected(self):
        self.rejects(self.blob + zlib.compress(b'second stream'))

    def test_truncated_compressed_payload_rejected(self):
        self.rejects(self.blob[:-1])

    def test_corrupt_compressed_payload_rejected(self):
        self.rejects(self.blob[:m.HEADER.size] + b'not a deflate stream')

    def test_expansion_above_limit_rejected(self):
        # Two valid declared lengths cannot cause the decoder to allocate the
        # much larger expansion hidden in a forged compressed payload.
        blob = m.HEADER.pack(m.MAGIC, 1, 1, m.digest(b'a'), m.digest(b'b'))
        self.rejects(blob + zlib.compress(b'a' * (m.DECODED_LIMIT * 4), 9))

    def test_encoding_oversize_or_empty_members_rejected(self):
        for model, projection in [(b'', b'a'), (b'a', b''), (b'a' * m.DECODED_LIMIT, b'b')]:
            with self.assertRaises(ValueError): m.encode(model, projection, 9)

    def test_undeclared_compression_level_rejected(self):
        with self.assertRaises(ValueError): m.encode(b'a', b'b', 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
