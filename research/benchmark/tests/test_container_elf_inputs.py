"""Opaque ELF metadata controls; no executable or library starts."""
import hashlib
import struct
import unittest
from unittest.mock import patch

from research.benchmark import container_elf_inputs as elf


def opaque_object():
    blob = bytearray(1024)
    ident = b'\x7fELF\x02\x01\x01' + b'\0' * 9
    struct.pack_into('<16sHHIQQQIHHHHHH', blob, 0, ident, 3, 62, 1, 0, 64, 0, 0,
                     64, 56, 3, 0, 0, 0)
    strings = b'\0libopaque.so\0$ORIGIN/opaque\0opaque-audit.so\0'
    interp = b'/opaque/loader\0'
    entries = [(5, 0x400000 + 768), (10, len(strings)), (1, 1),
               (29, strings.index(b'$ORIGIN')), (0x6ffffefc, strings.index(b'opaque-audit')),
               (0, 0)]
    for index, row in enumerate([(1, 4, 0, 0x400000, 0, 1024, 1024, 4096),
                                 (3, 4, 384, 0, 0, len(interp), len(interp), 1),
                                 (2, 4, 512, 0x400200, 0, len(entries) * 16, len(entries) * 16, 8)]):
        struct.pack_into('<IIQQQQQQ', blob, 64 + index * 56, *row)
    blob[384:384 + len(interp)] = interp
    blob[768:768 + len(strings)] = strings
    for index, row in enumerate(entries):
        struct.pack_into('<qQ', blob, 512 + index * 16, *row)
    return blob


class ElfInputControls(unittest.TestCase):
    def inspect(self, blob):
        blob = bytes(blob)
        with patch('subprocess.run', side_effect=AssertionError('candidate process started')), \
                patch('subprocess.Popen', side_effect=AssertionError('candidate process started')):
            return elf.inspect_elf_inputs(blob, hashlib.sha256(blob).hexdigest())

    def rejected(self, blob):
        with self.assertRaisesRegex(ValueError, '^compressed ELF input inspection rejected$'):
            self.inspect(blob)

    def test_declarations_are_unresolved_and_unadmitted(self):
        result = self.inspect(opaque_object())
        self.assertEqual(result['interpreter'], '/opaque/loader')
        self.assertEqual(result['declarations']['needed'], ['libopaque.so'])
        self.assertEqual(result['declarations']['runpath'], ['$ORIGIN/opaque'])
        self.assertEqual(result['declarations']['audit'], ['opaque-audit.so'])
        for flag in ['loader_resolution_verified', 'full_runtime_closure_verified',
                     'execution_admitted', 'official_tests_opened']:
            self.assertIs(result[flag], False)
        self.assertEqual(result['candidate_processes_started'], 0)
        self.assertTrue(all(result[key] is None for key in
                            ['mfs_v2', 'ptf_v1', 'release_safe', 'superiority']))

    def test_digest_precedes_header_parsing(self):
        with patch.object(elf.struct, 'unpack_from', side_effect=AssertionError('unbound ELF parsed')):
            with self.assertRaises(ValueError):
                elf.inspect_elf_inputs(bytes(opaque_object()), '0' * 64)

    def test_builtin_bytes_and_digest_required(self):
        class Permissive(str):
            def __eq__(self, other):
                return True
        blob = bytes(opaque_object()); digest = hashlib.sha256(blob).hexdigest()
        for data, value in [(bytearray(blob), digest), (blob, Permissive(digest)),
                            (blob, digest.upper()), (blob, None)]:
            with self.subTest(input_type=type(data).__name__, digest_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    elf.inspect_elf_inputs(data, value)

    def test_unsupported_header_rejected(self):
        for offset, value, fmt in [(4, 1, 'B'), (5, 2, 'B'), (6, 0, 'B'),
                                   (16, 1, 'H'), (18, 3, 'H'), (20, 0, 'I'),
                                   (32, 1, 'Q'), (52, 63, 'H'), (54, 55, 'H'),
                                   (56, 0, 'H'), (56, 1024, 'H')]:
            with self.subTest(offset=offset):
                blob = opaque_object(); struct.pack_into('<' + fmt, blob, offset, value)
                self.rejected(blob)

    def test_truncated_tables_and_segments_rejected(self):
        for length in [0, 63, 100, 600, 1000]:
            with self.subTest(length=length):
                self.rejected(opaque_object()[:length])
        blob = opaque_object(); struct.pack_into('<Q', blob, 64 + 40, 1)
        self.rejected(blob)

    def test_duplicate_interpreter_or_dynamic_rejected(self):
        for kind in [2, 3]:
            blob = opaque_object(); struct.pack_into('<I', blob, 64, kind)
            self.rejected(blob)

    def test_interpreter_must_be_one_absolute_terminated_string(self):
        for offset, value in [(384, ord('r')), (385, 0), (398, ord('x'))]:
            blob = opaque_object(); blob[offset] = value; self.rejected(blob)

    def test_dynamic_terminator_and_record_bounds_rejected(self):
        blob = opaque_object(); struct.pack_into('<qQ', blob, 512 + 5 * 16, 1, 1)
        self.rejected(blob)
        blob = opaque_object(); struct.pack_into('<Q', blob, 64 + 2 * 56 + 32, 95)
        self.rejected(blob)

    def test_dynamic_record_count_is_bounded(self):
        blob = opaque_object(); blob.extend(b'\0' * 100000)
        for offset in [64 + 32, 64 + 40]:
            struct.pack_into('<Q', blob, offset, len(blob))
        for offset in [64 + 2 * 56 + 32, 64 + 2 * 56 + 40]:
            struct.pack_into('<Q', blob, offset, 4097 * 16)
        self.rejected(blob)

    def test_dynamic_virtual_mapping_matches_file_offset(self):
        for address in [0xffff, 0x400300]:
            with self.subTest(address=address):
                blob = opaque_object()
                struct.pack_into('<Q', blob, 64 + 2 * 56 + 16, address)
                self.rejected(blob)

    def test_unmapped_or_ambiguous_string_table_rejected(self):
        blob = opaque_object(); struct.pack_into('<Q', blob, 512 + 8, 0xffff)
        self.rejected(blob)
        blob = opaque_object(); struct.pack_into('<H', blob, 56, 4)
        blob[64 + 3 * 56:64 + 4 * 56] = blob[64:64 + 56]
        self.rejected(blob)

    def test_missing_duplicate_or_unterminated_strings_rejected(self):
        blob = opaque_object(); struct.pack_into('<q', blob, 512, 999)
        self.rejected(blob)
        blob = opaque_object(); struct.pack_into('<qQ', blob, 512 + 4 * 16, 5, 0x400300)
        self.rejected(blob)
        blob = opaque_object(); struct.pack_into('<Q', blob, 512 + 2 * 16 + 8, 999)
        self.rejected(blob)
        blob = opaque_object(); blob[768:] = b'A' * (len(blob) - 768)
        self.rejected(blob)


if __name__ == '__main__':
    unittest.main()
