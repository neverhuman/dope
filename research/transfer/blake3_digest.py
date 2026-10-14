"""Default-length BLAKE3. Matches blake3::hash over the raw state||DPK bytes.

Keyed and derive-key modes are absent. This is the inspect-raw integrity
function, not a second sampler and not a package dependency.
"""

import struct

_IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_CHUNK_START = 1
_CHUNK_END = 2
_PARENT = 4
_ROOT = 8


def _rotr(value, bits):
    return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF


def _g(state, a, b, c, d, x, y):
    state[a] = (state[a] + state[b] + x) & 0xFFFFFFFF
    state[d] = _rotr(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotr(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + y) & 0xFFFFFFFF
    state[d] = _rotr(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] = _rotr(state[b] ^ state[c], 7)


def _compress(chaining, block, counter, block_len, flags):
    words = list(struct.unpack("<16I", block))
    state = [
        chaining[0], chaining[1], chaining[2], chaining[3],
        chaining[4], chaining[5], chaining[6], chaining[7],
        _IV[0], _IV[1], _IV[2], _IV[3],
        counter & 0xFFFFFFFF, (counter >> 32) & 0xFFFFFFFF,
        block_len, flags,
    ]
    for _ in range(7):
        _g(state, 0, 4, 8, 12, words[0], words[1])
        _g(state, 1, 5, 9, 13, words[2], words[3])
        _g(state, 2, 6, 10, 14, words[4], words[5])
        _g(state, 3, 7, 11, 15, words[6], words[7])
        _g(state, 0, 5, 10, 15, words[8], words[9])
        _g(state, 1, 6, 11, 12, words[10], words[11])
        _g(state, 2, 7, 8, 13, words[12], words[13])
        _g(state, 3, 4, 9, 14, words[14], words[15])
        words = [words[index] for index in _PERM]
    for index in range(8):
        state[index] ^= state[index + 8]
    return state[:8]


def _chunk(data, index, root):
    blocks = [data[offset:offset + 64] for offset in range(0, len(data), 64)] or [b""]
    chaining = _IV
    last = len(blocks) - 1
    for block_index, block in enumerate(blocks):
        flags = 0
        if block_index == 0:
            flags |= _CHUNK_START
        if block_index == last:
            flags |= _CHUNK_END
            if root:
                flags |= _ROOT
        chaining = _compress(chaining, block.ljust(64, b"\0"), index, len(block), flags)
    return chaining


def _parent(left, right, flags):
    block = struct.pack("<8I", *left) + struct.pack("<8I", *right)
    return _compress(_IV, block, 0, 64, flags)


def blake3_hash(data):
    """32-byte unkeyed BLAKE3 of data."""
    if type(data) is not bytes:
        raise TypeError("blake3 input must be bytes")
    count = max(1, (len(data) + 1023) // 1024)
    nodes = [_chunk(data[index * 1024:(index + 1) * 1024], index, count == 1) for index in range(count)]
    while len(nodes) > 1:
        merged = []
        index = 0
        while index < len(nodes):
            if index + 1 == len(nodes):
                merged.append(nodes[index])
                break
            flags = _PARENT | (_ROOT if len(nodes) == 2 else 0)
            merged.append(_parent(nodes[index], nodes[index + 1], flags))
            index += 2
        nodes = merged
    return struct.pack("<8I", *nodes[0])
