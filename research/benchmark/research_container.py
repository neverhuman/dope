"""Bounded lossless artifact-container draft; not a production wire format."""
import hashlib
import struct
import zlib

MAGIC = b'DOPECR01'
HEADER = struct.Struct('<8sII32s32s')
DECODED_LIMIT = 65536


def digest(body):
    return hashlib.sha256(body).digest()


def encode(model, projection, level):
    if level not in (1, 9) or not model or not projection or len(model) + len(projection) > DECODED_LIMIT:
        raise ValueError('research container input outside declared bound')
    return HEADER.pack(MAGIC, len(model), len(projection), digest(model), digest(projection)) + zlib.compress(model + projection, level)


def decode(blob):
    if len(blob) < HEADER.size or len(blob) > DECODED_LIMIT + HEADER.size + 128:
        raise ValueError('research container length invalid')
    magic, model_size, projection_size, model_hash, projection_hash = HEADER.unpack_from(blob)
    if magic != MAGIC or model_size < 1 or projection_size < 1 or model_size + projection_size > DECODED_LIMIT:
        raise ValueError('research container header invalid')
    try:
        decoder = zlib.decompressobj()
        body = decoder.decompress(blob[HEADER.size:], DECODED_LIMIT + 1)
    except zlib.error as error:
        raise ValueError('research container compressed payload invalid') from error
    if (len(body) != model_size + projection_size or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('research container decoded length invalid')
    model, projection = body[:model_size], body[model_size:]
    if digest(model) != model_hash or digest(projection) != projection_hash:
        raise ValueError('research container payload digest changed')
    return model, projection
