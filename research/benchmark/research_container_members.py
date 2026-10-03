"""Verify owned research-container members; never initialize or sample a model."""
from dataclasses import dataclass
import hashlib
import re

from . import research_container as codec

ARTIFACT_CAP = 10_240


@dataclass(frozen=True)
class DecodedMembers:
    model: bytes
    projection: bytes
    artifact_bytes: int
    artifact_sha256: str


def decode_verified_members(blob: bytes, *, artifact_sha256: str,
                            artifact_bytes: int, model_sha256: str,
                            projection_sha256: str) -> DecodedMembers:
    """Use three externally frozen digests, including the entire encoded artifact.

    Returned bytes are immutable owned copies. The caller must separately verify
    its job/source/runtime and obtain capacity admission before model execution.
    """
    if (type(blob) is not bytes or type(artifact_bytes) is not int
            or not codec.HEADER.size <= artifact_bytes <= ARTIFACT_CAP):
        raise ValueError('research container charge invalid')
    for value in (artifact_sha256, model_sha256, projection_sha256):
        if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError('research container frozen digest invalid')
    if len(blob) != artifact_bytes or hashlib.sha256(blob).hexdigest() != artifact_sha256:
        raise ValueError('research container encoded identity changed')
    model, projection = codec.decode(blob)
    if (hashlib.sha256(model).hexdigest() != model_sha256
            or hashlib.sha256(projection).hexdigest() != projection_sha256):
        raise ValueError('research container member identity changed')
    return DecodedMembers(model, projection, artifact_bytes, artifact_sha256)
