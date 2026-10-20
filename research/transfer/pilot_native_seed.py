"""Explicit successor seed policy; never reinterpret the immutable V3 plan."""
from dataclasses import dataclass
import hashlib
import json

FORMAT = "dope-e03-native-learner-seed32-v1"
OWNER_GRANT_SHA256 = "cdedea8f95fe054a387a82830dec6173333495cf356f9e082b54035d0068aaa4"
V3_PLAN_SHA256 = "7559801d9b57556c8992d8c810efe0f8a0d7e976484cd516b0a411ea06c3bac2"
ORIGINAL_STREAM_HEX = "0x000d0beef2026105"


def derive_native_seed32(original_stream, policy=FORMAT):
    if policy != FORMAT or type(original_stream) is not int or not 0 <= original_stream < 1 << 64:
        raise ValueError("native_seed_policy_invalid")
    encoded = FORMAT.encode("ascii") + b"\0" + original_stream.to_bytes(8, "little")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:4], "little"), digest.hex()


@dataclass(frozen=True)
class NativeSeed:
    original_stream_hex: str
    format: str = FORMAT
    owner_grant_sha256: str = OWNER_GRANT_SHA256
    original_v3_plan_sha256: str = V3_PLAN_SHA256

    def __post_init__(self):
        value = self.original_stream_hex
        if (type(value) is not str or len(value) != 18 or not value.startswith("0x") or
                any(c not in "0123456789abcdef" for c in value[2:]) or
                self.owner_grant_sha256 != OWNER_GRANT_SHA256 or
                self.original_v3_plan_sha256 != V3_PLAN_SHA256):
            raise ValueError("native_seed_provenance_invalid")
        derive_native_seed32(int(value, 16), self.format)

    @property
    def seed32(self):
        return derive_native_seed32(int(self.original_stream_hex, 16), self.format)[0]

    @property
    def provenance(self):
        return {"format": self.format, "original_stream_hex": self.original_stream_hex,
                "input_sha256": derive_native_seed32(int(self.original_stream_hex, 16), self.format)[1],
                "seed32": self.seed32, "owner_grant_sha256": self.owner_grant_sha256,
                "original_v3_plan_sha256": self.original_v3_plan_sha256}

    @property
    def sha256(self):
        encoded = json.dumps(self.provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
