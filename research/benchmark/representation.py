"""Real-train quantile map, HMAC manifest, and the MFS-v3 encoder distance.

The map is fit on real fit rows and applied to every other table. Refitting on
synthetic rows is a different map. Manifests store HMAC digests, not source text.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path


NORMALIZER = "empirical_midrank_quantile_v1"
HEADLINE_ENCODER = "kumo_tabular_l"
TABULAR_PROTOCOL = "train_on_synthetic_score_on_real"
TABULAR_AUDITORS = ("kumo_tabular_l", "mitra_v2", "tabicl2")
CATEGORICAL_BUCKETS = 65536
SENSITIVITY_ENCODERS = ("kumo_tabular_s", "mitra_v2", "tabicl2", "a0")
REFUSED_ENCODERS = (
    "context2048",
    "foundation",
    "hyperion_v6all",
    "hyperion_v7",
    "tabdpt13",
    "tabpfn35",
    "ta_dope_v2",
    "ta_hyperion_v8",
    "ta_hyperion_v8braw",
    "target",
)


def _hmac(key: bytes, message: bytes) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()


def salt_id(salt: bytes) -> str:
    return hashlib.sha256(salt).hexdigest()[:16]


def column_hash(salt: bytes, text: str) -> str:
    return _hmac(salt, text.encode()).hex()[:16]


def category_code(salt: bytes, level: str) -> float:
    digest = _hmac(salt, level.encode())
    bucket = int.from_bytes(digest[:2], "big")
    return (bucket + 0.5) / CATEGORICAL_BUCKETS


def midrank_quantile(fit: list[float], values: list[float]) -> list[float] | None:
    if not fit or any(not math.isfinite(value) for value in (*fit, *values)):
        return None
    ordered = sorted(fit)
    n = float(len(ordered))

    def one(value: float) -> float:
        less = _bisect_left(ordered, value)
        right = _bisect_right(ordered, value)
        if right == 0:
            return 0.5 / n
        if less == len(ordered):
            return (n - 0.5) / n
        if less == right:
            left_image = (less - 0.5) / n
            right_image = (less + 0.5) / n
            x0 = ordered[less - 1]
            x1 = ordered[less]
            if x1 == x0:
                return left_image
            t = (value - x0) / (x1 - x0)
            return left_image + t * (right_image - left_image)
        total = sum((index + 0.5) / n for index in range(less, right))
        return total / (right - less)

    return [one(value) for value in values]


def _bisect_left(ordered: list[float], value: float) -> int:
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _bisect_right(ordered: list[float], value: float) -> int:
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if ordered[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def unit_half_distance(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    if any(not math.isfinite(value) for value in (*left, *right)):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    total = sum((a / left_norm - b / right_norm) ** 2 for a, b in zip(left, right))
    distance = math.sqrt(total) / 2.0
    if math.isfinite(distance) and 0.0 <= distance <= 1.0:
        return distance
    return None


def representation_closeness(
    distance: float,
    d_null: float,
    d_match: float,
    gap_mitra: float,
    gap_tabicl: float,
) -> float | None:
    values = (distance, d_null, d_match, gap_mitra, gap_tabicl)
    if any(not math.isfinite(value) for value in values):
        return None
    if not 0.0 <= distance <= 1.0 or not 0.0 <= d_null <= 1.0 or not 0.0 < d_match <= 1.0:
        return None
    gap = distance - d_null
    if gap >= 0.0 or gap_mitra == 0.0 or gap_tabicl == 0.0:
        return None
    if math.copysign(1.0, gap) != math.copysign(1.0, gap_mitra):
        return None
    if math.copysign(1.0, gap) != math.copysign(1.0, gap_tabicl):
        return None
    return min(1.0, max(0.0, 1.0 - distance / d_match))


def tabular_transfer_ok(normalizer: str, protocol: str, auditors: list[str]) -> bool:
    """Utility counts only for the clean tabular models on the shared map."""
    return (
        normalizer == NORMALIZER
        and protocol == TABULAR_PROTOCOL
        and len(auditors) == len(TABULAR_AUDITORS)
        and all(name in auditors for name in TABULAR_AUDITORS)
    )


def artifact_has_cleartext(artifact: bytes, forbidden: list[bytes]) -> bool:
    return any(needle and needle in artifact for needle in forbidden)


def _inside_repo(directory: Path, repo_root: Path) -> bool:
    root = repo_root.resolve()
    cursor = directory if directory.is_absolute() else Path.cwd() / directory
    while True:
        if cursor.exists():
            resolved = cursor.resolve()
            return resolved == root or root in resolved.parents
        parent = cursor.parent
        if parent == cursor:
            return False
        cursor = parent


def write_local_lookup(
    directory: Path,
    repo_root: Path,
    salt: bytes,
    columns: list[tuple[str, int, int]],
    categories: list[str] | None = None,
) -> Path:
    """Write the salt and hash-to-label map outside the repository, mode 0600."""
    directory = Path(directory)
    if _inside_repo(directory, Path(repo_root)):
        raise ValueError("column lookup must stay outside the repository")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = {
        "categories": [
            {"hash": column_hash(salt, label), "label": label} for label in (categories or [])
        ],
        "columns": [
            {"dtype": dtype, "hash": column_hash(salt, name), "label": name, "role": role}
            for name, dtype, role in columns
        ],
        "format": "dope-mfs-v3-column-lookup",
        "normalizer": NORMALIZER,
        "salt_hex": salt.hex(),
        "salt_id": salt_id(salt),
    }
    path = directory / "mfs-v3-column-lookup.json"
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    path.chmod(0o600)
    return path


def hash_manifest(salt: bytes, columns: list[tuple[str, int, int]], encoder: str, byte_length: int) -> dict:
    return {
        "byte_length": byte_length,
        "columns": [
            {"dtype": dtype, "hash": column_hash(salt, name), "role": role}
            for name, dtype, role in columns
        ],
        "encoder": encoder,
        "normalizer": NORMALIZER,
        "salt_id": salt_id(salt),
    }
