"""Authenticate published source bytes using independently committed expectations.

The five ledger digests already existed in compute_panel.py/retained_evidence.py
at the original c36d7f2 statistical freeze. The receipt-panel digest is the later
pre-unseal source freeze in ops/ci/paper-review-inputs.json (53c09bb..04ade84),
whose own digest is independently fixed by paper-review-outputs.py. These are
published-input pins, not expectations supplied by the files being verified.
Future campaigns must explicitly freeze their own source expectations before
unsealing; this module preserves the published fit-11 campaign and its sources.
"""

import hashlib
from pathlib import Path

PUBLISHED_SOURCE_SHA256 = {
    "s3-lineage-record.json": "bbd0852c49f595315104261efa4eaa6f50db257e323b8c002aac6737a2dbe7d7",
    "density-matched-population-validation.json": "5264b88a40ad5efb21d9b119789caeb13e71e911a17f1d8b57f1d8e8ac31732a",
    "sdv-matched-population-validation.json": "4dc367eb78159a0386882d23ae1f99b0aa2cc9f425d1aae499b6f3453222a44e",
    "arf-matched-population-validation.json": "c69ce66e79b234bf5d1f9d56938450ba253ae39a3f6655a6fd586f890b8992f7",
    "s3-matched-forest-confirmation-validation.json": "97a25be902954cad16c5b5802d0e0ba468dea104221efa2465434c94e6774a1f",
    "review-fixes-receipts-v1/panel.json": "f0ecf5929825f1f07d877d4e01714f1ebd55868db3b863482f46be0f27ffdf47",
}


def authenticated_published_bytes(results: Path, names) -> dict[str, bytes]:
    """Capture and authenticate every requested source before a caller decodes.

    A mounted copy may be selected by results, but it must contain the exact
    frozen bytes. Callers decode these returned buffers without reopening paths.
    """
    names = tuple(dict.fromkeys(names))
    if any(name not in PUBLISHED_SOURCE_SHA256 for name in names):
        raise ValueError("unregistered published source")
    root = results.resolve()
    captured = {}
    for name in names:
        path = results / name
        if path.is_symlink() or path.resolve() != root / name:
            raise ValueError("redirected published source")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != PUBLISHED_SOURCE_SHA256[name]:
            raise ValueError("published source digest mismatch")
        captured[name] = raw
    return captured
