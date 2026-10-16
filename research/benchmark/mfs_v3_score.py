"""Deprecated historical MFS compatibility entry point."""
import warnings
from research.benchmark.legacy import mfs_v3_score as _legacy

warnings.warn("MFS benchmark metrics are historical; use bench.metrics",
              DeprecationWarning, stacklevel=2)
# Preserve private imports used by historical reducers and receipt verification.
globals().update({name: value for name, value in vars(_legacy).items()
                  if not name.startswith("__")})
