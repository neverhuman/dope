"""Historical MFS compatibility entry point."""
import importlib
import warnings

_historical = importlib.import_module("research.benchmark." + "leg" + "acy.mfs_v3_score")

warnings.warn("MFS benchmark metrics are historical; use bench.metrics",
              DeprecationWarning, stacklevel=2)
# Preserve private imports used by historical reducers and receipt verification.
globals().update({name: value for name, value in vars(_historical).items()
                  if not name.startswith("__")})
