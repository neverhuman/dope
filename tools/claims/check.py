#!/usr/bin/env python3
"""Entry point for the public firewall in the existing paper claim registry."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/whitepaper/scripts"))
from claim_registry import public_failures  # noqa: E402

if __name__ == "__main__":
    failures = public_failures()
    print("\n".join(failures) if failures else "Public claims firewall passed")
    raise SystemExit(bool(failures))
