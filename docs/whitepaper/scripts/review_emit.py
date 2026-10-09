#!/usr/bin/env python3
"""Emit review-fix tables from the committed validation ledgers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from research.benchmark.review_fixes.receipt_panel import main


if __name__ == "__main__":
    main()
