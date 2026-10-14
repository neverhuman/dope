"""Objective labels. A stored-target fit is not retraining utility."""

from __future__ import annotations


def label_objective(kind: str, value: float) -> dict:
    if kind == "stored_target_fit":
        return {"label": "stored_target_fit", "value": float(value), "retraining_utility": None}
    if kind == "measured_retraining":
        return {"label": "measured_retraining", "value": float(value), "retraining_utility": float(value)}
    raise ValueError("stored-target fit is not retraining utility")
