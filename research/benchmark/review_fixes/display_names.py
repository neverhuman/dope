"""Reader-facing names for identifiers printed in the paper's tables.

Only printed table text changes. Ledgers, panels, JSON outputs, receipts and
every wire identity keep the raw identifier. Lookup is by exact match, so a
dataset name or any value not listed here is printed unchanged.
"""

from __future__ import annotations

DISPLAY = {
    # Methods.
    "GaussianCopula": "Gaussian copula",
    "independent_marginals": "Independent marginals",
    "ForestDiffusion/Forest-Flow": "Forest-Flow",
    "predictor_only": "Predictor-only control",
    "real_bootstrap_4n": "Real rows resampled",
    # Configurations.
    "native_selected": "native selected",
    "author_default": "author default",
    "features12_steps2048": "DOPE profile",
    "scaled_200_vae_1000_diffusion": "scaled schedule",
    # Auditors.
    "catboost": "CatBoost",
    "linear": "Linear",
    "mlp": "MLP",
    # Privacy and detector metrics.
    "dcr_validation_median": "DCR validation",
    "dcr_fit_median": "DCR fit",
    "nndr_fit_median": "NNDR fit",
    "distance_mia_auc": "Distance MIA AUC",
    "c2st_catboost_auc": "C2ST CatBoost AUC",
    # Reasons a cell is unavailable.
    "sample_csv_unresolved": "sample file unresolved",
    "sample_csv_absent": "sample file absent",
}


def display(value) -> str:
    """The printed name of one identifier; unknown values pass through unchanged."""
    return DISPLAY.get(str(value), str(value))
