"""One row per v2 arm: display name, macro code, color, marker, and role.

Tables, figures, macros, and the README read this registry, so a method has
one name and one color everywhere. Colors avoid the collisions of the older
palette (ARF and CTGAN, TabDDPM and independent marginals). This module reads
no measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTROL_GREY = "#8C8C8C"


@dataclass(frozen=True)
class Arm:
    method: str
    configuration: str
    code: str          # letters only; becomes part of LaTeX macro names
    display: str       # plain-text name for tables, figures, and the README
    color: str
    marker: str
    role: str          # headline, generator, control, diagnostic
    selection: str     # how the configuration was chosen; never "tuned" for an untuned arm


ARMS = (
    Arm("DOPE", "headline", "Dope", "DOPE (ours)", "#76B900", "o", "headline",
        "profile chosen on a six-lineage discovery panel; compile search on fit rows"),
    Arm("DOPE", "product_default", "DopeDef", "DOPE, untuned default", "#A6D96A", "o", "diagnostic",
        "trainer default; no selection"),
    Arm("DOPE", "historical_seed11", "DopeHist", "DOPE, historical seed 11", "#4D7F00", "o", "diagnostic",
        "historical population at fit seed 11"),
    Arm("DOPE", "headline_bnew", "DopeBnew", "DOPE, B_new seed 11", "#4D7F00", "o", "diagnostic",
        "diagnostic binary"),
    Arm("DOPE", "fourseed_bnew", "DopeBnewFour", "DOPE, B_new seeds 23-71", "#4D7F00", "o", "diagnostic",
        "diagnostic binary"),
    Arm("TabSyn", "scaled_200_vae_1000_diffusion", "TabSyn", "TabSyn (scaled schedule)", "#6A3D9A", "D",
        "generator", "author architecture, scaled epochs; no selection"),
    Arm("ARF", "author_default", "Arf", "ARF (author default)", "#CC79A7", "s", "generator",
        "author default; no selection"),
    Arm("ARF", "native_selected", "ArfNat", "ARF (native selected)", "#E7A6CE", "s", "generator",
        "held-out log density on validation rows"),
    Arm("GaussianCopula", "native_selected", "Gauss", "Gaussian copula", "#222222", "^", "generator",
        "held-out log density on validation rows"),
    Arm("Chow-Liu", "native_selected", "Chow", "Chow-Liu tree", "#E69F00", "v", "generator",
        "held-out log density on validation rows"),
    Arm("independent_marginals", "native_selected", "Ind", "Independent marginals", "#0072B2", "<",
        "generator", "held-out log density on validation rows"),
    Arm("CTGAN", "native_selected", "Ctgan", "CTGAN", "#882255", "P", "generator",
        "linear and MLP validation skill on the scoring rows"),
    Arm("TVAE", "native_selected", "Tvae", "TVAE", "#56B4E9", "X", "generator",
        "linear and MLP validation skill on the scoring rows"),
    Arm("ForestDiffusion/Forest-Flow", "native_selected", "Forest", "Forest-Flow (historical)", "#666666",
        "h", "generator", "author regression objective on a frozen grid"),
    Arm("TabDDPM", "author_default", "TabDdpm", "TabDDPM", "#D55E00", "p", "generator", "author default"),
    Arm("TabDiff", "author_default", "TabDiff", "TabDiff", "#009E73", "*", "generator", "author default"),
    Arm("GReaT", "author_default_distilgpt2", "Great", "GReaT", "#DDAA33", "8", "generator",
        "author default"),
    Arm("synthpop_CART", "author_default", "Synthpop", "synthpop CART", "#44AA99", "d", "generator",
        "author default"),
    Arm("SMOTE", "k5_uniform_interpolation_on_features_and_target", "Smote", "SMOTE", "#AA4499", "1",
        "generator", "fixed k = 5"),
    Arm("predictor_only", "control", "Pred", "Predictor-only control", CONTROL_GREY, "_", "control",
        "ridge regression; no selection"),
    Arm("real_bootstrap_4n", "control", "Boot", "Real rows resampled", CONTROL_GREY, "_", "control",
        "no model"),
)

BY_KEY = {(arm.method, arm.configuration): arm for arm in ARMS}
BY_CODE = {arm.code: arm for arm in ARMS}
AUDITOR_CODE = {"catboost": "Cb", "linear": "Lin", "mlp": "Mlp"}
AUDITOR_DISPLAY = {"catboost": "CatBoost", "linear": "Linear", "mlp": "MLP"}
SIZE_CODE = {1: "One", 4: "Four"}


def arm(method: str, configuration: str) -> Arm:
    try:
        return BY_KEY[(method, configuration)]
    except KeyError as error:
        raise KeyError(f"arm {method}/{configuration} is not registered") from error


def macro_suffix(auditor: str, size: int) -> str:
    return AUDITOR_CODE[auditor] + SIZE_CODE[size]
