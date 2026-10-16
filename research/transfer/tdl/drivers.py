"""Evaluator-computed grouped importance consumer; no cross-model SHAP (T-032).

Ceilings and stable signs come from independent real-real controls on the same
real test rows. Required components are combined by minimum, never a blend.
"""
from .common import MetricError, failure, number, opt_default, require, sum_checked, unit, vector


def mass_agreement(real, synthetic):
    rtotal, stotal = sum_checked(real), sum_checked(synthetic)
    require(rtotal > 0, "degenerate_real_importance")
    if stotal <= 0:
        return 0.0
    l1 = sum_checked([abs(r / rtotal - s / stotal) for r, s in zip(real, synthetic)])
    return max(0.0, 1.0 - 0.5 * l1)


def rank_agreement(real, synthetic):
    """Pairwise rank agreement: ties on one side receive half credit."""
    pairs, score = 0, 0.0
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            r = (real[i] > real[j]) - (real[i] < real[j])
            s = (synthetic[i] > synthetic[j]) - (synthetic[i] < synthetic[j])
            pairs += 1
            score += 1.0 if r == s else (0.5 if r == 0 or s == 0 else 0.0)
    return score / pairs if pairs else 1.0


def evaluate(data):
    try:
        require(type(data) is dict)
        required = data.get("required_components", ["mass", "rank", "sign", "interaction"])
        require(type(required) is list and bool(required) and all(type(c) is str for c in required), "invalid_components")
        require(len(required) == len(set(required)), "invalid_components")
        require(set(required) <= {"mass", "rank", "sign", "interaction"}, "invalid_components")
        if any(data.get(k) is None for k in ("real_importances", "synthetic_importances", "real_real_ceiling")):
            return failure("missing_driver_component", "incomplete")
        real = vector(data["real_importances"], True)
        syn = vector(data["synthetic_importances"], True)
        require(len(real) == len(syn), "importance_shape_mismatch")
        if sum_checked(real) <= 0:
            return failure("degenerate_real_importance", "uninformative")
        raw = {"mass": mass_agreement(real, syn), "rank": rank_agreement(real, syn)}
        if "sign" in required:
            if data.get("real_signs") is None or data.get("synthetic_signs") is None:
                return failure("missing_driver_component", "incomplete")
            rs, ss = data["real_signs"], data["synthetic_signs"]
            require(type(rs) is list and type(ss) is list and len(rs) == len(ss) == len(real), "sign_shape_mismatch")
            require(all(type(v) is int and v in (-1, 0, 1) for v in rs + ss), "invalid_sign")
            active = [i for i, sign in enumerate(rs) if sign != 0 and real[i] > 0]
            if not active:
                return failure("no_stable_direction", "uninformative")
            raw["sign"] = sum(rs[i] == ss[i] and syn[i] > 0 for i in active) / len(active)
        if "interaction" in required:
            if data.get("real_interactions") is None or data.get("synthetic_interactions") is None:
                return failure("missing_driver_component", "incomplete")
            ri, si = vector(data["real_interactions"], True), vector(data["synthetic_interactions"], True)
            require(len(ri) == len(si), "interaction_shape_mismatch")
            # Preserve absolute strength as well as distribution; losing an
            # interaction cannot pass merely by retaining its normalized rank.
            rt, st = sum_checked(ri), sum_checked(si)
            if rt <= 0:
                return failure("no_stable_interaction", "uninformative")
            strength = min(rt, st) / max(rt, st)
            raw["interaction"] = mass_agreement(ri, si) * strength
        ceilings = data["real_real_ceiling"]
        require(type(ceilings) is dict, "invalid_ceiling")
        floor = unit(opt_default(data, "stability_floor", 0.70))
        parsed_ceilings = {}
        for component in required:
            if ceilings.get(component) is not None:
                parsed_ceilings[component] = unit(ceilings[component])
        if any(ceilings.get(component) is None for component in required):
            return failure("missing_driver_component", "incomplete")
        normalized = {}
        for component in required:
            ceiling = parsed_ceilings[component]
            if ceiling <= 0 or ceiling < floor:
                return failure("unstable_real_drivers", "uninformative")
            normalized[component] = min(1.0, max(0.0, raw[component] / ceiling))
        score = min(normalized.values())
        threshold = unit(opt_default(data, "driver_threshold", 0.80))
        return {"status": "complete", "reason": None,
                "raw_components": {k: raw[k] for k in required},
                "normalized_components": normalized, "driver_fidelity": score,
                "driver_pass": score >= threshold}
    except (MetricError, OverflowError) as error:
        return failure(str(error) if isinstance(error, MetricError) else "nonfinite_or_invalid_number")
