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
    version = data.get("protocol_version", "v1") if type(data) is dict else "v1"
    if version == "v1":
        return _evaluate_v1(data)
    if version != "v2":
        return failure("unsupported_driver_protocol")
    try:
        out = _evaluate_v2(data)
    except (MetricError, OverflowError) as error:
        out = failure(str(error) if isinstance(error, MetricError) else "nonfinite_or_invalid_number")
    return {"format": "driver-result-v1", "protocol_version": "v2", **out}


def _evaluate_v1(data):
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
        for component, ceiling in parsed_ceilings.items():
            if ceiling <= 0 or ceiling < floor:
                return failure("unstable_real_drivers", "uninformative")
        if any(ceilings.get(component) is None for component in required):
            return failure("missing_driver_component", "incomplete")
        normalized = {}
        for component in required:
            ceiling = parsed_ceilings[component]
            normalized[component] = min(1.0, max(0.0, raw[component] / ceiling))
        score = min(normalized.values())
        threshold = unit(opt_default(data, "driver_threshold", 0.80))
        return {"status": "complete", "reason": None,
                "raw_components": {k: raw[k] for k in required},
                "normalized_components": normalized, "driver_fidelity": score,
                "driver_pass": score >= threshold}
    except (MetricError, OverflowError) as error:
        return failure(str(error) if isinstance(error, MetricError) else "nonfinite_or_invalid_number")


def _evaluate_v2(data):
    """Consume evaluator-owned C02 full-family bounds, never infer from a ratio.

    Components and positive floors are explicit draft inputs. Common pins bind
    aligned real/synthetic/independent-real arms on D00 PreparedView groups.
    Neither these diagnostics nor fixture bounds authorize a campaign release.
    """
    from ..drivers import mass_agreement as mass, paired_margin, positive_importances
    missing = lambda: failure("missing_driver_component", "incomplete")
    state = data.get("perturbation_status")
    if state in (None, "missing"):
        return failure("missing_perturbation", "incomplete")
    if state in ("failed", "timed_out", "unsupported"):
        return failure("perturbation_" + state, state)
    require(state == "complete", "invalid_perturbation_status")
    pins = ("query_sha256", "groups_sha256", "perturbation_sha256", "prepared_view_sha256")
    if any(data.get(key) is None for key in pins):
        return missing()
    for key in pins:
        pin = data[key]
        require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin), "invalid_driver_pin")
    if data.get("required_components") is None:
        return missing()
    required = data["required_components"]
    require(type(required) is list and all(type(c) is str for c in required), "invalid_components")
    require(len(set(required)) == len(required) and "mass" in required and
            set(required) <= {"mass", "rank", "sign", "interaction"}, "invalid_components")
    fields = ["stability_floor", "importance_floor", "synthetic_strength_floor",
              "real_importances", "synthetic_importances", "real_control_importances"]
    if "interaction" in required:
        fields += ["real_interactions", "synthetic_interactions", "real_control_interactions"]
    if "sign" in required:
        fields += ["real_signs", "synthetic_signs", "real_control_signs"]
    if any(data.get(key) is None for key in fields):
        return missing()
    stability_floor, importance_floor, strength_floor = [number(data[k]) for k in fields[:3]]
    require(0 < stability_floor <= 1 and importance_floor > 0 and 0 < strength_floor <= 1, "invalid_driver_floor")
    real, syn, control = [positive_importances(data[k]) for k in fields[3:6]]
    require(len(real) == len(syn) == len(control), "importance_shape_mismatch")
    if data.get("inference") is None:
        return failure("missing_simultaneous_bounds", "incomplete")
    inference = data["inference"]
    require(type(inference) is dict, "invalid_driver_bounds")
    if inference.get("status") != "complete":
        return failure("incomplete_driver_inference", "incomplete")
    require(inference.get("bound_method") == "full_family_max_t", "non_simultaneous_driver_bounds")
    bounds = inference.get("bounds")
    require(type(bounds) is dict, "invalid_driver_bounds")
    components, passed, fidelity = {}, True, 1.0
    for name in sorted(required):
        r, s, c = ([positive_importances(data[k]) for k in fields[6:9]]
                   if name == "interaction" else (real, syn, control))
        require(len(r) == len(s) == len(c), "importance_shape_mismatch")
        rt, st, ct = [sum_checked(v) for v in (r, s, c)]
        if rt < importance_floor or ct < importance_floor:
            return failure("degenerate_real_importance", "uninformative")
        if name == "rank":
            rs, rr = rank_agreement(r, s), rank_agreement(r, c)
        elif name == "sign":
            a, b, d = [data[k] for k in ("real_signs", "synthetic_signs", "real_control_signs")]
            require(all(type(v) is list and len(v) == len(r) for v in (a, b, d)), "sign_shape_mismatch")
            require(all(type(v) is int and v in (-1, 0, 1) for v in a + b + d), "invalid_sign")
            active = [i for i in range(len(r)) if a[i] != 0 and r[i] > 0]
            if not active:
                return failure("no_stable_direction", "uninformative")
            def agreement(signs, imp):
                return sum(a[i] == signs[i] and imp[i] > 0 for i in active) / len(active)
            rs, rr = agreement(b, s), agreement(d, c)
        else:
            rs, rr = mass(r, s), mass(r, c)
        if rr < stability_floor:
            return failure("unstable_real_drivers", "uninformative")
        margin = paired_margin(rs, rr)
        endpoints = {"margin": margin, "stability": rr, "real_strength": rt,
                     "control_strength": ct, "synthetic_strength": number(st / rt)}
        lcbs = {}
        for endpoint, estimate in endpoints.items():
            coordinate = name + "." + endpoint
            if coordinate not in bounds:
                return failure("missing_simultaneous_bounds", "incomplete")
            bound = bounds[coordinate]
            require(type(bound) is dict, "invalid_driver_bounds")
            require(abs(number(bound.get("estimate")) - estimate) <= 1e-12, "driver_bound_estimate_mismatch")
            lower = number(bound.get("lower"))
            require(lower <= estimate + 1e-12, "invalid_driver_bounds")
            lcbs[endpoint] = lower
        passed &= (lcbs["margin"] >= 0 and lcbs["stability"] >= stability_floor and
                   lcbs["real_strength"] >= importance_floor and lcbs["control_strength"] >= importance_floor and
                   lcbs["synthetic_strength"] >= strength_floor)
        ratio = min(1.0, max(0.0, rs / rr))
        fidelity = min(fidelity, ratio)
        components[name] = {"a_rs": rs, "a_rr": rr, "margin": margin,
                            "descriptive_ratio": ratio, "lcbs": lcbs}
    return {"status": "complete", "reason": None, "components": components,
            "driver_fidelity": fidelity, "driver_pass": bool(passed)}
