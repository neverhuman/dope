//! Paired full-family max-T bounds and step-down tests of zero nulls.
//! Subset critical values apply only to tests, never to tightened intervals.
use super::*;

pub fn simultaneous_ucb(boot: &BootstrapEffects, alpha: f64) -> MetricResult<Value> {
    require(
        alpha.is_finite() && alpha > 0.0 && alpha < 0.5,
        "invalid_inference_alpha",
    )?;
    let refusal = |status| json!({"status": status, "bounds": null, "moment_assumption": "finite_variance", "distribution_free": false});
    if boot.status != "complete" {
        return Ok(refusal(boot.status));
    }
    if alpha * (boot.draws as f64) < 5.0 {
        return Ok(refusal("insufficient_tail_draws"));
    }
    require(
        boot.draws >= 99
            && !boot.samples.is_empty()
            && boot.samples.keys().eq(boot.estimates.keys())
            && boot.estimates.values().all(|v| v.is_finite()),
        "invalid_bootstrap_samples",
    )?;
    let names: Vec<_> = boot.samples.keys().collect();
    let mut scales = Vec::new();
    let mut deviations = Vec::new();
    for name in &names {
        let values = &boot.samples[*name];
        require(values.len() == boot.draws, "invalid_bootstrap_samples")?;
        let center = mean(values)?;
        let scale = (values.iter().map(|v| (v - center).powi(2)).sum::<f64>()
            / (boot.draws - 1) as f64)
            .sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Ok(refusal("insufficient_variation"));
        }
        scales.push(scale);
        deviations.push(
            values
                .iter()
                .map(|v| (v - boot.estimates[*name]) / scale)
                .collect::<Vec<_>>(),
        );
    }
    let mut bounds: BTreeMap<_, _> = names
        .iter()
        .map(|m| ((*m).clone(), json!({"estimate": boot.estimates[*m]})))
        .collect();
    for (sign, label) in [(1.0, "lower"), (-1.0, "upper")] {
        let mut maxima: Vec<_> = (0..boot.draws)
            .map(|b| {
                deviations
                    .iter()
                    .map(|c| sign * c[b])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        maxima.sort_by(f64::total_cmp);
        let pos = (1.0 - alpha) * (boot.draws - 1) as f64;
        let lo = pos.floor() as usize;
        let critical =
            (maxima[lo] + (maxima[pos.ceil() as usize] - maxima[lo]) * (pos - lo as f64)).max(0.0);
        let mut order: Vec<_> = (0..names.len()).collect();
        order.sort_by(|a, b| {
            (sign * boot.estimates[names[*b]] / scales[*b])
                .total_cmp(&(sign * boot.estimates[names[*a]] / scales[*a]))
                .then_with(|| names[*a].cmp(names[*b]))
        });
        let mut previous: f64 = 0.0;
        for (rank, &i) in order.iter().enumerate() {
            let observed = sign * boot.estimates[names[i]] / scales[i];
            let exceed = (0..boot.draws)
                .filter(|&b| {
                    order[rank..]
                        .iter()
                        .map(|&j| sign * deviations[j][b])
                        .fold(f64::NEG_INFINITY, f64::max)
                        >= observed
                })
                .count();
            previous = previous.max((1 + exceed) as f64 / (boot.draws + 1) as f64);
            bounds.get_mut(names[i]).unwrap()[label] =
                json!(boot.estimates[names[i]] - sign * critical * scales[i]);
            bounds.get_mut(names[i]).unwrap()[format!("{label}_stepdown_p")] = json!(previous);
        }
    }
    Ok(
        json!({"status": "complete", "alpha": alpha, "multiplicity": "step_down_max_t", "bound_method": "full_family_max_t", "simultaneous_one_sided_level": 1.0 - alpha, "two_sided_level_at_least": 1.0 - 2.0 * alpha, "bounds": bounds, "moment_assumption": "finite_variance", "distribution_free": false}),
    )
}
