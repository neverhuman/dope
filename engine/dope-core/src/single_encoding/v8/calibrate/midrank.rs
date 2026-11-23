use crate::stats::norm_ppf;
use std::ops::Range;

pub(super) fn sorted_observations(fit: &[f32]) -> Vec<(usize, f64)> {
    let mut observed: Vec<_> = fit
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, &value)| (index, f64::from(value)))
        .collect();
    observed.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    observed
}

fn tie_tolerance(value: f64) -> f64 {
    1e-12 * value.abs().max(1.0)
}

pub(super) fn groups(observations: &[(usize, f64)]) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        let first = observations.get(start)?.1;
        let mut end = start + 1;
        while end < observations.len()
            && (observations[end].1 - first).abs() <= tie_tolerance(first)
        {
            end += 1;
        }
        let group = start..end;
        start = end;
        Some(group)
    })
}

pub(super) fn normal_score(mid: f32) -> f64 {
    norm_ppf(f64::from(mid).clamp(1e-4, 1.0 - 1e-4)).clamp(-3.5, 3.5)
}

/// Project observed packet-unit values through one FIT column's empirical midranks.
/// Ties use the fitter's CDF intervals. Values between FIT groups use their insertion rank.
/// Missing observations and columns with fewer than two finite FIT values map to zero.
pub fn project_fit_midrank(fit: &[f32], observed: &[f32]) -> Vec<f32> {
    let fit = sorted_observations(fit);
    if fit.len() < 2 {
        return vec![0.0; observed.len()];
    }
    let groups: Vec<_> = groups(&fit).collect();
    let n = fit.len() as f64;
    let fraction = |rank: usize| (rank as f64 / n) as f32;
    observed
        .iter()
        .map(|&value| {
            if !value.is_finite() {
                return 0.0;
            }
            let value = f64::from(value);
            let index = groups.partition_point(|group| {
                let first = fit[group.start].1;
                value - first > tie_tolerance(first)
            });
            let Some(group) = groups.get(index) else {
                return normal_score(1.0) as f32;
            };
            let first = fit[group.start].1;
            let lo = fraction(group.start);
            let hi = if first - value <= tie_tolerance(first) {
                fraction(group.end)
            } else {
                lo
            };
            normal_score(0.5 * (lo + hi)) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::project_fit_midrank;
    use crate::single_encoding::v8::calibrate::CdfIntervals;

    #[test]
    fn projection_matches_native_fit_ties_and_ignores_other_queries() {
        let fit = vec![2.0, f32::NAN, -1.0, 2.0, 0.0, -0.0, 5e-13, 2e-12, 17.0];
        let expected: Vec<_> = CdfIntervals::build(std::slice::from_ref(&fit)).z_mid[0]
            .iter()
            .map(|&value| value as f32)
            .collect();
        assert_eq!(project_fit_midrank(&fit, &fit), expected);
        let before = project_fit_midrank(&fit, &[-2.0, 1.0, 80.0, f32::NAN]);
        let after = project_fit_midrank(&fit, &[-2.0, 1.0, 80.0, 1e20]);
        assert_eq!(before[..3], after[..3]);
        assert!(before[0] < before[1] && before[1] < before[2]);
        assert_eq!(before[3], 0.0);
        assert_eq!(project_fit_midrank(&[f32::NAN, 2.0], &[8.0]), [0.0]);
    }
}
