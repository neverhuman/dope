//! Stratified family bootstrap of a task-macro mean, Holm adjustment and
//! the Wilson interval of a replicate rate.
//!
//! The observations are family values grouped by task (one arm's family
//! values, or the paired differences candidate minus baseline). One draw
//! resamples the families of each task with replacement, averages within the
//! task and then averages the task means equally, exactly like the statistic.
//! The resampling plan (how often each family is drawn in each draw) depends
//! on the family counts and the seed only: it is built once and reused, so
//! both arms of a comparison and every replicate of `null` read the same
//! draws.
//!
//! Conventions of `dope_core::dvs::v3::qualify`: with `n` sorted draws the
//! lower bound is the draw at `floor(lower * (n - 1))`, the upper bound the
//! one at `ceil(upper * (n - 1))`, and the one-sided p-value for
//! `delta <= 0` is `(#{draw <= 0} + 1) / (n + 1)`.

use super::seeds::KpiRng;
use serde::{Deserialize, Serialize};

/// Point value and bootstrap summary of one mean.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Interval {
    pub value: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub p_one_sided: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BootStats {
    pub overall: Interval,
    /// One interval per task, in the order the values were given.
    pub per_task: Vec<Interval>,
}

#[derive(Clone, Debug)]
pub struct BootPlan {
    draws: usize,
    /// Family counts per task.
    sizes: Vec<usize>,
    /// Per task, `draws x size` multiplicities, draw-major.
    counts: Vec<Vec<u32>>,
    lower_rank: usize,
    upper_rank: usize,
}

/// Ranks of the lower and upper bound among `draws` sorted draws.
pub fn ranks(draws: usize, lower: f64, upper: f64) -> (usize, usize) {
    let last = draws.saturating_sub(1);
    let at = |position: f64| (position.max(0.0) as usize).min(last);
    (
        at((lower * last as f64).floor()),
        at((upper * last as f64).ceil()),
    )
}

impl BootPlan {
    /// `sizes` are the family counts of the tasks present (each at least 1).
    pub fn new(sizes: &[usize], draws: usize, seed: u64, quantiles: (f64, f64)) -> BootPlan {
        assert!(draws > 0 && sizes.iter().all(|&size| size > 0));
        let mut rng = KpiRng::new(seed);
        let mut counts: Vec<Vec<u32>> = sizes.iter().map(|size| vec![0; draws * size]).collect();
        for draw in 0..draws {
            for (task, &size) in sizes.iter().enumerate() {
                let row = &mut counts[task][draw * size..(draw + 1) * size];
                for _ in 0..size {
                    row[rng.below(size)] += 1;
                }
            }
        }
        let (lower_rank, upper_rank) = ranks(draws, quantiles.0, quantiles.1);
        BootPlan {
            draws,
            sizes: sizes.to_vec(),
            counts,
            lower_rank,
            upper_rank,
        }
    }

    /// Summaries of `values[task][family]`, whose shape must match the plan.
    pub fn run(&self, values: &[Vec<f64>]) -> BootStats {
        assert!(
            values.len() == self.sizes.len()
                && values
                    .iter()
                    .zip(&self.sizes)
                    .all(|(v, &size)| v.len() == size),
            "values do not match the bootstrap plan"
        );
        let tasks = values.len() as f64;
        let mut overall = vec![0.0; self.draws];
        let mut per_task = Vec::with_capacity(values.len());
        let mut point = 0.0;
        for (task, values) in values.iter().enumerate() {
            let size = values.len();
            let mut draws = Vec::with_capacity(self.draws);
            for (draw, row) in self.counts[task].chunks_exact(size).enumerate() {
                let mut sum = 0.0;
                for (&count, &value) in row.iter().zip(values) {
                    sum += f64::from(count) * value;
                }
                let mean = sum / size as f64;
                overall[draw] += mean / tasks;
                draws.push(mean);
            }
            let mean = values.iter().sum::<f64>() / size as f64;
            point += mean / tasks;
            per_task.push(self.summarize(mean, draws));
        }
        BootStats {
            overall: self.summarize(point, overall),
            per_task,
        }
    }

    fn summarize(&self, value: f64, mut draws: Vec<f64>) -> Interval {
        let n = draws.len();
        let at_most_zero = draws.iter().filter(|d| **d <= 0.0).count();
        let (_, low, _) = draws.select_nth_unstable_by(self.lower_rank, f64::total_cmp);
        let ci_low = *low;
        let (_, high, _) = draws.select_nth_unstable_by(self.upper_rank, f64::total_cmp);
        Interval {
            value,
            ci_low,
            ci_high: *high,
            p_one_sided: (at_most_zero as f64 + 1.0) / (n as f64 + 1.0),
        }
    }
}

/// Holm step-down adjustment of raw p-values (ties keep their input order).
pub fn holm(ps: &[f64]) -> Vec<f64> {
    let k = ps.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| ps[a].total_cmp(&ps[b]).then(a.cmp(&b)));
    let mut adjusted = vec![1.0; k];
    let mut running = 0.0_f64;
    for (rank, &index) in order.iter().enumerate() {
        running = running.max(((k - rank) as f64 * ps[index]).min(1.0));
        adjusted[index] = running;
    }
    adjusted
}

/// Wilson score interval at 95% for `hits` successes in `trials`.
pub fn wilson95(hits: usize, trials: usize) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let (z, n) = (1.959_963_984_540_054_f64, trials as f64);
    let rate = hits as f64 / n;
    let denominator = 1.0 + z * z / n;
    let centre = rate + z * z / (2.0 * n);
    let half = z * (rate * (1.0 - rate) / n + z * z / (4.0 * n * n)).sqrt();
    (
        ((centre - half) / denominator).max(0.0),
        ((centre + half) / denominator).min(1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: (f64, f64) = (0.025, 0.975);

    #[test]
    fn every_draw_resamples_each_task_to_its_own_size() {
        let plan = BootPlan::new(&[5, 1, 3], 300, 7, Q);
        for (task, &size) in plan.sizes.iter().enumerate() {
            for row in plan.counts[task].chunks_exact(size) {
                assert_eq!(row.iter().map(|&c| c as usize).sum::<usize>(), size);
            }
        }
        assert_eq!(plan.counts[1], vec![1u32; 300]);
        assert_eq!(BootPlan::new(&[5, 1, 3], 300, 7, Q).counts, plan.counts);
        assert_ne!(BootPlan::new(&[5, 1, 3], 300, 8, Q).counts, plan.counts);
        assert!(plan.counts[0].contains(&0) && plan.counts[0].iter().any(|&c| c >= 2));
    }

    #[test]
    fn ranks_follow_the_qualify_convention() {
        assert_eq!(ranks(10_000, 0.025, 0.975), (249, 9750));
        assert_eq!(ranks(1000, 0.025, 0.975), (24, 975));
        assert_eq!(ranks(1, 0.025, 0.975), (0, 0));
    }

    #[test]
    fn constant_values_have_a_degenerate_interval_and_zero_ties_count_against() {
        let plan = BootPlan::new(&[4, 3, 2], 400, 1, Q);
        let stats = plan.run(&[vec![0.5; 4], vec![0.25; 3], vec![-0.75; 2]]);
        assert!(stats.overall.value.abs() < 1e-15);
        assert!(stats.overall.ci_low.abs() < 1e-15 && stats.overall.ci_high.abs() < 1e-15);
        assert_eq!(stats.per_task[0].ci_low, 0.5);
        assert_eq!(stats.per_task[2].ci_high, -0.75);
        assert_eq!(stats.per_task[0].p_one_sided, 1.0 / 401.0);
        assert_eq!(stats.per_task[2].p_one_sided, 1.0);
        // A difference of exactly zero is a draw <= 0: p = 1.
        let zero = plan
            .run(&[vec![0.0; 4], vec![0.0; 3], vec![0.0; 2]])
            .overall;
        assert_eq!(
            (zero.value, zero.ci_low, zero.ci_high, zero.p_one_sided),
            (0.0, 0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn interval_matches_a_direct_recomputation_and_weights_tasks_equally() {
        let values = vec![
            vec![0.10, -0.20, 0.30, 0.05, 0.00, 0.15],
            vec![1.0, -1.0],
            vec![0.2],
        ];
        let plan = BootPlan::new(&[6, 2, 1], 1000, 42, Q);
        let stats = plan.run(&values);
        let expected = (0.4 / 6.0 + 0.0 + 0.2) / 3.0;
        assert!((stats.overall.value - expected).abs() < 1e-15);
        // Replay the same stream with index draws instead of multiplicities.
        let mut rng = KpiRng::new(42);
        let mut draws = Vec::new();
        for _ in 0..1000 {
            let mut total = 0.0;
            for task in &values {
                let n = task.len();
                let sum: f64 = (0..n).map(|_| task[rng.below(n)]).sum();
                total += sum / n as f64 / 3.0;
            }
            draws.push(total);
        }
        draws.sort_by(f64::total_cmp);
        assert!((stats.overall.ci_low - draws[24]).abs() < 1e-12);
        assert!((stats.overall.ci_high - draws[975]).abs() < 1e-12);
        // A draw within rounding of zero may land on either side in the two sums.
        let at_most_zero = draws.iter().filter(|d| **d <= 0.0).count();
        let replay = (at_most_zero as f64 + 1.0) / 1001.0;
        assert!((stats.overall.p_one_sided - replay).abs() <= 2.0 / 1001.0);
        assert_eq!(stats.per_task[2].ci_low, stats.per_task[2].ci_high);
        assert_eq!(
            (stats.per_task[1].ci_low, stats.per_task[1].ci_high),
            (-1.0, 1.0)
        );
    }

    #[test]
    fn holm_orders_scales_and_keeps_monotone() {
        assert_eq!(holm(&[0.01]), [0.01]);
        let adjusted = holm(&[0.04, 0.01, 0.03]);
        // Sorted 0.01 x 3 = 0.03; 0.03 x 2 = 0.06; 0.04 x 1 = 0.04 -> running max 0.06.
        let expected = [0.06, 0.03, 0.06];
        for (got, want) in adjusted.iter().zip(expected) {
            assert!((got - want).abs() < 1e-15, "{adjusted:?}");
        }
        assert_eq!(holm(&[0.6, 0.5]), [1.0, 1.0]);
        // Ties share the adjusted value.
        assert_eq!(holm(&[0.02, 0.02]), [0.04, 0.04]);
        assert!(holm(&[]).is_empty());
    }

    #[test]
    fn wilson_bounds_a_rate() {
        let (low, high) = wilson95(0, 2000);
        assert!(low < 1e-12 && (high - 0.001_917).abs() < 1e-5, "{high}");
        let (low, high) = wilson95(50, 1000);
        assert!((low - 0.0381).abs() < 5e-4 && (high - 0.0653).abs() < 5e-4);
        assert_eq!(wilson95(0, 0), (0.0, 1.0));
    }
}
