use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::data::Table;

const MAX_ROWS: usize = 128;
const MAX_DIMENSIONS: usize = 32;
const SLICES: usize = 16;
const PRDC_K: usize = 5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessDiagnostics {
    pub version: u8,
    pub generation_seed: u64,
    pub synthetic_rows: usize,
    pub real_evaluation_rows: usize,
    pub synthetic_evaluation_rows: usize,
    pub embedding_dimensions: usize,
    pub prdc_k_real: usize,
    pub prdc_k_synthetic: usize,
    pub sliced_wasserstein: f64,
    pub sliced_wasserstein_fidelity: f64,
    pub multi_bandwidth_mmd_squared: f64,
    pub mmd_fidelity: f64,
    pub prdc_precision: f64,
    pub prdc_recall: f64,
    pub prdc_density: f64,
    pub prdc_coverage: f64,
    pub coverage_realism: f64,
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn sampled_indices(rows: usize, seed: u64) -> Vec<usize> {
    let mut smallest = BinaryHeap::new();
    for row in 0..rows {
        let candidate = (mix(seed ^ row as u64), row);
        if smallest.len() < MAX_ROWS {
            smallest.push(candidate);
        } else if smallest.peek().is_some_and(|largest| candidate < *largest) {
            smallest.pop();
            smallest.push(candidate);
        }
    }
    smallest
        .into_sorted_vec()
        .into_iter()
        .map(|(_, row)| row)
        .collect()
}

fn embedded(table: &Table, rows: &[usize]) -> Vec<Vec<f64>> {
    let input_dimensions = table.features * 2 + 1;
    let dimensions = input_dimensions.min(MAX_DIMENSIONS);
    let scale = (input_dimensions as f64).sqrt();
    rows.iter()
        .map(|&row| {
            let mut output = vec![0.0; dimensions];
            for dimension in 0..input_dimensions {
                let value = if dimension == input_dimensions - 1 {
                    f64::from(table.target[row]) - 0.5
                } else {
                    let feature = dimension / 2;
                    let raw = table.columns[feature][row];
                    if dimension % 2 == 0 {
                        if raw.is_finite() {
                            f64::from(raw) - 0.5
                        } else {
                            0.0
                        }
                    } else {
                        (!raw.is_finite()) as u8 as f64
                    }
                };
                if input_dimensions <= MAX_DIMENSIONS {
                    output[dimension] = value / scale;
                } else {
                    for slot in 0..3u64 {
                        let hash =
                            mix((dimension as u64).wrapping_mul(37) ^ slot.wrapping_mul(101));
                        let bucket = hash as usize % dimensions;
                        let sign = if hash & (1 << 32) == 0 { 1.0 } else { -1.0 };
                        output[bucket] += sign * value / (3.0f64).sqrt() / scale;
                    }
                }
            }
            output
        })
        .collect()
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum()
}

fn distances(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    left.iter()
        .map(|a| right.iter().map(|b| squared_distance(a, b)).collect())
        .collect()
}

fn neighborhood_k(rows: usize) -> usize {
    PRDC_K.min((rows - 1) / 2).max(1)
}

fn kth_radii(distances: &[Vec<f64>], k: usize) -> Vec<f64> {
    distances
        .iter()
        .enumerate()
        .map(|(row, values)| {
            let mut peers: Vec<_> = values
                .iter()
                .enumerate()
                .filter_map(|(column, &distance)| (column != row).then_some(distance))
                .collect();
            peers.sort_by(f64::total_cmp);
            peers[k - 1]
        })
        .collect()
}

fn prdc(real: &[Vec<f64>], synthetic: &[Vec<f64>]) -> (f64, f64, f64, f64) {
    let real_k = neighborhood_k(real.len());
    let synthetic_k = neighborhood_k(synthetic.len());
    let real_radii = kth_radii(&distances(real, real), real_k);
    let synthetic_radii = kth_radii(&distances(synthetic, synthetic), synthetic_k);
    let cross = distances(real, synthetic);
    let precision = (0..synthetic.len())
        .filter(|&synthetic_row| {
            (0..real.len()).any(|real_row| cross[real_row][synthetic_row] <= real_radii[real_row])
        })
        .count() as f64
        / synthetic.len() as f64;
    let recall = (0..real.len())
        .filter(|&real_row| {
            (0..synthetic.len()).any(|synthetic_row| {
                cross[real_row][synthetic_row] <= synthetic_radii[synthetic_row]
            })
        })
        .count() as f64
        / real.len() as f64;
    let density = (0..synthetic.len())
        .map(|synthetic_row| {
            (0..real.len())
                .filter(|&real_row| cross[real_row][synthetic_row] <= real_radii[real_row])
                .count()
        })
        .sum::<usize>() as f64
        / (real_k * synthetic.len()) as f64;
    let coverage = (0..real.len())
        .filter(|&real_row| {
            cross[real_row]
                .iter()
                .copied()
                .reduce(f64::min)
                .is_some_and(|distance| distance <= real_radii[real_row])
        })
        .count() as f64
        / real.len() as f64;
    (precision, recall, density, coverage)
}

fn sliced_wasserstein(real: &[Vec<f64>], synthetic: &[Vec<f64>]) -> f64 {
    let dimensions = real[0].len();
    (0..SLICES)
        .map(|slice| {
            let signs: Vec<_> = (0..dimensions)
                .map(|dimension| {
                    if mix(((slice as u64) << 32) ^ dimension as u64) & 1 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            let project = |rows: &[Vec<f64>]| {
                let mut values: Vec<_> = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(&signs)
                            .map(|(value, sign)| value * sign)
                            .sum::<f64>()
                            / (dimensions as f64).sqrt()
                    })
                    .collect();
                values.sort_by(f64::total_cmp);
                values
            };
            let left = project(real);
            let right = project(synthetic);
            (0..64)
                .map(|index| {
                    (left[index * (left.len() - 1) / 63] - right[index * (right.len() - 1) / 63])
                        .abs()
                })
                .sum::<f64>()
                / 64.0
        })
        .sum::<f64>()
        / SLICES as f64
}

fn multi_bandwidth_mmd_squared(real: &[Vec<f64>], synthetic: &[Vec<f64>]) -> f64 {
    let within_real = distances(real, real);
    let within_synthetic = distances(synthetic, synthetic);
    let cross = distances(real, synthetic);
    let mut positive_distances: Vec<_> = within_real
        .iter()
        .flatten()
        .chain(within_synthetic.iter().flatten())
        .chain(cross.iter().flatten())
        .copied()
        .filter(|distance| *distance > 0.0)
        .collect();
    positive_distances.sort_by(f64::total_cmp);
    let median = positive_distances
        .get(positive_distances.len() / 2)
        .copied()
        .unwrap_or(1e-6)
        .max(1e-6);
    let mean_kernel = |matrix: &[Vec<f64>], bandwidth: f64| {
        matrix
            .iter()
            .flatten()
            .map(|distance| (-distance / bandwidth).exp())
            .sum::<f64>()
            / matrix.iter().map(Vec::len).sum::<usize>() as f64
    };
    let mmd = [0.5, 1.0, 2.0]
        .into_iter()
        .map(|multiplier| {
            let bandwidth = median * multiplier;
            mean_kernel(&within_real, bandwidth) + mean_kernel(&within_synthetic, bandwidth)
                - 2.0 * mean_kernel(&cross, bandwidth)
        })
        .sum::<f64>()
        / 3.0;
    mmd.clamp(0.0, 2.0)
}

pub fn evaluate(
    real_holdout: &Table,
    synthetic: &Table,
    generation_seed: u64,
) -> Option<FitnessDiagnostics> {
    if real_holdout.rows < 6 || synthetic.rows < 6 || real_holdout.features != synthetic.features {
        return None;
    }
    let real_indices = sampled_indices(real_holdout.rows, 0x5245_414c_5f46_4954);
    let synthetic_indices = sampled_indices(synthetic.rows, generation_seed);
    let real = embedded(real_holdout, &real_indices);
    let synth = embedded(synthetic, &synthetic_indices);
    let sw = sliced_wasserstein(&real, &synth);
    let mmd_squared = multi_bandwidth_mmd_squared(&real, &synth);
    let (precision, recall, density, coverage) = prdc(&real, &synth);
    Some(FitnessDiagnostics {
        version: 1,
        generation_seed,
        synthetic_rows: synthetic.rows,
        real_evaluation_rows: real.len(),
        synthetic_evaluation_rows: synth.len(),
        embedding_dimensions: real[0].len(),
        prdc_k_real: neighborhood_k(real.len()),
        prdc_k_synthetic: neighborhood_k(synth.len()),
        sliced_wasserstein: sw,
        sliced_wasserstein_fidelity: (1.0 - sw / 0.25).clamp(0.0, 1.0),
        multi_bandwidth_mmd_squared: mmd_squared,
        mmd_fidelity: (1.0 - (mmd_squared / 2.0).sqrt()).clamp(0.0, 1.0),
        prdc_precision: precision,
        prdc_recall: recall,
        prdc_density: density,
        prdc_coverage: coverage,
        coverage_realism: 0.25 * precision
            + 0.25 * recall
            + 0.20 * density.min(1.0)
            + 0.30 * coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(values: &[f32]) -> Table {
        Table {
            rows: values.len(),
            features: 1,
            columns: vec![values.to_vec()],
            target: values.to_vec(),
        }
    }

    #[test]
    fn identical_samples_have_zero_distribution_distance_and_full_support() {
        let values = [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95];
        let report = evaluate(&table(&values), &table(&values), 7).unwrap();
        assert!(report.sliced_wasserstein < 1e-12);
        assert!(report.multi_bandwidth_mmd_squared < 1e-12);
        assert_eq!(report.prdc_precision, 1.0);
        assert_eq!(report.prdc_recall, 1.0);
        assert_eq!(report.prdc_coverage, 1.0);
    }

    #[test]
    fn mode_collapse_loses_coverage_and_repeat_is_exact() {
        let real = table(&[0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99]);
        let synth = table(&[0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04]);
        let first = evaluate(&real, &synth, 19).unwrap();
        let second = evaluate(&real, &synth, 19).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert!(first.prdc_coverage < 1.0);
        assert!(first.sliced_wasserstein > 0.0);
        assert!(first.multi_bandwidth_mmd_squared > 0.0);
    }

    #[test]
    fn too_few_rows_leave_evidence_unmeasured() {
        assert!(evaluate(&table(&[0.0, 1.0]), &table(&[0.0, 1.0]), 1).is_none());
    }

    #[test]
    fn wide_missing_data_uses_bounded_finite_embedding() {
        let rows = 8;
        let columns = (0..64)
            .map(|feature| {
                (0..rows)
                    .map(|row| {
                        if feature == 63 && row == 3 {
                            f32::NAN
                        } else {
                            ((feature + row) % 11) as f32 / 10.0
                        }
                    })
                    .collect()
            })
            .collect();
        let table = Table {
            rows,
            features: 64,
            columns,
            target: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        };
        let report = evaluate(&table, &table, 31).unwrap();
        assert_eq!(report.embedding_dimensions, MAX_DIMENSIONS);
        assert!(report.sliced_wasserstein.is_finite());
        assert!(report.multi_bandwidth_mmd_squared.is_finite());
        assert!(report.coverage_realism.is_finite());
    }
}
