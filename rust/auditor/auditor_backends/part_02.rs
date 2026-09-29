

impl AuditorBackend for Ga2m {
    fn id(&self) -> &'static str {
        "ga2m"
    }

    #[allow(clippy::needless_range_loop)]
    fn predict(&self, train: &Table, test: &Table, task: Task, _seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let mean = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / train.rows as f64;
        let intercept = if task == Task::Binary {
            (mean.clamp(1e-5, 1.0 - 1e-5) / (1.0 - mean.clamp(1e-5, 1.0 - 1e-5))).ln()
        } else {
            mean
        };
        let mut main = vec![[0.0f64; 8]; train.features];
        let pairs = (0..train.features.saturating_sub(1).min(8))
            .map(|feature| (feature, feature + 1))
            .collect::<Vec<_>>();
        let train_bins = (0..train.features)
            .map(|feature| {
                (0..train.rows)
                    .map(|row| bin(value(train, feature, row, &means)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut interactions = vec![[0.0f64; 64]; pairs.len()];
        let mut linear = vec![intercept; train.rows];
        for _ in 0..12 {
            for (feature, feature_bins) in train_bins.iter().enumerate() {
                let mut sums = [0.0; 8];
                let mut counts = [0usize; 8];
                for row in 0..train.rows {
                    let slot = feature_bins[row];
                    let prediction = if task == Task::Binary {
                        sigmoid(linear[row])
                    } else {
                        linear[row]
                    };
                    sums[slot] += f64::from(train.target[row]) - prediction;
                    counts[slot] += 1;
                }
                let mut updates = [0.0; 8];
                for slot in 0..8 {
                    let update = 0.35 * sums[slot] / counts[slot].max(1) as f64;
                    main[feature][slot] += update;
                    updates[slot] = update;
                }
                for row in 0..train.rows {
                    linear[row] += updates[train_bins[feature][row]];
                }
            }
            for (pair_index, &(left, right)) in pairs.iter().enumerate() {
                let mut sums = [0.0; 64];
                let mut counts = [0usize; 64];
                for row in 0..train.rows {
                    let slot = train_bins[left][row] * 8 + train_bins[right][row];
                    let prediction = if task == Task::Binary {
                        sigmoid(linear[row])
                    } else {
                        linear[row]
                    };
                    sums[slot] += f64::from(train.target[row]) - prediction;
                    counts[slot] += 1;
                }
                for slot in 0..64 {
                    interactions[pair_index][slot] += 0.2 * sums[slot] / counts[slot].max(1) as f64;
                }
                for row in 0..train.rows {
                    let slot = train_bins[left][row] * 8 + train_bins[right][row];
                    linear[row] += 0.2 * sums[slot] / counts[slot].max(1) as f64;
                }
            }
        }
        Ok((0..test.rows)
            .map(|row| {
                let mut linear = intercept;
                for feature in 0..test.features {
                    linear += main[feature][bin(value(test, feature, row, &means))];
                }
                for (pair_index, &(left, right)) in pairs.iter().enumerate() {
                    let slot = bin(value(test, left, row, &means)) * 8
                        + bin(value(test, right, row, &means));
                    linear += interactions[pair_index][slot];
                }
                if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

#[derive(Clone, Copy)]
struct Stump {
    feature: usize,
    threshold: f64,
    left: f64,
    right: f64,
}

impl Stump {
    fn predict(self, value: f64) -> f64 {
        if value < self.threshold {
            self.left
        } else {
            self.right
        }
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn fit_stump(
    table: &Table,
    means: &[f64],
    targets: &[f64],
    feature: usize,
    threshold: f64,
) -> Stump {
    let mut sums = [0.0; 2];
    let mut counts = [0usize; 2];
    for (row, &target) in targets.iter().enumerate().take(table.rows) {
        let side = usize::from(value(table, feature, row, means) > threshold);
        sums[side] += target;
        counts[side] += 1;
    }
    let mean_prediction = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
    Stump {
        feature,
        threshold,
        left: if counts[0] == 0 {
            mean_prediction
        } else {
            sums[0] / counts[0] as f64
        },
        right: if counts[1] == 0 {
            mean_prediction
        } else {
            sums[1] / counts[1] as f64
        },
    }
}

struct ExtraTrees;

impl AuditorBackend for ExtraTrees {
    fn id(&self) -> &'static str {
        "extremely_randomized_trees"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let targets = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        let mut rng = SplitMix64(seed);
        let stumps = (0..192)
            .map(|_| {
                let feature = rng.next() as usize % train.features;
                let threshold = (rng.next() >> 11) as f64 / ((1u64 << 53) as f64);
                fit_stump(train, &means, &targets, feature, threshold)
            })
            .collect::<Vec<_>>();
        Ok((0..test.rows)
            .map(|row| {
                let prediction = stumps
                    .iter()
                    .map(|stump| stump.predict(value(test, stump.feature, row, &means)))
                    .sum::<f64>()
                    / stumps.len() as f64;
                if task == Task::Binary {
                    prediction.clamp(1e-7, 1.0 - 1e-7)
                } else {
                    prediction.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

struct HistogramGbdt;

impl AuditorBackend for HistogramGbdt {
    fn id(&self) -> &'static str {
        "histogram_gbdt"
    }

    fn predict(&self, train: &Table, test: &Table, task: Task, _seed: u64) -> Result<Vec<f64>> {
        validate_tables(train, test)?;
        let means = column_means(train);
        let mean = train
            .target
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / train.rows as f64;
        let intercept = if task == Task::Binary {
            (mean.clamp(1e-5, 1.0 - 1e-5) / (1.0 - mean.clamp(1e-5, 1.0 - 1e-5))).ln()
        } else {
            mean
        };
        let train_bins = (0..train.features)
            .map(|feature| {
                (0..train.rows)
                    .map(|row| {
                        (value(train, feature, row, &means).clamp(0.0, 1.0) * 16.0)
                            .floor()
                            .min(15.0) as usize
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut train_linear = vec![intercept; train.rows];
        let mut stumps = Vec::new();
        for _ in 0..96 {
            let residuals = (0..train.rows)
                .map(|row| {
                    f64::from(train.target[row])
                        - if task == Task::Binary {
                            sigmoid(train_linear[row])
                        } else {
                            train_linear[row]
                        }
                })
                .collect::<Vec<_>>();
            let mut best = None::<(f64, Stump)>;
            for (feature, feature_bins) in train_bins.iter().enumerate() {
                let mut sums = [0.0f64; 16];
                let mut squares = [0.0f64; 16];
                let mut counts = [0usize; 16];
                for (row, residual) in residuals.iter().copied().enumerate() {
                    let slot = feature_bins[row];
                    sums[slot] += residual;
                    squares[slot] += residual * residual;
                    counts[slot] += 1;
                }
                let total_sum = sums.iter().sum::<f64>();
                let total_square = squares.iter().sum::<f64>();
                let mut left_sum = 0.0;
                let mut left_square = 0.0;
                let mut left_count = 0usize;
                for threshold_bin in 1..16 {
                    let slot = threshold_bin - 1;
                    left_sum += sums[slot];
                    left_square += squares[slot];
                    left_count += counts[slot];
                    let right_sum = total_sum - left_sum;
                    let right_square = total_square - left_square;
                    let right_count = train.rows - left_count;
                    let left_mean = left_sum / left_count.max(1) as f64;
                    let right_mean = right_sum / right_count.max(1) as f64;
                    let error = left_square - left_sum * left_sum / left_count.max(1) as f64
                        + right_square
                        - right_sum * right_sum / right_count.max(1) as f64;
                    let stump = Stump {
                        feature,
                        threshold: threshold_bin as f64 / 16.0,
                        left: left_mean,
                        right: right_mean,
                    };
                    if best
                        .as_ref()
                        .is_none_or(|(best_error, _)| error < *best_error)
                    {
                        best = Some((error, stump));
                    }
                }
            }
            let mut stump = best.expect("nonempty feature search").1;
            stump.left *= 0.08;
            stump.right *= 0.08;
            for (row, linear) in train_linear.iter_mut().enumerate() {
                *linear += stump.predict(value(train, stump.feature, row, &means));
            }
            stumps.push(stump);
        }
        Ok((0..test.rows)
            .map(|row| {
                let linear = intercept
                    + stumps
                        .iter()
                        .map(|stump| stump.predict(value(test, stump.feature, row, &means)))
                        .sum::<f64>();
                if task == Task::Binary {
                    sigmoid(linear)
                } else {
                    linear.clamp(0.0, 1.0)
                }
            })
            .collect())
    }
}

#[cfg(test)]
include!("tests.rs");
