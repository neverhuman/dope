// Fit-only bounded pair exploration independent of marginal parent support.
// Row folds use a fixed row-index hash, never target/candidate-dependent folds.

fn crossfit_interaction_pairs(table: &Table, columns: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let rows = table.rows.min(512);
    if rows < 16 || columns.len() < 2 {
        return Vec::new();
    }
    let row_ids = (0..rows).map(|r| r * table.rows / rows).collect::<Vec<_>>();
    let folds = row_ids
        .iter()
        .map(|r| (splitmix64(*r as u64 ^ 0x6d61_696e_666f_6c64) % 4) as usize)
        .collect::<Vec<_>>();
    let mut pairs = Vec::new();
    if columns.len() <= 64 {
        for left in 0..columns.len() {
            for right in left + 1..columns.len() {
                pairs.push((left, right));
            }
        }
    } else {
        // Wide inputs have a deterministic exploration quota over every column,
        // rather than requiring marginally useful parents for all proposals.
        let mut seen = std::collections::BTreeSet::new();
        for at in 0..2048u64 {
            let left = (splitmix64(at ^ 0xa17e) % columns.len() as u64) as usize;
            let right = (splitmix64(at ^ 0xc0fe) % columns.len() as u64) as usize;
            if left != right {
                seen.insert((left.min(right), left.max(right)));
            }
        }
        pairs.extend(seen);
    }
    let mut scored = Vec::new();
    for (left, right) in pairs {
        let mut null_error = 0.0;
        let mut pair_error = 0.0;
        for fold in 0..4 {
            let train = row_ids
                .iter()
                .enumerate()
                .filter(|(at, _)| folds[*at] != fold)
                .map(|(_, row)| *row)
                .collect::<Vec<_>>();
            let eval = row_ids
                .iter()
                .enumerate()
                .filter(|(at, _)| folds[*at] == fold)
                .map(|(_, row)| *row)
                .collect::<Vec<_>>();
            if train.is_empty() || eval.is_empty() {
                continue;
            }
            let mean = train
                .iter()
                .map(|r| f64::from(table.target[*r]))
                .sum::<f64>()
                / train.len() as f64;
            let median = |column: usize| {
                let mut values = train
                    .iter()
                    .map(|r| columns[column][*r])
                    .collect::<Vec<_>>();
                values.sort_by(f32::total_cmp);
                let low = values[0];
                let high = values[values.len() - 1];
                if values.iter().all(|v| *v == low || *v == high) {
                    return (low + high) * 0.5;
                }
                values[values.len() / 2]
            };
            let lm = median(left);
            let rm = median(right);
            let mut sums = [0.0; 4];
            let mut counts = [0usize; 4];
            let cell = |r: usize| {
                usize::from(columns[left][r] > lm) * 2 + usize::from(columns[right][r] > rm)
            };
            for &r in &train {
                let at = cell(r);
                sums[at] += f64::from(table.target[r]);
                counts[at] += 1;
            }
            for &r in &eval {
                let at = cell(r);
                let prediction = if counts[at] > 0 {
                    sums[at] / counts[at] as f64
                } else {
                    mean
                };
                let actual = f64::from(table.target[r]);
                null_error += (actual - mean).powi(2);
                pair_error += (actual - prediction).powi(2);
            }
        }
        if null_error > 0.0 {
            let gain = (null_error - pair_error) / null_error;
            if gain > 0.01 {
                scored.push((left, right, gain));
            }
        }
    }
    scored.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    scored.into_iter().take(4).map(|(l, r, _)| (l, r)).collect()
}

#[cfg(test)]
mod interaction_screen_tests {
    use super::*;
    #[test]
    fn pure_shifted_pair_beyond_marginal_parents_survives_target_units_and_permutation() {
        let n = 256;
        let p = 40;
        let mut columns = vec![vec![0.0; n]; p];
        let mut target = vec![0.0; n];
        for row in 0..n {
            for (col, values) in columns.iter_mut().enumerate() {
                values[row] = (splitmix64((row as u64) << 8 ^ col as u64) % 1000) as f32 / 1000.0;
            }
            let a = (row % 2) as f32;
            let b = ((row / 2) % 2) as f32;
            columns[30][row] = a + 12.0;
            columns[31][row] = b + 19.0;
            target[row] = u8::from(a != b) as f32;
        }
        let table = Table {
            rows: n,
            features: p,
            columns: columns.clone(),
            target: target.clone(),
        };
        assert!(crossfit_interaction_pairs(&table, &columns).contains(&(30, 31)));
        let scaled = Table {
            target: target.iter().map(|v| 100.0 + *v * 100.0).collect(),
            ..table.clone()
        };
        assert!(crossfit_interaction_pairs(&scaled, &columns).contains(&(30, 31)));
        columns.swap(0, 30);
        columns.swap(1, 31);
        let permuted = Table {
            columns: columns.clone(),
            ..table
        };
        assert!(crossfit_interaction_pairs(&permuted, &columns).contains(&(0, 1)));
    }
    #[test]
    fn actual_compiler_ga2m_consumes_pure_pair_and_encoded_cap() {
        let n = 256;
        let p = 40;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for row in 0..n {
            for col in 0..p {
                x.push(if col == 30 {
                    (row % 2) as f32
                } else if col == 31 {
                    ((row / 2) % 2) as f32
                } else {
                    (splitmix64((row as u64) << 8 ^ col as u64) % 1000) as f32 / 1000.0
                });
            }
            y.push(u8::from(row % 2 != (row / 2) % 2) as f32);
        }
        let table = Table::from_arrays(&x, &y, n, p, Task::Regression).unwrap();
        let screen = target_screen(&table.columns, &table.target);
        let target = fit_ga2m_target(&table, &table.columns, &screen, Task::Regression);
        let Target::Ga2m { interactions, .. } = &target else {
            panic!("ga2m");
        };
        assert!(interactions.iter().any(|t| t.left == 30 && t.right == 31));
        let mse = (0..n)
            .map(|row| {
                let values = table.columns.iter().map(|c| c[row]).collect::<Vec<_>>();
                let prediction = evaluate_target(&target, &values).0;
                (prediction - y[row]).powi(2)
            })
            .sum::<f32>()
            / n as f32;
        assert!(mse < 0.12, "{mse}");
        let options = CompileOptions {
            seed: Some(1729),
            beam_width: Some(42),
            quantization_profiles: vec![8],
            ..CompileOptions::default()
        };
        let result = compile_kernel_from_arrays(&x, &y, n, p, Task::Regression, &options).unwrap();
        assert!(result.candidates.iter().any(|c| c.target == "ga2m"));
        assert!(result.candidate_artifacts.iter().any(|(_,bytes)|{
            let k=crate::codec::decode_kernel(bytes).unwrap();
            matches!(k.program,KernelProgram::Symbolic{target:Target::Ga2m{interactions,..},..} if interactions.iter().any(|t|t.left==30 && t.right==31))
        }));
        assert!(
            result
                .report
                .search_history
                .iter()
                .all(|a| a.actual_artifact_bytes.is_some())
        );
    }
}
