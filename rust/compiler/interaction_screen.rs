// Fit-only bounded pair exploration independent of marginal parent support.
// Row folds use a fixed row-index hash, never target/candidate-dependent folds.

// Deduplicate complete fit columns, not just the screening row subset. Verify
// equality inside hash buckets and canonicalize signed zero's numeric identity.
fn interaction_column_aliases(columns: &[Vec<f32>]) -> Vec<usize> {
    let mut buckets = std::collections::BTreeMap::<[u8; 32], Vec<usize>>::new();
    let mut aliases = Vec::with_capacity(columns.len());
    for (index, column) in columns.iter().enumerate() {
        let mut hash = blake3::Hasher::new();
        for &value in column {
            hash.update(&(if value == 0.0 { 0.0 } else { value }).to_le_bytes());
        }
        let bucket = buckets.entry(*hash.finalize().as_bytes()).or_default();
        let representative = bucket
            .iter()
            .copied()
            .find(|&other| columns[other] == *column)
            .unwrap_or_else(|| {
                bucket.push(index);
                index
            });
        aliases.push(representative);
    }
    aliases
}

fn crossfit_interaction_pairs(table: &Table, columns: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let rows = table.rows.min(512);
    let aliases = interaction_column_aliases(columns);
    let unique = aliases
        .iter()
        .enumerate()
        .filter_map(|(index, &parent)| (index == parent).then_some(index))
        .collect::<Vec<_>>();
    if rows < 16 || unique.len() < 2 {
        return Vec::new();
    }
    let row_ids = (0..rows).map(|r| r * table.rows / rows).collect::<Vec<_>>();
    let folds = row_ids
        .iter()
        .map(|r| (splitmix64(*r as u64 ^ 0x6d61_696e_666f_6c64) % 4) as usize)
        .collect::<Vec<_>>();
    let mut pairs = Vec::new();
    if unique.len() <= 64 {
        for (at, &left) in unique.iter().enumerate() {
            for &right in &unique[at + 1..] {
                pairs.push((left, right));
            }
        }
    } else {
        // Wide inputs have a deterministic exploration quota over every column,
        // rather than requiring marginally useful parents for all proposals.
        let mut seen = std::collections::BTreeSet::new();
        for at in 0..2048u64 {
            let left = unique[(splitmix64(at ^ 0xa17e) % unique.len() as u64) as usize];
            let right = unique[(splitmix64(at ^ 0xc0fe) % unique.len() as u64) as usize];
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

// Explore a bounded semantic-column-order sample plus strong single-column
// proposals. Tail cuts are estimated in each training fold; no audit/final
// labels or candidate-dependent folds enter proposal construction.
fn exploratory_pair_proposals(
    table: &Table,
    columns: &[Vec<f32>],
    screen: &[(usize, f32)],
    limit: usize,
) -> Vec<(usize, usize)> {
    let aliases = interaction_column_aliases(columns);
    if table.rows < 24
        || aliases
            .iter()
            .enumerate()
            .filter(|(index, parent)| *index == **parent)
            .count()
            < 2
    {
        return Vec::new();
    }
    let count = table.rows.min(256);
    let ids = (0..count)
        .map(|at| at * table.rows / count)
        .collect::<Vec<_>>();
    let folds = ids
        .iter()
        .map(|id| (splitmix64(*id as u64 ^ 0x6d61_696e_666f_6c64) % 4) as usize)
        .collect::<Vec<_>>();
    let mut semantic = columns
        .iter()
        .enumerate()
        .filter(|(index, _)| aliases[*index] == *index)
        .map(|(index, column)| {
            let mut hash = blake3::Hasher::new();
            for &row in &ids {
                hash.update(&column[row].to_le_bytes());
            }
            (hash.finalize().to_hex().to_string(), index)
        })
        .collect::<Vec<_>>();
    semantic.sort();
    let mut proposed = std::collections::BTreeSet::new();
    let mut seen_parents = std::collections::BTreeSet::new();
    let strongest = screen
        .iter()
        .map(|(feature, _)| aliases[*feature])
        .filter(|feature| seen_parents.insert(*feature))
        .take(8)
        .collect::<Vec<_>>();
    for (at, &left) in strongest.iter().enumerate() {
        for &right in &strongest[at + 1..] {
            proposed.insert((left.min(right), left.max(right)));
        }
    }
    for at in 0..128u64 {
        let left = semantic[(splitmix64(at ^ 0xc2a1) % semantic.len() as u64) as usize].1;
        let right = semantic[(splitmix64(at ^ 0xc2b2) % semantic.len() as u64) as usize].1;
        if left != right {
            proposed.insert((left.min(right), left.max(right)));
        }
    }
    let mut scored = Vec::new();
    for (left, right) in proposed {
        let mut best = 0.0f64;
        for (left_q, right_q) in [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)] {
            let mut null_error = 0.0;
            let mut pair_error = 0.0;
            for fold in 0..4 {
                let train = ids
                    .iter()
                    .enumerate()
                    .filter(|(at, _)| folds[*at] != fold)
                    .map(|(_, row)| *row)
                    .collect::<Vec<_>>();
                let eval = ids
                    .iter()
                    .enumerate()
                    .filter(|(at, _)| folds[*at] == fold)
                    .map(|(_, row)| *row)
                    .collect::<Vec<_>>();
                if train.is_empty() || eval.is_empty() {
                    continue;
                }
                let cut = |feature: usize, q: f32| {
                    let mut values = train
                        .iter()
                        .map(|row| columns[feature][*row])
                        .collect::<Vec<_>>();
                    values.sort_by(f32::total_cmp);
                    quantile(&values, q)
                };
                let lm = cut(left, left_q);
                let rm = cut(right, right_q);
                let mean = train
                    .iter()
                    .map(|row| f64::from(table.target[*row]))
                    .sum::<f64>()
                    / train.len() as f64;
                let cell = |row: usize| {
                    usize::from(columns[left][row] > lm) * 2 + usize::from(columns[right][row] > rm)
                };
                let mut sums = [0.0; 4];
                let mut counts = [0; 4];
                for &row in &train {
                    let at = cell(row);
                    sums[at] += f64::from(table.target[row]);
                    counts[at] += 1;
                }
                for &row in &eval {
                    let at = cell(row);
                    let prediction = if counts[at] == 0 {
                        mean
                    } else {
                        sums[at] / f64::from(counts[at])
                    };
                    let actual = f64::from(table.target[row]);
                    null_error += (actual - mean).powi(2);
                    pair_error += (actual - prediction).powi(2);
                }
            }
            if null_error > 0.0 {
                best = best.max((null_error - pair_error) / null_error);
            }
        }
        if best > 0.01 {
            scored.push((left, right, best));
        }
    }
    scored.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(left, right, _)| (left, right))
        .collect()
}

#[cfg(test)]
mod interaction_screen_tests {
    use super::*;
    #[test]
    fn duplicated_parents_never_form_self_or_repeated_interactions() {
        let rows = 256;
        let parent = (0..rows)
            .map(|row| (row % 16) as f32 / 15.0)
            .collect::<Vec<_>>();
        let columns = vec![parent.clone(); 70];
        let table = Table {
            rows,
            features: columns.len(),
            columns: columns.clone(),
            target: parent.clone(),
        };
        let screen = target_screen(&columns, &table.target);
        assert!(crossfit_interaction_pairs(&table, &columns).is_empty());
        assert!(exploratory_pair_proposals(&table, &columns, &screen, 4).is_empty());
        let target = fit_ga2m_target(&table, &columns, &screen, Task::Regression);
        let Target::Ga2m { interactions, .. } = target else {
            panic!("ga2m");
        };
        assert!(interactions.is_empty());
        let other = (0..rows)
            .map(|row| ((row / 16) % 16) as f32 / 15.0)
            .collect::<Vec<_>>();
        let columns = vec![parent.clone(), other.clone(), parent, other];
        let table = Table {
            rows,
            features: 4,
            columns: columns.clone(),
            target: (0..rows)
                .map(|row| u8::from((row % 16 > 7) != (row / 16 > 7)) as f32)
                .collect(),
        };
        let pairs = crossfit_interaction_pairs(&table, &columns);
        assert_eq!(pairs, vec![(0, 1)]);
        assert!(pairs.len() <= 4);
        let screen = target_screen(&columns, &table.target);
        let target = fit_ga2m_target(&table, &columns, &screen, Task::Regression);
        let Target::Ga2m { interactions, .. } = target else {
            panic!("ga2m");
        };
        assert!(interactions.len() <= 4);
        assert!(
            interactions
                .iter()
                .all(|term| term.left == 0 && term.right == 1)
        );
    }

    #[test]
    fn duplicate_identity_uses_all_rows_and_numeric_signed_zero() {
        let mut a = vec![0.0; 1024];
        let mut b = a.clone();
        b[1] = 1.0; // Outside the 512-row screening subset.
        a[0] = -0.0;
        assert_eq!(interaction_column_aliases(&[a.clone(), b, a]), [0, 1, 0]);
    }
    #[test]
    fn rare_gate_proposals_survive_shifted_features_target_units_and_permutation() {
        let rows = 256;
        let mut columns = (0..40)
            .map(|feature| {
                (0..rows)
                    .map(|row| {
                        (splitmix64(row as u64 ^ ((feature as u64 + 1) << 24)) % 1000) as f32
                            / 999.0
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut target = Vec::new();
        for row in 0..rows {
            let left = (row % 16) as f32 / 15.0;
            let right = (row / 16) as f32 / 15.0;
            columns[30][row] = 12.0 + left;
            columns[31][row] = 19.0 + right;
            target.push(if left > 0.8 && right > 0.8 { 0.9 } else { 0.1 });
        }
        let mut table = Table {
            rows,
            features: 40,
            columns: columns.clone(),
            target,
        };
        for scale in [1.0, 1000.0] {
            table
                .target
                .iter_mut()
                .for_each(|value| *value = *value * scale + 7.0);
            let screen = target_screen(&columns, &table.target);
            assert!(exploratory_pair_proposals(&table, &columns, &screen, 8).contains(&(30, 31)));
        }
        columns.reverse();
        table.columns = columns.clone();
        let screen = target_screen(&columns, &table.target);
        assert!(exploratory_pair_proposals(&table, &columns, &screen, 8).contains(&(8, 9)));
    }
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
