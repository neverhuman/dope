use dope_kernel::compiler::interactions::{
    ColumnMatrix, InteractionError, ProposalKind, feature_budget_search,
};

fn bits(rows: usize, salt: u32) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let mut mixed = (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ u64::from(salt).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            mixed ^= mixed >> 31;
            if (mixed >> 33) & 1 == 0 { -1.0 } else { 1.0 }
        })
        .collect()
}

fn matrix_from(columns: Vec<Vec<f32>>, target: Vec<f32>) -> ColumnMatrix {
    ColumnMatrix::new(columns, target).unwrap()
}

#[test]
fn tariff_and_shape_errors_carry_codes_only() {
    assert_eq!(ProposalKind::Rank.bytes(), 14);
    assert_eq!(ProposalKind::Product.bytes(), 12);
    assert_eq!(ProposalKind::Gate.bytes(), 20);
    let shape = ColumnMatrix::new(vec![vec![1.0, 2.0, 3.0]], vec![0.0; 4]);
    let shape_error = shape.unwrap_err();
    assert_eq!(shape_error, InteractionError::InvalidShape);
    assert_eq!(shape_error.to_string(), "invalid_shape");
    let columns = vec![bits(16, 1), bits(16, 2)];
    let target: Vec<f32> = (0..16)
        .map(|row| columns[0][row] * columns[1][row])
        .collect();
    let matrix = matrix_from(columns, target);
    let budget = feature_budget_search(&matrix, 0, 1);
    assert_eq!(budget.unwrap_err().to_string(), "invalid_budget");
}

#[test]
fn product_xor_and_rare_gate_select_the_generating_pair() {
    let rows = 64;
    let mut columns: Vec<Vec<f32>> = (0..6).map(|salt| bits(rows, salt + 3)).collect();
    columns.push((0..rows).map(|row| row as f32).collect());
    let id = columns.len() - 1;
    let product: Vec<f32> = (0..rows)
        .map(|row| columns[0][row] * columns[1][row])
        .collect();
    let matrix = matrix_from(columns, product);
    let search = feature_budget_search(&matrix, ProposalKind::Product.bytes(), 11).unwrap();
    let chosen = &search.support.proposals;
    assert_eq!(chosen.len(), 1);
    assert_eq!(chosen[0].kind, ProposalKind::Product);
    assert_eq!((chosen[0].left, chosen[0].right), (0, Some(1)));
    assert!(chosen[0].held_out_mse < 1.0e-8);
    assert!(
        search
            .support
            .proposals
            .iter()
            .all(|term| usize::from(term.left) != id)
    );
    assert_eq!(search.support.charged_bytes, ProposalKind::Product.bytes());

    let rows = 140;
    let left: Vec<f32> = (0..rows)
        .map(|row| if row % 5 == 0 { 1.0 } else { -1.0 })
        .collect();
    let right: Vec<f32> = (0..rows)
        .map(|row| if row % 7 == 0 { 1.0 } else { -1.0 })
        .collect();
    let noise = bits(rows, 9);
    let target: Vec<f32> = (0..rows)
        .map(|row| {
            if left[row] > 0.0 && right[row] > 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let matrix = matrix_from(vec![left, right, noise], target);
    let search = feature_budget_search(&matrix, ProposalKind::Gate.bytes(), 11).unwrap();
    let chosen = &search.support.proposals[0];
    assert_eq!(chosen.kind, ProposalKind::Gate);
    assert_eq!((chosen.left, chosen.right), (0, Some(1)));
    assert!(chosen.held_out_mse < 1.0e-8);
}

#[test]
fn selection_fits_inside_folds_and_ignores_fixture_names() {
    let rows = 64;
    let columns: Vec<Vec<f32>> = (0..4).map(|salt| bits(rows, salt + 8)).collect();
    let target: Vec<f32> = (0..rows)
        .map(|row| columns[0][row] * columns[1][row])
        .collect();
    let matrix = matrix_from(columns, target);
    let search = feature_budget_search(&matrix, ProposalKind::Product.bytes(), 3).unwrap();
    let rows = matrix.rows();
    assert!(search.support.fold_count >= 2);
    assert_eq!(search.support.eval_counts.iter().sum::<usize>(), rows);
    for (train, eval) in search
        .support
        .train_counts
        .iter()
        .zip(&search.support.eval_counts)
    {
        assert_eq!(train + eval, rows);
        assert!(*eval > 0);
        assert!(*train > 0 && *train < rows);
    }
    let source = include_str!("../rust/compiler/interactions.rs");
    for name in [
        "pure_product",
        "rare_gate",
        "weak_distributed",
        "irrelevant_id",
    ] {
        assert!(!source.contains(name), "{name}");
    }
}

#[test]
fn order_collinear_and_teacher_changes_are_measured() {
    let rows = 64;
    let signal = bits(rows, 4);
    let copy = signal.clone();
    let noise = bits(rows, 5);
    let target = signal.clone();
    let matrix = matrix_from(vec![signal, copy, noise], target);
    let search = feature_budget_search(&matrix, ProposalKind::Rank.bytes(), 5).unwrap();
    assert_eq!(search.support.proposals[0].kind, ProposalKind::Rank);
    assert_eq!(search.support.proposals[0].left, 0);
    let substitute = matrix_from(
        vec![bits(rows, 99), bits(rows, 4), bits(rows, 5)],
        bits(rows, 4),
    );
    let swapped = feature_budget_search(&substitute, ProposalKind::Rank.bytes(), 5).unwrap();
    assert_eq!(swapped.support.proposals[0].left, 1);
    assert!((search.support.held_out_mse - swapped.support.held_out_mse).abs() < 1.0e-8);

    let columns: Vec<Vec<f32>> = (0..4).map(|salt| bits(rows, salt + 15)).collect();
    let product = forward_target(&columns);
    let forward = matrix_from(columns.clone(), product);
    let reversed_target: Vec<f32> = forward_target(&columns).into_iter().rev().collect();
    let backward = matrix_from(
        columns
            .iter()
            .map(|column| column.iter().rev().copied().collect())
            .collect(),
        reversed_target,
    );
    let ahead = feature_budget_search(&forward, ProposalKind::Product.bytes(), 9).unwrap();
    let back = feature_budget_search(&backward, ProposalKind::Product.bytes(), 9).unwrap();
    assert_eq!(
        ahead.support.proposals[0].left,
        back.support.proposals[0].left
    );
    assert_eq!(
        ahead.support.proposals[0].right,
        back.support.proposals[0].right
    );

    let mut permuted = columns.clone();
    permuted.swap(0, 2);
    let permuted_target: Vec<f32> = (0..rows)
        .map(|row| permuted[2][row] * permuted[0][row])
        .collect();
    let moved = matrix_from(permuted, permuted_target);
    let moved_search = feature_budget_search(&moved, ProposalKind::Product.bytes(), 9).unwrap();
    let term = &moved_search.support.proposals[0];
    let pair = (
        term.left.min(term.right.unwrap()),
        term.left.max(term.right.unwrap()),
    );
    assert_eq!(pair, (0, 2));

    let teacher_a = feature_budget_search(&forward, ProposalKind::Product.bytes(), 9).unwrap();
    let only_third: Vec<f32> = columns[3].clone();
    let teacher_b = matrix_from(columns, only_third);
    let other = feature_budget_search(&teacher_b, ProposalKind::Rank.bytes(), 9).unwrap();
    assert_ne!(
        teacher_a.support.proposals[0].kind,
        other.support.proposals[0].kind
    );
    assert_eq!(other.support.proposals[0].left, 3);
}

fn forward_target(columns: &[Vec<f32>]) -> Vec<f32> {
    (0..columns[0].len())
        .map(|row| columns[0][row] * columns[2][row])
        .collect()
}

#[test]
fn equal_byte_widths_keep_weak_signal_and_drop_ids() {
    let rows = 96;
    let mut columns: Vec<Vec<f32>> = (0..12).map(|salt| bits(rows, salt + 30)).collect();
    let id = columns.len();
    columns.push((0..rows).map(|row| row as f32).collect());
    let target: Vec<f32> = (0..rows)
        .map(|row| columns[..id].iter().map(|column| column[row]).sum())
        .collect();
    let matrix = matrix_from(columns, target);
    let budget = ProposalKind::Rank.bytes() * 24;
    let search = feature_budget_search(&matrix, budget, 19).unwrap();
    assert_eq!(search.widths.len(), 4);
    for (report, max_terms) in search.widths.iter().zip([12, 24, 48, 48]) {
        assert!(report.terms.len() <= max_terms);
        assert!(report.charged_bytes <= budget);
        assert_eq!(
            report.charged_bytes,
            report.terms.iter().map(|term| term.bytes).sum::<u32>()
        );
    }
    let adaptive = search
        .widths
        .iter()
        .find(|report| report.label == "adaptive")
        .unwrap();
    let narrow = search
        .widths
        .iter()
        .find(|report| report.label == "12")
        .unwrap();
    assert!(adaptive.held_out_mse <= narrow.held_out_mse + 1.0e-8);
    let signal_ranks = adaptive
        .terms
        .iter()
        .filter(|term| term.kind == ProposalKind::Rank && usize::from(term.left) < id)
        .count();
    assert!(signal_ranks >= 8, "{signal_ranks}");
    assert!(adaptive.terms.iter().all(|term| {
        usize::from(term.left) != id && term.right.is_none_or(|right| usize::from(right) != id)
    }));
}

#[test]
fn randomized_proposals_follow_the_seed() {
    let rows = 48;
    let columns: Vec<Vec<f32>> = (0..10).map(|salt| bits(rows, salt + 70)).collect();
    let target = bits(rows, 71);
    let matrix = matrix_from(columns, target);
    let first = feature_budget_search(&matrix, 80, 1).unwrap();
    let second = feature_budget_search(&matrix, 80, 2).unwrap();
    assert_ne!(first.randomized_pairs, second.randomized_pairs);
    let repeated = feature_budget_search(&matrix, 80, 1).unwrap();
    assert_eq!(first.randomized_pairs, repeated.randomized_pairs);
}

#[test]
fn finite_large_inputs_keep_interaction_reports_finite() {
    let rows = 64;
    let left = bits(rows, 3);
    let right = bits(rows, 4);
    let target = left.iter().zip(&right).map(|(l, r)| l * r).collect();
    let matrix = matrix_from(
        vec![
            left.iter().map(|value| value * 1.0e20).collect(),
            right.iter().map(|value| value * 1.0e20).collect(),
            bits(rows, 9),
        ],
        target,
    );
    let wide = feature_budget_search(&matrix, 1_024, 11).unwrap();
    assert!(
        wide.widths
            .iter()
            .all(|report| report.held_out_mse.is_finite())
    );
    let narrow = feature_budget_search(&matrix, ProposalKind::Product.bytes(), 11).unwrap();
    assert_eq!(narrow.support.proposals.len(), 1);
    assert_eq!(narrow.support.proposals[0].kind, ProposalKind::Product);
    assert_eq!(
        (
            narrow.support.proposals[0].left,
            narrow.support.proposals[0].right
        ),
        (0, Some(1))
    );
    assert!(narrow.support.held_out_mse < 1.0e-8);
}

#[test]
fn unseen_slot_uses_only_the_columns_that_generate_it() {
    let rows = 64;
    for slot in 0..3u16 {
        let columns: Vec<Vec<f32>> = (0..6)
            .map(|salt| bits(rows, salt + 80 + u32::from(slot)))
            .collect();
        let right = slot + 1;
        let target: Vec<f32> = (0..rows)
            .map(|row| columns[usize::from(slot)][row] * columns[usize::from(right)][row])
            .collect();
        let matrix = matrix_from(columns, target);
        let search = feature_budget_search(&matrix, ProposalKind::Product.bytes(), 4).unwrap();
        let term = &search.support.proposals[0];
        assert_eq!(term.kind, ProposalKind::Product);
        let pair = (
            term.left.min(term.right.unwrap()),
            term.left.max(term.right.unwrap()),
        );
        assert_eq!(pair, (slot, right));
    }
}
