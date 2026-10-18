use super::coupled::{compare, CoupledEdit, EditOperator, Selector, SelectorMetrics};

fn edits() -> Vec<CoupledEdit> {
    [
        ("A", EditOperator::Knot),
        ("B", EditOperator::Edge),
    ]
    .into_iter()
    .map(|(payload, operator)| CoupledEdit {
        section: "target".into(),
        operator,
        payload: payload.into(),
    })
    .collect()
}

fn xor_metrics(selected: &[usize]) -> SelectorMetrics {
    let aggregate_tstr = match selected {
        [] => 0.5,
        [0] | [1] => 0.4,
        [0, 1] => 0.9,
        _ => 0.1,
    };
    SelectorMetrics {
        aggregate_tstr,
        critical_tail_tstr: 0.8,
    }
}

#[test]
fn matched_pool_xor_fixture_defeats_greedy_but_coupled_searches_find_pair() {
    let results = compare(b"base", &edits(), 256, 0.5, 32, xor_metrics);
    assert_eq!(results.len(), 6);
    assert!(results.iter().all(|result| result.evaluations <= 32));
    let by_name = |selector| results.iter().find(|result| result.selector == selector).unwrap();

    assert!(by_name(Selector::Greedy).selected.is_empty());
    for selector in [Selector::Beam, Selector::Exchange, Selector::Lookahead] {
        let result = by_name(selector);
        assert_eq!(result.selected, vec![0, 1], "{}", selector.name());
        assert_eq!(result.metrics.aggregate_tstr, 0.9);
    }
    assert!(by_name(Selector::ExistingSelector).selected.is_empty());
    assert!(by_name(Selector::IndependentSectionDp).selected.is_empty());
}

#[test]
fn candidate_size_is_measured_from_whole_serialized_file() {
    let pool = edits();
    let result = compare(b"base", &pool, 256, 0.5, 32, xor_metrics)
        .into_iter()
        .find(|result| result.selector == Selector::Lookahead)
        .unwrap();
    let whole = serde_json::to_vec(&(
        b"base".as_slice(),
        result.selected.iter().map(|&index| &pool[index]).collect::<Vec<_>>(),
    ))
    .unwrap();
    let one_a = serde_json::to_vec(&(b"base".as_slice(), vec![&pool[0]])).unwrap().len();
    let one_b = serde_json::to_vec(&(b"base".as_slice(), vec![&pool[1]])).unwrap().len();
    let empty = serde_json::to_vec(&(b"base".as_slice(), Vec::<&CoupledEdit>::new())).unwrap().len();

    assert_eq!(result.whole_file_bytes, whole.len());
    assert_ne!(result.whole_file_bytes - empty, (one_a - empty) + (one_b - empty));
}

#[test]
fn critical_tail_gate_rejects_aggregate_winner() {
    let results = compare(b"base", &edits(), 256, 0.5, 32, |selected| {
        if selected == [0, 1] {
            SelectorMetrics { aggregate_tstr: 0.99, critical_tail_tstr: 0.2 }
        } else {
            xor_metrics(selected)
        }
    });
    for result in results.iter().filter(|result| {
        matches!(result.selector, Selector::Beam | Selector::Exchange | Selector::Lookahead)
    }) {
        assert!(result.selected.is_empty());
        assert!(result.stopped_for_critical_tail);
        assert_eq!(result.metrics.aggregate_tstr, 0.5);
    }
    assert!(results.iter().all(|result| result.selected.is_empty()));
}

#[test]
fn operator_vocabulary_covers_the_preregistered_edit_families() {
    let operators = [
        EditOperator::Knot,
        EditOperator::Edge,
        EditOperator::Lowrank,
        EditOperator::Ga2m,
        EditOperator::Noise,
        EditOperator::Precision,
        EditOperator::Subexpression,
        EditOperator::Category,
        EditOperator::Calibration,
    ];
    assert_eq!(operators.len(), 9);
}
