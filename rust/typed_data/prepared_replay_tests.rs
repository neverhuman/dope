#[test]
fn fit_row_binding_is_shared_with_affine_state_and_legacy_replay_is_explicit() {
    use super::super::FittedTransforms;
    let input = table("10,20,100\n20,40,200\n1000,2000,300\n", &declaration(false));
    let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
    let affine = FittedTransforms::fit_transform(&input, &[0, 1]).unwrap();
    let digest = &fit.metadata.provenance.fit_digest;
    assert_eq!(affine.fit_row_digest(), Some(digest.as_str()));
    assert!(affine.fit_only());
    let bytes = affine.sampling_state_bytes().unwrap();
    let restored = FittedTransforms::reload_for_fit(&bytes, digest).unwrap();
    assert_eq!(restored.sampling_state_bytes().unwrap(), bytes);
    assert_eq!(
        restored.inverse_transform("c0001", &[99.]).unwrap(),
        [1000.]
    );
    assert_eq!(
        restored
            .transform_cell("c0001", input.cell(2, "c0001").unwrap())
            .unwrap(),
        Cell::Number(99.)
    );
    assert_eq!(
        FittedTransforms::reload_for_fit(&bytes, &"f".repeat(64))
            .unwrap_err()
            .code,
        "fit_row_digest_mismatch"
    );
    let mut legacy: Value = serde_json::from_slice(&bytes).unwrap();
    legacy.as_object_mut().unwrap().remove("fit_row_digest");
    let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
    let unbound = FittedTransforms::reload(&legacy_bytes).unwrap();
    assert!(!unbound.fit_only());
    assert_eq!(unbound.inverse_transform("c0003", &[2.]).unwrap(), [300.]);
    assert_eq!(
        FittedTransforms::reload_for_fit(&legacy_bytes, digest)
            .unwrap_err()
            .code,
        "fit_row_digest_mismatch"
    );
    for bad in [json!("secret_digest"), json!("A".repeat(64)), json!(1)] {
        legacy["fit_row_digest"] = bad;
        let e = FittedTransforms::reload(&serde_json::to_vec(&legacy).unwrap()).unwrap_err();
        assert_eq!(e.code, "invalid_transform_state");
        assert!(!format!("{e} {e:?}").contains("secret_"));
    }
    let reversed = FittedTransforms::fit_transform(&input, &[1, 0]).unwrap();
    assert_ne!(reversed.fit_row_digest(), affine.fit_row_digest());
    // Equal extrema do not erase changes to interior fitting labels.
    let other = table("10,20,100\n20,40,201\n1000,2000,300\n", &declaration(false));
    let a = FittedTransforms::fit_transform(&input, &[0, 1, 2]).unwrap();
    let b = FittedTransforms::fit_transform(&other, &[0, 1, 2]).unwrap();
    assert_ne!(a.fit_row_digest(), b.fit_row_digest());
    assert_eq!(
        a.transform_numeric("c0003", &[200.]).unwrap(),
        b.transform_numeric("c0003", &[200.]).unwrap()
    );
}

#[test]
fn all_arms_use_one_frozen_map_and_original_unit_replay() {
    let input = table(
        "10,20,100\n20,40,200\n30,60,300\n-10,-20,-100\n",
        &declaration(false),
    );
    let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
    let contract = fit.freeze().unwrap();
    let synthetic = table("15,30,150\n40,80,400\n", &declaration(false));
    for (raw, rows) in [
        (&input, vec![2]),
        (&input, vec![3]),
        (&synthetic, vec![0, 1]),
    ] {
        let view = contract.replay(ReplayInput::Raw(raw, &rows)).unwrap();
        assert!(view.metadata == fit.metadata);
        // Reload only validates the same frozen map; it never refits it.
        let loaded = validate_input(&view.private_json().unwrap(), &contract).unwrap();
        let original = contract.original_rows(&loaded).unwrap();
        assert_eq!(original, contract.original_rows(&loaded).unwrap());
        for (actual, source_row) in original.iter().zip(&rows) {
            for (cell, id) in actual.iter().zip(["c0001", "c0002", "c0003"]) {
                assert_eq!(cell, raw.cell(*source_row, id).unwrap());
            }
        }
        assert_eq!(
            contract
                .replay(ReplayInput::Prepared(&loaded))
                .unwrap_err()
                .code,
            "prepared_already_transformed"
        );
        assert_eq!(
            contract.compile_input(&loaded).unwrap().view().rows(),
            view.rows()
        );
    }
    assert_eq!(
        contract.apply(&input, &[2]).unwrap().rows(),
        &[vec![2., 2., 300.]]
    );
    assert_eq!(contract.regression_prediction(2.).unwrap(), 300.);
    let mut conflicting = fit.clone();
    conflicting.metadata.columns[0].transform = PreparedTransform::Affine {
        offset: 10.,
        scale: 20.,
    };
    assert_eq!(
        contract.original_rows(&conflicting).unwrap_err().code,
        "prepared_column_mismatch"
    );
}

#[test]
fn replay_preserves_categories_booleans_and_discrete_labels() {
    for task in [TaskKind::Regression, TaskKind::Binary, TaskKind::Multiclass] {
        let mut decl = declaration(true);
        decl["columns"][0]["dtype"] = json!("boolean");
        let input = table(
            if task == TaskKind::Regression {
                "false,private_category,100\ntrue,other_category,200\ntrue,private_category,300\n"
            } else {
                "false,private_category,0\ntrue,other_category,1\ntrue,private_category,2\n"
            },
            &decl,
        );
        let fit = prepare(&input, &[0, 1], spec(task)).unwrap();
        let contract = fit.freeze().unwrap();
        let rows = contract.original_rows(&fit).unwrap();
        for (row, cells) in rows.iter().enumerate() {
            for (cell, id) in cells.iter().zip(["c0001", "c0002", "c0003"]) {
                assert_eq!(cell, input.cell(row, id).unwrap());
            }
        }
        for bad in [-1., 0.5, 2., f64::NAN, f64::INFINITY] {
            let mut invalid = fit.clone();
            invalid.rows[0][1] = bad;
            assert!(contract.original_rows(&invalid).is_err());
        }
        let mut conflicting = fit.clone();
        conflicting.metadata.columns[0].transform = PreparedTransform::Affine {
            offset: 0.,
            scale: 2.,
        };
        assert!(contract.original_rows(&conflicting).is_err());
    }
}

#[test]
fn replay_debug_errors_and_public_projection_never_echo_cells_or_extrema() {
    let input = table(
        "1234567,private_category,7654321\n2234567,other_category,8654321\n",
        &declaration(true),
    );
    let fit = prepare(&input, &[0, 1], spec(TaskKind::Regression)).unwrap();
    let contract = fit.freeze().unwrap();
    let e = contract.replay(ReplayInput::Prepared(&fit)).unwrap_err();
    let raw_debug = format!("{:?}", contract.original_rows(&fit).unwrap());
    let public = serde_json::to_string(&fit.public_projection().unwrap()).unwrap();
    let debug = format!(
        "{input:?} {fit:?} {contract:?} {:?} {e} {e:?} {raw_debug}",
        ReplayInput::Raw(&input, &[0])
    );
    for secret in [
        "private_",
        "other_category",
        "1234567",
        "7654321",
        "offset",
        "scale",
    ] {
        assert!(!debug.contains(secret));
        assert!(!public.contains(secret));
    }
}
