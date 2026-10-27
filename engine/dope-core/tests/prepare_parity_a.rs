use dope_core::prepare::{
    compute_table_descriptors, extract_report_descriptors, prepare_table, prepare_table_cells,
    prepare_table_cells_with_ids, prepare_table_mixed, prepare_table_with_ids, ColumnKind, Level,
    NativeReportView, PrepTask, ShapeOverride,
};
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/prepare-v1")
}

fn read_f64_tensor(st: &SafeTensors<'_>, name: &str) -> Vec<f64> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("missing tensor {name}: {e}"));
    assert_eq!(view.dtype(), Dtype::F64, "expected F64 for {name}");
    view.data()
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_f32_tensor(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("missing tensor {name}: {e}"));
    assert_eq!(view.dtype(), Dtype::F32, "expected F32 for {name}");
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn max_abs_diff_f64(a: &[f64], b: &[f64], label: &str) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch for {label}");
    let mut max_diff = 0.0f64;
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite value at {label}[{idx}]: actual={x} expected={y}"
        );
        let d = (x - y).abs();
        assert!(
            d.is_finite(),
            "non-finite diff at {label}[{idx}]: actual={x} expected={y}"
        );
        if d > max_diff {
            max_diff = d;
        }
    }
    max_diff
}

fn max_abs_diff_f32(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch for {label}");
    let mut max_diff = 0.0f32;
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite value at {label}[{idx}]: actual={x} expected={y}"
        );
        let d = (x - y).abs();
        assert!(
            d.is_finite(),
            "non-finite diff at {label}[{idx}]: actual={x} expected={y}"
        );
        if d > max_diff {
            max_diff = d;
        }
    }
    max_diff
}

#[test]
fn prepare_scaling_and_descriptors_parity_against_python_goldens() {
    let dir = fixtures_dir();
    let manifest_bytes = fs::read(dir.join("manifest.json")).expect("read manifest.json");
    let manifest: Value = serde_json::from_slice(&manifest_bytes).expect("parse manifest.json");
    assert_eq!(manifest["schema"], "dope.prepare-goldens/v1");

    let levels: Vec<Level> = manifest["a_levels"]
        .as_array()
        .expect("a_levels array")
        .iter()
        .map(|v| Level::parse(v.as_str().unwrap()).unwrap())
        .collect();
    let cases = manifest["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let st_bytes = fs::read(dir.join(format!("{name}.safetensors")))
            .unwrap_or_else(|e| panic!("read {name}.safetensors: {e}"));
        let st = SafeTensors::deserialize(&st_bytes).expect("deserialize safetensors");
        let doc_bytes = fs::read(dir.join(format!("{name}.json")))
            .unwrap_or_else(|e| panic!("read {name}.json: {e}"));
        let doc: Value = serde_json::from_slice(&doc_bytes).expect("parse case json");

        let n = doc["n"].as_u64().unwrap() as usize;
        let p = doc["p"].as_u64().unwrap() as usize;
        let task = PrepTask::parse(doc["task"].as_str().unwrap()).expect("parse task");
        let input_kind = doc["input"].as_str().unwrap();

        let custom_ids: Vec<u64> = (0..n).map(|i| 1000 + (i as u64) * 3).collect();
        let (prepared, prepared_custom) = if input_kind == "cells" {
            let x_cells: Vec<Vec<String>> = doc["x_cells"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|c| c.as_str().unwrap().to_string())
                        .collect()
                })
                .collect();
            let y_cells: Vec<String> = doc["y_cells"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect();
            let p1 = prepare_table_cells(&x_cells, &y_cells, task)
                .unwrap_or_else(|e| panic!("{name}: prepare_table_cells failed: {e}"));
            let p2 = prepare_table_cells_with_ids(&x_cells, &y_cells, task, custom_ids.clone())
                .unwrap_or_else(|e| panic!("{name}: prepare_table_cells_with_ids failed: {e}"));
            (p1, p2)
        } else {
            let x_raw_flat = read_f64_tensor(&st, "x_raw");
            let y_raw = read_f64_tensor(&st, "y_raw");
            let x_rows: Vec<Vec<f64>> = (0..n)
                .map(|i| x_raw_flat[i * p..(i + 1) * p].to_vec())
                .collect();
            let p1 = prepare_table(&x_rows, &y_raw, task)
                .unwrap_or_else(|e| panic!("{name}: prepare_table failed: {e}"));
            let p2 = prepare_table_with_ids(&x_rows, &y_raw, task, custom_ids.clone())
                .unwrap_or_else(|e| panic!("{name}: prepare_table_with_ids failed: {e}"));
            (p1, p2)
        };

        assert_eq!(prepared.rows, n, "{name}: rows");
        assert_eq!(prepared.features, p, "{name}: features");
        assert_eq!(
            prepared.classes,
            doc["classes"].as_u64().unwrap() as usize,
            "{name}: classes"
        );
        assert_eq!(
            prepared.row_ids,
            (0..n as u64).collect::<Vec<_>>(),
            "{name}: sequential row_ids"
        );
        assert_eq!(
            prepared_custom.row_ids, custom_ids,
            "{name}: custom row_ids"
        );
        assert_ne!(
            prepared.source_rows_digest(),
            prepared_custom.source_rows_digest(),
            "{name}: source_rows_digest must bind physical row_ids"
        );
        assert_eq!(
            prepared
                .clone()
                .with_row_ids(custom_ids)
                .unwrap()
                .source_rows_digest(),
            prepared_custom.source_rows_digest(),
            "{name}: with_row_ids digest matches prepare_table*_with_ids"
        );

        let expected_kinds: Vec<ColumnKind> = doc["kinds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| match k.as_str().unwrap() {
                "numeric" => ColumnKind::Numeric,
                "categorical" => ColumnKind::Categorical,
                other => panic!("unexpected kind {other}"),
            })
            .collect();
        assert_eq!(prepared.transform.kinds, expected_kinds, "{name}: kinds");

        let x_scaled_ref = read_f64_tensor(&st, "x_scaled");
        let y_scaled_ref = read_f64_tensor(&st, "y_scaled");
        let dx = max_abs_diff_f64(
            &prepared.x_scaled,
            &x_scaled_ref,
            &format!("{name}.x_scaled"),
        );
        let dy = max_abs_diff_f64(
            &prepared.y_scaled,
            &y_scaled_ref,
            &format!("{name}.y_scaled"),
        );
        assert!(dx <= 1e-12, "{name}: x_scaled max abs diff {dx:e} > 1e-12");
        assert!(dy <= 1e-12, "{name}: y_scaled max abs diff {dy:e} > 1e-12");

        // Verify cached observed_target_stats against Python reference.
        match (
            &prepared.observed_target_stats,
            &doc["observed_target_stats"],
        ) {
            (None, Value::Null) => {}
            (Some(actual), Value::Array(arr)) => {
                let expected = [arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap()];
                let dts =
                    max_abs_diff_f64(actual, &expected, &format!("{name}.observed_target_stats"));
                assert!(
                    dts <= 1e-12,
                    "{name}: observed_target_stats diff {dts:e} > 1e-12"
                );
                let dts_full = max_abs_diff_f64(
                    &prepared.target_stats,
                    &expected,
                    &format!("{name}.target_stats"),
                );
                assert!(
                    dts_full <= 1e-12,
                    "{name}: target_stats diff {dts_full:e} > 1e-12"
                );
            }
            (a, b) => panic!("{name}: observed_target_stats mismatch: actual={a:?} expected={b:?}"),
        }
        assert!(
            prepared.target_stats[0].is_finite() && prepared.target_stats[1].is_finite(),
            "{name}: full target_stats must be finite"
        );

        // Verify LandmarkFrame conversion on tables meeting the native 20-row minimum.
        if n >= 20 {
            let frame = prepared
                .to_landmark_frame()
                .unwrap_or_else(|e| panic!("{name}: to_landmark_frame failed: {e}"));
            assert_eq!(frame.rows(), n);
            assert_eq!(frame.features(), p);
        }

        // Verify compute_table_descriptors & cached high_table_descriptors across all levels.
        for &lvl in &levels {
            let ls = lvl.as_str();
            let td = compute_table_descriptors(
                &prepared.x_scaled,
                prepared.rows,
                prepared.features,
                &prepared.y_scaled,
                lvl,
            );
            assert_eq!(
                prepared.table_descriptors(lvl),
                td,
                "{name}.{ls}: table_descriptors(lvl)"
            );
            if lvl == Level::High {
                assert_eq!(
                    prepared.high_table_descriptors(),
                    &td,
                    "{name}: cached high_table_descriptors"
                );
            }
            let tv_ref = read_f32_tensor(&st, &format!("table_vec24.{ls}"));
            let cd_ref = read_f32_tensor(&st, &format!("column_desc.{ls}"));
            let ew_ref = read_f32_tensor(&st, &format!("emp_w.{ls}"));
            let cd_flat: Vec<f32> = td.column_desc.iter().flatten().copied().collect();

            let d_tv = max_abs_diff_f32(
                &td.table_vec24,
                &tv_ref,
                &format!("{name}.table_vec24.{ls}"),
            );
            let d_cd = max_abs_diff_f32(&cd_flat, &cd_ref, &format!("{name}.column_desc.{ls}"));
            let d_ew = max_abs_diff_f32(
                &td.empirical_driver_weights,
                &ew_ref,
                &format!("{name}.emp_w.{ls}"),
            );
            assert!(d_tv <= 1e-6, "{name}.table_vec24.{ls}: diff {d_tv:e}");
            assert!(d_cd <= 1e-6, "{name}.column_desc.{ls}: diff {d_cd:e}");
            assert!(d_ew <= 1e-6, "{name}.emp_w.{ls}: diff {d_ew:e}");
        }

        // Verify extract_report_descriptors across all report variants and ShapeOverride.
        let variants = doc["variants"].as_array().unwrap();
        for v_val in variants {
            let v = v_val.as_str().unwrap();
            let rep_json = &doc["reports"][v];
            let rep_view = NativeReportView::from_json(rep_json)
                .unwrap_or_else(|e| panic!("{name}.{v}: NativeReportView::from_json: {e}"));
            for &lvl in &levels {
                let ls = lvl.as_str();
                let mut checks = vec![(format!("{v}.{ls}"), ShapeOverride::default())];
                if v == "partial_override" || v == "empty" {
                    checks.push((format!("{v}+ovr.{ls}"), ShapeOverride::shape_only(n, p)));
                }
                for (key, ovr) in checks {
                    let rd = extract_report_descriptors(&rep_view, lvl, ovr);
                    let v116_ref = read_f32_tensor(&st, &format!("vector116.{key}"));
                    let cs_ref = read_f32_tensor(&st, &format!("column_spectrum.{key}"));
                    let cp_ref = read_f32_tensor(&st, &format!("col_prior.{key}"));
                    let cs_flat: Vec<f32> = rd.column_spectrum.iter().flatten().copied().collect();

                    let d_v116 = max_abs_diff_f32(
                        &rd.vector116,
                        &v116_ref,
                        &format!("{name}.vector116.{key}"),
                    );
                    let d_cs = max_abs_diff_f32(
                        &cs_flat,
                        &cs_ref,
                        &format!("{name}.column_spectrum.{key}"),
                    );
                    let d_cp = max_abs_diff_f32(
                        &rd.column_prior_importance,
                        &cp_ref,
                        &format!("{name}.col_prior.{key}"),
                    );
                    assert!(d_v116 <= 1e-6, "{name}.vector116.{key}: diff {d_v116:e}");
                    assert!(d_cs <= 1e-6, "{name}.column_spectrum.{key}: diff {d_cs:e}");
                    assert!(d_cp <= 1e-6, "{name}.col_prior.{key}: diff {d_cp:e}");
                }
            }
        }
    }
}

#[test]
fn prepared_table_row_ids_and_categories_validation() {
    let x: Vec<Vec<f64>> = (0..24)
        .map(|i| vec![(i % 3) as f64, (i as f64) * 0.5])
        .collect();
    let y: Vec<f64> = (0..24).map(|i| (i % 2) as f64).collect();
    let table = prepare_table(&x, &y, PrepTask::Classification).expect("prepare_table");

    // Duplicate row_ids rejected.
    let mut dup_ids: Vec<u64> = (0..24).collect();
    dup_ids[5] = dup_ids[2];
    assert!(table.clone().with_row_ids(dup_ids).is_err());

    // Wrong length row_ids rejected.
    assert!(table.clone().with_row_ids(vec![0, 1, 2]).is_err());

    // Explicit categorical metadata reaches LandmarkFrame::from_columns.
    let with_cats = table
        .clone()
        .with_category_tokens(vec![vec!["a".into(), "b".into(), "c".into()], Vec::new()])
        .expect("with_category_tokens");
    assert_eq!(with_cats.categories, vec![Some(3), None]);
    assert_ne!(table.source_rows_digest(), with_cats.source_rows_digest());
    let frame = with_cats.to_landmark_frame().expect("to_landmark_frame");
    assert_eq!(frame.rows(), 24);
    assert_eq!(frame.features(), 2);

    // prepare_table_mixed with declared kinds and target_stats override on ShapeOverride.
    let x_cells: Vec<Vec<String>> = (0..24)
        .map(|i| {
            vec![
                if i % 3 == 0 {
                    "cat_a".into()
                } else if i % 3 == 1 {
                    "cat_b".into()
                } else {
                    "cat_c".into()
                },
                format!("{}", (i as f64) * 0.25),
            ]
        })
        .collect();
    let mixed = prepare_table_mixed(
        &x_cells,
        &y,
        Some(&[ColumnKind::Categorical, ColumnKind::Numeric]),
        PrepTask::Classification,
        Some((100..124).collect()),
    )
    .expect("prepare_table_mixed");
    assert_eq!(
        mixed.transform.kinds,
        vec![ColumnKind::Categorical, ColumnKind::Numeric]
    );
    assert_eq!(mixed.categories, vec![Some(4), None]); // __other__ + 3 retained tokens
    assert_eq!(mixed.row_ids, (100..124).collect::<Vec<_>>());
    let mixed_frame = mixed.to_landmark_frame().expect("mixed to_landmark_frame");
    assert_eq!(mixed_frame.rows(), 24);
    assert_eq!(mixed_frame.features(), 2);

    let empty_rep = NativeReportView::empty();
    let rd_full_ovr = extract_report_descriptors(&empty_rep, Level::High, mixed.shape_override());
    let sq_ts = mixed.squashed_target_stats();
    assert!((rd_full_ovr.vector116[85] - sq_ts[0] as f32).abs() <= 1e-6);
    assert!((rd_full_ovr.vector116[86] - sq_ts[1] as f32).abs() <= 1e-6);
    assert_eq!(rd_full_ovr.vector116[87], 1.0); // is_clf
}

#[test]
#[should_panic(expected = "non-finite value")]
fn max_abs_diff_f64_rejects_one_sided_nan() {
    let _ = max_abs_diff_f64(&[0.0, f64::NAN], &[0.0, 1.0], "nan_check");
}

#[test]
#[should_panic(expected = "non-finite value")]
fn max_abs_diff_f32_rejects_one_sided_nan() {
    let _ = max_abs_diff_f32(&[0.0, f32::NAN], &[0.0, 1.0], "nan_check32");
}
