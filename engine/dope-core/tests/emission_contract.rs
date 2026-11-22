//! Behavioral tests for Phase P2 row-keyed emission contract (`T11`, `T12`, `T25`, `distill`).

use dope_core::rng::{row_keyed_normal, DEFAULT_NAMESPACE_128, PURPOSE_COPULA_EPS};
use dope_core::{
    decode, fit, output, sample_with_mode, EmissionMode, EmitOptions, FitOptions, RawTable,
};

fn synthetic_reg_table(n: usize) -> RawTable {
    let mut csv = String::from("x0,x1,x2,y\n");
    for i in 0..n {
        let x0 = ((i * 29) % 100) as f64 / 50.0 - 1.0;
        let x1 = ((i * 47) % 100) as f64 / 50.0 - 1.0;
        let x2 = ((i * 71) % 100) as f64 / 50.0 - 1.0;
        let y = 1.8 * x0 - 0.9 * x1 + 0.5 * x2 + 0.05 * ((i % 7) as f64 - 3.0);
        csv += &format!("{x0:.4},{x1:.4},{x2:.4},{y:.4}\n");
    }
    RawTable::from_reader(csv.as_bytes()).unwrap()
}

fn synthetic_mc_table(n: usize) -> RawTable {
    let mut csv = String::from("x0,x1,cls\n");
    for i in 0..n {
        let x0 = ((i * 31) % 100) as f64 / 50.0 - 1.0;
        let x1 = ((i * 53) % 100) as f64 / 50.0 - 1.0;
        let cls = if x0 + x1 > 0.4 {
            "c2"
        } else if x0 - x1 < -0.4 {
            "c0"
        } else {
            "c1"
        };
        csv += &format!("{x0:.4},{x1:.4},{cls}\n");
    }
    RawTable::from_reader(csv.as_bytes()).unwrap()
}

#[test]
fn t11_row_keyed_emission_is_batch_chunk_and_offset_invariant() {
    let raw = synthetic_reg_table(160);
    let fitted = fit(
        &raw,
        3,
        &FitOptions {
            budget: Some(8192),
            seed: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let program = decode(&fitted.bytes).unwrap();

    let whole = sample_with_mode(
        &program,
        &EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 100,
            seed: 99,
            table_id: 3,
            row_offset: 0,
            ..Default::default()
        },
    )
    .unwrap();

    // Split into arbitrary non-uniform chunks (0..37, 37..73, 73..100).
    let mut stitched_y = Vec::new();
    let mut stitched_x = vec![Vec::new(); program.features()];
    for &(off, len) in &[(0u64, 37usize), (37, 36), (73, 27)] {
        let part = sample_with_mode(
            &program,
            &EmitOptions {
                mode: EmissionMode::Faithful,
                rows: len,
                seed: 99,
                table_id: 3,
                row_offset: off,
                ..Default::default()
            },
        )
        .unwrap();
        stitched_y.extend_from_slice(&part.samples.y);
        for (col, part_col) in stitched_x.iter_mut().zip(&part.samples.x) {
            col.extend_from_slice(part_col);
        }
    }
    assert_eq!(whole.samples.y.len(), stitched_y.len());
    for (i, (&wy, &sy)) in whole.samples.y.iter().zip(&stitched_y).enumerate() {
        assert_eq!(wy.to_bits(), sy.to_bits());
        for (wc, sc) in whole.samples.x.iter().zip(&stitched_x) {
            assert_eq!(wc[i].to_bits(), sc[i].to_bits());
        }
    }
}

#[test]
fn t12_distinct_table_ids_yield_independent_streams() {
    let raw = synthetic_reg_table(140);
    let fitted = fit(
        &raw,
        3,
        &FitOptions {
            budget: Some(8192),
            seed: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let program = decode(&fitted.bytes).unwrap();

    let t0 = sample_with_mode(
        &program,
        &EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 500,
            seed: 42,
            table_id: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let t1 = sample_with_mode(
        &program,
        &EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 500,
            seed: 42,
            table_id: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let ma = t0.samples.x[0].iter().sum::<f64>() / 500.0;
    let mb = t1.samples.x[0].iter().sum::<f64>() / 500.0;
    let (mut num, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&a, &b) in t0.samples.x[0].iter().zip(&t1.samples.x[0]) {
        num += (a - ma) * (b - mb);
        va += (a - ma).powi(2);
        vb += (b - mb).powi(2);
    }
    let corr = num / (va * vb).sqrt().max(1e-12);
    assert!(
        corr.abs() < 0.15,
        "distinct table_id streams must be uncorrelated, got corr={corr}"
    );
}

#[test]
fn t25_crn_excludes_packet_digest_and_correlates_across_candidate_packets() {
    let raw = synthetic_reg_table(180);
    let p_small = decode(
        &fit(
            &raw,
            3,
            &FitOptions {
                budget: Some(4096),
                seed: 7,
                ..Default::default()
            },
        )
        .unwrap()
        .bytes,
    )
    .unwrap();
    let p_large = decode(
        &fit(
            &raw,
            3,
            &FitOptions {
                budget: Some(16384),
                seed: 7,
                ..Default::default()
            },
        )
        .unwrap()
        .bytes,
    )
    .unwrap();

    // Underlying stateless normal draw depends only on key, never packet bytes.
    let z_a = row_keyed_normal(&DEFAULT_NAMESPACE_128, 7, 0, 12, 0, PURPOSE_COPULA_EPS, 0);
    let z_b = row_keyed_normal(&DEFAULT_NAMESPACE_128, 7, 0, 12, 0, PURPOSE_COPULA_EPS, 0);
    assert_eq!(z_a.to_bits(), z_b.to_bits());

    let opts = EmitOptions {
        mode: EmissionMode::Matched,
        rows: 300,
        seed: 7,
        table_id: 0,
        ..Default::default()
    };
    let s1 = sample_with_mode(&p_small, &opts).unwrap();
    let s2 = sample_with_mode(&p_large, &opts).unwrap();
    let m1 = s1.samples.y.iter().sum::<f64>() / 300.0;
    let m2 = s2.samples.y.iter().sum::<f64>() / 300.0;
    let (mut num, mut v1, mut v2) = (0.0, 0.0, 0.0);
    for (&a, &b) in s1.samples.y.iter().zip(&s2.samples.y) {
        num += (a - m1) * (b - m2);
        v1 += (a - m1).powi(2);
        v2 += (b - m2).powi(2);
    }
    let corr = num / (v1 * v2).sqrt().max(1e-12);
    assert!(
        corr > 0.70,
        "CRN draws across two packets on the same table must strongly couple, got corr={corr}"
    );
}

#[test]
fn distill_mode_emits_normalized_weights_pair_ids_and_sidecar_csv() {
    let raw_reg = synthetic_reg_table(140);
    let p_reg = decode(
        &fit(
            &raw_reg,
            3,
            &FitOptions {
                budget: Some(8192),
                seed: 3,
                ..Default::default()
            },
        )
        .unwrap()
        .bytes,
    )
    .unwrap();
    let d_reg = sample_with_mode(
        &p_reg,
        &EmitOptions {
            mode: EmissionMode::Distill,
            rows: 40,
            seed: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let w_reg = d_reg.weights.as_ref().unwrap();
    let pids_reg = d_reg.pair_ids.as_ref().unwrap();
    assert_eq!(d_reg.samples.y.len(), 40);
    assert_eq!(w_reg.len(), 40);
    assert_eq!(pids_reg.len(), 40);
    for pair in 0..20 {
        let i0 = pair * 2;
        let i1 = pair * 2 + 1;
        assert_eq!(pids_reg[i0], pids_reg[i1]);
        assert!((w_reg[i0] + w_reg[i1] - 1.0).abs() < 1e-12);
        for j in 0..p_reg.features() {
            assert_eq!(
                d_reg.samples.x[j][i0].to_bits(),
                d_reg.samples.x[j][i1].to_bits()
            );
        }
    }
    let mut sidecar = Vec::new();
    output::write_distill_sidecar(w_reg, pids_reg, &mut sidecar).unwrap();
    let sc_str = String::from_utf8(sidecar).unwrap();
    assert!(sc_str.starts_with("row_id,weight,pair_id\n"));
    assert_eq!(sc_str.lines().count(), 41);

    // Multiclass distill emits Rao-Blackwellized class probability weights summing to 1 per pair_id.
    let raw_mc = synthetic_mc_table(150);
    let p_mc = decode(
        &fit(
            &raw_mc,
            2,
            &FitOptions {
                budget: Some(8192),
                seed: 4,
                ..Default::default()
            },
        )
        .unwrap()
        .bytes,
    )
    .unwrap();
    let d_mc = sample_with_mode(
        &p_mc,
        &EmitOptions {
            mode: EmissionMode::Distill,
            rows: 30,
            seed: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let w_mc = d_mc.weights.as_ref().unwrap();
    let pids_mc = d_mc.pair_ids.as_ref().unwrap();
    assert_eq!(w_mc.len(), 30);
    for pair in 0..10 {
        let s: f64 = w_mc[pair * 3..(pair + 1) * 3].iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert_eq!(pids_mc[pair * 3], pids_mc[pair * 3 + 2]);
    }
}
