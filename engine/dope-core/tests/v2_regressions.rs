//! Regression tests for DOPE v2.0.0:
//! 1. DSE7 binary custom-label encoding (`{-1, 1}`, `{1, 2}`).
//! 2. Format-2 chunk-invariance honoring `row_offset` and `namespace_128`.
//! 3. `embed_packet` & `embed_to_629_bridge` across `DSE7` packets with zero raw-value echo.

use dope_core::artifact::format2_codec::decode_format2_program;
use dope_core::compile::v2::{fit_v2, FitV2Options};
use dope_core::embed::{embed_packet, embed_to_629_bridge, PACKET_FEATURES_V2_DIM};
use dope_core::rng::Rng;
use dope_core::sample::{EmissionMode, EmitOptions};
use dope_core::schema::Task;
use dope_core::single_encoding::{fit_dse7_packet, unpack_dse7_packet};
use dope_core::table::RawTable;

#[test]
fn dse7_binary_preserves_non_canonical_labels_minus1_plus1_and_1_2() {
    let n = 120;
    let mut rng = Rng::new(0xB1A5_ED12);
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    for _ in 0..n {
        x0.push(rng.normal());
        x1.push(rng.normal());
    }
    for (lo, hi) in [(-1.0f64, 1.0f64), (1.0f64, 2.0f64)] {
        let y: Vec<f64> = (0..n)
            .map(|i| if x0[i] + 0.3 * x1[i] > 0.0 { hi } else { lo })
            .collect();
        let pkt_bytes = fit_dse7_packet(&[x0.clone(), x1.clone()], &y, 1, 2048)
            .expect("binary DSE7 fit must succeed for custom class labels");
        assert!(pkt_bytes.len() <= 2048);
        let (pkt, hdr, card) = unpack_dse7_packet(&pkt_bytes).expect("unpack binary DSE7");
        assert_eq!(hdr.task_id, 1);
        assert_eq!(hdr.k_eff, 2);
        assert_eq!(card.classes, vec![lo as i32, hi as i32]);
        assert!(
            pkt.w_base[0][0] > 0.2,
            "expected positive slope on x0 for ({lo}, {hi}), got {}",
            pkt.w_base[0][0]
        );
    }
}

#[test]
fn format2_chunk_invariance_honors_row_offset_and_namespace() {
    let n = 160;
    let mut rng = Rng::new(0xF022_C0DE);
    let x0: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let x1: Vec<f64> = (0..n).map(|i| 0.6 * x0[i] + 0.8 * rng.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.5 * x0[i] - 0.7 * x1[i] + 0.3 * rng.normal())
        .collect();
    let table = RawTable {
        headers: vec!["x0".into(), "x1".into(), "y".into()],
        columns: vec![
            x0.iter().map(|v| format!("{v:.6}")).collect(),
            x1.iter().map(|v| format!("{v:.6}")).collect(),
            y.iter().map(|v| format!("{v:.6}")).collect(),
        ],
    };
    let res = fit_v2(
        &table,
        2,
        &FitV2Options {
            budget: Some(2048),
            task: Some(Task::Regression),
            seed: 42,
            ..Default::default()
        },
    )
    .expect("format2 fit must succeed");
    let prog = decode_format2_program(&res.bytes).expect("decode format2 program");

    for mode in [EmissionMode::Faithful, EmissionMode::Distill] {
        let full = prog
            .sample_with_mode(&EmitOptions {
                mode,
                rows: 30,
                seed: 99,
                table_id: 7,
                row_offset: 0,
                ..Default::default()
            })
            .expect("full sample");
        let part_a = prog
            .sample_with_mode(&EmitOptions {
                mode,
                rows: 12,
                seed: 99,
                table_id: 7,
                row_offset: 0,
                ..Default::default()
            })
            .expect("part_a sample");
        let part_b = prog
            .sample_with_mode(&EmitOptions {
                mode,
                rows: 18,
                seed: 99,
                table_id: 7,
                row_offset: 12,
                ..Default::default()
            })
            .expect("part_b sample");
        for r in 0..12 {
            assert_eq!(full.samples.y[r], part_a.samples.y[r]);
            assert_eq!(full.samples.x[0][r], part_a.samples.x[0][r]);
        }
        for r in 0..18 {
            assert_eq!(full.samples.y[12 + r], part_b.samples.y[r]);
            assert_eq!(full.samples.x[0][12 + r], part_b.samples.x[0][r]);
        }
        if let (Some(pf), Some(pb)) = (&full.pair_ids, &part_b.pair_ids) {
            for r in 0..18 {
                assert_eq!(pf[12 + r], pb[r]);
            }
        }
    }

    let ns_a = prog
        .sample_with_mode(&EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 16,
            seed: 99,
            table_id: 7,
            namespace_128: [1u8; 16],
            ..Default::default()
        })
        .expect("ns_a");
    let ns_b = prog
        .sample_with_mode(&EmitOptions {
            mode: EmissionMode::Faithful,
            rows: 16,
            seed: 99,
            table_id: 7,
            namespace_128: [2u8; 16],
            ..Default::default()
        })
        .expect("ns_b");
    assert_ne!(ns_a.samples.y, ns_b.samples.y);
}

#[test]
fn embed_packet_and_bridge_support_dse7_without_raw_echo() {
    let n = 120;
    let mut rng = Rng::new(0xD5E7_E8B3);
    let x0: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let x1: Vec<f64> = (0..n).map(|i| 0.5 * x0[i] + 0.8 * rng.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.2 * x0[i] - 0.5 * x1[i] + 0.2 * rng.normal())
        .collect();
    let pkt_base = fit_dse7_packet(&[x0.clone(), x1.clone()], &y, 0, 2048).expect("dse7 base");
    let x0_shifted: Vec<f64> = x0.iter().map(|v| 1e6 + 500.0 * v).collect();
    let y_shifted: Vec<f64> = y.iter().map(|v| -2e5 + 100.0 * v).collect();
    let pkt_shifted =
        fit_dse7_packet(&[x0_shifted, x1], &y_shifted, 0, 2048).expect("dse7 shifted");

    let e1 = embed_packet(&pkt_base).expect("embed base");
    let e2 = embed_packet(&pkt_base).expect("embed deterministic");
    let es = embed_packet(&pkt_shifted).expect("embed shifted");
    assert_eq!(e1.protocol, "dope-packet-features-v2");
    assert_eq!(e1.format_version, 7);
    assert_eq!(e1.features.len(), PACKET_FEATURES_V2_DIM);
    assert_eq!(e1.features, e2.features);
    assert!(e1.features.iter().all(|v| v.is_finite()));
    // Structural features must NOT echo the 1e6 raw shift/scale!
    for (a, b) in e1.features.iter().zip(&es.features) {
        assert!(
            (a - b).abs() < 1e-2,
            "dse7 embedding leaked raw scale/offset: {a} vs {b}"
        );
    }

    let bridge = embed_to_629_bridge(&pkt_base, 123).expect("629 bridge for dse7");
    assert_eq!(bridge.synthetic_rows, 4096);
    assert_eq!(bridge.feature_count, 2);
    assert!(bridge.target_mean.is_finite() && bridge.target_std.is_finite());
}
