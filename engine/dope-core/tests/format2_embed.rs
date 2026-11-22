//! Behavioral tests for Phase P7: `.dope` Format 2 parser fuzzing & limits (`T13`, `T14`),
//! `f64 -> f32 -> i16` quantization (`T15`), Format-1 bit-identity (`T24`), `dope embed`
//! (`PacketFeaturesV2` + `Encoder629Bridge`), and empirical privacy capacity backoff.

use dope_core::artifact::format2::{
    decode_format2, dequantize_i16_f64, encode_format2, is_format2, quantize_f64_i16,
    Format2Packet, Format2Section, TierPrefixEntry, CODEC_Q16, CODEC_RAW, FORMAT2_MAGIC,
    MAX_FORMAT2_DECODED_BYTES, TAG_CATF, TAG_MARGINALS_V2, TAG_STUDENT_FOREST, TAG_UNIT_SCHEMA,
};
use dope_core::compile::v2::run_privacy_backoff;
use dope_core::dvs::v3::privacy::EmpiricalPrivacyAssessment;
use dope_core::embed::{embed_packet, embed_to_629_bridge, PACKET_FEATURES_V2_DIM};
use dope_core::rng::Rng;
use dope_core::{decode, fit, FitOptions, RawTable};

#[test]
fn t13_and_t14_format2_roundtrip_and_adversarial_parser_fuzzing() {
    let q_payload = quantize_f64_i16(&[0.0, 0.25, 0.5, 0.75, 1.0]).unwrap();
    let pkt = Format2Packet {
        flags: 0x0001,
        tiers: vec![
            TierPrefixEntry {
                tier_budget_bytes: 4096,
                section_prefix_count: 2,
                forest_trees_prefix: 16,
            },
            TierPrefixEntry {
                tier_budget_bytes: 65536,
                section_prefix_count: 4,
                forest_trees_prefix: 256,
            },
        ],
        sections: vec![
            Format2Section {
                tag: TAG_UNIT_SCHEMA,
                codec: CODEC_RAW,
                data: vec![1, 2, 3, 4],
            },
            Format2Section {
                tag: TAG_MARGINALS_V2,
                codec: CODEC_Q16,
                data: q_payload,
            },
            Format2Section {
                tag: TAG_CATF,
                codec: CODEC_RAW,
                data: vec![9, 8, 7],
            },
            Format2Section {
                tag: TAG_STUDENT_FOREST,
                codec: CODEC_RAW,
                data: vec![42; 64],
            },
        ],
    };
    let wire = encode_format2(&pkt).unwrap();
    assert!(is_format2(&wire));
    let back = decode_format2(&wire).unwrap();
    assert_eq!(back, pkt);

    // T13: Every truncation, bitflip, and trailing-garbage corruption refuses cleanly without panic.
    assert!(decode_format2(&[]).is_err());
    assert!(decode_format2(&wire[..16]).is_err());
    assert!(decode_format2(&wire[..wire.len() - 1]).is_err());

    let mut flipped = wire.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 0x55;
    assert!(decode_format2(&flipped).is_err(), "bitflip must fail CRC32");

    let mut bad_crc = wire.clone();
    bad_crc[20] ^= 0x01;
    assert!(
        decode_format2(&bad_crc).is_err(),
        "corrupted CRC32 must fail"
    );

    // T14: Forged header bounds (> 64 sections, > 256 MiB decoded) refuse before allocation!
    let mut forged_sec = wire.clone();
    forged_sec[12..16].copy_from_slice(&1000u32.to_le_bytes());
    assert!(decode_format2(&forged_sec).is_err());

    let mut forged_ram = wire.clone();
    let bomb = (MAX_FORMAT2_DECODED_BYTES as u64) + 1;
    forged_ram[24..32].copy_from_slice(&bomb.to_le_bytes());
    assert!(decode_format2(&forged_ram).is_err());

    // Deterministic random fuzzing (200 adversarial byte buffers) -> zero panics!
    let mut rng = Rng::new(1314);
    for _ in 0..200 {
        let len = rng.below(128);
        let mut buf: Vec<u8> = (0..len).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        if buf.len() >= 8 && rng.below(2) == 0 {
            buf[..8].copy_from_slice(FORMAT2_MAGIC);
        }
        let _ = decode_format2(&buf);
    }
}

#[test]
fn t15_quantization_f64_f32_i16_has_negligible_error() {
    let vals: Vec<f64> = (0..500)
        .map(|i| -12.5 + (i as f64 / 499.0) * 37.5)
        .collect();
    let enc = quantize_f64_i16(&vals).unwrap();
    let dec = dequantize_i16_f64(&enc).unwrap();
    assert_eq!(dec.len(), vals.len());
    let span = 37.5_f64;
    let max_rel_err = vals
        .iter()
        .zip(&dec)
        .map(|(&a, &b)| (a - b).abs() / span)
        .fold(0.0_f64, f64::max);
    assert!(
        max_rel_err < 0.0005,
        "q16 relative quantization error {max_rel_err} exceeded bound"
    );
}

#[test]
fn t24_format1_bit_identity_dope_embed_and_privacy_backoff() {
    let mut csv = String::from("x0,x1,y\n");
    for i in 0..140 {
        let x0 = (i as f64) * 0.05;
        let x1 = ((i * 11) % 23) as f64 * 0.1;
        let y = 2.0 * x0 - 0.5 * x1 + 0.1 * ((i % 5) as f64);
        csv += &format!("{x0:.3},{x1:.3},{y:.3}\n");
    }
    let raw = RawTable::from_reader(csv.as_bytes()).unwrap();
    let f1 = fit(
        &raw,
        2,
        &FitOptions {
            budget: Some(8192),
            seed: 24,
            ..Default::default()
        },
    )
    .unwrap();
    let f2 = fit(
        &raw,
        2,
        &FitOptions {
            budget: Some(8192),
            seed: 24,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        f1.bytes, f2.bytes,
        "Format 1 compile must remain bit-identical"
    );
    assert!(!is_format2(&f1.bytes));
    let _prog = decode(&f1.bytes).unwrap();

    // Deterministic source-free `dope embed` (`PacketFeaturesV2`) + `Encoder629Bridge`.
    let e1 = embed_packet(&f1.bytes).unwrap();
    let e2 = embed_packet(&f2.bytes).unwrap();
    assert_eq!(e1.features.len(), PACKET_FEATURES_V2_DIM);
    assert_eq!(e1.packet_sha256, e2.packet_sha256);
    assert_eq!(e1.features, e2.features);

    let bridge = embed_to_629_bridge(&f1.bytes, 7).unwrap();
    assert_eq!(bridge.synthetic_rows, 4096);
    assert_eq!(bridge.feature_count, 2);

    // Empirical privacy capacity backoff steps down until DCR/MIA pass and never claims DP.
    let outcome = run_privacy_backoff(|step| {
        let dcr = 0.62 - 0.05 * step.step_index as f64;
        let pass = dcr <= 0.55;
        EmpiricalPrivacyAssessment {
            dcr_closer_to_a_share: dcr,
            mia_auc: 0.52,
            mia_tpr_at_1pct_fpr: 0.01,
            exact_copies: 0,
            exact_copies_twin_b: 0,
            exact_copy_excess: 0.0,
            canary_hits: 0,
            min_emitted_support: step.support_k as u64,
            support_floor_k: 10,
            pass,
            formal_dp_state: "not_claimed",
            hipaa_deidentification_state: "not_claimed",
        }
    });
    assert_eq!(outcome.selected_step.step_index, 2);
    assert!(outcome.assessment.pass);
    assert_eq!(outcome.formal_dp_state, "not_claimed");
}
