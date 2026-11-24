use super::build_zero_init_v8_safetensors;
use dope_core::rng::Rng;
use dope_core::single_encoding::v8::model::{
    ACTIVE_RANK_V1, ACTIVE_RANK_V1_MARKER_TENSOR, DECODER_CAPABILITY_KEY,
};
use dope_core::single_encoding::v8::{
    build_decode_plan_v8, decode_dse8_row_at, fit_dse8_packet, pack_dse8_packet,
    unpack_dse8_packet, Dse8FitOptions, ReplayArmV8, ReplayPlanV8, SingleEncV8Weights,
    FLAG_ACTIVE_RANK_V1, V8_MAX_RANK,
};

fn checkpoint(capability: Option<&str>, first_changed_lane: usize, target_bias: f32) -> Vec<u8> {
    checkpoint_with_marker(
        capability,
        capability == Some(ACTIVE_RANK_V1),
        first_changed_lane,
        target_bias,
    )
}

fn checkpoint_with_marker(
    capability: Option<&str>,
    with_u8_marker: bool,
    first_changed_lane: usize,
    target_bias: f32,
) -> Vec<u8> {
    let bytes = build_zero_init_v8_safetensors(16, 4);
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data_at = 8 + header_len;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..data_at]).unwrap();
    let mut data = bytes[data_at..].to_vec();
    let bias_at = header["delta_L.bias"]["data_offsets"][0].as_u64().unwrap() as usize;
    for lane in first_changed_lane..V8_MAX_RANK {
        let at = bias_at + 4 * lane;
        data[at..at + 4].copy_from_slice(&4.0f32.to_le_bytes());
    }
    let at = header["delta_W.bias"]["data_offsets"][0].as_u64().unwrap() as usize;
    data[at..at + 4].copy_from_slice(&target_bias.to_le_bytes());
    if let Some(capability) = capability {
        header["__metadata__"] = serde_json::json!({ (DECODER_CAPABILITY_KEY): capability });
    }
    if with_u8_marker {
        let off = data.len();
        header[ACTIVE_RANK_V1_MARKER_TENSOR] = serde_json::json!({
            "dtype": "U8",
            "shape": [1],
            "data_offsets": [off, off + 1],
        });
        data.push(1u8);
    }
    let header = serde_json::to_vec(&header).unwrap();
    [
        &(header.len() as u64).to_le_bytes(),
        header.as_slice(),
        &data,
    ]
    .concat()
}

#[test]
fn corrected_checkpoint_ignores_inactive_loading_sentinels_at_every_rank() {
    for width in [48, 64] {
        let mut rng = Rng::new(129);
        let x: Vec<Vec<_>> = (0..width)
            .map(|_| (0..80).map(|_| rng.normal()).collect())
            .collect();
        let y: Vec<_> = x[0].iter().map(|&value| 0.4 * value).collect();
        let opts = Dse8FitOptions {
            budget_bytes: 8192,
            normalized_only: true,
            enable_regimes: false,
            ..Default::default()
        };
        let fitted = fit_dse8_packet(&x, &y, &opts).unwrap();
        for rank in [0, 2, 8, 12] {
            let mut packet = unpack_dse8_packet(&fitted).unwrap();
            packet.header.rank_use = rank as u8;
            packet = unpack_dse8_packet(&pack_dse8_packet(&packet, 8192).unwrap()).unwrap();
            assert_eq!(packet.card.col_indices.len(), width);
            assert_eq!(usize::from(packet.header.rank_use), rank);
            let base = build_decode_plan_v8(&packet, None).unwrap();
            let legacy =
                SingleEncV8Weights::from_safetensors_bytes(&checkpoint(None, rank, 0.0), 4);
            let corrected = SingleEncV8Weights::from_safetensors_bytes(
                &checkpoint(Some(ACTIVE_RANK_V1), rank, 0.0),
                4,
            )
            .unwrap();
            let legacy = legacy.unwrap();
            assert_ne!(legacy.fingerprint(), corrected.fingerprint());
            assert!(build_decode_plan_v8(&packet, Some(&corrected)).is_err());
            packet.weights_fp = Some(legacy.fingerprint());
            let legacy_plan = build_decode_plan_v8(&packet, Some(&legacy)).unwrap();
            assert!(build_decode_plan_v8(&packet, Some(&corrected)).is_err());
            packet.weights_fp = Some(corrected.fingerprint());
            // Without FLAG_ACTIVE_RANK_V1 on the packet, corrected checkpoint must refuse!
            assert!(build_decode_plan_v8(&packet, Some(&corrected)).is_err());
            packet.header.flags |= FLAG_ACTIVE_RANK_V1;
            let corrected_plan = build_decode_plan_v8(&packet, Some(&corrected)).unwrap();

            assert_eq!(corrected_plan.l_eff, base.l_eff);
            assert_eq!(corrected_plan.noise_sd, base.noise_sd);
            assert_eq!(legacy_plan.noise_sd == base.noise_sd, rank == V8_MAX_RANK);
            for row in 0..16 {
                let expected = decode_dse8_row_at(&base, 7, row);
                let actual = decode_dse8_row_at(&corrected_plan, 7, row);
                assert_eq!(actual.x_unit, expected.x_unit);
                assert_eq!(actual.missing_active, expected.missing_active);
                assert_eq!(actual.logits, expected.logits);
                assert_eq!(actual.y_unit.to_bits(), expected.y_unit.to_bits());
            }
        }
    }
}

#[test]
fn unknown_decoder_capability_is_refused() {
    let checkpoint = checkpoint(Some("unsupported"), V8_MAX_RANK, 0.0);
    assert!(SingleEncV8Weights::from_safetensors_bytes(&checkpoint, 4).is_err());
    // Metadata-only active-rank-v1 lacking the U8 guard tensor must also be refused.
    let meta_only = checkpoint_with_marker(Some(ACTIVE_RANK_V1), false, V8_MAX_RANK, 0.0);
    assert!(SingleEncV8Weights::from_safetensors_bytes(&meta_only, 4).is_err());
}

#[test]
fn analytic_features_bypass_tail_clipping_with_a_learned_target() {
    let mut rng = Rng::new(9);
    let x: Vec<Vec<_>> = (0..2)
        .map(|_| (0..64).map(|_| rng.normal()).collect())
        .collect();
    let y = x[0].clone();
    let opts = Dse8FitOptions {
        enable_regimes: false,
        ..Default::default()
    };
    let fitted = fit_dse8_packet(&x, &y, &opts).unwrap();
    let weights =
        SingleEncV8Weights::from_safetensors_bytes(&checkpoint(None, V8_MAX_RANK, 2.0), 4).unwrap();
    let mut packet = unpack_dse8_packet(&fitted).unwrap();
    packet.header.rank_use = 0;
    packet.header.inv_nu = 0.45;
    packet.weights_fp = Some(weights.fingerprint());
    let packet = unpack_dse8_packet(&pack_dse8_packet(&packet, 2048).unwrap()).unwrap();
    let analytic = ReplayPlanV8::new(&packet, &weights, ReplayArmV8::Analytic).unwrap();
    let learned_y = ReplayPlanV8::new(&packet, &weights, ReplayArmV8::LearnedY).unwrap();
    let learned = ReplayPlanV8::new(&packet, &weights, ReplayArmV8::Learned).unwrap();
    let mut found_tail = false;
    for index in 0..2000 {
        let mut base_latents = Vec::new();
        let base = analytic.row_at(11, index, |_, z| base_latents = z);
        if base_latents.iter().all(|value| value.abs() <= 5.0) {
            continue;
        }
        let hybrid = learned_y.row_at(11, index, |_, z| assert_eq!(z, base_latents));
        assert_eq!(base.x_unit, hybrid.x_unit);
        assert_eq!(base.missing_active, hybrid.missing_active);
        learned.row_at(11, index, |_, z| {
            assert!(z.iter().all(|v| v.abs() <= 5.0));
        });
        found_tail = true;
        break;
    }
    assert!(
        found_tail,
        "the common counters must exercise a clipped tail"
    );
}
