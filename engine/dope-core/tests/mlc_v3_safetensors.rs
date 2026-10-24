use dope_core::mlc_v3::{
    inflate::crc32_ieee, sample_synthetic_table, sample_synthetic_table_with_draws, Dml3Envelope,
    FixedSampleDraws, MlcV3ForwardCheckInput, MlcV3ForwardCheckOutput, MlcV3Model,
    MlcV3SampledTable, MLC_V3_EXPECTED_PARAMETERS, MLC_V3_EXPECTED_TENSORS,
};
use half::f16;
use safetensors::tensor::{Dtype, TensorView};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

const EXPECTED_TRAINED_ST_SHA256: &str =
    "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c";

fn build_deterministic_safetensors_bytes() -> Vec<u8> {
    let specs: &[(&str, &[usize])] = &[
        ("cov_net.net.0.bias", &[128]),
        ("cov_net.net.0.weight", &[128, 192]),
        ("cov_net.net.2.bias", &[128]),
        ("cov_net.net.2.weight", &[128]),
        ("cov_net.net.3.bias", &[64]),
        ("cov_net.net.3.weight", &[64, 128]),
        ("router.backbone.0.bias", &[256]),
        ("router.backbone.0.weight", &[256, 140]),
        ("router.backbone.2.bias", &[256]),
        ("router.backbone.2.weight", &[256]),
        ("router.backbone.3.bias", &[256]),
        ("router.backbone.3.weight", &[256, 256]),
        ("router.backbone.5.bias", &[256]),
        ("router.backbone.5.weight", &[256]),
        ("router.bb_recon_head.bias", &[16]),
        ("router.bb_recon_head.weight", &[16, 256]),
        ("router.dist_latent_head.bias", &[128]),
        ("router.dist_latent_head.weight", &[128, 256]),
        ("router.gate_head.bias", &[10]),
        ("router.gate_head.weight", &[10, 256]),
        ("router.input_norm.bias", &[140]),
        ("router.input_norm.weight", &[140]),
        ("router.inter_attn.bilinear_u.bias", &[128]),
        ("router.inter_attn.bilinear_u.weight", &[128, 256]),
        ("router.inter_attn.bilinear_v.bias", &[128]),
        ("router.inter_attn.bilinear_v.weight", &[128, 256]),
        ("router.inter_attn.norm.bias", &[256]),
        ("router.inter_attn.norm.weight", &[256]),
        ("router.inter_attn.out_proj.bias", &[256]),
        ("router.inter_attn.out_proj.weight", &[256, 384]),
        ("router.inter_attn.qkv.bias", &[96]),
        ("router.inter_attn.qkv.weight", &[96, 32]),
        ("router.inter_attn.token_proj.bias", &[32]),
        ("router.inter_attn.token_proj.weight", &[32, 32]),
        ("router.landmark_recon_head.bias", &[23]),
        ("router.landmark_recon_head.weight", &[23, 256]),
        ("router.level_embed.weight", &[3, 256]),
        ("router.small_embed", &[256]),
        ("router.task_latent_head.bias", &[128]),
        ("router.task_latent_head.weight", &[128, 256]),
        ("router.v3_gate_head.bias", &[4]),
        ("router.v3_gate_head.weight", &[4, 256]),
        ("router.v3_inter_norm.bias", &[116]),
        ("router.v3_inter_norm.weight", &[116]),
        ("target_net.col_gate_head.bias", &[1]),
        ("target_net.col_gate_head.weight", &[1, 145]),
        ("target_net.fc1.bias", &[128]),
        ("target_net.fc1.weight", &[128, 192]),
        ("target_net.fc2.bias", &[128]),
        ("target_net.fc2.weight", &[128, 128]),
        ("target_net.film_beta.bias", &[64]),
        ("target_net.film_beta.weight", &[64, 128]),
        ("target_net.film_gamma.bias", &[64]),
        ("target_net.film_gamma.weight", &[64, 128]),
        ("target_net.out_head.bias", &[1]),
        ("target_net.out_head.weight", &[1, 128]),
    ];
    let mut buffers: Vec<(&str, Vec<usize>, Vec<u8>)> = Vec::with_capacity(specs.len());
    for (idx, &(name, shape)) in specs.iter().enumerate() {
        let count: usize = shape.iter().product();
        let is_norm_w = name.ends_with("norm.weight")
            || name.ends_with(".2.weight")
            || name.ends_with(".5.weight");
        let mut bytes = Vec::with_capacity(count * 4);
        for k in 0..count {
            let val: f32 = if is_norm_w {
                1.0
            } else {
                (((idx * 131 + k * 17) % 97) as f32 - 48.0) * 0.001
            };
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        buffers.push((name, shape.to_vec(), bytes));
    }
    let mut views = BTreeMap::new();
    for (name, shape, bytes) in &buffers {
        views.insert(
            *name,
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        );
    }
    safetensors::tensor::serialize(&views, None).unwrap()
}

fn push_f16(buf: &mut Vec<u8>, vals: &[f64]) {
    for &v in vals {
        buf.extend_from_slice(&f16::from_f64(v).to_le_bytes());
    }
}

fn build_synthetic_dml3_packet(level_id: u8, task_id: u8) -> Vec<u8> {
    let mut core = Vec::new();
    core.extend_from_slice(&[2u8, 1u8, u8::from(task_id == 1), 1u8]);
    core.extend_from_slice(&64u32.to_le_bytes());
    core.extend_from_slice(&4u32.to_le_bytes());
    for &u in &[1u16, 2u16, 5u16, 1u16, 18u16, 0u16] {
        core.extend_from_slice(&u.to_le_bytes());
    }
    for &q in &[0i32, 35_000, 120_000, 15_000, 500_000, -250_000, 250_000] {
        core.extend_from_slice(&q.to_le_bytes());
    }
    for &g in &[0.55f32, 0.45, 0.50, 0.70] {
        core.extend_from_slice(&g.to_le_bytes());
    }
    push_f16(&mut core, &[1.0]);
    push_f16(&mut core, &[0.0; 4]);
    push_f16(&mut core, &[1.0; 4]);
    push_f16(&mut core, &[0.6, 0.1, 0.2, 0.5, -0.3, 0.4, 0.1, 0.7]);
    push_f16(&mut core, &[0.5; 4]);
    push_f16(&mut core, &[0.02; 128]);
    push_f16(&mut core, &[-0.01; 128]);
    for _ in 0..4 {
        push_f16(&mut core, &[-1.5, -0.5, 0.0, 0.5, 1.5]);
    }
    push_f16(&mut core, &[0.0; 4]);
    push_f16(&mut core, &[0.0; 8]);
    core.push(0u8);
    core.extend_from_slice(&[0u8, 1u8]);
    push_f16(&mut core, &[0.05; 18]);
    core.push(2u8);
    core.extend_from_slice(&[0u8, 1u8]);
    core.extend_from_slice(&0i16.to_le_bytes());
    core.extend_from_slice(&1000i16.to_le_bytes());
    push_f16(&mut core, &[-0.2, -0.05, 0.05, 0.2]);

    let mut sc = Vec::new();
    sc.extend_from_slice(&[task_id, if task_id == 2 { 3 } else { 0 }]);
    sc.extend_from_slice(&4u16.to_le_bytes());
    sc.push(1u8);
    for &tv in &[10.0f32, 2.5, 10.0, 3.0, 0.5] {
        sc.extend_from_slice(&tv.to_le_bytes());
    }
    if task_id == 2 {
        for &lbl in &[0.0f32, 1.0, 2.0] {
            sc.extend_from_slice(&lbl.to_le_bytes());
        }
        push_f16(&mut sc, &[0.34, 0.33, 0.33]);
    }
    for _ in 0..4 {
        sc.push(3u8);
        push_f16(&mut sc, &[0.0, 1.0, 5.0, 2.0, 0.0, 0.3]);
        push_f16(&mut sc, &[-2.0, 0.0, 2.0]);
    }

    let mut mc = Vec::new();
    if task_id == 2 {
        for &h in &[3u16, 18u16, 2u16, 1u16] {
            mc.extend_from_slice(&h.to_le_bytes());
        }
        push_f16(&mut mc, &[0.34, 0.33, 0.33]);
        push_f16(&mut mc, &[0.05; 36]);
        push_f16(&mut mc, &[0.1; 6]);
        mc.push(1u8);
        mc.push(0u8);
        push_f16(&mut mc, &[0.0]);
        push_f16(&mut mc, &[-0.1, 0.0, 0.1, 0.1, 0.0, -0.1]);
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&[level_id, task_id, 1u8, 0u8]);
    payload.extend_from_slice(&(core.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(sc.len() as u32).to_le_bytes());
    payload.extend_from_slice(&(mc.len() as u32).to_le_bytes());
    payload.extend_from_slice(&core);
    payload.extend_from_slice(&sc);
    payload.extend_from_slice(&mc);

    let crc = crc32_ieee(&payload);
    let mut wire = b"DML3".to_vec();
    wire.push(3u8);
    wire.push(0u8);
    wire.extend_from_slice(&crc.to_le_bytes());
    wire.extend_from_slice(&payload);
    wire
}

fn max_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

#[derive(Deserialize)]
struct LevelRef {
    conditioning_vec256: Vec<f32>,
    forward_input: MlcV3ForwardCheckInput,
    expected: MlcV3ForwardCheckOutput,
}

#[derive(Deserialize)]
struct ReferenceBundle {
    forward_by_level: BTreeMap<String, LevelRef>,
}

#[test]
fn loads_safetensors_and_verifies_trained_parity_and_fail_closed_wire() {
    let raw_st = build_deterministic_safetensors_bytes();
    let det_model =
        MlcV3Model::from_safetensors_bytes(&raw_st).expect("load deterministic safetensors");
    assert_eq!(det_model.inventory.tensor_count, MLC_V3_EXPECTED_TENSORS);
    assert_eq!(
        det_model.inventory.parameter_count,
        MLC_V3_EXPECTED_PARAMETERS
    );

    // Fail-closed weight checks
    assert!(MlcV3Model::from_safetensors_bytes(b"").is_err());
    assert!(MlcV3Model::from_safetensors_bytes(b"corrupt_safetensors").is_err());

    for (lvl_id, task_id) in [(0u8, 0u8), (1u8, 1u8), (2u8, 2u8), (3u8, 0u8)] {
        let wire = build_synthetic_dml3_packet(lvl_id, task_id);
        let env = Dml3Envelope::from_bytes(&wire).expect("decode synthetic DML3");
        assert!(env.within_tier_cap);
        let sampled =
            sample_synthetic_table(&det_model, &env, 24, 20261023).expect("sample synthetic");
        assert_eq!(sampled.x_raw.len(), 24);
        assert_eq!(sampled.headers.last().unwrap(), "target");

        // Fail-closed wire checks: CRC mismatch, bad magic, bad version, bad flag, truncated
        let mut crc_bad = wire.clone();
        let last = crc_bad.len() - 1;
        crc_bad[last] ^= 0x55;
        assert!(Dml3Envelope::from_bytes(&crc_bad).is_err());
        let mut ver_bad = wire.clone();
        ver_bad[4] = 9;
        assert!(Dml3Envelope::from_bytes(&ver_bad).is_err());
        assert!(Dml3Envelope::from_bytes(&wire[..7]).is_err());
    }

    let ckpt_path = Path::new(
        "../../target/s3-multilevel-training-20261023/v3/dope_mlc_v3_multitask_best.safetensors",
    );
    let fixtures_dir =
        Path::new("../../target/s3-multilevel-training-20261023/v3/rust_parity_fixtures");
    let ref_bundle_path = fixtures_dir.join("reference_bundle.json");
    let require_strict = std::env::var("DOPE_REQUIRE_RUST_PARITY").as_deref() == Ok("1");

    if !ckpt_path.is_file() || !ref_bundle_path.is_file() {
        assert!(
            !require_strict,
            "DOPE_REQUIRE_RUST_PARITY=1 requires trained safetensors and parity fixtures"
        );
        return;
    }

    let trained = MlcV3Model::load_safetensors(ckpt_path).expect("load trained safetensors");
    assert_eq!(trained.inventory.sha256, EXPECTED_TRAINED_ST_SHA256);
    assert_eq!(trained.inventory.tensor_count, 56);
    assert_eq!(trained.inventory.parameter_count, 444_168);

    let bundle: ReferenceBundle =
        serde_json::from_str(&std::fs::read_to_string(&ref_bundle_path).unwrap()).unwrap();
    for lvl in ["small", "medium", "high", "ultra"] {
        let lref = bundle.forward_by_level.get(lvl).expect("level ref");
        let routed = trained
            .router
            .forward(&lref.conditioning_vec256, lvl)
            .expect("route trained");
        let fwd = MlcV3ForwardCheckOutput::evaluate(
            routed,
            &trained.target_net,
            &trained.cov_net,
            &lref.forward_input,
        )
        .expect("eval forward");
        for (k, &v) in &lref.expected.gates {
            assert!((v - fwd.gates[k]).abs() < 1e-5, "gate diff at {lvl}:{k}");
        }
        assert!(max_diff_f32(&lref.expected.task_latent, &fwd.task_latent) < 1e-5);
        assert!(max_diff_f32(&lref.expected.dist_latent, &fwd.dist_latent) < 1e-5);
        assert!(max_diff_f32(&lref.expected.landmark_pred, &fwd.landmark_pred) < 1e-5);
        assert!(max_diff_f32(&lref.expected.bb_pred, &fwd.bb_pred) < 1e-5);
        assert!(
            max_diff_f32(
                &lref.expected.col_gates_from_landmarks,
                &fwd.col_gates_from_landmarks
            ) < 1e-5
        );
        assert!(max_diff_f32(&lref.expected.col_gates_from_xpad, &fwd.col_gates_from_xpad) < 1e-5);
        assert!(
            max_diff_f32(
                &lref.expected.target_net_out_with_landmarks,
                &fwd.target_net_out_with_landmarks
            ) < 1e-5
        );
        assert!(
            max_diff_f32(
                &lref.expected.target_net_out_from_xpad,
                &fwd.target_net_out_from_xpad
            ) < 1e-5
        );
        for (ra, rb) in lref.expected.cov_net_out.iter().zip(fwd.cov_net_out.iter()) {
            assert!(max_diff_f32(ra, rb) < 1e-5);
        }

        for task in ["regression", "classification", "multiclass"] {
            let key = format!("{lvl}_{task}");
            let wire_bytes = std::fs::read(fixtures_dir.join(format!("packet_{key}.dml3")))
                .expect("read packet");
            let env = Dml3Envelope::from_bytes(&wire_bytes).expect("decode python DML3");
            assert!(env.within_tier_cap);
            let draws: FixedSampleDraws = serde_json::from_str(
                &std::fs::read_to_string(fixtures_dir.join(format!("draws_{key}.json"))).unwrap(),
            )
            .unwrap();
            let exp_fixed: MlcV3SampledTable = serde_json::from_str(
                &std::fs::read_to_string(fixtures_dir.join(format!("rs_fixed_{key}.json")))
                    .unwrap(),
            )
            .unwrap();
            let got = sample_synthetic_table_with_draws(&trained, &env, 36, 42, Some(&draws))
                .expect("fixed draw sample");
            for (ra, rb) in exp_fixed.x_raw.iter().zip(got.x_raw.iter()) {
                assert!(max_diff_f64(ra, rb) < 1e-5);
            }
            assert!(max_diff_f64(&exp_fixed.y_raw, &got.y_raw) < 1e-5);
        }
    }

    for ver in [1, 2] {
        let dml2 =
            std::fs::read(fixtures_dir.join(format!("compat_v{ver}.dml2"))).expect("read dml2");
        let env = Dml3Envelope::from_bytes(&dml2).expect("decode dml2");
        assert_eq!(env.wire_format, "DML2");
    }
}
