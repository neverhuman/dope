use dope_core::{
    landmark::{
        ExecutionPlan, LandmarkBundle, LandmarkMode, LandmarkOptions, LandmarkScope,
        ProgressiveLandmarks,
    },
    mlc_v3::MlcV3Model as CoreModel,
    prepare::{
        build_conditioning, build_conditioning_at_level, prepare_table_with_policy,
        ClassVocabPolicy, Level as CoreLevel, PrepTask,
    },
};
use dope_sdk::v0::encode_prepared;
use dope_sdk::v0::{
    landmark::{
        ExecutionPlan as SdkPlan, LandmarkScope as SdkScope, ProgressiveLandmarks as SdkProgressive,
    },
    mlc_v3::MlcV3Model as SdkModel,
    prepare::{
        build_conditioning as sdk_condition, build_conditioning_at_level as sdk_condition_at_level,
        prepare_table_with_policy as sdk_prepare, ClassVocabPolicy as SdkVocab, Level as SdkLevel,
        PrepTask as SdkTask,
    },
};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;

fn fixture(task: PrepTask) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..48)
        .map(|i| {
            let a = ((i * 17 % 53) as f64) / 53.0;
            vec![a, (i as f64 * 0.23).sin(), (i % 7) as f64]
        })
        .collect();
    let y = x
        .iter()
        .enumerate()
        .map(|(i, r)| match task {
            PrepTask::Regression => 1.7 * r[0] - 0.4 * r[1] + (i as f64 * 0.13).cos(),
            PrepTask::Classification => (i % 2) as f64,
            PrepTask::Multiclass => (i % 3) as f64,
        })
        .collect();
    (x, y)
}

// Same deterministic, nonzero 56-tensor source fixture used by the core loader test.
fn synthetic_checkpoint() -> Vec<u8> {
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
    let buffers: Vec<_> = specs
        .iter()
        .enumerate()
        .map(|(idx, &(name, shape))| {
            let mut bytes = Vec::new();
            let norm = name.ends_with("norm.weight")
                || name.ends_with(".2.weight")
                || name.ends_with(".5.weight");
            for k in 0..shape.iter().product::<usize>() {
                let value = if norm {
                    1.0f32
                } else {
                    (((idx * 131 + k * 17) % 97) as f32 - 48.0) * 0.001
                };
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            (name, shape.to_vec(), bytes)
        })
        .collect();
    let views: BTreeMap<_, _> = buffers
        .iter()
        .map(|(name, shape, bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::tensor::serialize(&views, None).unwrap()
}

fn bits<T: Copy>(values: &[T], map: impl Fn(T) -> u32) -> Vec<u32> {
    values.iter().copied().map(map).collect()
}
fn cols(values: &[[f32; 17]]) -> Vec<Vec<f32>> {
    values.iter().map(|r| r.to_vec()).collect()
}
fn col_bits(values: &[[f32; 17]]) -> Vec<[u32; 17]> {
    values.iter().map(|r| r.map(f32::to_bits)).collect()
}
fn options() -> LandmarkOptions {
    LandmarkOptions {
        folds: 2,
        ridge_rows: 40,
        small_rows: 40,
        knn_reference: 8,
        spectrum_features: 2,
        seed: 17,
        work_budget: u64::MAX,
        ..Default::default()
    }
}

fn assert_safe_refusal(
    model: &SdkModel,
    value: &dope_core::prepare::DopeConditioning,
    recipe: &str,
    message: &str,
    canary: &str,
) {
    let error = encode_prepared(model, value, recipe)
        .unwrap_err()
        .to_string();
    assert_eq!(error, message);
    assert!(!error.contains(canary));
}

fn assert_checked_refusals(model: &SdkModel, valid: &dope_core::prepare::DopeConditioning) {
    let expected = dope_sdk::v0::prepare::FULL_REPORT_RECIPE;
    let canary = "sdk-private-canary-74a1";
    let mut wrong_known_recipe = valid.clone();
    wrong_known_recipe.recipe = dope_sdk::v0::prepare::OBSERVED_RECIPE.into();
    wrong_known_recipe.vec256[0] = f32::NAN;
    assert_safe_refusal(
        model,
        &wrong_known_recipe,
        expected,
        "input: conditioning recipe mismatch",
        canary,
    );

    let mut unknown_recipe = valid.clone();
    unknown_recipe.recipe = canary.into();
    unknown_recipe.vec256[0] = f32::NAN;
    assert_safe_refusal(
        model,
        &unknown_recipe,
        expected,
        "input: unknown conditioning recipe",
        canary,
    );

    let mut invalid_level = valid.clone();
    invalid_level.level = canary.into();
    invalid_level.vec256[0] = f32::NAN;
    assert_safe_refusal(
        model,
        &invalid_level,
        expected,
        "input: unsupported conditioning level",
        canary,
    );

    let mut unknown_expected = valid.clone();
    unknown_expected.vec256[0] = f32::NAN;
    assert_safe_refusal(
        model,
        &unknown_expected,
        canary,
        "input: unknown expected conditioning recipe",
        canary,
    );

    let mut malformed_shape = valid.clone();
    malformed_shape.col_landmarks.truncate(63);
    assert_safe_refusal(
        model,
        &malformed_shape,
        expected,
        "artifact: col_landmarks requires shape [64, 17]",
        canary,
    );
}

#[test]
fn direct_core_and_sdk_match_prepared_views_conditioning_and_nonzero_encoder() {
    let model_bytes = synthetic_checkpoint();
    let core_model = CoreModel::from_safetensors_bytes(&model_bytes).unwrap();
    let sdk_model = SdkModel::from_safetensors_bytes(&model_bytes).unwrap();
    assert_eq!(core_model.inventory.sha256, sdk_model.inventory.sha256);
    assert_eq!(core_model.inventory.parameter_count, 444_168);

    for task in [
        PrepTask::Regression,
        PrepTask::Classification,
        PrepTask::Multiclass,
    ] {
        let policy = if task == PrepTask::Multiclass {
            ClassVocabPolicy::LosslessV1
        } else {
            ClassVocabPolicy::Cap32V1
        };
        let (x, y) = fixture(task);
        let ids: Vec<u64> = (0..x.len()).map(|i| 10_003 + 7 * i as u64).collect();
        let core = prepare_table_with_policy(&x, &y, task, ids.clone(), policy).unwrap();
        let sdk = sdk_prepare(&x, &y, task, ids.clone(), policy).unwrap();
        assert_eq!(core.row_ids, ids);
        assert_eq!(
            core.x_scaled
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            sdk.x_scaled.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            core.y_scaled
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            sdk.y_scaled.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(core.source_rows_digest(), sdk.source_rows_digest());
        assert_eq!(
            core.computed_source_rows_digest(),
            sdk.computed_source_rows_digest()
        );
        assert_eq!(core.classes, sdk.classes);
        assert_eq!(core.task.as_str(), sdk.task.as_str());
        assert_eq!(
            core.transform.class_vocab_policy,
            sdk.transform.class_vocab_policy
        );

        let opts = options();
        let frame = core.landmark_frame().unwrap();
        let sdk_frame = sdk.landmark_frame().unwrap();
        let scope = LandmarkScope::for_source_digest(&opts, None, core.source_rows_digest());
        let mut full = ProgressiveLandmarks::new_with_scope_and_plan(
            &frame,
            &opts,
            scope.clone(),
            ExecutionPlan::FullPanelV1,
        )
        .unwrap();
        let mut high = ProgressiveLandmarks::new_with_scope_and_plan(
            &frame,
            &opts,
            scope,
            ExecutionPlan::HighRequiredV1,
        )
        .unwrap();
        let sdk_opts = opts.clone();
        let sdk_scope = SdkScope::for_source_digest(&sdk_opts, None, sdk.source_rows_digest());
        let mut sdk_full = SdkProgressive::new_with_scope_and_plan(
            &sdk_frame,
            &sdk_opts,
            sdk_scope.clone(),
            SdkPlan::FullPanelV1,
        )
        .unwrap();
        let mut sdk_high = SdkProgressive::new_with_scope_and_plan(
            &sdk_frame,
            &sdk_opts,
            sdk_scope,
            SdkPlan::HighRequiredV1,
        )
        .unwrap();
        full.run_required().unwrap();
        high.run_required().unwrap();
        sdk_full.run_required().unwrap();
        sdk_high.run_required().unwrap();
        assert_eq!(full.scope(), sdk_full.scope());
        assert_eq!(high.scope(), sdk_high.scope());
        assert_eq!(
            serde_json::to_vec(&full.report().private_report()).unwrap(),
            serde_json::to_vec(&sdk_full.report().private_report()).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&high.report().private_report()).unwrap(),
            serde_json::to_vec(&sdk_high.report().private_report()).unwrap()
        );
        assert_eq!(
            full.bundle() == high.bundle(),
            full.bundle().conditioning_digest() == high.bundle().conditioning_digest()
        );
        assert_ne!(
            serde_json::to_vec(&full.report().private_report()).unwrap(),
            serde_json::to_vec(&high.report().private_report()).unwrap(),
            "different execution plans retain distinct private report states"
        );
        let full_report = dope_core::prepare::NativeReportView::from_signature(full.report());
        let high_report = dope_core::prepare::NativeReportView::from_signature(high.report());
        let sdk_full_report =
            dope_sdk::v0::prepare::NativeReportView::from_signature(sdk_full.report());
        let sdk_high_report =
            dope_sdk::v0::prepare::NativeReportView::from_signature(sdk_high.report());
        let cases = [
            (
                LandmarkMode::None,
                LandmarkBundle::zero(),
                LandmarkBundle::zero(),
                None,
                None,
            ),
            (
                LandmarkMode::Full,
                full.bundle().clone(),
                sdk_full.bundle().clone(),
                Some(&full_report),
                Some(&sdk_full_report),
            ),
            (
                LandmarkMode::Full,
                high.bundle().clone(),
                sdk_high.bundle().clone(),
                Some(&high_report),
                Some(&sdk_high_report),
            ),
        ];
        let mut high_condition_pair = None;
        for (mode, core_bundle, sdk_bundle, report, sdk_report) in cases {
            let (a, b) = if mode == LandmarkMode::None {
                (
                    build_conditioning(&core, &core_bundle, report, mode).unwrap(),
                    sdk_condition(&sdk, &sdk_bundle, sdk_report, mode).unwrap(),
                )
            } else {
                (
                    build_conditioning_at_level(
                        &core,
                        &core_bundle,
                        report,
                        mode,
                        CoreLevel::High,
                        None,
                    )
                    .unwrap(),
                    sdk_condition_at_level(
                        &sdk,
                        &sdk_bundle,
                        sdk_report,
                        mode,
                        SdkLevel::High,
                        None,
                    )
                    .unwrap(),
                )
            };
            assert_eq!(
                bits(&a.vec256, f32::to_bits),
                bits(&b.vec256, f32::to_bits),
                "{task:?}/{mode:?}"
            );
            assert_eq!(col_bits(&a.col_landmarks), col_bits(&b.col_landmarks));
            assert_eq!(a.digest(), b.digest());
            assert_eq!(a.bundle, core_bundle);
            assert_eq!(b.bundle, sdk_bundle);
            let core_columns = cols(&a.col_landmarks);
            let core_output = core_model
                .encode_features_629(&a.vec256, &core_columns, "high")
                .unwrap();
            let expected_recipe = if mode == LandmarkMode::None {
                dope_sdk::v0::prepare::OBSERVED_RECIPE
            } else {
                dope_sdk::v0::prepare::FULL_REPORT_RECIPE
            };
            let checked_output = encode_prepared(&sdk_model, &b, expected_recipe).unwrap();
            assert_eq!(
                bits(&core_output, f32::to_bits),
                bits(&checked_output, f32::to_bits)
            );
            if mode == LandmarkMode::Full
                && report.is_some_and(|r| {
                    r.cells
                        .iter()
                        .filter(|c| c.state == "not_requested")
                        .count()
                        == 5
                })
            {
                assert_checked_refusals(&sdk_model, &b);
            }
            if report.is_some_and(|r| {
                r.cells
                    .iter()
                    .filter(|c| c.state == "not_requested")
                    .count()
                    == 5
            }) {
                high_condition_pair = Some((a, b, core_output, checked_output));
            }
        }
        let (core_high_condition, sdk_high_condition, core_vec, sdk_vec) =
            high_condition_pair.expect("HighRequiredV1 conditioning");
        let core_lm = cols(&core_high_condition.col_landmarks);
        let sdk_lm = cols(&sdk_high_condition.col_landmarks);
        assert!(core_vec.iter().any(|v| *v != 0.0));
        let mut core_perturbed = core_high_condition.vec256;
        let mut sdk_perturbed = sdk_high_condition.vec256;
        for (i, value) in core_perturbed[..140].iter_mut().enumerate() {
            *value += (i as f32 + 1.0) * 0.01;
        }
        for (i, value) in sdk_perturbed[..140].iter_mut().enumerate() {
            *value += (i as f32 + 1.0) * 0.01;
        }
        let core_changed = core_model
            .encode_features_629(&core_perturbed, &core_lm, "high")
            .unwrap();
        let sdk_changed = sdk_model
            .encode_features_629(&sdk_perturbed, &sdk_lm, "high")
            .unwrap();
        assert_eq!(
            bits(&core_changed, f32::to_bits),
            bits(&sdk_changed, f32::to_bits)
        );
        assert_ne!(
            bits(&core_vec[256..512], f32::to_bits),
            bits(&core_changed[256..512], f32::to_bits),
            "conditioning perturbation must change actual routed latents"
        );
    }
}

#[test]
fn sdk_refuses_malformed_weights_and_unsupported_preparation_without_canary_echo() {
    let canary = "sdk-private-canary-74a1";
    let x = vec![vec![canary.to_owned()], vec!["1".to_owned()]];
    let errors = [
        SdkModel::from_safetensors_bytes(canary.as_bytes())
            .err()
            .expect("malformed weights accepted"),
        dope_sdk::v0::prepare::prepare_table_mixed_with_policy(
            &x,
            &[0.0, 1.0],
            Some(&[]),
            SdkTask::Classification,
            Some(vec![10, 30]),
            SdkVocab::Cap32V1,
        )
        .unwrap_err(),
        dope_sdk::v0::prepare::prepare_table_with_ids(
            &[vec![1.0], vec![2.0]],
            &[0.0],
            PrepTask::Regression,
            vec![10, 30],
        )
        .unwrap_err(),
    ];
    for error in errors {
        assert!(!error.to_string().contains(canary));
    }
}
