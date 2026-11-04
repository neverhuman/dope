use dope_core::landmark::bundle::FINE_SLOTS;
use dope_core::{
    landmark::{
        CellState, ExecutionPlan, LandmarkBundle, LandmarkGroup, LandmarkMode, LandmarkOptions,
        LandmarkScope, ProgressiveLandmarks, SubsampleSpec,
    },
    mlc_v3::{MlcV3Model, MLC_V3_EXPECTED_PARAMETERS},
    prepare::{
        build_conditioning, build_conditioning_at_level, prepare_table_with_policy,
        ClassVocabPolicy, Level, NativeReportView, PrepTask, PreparedTable,
    },
};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;

fn data(task: PrepTask) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..96)
        .map(|i| {
            let a = ((i * 17 % 101) as f64) / 101.0;
            vec![a, (i as f64 * 0.17).sin(), ((i * 7 % 29) as f64) / 29.0]
        })
        .collect();
    let y = x
        .iter()
        .enumerate()
        .map(|(i, r)| match task {
            PrepTask::Regression => 1.4 * r[0] - 0.7 * r[1] + (i as f64 * 0.11).cos(),
            PrepTask::Classification => (i % 2) as f64,
            PrepTask::Multiclass => (i % 3) as f64,
        })
        .collect();
    (x, y)
}

fn prepared(task: PrepTask) -> PreparedTable {
    let (x, y) = data(task);
    let ids = (0..x.len()).map(|i| 1_000_009 + 11 * i as u64).collect();
    let policy = if task == PrepTask::Multiclass {
        ClassVocabPolicy::LosslessV1
    } else {
        ClassVocabPolicy::Cap32V1
    };
    prepare_table_with_policy(&x, &y, task, ids, policy).unwrap()
}

fn options(budget: u64) -> LandmarkOptions {
    LandmarkOptions {
        folds: 2,
        ridge_rows: 40,
        small_rows: 40,
        knn_reference: 8,
        spectrum_features: 2,
        seed: 17,
        work_budget: budget,
        ..Default::default()
    }
}

// Same deterministic, nonzero 56-tensor source fixture used by the existing loader test.
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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn columns(values: &[[f32; 17]]) -> Vec<Vec<f32>> {
    values.iter().map(|r| r.to_vec()).collect()
}
fn report_view(p: &ProgressiveLandmarks) -> NativeReportView {
    NativeReportView::from_signature(p.report())
}
fn session(
    table: &PreparedTable,
    opts: &LandmarkOptions,
    plan: ExecutionPlan,
    sample: Option<SubsampleSpec>,
) -> ProgressiveLandmarks {
    let frame = table.landmark_frame().unwrap();
    let scope = LandmarkScope::for_source_digest(opts, sample, table.source_rows_digest());
    ProgressiveLandmarks::new_with_scope_and_plan(&frame, opts, scope, plan).unwrap()
}

fn assert_high_equal(
    table: &PreparedTable,
    full: &ProgressiveLandmarks,
    high: &ProgressiveLandmarks,
    model: &MlcV3Model,
) {
    let full_cond = build_conditioning_at_level(
        table,
        full.bundle(),
        Some(&report_view(full)),
        LandmarkMode::Full,
        Level::High,
        None,
    )
    .unwrap();
    let high_cond = build_conditioning_at_level(
        table,
        high.bundle(),
        Some(&report_view(high)),
        LandmarkMode::Full,
        Level::High,
        None,
    )
    .unwrap();
    assert_eq!(bits(&full_cond.vec256), bits(&high_cond.vec256));
    assert_eq!(
        full_cond
            .col_landmarks
            .iter()
            .map(|r| r.map(f32::to_bits))
            .collect::<Vec<_>>(),
        high_cond
            .col_landmarks
            .iter()
            .map(|r| r.map(f32::to_bits))
            .collect::<Vec<_>>()
    );
    assert_eq!(full_cond.digest(), high_cond.digest());
    assert_eq!(
        full.bundle() == high.bundle(),
        full.bundle().conditioning_digest() == high.bundle().conditioning_digest()
    );
    assert_ne!(
        serde_json::to_vec(&full.report().private_report()).unwrap(),
        serde_json::to_vec(&high.report().private_report()).unwrap(),
        "different execution plans retain distinct private report states"
    );
    let full_lm = columns(&full_cond.col_landmarks);
    let high_lm = columns(&high_cond.col_landmarks);
    let full_vec = model
        .encode_features_629(&full_cond.vec256, &full_lm, "high")
        .unwrap();
    let high_vec = model
        .encode_features_629(&high_cond.vec256, &high_lm, "high")
        .unwrap();
    assert_eq!(bits(&full_vec), bits(&high_vec));
    assert!(full_vec.iter().any(|v| *v != 0.0));
    let mut changed_input = high_cond.vec256;
    for (i, value) in changed_input[..140].iter_mut().enumerate() {
        *value += (i as f32 + 1.0) * 0.01;
    }
    let changed = model
        .encode_features_629(&changed_input, &high_lm, "high")
        .unwrap();
    assert_ne!(
        bits(&high_vec[256..512]),
        bits(&changed[256..512]),
        "perturbed conditioning changes actual routed encoder latents"
    );
}

#[test]
fn full_high_none_policy_matrix_uses_real_loader_and_forward_for_all_tasks() {
    let model = MlcV3Model::from_safetensors_bytes(&synthetic_checkpoint()).unwrap();
    assert_eq!(model.inventory.parameter_count, MLC_V3_EXPECTED_PARAMETERS);
    for task in [
        PrepTask::Regression,
        PrepTask::Classification,
        PrepTask::Multiclass,
    ] {
        let table = prepared(task);
        let opts = options(u64::MAX);
        let mut full = session(&table, &opts, ExecutionPlan::FullPanelV1, None);
        let mut high = session(&table, &opts, ExecutionPlan::HighRequiredV1, None);
        full.run_required().unwrap();
        high.run_required().unwrap();
        assert_eq!(full.scope(), high.scope());
        assert_eq!(
            full.execution_work().reserved_panel_units,
            high.execution_work().reserved_panel_units
        );
        assert_eq!(high.execution_work().computed_cells, 18);
        assert!(high.is_execution_complete() && !high.is_complete());
        for slot in FINE_SLOTS {
            assert_eq!(high.report().cells[slot].state, CellState::NotRequested);
            assert!(high.report().cells[slot].gain.is_none());
            assert!(!high.bundle().mask[slot]);
        }
        assert_high_equal(&table, &full, &high, &model);

        let none =
            build_conditioning(&table, &LandmarkBundle::zero(), None, LandmarkMode::None).unwrap();
        assert_eq!(none.mode, "none");
        assert!(none.bundle.is_zero());
        let high_work_before_upgrade = high.execution_work();
        high.upgrade_to_full_panel();
        assert!(!high.is_execution_complete());
        high.next_group(LandmarkGroup::Fine).unwrap();
        assert!(high.is_complete() && high.is_execution_complete());
        assert_eq!(high.execution_work().computed_cells, 23);
        assert_eq!(
            high.execution_work().reserved_panel_units,
            high_work_before_upgrade.reserved_panel_units
        );
        assert_eq!(
            serde_json::to_vec(&high.report().private_report()).unwrap(),
            serde_json::to_vec(&full.report().private_report()).unwrap()
        );
        assert_eq!(high.bundle(), full.bundle());
    }
}

#[test]
fn subsampled_tight_budget_keeps_full_reservation_and_high_admission() {
    let table = prepared(PrepTask::Regression);
    let opts = options(500_000);
    let sample = Some(SubsampleSpec::new(32, 7));
    let mut full = session(&table, &opts, ExecutionPlan::FullPanelV1, sample);
    let mut high = session(&table, &opts, ExecutionPlan::HighRequiredV1, sample);
    full.run_required().unwrap();
    high.run_required().unwrap();
    assert_eq!(
        full.selection().unwrap().row_ids,
        high.selection().unwrap().row_ids
    );
    assert_eq!(
        full.execution_work().reserved_panel_units,
        high.execution_work().reserved_panel_units
    );
    assert_eq!(full.report().work.dropped, high.report().work.dropped);
    assert_high_equal(
        &table,
        &full,
        &high,
        &MlcV3Model::from_safetensors_bytes(&synthetic_checkpoint()).unwrap(),
    );
}
