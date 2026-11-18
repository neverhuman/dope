use super::build_deterministic_safetensors_bytes;
use candle_core::{Device, Tensor};
use dope_core::mlc_v3::{
    packet_features_v1_from_encoding, packet_global_v1_from_encoding, MlcV3Encoding, MlcV3Model,
    ObliviousTree, ResearchConditioningLevel, TargetBundle, VectorLeafTree, MLC_V3_GATE_KEYS_14,
    PACKET_FEATURES_V1_DIM, PACKET_GLOBAL_V1_DIM, PACKET_GLOBAL_V1_FIELDS,
    RESEARCH_ENCODER_VIEW_V1_RECIPE,
};
use std::collections::{BTreeMap, BTreeSet};

fn sample_encoding(task: &str, y_min: f64, y_max: f64) -> MlcV3Encoding {
    let mut gates = BTreeMap::new();
    for (i, k) in MLC_V3_GATE_KEYS_14.iter().enumerate() {
        gates.insert((*k).to_string(), 0.05 * (i as f64 + 1.0));
    }
    let mut disc = BTreeMap::new();
    disc.insert(
        2usize,
        (vec![-1.0, 0.0, 1.0, 2.0], vec![0.25, 0.5, 0.75, 1.0]),
    );
    MlcV3Encoding {
        level: "high".to_string(),
        task: task.to_string(),
        code_width: 8192,
        n_train: 200,
        p_use: 4,
        gates: gates.clone(),
        task_latent: (0..128).map(|i| ((i as f32) - 64.0) * 0.01).collect(),
        dist_latent: (0..128).map(|i| ((i as f32) - 32.0) * 0.005).collect(),
        knot_grid: vec![0.01, 0.25, 0.5, 0.75, 0.99],
        col_knots: vec![
            vec![-2.0, -0.5, 0.0, 0.6, 2.2],
            vec![-1.5, -0.4, 0.2, 0.9, 1.8],
            vec![-1.0, 0.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        is_binary_col: vec![false, false, false, true],
        binary_probs: vec![0.0, 0.0, 0.0, 0.3],
        binary_low_high: vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        discrete_support: disc,
        chol_factor: vec![
            vec![0.8, 0.1],
            vec![0.3, 0.7],
            vec![0.5, -0.2],
            vec![0.2, 0.4],
        ],
        diag_res_std: vec![0.59, 0.65, 0.84, 0.89],
        regime_weights: vec![0.2, 0.5, 0.3],
        regime_means: vec![
            vec![-0.4, 0.1, 0.0, -0.2],
            vec![0.0, -0.1, 0.2, 0.0],
            vec![0.5, 0.2, -0.3, 0.3],
        ],
        regime_scales: vec![
            vec![0.9, 1.0, 1.1, 0.8],
            vec![1.0, 1.0, 0.9, 1.1],
            vec![1.2, 0.8, 1.0, 1.0],
        ],
        target_bundle: TargetBundle {
            task_kind: task.to_string(),
            gates,
            pairs: vec![(0, 1), (1, 3)],
            include_v3_primitives: true,
            w_basis: vec![
                0.1, // bias
                0.8, -0.3, 0.2, 0.05, // lin (4)
                0.4, 0.1, -0.2, 0.0, // quad (4)
                0.15, 0.05, 0.1, 0.02, // abs (4)
                0.35, -0.25, 0.12, // tail
            ],
            res_net_mean: 0.12,
            neural_w: 0.45,
            trees: vec![
                ObliviousTree {
                    features: vec![0, 1, 3],
                    thresholds: vec![0.1, -0.2, 0.5],
                    leaf_values: vec![-0.4, -0.1, 0.1, 0.3, -0.2, 0.0, 0.2, 0.5],
                },
                ObliviousTree {
                    features: vec![1, 2, 0],
                    thresholds: vec![0.0, 0.4, -0.3],
                    leaf_values: vec![-0.1, 0.05, -0.05, 0.1, 0.0, 0.08, -0.08, 0.12],
                },
            ],
            residual_std: 0.42,
            noise_scale: 0.35,
            class_prior: 0.35,
            y_min,
            y_max,
            k_classes: if task == "multiclass" { 3 } else { 1 },
            class_priors: if task == "multiclass" {
                vec![0.1, 0.7, 0.2]
            } else {
                Vec::new()
            },
            u_basis: if task == "multiclass" {
                (0..16)
                    .map(|r| vec![0.1 * ((r % 5) as f64), 0.05 * ((r % 3) as f64)])
                    .collect()
            } else {
                Vec::new()
            },
            v_basis: Vec::new(),
            mc_trees: if task == "multiclass" {
                vec![VectorLeafTree {
                    features: vec![0, 2],
                    thresholds: vec![0.1, -0.2],
                    leaf_values: vec![
                        vec![0.1, -0.1, 0.0],
                        vec![-0.2, 0.2, 0.0],
                        vec![0.0, 0.1, -0.1],
                        vec![0.05, 0.05, -0.1],
                    ],
                }]
            } else {
                Vec::new()
            },
        },
    }
}

#[test]
fn research_view_schema_deduplicates_noise_scale_and_drops_extrema() {
    assert_eq!(RESEARCH_ENCODER_VIEW_V1_RECIPE, "research-encoder-view/v1");
    let unique: BTreeSet<&str> = PACKET_GLOBAL_V1_FIELDS.iter().copied().collect();
    assert_eq!(unique.len(), PACKET_GLOBAL_V1_DIM);
    assert!(unique.contains("noise_scale"));
    assert!(!unique.contains("y_min"));
    assert!(!unique.contains("y_max"));
}

#[test]
fn research_view_629_verifies_level_and_matches_tensor_path() {
    let model =
        MlcV3Model::from_safetensors_bytes(&build_deterministic_safetensors_bytes()).unwrap();
    let v256: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.01).collect();
    let cols: Vec<Vec<f32>> = (0..64)
        .map(|r| {
            (0..17)
                .map(|c| ((r * 17 + c) as f32 % 19.0 - 9.0) * 0.02)
                .collect()
        })
        .collect();

    let direct_high = model.encode_features_629(&v256, &cols, "high").unwrap();
    let view_high = model
        .encode_research_view_629(
            &v256,
            &cols,
            ResearchConditioningLevel::High,
            ResearchConditioningLevel::High,
        )
        .unwrap();
    assert_eq!(direct_high, view_high);

    // Mismatched conditioning level is refused in both directions.
    assert!(model
        .encode_research_view_629(
            &v256,
            &cols,
            ResearchConditioningLevel::High,
            ResearchConditioningLevel::Ultra,
        )
        .is_err());
    assert!(model
        .encode_research_view_629(
            &v256,
            &cols,
            ResearchConditioningLevel::Ultra,
            ResearchConditioningLevel::High,
        )
        .is_err());

    let view_ultra = model
        .encode_research_view_629(
            &v256,
            &cols,
            ResearchConditioningLevel::Ultra,
            ResearchConditioningLevel::Ultra,
        )
        .unwrap();
    assert_eq!(view_ultra.len(), 629);
    assert_ne!(&view_ultra[512..526], &view_high[512..526]);

    let dev = Device::Cpu;
    let t_desc = Tensor::from_slice(&v256, (1, 256), &dev).unwrap();
    let flat_cols: Vec<f32> = cols.iter().flatten().copied().collect();
    let t_cols = Tensor::from_vec(flat_cols, (1, 64, 17), &dev).unwrap();
    let tensor_ultra = model
        .encode_research_view_629_tensor(
            &t_desc,
            &t_cols,
            ResearchConditioningLevel::Ultra,
            ResearchConditioningLevel::Ultra,
        )
        .unwrap()
        .squeeze(0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, b) in view_ultra.iter().zip(&tensor_ultra) {
        assert!((a - b).abs() < 1e-6);
    }

    // Non-finite input is refused on Ultra so LayerNorm cannot silently zero all 373 learned coords.
    let mut bad_v256 = v256.clone();
    bad_v256[0] = f32::NAN;
    assert!(model
        .encode_research_view_629(
            &bad_v256,
            &cols,
            ResearchConditioningLevel::Ultra,
            ResearchConditioningLevel::Ultra,
        )
        .is_err());
}

#[test]
fn packet_features_v1_extrema_and_permutation_invariance() {
    let enc_a = sample_encoding("regression", -2.5, 3.5);
    let enc_b = sample_encoding("regression", -987_654.0, 123_456.0);
    let feat_a = packet_features_v1_from_encoding(&enc_a, true).unwrap();
    let feat_b = packet_features_v1_from_encoding(&enc_b, true).unwrap();
    assert_eq!(feat_a.len(), PACKET_FEATURES_V1_DIM);
    assert_eq!(
        feat_a, feat_b,
        "target extrema must never leak into packet_features_v1"
    );

    let glob_a = packet_global_v1_from_encoding(&enc_a, true).unwrap();
    assert_eq!(&glob_a[..], &feat_a[..PACKET_GLOBAL_V1_DIM]);

    // P-S2 covariate-only ablation zeros target-bundle slots while preserving covariates & latents.
    let feat_notarget = packet_features_v1_from_encoding(&enc_a, false).unwrap();
    assert_eq!(feat_notarget[14], 0.0);
    assert_eq!(feat_notarget[25], 0.0);
    assert_eq!(&feat_notarget[40..44], &[0.0; 4]);
    assert_eq!(&feat_notarget[172..188], &[0.0; 16]);
    assert_eq!(&feat_notarget[188..196], &feat_a[188..196]);

    // Permute columns [0, 1, 2, 3] -> [2, 0, 3, 1].
    let perm = [2usize, 0, 3, 1];
    let mut inv = [0usize; 4];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv[old_idx] = new_idx;
    }
    let mut enc_p = enc_a.clone();
    enc_p.col_knots = perm.iter().map(|&j| enc_a.col_knots[j].clone()).collect();
    enc_p.is_binary_col = perm.iter().map(|&j| enc_a.is_binary_col[j]).collect();
    enc_p.binary_probs = perm.iter().map(|&j| enc_a.binary_probs[j]).collect();
    enc_p.chol_factor = perm.iter().map(|&j| enc_a.chol_factor[j].clone()).collect();
    enc_p.diag_res_std = perm.iter().map(|&j| enc_a.diag_res_std[j]).collect();
    enc_p.regime_means = enc_a
        .regime_means
        .iter()
        .map(|r| perm.iter().map(|&j| r[j]).collect())
        .collect();
    enc_p.regime_scales = enc_a
        .regime_scales
        .iter()
        .map(|r| perm.iter().map(|&j| r[j]).collect())
        .collect();
    enc_p.discrete_support.clear();
    for (&old_j, val) in &enc_a.discrete_support {
        enc_p.discrete_support.insert(inv[old_j], val.clone());
    }
    enc_p.target_bundle.pairs = enc_a
        .target_bundle
        .pairs
        .iter()
        .map(|&(a, b)| (inv[a], inv[b]))
        .collect();
    for (ti, t) in enc_a.target_bundle.trees.iter().enumerate() {
        enc_p.target_bundle.trees[ti].features = t.features.iter().map(|&f| inv[f]).collect();
    }
    for (new_j, &old_j) in perm.iter().enumerate() {
        enc_p.target_bundle.w_basis[1 + new_j] = enc_a.target_bundle.w_basis[1 + old_j];
        enc_p.target_bundle.w_basis[1 + 4 + new_j] = enc_a.target_bundle.w_basis[1 + 4 + old_j];
        enc_p.target_bundle.w_basis[1 + 8 + new_j] = enc_a.target_bundle.w_basis[1 + 8 + old_j];
    }
    let feat_p = packet_features_v1_from_encoding(&enc_p, true).unwrap();
    for (idx, (a, b)) in feat_a.iter().zip(&feat_p).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "column permutation drift at coord {idx}: {a} vs {b}"
        );
    }

    // Multiclass class-label permutation invariance.
    let mc_a = sample_encoding("multiclass", 0.0, 2.0);
    let mut mc_b = mc_a.clone();
    mc_b.target_bundle.class_priors = vec![0.7, 0.2, 0.1];
    let f_mca = packet_global_v1_from_encoding(&mc_a, true).unwrap();
    let f_mcb = packet_global_v1_from_encoding(&mc_b, true).unwrap();
    for (a, b) in f_mca.iter().zip(&f_mcb) {
        assert!((a - b).abs() < 1e-6);
    }
}
