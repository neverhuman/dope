//! End-to-end: intake → fit → safetensors → decode → sample → retention.

use dope_core::eval::{evaluate, EvalOptions};
use dope_core::rng::Rng;
use dope_core::{decode, fit, sample, FitOptions, RawTable, SampleOptions};

/// A mixed-type table: two numeric drivers, a categorical driver, a skewed
/// count, missing cells, and a nonlinear target.
fn table(rows: usize, seed: u64, classify: bool) -> RawTable {
    let mut rng = Rng::new(seed);
    let mut csv = String::from("x1,x2,colour,visits,y\n");
    for _ in 0..rows {
        let x1 = rng.normal();
        let x2 = 0.6 * x1 + 0.8 * rng.normal();
        let colour = ["red", "green", "blue"][rng.below(3)];
        let effect = match colour {
            "red" => 1.0,
            "green" => 0.0,
            _ => -1.0,
        };
        let visits = (rng.normal() * 1.2).exp().round();
        let signal = (1.5 * x1).sin() + 0.5 * x2 * x2 + effect + 0.1 * visits;
        let y = if classify {
            if signal + 0.5 * rng.normal() > 0.5 {
                "yes".to_string()
            } else {
                "no".to_string()
            }
        } else {
            format!("{:.4}", signal + 0.3 * rng.normal())
        };
        let x2s = if rng.uniform() < 0.05 {
            String::new()
        } else {
            format!("{x2:.4}")
        };
        csv += &format!("{x1:.4},{x2s},{colour},{visits},{y}\n");
    }
    RawTable::from_reader(csv.as_bytes()).unwrap()
}

#[test]
fn regression_roundtrip_respects_budget_and_retains_signal() {
    let train = table(3000, 1, false);
    let test = table(1000, 2, false);
    let opts = FitOptions {
        budget: Some(3072),
        max_steps: 400,
        ..Default::default()
    };
    let fitted = fit(&train, 4, &opts).unwrap();
    assert!(
        fitted.bytes.len() <= 3072,
        "artifact {} bytes",
        fitted.bytes.len()
    );
    // The file is a valid safetensors archive and decodes to the evaluated program.
    let program = decode(&fitted.bytes).unwrap();
    assert_eq!(program, fitted.program);
    // Sampling is deterministic, and chunking changes values only by float
    // rounding (GEMM blocking), never which random inputs a row receives.
    let a = sample(
        &program,
        &SampleOptions {
            rows: 500,
            seed: 9,
            chunk: 500,
            ..Default::default()
        },
    )
    .unwrap();
    let b = sample(
        &program,
        &SampleOptions {
            rows: 500,
            seed: 9,
            chunk: 64,
            ..Default::default()
        },
    )
    .unwrap();
    let again = sample(
        &program,
        &SampleOptions {
            rows: 500,
            seed: 9,
            chunk: 500,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(a.y, again.y);
    let close = |u: &f64, v: &f64| (u - v).abs() < 1e-5 || (u.is_nan() && v.is_nan());
    assert!(a.y.iter().zip(&b.y).all(|(u, v)| close(u, v)));
    for j in 0..program.features() {
        assert!(
            a.x[j].iter().zip(&b.x[j]).all(|(u, v)| close(u, v)),
            "feature {j} differs across chunkings"
        );
    }
    // Missing cells are reproduced at roughly the observed rate.
    let miss = a.x[1].iter().filter(|v| v.is_nan()).count() as f64 / 500.0;
    assert!((0.01..0.12).contains(&miss), "missing rate {miss}");
    let opts = EvalOptions {
        auditors: vec!["linear".into(), "gbdt".into()],
        factor: 1.0,
        seed: 0,
    };
    let r = evaluate(&fitted.bytes, &train, &test, None, &opts).unwrap();
    for a in &r.auditors {
        assert!(
            a.informative && a.retention > 0.7,
            "{} retention {}",
            a.auditor,
            a.retention
        );
    }
}

#[test]
fn classification_samples_valid_labels_and_categories() {
    let train = table(2000, 3, true);
    let fitted = fit(
        &train,
        4,
        &FitOptions {
            max_steps: 300,
            ..Default::default()
        },
    )
    .unwrap();
    let program = decode(&fitted.bytes).unwrap();
    let s = sample(
        &program,
        &SampleOptions {
            rows: 400,
            seed: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let mut out = Vec::new();
    dope_core::output::write_raw(&program, &s, &mut out).unwrap();
    let synth = RawTable::from_reader(out.as_slice()).unwrap();
    assert_eq!(synth.headers, vec!["x1", "x2", "colour", "visits", "y"]);
    assert!(synth.columns[2]
        .iter()
        .all(|c| ["red", "green", "blue"].contains(&c.as_str())));
    let yes = synth.columns[4].iter().filter(|c| *c == "yes").count();
    assert!(yes > 40 && yes < 360, "label balance {yes}/400");
    assert!(synth.columns[3].iter().filter(|c| !c.is_empty()).all(|c| c
        .parse::<f64>()
        .unwrap()
        .fract()
        == 0.0));
}

#[test]
fn anonymized_artifacts_carry_no_names_or_labels() {
    let train = table(600, 5, true);
    let fitted = fit(
        &train,
        4,
        &FitOptions {
            anonymize: true,
            max_steps: 100,
            ..Default::default()
        },
    )
    .unwrap();
    let text = String::from_utf8_lossy(&fitted.bytes);
    for secret in ["colour", "red", "green", "visits", "yes"] {
        assert!(!text.contains(secret), "artifact leaks `{secret}`");
    }
}

#[test]
fn ml_space_is_one_hot_and_unit_scaled() {
    let train = table(800, 7, false);
    let fitted = fit(
        &train,
        4,
        &FitOptions {
            max_steps: 100,
            ..Default::default()
        },
    )
    .unwrap();
    let program = decode(&fitted.bytes).unwrap();
    let enc = dope_core::MlEncoder::new(&program, dope_core::Scaling::Unit);
    let headers = enc.headers();
    assert!(
        headers.contains(&"colour=red".to_string())
            && headers.contains(&"x2=__missing__".to_string())
    );
    let s = sample(
        &program,
        &SampleOptions {
            rows: 200,
            seed: 3,
            ..Default::default()
        },
    )
    .unwrap();
    for row in enc.rows(&s.x, &s.y) {
        let features = &row[..row.len() - 1];
        assert!(features.iter().all(|v| (0.0..=1.0).contains(v)));
        let hot: f64 = headers
            .iter()
            .zip(features)
            .filter(|(h, _)| h.starts_with("colour="))
            .map(|(_, v)| v)
            .sum();
        assert_eq!(hot, 1.0);
    }
}

#[test]
fn native_and_tensor_samplers_agree() {
    use dope_core::sample::sample_device;
    let mut forests = 0;
    for (seed, classify) in [(11, false), (12, true)] {
        let train = table(1500, seed, classify);
        let fitted = fit(
            &train,
            4,
            &FitOptions {
                max_steps: 200,
                budget: Some(8192),
                ..Default::default()
            },
        )
        .unwrap();
        let program = decode(&fitted.bytes).unwrap();
        forests += usize::from(program.forest.trees() > 0);
        let opts = SampleOptions {
            rows: 400,
            seed: 5,
            ..Default::default()
        };
        let native = sample(&program, &opts).unwrap();
        let tensor = sample_device(&program, &opts).unwrap();
        let close = |u: &f64, v: &f64| (u - v).abs() < 1e-3 || (u.is_nan() && v.is_nan());
        for j in 0..program.features() {
            let agree = native.x[j]
                .iter()
                .zip(&tensor.x[j])
                .filter(|(u, v)| close(u, v))
                .count();
            assert!(agree >= 398, "feature {j}: {agree}/400 rows agree");
        }
        let agree = native
            .y
            .iter()
            .zip(&tensor.y)
            .filter(|(u, v)| close(u, v))
            .count();
        assert!(agree >= 396, "target: {agree}/400 rows agree");
    }
    assert!(forests > 0, "no program exercised the forest path");
}

#[test]
fn trees_buy_held_out_gain_within_the_budget() {
    let train = table(4000, 21, false);
    let base = FitOptions {
        budget: Some(8192),
        max_steps: 150,
        audit: false,
        ..Default::default()
    };
    let with = fit(&train, 4, &base).unwrap();
    let without = fit(
        &train,
        4,
        &FitOptions {
            trees: false,
            ..base.clone()
        },
    )
    .unwrap();
    assert!(with.bytes.len() <= 8192 && without.bytes.len() <= 8192);
    assert_eq!(without.program.forest.trees(), 0);
    assert!(with.report.trees > 0, "no trees were kept");
    assert!(
        with.report.stored_gain >= without.report.stored_gain,
        "{} < {}",
        with.report.stored_gain,
        without.report.stored_gain
    );
    // The forest survives the safetensors round trip exactly.
    assert_eq!(decode(&with.bytes).unwrap(), with.program);
}

#[test]
fn artifact_decoding_refuses_malformed_input() {
    let train = table(1500, 31, false);
    let fitted = fit(
        &train,
        4,
        &FitOptions {
            max_steps: 100,
            budget: Some(8192),
            audit: false,
            ..Default::default()
        },
    )
    .unwrap();
    let bytes = &fitted.bytes;
    // Garbage, truncation and a corrupted header length are refused, never panicked on.
    assert!(decode(b"not a safetensors file").is_err());
    assert!(decode(&bytes[..bytes.len() / 2]).is_err());
    let mut bad = bytes.clone();
    bad[0] ^= 0x40;
    assert!(decode(&bad).is_err());
    // A forest entry that disagrees with its blob is refused (other tree count, same sections).
    assert!(fitted.program.forest.trees() > 0, "fixture has no forest");
    let parts = dope_core::artifact::unpack(bytes).unwrap();
    let mut meta = parts.meta.clone();
    let trees = meta["forest"][0].as_u64().unwrap();
    meta["forest"][0] = serde_json::json!(trees + 1);
    let tampered = dope_core::artifact::container::repack(&meta, &parts.sections).unwrap();
    assert!(decode(&tampered).is_err());
    // The untouched parts re-pack to an artifact that decodes to the same program.
    let same = dope_core::artifact::container::repack(&parts.meta, &parts.sections).unwrap();
    assert_eq!(decode(&same).unwrap(), fitted.program);
    // Another format version is refused.
    let mut later = parts.meta.clone();
    later["v"] = serde_json::json!(99);
    let later = dope_core::artifact::container::repack(&later, &parts.sections).unwrap();
    assert!(decode(&later).is_err());
    // Errors carry a repair hint and a documentation pointer.
    let err = decode(b"xx").unwrap_err();
    assert!(!err.repair_hint().is_empty() && err.docs_url().starts_with("engine/"));
}
