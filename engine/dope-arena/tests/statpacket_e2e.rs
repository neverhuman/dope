//! End-to-end tests of `dope_statpacket` through files only: fit at a 2,048
//! byte budget, sample four times the FIT rows, train a plain ridge or
//! logistic model on the real FIT rows and on the sampled rows, and compare
//! their losses on a disjoint real test set.
//!
//! Retention `R = (L0 - L_S) / (L0 - L_A)`: `L0` is the loss of the best
//! constant, `L_A` of the model trained on the real FIT rows, `L_S` of the
//! model trained on the sampled rows.

mod statpacket_support;

use statpacket_support::eval::{max_corr_gap, null_loss, retention, test_loss, Table};
use statpacket_support::{
    class_shares, dir, fit, fixtures, sample, truncate, Fixture, FIT_ROWS, LEVELS,
};
use std::path::Path;

const BUDGET: usize = 2048;
const SAMPLE_ROWS: usize = 4 * FIT_ROWS;
const SEED: u64 = 20_261_122;
/// Independent samples used for the label-share check.
const LABEL_SAMPLES: usize = 8;

/// Retention of one packet for a fixture.
fn retention_of(f: &Fixture, sampled: &Table) -> f64 {
    let null = null_loss(&f.fit, &f.test, f.classes);
    let real = test_loss(&f.fit, &f.test, f.classes);
    retention(null, real, test_loss(sampled, &f.test, f.classes))
}

/// Largest gap, over classes, between a label share of sampled rows and the
/// FIT share, in standard errors of a share estimated from one sample of
/// `SAMPLE_ROWS` rows. `shares` is one share vector per sample; the gap is
/// taken on their mean.
fn label_gap_in_se(f: &Fixture, shares: &[Vec<f64>]) -> f64 {
    let fit = class_shares(&f.fit, f.classes);
    (0..f.classes)
        .map(|c| {
            let mean = shares.iter().map(|s| s[c]).sum::<f64>() / shares.len() as f64;
            (mean - fit[c]).abs() / (fit[c] * (1.0 - fit[c]) / SAMPLE_ROWS as f64).sqrt()
        })
        .fold(0.0, f64::max)
}

struct LevelRow {
    level: String,
    bytes: usize,
    retention: f64,
    corr_gap: f64,
    /// Label gap of the first sample, of the mean of all samples, and the
    /// number of single samples within 3 SE.
    label_se: f64,
    label_se_mean: f64,
    label_within: usize,
}

/// Fits `f` at the budget, truncates to every level and scores each packet.
fn score_levels(f: &Fixture, d: &Path) -> Vec<LevelRow> {
    let (packet, card) = (
        d.join(format!("{}.dse", f.name)),
        d.join(format!("{}.json", f.name)),
    );
    let report = fit(&f.fit, f.task, f.classes, BUDGET, &packet, &card);
    assert_eq!(report["status"], "ok", "{}", f.name);
    assert_eq!(report["level"], "E", "{}", f.name);
    let top = std::fs::read(&packet).unwrap();
    assert!(top.len() <= BUDGET);
    assert_eq!(report["packet_bytes"], top.len());
    let mut rows = Vec::new();
    for level in LEVELS {
        let cut = d.join(format!("{}.{level}.dse", f.name));
        let info = truncate(&packet, level, &cut);
        let bytes = std::fs::read(&cut).unwrap();
        assert_eq!(info["header"]["level"], level);
        assert_eq!(info["packet_bytes"], bytes.len());
        if level == "E" {
            assert_eq!(bytes, top, "truncating to the same level is the identity");
        }
        let sampled = sample(&cut, &card, SAMPLE_ROWS, 0, SEED, "sampled");
        assert_eq!((sampled.rows(), sampled.p), (SAMPLE_ROWS, f.fit.p));
        assert!(sampled.x.iter().chain(&sampled.y).all(|v| v.is_finite()));
        // Label shares over LABEL_SAMPLES independent samples of the same size.
        let shares: Vec<Vec<f64>> = (0..LABEL_SAMPLES)
            .filter(|_| f.classes > 0)
            .map(|s| {
                let rows = sample(&cut, &card, SAMPLE_ROWS, 0, SEED + s as u64, "sampled");
                class_shares(&rows, f.classes)
            })
            .collect();
        let single: Vec<f64> = shares
            .iter()
            .map(|s| label_gap_in_se(f, std::slice::from_ref(s)))
            .collect();
        rows.push(LevelRow {
            level: level.to_string(),
            bytes: bytes.len(),
            retention: retention_of(f, &sampled),
            corr_gap: max_corr_gap(&f.fit, &sampled),
            label_se: single.first().copied().unwrap_or(0.0),
            label_se_mean: if shares.is_empty() {
                0.0
            } else {
                label_gap_in_se(f, &shares)
            },
            label_within: single.iter().filter(|&&g| g <= 3.0).count(),
        });
    }
    rows
}

/// The headline test: three tables, budget 2,048 bytes, every level.
#[test]
fn retention_by_level_on_three_tables() {
    let d = dir("retention");
    for f in fixtures() {
        let null = null_loss(&f.fit, &f.test, f.classes);
        let real = test_loss(&f.fit, &f.test, f.classes);
        println!(
            "{}: P={} n_fit={} n_test={} L0={null:.4} L_A={real:.4}",
            f.name,
            f.fit.p,
            f.fit.rows(),
            f.test.rows()
        );
        let rows = score_levels(&f, &d);
        for r in &rows {
            let labels = if f.classes > 0 {
                format!(
                    "  label gap {:.2} SE (mean of {LABEL_SAMPLES} samples {:.2} SE, {} of {LABEL_SAMPLES} within 3 SE)",
                    r.label_se, r.label_se_mean, r.label_within
                )
            } else {
                String::new()
            };
            println!(
                "  {} level {}: bytes {:4}  retention {:.4}  max|corr gap| {:.4}{labels}",
                f.name, r.level, r.bytes, r.retention, r.corr_gap
            );
        }
        // Truncation never grows a packet.
        for pair in rows.windows(2) {
            assert!(pair[0].bytes <= pair[1].bytes, "{}", f.name);
        }
        let top = rows.last().unwrap();
        assert!(top.bytes <= BUDGET);
        assert!(top.retention.is_finite());
        if f.name != "multiclass" {
            assert!(
                top.retention >= 0.85,
                "{} level E: {}",
                f.name,
                top.retention
            );
        }
        if f.classes > 0 {
            // Label shares of the fitted (level E) packet: the mean over the
            // samples is within 3 SE of the FIT shares, and so is every single
            // sample but at most one (with five classes a single sample of
            // exact draws leaves 3 SE about once in fifty).
            assert!(
                top.label_se_mean <= 3.0,
                "{}: {}",
                f.name,
                top.label_se_mean
            );
            assert!(
                top.label_within + 1 >= LABEL_SAMPLES,
                "{}: {}",
                f.name,
                top.label_within
            );
        }
    }
}

/// Retention of the two label rules, averaged over `seeds` samples.
fn rule_retention(f: &Fixture, packet: &Path, card: &Path, rule: &str, seeds: u64) -> f64 {
    let total: f64 = (0..seeds)
        .map(|s| retention_of(f, &sample(packet, card, SAMPLE_ROWS, 0, SEED + s, rule)))
        .sum();
    total / seeds as f64
}

/// The default label rule (a separate draw from `softmax(T * eta)`) against
/// tying the label to the feature regime, on the two class tables (the
/// multiclass one has class-separated means). Measured, mean of 8 samples:
/// the tied rule is ahead by about 0.02 at level E, where every class mean is
/// represented, and far behind at every lower level, so the separate draw
/// stays the default.
#[test]
fn sampled_labels_against_regime_labels() {
    let d = dir("label_rule");
    for f in fixtures().into_iter().filter(|f| f.classes > 0) {
        let (packet, card) = (
            d.join(format!("{}.dse", f.name)),
            d.join(format!("{}.json", f.name)),
        );
        fit(&f.fit, f.task, f.classes, BUDGET, &packet, &card);
        let mut line = Vec::new();
        for level in ["B", "D", "E"] {
            let cut = d.join(format!("{}.{level}.dse", f.name));
            truncate(&packet, level, &cut);
            let sampled = rule_retention(&f, &cut, &card, "sampled", 8);
            let regime = rule_retention(&f, &cut, &card, "regime", 8);
            line.push(format!(
                "level {level}: sampled {sampled:.4} regime {regime:.4}"
            ));
            // The default holds its retention at every level.
            assert!(sampled >= 0.9, "{} level {level}: {sampled}", f.name);
            if level == "E" {
                assert!(sampled >= regime - 0.05, "{}: {sampled} {regime}", f.name);
            } else {
                assert!(
                    sampled > regime,
                    "{} level {level}: {sampled} {regime}",
                    f.name
                );
            }
        }
        println!(
            "label rule, {} (mean of 8 samples): {}",
            f.name,
            line.join("; ")
        );
    }
}
