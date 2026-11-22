//! Integration test for `dope sample --given <X>` diagnostic swap-ladder control.

use std::process::Command;

#[test]
fn sample_given_preserves_caller_features_and_draws_conditional_targets() {
    let bin = env!("CARGO_BIN_EXE_dope");
    let dir = std::env::temp_dir().join(format!("dope-given-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let train_csv = dir.join("train.csv");
    let given_csv = dir.join("given_x.csv");
    let model_path = dir.join("model.dope");
    let out_csv = dir.join("out.csv");

    let mut train = String::from("a,b,y\n");
    for i in 0..120 {
        let a = (i as f64) * 0.1;
        let b = ((i * 7) % 13) as f64;
        let y = 2.0 * a - 0.5 * b + 1.0;
        train += &format!("{a:.2},{b:.2},{y:.3}\n");
    }
    std::fs::write(&train_csv, train).unwrap();

    // Features-only CSV (no target column `y`!) to verify target-free --given support.
    std::fs::write(&given_csv, "a,b\n1.00,2.00\n5.00,4.00\n9.00,6.00\n").unwrap();

    let fit = Command::new(bin)
        .args([
            "fit",
            train_csv.to_str().unwrap(),
            "--target",
            "y",
            "--budget",
            "8KiB",
            "--seed",
            "7",
            "--quiet",
            "-o",
            model_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        fit.status.success(),
        "fit failed: {}",
        String::from_utf8_lossy(&fit.stderr)
    );

    let samp = Command::new(bin)
        .args([
            "sample",
            model_path.to_str().unwrap(),
            "--given",
            given_csv.to_str().unwrap(),
            "--seed",
            "7",
            "-o",
            out_csv.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        samp.status.success(),
        "sample --given failed: {}",
        String::from_utf8_lossy(&samp.stderr)
    );

    let out = std::fs::read_to_string(&out_csv).unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 4, "header + 3 given rows");
    assert_eq!(lines[0], "a,b,y");
    let row0: Vec<f64> = lines[1]
        .split(',')
        .map(|s| s.parse::<f64>().unwrap())
        .collect();
    let row2: Vec<f64> = lines[3]
        .split(',')
        .map(|s| s.parse::<f64>().unwrap())
        .collect();
    assert!((row0[0] - 1.0).abs() < 0.2);
    assert!((row2[0] - 9.0).abs() < 0.2);
    assert!(row2[2] > row0[2], "conditional target must track a=9 > a=1");
    let _ = std::fs::remove_dir_all(&dir);
}
