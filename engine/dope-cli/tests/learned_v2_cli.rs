//! A real native v2 CLI journey with all selected widths and no recipient encoder.
use std::{path::PathBuf, process::Command};

#[test]
fn v2_encode_and_all_tier_recipient_sample_use_declared_layout_widths() {
    for variant in ["slot", "slot-bounded"] {
        let root =
            std::env::temp_dir().join(format!("dope-v2-cli-{}-{variant}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../dope-core/tests/fixtures/learned-v2/{variant}/model"
        ));
        let csv = root.join("fit.csv");
        let mut text = String::from("PRIVATE_HEADER,c,target\n");
        for i in 0..80 {
            text += &format!("{},PRIVATE_CATEGORY_{},{}\n", i, i % 2, i * 2);
        }
        std::fs::write(&csv, text).unwrap();
        let code = root.join("all.code");
        let encoded = Command::new(env!("CARGO_BIN_EXE_dope"))
            .args(["--threads", "1", "learned", "encode", "--model"])
            .arg(&model)
            .arg("--csv")
            .arg(&csv)
            .args([
                "--target",
                "target",
                "--kinds",
                "numeric,categorical,numeric",
                "--task",
                "regression",
                "--levels",
                "low,medium,high,ultra",
                "--out",
            ])
            .arg(&code)
            .output()
            .unwrap();
        assert!(encoded.status.success(), "{encoded:?}");
        assert!(!String::from_utf8_lossy(&std::fs::read(&code).unwrap()).contains("PRIVATE_"));
        let recipient = root.join("recipient");
        std::fs::create_dir(&recipient).unwrap();
        for name in ["model.json", "decoder.safetensors"] {
            std::fs::copy(model.join(name), recipient.join(name)).unwrap();
        }
        std::fs::remove_file(&csv).unwrap();
        for level in ["low", "medium", "high", "ultra"] {
            let out = root.join(format!("{level}.csv"));
            let sampled = Command::new(env!("CARGO_BIN_EXE_dope"))
                .env("PATH", "/nonexistent")
                .args(["--threads", "1", "learned", "sample", "--model"])
                .arg(&recipient)
                .arg("--code")
                .arg(&code)
                .args(["--level", level, "--rows", "4", "--steps", "8", "--out"])
                .arg(&out)
                .output()
                .unwrap();
            assert!(sampled.status.success(), "{level}: {sampled:?}");
            assert_eq!(std::fs::read_to_string(out).unwrap().lines().count(), 5);
        }
        assert!(!recipient.join("encoder.safetensors").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
