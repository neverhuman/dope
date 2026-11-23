//! The SingleEncV7 checkpoint compiled into `dope` for `fit --format auto|v7` and for
//! sampling V7 `DPA1` artifacts, and the `--v7-weights` override.
//!
//! Swapping the compiled weights is a one-path change: overwrite
//! `models/single-enc-v7/model.safetensors`, or point `v7_weights_path!` at another
//! SingleEncV7 F32 `.safetensors` file with the same layout, then rebuild. Keep
//! `models/single-enc-v7/manifest.json` describing the compiled file; the test below checks
//! its byte length, SHA-256 and shape against the compiled bytes. Artifacts record the
//! weights' fingerprint, so V7 artifacts fitted with other weights are refused rather than
//! decoded wrongly.

use dope_core::auto::V7Model;
use std::path::Path;

/// Path of the compiled checkpoint, relative to this source file.
macro_rules! v7_weights_path {
    () => {
        "../../../models/single-enc-v7/model.safetensors"
    };
}

/// The compiled checkpoint.
pub(crate) static V7_WEIGHTS: &[u8] = include_bytes!(v7_weights_path!());

/// Attention heads of the compiled checkpoint (its manifest's `n_heads`), also assumed for
/// `--v7-weights` files.
pub(crate) const V7_N_HEADS: usize = 4;

/// The weights to fit or sample V7 with: the `--v7-weights` file when given, else the
/// compiled checkpoint. Both load fail-closed, as the runtime loader.
pub(crate) fn load_model(path: Option<&Path>) -> Result<V7Model, String> {
    let Some(path) = path else {
        return V7Model::from_safetensors(V7_WEIGHTS, V7_N_HEADS)
            .map_err(|e| format!("the embedded SingleEncV7 weights did not load: {e}"));
    };
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read --v7-weights: {e}"))?;
    V7Model::from_safetensors(&bytes, V7_N_HEADS).map_err(|e| {
        format!("--v7-weights is not SingleEncV7 F32 SafeTensors with {V7_N_HEADS} heads: {e}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = include_str!("../../../models/single-enc-v7/manifest.json");
    const DECISION: &str = include_str!("../../../models/single-enc-v7/V7_RELEASE_CHECKPOINT.md");

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn embedded_weights_match_their_manifest_and_decision_record() {
        let m: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
        assert_eq!(m["bytes"].as_u64(), Some(V7_WEIGHTS.len() as u64));
        let model = load_model(None).unwrap();
        let sha = hex(&model.sha256());
        assert_eq!(
            m["sha256"].as_str(),
            Some(sha.as_str()),
            "embedded bytes vs manifest"
        );
        let source = m["source_sha256"].as_str().unwrap();
        assert!(
            source.len() == 64 && source != sha,
            "source checkpoint digest"
        );
        assert_eq!(m["decision_record"], "V7_RELEASE_CHECKPOINT.md");
        assert!(
            DECISION.contains(&sha),
            "decision record names the embedded bytes"
        );
        assert_eq!(m["n_heads"].as_u64(), Some(V7_N_HEADS as u64));
        let w = model.weights();
        assert_eq!(m["d_model"].as_u64(), Some(w.d_model as u64));
        assert_eq!(m["n_layers"].as_u64(), Some(w.n_layers as u64));
        assert_eq!(m["h_target"].as_u64(), Some(w.h_target as u64));
        let text = MANIFEST.to_ascii_lowercase();
        assert!(
            !text.contains("/home/") && !text.contains("xbabe"),
            "host detail in manifest"
        );
    }

    #[test]
    fn override_weights_fail_closed() {
        assert!(load_model(Some(Path::new("no-such-weights.safetensors")))
            .unwrap_err()
            .starts_with("cannot read --v7-weights"));
        let dir = std::env::temp_dir().join(format!("dope-v7w-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cut = dir.join("cut.safetensors");
        std::fs::write(&cut, &V7_WEIGHTS[..V7_WEIGHTS.len() / 2]).unwrap();
        assert!(load_model(Some(&cut)).is_err());
        let same = dir.join("same.safetensors");
        std::fs::write(&same, V7_WEIGHTS).unwrap();
        let loaded = load_model(Some(&same)).unwrap();
        assert_eq!(
            loaded.fingerprint(),
            load_model(None).unwrap().fingerprint()
        );
        std::fs::remove_dir_all(dir).ok();
    }
}
