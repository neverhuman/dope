//! Loader for `SingleEncV8` F32/BF16 `rank <= 2` SafeTensors checkpoints (`--v8-weights`, `<= 1 GiB`).
//!
//! Before the v3.0.0 release checkpoint passes development qualification and is embedded under
//! `models/single-enc-v8/model.safetensors`, omitting `--v8-weights` cleanly yields `Ok(None)`,
//! selecting the weightless analytic Step-0 `DSE8` decoder (`NO_FINGERPRINT`). Passing
//! `--v8-weights` loads and validates the checkpoint fail-closed (auto-inferring attention heads
//! and Tier 1/2/3 nested code width `64|128|256`) and enforces `TAG_WEIGHTS_FP` + `DPA1` tag `8`
//! fingerprint equality on decode.

use dope_core::auto::V8Model;
use std::path::Path;

/// Load `--v8-weights` if supplied, failing closed on missing/corrupt/unsupported files.
pub(crate) fn load_v8_model(path: Option<&Path>) -> Result<Option<V8Model>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read --v8-weights: {e}"))?;
    let model = V8Model::from_safetensors_auto(&bytes)
        .map_err(|e| format!("--v8-weights is not valid SingleEncV8 F32/BF16 SafeTensors: {e}"))?;
    Ok(Some(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_corrupt_v8_weights_fail_closed() {
        assert!(load_v8_model(None).unwrap().is_none());
        assert!(load_v8_model(Some(Path::new("no-such-v8.safetensors")))
            .unwrap_err()
            .starts_with("cannot read --v8-weights"));
        let dir = std::env::temp_dir().join(format!("dope-v8w-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("bad.safetensors");
        std::fs::write(&bad, b"not-a-safetensors-file").unwrap();
        assert!(load_v8_model(Some(&bad))
            .unwrap_err()
            .contains("SingleEncV8"));
        std::fs::remove_dir_all(dir).ok();
    }
}
