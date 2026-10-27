//! Run with a previously verified, pinned SafeTensors file.
use dope_core::mlc_v3::{MlcV3Model, MLC_V3_FEATURES_629_DIM};
use sha2::{Digest, Sha256};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected checkpoint path")?;
    let raw = std::fs::read(Path::new(&path))?;
    let digest = format!("{:x}", Sha256::digest(&raw));
    if digest != "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c" {
        return Err("checkpoint SHA-256 mismatch".into());
    }
    let model = MlcV3Model::from_safetensors_bytes(&raw)?;
    let features = model.encode_features_629(&[0.0; 256], &vec![vec![0.0; 17]; 64], "high")?;
    assert_eq!(features.len(), MLC_V3_FEATURES_629_DIM);
    println!("encoded {} coordinates", features.len());
    Ok(())
}
