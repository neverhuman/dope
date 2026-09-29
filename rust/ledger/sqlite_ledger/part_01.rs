
#[derive(Clone, Debug)]
pub struct CampaignLedger {
    path: PathBuf,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}