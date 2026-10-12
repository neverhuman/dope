include!("certification/part_01.rs");
include!("certification/part_02.rs");

// Legacy entry points keep their receipt layout. Protected evaluation supplies
// tables read by the owner endpoint after its durable access event.
struct CertificationInputs<'a> {
    real_dir: &'a Path,
    kernel_path: &'a Path,
    out: &'a Path,
    seed: u64,
    runtime_dictionary_bytes: usize,
    supported_datasets: usize,
    policy: &'a ReleasePolicy,
    tables: Option<(Table, Table)>,
}

pub(crate) fn certify_frozen_tables(
    frozen: &crate::evaluation::FrozenSelection,
    kernel_path: &Path,
    out: &Path,
    train: Table,
    test: Table,
) -> Result<CertificationReport> {
    let (seed, runtime_dictionary_bytes, supported_datasets, policy) = frozen.parameters();
    certify_inputs(CertificationInputs {
        real_dir: Path::new("."),
        kernel_path,
        out,
        seed,
        runtime_dictionary_bytes,
        supported_datasets,
        policy,
        tables: Some((train, test)),
    })
}
