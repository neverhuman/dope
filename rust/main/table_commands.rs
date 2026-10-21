fn run_table(command: Command) -> Result<()> {
    use dope_kernel::table;
    match command {
        Command::FitTable {
            csv,
            target,
            out,
            drop,
            task,
            seed,
            candidate,
            native_target_profile,
            min_level_count,
        } => {
            let mut options = table::FitTableOptions::new(target);
            options.table.drop = drop;
            options.table.task = match task.as_str() {
                "auto" => table::TaskChoice::Auto,
                "regression" => table::TaskChoice::Regression,
                "binary" => table::TaskChoice::Binary,
                _ => return Err(DopeError::Data("task must be auto, regression or binary".into())),
            };
            options.table.min_level_count = min_level_count.max(1);
            options.seed = seed;
            options.candidate = candidate;
            options.native_target_profile = match native_target_profile.as_str() {
                "none" => None,
                name => Some(dope_kernel::compiler::NativeTargetProfile::parse(name)?),
            };
            if out.exists() {
                return Err(DopeError::Data("output exists; choose a new --out path".into()));
            }
            let (header, rows) = table::read_csv_table(&csv)?;
            let (artifact, report) = table::fit_table(&header, &rows, &options)?;
            fs::write(&out, &artifact).map_err(|error| dope_kernel::error::io_error(&out, error))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::SampleTable { model, rows, seed, out } => {
            let bytes = fs::read(&model).map_err(|error| dope_kernel::error::io_error(&model, error))?;
            let (header, sampled) = table::sample_table(&bytes, rows, seed)?;
            table::write_csv_table(&out, &header, &sampled)?;
            println!("{}", serde_json::json!({"out": out, "rows": rows}));
        }
        Command::InspectTable { model } => {
            let bytes = fs::read(&model).map_err(|error| dope_kernel::error::io_error(&model, error))?;
            println!("{}", serde_json::to_string_pretty(&table::inspect_table(&bytes)?)?);
        }
        _ => unreachable!("table commands only"),
    }
    Ok(())
}
