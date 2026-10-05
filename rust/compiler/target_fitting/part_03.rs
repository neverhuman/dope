#[cfg(any(test, feature = "gpu-training"))]
fn research_loss_record(step: usize, train: f64, validation: f64) -> String {
    format!("{step}\t{train:.8}\t{validation:.8}\n")
}

#[cfg(feature = "gpu-training")]
fn open_research_loss_log(
    table: &Table,
    logistic: bool,
    selected: &[usize],
    completed: &[Vec<f32>],
    device: tch::Device,
) -> Result<Option<(std::fs::File, tch::Tensor, tch::Tensor)>> {
    let path = match std::env::var("DOPE_RESEARCH_LOSS_LOG") {
        Ok(path) => path,
        Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(DopeError::Unsupported(
                "research loss log path is not Unicode".into(),
            ));
        }
    };
    let file = std::fs::File::create(&path)
        .map_err(|error| DopeError::Data(format!("research loss log: {error}")))?;
    let validation_path = match std::env::var("DOPE_RESEARCH_VALIDATION_CSV") {
        Ok(path) => path,
        Err(std::env::VarError::NotPresent) => {
            return Err(DopeError::Unsupported(
                "research loss log requires a validation CSV".into(),
            ));
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(DopeError::Unsupported(
                "research validation path is not Unicode".into(),
            ));
        }
    };
    let task = if logistic { Task::Binary } else { Task::Regression };
    let held_out = Table::read_csv(Path::new(&validation_path), task)?;
    if held_out.features != table.features || held_out.rows == 0 {
        return Err(DopeError::Data(
            "research validation width differs from the fit table".into(),
        ));
    }
    let fallbacks = selected
        .iter()
        .map(|&feature| {
            let mut observed: Vec<f32> = completed[feature]
                .iter()
                .copied()
                .filter(|item| item.is_finite())
                .collect();
            observed.sort_by(f32::total_cmp);
            quantile(&observed, 0.5)
        })
        .collect::<Vec<_>>();
    let mut rows = Vec::with_capacity(held_out.rows * selected.len());
    let mut targets = Vec::with_capacity(held_out.rows);
    for row in 0..held_out.rows {
        for (index, &feature) in selected.iter().enumerate() {
            let value = held_out.columns[feature][row];
            rows.push(if value.is_finite() { value } else { fallbacks[index] });
        }
        targets.push(held_out.target[row]);
    }
    let vx = tch::Tensor::from_slice(&rows)
        .view([held_out.rows as i64, selected.len() as i64])
        .to_device(device);
    let vy = tch::Tensor::from_slice(&targets)
        .view([held_out.rows as i64, 1])
        .to_device(device);
    Ok(Some((file, vx, vy)))
}

#[cfg(test)]
mod research_loss_log_tests {
    use super::*;

    #[test]
    fn research_loss_record_is_one_tsv_row() {
        assert_eq!(
            research_loss_record(0, 1.0, 0.5),
            "0\t1.00000000\t0.50000000\n"
        );
        assert_eq!(
            research_loss_record(2047, 0.25, 0.125),
            "2047\t0.25000000\t0.12500000\n"
        );
    }
}
