

fn mean_and_upper_95(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    Some((
        average,
        average + 1.645 * deviation / (values.len() as f64).sqrt(),
    ))
}

fn mean_and_lower_95(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let average = values.iter().sum::<f64>() / values.len() as f64;
    let deviation = if values.len() > 1 {
        (values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / (values.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    Some((
        average,
        average - 1.96 * deviation / (values.len() as f64).sqrt(),
    ))
}

fn float_percentile(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * quantile).ceil() as usize])
}

fn update_optional_max(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *target = Some(target.map_or(value, |current| current.max(value)));
    }
}

fn update_optional_min(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        *target = Some(target.map_or(value, |current| current.min(value)));
    }
}

fn is_timeout_failure(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("timeout") || error.contains("timed out") || error.contains("deadline")
}
