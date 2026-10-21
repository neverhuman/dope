/// A planned table: the plan plus the encoded unit-interval matrix it implies.
pub struct PlannedTable {
    pub plan: TablePlan,
    /// Row-major features (NaN marks a missing cell), kernel column order.
    pub features: Vec<f32>,
    pub target: Vec<f32>,
    pub rows: usize,
    /// Source columns left out, with the reason.
    pub dropped: Vec<(String, &'static str)>,
    /// Rows without a target value, which are not fitted.
    pub rows_without_target: usize,
}

impl TablePlan {
    pub fn feature_count(&self) -> usize {
        self.columns.len() - 1
    }

    /// Kernel column order: every planned column except the target, then it.
    pub fn kernel_order(&self) -> Vec<usize> {
        (0..self.columns.len())
            .filter(|&index| index != self.target)
            .chain(std::iter::once(self.target))
            .collect()
    }

    pub fn header(&self) -> Vec<&str> {
        self.columns.iter().map(|column| column.name.as_str()).collect()
    }
}

fn plan_target(cells: &[&str], choice: TaskChoice, skew_threshold: f64) -> Result<(Task, Encoding)> {
    let present = cells.iter().filter(|c| !is_missing(c)).collect::<Vec<_>>();
    let numeric = present.iter().all(|c| parse_number(c).is_some());
    // Numeric labels are distinct by value, so "1" and "1.0" are one level;
    // the first spelling seen names it.
    let mut seen = BTreeSet::new();
    let distinct = present
        .iter()
        .map(|c| c.trim())
        .filter(|c| seen.insert(if numeric { parse_number(c).map_or(0, |v| (v + 0.0).to_bits()).to_string() } else { c.to_string() }))
        .collect::<Vec<_>>();
    if distinct.len() < 2 {
        return Err(DopeError::Data("table_target_constant".into()));
    }
    let binary = match choice {
        TaskChoice::Binary => true,
        TaskChoice::Regression => false,
        TaskChoice::Auto => distinct.len() == 2,
    };
    if binary {
        if distinct.len() != 2 {
            return Err(DopeError::Data("table_binary_target_needs_two_levels".into()));
        }
        let mut levels = distinct.into_iter().map(str::to_string).collect::<Vec<_>>();
        // The second level is the positive class: numeric labels by value, others by text,
        // never by row order.
        if numeric {
            levels.sort_by(|a, b| parse_number(a).unwrap().total_cmp(&parse_number(b).unwrap()));
        } else {
            levels.sort();
        }
        return Ok((Task::Binary, Encoding::Category { levels }));
    }
    if !numeric {
        return Err(DopeError::Unsupported(
            "table_multiclass_target_unsupported: use a numeric or two-level target".into(),
        ));
    }
    Ok((Task::Regression, plan_numeric(cells, skew_threshold)))
}

impl Encoding {
    /// Map one source cell to the unit interval; NaN marks a missing cell.
    pub fn encode(&self, cell: &str) -> f32 {
        match self {
            Encoding::Numeric { log, low, high, .. } => {
                let Some(value) = (!is_missing(cell)).then(|| parse_number(cell)).flatten() else {
                    return f32::NAN;
                };
                let value = log.map_or(value, |scale| scale.forward(value));
                unit_position(value, f64::from(*low), f64::from(*high))
            }
            Encoding::Date { low, high, .. } => match parse_date(cell) {
                Some((days, _)) if !is_missing(cell) => unit_position(days, *low, *high),
                _ => f32::NAN,
            },
            Encoding::Category { levels } => {
                let key = cell.trim();
                let numeric = parse_number(key);
                let index = levels
                    .iter()
                    .position(|level| level == key)
                    .or_else(|| numeric.and_then(|v| levels.iter().position(|level| parse_number(level) == Some(v))))
                    .or_else(|| levels.iter().position(|level| level == RARE_LEVEL))
                    .unwrap_or(0);
                (index as f32 + 0.5) / levels.len() as f32
            }
        }
    }

    /// Map a sampled unit value back to a source cell.
    pub fn decode(&self, value: f32) -> String {
        if value.is_nan() {
            return String::new();
        }
        match self {
            Encoding::Numeric { integer, decimals, log, low, high } => {
                let t = f64::from(*low) + f64::from(value.clamp(0.0, 1.0)) * f64::from(high - low);
                let x = log.map_or(t, |scale| scale.inverse(t));
                if *integer {
                    format!("{}", x.round() as i64)
                } else {
                    format!("{x:.prec$}", prec = usize::from(*decimals))
                }
            }
            Encoding::Date { time, low, high } => {
                let days = low + f64::from(value.clamp(0.0, 1.0)) * (high - low);
                format_date(days, *time)
            }
            Encoding::Category { levels } => {
                let index = ((value.clamp(0.0, 1.0) * levels.len() as f32) as usize).min(levels.len() - 1);
                levels[index].clone()
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Encoding::Numeric { log: Some(_), .. } => "numeric_log",
            Encoding::Numeric { integer: true, .. } => "integer",
            Encoding::Numeric { .. } => "numeric",
            Encoding::Category { .. } => "category",
            Encoding::Date { .. } => "date",
        }
    }
}

fn unit_position(value: f64, low: f64, high: f64) -> f32 {
    let spread = high - low;
    if spread <= 0.0 {
        return 0.0;
    }
    ((value - low) / spread).clamp(0.0, 1.0) as f32
}

fn plan_feature(cells: &[&str], target: &[f64], options: &TableOptions) -> std::result::Result<Encoding, &'static str> {
    let present = cells.iter().copied().filter(|c| !is_missing(c)).collect::<Vec<_>>();
    if present.is_empty() {
        return Err("all_missing");
    }
    if present.iter().all(|c| parse_number(c).is_some()) {
        return Ok(plan_numeric(cells, options.skew_threshold));
    }
    if present.iter().all(|c| parse_date(c).is_some()) {
        let parsed = present.iter().filter_map(|c| parse_date(c)).collect::<Vec<_>>();
        let time = parsed.iter().any(|(_, has_time)| *has_time);
        let (low, high) = parsed.iter().fold((f64::MAX, f64::MIN), |(lo, hi), (d, _)| (lo.min(*d), hi.max(*d)));
        return Ok(Encoding::Date { time, low, high });
    }
    plan_category(cells, target, options.min_level_count).ok_or("identifier_like_or_too_many_levels")
}

/// Plan and encode a headered table held as rows of string cells.
pub fn plan_table(header: &[String], rows: &[Vec<String>], options: &TableOptions) -> Result<PlannedTable> {
    if header.len() < 2
        || header.iter().any(|name| name.trim().is_empty())
        || header.iter().collect::<BTreeSet<_>>().len() != header.len()
        || rows.iter().any(|row| row.len() != header.len())
        || rows.len().saturating_mul(header.len()) > MAX_TABLE_CELLS
    {
        return Err(DopeError::Data("table_invalid_shape_or_header".into()));
    }
    let target_source = header
        .iter()
        .position(|name| name == &options.target)
        .ok_or_else(|| DopeError::Data("table_target_not_in_header".into()))?;
    let kept_rows = rows
        .iter()
        .filter(|row| row.get(target_source).is_some_and(|cell| !is_missing(cell)))
        .collect::<Vec<_>>();
    if kept_rows.len() < 8 {
        return Err(DopeError::Data("table_needs_at_least_eight_labelled_rows".into()));
    }
    let column = |index: usize| kept_rows.iter().map(|row| row.get(index).map_or("", String::as_str)).collect::<Vec<_>>();
    let target_cells = column(target_source);
    let (task, target_encoding) = plan_target(&target_cells, options.task, options.skew_threshold)?;
    let target_unit = target_cells.iter().map(|cell| f64::from(target_encoding.encode(cell))).collect::<Vec<_>>();
    let target_unit = match task {
        Task::Binary => target_unit.iter().map(|v| if *v >= 0.5 { 1.0 } else { 0.0 }).collect(),
        Task::Regression => target_unit,
    };
    let mut columns = Vec::new();
    let mut dropped = Vec::new();
    let mut target = 0;
    for (index, name) in header.iter().enumerate() {
        if index == target_source {
            target = columns.len();
            columns.push(ColumnPlan { name: name.clone(), encoding: target_encoding.clone() });
        } else if options.drop.contains(name) {
            dropped.push((name.clone(), "requested"));
        } else {
            match plan_feature(&column(index), &target_unit, options) {
                Ok(encoding) => columns.push(ColumnPlan { name: name.clone(), encoding }),
                Err(reason) => dropped.push((name.clone(), reason)),
            }
        }
    }
    let target_column = columns.remove(target);
    columns.push(target_column);
    let target = columns.len() - 1;
    let plan = TablePlan { task, columns, target };
    if plan.feature_count() == 0 {
        return Err(DopeError::Data("table_has_no_usable_feature".into()));
    }
    let source_index = |planned: usize| header.iter().position(|n| n == &plan.columns[planned].name).expect("planned column");
    let order = plan.kernel_order();
    let mut features = Vec::with_capacity(kept_rows.len() * plan.feature_count());
    let mut encoded_target = Vec::with_capacity(kept_rows.len());
    for row in &kept_rows {
        for &planned in &order[..order.len() - 1] {
            let cell = row.get(source_index(planned)).map_or("", String::as_str);
            features.push(plan.columns[planned].encoding.encode(cell));
        }
        let value = plan.columns[plan.target].encoding.encode(&row[target_source]);
        encoded_target.push(match task {
            Task::Binary => if value >= 0.5 { 1.0 } else { 0.0 },
            Task::Regression => value,
        });
    }
    Ok(PlannedTable {
        rows: kept_rows.len(),
        rows_without_target: rows.len() - kept_rows.len(),
        plan,
        features,
        target: encoded_target,
        dropped,
    })
}

/// Decode kernel samples (row-major, kernel column order) to source-order cells.
pub fn decode_rows(plan: &TablePlan, samples: &[f32]) -> Vec<Vec<String>> {
    let order = plan.kernel_order();
    samples
        .chunks_exact(order.len())
        .map(|row| {
            let mut cells = vec![String::new(); plan.columns.len()];
            for (position, &planned) in order.iter().enumerate() {
                let value = row[position];
                cells[planned] = if planned == plan.target && plan.task == Task::Binary {
                    plan.columns[planned].encoding.decode(if value >= 0.5 { 0.75 } else { 0.25 })
                } else {
                    plan.columns[planned].encoding.decode(value)
                };
            }
            cells
        })
        .collect()
}
