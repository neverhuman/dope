/// Tokens read as a missing cell (after trimming).
const MISSING_TOKENS: [&str; 10] = ["", "NA", "N/A", "NaN", "nan", "null", "NULL", "None", "?", "-"];
/// Label of the pooled level that holds categories rarer than the threshold.
pub const RARE_LEVEL: &str = "__rare__";
/// Categories above this count are refused or, for features, dropped as
/// identifier-like. 127 keeps adjacent codes two grid steps apart at 8 bits.
const MAX_LEVELS: usize = 127;
/// Smoothing weight (in rows) for the target statistic that orders levels.
const LEVEL_SMOOTHING: f64 = 10.0;

/// How the target is modelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskChoice {
    Auto,
    Regression,
    Binary,
}

/// Options for planning a headered table.
#[derive(Clone, Debug)]
pub struct TableOptions {
    pub target: String,
    pub drop: Vec<String>,
    pub task: TaskChoice,
    /// Categories seen fewer times are pooled into one rare level.
    pub min_level_count: usize,
    /// Absolute skewness above which a numeric column is log-scaled.
    pub skew_threshold: f64,
}

impl TableOptions {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            drop: Vec::new(),
            task: TaskChoice::Auto,
            min_level_count: 5,
            skew_threshold: 2.0,
        }
    }
}

/// Invertible map from one source column to the unit interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Encoding {
    /// u = (t(x) - low) / (high - low), where t is the identity or, for a
    /// heavily skewed column, sign * ln(1 + sign * (x - origin) / scale).
    Numeric {
        integer: bool,
        decimals: u8,
        log: Option<LogScale>,
        low: f32,
        high: f32,
    },
    /// Level i of L, ordered by its smoothed target statistic, sits at the
    /// code (i + 0.5) / L, so the code carries the level's main effect.
    Category { levels: Vec<String> },
    /// Days since 1970-01-01 (fractional when a time of day is present). The
    /// bounds are f64 so a time of day keeps its seconds.
    Date { time: bool, low: f64, high: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogScale {
    pub origin: f32,
    pub scale: f32,
    pub mirrored: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColumnPlan {
    pub name: String,
    pub encoding: Encoding,
}

/// The complete intake plan. `columns` keeps the source order without dropped
/// columns; the kernel sees every column except `target`, in that order, and
/// then the target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TablePlan {
    pub task: Task,
    pub columns: Vec<ColumnPlan>,
    pub target: usize,
}

fn is_missing(cell: &str) -> bool {
    MISSING_TOKENS.contains(&cell.trim())
}

fn parse_number(cell: &str) -> Option<f64> {
    cell.trim().parse::<f64>().ok().filter(|value| value.is_finite())
}

/// Days from 1970-01-01 for an ISO date `YYYY-MM-DD`, optionally followed by
/// `T` or a space and `HH:MM[:SS]`.
fn parse_date(cell: &str) -> Option<(f64, bool)> {
    let cell = cell.trim();
    let (date, time) = match cell.find(['T', ' ']) {
        Some(at) => (&cell[..at], Some(&cell[at + 1..])),
        None => (cell, None),
    };
    let parts = date.split('-').collect::<Vec<_>>();
    let [year, month, day] = parts.as_slice() else {
        return None;
    };
    if year.len() != 4 || month.len() != 2 || day.len() != 2 {
        return None;
    }
    let (year, month, day) = (year.parse::<i64>().ok()?, month.parse::<i64>().ok()?, day.parse::<i64>().ok()?);
    if !(1..=12).contains(&month) || !(1..=days_in_month(year, month)).contains(&day) {
        return None;
    }
    // Days from civil (Howard Hinnant's algorithm).
    let shifted = if month <= 2 { year - 1 } else { year };
    let era = shifted.div_euclid(400);
    let year_of_era = shifted - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = (era * 146_097 + day_of_era - 719_468) as f64;
    let Some(time) = time else {
        return Some((days, false));
    };
    let fields = time.split(':').map(|f| f.parse::<f64>().ok()).collect::<Option<Vec<_>>>()?;
    let (hours, minutes, seconds) = match fields.as_slice() {
        [h, m] => (*h, *m, 0.0),
        [h, m, s] => (*h, *m, *s),
        _ => return None,
    };
    if !(0.0..24.0).contains(&hours) || !(0.0..60.0).contains(&minutes) || !(0.0..61.0).contains(&seconds) {
        return None;
    }
    Some((days + (hours * 3600.0 + minutes * 60.0 + seconds) / 86_400.0, true))
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

fn format_date(days: f64, time: bool) -> String {
    // Round once at the output resolution, then split, so a time that rounds
    // up to midnight moves to the next day instead of printing 24:00:00.
    let (whole, seconds) = if time {
        let total = (days * 86_400.0).round() as i64;
        (total.div_euclid(86_400), total.rem_euclid(86_400))
    } else {
        (days.round() as i64, 0)
    };
    let z = whole + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    if !time {
        return format!("{year:04}-{month:02}-{day:02}");
    }
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}", seconds / 3600, seconds / 60 % 60, seconds % 60)
}

fn decimals_of(cell: &str) -> u8 {
    let cell = cell.trim();
    if cell.contains(['e', 'E']) {
        return 6;
    }
    cell.split_once('.').map_or(0, |(_, fraction)| fraction.len().min(6) as u8)
}

fn skewness(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let m2 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let m3 = values.iter().map(|v| (v - mean).powi(3)).sum::<f64>() / n;
    if m2 <= 1e-300 { 0.0 } else { m3 / m2.powf(1.5) }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

impl LogScale {
    fn forward(&self, x: f64) -> f64 {
        let sign = if self.mirrored { -1.0 } else { 1.0 };
        sign * (1.0 + (sign * (x - f64::from(self.origin)) / f64::from(self.scale)).max(0.0)).ln()
    }

    fn inverse(&self, t: f64) -> f64 {
        let sign = if self.mirrored { -1.0 } else { 1.0 };
        f64::from(self.origin) + sign * f64::from(self.scale) * ((sign * t).exp() - 1.0)
    }
}

fn plan_numeric(cells: &[&str], skew_threshold: f64) -> Encoding {
    let values = cells.iter().filter_map(|cell| parse_number(cell)).collect::<Vec<_>>();
    let integer = values.iter().all(|v| v.fract() == 0.0);
    let decimals = cells.iter().filter(|c| !is_missing(c)).map(|c| decimals_of(c)).max().unwrap_or(0);
    let distinct = values.iter().map(|v| v.to_bits()).collect::<BTreeSet<_>>().len();
    let skew = skewness(&values);
    let log = (distinct > 16 && skew.abs() > skew_threshold).then(|| {
        let mirrored = skew < 0.0;
        let origin = if mirrored {
            values.iter().copied().fold(f64::MIN, f64::max)
        } else {
            values.iter().copied().fold(f64::MAX, f64::min)
        };
        let mut distances = values.iter().map(|v| (v - origin).abs()).collect::<Vec<_>>();
        let typical = median(&mut distances);
        let mean = distances.iter().sum::<f64>() / distances.len() as f64;
        let scale = if typical > 0.0 { typical } else if mean > 0.0 { mean } else { 1.0 };
        LogScale { origin: origin as f32, scale: scale as f32, mirrored }
    });
    let transformed = values.iter().map(|&v| log.map_or(v, |l| l.forward(v)));
    let (low, high) = transformed.fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    Encoding::Numeric { integer, decimals, log, low: low as f32, high: high as f32 }
}

/// Order levels by their smoothed target statistic (ties by count, then name),
/// pooling rare levels.
fn plan_category(cells: &[&str], target: &[f64], min_count: usize) -> Option<Encoding> {
    let mut stats = BTreeMap::<String, (usize, f64)>::new();
    for (cell, &y) in cells.iter().zip(target) {
        // Missing tokens stay literal levels, so they decode to themselves.
        let key = cell.trim().to_string();
        let entry = stats.entry(key).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += y;
    }
    let rare = stats.iter().filter(|(_, (n, _))| *n < min_count).map(|(k, _)| k.clone()).collect::<Vec<_>>();
    // A single rare level keeps its label: pooling it with nothing would only lose the name.
    if rare.len() > 1 {
        let mut pooled = (0, 0.0);
        for key in rare {
            let (n, sum) = stats.remove(&key).expect("rare level present");
            pooled = (pooled.0 + n, pooled.1 + sum);
        }
        let entry = stats.entry(RARE_LEVEL.to_string()).or_insert((0, 0.0));
        entry.0 += pooled.0;
        entry.1 += pooled.1;
    }
    if stats.len() > MAX_LEVELS {
        return None;
    }
    let global = target.iter().sum::<f64>() / target.len().max(1) as f64;
    let mut levels = stats
        .into_iter()
        .map(|(name, (n, sum))| ((sum + LEVEL_SMOOTHING * global) / (n as f64 + LEVEL_SMOOTHING), n, name))
        .collect::<Vec<_>>();
    levels.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    Some(Encoding::Category { levels: levels.into_iter().map(|(_, _, name)| name).collect() })
}
