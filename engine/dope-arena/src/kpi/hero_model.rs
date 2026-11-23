//! Inputs and text of `dope_kpi render-hero`: the points, reference lines
//! and theme tokens of the README hook graphic, and the fixed number formats
//! of every label. Standard library only.

pub const DEFAULT_TITLE: &str = "Predictive structure kept vs. artifact size";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Light,
    Dark,
}

/// Colour tokens of a theme. Text only ever wears `ink`, `ink2` or `muted`.
#[derive(Debug)]
pub struct Tokens {
    pub surface: &'static str,
    pub ink: &'static str,
    pub ink2: &'static str,
    pub muted: &'static str,
    pub grid: &'static str,
    pub axis: &'static str,
    pub accent: &'static str,
    pub accent_light: &'static str,
    pub gray: &'static str,
}

pub const LIGHT: Tokens = Tokens {
    surface: "#fcfcfb",
    ink: "#0b0b0b",
    ink2: "#52514e",
    muted: "#898781",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    accent: "#2a78d6",
    accent_light: "#86b6ef",
    gray: "#898781",
};

pub const DARK: Tokens = Tokens {
    surface: "#1a1a19",
    ink: "#f0efec",
    ink2: "#c3c2b7",
    muted: "#898781",
    grid: "#2c2c2a",
    axis: "#383835",
    // Darker previous-release step: on a dark surface the step toward the
    // surface is the darker one (OKLab difference 15.6, both >= 3:1).
    accent: "#4a95ee",
    accent_light: "#2e66ad",
    gray: "#898781",
};

impl Theme {
    pub fn tokens(self) -> &'static Tokens {
        match self {
            Theme::Light => &LIGHT,
            Theme::Dark => &DARK,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Theme::Light => "light",
            Theme::Dark => "dark",
        }
    }
}

/// Emphasis of a point, in drawing order (the highlight is drawn last).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Emphasis {
    Other,
    Previous,
    Highlight,
}

impl Emphasis {
    pub fn class(self) -> &'static str {
        match self {
            Emphasis::Highlight => "hl",
            Emphasis::Previous => "pv",
            Emphasis::Other => "ot",
        }
    }

    pub fn color(self, t: &Tokens) -> &'static str {
        match self {
            Emphasis::Highlight => t.accent,
            Emphasis::Previous => t.accent_light,
            Emphasis::Other => t.gray,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub label: String,
    /// Median emitted-file size, positive.
    pub bytes: f64,
    pub trg: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub tier: Emphasis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    pub label: String,
    pub trg: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hero {
    pub title: String,
    pub note: Option<String>,
    /// At least one.
    pub points: Vec<Point>,
    pub references: Vec<Reference>,
    /// Headline byte budget, drawn as the dashed reference line.
    pub budget: f64,
    /// Small print at the bottom right (contract and receipt digests).
    pub footer: String,
}

/// A coordinate with one decimal and no negative zero.
pub fn coord(x: f64) -> String {
    let text = format!("{x:.1}");
    if text == "-0.0" {
        "0.0".to_string()
    } else {
        text
    }
}

fn minus(text: String) -> String {
    text.replacen('-', "\u{2212}", 1)
}

/// `x` with 3 decimals, no negative zero and a true minus sign.
pub fn trg_text(x: f64) -> String {
    let text = format!("{x:.3}");
    if text == "-0.000" {
        "0.000".to_string()
    } else {
        minus(text)
    }
}

/// A gridline value: `0`, else 2 decimals with a true minus sign.
pub fn tick_text(value: f64) -> String {
    let text = format!("{value:.2}");
    if text == "0.00" || text == "-0.00" {
        "0".to_string()
    } else {
        minus(text)
    }
}

/// Bytes with thousands separators: `1,462 B`, `1,462.5 B`.
pub fn bytes_text(bytes: f64) -> String {
    let text = if bytes.fract() == 0.0 {
        format!("{bytes:.0}")
    } else {
        format!("{bytes:.1}")
    };
    let (whole, frac) = text.split_once('.').unwrap_or((&text, ""));
    let mut grouped = String::new();
    for (i, digit) in whole.chars().enumerate() {
        if i > 0 && (whole.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    if frac.is_empty() {
        format!("{grouped} B")
    } else {
        format!("{grouped}.{frac} B")
    }
}

/// `2 KiB budget` for 2,048 bytes; whole MiB and KiB are named, the rest in bytes.
pub fn budget_text(budget: f64) -> String {
    let whole = budget as u64;
    let exact = budget.fract() == 0.0 && whole > 0;
    if exact && whole % 1_048_576 == 0 {
        format!("{} MiB budget", whole / 1_048_576)
    } else if exact && whole % 1024 == 0 {
        format!("{} KiB budget", whole / 1024)
    } else {
        format!("{} budget", bytes_text(budget))
    }
}

/// `text` safe inside XML character data and attributes (control characters dropped).
pub fn esc(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            ch if ch.is_control() => {}
            ch => out.push(ch),
        }
    }
    out
}

/// `DOPE v2.0.0: TRG −0.232 [−0.347, −0.131], 1,462 B`.
pub fn point_title(p: &Point) -> String {
    format!(
        "{}: TRG {} [{}, {}], {}",
        p.label,
        trg_text(p.trg),
        trg_text(p.ci_low),
        trg_text(p.ci_high),
        bytes_text(p.bytes)
    )
}

/// The text alternative: the note, what is plotted, every point and control.
pub fn describe(hero: &Hero) -> String {
    let mut parts: Vec<String> = hero.note.iter().cloned().collect();
    parts.push(format!(
        "TRG with its 95% interval against the median emitted-file size on a log axis; a dashed line marks the {} and dotted lines mark controls.",
        budget_text(hero.budget)
    ));
    parts.extend(hero.points.iter().map(|p| format!("{}.", point_title(p))));
    if !hero.references.is_empty() {
        let refs: Vec<String> = hero
            .references
            .iter()
            .map(|r| format!("{} {}", r.label, trg_text(r.trg)))
            .collect();
        parts.push(format!("Controls: {}.", refs.join("; ")));
    }
    parts.join(" ")
}
