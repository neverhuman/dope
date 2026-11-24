//! The release comparison graphic combines the public equal-cap and default-fit evidence.
use super::hero::METHODS_LABELS;
use super::hero_model::{
    budget_text, bytes_text, esc, score_text, Emphasis, Hero, Point, Theme, YScale,
};
use super::receipt::Receipt;
use super::rundir::contract;
use dope_arena::{ArenaError, ArenaResult};
use serde::Deserialize;
use std::fmt::Write;

const DOPE: &str = "dope-v3b-auto-v8";

#[derive(Deserialize)]
struct BudgetRow {
    cap: u64,
    score: f64,
    low: f64,
    high: f64,
    median: f64,
    informative: usize,
}

fn text(svg: &mut String, x: u32, y: u32, class: &str, value: &str) {
    writeln!(
        svg,
        r#"<text x="{x}" y="{y}" class="{class}">{}</text>"#,
        esc(value)
    )
    .unwrap();
}

fn interval(svg: &mut String, point: &Point, y: u32, x: impl Fn(f64) -> f64) {
    let scale = YScale::TransferScore;
    let low = x(scale.convert(point.ci_low));
    let high = x(scale.convert(point.ci_high));
    let score = x(scale.convert(point.trg));
    let class = if point.tier == Emphasis::Highlight {
        "winner"
    } else {
        "peer"
    };
    writeln!(
        svg,
        r#"<g class="{class}"><line x1="{low:.1}" x2="{high:.1}" y1="{y}" y2="{y}"/><circle cx="{score:.1}" cy="{y}" r="5"/></g>"#,
    )
    .unwrap();
}

fn axis(svg: &mut String, x: impl Fn(f64) -> f64, top: u32, bottom: u32) {
    for tick in [-100, 0, 100] {
        let pos = x(f64::from(tick));
        writeln!(
            svg,
            r#"<line class="grid" x1="{pos:.1}" x2="{pos:.1}" y1="{top}" y2="{bottom}"/>"#
        )
        .unwrap();
        text(
            svg,
            pos.round() as u32,
            bottom + 19,
            "tick",
            &score_text(f64::from(tick)).replace(".0", ""),
        );
    }
}

fn compact_size(bytes: f64) -> String {
    if bytes >= 1_000_000.0 {
        format!("{:.2} MB", bytes / 1_000_000.0)
    } else if bytes >= 10_000.0 {
        format!("{:.0} kB", bytes / 1_000.0)
    } else {
        bytes_text(bytes)
    }
}

pub(super) fn render(
    hero: &Hero,
    theme: Theme,
    rows: &[u8],
    receipts: &[Receipt],
) -> ArenaResult<String> {
    let mut points = hero.points.clone();
    points.sort_by(|a, b| b.trg.total_cmp(&a.trg));
    let v3 = points.iter().find(|p| p.tier == Emphasis::Highlight);
    let Some(v3) = v3.filter(|_| points.len() == 5) else {
        return contract("comparison hero requires the v3 and four external arms");
    };
    let mut matching = csv::Reader::from_reader(rows)
        .deserialize::<BudgetRow>()
        .filter_map(|row| match row {
            Ok(row) if row.cap as f64 == hero.budget => Some(Ok(row)),
            Ok(_) => None,
            Err(err) => Some(Err(ArenaError::Contract(format!(
                "invalid budget sweep CSV: {err}"
            )))),
        })
        .collect::<ArenaResult<Vec<_>>>()?;
    let Some(rows) = matching.pop().filter(|_| matching.is_empty()) else {
        return contract("budget sweep must have one row for the reported cap");
    };
    if ![rows.score, rows.low, rows.high, rows.median]
        .iter()
        .all(|v| v.is_finite())
        || rows.median <= 0.0
        || rows.median > hero.budget
        || rows.low > rows.high
    {
        return contract("invalid stored-row size or score interval");
    }
    let as_trg = |score: f64| score / 100.0 - 1.0;
    let uniform = Point {
        label: "Stored real rows".into(),
        bytes: rows.median,
        trg: as_trg(rows.score),
        ci_low: as_trg(rows.low),
        ci_high: as_trg(rows.high),
        tier: Emphasis::Other,
    };
    let scale = YScale::TransferScore;
    let low = points
        .iter()
        .chain([&uniform])
        .map(|p| scale.convert(p.ci_low))
        .fold(-100.0, f64::min);
    let high = points
        .iter()
        .chain([&uniform])
        .map(|p| scale.convert(p.ci_high))
        .fold(100.0, f64::max);
    let fraction = |v| (v - low) / (high - low);
    let left_axis = |v| 43.0 + 238.0 * fraction(v);
    let right_axis = |v| 473.0 + 166.0 * fraction(v);
    let budget = budget_text(hero.budget);
    let title = if points[0].tier == Emphasis::Highlight && points[0].trg > points[1].trg {
        format!("DOPE v3 leads with a {budget}")
    } else {
        format!("Predictive utility under a {budget}")
    };
    let tokens = theme.tokens();
    let accent = match theme {
        Theme::Light => "#086a58",
        Theme::Dark => "#68dbc2",
    };
    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="988" height="438" viewBox="0 0 988 438" role="img" aria-labelledby="title desc">
<title id="title">{title_xml}</title>
<desc id="desc">Two comparisons on the development report splits at four times the fitting rows. The left panel compares signed DOPE v3 with storing fitting rows under the same cap. The right panel compares v3 with four uncapped library-default generators, with scores, intervals and file sizes.</desc>
<style>
text {{font-family:DejaVu Sans,sans-serif;font-size:15px;fill:{ink}}}
.headline {{font-size:32px;font-weight:700}}
.heading {{font-size:20px;font-weight:700}}
.note,.tick,.column,.gain {{fill:{muted};font-size:13px}}
.column,.value,.gain,.lead-value {{text-anchor:end}}
.gain {{fill:{accent};font-weight:700}}
.detail {{fill:{muted};font-size:12px;text-anchor:end}}
.value,.lead {{font-weight:700}}
.lead {{fill:{accent}}}
.lead-value {{font-size:31px;font-weight:700}}
.lead-value.lead {{fill:{accent}}}
.tick {{text-anchor:middle}}
.grid {{stroke:{grid}}}
.winner {{stroke:{accent};fill:{accent};stroke-width:2.5}}
.peer {{stroke:{muted};fill:{muted};stroke-width:1.8}}
</style>
<rect width="988" height="438" rx="14" fill="{surface}"/>
<line x1="315" x2="315" y1="88" y2="365" class="grid"/>
"#,
        title_xml = esc(&title),
        ink = tokens.ink,
        muted = tokens.ink2,
        grid = tokens.grid,
        surface = tokens.surface,
    );
    text(&mut svg, 24, 43, "headline", &title);
    text(
        &mut svg,
        24,
        68,
        "note",
        "Transfer Score ↑  ·  Development report splits  ·  4× sampling",
    );
    text(&mut svg, 24, 105, "heading", "Same byte cap");
    text(
        &mut svg,
        24,
        126,
        "note",
        "Signed v3 vs. storing and resampling rows",
    );
    axis(&mut svg, left_axis, 159, 335);
    for (point, y) in [(v3, 191), (&uniform, 276)] {
        let value = scale.value_text(point.trg);
        text(
            &mut svg,
            24,
            y - 21,
            if point.tier == Emphasis::Highlight {
                "lead"
            } else {
                ""
            },
            &point.label,
        );
        let class = if point.tier == Emphasis::Highlight {
            "lead-value lead"
        } else {
            "lead-value"
        };
        text(&mut svg, 281, y - 21, class, &value);
        interval(&mut svg, point, y, left_axis);
    }
    text(&mut svg, 336, 105, "heading", "Third-party generators");
    text(
        &mut svg,
        336,
        126,
        "note",
        "Library defaults with no cap on external files",
    );
    text(&mut svg, 691, 151, "column", "Score");
    text(&mut svg, 783, 151, "column", "Median file");
    text(&mut svg, 964, 151, "column", "DOPE lead [95% CI]");
    axis(&mut svg, right_axis, 159, 335);
    for (i, point) in points.iter().enumerate() {
        let y = 175 + i as u32 * 36;
        text(
            &mut svg,
            336,
            y + 5,
            if point.tier == Emphasis::Highlight {
                "lead"
            } else {
                ""
            },
            &point.label,
        );
        interval(&mut svg, point, y, right_axis);
        text(&mut svg, 691, y + 5, "value", &scale.value_text(point.trg));
        text(&mut svg, 783, y + 5, "column", &compact_size(point.bytes));
        if point.tier == Emphasis::Highlight {
            text(&mut svg, 964, y + 5, "column", "—");
            continue;
        }
        let Some(pair) = receipts.iter().flat_map(|r| &r.comparisons).find(|pair| {
            pair.id == DOPE
                && METHODS_LABELS.contains(&(pair.baseline.as_str(), point.label.as_str()))
        }) else {
            return contract("comparison hero requires a paired receipt for every external arm");
        };
        if pair.cells != rows.informative || pair.ci_low > pair.ci_high {
            return contract("comparison hero inputs disagree on coverage or interval");
        }
        let paired = format!(
            "{:+.1}  [{}, {}]",
            100.0 * pair.delta,
            score_text(100.0 * pair.ci_low),
            score_text(100.0 * pair.ci_high)
        );
        let class = if pair.ci_low > 0.0 { "gain" } else { "column" };
        text(&mut svg, 964, y + 5, class, &paired);
        let outcome = if pair.ci_low > 0.0 {
            "DOPE ahead"
        } else if pair.ci_high < 0.0 {
            "Baseline ahead"
        } else {
            "Inconclusive"
        };
        text(&mut svg, 964, y + 19, "detail", outcome);
        let ratio = bytes_text((point.bytes / v3.bytes).round());
        let ratio = ratio.strip_suffix(" B").unwrap_or(&ratio);
        text(&mut svg, 783, y + 19, "detail", &format!("{ratio}× DOPE"));
    }
    text(
        &mut svg,
        336,
        377,
        "note",
        "DOPE ahead means the paired interval is above zero. See the paper for guard decisions.",
    );
    text(&mut svg, 24, 411, "note", &format!(
        "95% family-bootstrap intervals  ·  100 matches a second real sample  ·  {} informative cells", rows.informative
    ));
    svg.push_str("</svg>\n");
    Ok(svg)
}
