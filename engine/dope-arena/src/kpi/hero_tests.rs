//! Unit tests of the `render-hero` drawing: number formats, domains,
//! wrapping, the label placer, and the SVG of a fixture shaped like a real
//! receipt (two coincident DOPE releases, two research arms, a third-party
//! baseline far past the budget, five controls in two tight pairs).

use super::hero_layout::{
    byte_domain, text_width, trg_steps, wrap, Anchor, Candidate, Placer, Rect,
};
use super::hero_model::{
    budget_text, bytes_text, esc, tick_text, trg_text, Hero, Point, Reference, Theme, Tier,
    DEFAULT_TITLE,
};
use super::hero_svg::{draw, render};

fn point(label: &str, bytes: f64, trg: f64, half: f64, tier: Tier) -> Point {
    Point {
        label: label.to_string(),
        bytes,
        trg,
        ci_low: trg - half,
        ci_high: trg + half,
        tier,
    }
}

fn reference(label: &str, trg: f64) -> Reference {
    Reference {
        label: label.to_string(),
        trg,
    }
}

pub fn fixture() -> Hero {
    Hero {
        title: DEFAULT_TITLE.to_string(),
        note: Some(
            "Exploratory panel: 117 families, 125 cells, scored by dope_kpi. TRG 0 = matches the real B sample under both auditors; baselines are not byte-capped."
                .to_string(),
        ),
        points: vec![
            point("DOPE v2.0.0", 1462.0, -0.232, 0.108, Tier::Highlight),
            point("DOPE v1.3.0", 1462.0, -0.234, 0.108, Tier::Previous),
            point("v7 step 0 (PR 390)", 535.0, -0.381, 0.071, Tier::Other),
            point("format2 2 KiB (PR 390)", 1992.0, -0.454, 0.097, Tier::Other),
            point("SDV Gaussian copula", 105_848.0, -0.686, 0.069, Tier::Other),
        ],
        references: vec![
            reference("keep the training rows (copy of A)", -0.030),
            reference("resample the training rows (bootstrap of A)", -0.055),
            reference("fresh real rows (twin sample C)", -0.173),
            reference("shuffled target (label shuffle)", -1.156),
            reference("independent marginals", -1.173),
        ],
        budget: 2048.0,
        footer: "contract 200ea0ed \u{b7} receipt 5c999726".to_string(),
    }
}

/// Minimal XML check: balanced tags, even attribute quotes, and every `&`
/// in character data opening one of the five predefined entities.
fn well_formed(svg: &str) -> Result<(), String> {
    let entity_ok = |text: &str| {
        text.match_indices('&').all(|(i, _)| {
            ["&amp;", "&lt;", "&gt;", "&quot;", "&apos;"]
                .iter()
                .any(|e| text[i..].starts_with(e))
        })
    };
    let mut stack: Vec<String> = Vec::new();
    let mut rest = svg;
    while let Some(open) = rest.find('<') {
        if !entity_ok(&rest[..open]) {
            return Err("bare & in text".to_string());
        }
        let close = rest[open..].find('>').ok_or("unclosed tag")? + open;
        let tag = &rest[open + 1..close];
        rest = &rest[close + 1..];
        if tag.starts_with('?') {
            continue;
        }
        if tag.matches('"').count() % 2 != 0 || !entity_ok(tag) {
            return Err(format!("bad attributes in <{tag}>"));
        }
        if let Some(name) = tag.strip_prefix('/') {
            let top = stack.pop().ok_or("close without open")?;
            if top != name {
                return Err(format!("</{name}> closes <{top}>"));
            }
            continue;
        }
        let name = tag.trim_end_matches('/').split_whitespace().next();
        let name = name.ok_or("empty tag")?.to_string();
        if !tag.ends_with('/') {
            stack.push(name);
        }
    }
    if !rest.trim().is_empty() || !stack.is_empty() {
        return Err(format!("trailing text or open tags {stack:?}"));
    }
    Ok(())
}

fn count(svg: &str, needle: &str) -> usize {
    svg.matches(needle).count()
}

#[test]
fn numbers_and_text_have_fixed_formats() {
    assert_eq!(trg_text(-0.2322), "\u{2212}0.232");
    assert_eq!(trg_text(-0.0004), "0.000");
    assert_eq!(trg_text(0.05), "0.050");
    assert_eq!(tick_text(-1.25), "\u{2212}1.25");
    assert_eq!(tick_text(0.0), "0");
    assert_eq!(tick_text(-0.0), "0");
    assert_eq!(bytes_text(1462.0), "1,462 B");
    assert_eq!(bytes_text(105_848.0), "105,848 B");
    assert_eq!(bytes_text(1462.5), "1,462.5 B");
    assert_eq!(bytes_text(950.0), "950 B");
    assert_eq!(budget_text(2048.0), "2 KiB budget");
    assert_eq!(budget_text(1_048_576.0), "1 MiB budget");
    assert_eq!(budget_text(1500.0), "1,500 B budget");
    assert_eq!(
        esc("a<b & 'c'>\"\u{7}"),
        "a&lt;b &amp; &apos;c&apos;&gt;&quot;"
    );
    assert!(text_width("DOPE", 12.0, true) > text_width("DOPE", 12.0, false));
}

#[test]
fn domains_snap_to_ticks_and_hold_zero() {
    let (lo, hi) = byte_domain(535.0, 2048.0);
    assert_eq!((lo, hi), (100f64.log10(), 10240f64.log10()));
    let (lo, hi) = byte_domain(535.0, 105_848.0);
    assert_eq!((lo, hi), (100f64.log10(), 1_048_576f64.log10()));
    // Beyond the tick range the padded value stands.
    let (lo, _) = byte_domain(50.0, 2048.0);
    assert!((lo - (50f64.log10() - 0.12)).abs() < 1e-12);
    assert_eq!(trg_steps(-1.173, -0.015), (-5, 0));
    assert_eq!(trg_steps(0.1, 0.3), (0, 2));
    assert_eq!(trg_steps(-0.25, -0.25), (-1, 0));
    assert_eq!(trg_steps(0.0, 0.0), (-1, 0));
}

#[test]
fn notes_wrap_within_the_width() {
    let note = fixture().note.unwrap();
    let lines = wrap(&note, 13.0, 400.0);
    assert!(lines.len() > 1);
    assert_eq!(lines.join(" "), note);
    assert!(lines.iter().all(|l| text_width(l, 13.0, false) <= 400.0));
    assert_eq!(wrap("one", 13.0, 1.0), vec!["one".to_string()]);
    let kept = wrap("a TRG\u{a0}0\u{a0}=\u{a0}match", 13.0, 40.0);
    assert_eq!(
        kept,
        vec!["a".to_string(), "TRG\u{a0}0\u{a0}=\u{a0}match".to_string()]
    );
}

#[test]
fn the_placer_takes_the_first_free_box_else_the_least_overlap() {
    let rect = |x: f64, y: f64| Rect {
        x,
        y,
        w: 10.0,
        h: 10.0,
    };
    let at = |x: f64, y: f64| Candidate {
        rect: rect(x, y),
        anchor: Anchor::Start,
    };
    let area = Rect {
        x: 0.0,
        y: 0.0,
        w: 100.0,
        h: 100.0,
    };
    let mut placer = Placer::new(area);
    placer.block(rect(0.0, 0.0));
    // Out of bounds and blocked candidates are skipped.
    let chosen = placer.place(&[at(-5.0, 0.0), at(5.0, 5.0), at(20.0, 0.0)], &[]);
    assert_eq!(chosen.unwrap().rect, rect(20.0, 0.0));
    // Nothing free: the least overlap wins, moved inside the bounds.
    placer.block(Rect {
        x: 0.0,
        y: 0.0,
        w: 100.0,
        h: 100.0,
    });
    let chosen = placer.place(&[at(95.0, 95.0)], &[]);
    assert_eq!(chosen.unwrap().rect, rect(90.0, 90.0));
    // Extra boxes to avoid count like reserved ones.
    let mut placer = Placer::new(Rect {
        x: 0.0,
        y: 0.0,
        w: 100.0,
        h: 100.0,
    });
    let chosen = placer.place(&[at(0.0, 0.0), at(50.0, 0.0)], &[rect(0.0, 0.0)]);
    assert_eq!(chosen.unwrap().rect, rect(50.0, 0.0));
}

#[test]
fn the_fixture_renders_well_formed_and_byte_identical() {
    let hero = fixture();
    let svg = render(&hero, Theme::Light);
    assert_eq!(svg, render(&hero, Theme::Light), "same input, same bytes");
    well_formed(&svg).unwrap();
    assert!(svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg "));
    assert!(svg.contains("role=\"img\"") && svg.contains("viewBox=\"0 0 960 540\""));
    assert!(svg
        .contains("<title id=\"hero-title\">Predictive structure kept vs. artifact size</title>"));
    assert_eq!(count(&svg, "<desc id=\"hero-desc\">"), 1);
    assert_eq!(count(&svg, "class=\"pt "), 5, "one marker per point");
    assert_eq!(count(&svg, "class=\"ci "), 5, "one whisker per point");
    assert_eq!(count(&svg, "class=\"ref\""), 5, "one line per control");
    assert_eq!(count(&svg, "class=\"ref-label\""), 5);
    assert_eq!(count(&svg, "class=\"budget\""), 1);
    assert_eq!(count(&svg, "class=\"label"), 5, "every point labelled");
    assert_eq!(count(&svg, "class=\"pt hl\""), 1);
    assert!(svg.contains("class=\"pt hl\"") && svg.contains("fill=\"#2a78d6\""));
    let highlight = svg
        .lines()
        .find(|l| l.contains("class=\"label hl\""))
        .unwrap();
    assert!(highlight.contains("font-weight=\"600\""));
    assert!(highlight.contains(">DOPE v2.0.0 \u{b7} TRG \u{2212}0.232</text>"));
    // Only the highlight label carries a number.
    let others: Vec<&str> = svg
        .lines()
        .filter(|l| l.contains("class=\"label\""))
        .collect();
    assert_eq!(others.len(), 4);
    assert!(others.iter().all(|l| !l.contains("TRG")));
    assert!(svg.contains(">2 KiB budget</text>") && svg.contains(">1 MiB</text>"));
    assert!(svg.contains(">contract 200ea0ed \u{b7} receipt 5c999726</text>"));
    for kind in ["previous release", "other methods", "controls"] {
        assert!(
            svg.contains(&format!(">{kind}</text>")),
            "{kind} in the key"
        );
    }
}

#[test]
fn every_label_is_inside_the_plot_and_none_overlap() {
    for theme in [Theme::Light, Theme::Dark] {
        let (_, bounds, boxes) = draw(&fixture(), theme);
        assert_eq!(boxes.len(), 11, "budget, five controls, five points");
        for (i, a) in boxes.iter().enumerate() {
            assert!(a.within(&bounds), "label {i} inside the plot");
            for (j, b) in boxes.iter().enumerate().skip(i + 1) {
                assert_eq!(a.overlap(b), 0.0, "labels {i} and {j} overlap");
            }
        }
    }
}

/// The number in attribute `name` of an SVG element line.
fn attr(element: &str, name: &str) -> f64 {
    let start = element.find(&format!(" {name}=\"")).unwrap() + name.len() + 3;
    let end = start + element[start..].find('"').unwrap();
    element[start..end].parse().unwrap()
}

#[test]
fn no_line_runs_through_a_label() {
    let (svg, bounds, boxes) = draw(&fixture(), Theme::Light);
    let budget = svg
        .lines()
        .find(|l| l.contains("class=\"budget\""))
        .unwrap();
    let x = attr(budget, "x1");
    let mut lines = vec![Rect {
        x: x - 1.0,
        y: bounds.y,
        w: 2.0,
        h: bounds.h,
    }];
    for line in svg.lines().filter(|l| l.contains("class=\"ref\"")) {
        let y = attr(line, "y1");
        let w = attr(line, "x2") - attr(line, "x1");
        lines.push(Rect {
            x: attr(line, "x1"),
            y: y - 0.5,
            w,
            h: 1.0,
        });
    }
    assert_eq!(lines.len(), 6);
    for (i, label) in boxes.iter().enumerate() {
        for line in &lines {
            assert_eq!(label.overlap(line), 0.0, "a line crosses label {i}");
        }
    }
}

#[test]
fn the_dark_theme_uses_only_dark_tokens() {
    let dark = render(&fixture(), Theme::Dark);
    well_formed(&dark).unwrap();
    assert!(dark.contains("fill=\"#1a1a19\"") && dark.contains("fill=\"#3987e5\""));
    for light in [
        "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#86b6ef", "#e1e0d9",
    ] {
        assert!(!dark.contains(light), "{light} in the dark theme");
    }
    assert_ne!(dark, render(&fixture(), Theme::Light));
}

#[test]
fn the_budget_is_a_line_not_a_filter() {
    let mut hero = fixture();
    hero.points.retain(|p| p.bytes > 2048.0);
    hero.points[0].tier = Tier::Highlight;
    let svg = render(&hero, Theme::Light);
    well_formed(&svg).unwrap();
    assert_eq!(
        count(&svg, "class=\"pt "),
        1,
        "a point past the budget is drawn"
    );
    assert_eq!(count(&svg, "class=\"budget\""), 1);
    assert!(svg.contains(">100 KiB</text>") && svg.contains(">1 KiB</text>"));
    // No highlight, no previous release and no controls: the key shrinks.
    hero.points[0].tier = Tier::Other;
    hero.references.clear();
    let svg = render(&hero, Theme::Light);
    assert!(!svg.contains(">controls</text>") && !svg.contains(">previous release</text>"));
    assert_eq!(count(&svg, "class=\"ref\""), 0);
}

#[test]
fn labels_and_titles_are_escaped() {
    let mut hero = fixture();
    hero.title = "A <b> & \"c\"".to_string();
    hero.points[2].label = "x < y & z".to_string();
    let svg = render(&hero, Theme::Light);
    well_formed(&svg).unwrap();
    assert!(svg.contains("A &lt;b&gt; &amp; &quot;c&quot;"));
    assert!(svg.contains(">x &lt; y &amp; z</text>"));
}
