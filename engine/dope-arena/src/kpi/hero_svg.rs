//! The README hook graphic of `dope_kpi render-hero`: TRG against artifact
//! bytes as a deterministic SVG (960 x 540, light or dark theme).
//!
//! x is an arm's median emitted-file size on a log10 axis and y its TRG,
//! with the 95% interval as a thin whisker and gridlines every 0.25. The
//! highlighted arm wears the accent and a bold label. On the Transfer Score
//! scale every generator's label carries its score; on the TRG scale only the
//! highlight does. Earlier DOPE releases wear a lighter step of the accent and
//! every other method neutral gray. Every point has a direct label and a small key names the
//! four kinds of mark. Controls without bytes are dotted horizontal reference
//! lines labelled at the right end, and the headline byte budget is a dashed
//! vertical line: a reference, not a filter. Coordinates have one decimal and
//! arms keep their input order, so the same numbers give the same bytes.
//! Nothing but labels, numbers and digests is written: no path, host, source
//! value or header.

use super::hero_labels::{place, Guides, Label, Mark};
use super::hero_layout::{
    baseline, byte_domain, text_width, wrap, y_steps, Anchor, Axis, Rect, BYTE_TICKS,
};
use super::hero_model::{coord, describe, esc, point_title, Emphasis, Hero, Theme, Tokens, YScale};

pub const WIDTH: f64 = 960.0;
pub const HEIGHT: f64 = 540.0;
const MARGIN: f64 = 32.0;
const FONT: &str = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const TITLE_SIZE: f64 = 22.0;
const NOTE_SIZE: f64 = 13.0;
const SMALL: f64 = 12.0;
const FOOT_SIZE: f64 = 10.0;
/// Space between the plot edge and the outermost gridline, kept for labels.
const INNER_PAD: f64 = 18.0;
const X_TITLE: &str = "Artifact bytes: median emitted file (log scale)";

/// Style of one text element.
#[derive(Clone, Copy)]
struct Style {
    size: f64,
    fill: &'static str,
    bold: bool,
    anchor: Anchor,
    /// A surface-coloured outline that keeps text legible over lines.
    halo: bool,
}

impl Style {
    fn plain(size: f64, fill: &'static str) -> Style {
        Style {
            size,
            fill,
            bold: false,
            anchor: Anchor::Start,
            halo: false,
        }
    }

    fn anchored(self, anchor: Anchor) -> Style {
        Style { anchor, ..self }
    }
}

struct Svg {
    out: String,
    surface: &'static str,
}

impl Svg {
    fn push(&mut self, element: String) {
        self.out.push_str(&element);
        self.out.push('\n');
    }

    fn line(&mut self, class: &str, from: (f64, f64), to: (f64, f64), stroke: &str, dash: &str) {
        let dash = if dash.is_empty() {
            String::new()
        } else {
            format!(" stroke-dasharray=\"{dash}\"")
        };
        self.push(format!(
            "<line class=\"{class}\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"1\"{dash}/>",
            coord(from.0),
            coord(from.1),
            coord(to.0),
            coord(to.1)
        ));
    }

    /// A filled marker with a 2 px ring in the surface colour.
    fn dot(&mut self, class: &str, cx: f64, cy: f64, r: f64, fill: &str) {
        self.push(format!(
            "<circle class=\"{class}\" cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{fill}\" stroke=\"{}\" stroke-width=\"4\" paint-order=\"stroke\"/>",
            coord(cx),
            coord(cy),
            coord(r),
            self.surface
        ));
    }

    fn text(&mut self, class: &str, at: (f64, f64), style: Style, text: &str) {
        let mut attrs = format!(
            "class=\"{class}\" x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"{}\"",
            coord(at.0),
            coord(at.1),
            coord(style.size),
            style.fill
        );
        if style.bold {
            attrs.push_str(" font-weight=\"600\"");
        }
        match style.anchor {
            Anchor::Start => {}
            Anchor::Middle => attrs.push_str(" text-anchor=\"middle\""),
            Anchor::End => attrs.push_str(" text-anchor=\"end\""),
        }
        if style.halo {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"3\" stroke-linejoin=\"round\" paint-order=\"stroke\"",
                self.surface
            ));
        }
        self.push(format!("<text {attrs}>{}</text>", esc(text)));
    }

    fn label(&mut self, label: &Label) {
        let rect = label.at.rect;
        let x = match label.at.anchor {
            Anchor::Start => rect.x,
            Anchor::Middle => rect.x + rect.w / 2.0,
            Anchor::End => rect.right(),
        };
        let style = Style {
            size: label.size,
            fill: label.ink,
            bold: label.bold,
            anchor: label.at.anchor,
            halo: true,
        };
        self.text(
            label.class,
            (x, baseline(rect.y, label.size)),
            style,
            &label.text,
        );
    }
}

enum KeyMark {
    Dot(&'static str, f64),
    Dash,
}

impl KeyMark {
    fn width(&self) -> f64 {
        match self {
            KeyMark::Dot(_, r) => 2.0 * r,
            KeyMark::Dash => 16.0,
        }
    }
}

/// The key: highlight, earlier releases, other methods, controls (those present).
fn key_items(hero: &Hero, t: &Tokens) -> Vec<(KeyMark, String)> {
    let mut items = Vec::new();
    if let Some(p) = hero.points.iter().find(|p| p.tier == Emphasis::Highlight) {
        items.push((KeyMark::Dot(t.accent, 5.0), p.label.clone()));
    }
    if hero.points.iter().any(|p| p.tier == Emphasis::Previous) {
        let text = "previous release".to_string();
        items.push((KeyMark::Dot(t.accent_light, 4.0), text));
    }
    if hero.points.iter().any(|p| p.tier == Emphasis::Other) {
        items.push((KeyMark::Dot(t.gray, 4.0), "other methods".to_string()));
    }
    if !hero.references.is_empty() {
        items.push((KeyMark::Dash, "controls".to_string()));
    }
    items
}

fn key_width(items: &[(KeyMark, String)]) -> f64 {
    let marks: f64 = items
        .iter()
        .map(|(mark, text)| mark.width() + 6.0 + text_width(text, SMALL, false))
        .sum();
    marks + 16.0 * items.len().saturating_sub(1) as f64
}

/// Pixel layout of the graphic.
struct Frame {
    note: Vec<String>,
    title_base: f64,
    key_x: f64,
    key_base: f64,
    y_title_base: f64,
    y_scale: YScale,
    x0: f64,
    x1: f64,
    top: f64,
    bottom: f64,
    steps: (i64, i64),
    xs: Axis,
    ys: Axis,
}

fn extent(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    })
}

impl Frame {
    fn new(hero: &Hero, key_w: f64) -> Frame {
        let title_base = 42.0;
        let y_scale = hero.y_scale;
        let note = hero
            .note
            .as_deref()
            .map(|n| wrap(n, NOTE_SIZE, WIDTH - 2.0 * MARGIN))
            .unwrap_or_default();
        let header = match note.len() {
            0 => title_base,
            n => title_base + 24.0 + 18.0 * (n - 1) as f64,
        };
        let key_x = WIDTH - MARGIN - key_w;
        let key_base = header + 28.0;
        // The y title shares the key row unless the key reaches it.
        let shared = key_x >= MARGIN + text_width(y_scale.title(), SMALL, false) + 24.0;
        let y_title_base = if shared { key_base } else { key_base + 18.0 };
        let (top, bottom) = (y_title_base + 14.0, HEIGHT - 64.0);
        let y_vals = hero.points.iter().flat_map(|p| {
            [
                y_scale.convert(p.trg),
                y_scale.convert(p.ci_low),
                y_scale.convert(p.ci_high),
            ]
        });
        let (lo, hi) = extent(y_vals.chain(hero.references.iter().map(|r| y_scale.convert(r.trg))));
        let step = y_scale.step();
        let steps = y_steps(lo, hi, step, y_scale.anchor_step());
        let tick_w = (steps.0..=steps.1)
            .map(|k| text_width(&y_scale.tick_label(k as f64 * step), SMALL, false))
            .fold(0.0, f64::max);
        let (x0, x1) = (MARGIN + tick_w + 10.0, WIDTH - MARGIN);
        let ys = Axis {
            lo: steps.0 as f64 * step,
            hi: steps.1 as f64 * step,
            start: bottom - INNER_PAD,
            end: top + INNER_PAD,
        };
        let (b_lo, b_hi) = extent(hero.points.iter().map(|p| p.bytes).chain([hero.budget]));
        let (lo, hi) = byte_domain(b_lo, b_hi);
        let xs = Axis {
            lo,
            hi,
            start: x0,
            end: x1,
        };
        Frame {
            note,
            title_base,
            key_x,
            key_base,
            y_title_base,
            y_scale,
            x0,
            x1,
            top,
            bottom,
            steps,
            xs,
            ys,
        }
    }
}

/// Gridlines with their values, then the x axis with its byte ticks.
fn axes(svg: &mut Svg, f: &Frame, t: &Tokens) {
    let tick = Style::plain(SMALL, t.ink2);
    let step = f.y_scale.step();
    for k in f.steps.0..=f.steps.1 {
        let value = k as f64 * step;
        let y = f.ys.at(value);
        svg.line("grid", (f.x0, y), (f.x1, y), t.grid, "");
        let at = (f.x0 - 8.0, y + 4.0);
        svg.text(
            "tick",
            at,
            tick.anchored(Anchor::End),
            &f.y_scale.tick_label(value),
        );
    }
    svg.line("axis", (f.x0, f.bottom), (f.x1, f.bottom), t.axis, "");
    for (value, label) in BYTE_TICKS {
        let at = value.log10();
        if at < f.xs.lo - 1e-9 || at > f.xs.hi + 1e-9 {
            continue;
        }
        let x = f.xs.at(at);
        svg.line("tick", (x, f.bottom), (x, f.bottom + 5.0), t.axis, "");
        let style = tick.anchored(Anchor::Middle);
        svg.text("tick", (x, f.bottom + 19.0), style, label);
    }
}

fn marks(hero: &Hero, f: &Frame) -> Vec<Mark> {
    let mark = |p: &super::hero_model::Point| {
        let r = if p.tier == Emphasis::Highlight {
            5.0
        } else {
            4.0
        };
        let cy = f.ys.at(f.y_scale.convert(p.trg));
        Mark {
            cx: f.xs.at(p.bytes.log10()),
            cy,
            r,
            top: f.ys.at(f.y_scale.convert(p.ci_high)).min(cy - r - 2.0),
            bottom: f.ys.at(f.y_scale.convert(p.ci_low)).max(cy + r + 2.0),
        }
    };
    hero.points.iter().map(mark).collect()
}

/// Whiskers and markers, each point in a group with its tooltip; the highlight last.
fn points(svg: &mut Svg, hero: &Hero, marks: &[Mark], f: &Frame, t: &Tokens) {
    let mut order: Vec<usize> = (0..hero.points.len()).collect();
    order.sort_by_key(|&i| hero.points[i].tier);
    for i in order {
        let (p, m) = (&hero.points[i], &marks[i]);
        let (color, class) = (p.tier.color(t), p.tier.class());
        svg.push(format!(
            "<g><title>{}</title>",
            esc(&point_title(p, f.y_scale))
        ));
        svg.push(format!(
            "<line class=\"ci {class}\" x1=\"{x}\" y1=\"{}\" x2=\"{x}\" y2=\"{}\" stroke=\"{color}\" stroke-width=\"1.5\" stroke-linecap=\"round\"/>",
            coord(f.ys.at(f.y_scale.convert(p.ci_high))),
            coord(f.ys.at(f.y_scale.convert(p.ci_low))),
            x = coord(m.cx)
        ));
        svg.dot(&format!("pt {class}"), m.cx, m.cy, m.r, color);
        svg.push("</g>".to_string());
    }
}

/// Title, note, key, axis titles and the small print.
fn furniture(svg: &mut Svg, hero: &Hero, f: &Frame, key: &[(KeyMark, String)], t: &Tokens) {
    let title = Style {
        bold: true,
        ..Style::plain(TITLE_SIZE, t.ink)
    };
    svg.text("title", (MARGIN, f.title_base), title, &hero.title);
    for (i, line) in f.note.iter().enumerate() {
        let y = f.title_base + 24.0 + 18.0 * i as f64;
        svg.text("note", (MARGIN, y), Style::plain(NOTE_SIZE, t.ink2), line);
    }
    let small = Style::plain(SMALL, t.ink2);
    let mut x = f.key_x;
    for (mark, text) in key {
        let y = f.key_base - 4.0;
        match mark {
            KeyMark::Dot(fill, r) => svg.dot("key", x + r, y, *r, fill),
            KeyMark::Dash => svg.line("key", (x, y), (x + 16.0, y), t.muted, "2 3"),
        }
        let text_x = x + mark.width() + 6.0;
        svg.text("key", (text_x, f.key_base), small, text);
        x = text_x + text_width(text, SMALL, false) + 16.0;
    }
    svg.text(
        "axis-title",
        (MARGIN, f.y_title_base),
        small,
        f.y_scale.title(),
    );
    let at = ((f.x0 + f.x1) / 2.0, f.bottom + 38.0);
    svg.text("axis-title", at, small.anchored(Anchor::Middle), X_TITLE);
    let foot = Style::plain(FOOT_SIZE, t.muted).anchored(Anchor::End);
    svg.text("footer", (f.x1, HEIGHT - 10.0), foot, &hero.footer);
}

/// The SVG document for `hero` in `theme`. `hero.points` must not be empty
/// and every byte value must be positive.
pub fn render(hero: &Hero, theme: Theme) -> String {
    draw(hero, theme).0
}

/// The document, the box labels must stay in and the box of every label.
pub(super) fn draw(hero: &Hero, theme: Theme) -> (String, Rect, Vec<Rect>) {
    let t = theme.tokens();
    let key = key_items(hero, t);
    let f = Frame::new(hero, key_width(&key));
    let mut svg = Svg {
        out: String::new(),
        surface: t.surface,
    };
    svg.push("<?xml version=\"1.0\" encoding=\"UTF-8\"?>".to_string());
    svg.push(format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{WIDTH}\" height=\"{HEIGHT}\" viewBox=\"0 0 {WIDTH} {HEIGHT}\" role=\"img\" aria-labelledby=\"hero-title hero-desc\" font-family=\"{FONT}\">"
    ));
    svg.push(format!(
        "<title id=\"hero-title\">{}</title>",
        esc(&hero.title)
    ));
    svg.push(format!(
        "<desc id=\"hero-desc\">{}</desc>",
        esc(&describe(hero))
    ));
    svg.push(format!(
        "<rect class=\"bg\" width=\"{WIDTH}\" height=\"{HEIGHT}\" fill=\"{}\"/>",
        t.surface
    ));
    axes(&mut svg, &f, t);
    let budget_x = f.xs.at(hero.budget.log10());
    svg.line(
        "budget",
        (budget_x, f.top),
        (budget_x, f.bottom),
        t.ink2,
        "4 3",
    );
    let ref_y: Vec<f64> = hero
        .references
        .iter()
        .map(|r| f.ys.at(f.y_scale.convert(r.trg)))
        .collect();
    for y in &ref_y {
        svg.line("ref", (f.x0, *y), (f.x1, *y), t.muted, "2 3");
    }
    let marks = marks(hero, &f);
    points(&mut svg, hero, &marks, &f, t);
    let bounds = Rect {
        x: f.x0 + 2.0,
        y: f.top,
        w: f.x1 - f.x0 - 2.0,
        h: f.bottom - f.top - 2.0,
    };
    let guides = Guides {
        bounds,
        budget_x,
        top: f.top,
        ref_y,
    };
    let labels = place(hero, t, &marks, &guides);
    for label in &labels {
        svg.label(label);
    }
    furniture(&mut svg, hero, &f, &key, t);
    svg.push("</svg>".to_string());
    let boxes = labels.iter().map(|l| l.at.rect).collect();
    (svg.out, bounds, boxes)
}
