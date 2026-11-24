//! Direct labels of `dope_kpi render-hero`. Every point marker and whisker
//! is reserved first; then the budget label, the control labels (in their
//! command-line order) and the point labels (highlight, earlier releases,
//! the rest) are placed one at a time on the first free candidate box:
//! right, left, above and below a point, then right and left one to three
//! rows up or down; at the right end of a control line, above or below it,
//! sliding left in 24 px steps. Point labels keep off every control line and
//! the budget line, control labels off the other lines, so no line runs
//! through text. A label that finds no free box takes the one that overlaps
//! least.

use super::hero_layout::{line_box, text_width, Anchor, Candidate, Placer, Rect};
use super::hero_model::{budget_text, Emphasis, Hero, Tokens, YScale};

pub const HIGHLIGHT_SIZE: f64 = 13.0;
pub const LABEL_SIZE: f64 = 12.0;
pub const REF_SIZE: f64 = 11.0;

/// A point's marker and whisker, in pixels (`top` and `bottom` clear the ring).
pub struct Mark {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub top: f64,
    pub bottom: f64,
}

impl Mark {
    fn ring_box(&self) -> Rect {
        let r = self.r + 2.0;
        Rect {
            x: self.cx - r,
            y: self.cy - r,
            w: 2.0 * r,
            h: 2.0 * r,
        }
    }

    fn whisker_box(&self) -> Rect {
        Rect {
            x: self.cx - 1.5,
            y: self.top,
            w: 3.0,
            h: self.bottom - self.top,
        }
    }
}

pub struct Label {
    pub text: String,
    pub at: Candidate,
    pub size: f64,
    pub bold: bool,
    pub ink: &'static str,
    pub class: &'static str,
}

/// Where the guides of the plot are.
pub struct Guides {
    /// Every label stays inside.
    pub bounds: Rect,
    pub budget_x: f64,
    /// Top of the plot, where the budget label sits.
    pub top: f64,
    /// y of each control line, in `hero.references` order.
    pub ref_y: Vec<f64>,
}

fn box_at(x: f64, y: f64, w: f64, h: f64, anchor: Anchor) -> Candidate {
    Candidate {
        rect: Rect { x, y, w, h },
        anchor,
    }
}

fn reference_candidates(guides: &Guides, y: f64, w: f64, h: f64) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut right = guides.bounds.right() - 2.0;
    loop {
        for top in [y - 2.0 - h, y + 2.0] {
            candidates.push(box_at(right - w, top, w, h, Anchor::End));
        }
        right -= 24.0;
        if right - w < guides.bounds.x + 2.0 {
            return candidates;
        }
    }
}

fn point_candidates(m: &Mark, budget_x: f64, w: f64, h: f64) -> Vec<Candidate> {
    let side = |dy: f64| {
        let y = m.cy - h / 2.0 + dy;
        [
            box_at(m.cx + m.r + 6.0, y, w, h, Anchor::Start),
            box_at(m.cx - m.r - 6.0 - w, y, w, h, Anchor::End),
        ]
    };
    let mut candidates = side(0.0).to_vec();
    candidates.push(box_at(
        m.cx - w / 2.0,
        m.top - 3.0 - h,
        w,
        h,
        Anchor::Middle,
    ));
    candidates.push(box_at(m.cx - w / 2.0, m.bottom + 3.0, w, h, Anchor::Middle));
    for k in 1..=3 {
        let dy = f64::from(k) * (h + 2.0);
        candidates.extend(side(-dy));
        candidates.extend(side(dy));
    }
    // If a point sits right beside the vertical budget line, allow clearing it cleanly.
    for dy in [0.0, -(h + 2.0), h + 2.0] {
        let y = m.cy - h / 2.0 + dy;
        candidates.push(box_at(budget_x + 5.0, y, w, h, Anchor::Start));
        candidates.push(box_at(budget_x - 5.0 - w, y, w, h, Anchor::End));
    }
    candidates
}

/// Every label of `hero`; `marks` follow `hero.points`.
pub fn place(hero: &Hero, t: &Tokens, marks: &[Mark], guides: &Guides) -> Vec<Label> {
    let mut placer = Placer::new(guides.bounds);
    for m in marks {
        placer.block(m.ring_box());
        placer.block(m.whisker_box());
    }
    let mut labels = Vec::new();
    let budget = budget_text(hero.budget);
    let (w, h) = (text_width(&budget, REF_SIZE, false), line_box(REF_SIZE));
    let bx = guides.budget_x;
    let budget_at = [
        box_at(bx + 5.0, guides.top, w, h, Anchor::Start),
        box_at(bx - 5.0 - w, guides.top, w, h, Anchor::End),
    ];
    if let Some(at) = placer.place(&budget_at, &[]) {
        labels.push(Label {
            text: budget,
            at,
            size: REF_SIZE,
            bold: false,
            ink: t.ink2,
            class: "budget-label",
        });
    }
    let (x0, x1) = (guides.bounds.x, guides.bounds.right());
    let lines: Vec<Rect> = guides
        .ref_y
        .iter()
        .map(|y| Rect {
            x: x0,
            y: y - 0.5,
            w: x1 - x0,
            h: 1.0,
        })
        .collect();
    let budget_line = Rect {
        x: bx - 1.0,
        y: guides.bounds.y,
        w: 2.0,
        h: guides.bounds.h,
    };
    for (i, r) in hero.references.iter().enumerate() {
        let (w, h) = (text_width(&r.label, REF_SIZE, false), line_box(REF_SIZE));
        let candidates = reference_candidates(guides, guides.ref_y[i], w, h);
        let avoid: Vec<Rect> = (0..lines.len())
            .filter(|j| *j != i)
            .map(|j| lines[j])
            .chain([budget_line])
            .collect();
        if let Some(at) = placer.place(&candidates, &avoid) {
            labels.push(Label {
                text: r.label.clone(),
                at,
                size: REF_SIZE,
                bold: false,
                ink: t.ink2,
                class: "ref-label",
            });
        }
    }
    let avoid: Vec<Rect> = lines.iter().copied().chain([budget_line]).collect();
    let mut order: Vec<usize> = (0..hero.points.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(hero.points[i].tier));
    for i in order {
        let p = &hero.points[i];
        let highlight = p.tier == Emphasis::Highlight;
        // On the Transfer Score chart every generator carries its score, so the
        // ranking is readable without a second table. The TRG chart still names
        // only the highlighted arm.
        let (text, size) = if highlight || hero.y_scale == YScale::TransferScore {
            (
                hero.y_scale.highlight_label(&p.label, p.trg),
                if highlight { HIGHLIGHT_SIZE } else { LABEL_SIZE },
            )
        } else {
            (p.label.clone(), LABEL_SIZE)
        };
        let (w, h) = (text_width(&text, size, highlight), line_box(size));
        if let Some(at) = placer.place(&point_candidates(&marks[i], bx, w, h), &avoid) {
            labels.push(Label {
                text,
                at,
                size,
                bold: highlight,
                ink: if p.tier == Emphasis::Other {
                    t.ink2
                } else {
                    t.ink
                },
                class: if highlight { "label hl" } else { "label" },
            });
        }
    }
    labels
}
