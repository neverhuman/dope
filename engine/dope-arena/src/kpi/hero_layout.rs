//! Geometry of `dope_kpi render-hero`: text extents, the axis domains and
//! scales, word wrapping and the deterministic label placer. Standard
//! library only, so the drawing is a pure function of its numbers.

/// Advance widths (1/1000 em) of DejaVu Sans Book for ASCII 32..=126. DejaVu
/// is the widest common `system-ui` fallback, so boxes sized with it leave
/// room for the narrower faces a browser picks.
const REGULAR: [u16; 95] = [
    318, 401, 460, 838, 636, 950, 780, 275, 390, 390, 500, 838, 318, 361, 318, 337, 636, 636, 636,
    636, 636, 636, 636, 636, 636, 636, 337, 337, 838, 838, 838, 531, 1000, 684, 686, 698, 770, 632,
    575, 775, 752, 295, 295, 656, 557, 863, 748, 787, 603, 787, 695, 635, 611, 732, 684, 989, 685,
    611, 685, 390, 337, 390, 838, 500, 500, 613, 635, 550, 635, 615, 352, 635, 634, 278, 278, 579,
    278, 974, 634, 612, 635, 635, 411, 521, 392, 634, 592, 818, 592, 592, 525, 636, 337, 636, 838,
];
/// The same for DejaVu Sans Bold.
const BOLD: [u16; 95] = [
    348, 456, 521, 838, 696, 1002, 872, 306, 457, 457, 523, 838, 380, 415, 380, 365, 696, 696, 696,
    696, 696, 696, 696, 696, 696, 696, 400, 400, 838, 838, 838, 580, 1000, 774, 762, 734, 830, 683,
    683, 821, 837, 372, 372, 775, 637, 995, 837, 850, 733, 850, 770, 720, 682, 812, 774, 1103, 771,
    724, 725, 457, 365, 457, 838, 500, 500, 675, 716, 593, 716, 678, 435, 716, 712, 343, 343, 665,
    343, 1042, 712, 687, 716, 716, 493, 595, 478, 712, 652, 924, 645, 652, 582, 712, 365, 712, 838,
];

fn advance(c: char, bold: bool) -> u16 {
    let table = if bold { &BOLD } else { &REGULAR };
    match c {
        ' '..='~' => table[c as usize - 32],
        // No-break space: as wide as a space.
        '\u{a0}' => table[0],
        '\u{b7}' if bold => 380,
        '\u{b7}' => 318,
        '\u{2013}' => 500,
        // Minus sign, and a full em for anything else.
        '\u{2212}' => 838,
        _ => 1000,
    }
}

/// Estimated advance of `text` at `size` px.
pub fn text_width(text: &str, size: f64, bold: bool) -> f64 {
    let units: f64 = text.chars().map(|c| f64::from(advance(c, bold))).sum();
    units * size / 1000.0
}

/// Height of a one-line label box at `size` px (2 px padding above and below).
pub fn line_box(size: f64) -> f64 {
    size + 4.0
}

/// Baseline of the text in a label box whose top is `top`.
pub fn baseline(top: f64, size: f64) -> f64 {
    top + 2.0 + 0.76 * size
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub fn right(&self) -> f64 {
        self.x + self.w
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.h
    }

    /// Area shared with `other` (0 when they only touch).
    pub fn overlap(&self, other: &Rect) -> f64 {
        let w = self.right().min(other.right()) - self.x.max(other.x);
        let h = self.bottom().min(other.bottom()) - self.y.max(other.y);
        if w > 0.0 && h > 0.0 {
            w * h
        } else {
            0.0
        }
    }

    pub fn within(&self, bounds: &Rect) -> bool {
        self.x >= bounds.x
            && self.y >= bounds.y
            && self.right() <= bounds.right()
            && self.bottom() <= bounds.bottom()
    }

    /// The same box moved the least distance to lie inside `bounds`.
    pub fn moved_into(&self, bounds: &Rect) -> Rect {
        let x = self.x.min(bounds.right() - self.w).max(bounds.x);
        let y = self.y.min(bounds.bottom() - self.h).max(bounds.y);
        Rect { x, y, ..*self }
    }
}

/// Where a label's text is anchored in its box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub rect: Rect,
    pub anchor: Anchor,
}

/// Places label boxes one at a time: the first candidate inside the bounds
/// that overlaps nothing placed or blocked before it, else the candidate
/// (moved inside) with the least overlap, the earliest on ties.
pub struct Placer {
    bounds: Rect,
    taken: Vec<Rect>,
}

impl Placer {
    pub fn new(bounds: Rect) -> Placer {
        Placer {
            bounds,
            taken: Vec::new(),
        }
    }

    /// Reserve `rect` (a marker, a whisker or a label placed by hand).
    pub fn block(&mut self, rect: Rect) {
        self.taken.push(rect);
    }

    fn cost(&self, rect: &Rect, avoid: &[Rect]) -> f64 {
        self.taken
            .iter()
            .chain(avoid)
            .map(|other| rect.overlap(other))
            .sum()
    }

    /// The chosen candidate (moved inside the bounds if none fitted), now
    /// reserved. `avoid` holds boxes only this label must keep clear of.
    pub fn place(&mut self, candidates: &[Candidate], avoid: &[Rect]) -> Option<Candidate> {
        let free = candidates
            .iter()
            .find(|c| c.rect.within(&self.bounds) && self.cost(&c.rect, avoid) == 0.0);
        let chosen = match free {
            Some(c) => *c,
            None => {
                let mut best: Option<(f64, Candidate)> = None;
                for c in candidates {
                    let moved = Candidate {
                        rect: c.rect.moved_into(&self.bounds),
                        anchor: c.anchor,
                    };
                    let cost = self.cost(&moved.rect, avoid);
                    if best.is_none_or(|(least, _)| cost < least) {
                        best = Some((cost, moved));
                    }
                }
                best?.1
            }
        };
        self.taken.push(chosen.rect);
        Some(chosen)
    }
}

/// A linear map from `lo..hi` to pixels `start..end`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub lo: f64,
    pub hi: f64,
    pub start: f64,
    pub end: f64,
}

impl Axis {
    pub fn at(&self, value: f64) -> f64 {
        self.start + (value - self.lo) / (self.hi - self.lo) * (self.end - self.start)
    }
}

/// Byte ticks of the x axis, in bytes, with their labels.
pub const BYTE_TICKS: [(f64, &str); 6] = [
    (100.0, "100 B"),
    (1024.0, "1 KiB"),
    (10240.0, "10 KiB"),
    (102_400.0, "100 KiB"),
    (1_048_576.0, "1 MiB"),
    (10_485_760.0, "10 MiB"),
];
/// Padding, in decades, between the extreme byte values and the domain.
const BYTE_PAD: f64 = 0.12;

/// The log10 domain of the x axis for byte values in `lo..=hi`: padded, then
/// widened to the nearest tick on each side when one exists.
pub fn byte_domain(lo: f64, hi: f64) -> (f64, f64) {
    let low = lo.log10() - BYTE_PAD;
    let high = hi.log10() + BYTE_PAD;
    let ticks = BYTE_TICKS.iter().map(|(bytes, _)| bytes.log10());
    let below = ticks.clone().filter(|t| *t <= low).fold(f64::NAN, f64::max);
    let above = ticks.filter(|t| *t >= high).fold(f64::NAN, f64::min);
    (
        if below.is_nan() { low } else { below },
        if above.is_nan() { high } else { above },
    )
}

/// TRG gridline step.
#[cfg(test)]
pub const TRG_STEP: f64 = 0.25;

/// The y domain for values in `lo..=hi`, in whole gridline steps of `step`,
/// always holding `anchor * step`.
pub fn y_steps(lo: f64, hi: f64, step: f64, anchor: i64) -> (i64, i64) {
    let mut low = ((lo / step).floor() as i64).min(anchor);
    let high = ((hi / step).ceil() as i64).max(anchor);
    if low == high {
        low -= 1;
    }
    (low, high)
}

/// The y domain for TRG values in `lo..=hi`, in whole gridline steps, always
/// holding 0 (a TRG of 0 matches the real B sample).
#[cfg(test)]
pub fn trg_steps(lo: f64, hi: f64) -> (i64, i64) {
    y_steps(lo, hi, TRG_STEP, 0)
}

/// Greedy wrap of `text` at spaces into lines at most `max` px wide at
/// `size` px (a no-break space keeps its words together; a single word wider
/// than `max` keeps its own line).
pub fn wrap(text: &str, size: f64, max: f64) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split(' ').filter(|w| !w.is_empty()) {
        if line.is_empty() {
            line = word.to_string();
            continue;
        }
        let longer = format!("{line} {word}");
        if text_width(&longer, size, false) > max {
            lines.push(std::mem::replace(&mut line, word.to_string()));
        } else {
            line = longer;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}
