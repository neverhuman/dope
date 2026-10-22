"""Dependency-free deterministic Elo forest plot; text is escaped for SVG."""
from html import escape
from elo_math import LABELS


def svg(result: dict) -> str:
    rows = result["leaderboard"]
    w, h = 1100, 170 + 58 * len(rows)
    lo = min(1300., min(r["lo"] for r in rows) - 70)
    hi = max(1700., max(r["hi"] for r in rows) + 100)
    x = lambda v: 240 + 650 * (v - lo) / (hi - lo)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
           '<title id="title">Budgeted transfer Elo on a common dataset cohort</title>',
           '<desc id="desc">Batch Bradley–Terry ratings with 95 percent dataset bootstrap intervals. Selection-exposed descriptive evidence.</desc>',
           '<rect width="100%" height="100%" fill="#fcfcfb"/>',
           '<g font-family="DejaVu Sans, sans-serif" fill="#242424">',
           '<text x="30" y="34" font-size="25" font-weight="bold">Predictive information kept across byte budgets</text>',
           f'<text x="30" y="61" font-size="15">BTR-AUC → Elo · {len(result["common_datasets"])} common datasets · 95% dataset bootstrap intervals</text>',
           '<text x="30" y="85" font-size="13" fill="#666">Development test envelope · 1 KiB–64 MiB default grid · not a sealed study</text>']
    # The chosen grid, not the default, must be visible for custom studies.
    out[-1] = f'<text x="30" y="85" font-size="13" fill="#666">Development test envelope · {result["budgets_bytes"][0]:,}–{result["budgets_bytes"][-1]:,} bytes · not a sealed study</text>'
    for tick in range(int(lo // 100 + 1) * 100, int(hi) + 1, 100):
        out += [f'<line x1="{x(tick):.2f}" x2="{x(tick):.2f}" y1="105" y2="{h-55}" stroke="#e0e0dd"/>',
                f'<text x="{x(tick):.2f}" y="{h-30}" text-anchor="middle" font-size="13">{tick}</text>']
    for i, r in enumerate(rows):
        y, color = 128 + 58 * i, "#2a78d6" if r["method"] == "dope" else "#777770"
        label = ('<tspan x="215" dy="-9">Compressed original</tspan>'
                 '<tspan x="215" dy="18">training rows</tspan>' if r["method"] == "rows"
                 else escape(LABELS.get(r["method"], r["method"])))
        out += [f'<text x="215" y="{y+5}" text-anchor="end" font-size="17">{label}</text>',
                f'<line x1="{x(r["lo"]):.2f}" x2="{x(r["hi"]):.2f}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>',
                f'<circle cx="{x(r["elo"]):.2f}" cy="{y}" r="7" fill="{color}" stroke="#fcfcfb" stroke-width="2"/>',
                f'<text x="920" y="{y+5}" font-size="15">{r["elo"]:.0f} [{r["lo"]:.0f}, {r["hi"]:.0f}]</text>']
    out += [f'<text x="30" y="{h-6}" font-size="12" fill="#666">1500 = cohort center · local per-dataset rankings and coverage are in elo.md / datasets.csv</text>', '</g></svg>']
    return "\n".join(out) + "\n"


def dataset_svg(result: dict) -> str:
    methods = [r["method"] for r in result["leaderboard"]]
    datasets = result["per_dataset"]
    w, h = 220 + 105 * len(methods), 145 + 34 * len(datasets)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">',
           '<title id="title">Local Elo by dataset</title>',
           '<desc id="desc">Compare methods within each dataset row. Missing alternatives are not rated. Original rows are training examples stored with lossless xz compression.</desc>',
           '<rect width="100%" height="100%" fill="#fcfcfb"/>',
           '<g font-family="DejaVu Sans, sans-serif" fill="#242424">',
           '<text x="25" y="30" font-size="23" font-weight="bold">Local Elo by dataset</text>',
           '<text x="25" y="54" font-size="13">Compare within rows · 1500 = local center · — = incomplete / unqualified</text>']
    for j, m in enumerate(methods):
        label = {"dope": "DOPE", "rows": "Original rows", "kernel": "Kernel", "copula": "Copula"}.get(m, m.upper())
        out.append(f'<text x="{242+105*j}" y="85" font-size="14" text-anchor="middle">{escape(label)}</text>')
    for i, (ds, cells) in enumerate(datasets.items()):
        y = 98 + 34 * i
        out.append(f'<text x="185" y="{y+20}" text-anchor="end" font-size="14">{escape(ds)}</text>')
        for j, m in enumerate(methods):
            e = cells[m].get("elo")
            if e is None:
                color, label = "#efefeb", "—"
            else:
                alpha = min(1., abs(e - 1500) / 400)
                endpoint = (115, 169, 223) if e >= 1500 else (234, 183, 131)
                rgb = [round(248 + alpha * (c - 248)) for c in endpoint]
                color, label = "#" + "".join(f"{c:02x}" for c in rgb), f"{e:.0f}"
            out += [f'<rect x="{197+105*j}" y="{y}" width="99" height="29" rx="3" fill="{color}"/>',
                    f'<text x="{246+105*j}" y="{y+20}" text-anchor="middle" font-size="14">{label}</text>']
    out += [f'<text x="25" y="{h-10}" font-size="12">Descriptive BTR-AUC matches · opponent sets differ where methods are missing · no sealed-test claim</text>', '</g></svg>']
    return "\n".join(out) + "\n"
