# STYLE_SPEC — emulate the UNREAL paper layout for the DOPE whitepaper

Style reference for the DOPE whitepaper restyle (2026-10-08). Paths below are relative to the restyle working folder on xbabe2 (~/dope_paper_restyle).

## 0. Which paper is the "reference"?
Jepson's link (`neverhuman/dope/blob/main/docs/whitepaper/dope-mfs.pdf`) is **our own current paper**:
IEEEtran journal, two-column, 15 pp, title "Downstream Objective-Preserving Encoding".
It contains no "UNREAL". The paper he praised ("start with the name UNREAL and then define it",
"plots showing clear value on the front page") is:

> **UNREAL: Unifying Retrieval and Long-Context with a Single Model** — Kinderman, Hoffer, Blau,
> Chmiel, Banner, Soudry, Ginsburg (NVIDIA/Technion), arXiv:2610.08463v1, 6 Oct 2026.
> PDF: https://arxiv.org/pdf/2610.08463  · LaTeX source: https://arxiv.org/src/2610.08463

So: **STYLE = UNREAL (arXiv 2610.08463). CONTENT = docs/whitepaper/dope-mfs.tex (ours).**
Local copies: `ref/unreal-2610.08463.pdf`, `ref/src/` (main.tex, iclr2027_conference.sty/.bst,
fancyhdr.sty, natbib.sty, math_commands.tex, figs/), page renders in `pages/unreal-NN.png`,
current DOPE page renders in `pages/dopecur-NN.png`.

## 1. Document class, page, columns, margins
- `\documentclass{article}` + `\usepackage{iclr2027_conference,times}` + `\iclrfinalcopy`.
- US Letter 612x792 pt. **Single column.** `\textwidth 5.5in`, `\textheight 9.0in`,
  `\oddsidemargin .5in`, `\topmargin -0.625in` (all from iclr2027_conference.sty).
- 10 pt body, generous leading, paragraphs separated by vertical space with no indent (ICLR style).
- Running header removed: UNREAL does `\fancyhead{}` + `\renewcommand{\headrulewidth}{0pt}` after
  `\maketitle`, so no "Published as a conference paper at ICLR" line. **We must do the same; never claim an ICLR venue.**
- Page number centered in footer. 26 pp total (main ≈ 11 pp + refs + appendix).

## 2. Fonts (pdffonts)
- Body: **Times** → NimbusRomNo9L-Regu / -Medi (via `times` package, pdflatex). Bold = Medi.
- Mono: NimbusMonL-Regu (emails, code).
- Math: Computer Modern (CMR/CMMI/CMSY) + MSBM (amssymb). No newtx — math stays CM.
- Figures: matplotlib **DejaVu Sans** (regular + bold panel titles); some Helvetica in drawio diagram.
- Title, abstract heading and section headings set in **small caps** (`\sc`) by the sty.

## 3. Title block and opener (the thing Jepson loves)
- Title: `{\LARGE\sc ...}`, left aligned, two lines: "UNREAL: Unifying Retrieval and Long-Context with a Single Model".
  Pattern = **ACRONYM: expansion-ish descriptive title**.
- Authors bold with superscript affiliation numbers, affiliations line (`$^1$NVIDIA \quad $^2$Technion`),
  then a `\small\texttt{\{a,b,c\}@domain}` email line.
- Abstract heading "ABSTRACT" centered small caps; abstract indented both sides, one paragraph.
- **The definition move:** the abstract says "We introduce **UN**ifying **RE**trieval **A**nd **L**ong-Context
  with a Single Model (UNREAL), a model-native ..." — the letters forming the acronym are **bold**,
  followed immediately by "(UNREAL), a <one-line definition>". Intro §1.1 repeats the same bolded expansion.
- DOPE mirror: title **"DOPE: Downstream Objective-Preserving Encoding for <short claim>"**, and abstract/intro:
  "We introduce **D**ownstream **O**bjective-**P**reserving **E**ncoding (DOPE), a <one-line definition> ..."
  Drop the "D.O.P.E." dotted form.

## 4. Page-1 teaser figure ("clear value on the front page")
- Placed right after the abstract with `\begin{figure}[H]` (float package), `width=1.0\linewidth`,
  then `\newpage` before §1 Introduction. Page 1 = title + abstract + teaser only.
- Teaser: **one row of 3 panels** (one per backbone), each with bold panel title, light grid,
  line + marker series per method, a **dashed vertical divider** splitting regimes ("Long context" | "RAG"),
  and **one shared legend below** all panels (2-3 rows, ~6 entries); "(ours)" method is the visually
  dominant top line (green), baselines in black/grey/pink/orange/blue.
- Caption: "Figure 1: <what is measured>. <setup>. Unlike baselines ..., UNREAL ... maintaining the highest ..."
  — one sentence of value at the end. Caption font = body size, plain (not bold label).

## 5. Sections, headings
- `\section`: `\large\sc`, numbered "1 INTRODUCTION" (small caps). Subsections "1.1 MODEL-INTERNAL ..." small caps, normal size.
- Paragraph-lead style: `\textbf{Full corpus evaluation.}` run-in bold heads everywhere in Results.
- Intro ends with **§1.2 Contributions** as numbered list whose items start with a **bold one-line claim.**
  followed by detail and section pointer "(Section 3)".
- Order: Introduction → Method → (two results sections, each "UNREAL as a ...") → Related Work →
  Limitations → Discussion and Future Directions → References → Appendix. Related Work is late, not §2.

## 6. Figures and tables
- Figures `[t]`, mostly 0.8–1.0 `\linewidth`, matplotlib, DejaVu Sans, white bg, light grid, bar charts with
  95% CI whiskers, method colors consistent across all figures, ours in green.
- Captions start with a **bold one-line takeaway**: "\textbf{UNREAL scales to full-corpus Wikipedia retrieval.} ..."
  then details and "Bars show means with 95% BCa confidence intervals."
- Tables: `booktabs` (`\toprule/\midrule/\cmidrule(lr)/\bottomrule`), `\small`, **caption above**,
  grouped column headers via `\multicolumn` + `\cmidrule`, reference rows (No context/Oracle) separated by `\midrule`,
  our row last with **bold best values**.
- Cross-refs via `cleveref` (`\cref` → "Fig. 1", "Table 2", "Section 3"); `subcaption`, `enumitem`, `xcolor`.

## 7. Colors, links, references
- `hyperref` with `colorlinks=true, urlcolor=blue`; default citecolor (**green** author-year citations),
  default linkcolor (red section/figure numbers).
- Citations natbib **author–year**: `\citep{}` → "(Lewis et al., 2020)", `\citet{}` in text.
  Bibliography style `iclr2027_conference.bst` (alphabetical, author-year). Our IEEE `\cite{}`
  numeric style must be converted to `\citep`/`\citet`.

## 8. What the current DOPE paper does differently (to change)
IEEEtran two-column journal; Times-like IEEE font; centered title w/o acronym opener; bold abstract in
spanning box; IEEE Index Terms; Roman-numeral small-caps sections ("I. INTRODUCTION"); numeric [n] citations;
no page-1 figure; figure* wide floats; running header "TAYLOR AND ALEXANDER".
