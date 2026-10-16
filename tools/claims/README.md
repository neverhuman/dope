# Public claims tooling

`docs/whitepaper/scripts/claim_registry.py` owns the publication rules. Its
historical compiler remains active; `tools/claims/check.py` delegates the new
public checks to the same module.

Run these commands from the repository root:

```sh
python3 tools/claims/registry.py
python3 tools/claims/version.py
python3 tools/claims/render.py
python3 tools/claims/render.py --check
python3 tools/claims/check.py
python3 docs/whitepaper/scripts/claim_registry.py --check
python3 -m unittest discover -s tools/claims -p 'test_*.py'
```

The canonical public result rows are in
`paper/generated/headline-results.json`. Each row binds its historical source,
cohort, byte scope and evidence status. The current source is historical,
selection-exposed CatBoost validation; it supplies no TDL result. The renderer
emits `STATUS.md`, a README block, TeX macros, figure CSV, PNG/PDF/SVG and their
SHA256 manifest. Every image comes from that CSV. These historical points do not
establish an optimized budget frontier, a driver certificate or cold closure.
The figure's threshold line and shading come from the `thresholds` field of
`headline-results.json`. Rendering refuses any point, interval or threshold that would fall
outside the fixed axes, so a value is never clipped silently (`test_figure.py`).
Smaller cohorts remain separate and grey in the figure.

Place `paper/generated/readme-result-card.md` verbatim in the README, including
its `BEGIN TDL RESULTS` / `END TDL RESULTS` markers. Include
`generated/headline-macros.tex` from the canonical paper. The generated
`result-card.tex` supplies its evidence table. Preserve native engineering
examples using the exact `readme-demo-notes.md` block and its PINNED DEMOS markers;
the native receipt and toy INSPECT transcript remain SHA-pinned. Example macro names
are `\HistoricalDopeRetention`, `\HistoricalDopeLower`,
`\HistoricalDopeUpper`, `\HistoricalDopeBytes` and
`\HistoricalDopeLineages`. Result numbers outside generated inputs are refused;
CLI command examples, citation keys and document layout are not result claims.

The firewall rejects non-citable fixtures, unqualified headline auditors,
incomplete or mixed headline cohorts, non-sealed final claims, unmeasured driver
and budget gates, source drift, and generated output/figure drift. Prohibited
claims are checked here and in the existing historical registry. Fixture tests
seed each failure class. The old BENCHMARK marker remains admitted during lane
integration because the historical compiler already verifies that block.

Python package metadata reads the Cargo version through `package_version.py`.
`version.py --write` synchronizes the static citation version; the public gate
checks both. Python retains the `dope-kernel-v1` shim; the native Cargo binary
provides `dope`.
