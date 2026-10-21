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
The hero has two explicitly separate cohorts: the matched classical size/transfer
panel uses `legacy-v2-catboost.csv`, while the foundation-model bars use
`hero-frontier.csv`. Both CSVs are emitted in the same render from authenticated
inputs. The classical panel includes the untuned default and all comparable historical
generator configurations; smaller cohorts stay in the full evidence card. Its size ratio
is a ratio of cohort medians, not a paired compression estimate or proof of equivalence.
Rendering refuses points and intervals outside the fixed axes, missing/duplicate methods,
and mixed classical cohorts (`test_figure.py`). Proposed TDL thresholds are checked
against the contract but are not drawn as achieved gates.

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
