# Original metadata supporting the paper

This snapshot repairs TRACE-01 in the historical [paper audit](../paper-receipt-redteam-20261008-v1/report.json). The source bytes were observed on 8 October 2026; this does **not** reconstruct an earlier frozen custody chain.

- `replay/`: 100 lineages × two budgets, each with a byte-exact fit receipt, compile report, and deterministic gzip of the three numeric loss-log columns.
- `beyond/`: five families × two budgets with the same metadata.
- `filename-inventory.json`: directory identities and CSV-name presence. No CSV contents, including sealed tests, were opened or hashed.
- `beyond-prepare-projection.json`: safe scalar preparation outcomes, bound to the original private summary hash. Target headers, source values, raw failure details and traces are excluded. Its original private document is not committed.
- `index.json`: source namespaces, capture date, compressed and original byte lengths and SHA-256 digests; pinned by `docs/whitepaper/scripts/paper_receipts.py`.

`docs/whitepaper/generated/original-field-map.json` maps the original file and field used for every replay cost and BeyondArena value. Per-step replay medians and bootstrap bands are regenerated from these numeric logs by `render_figures.py`, including on machines without campaign scratch. The logs describe auxiliary replay fits and **do not replace the scored artifacts**. Artifact byte counts are original receipt fields; this publication does not rehash model weights.

Run `python3 docs/whitepaper/scripts/local_receipts.py --check` to authenticate all original metadata and compare the reduced extracts. Run `just paper-check` to regenerate the loss-curve PDF and bootstrap summary and compare their exact hashes, alongside the manuscript checks.

The source reports retain failed release gates. Official tests remain sealed; MFS-v2, PTF-v1 and production certification are not measured by this repair.
