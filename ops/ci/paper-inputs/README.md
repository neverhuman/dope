These are immutable source snapshots extracted from b774807, before review
fixes. `paper-prepare.py` authenticates their bytes and materializes them into
`docs/whitepaper/generated/` during a clean rebuild.

`provenance.json` records the historical inventory observation, including
custody gaps. Rebuilding the paper does not make a fresh filesystem observation.
`licenses.json` records the prior hashed license extracts; the existing license
reader can recheck mounted license files. `related.bib` is the static additional
bibliography source. These files are source inputs, not new experimental results.
