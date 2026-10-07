# Work Order B: five closed paid operations after PR177

Fixed metadata snapshot anchored by the fifth native terminal receipt. The five new physical operations are two OK and three infrastructure outcomes. Actual parent waits are all zero; their scientific statuses remain distinct.

PR177's immutable cutoff is [aaa8e638](https://github.com/neverhuman/dope/commit/aaa8e63857284faca936048339a8b2985cf7796f): 924.4300867351703 parent seconds and 111 paid trials. This delta adds 699.9920627153479 seconds and five paid trials, reaching 1624.4221494505182 seconds and 116 paid trials. Accepted model plus projection bytes add 5813. Failed operations have no accepted artifact charge.

The historical prefit infrastructure cost 18.162802021950483 seconds is already included in the cutoff. Generated qualification 337.20003135968 seconds, original qualification clocks, retained seed11 reuse, and earlier native publications are not charged again.

The released sampler's 1670.3220896930434-second metadata hold had zero scientific entries and zero native trials. It is a separate elapsed hold observation, excluded from the native paid clock. Unpaid admission refusals are not new scientific or paid infrastructure outcomes. Operator, transport, nested operation, and campaign wall clocks are not added to parent time.

Sharing is taken only from actual GPU-clock observations. An empty GPU clock yields unknown sharing even if a receipt's default flag is false. Observed own GPU residency is not a true peak. Hardware model, CPU core seconds, GPU kernel time, energy, true RAM/VRAM peaks, updated logical progress, sample completeness, fivefit stability, method completion, release and superiority remain unavailable.

`delta.json` is the rights-safe scalar ledger. `source-proof.json` binds immutable input hashes and byte counts without private paths or payloads. The frozen strict schemas reject extra fields. `render.py` reads only these two committed JSON files after digest checks and regenerates `costs.csv` and `costs.md`. It does not reopen private receipts, models, samples or data.

Regenerate: `python3 -I -S -B render.py`. Check projections: `python3 -I -S -B render.py --check`.
