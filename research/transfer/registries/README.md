# TDL candidate registries

`auditors.json`, `generator-panel.json` and `../manifests/datasets.json` are
unfrozen TDL scaffolding. Every model is specified and unqualified. Declared
upstream licenses are not clearance, and no qualification/download campaign is
performed here. PMLB belongs to MechanismLab only. Unknown rights,
contamination, source identity and split identity exclude dataset candidates
from headline evidence.

The older `models.json`, `generators.json` and source-family registry remain
historical adapter compatibility inputs. They grant no TDL admission.

The authoritative typed lock migration is `lock-states.json`. It maps every
legacy lock to its exact SHA256 and records model/version, source/code/weight
identity, licenses, tasks, mode/context, preprocessing, parity/resource evidence,
admission, rejection reason and verification date. Unmeasured fields remain
null. Legacy lock bytes and their closed decoder formats are retained. This
preserves failed attempts and existing adapter refusals while providing one
TDL state machine:

`specified → source_verified → weights_verified → environment_verified → parity_verified → resource_verified → qualified`

Run `python3 tools/claims/registry.py --migrate` to map a newly added legacy lock.
The migration is idempotent, refuses changed lock hashes, and preserves existing
state. `registry.advance` returns a new entry and refuses skipped transitions,
missing evidence and missing receipt digests. Each transition binds its own
receipt. Until the last transition, admission stays unqualified. Qualification
requires license/contamination review digests and a verification date as well
as source, weights, environment, native parity and resources. No manually
reviewed historical adapter status is a substitute for these gates.

`python3 tools/claims/registry.py` validates all three JSON schemas, candidate
states, and every existing lock mapping. D-07 remains an owner decision; neither
these files nor fixtures authorize acquiring weights or opening sealed data.
