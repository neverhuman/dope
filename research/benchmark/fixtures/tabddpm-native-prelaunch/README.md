# Frozen TabDDPM prelaunch fixtures

These are exact copies of the standard-library-only `common.py` and `entry.py`
from the native research round frozen at SHA-256
`95ef4ac804c5132fb2e1f63bda4443f522bd257886863c2a0a00e70d2f17d180`.
They contain no rows, models, credentials, or third-party implementation.
The CPU and GPU entry files in that round are identical.

The tests pin both fixture hashes, create a miniature local runtime and worker
under the lane's `target/`, and invoke the actual entry code with isolated Python
startup. Dependency canaries prove that fit, sample and native entry modes reject
runtime drift before initialization. A positive native-mode control confirms that
valid inputs can reach initialization. Host identity is mocked in the child so
the controls also work on CI runners; no GPU or host query is executed.

The request receipt must use the complete frozen job digest and `attempt-0001`
directory. The positive control tests this path contract instead of failing
early on an unrelated output-path assertion.

These fixtures establish prelaunch behavior for this source version. They do
not certify a dataset, open an official test, measure model quality, or establish
complete system-library closure. Future source versions need their own frozen
identities and proof. The separate digest-acceleration experiment is not adopted
by this round or these fixtures.

Run:

```sh
python3 -m unittest research.benchmark.tests.test_tabddpm_prelaunch -v
```
