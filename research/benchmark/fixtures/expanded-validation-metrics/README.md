# Expanded validation diagnostic controls

The generated controls extract only the pinned Synthcity alpha-precision and beta-recall equations and the DOMIAS KDE density-ratio equation. They do not execute either author training pipeline. The full unchanged source files are retained so the extraction is checked against an exact upstream SHA256.

Synthcity commit `23f322fe381326ed01c41b13d469a06e38cce545` is distributed under its included Apache 2.0 license. DOMIAS commit `9527dab320f9d41372e1f67e0d5c2b80becfdd61` is distributed under its included MIT license. Source URLs, byte counts and hashes are recorded in `author-source-pins.json`.

Run `python -m unittest research.benchmark.tests.test_expanded_validation_metrics`. All arrays in these controls are generated. No dataset, sample, weights or official test partition are included.

Python source buffers have a `.py.txt` suffix to keep them opaque; byte hashes refer to unchanged author contents.
