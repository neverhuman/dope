#!/usr/bin/env bash
# Admit only the standalone Rust CPU application closure, not research bindings.
set -euo pipefail
[[ $# == 1 && -s "$1" ]] || { echo 'engine_runtime_graph_missing' >&2; exit 2; }
if awk '{print $1}' "$1" | grep -Eiq '^(tch|torch-sys|libtorch.*|pyo3.*|cpython|python.*|libpython.*|rustpython.*|pyembed|ort|onnxruntime.*|tensorflow.*|tflite.*|numpy|cudarc|candle-kernels|candle-metal-kernels|metal|accelerate-src|intel-mkl.*)$'; then
  echo 'engine_runtime_foreign_binding_or_gpu_dependency_refused' >&2
  exit 1
fi
grep -Eq '^dope-cli v' "$1" || { echo 'engine_runtime_cli_missing' >&2; exit 1; }
grep -Eq '^dope-core v' "$1" || { echo 'engine_runtime_core_missing' >&2; exit 1; }
