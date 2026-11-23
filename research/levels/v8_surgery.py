"""Model surgery: start ``SingleEncV8``'s receiver decoder from trained ``SingleEncV7`` weights.

V7 is a packet-in, packet-residual-out transformer; V8's receiver decoder has the same role and
the same pre-LN split-QKV GELU block, plus a pairwise edge bias and an owner-code token. The DSE8
decoder inputs extend V7's packet features (knots, loadings and per-class weights share slots
0-36), so the trained V7 trunk carries over:

* blocks ``0..n_dec-1`` -> ``dec_blocks`` and ``final_norm`` -> ``dec_final_norm`` by consistent
  weight selection (Xu et al., "Initializing Models with Larger Ones"): every tensor keeps the
  same evenly spaced residual indices, each attention head keeps its own evenly spaced slice, and
  the feed-forward keeps evenly spaced hidden units. With equal widths the map is the identity.
* ``col_in`` / ``glob_in`` -> ``dec_col_in`` / ``dec_glob_in`` by the feature map below. V8-only
  inputs start at zero weight, and the width feature is rescaled (V7 ``p/48``, V8 ``p/64``).
* ``edge_proj``, ``dec_code_in`` and every residual head start at zero. V8's plan is then exactly
  the analytic DSE8 packet at step 0, as with the default initialisation: only the decoder's
  hidden state, which the heads learn from, starts trained. ``--with-delta-w`` also copies V7's
  per-class weight head, which has V8's ``0.50 * tanh`` form and meaning.

The owner evidence encoder has no V7 counterpart. Its tensors are left out of the file, so the
trainer's seeded initialisation stays (``s3_v8_trainer --init-from`` loads non-strictly).

    python -m research.levels.v8_surgery --v7 models/single-enc-v7/model.safetensors \\
        --out v8-from-v7.safetensors --report v8-from-v7.json [--with-delta-w]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from research.levels.v7_single_encoding import MAX_COLS as V7_MAX_COLS
from research.levels.v8_packet import V8_MAX_COLS
from research.levels.v8_single_encoding import SingleEncV8

# V7 packet feature slot -> DSE8 decoder input slot (same meaning; see each packet's builder).
V7_TO_V8_COL = {**{i: i for i in range(37)}, 38: 39, 39: 41, 40: 42, 41: 43}  # is_disc->is_int, log span, z_max, knot level
V7_TO_V8_GLOB = {0: 0, 1: 1, 2: 2, **{i: i for i in range(4, 16)}}  # task one-hot, prior, n_fit, width, resid sd, classes
GLOB_SCALE = {13: V8_MAX_COLS / V7_MAX_COLS}  # V8 width input is p/64 where V7's was p/48
ZERO_HEADS = ("delta_knots", "delta_L", "delta_W", "node_sel", "node_thr", "node_leaf", "bilin_u", "bilin_v",
              "bilin_w", "flow_scale_head", "flow_shift_head", "gate_head", "dec_code_in")


def even(n_src: int, n_dst: int) -> list[int]:
    """``n_dst`` evenly spaced indices of ``range(n_src)`` (the identity when the sizes are equal)."""
    if n_dst > n_src:
        raise ValueError("surgery_target_wider_than_source")
    return [(i * n_src) // n_dst for i in range(n_dst)]


def per_head(d_src: int, d_dst: int, n_heads: int) -> list[int]:
    hs, hd = d_src // n_heads, d_dst // n_heads
    return [h * hs + j for h in range(n_heads) for j in even(hs, hd)]


def _map_inputs(w7: torch.Tensor, rows: list[int], n_in: int, pairs: Mapping[int, int],
                scale: Mapping[int, float]) -> torch.Tensor:
    w = torch.zeros(len(rows), n_in, dtype=torch.float32)
    for i7, i8 in pairs.items():
        if i7 < w7.shape[1]:
            w[:, i8] = w7[rows, i7] * float(scale.get(i8, 1.0))
    return w


def transplant(v7: Mapping[str, torch.Tensor], v8: SingleEncV8, n_heads: int = 4,
               with_delta_w: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return the V8 tensors to load (decoder only) and a report of what was carried over."""
    d7, d8 = v7["final_norm.weight"].shape[0], v8.d_model
    n7 = len({k.split(".")[1] for k in v7 if k.startswith("blocks.")})
    if n7 < v8.n_dec_layers:
        raise ValueError("surgery_source_has_fewer_layers")
    res, head = even(d7, d8), per_head(d7, d8, n_heads)
    target = {k: v.detach().clone().float() for k, v in v8.state_dict().items()}
    out: dict[str, torch.Tensor] = {}

    def put(name: str, t: torch.Tensor) -> None:
        if tuple(t.shape) != tuple(target[name].shape):
            raise ValueError(f"surgery_shape_mismatch:{name}")
        out[name] = t.contiguous().float()

    put("dec_col_in.weight", _map_inputs(v7["col_in.weight"], res, target["dec_col_in.weight"].shape[1], V7_TO_V8_COL, {}))
    put("dec_col_in.bias", v7["col_in.bias"][res])
    put("dec_glob_in.weight", _map_inputs(v7["glob_in.weight"], res, target["dec_glob_in.weight"].shape[1], V7_TO_V8_GLOB, GLOB_SCALE))
    put("dec_glob_in.bias", v7["glob_in.bias"][res])
    for layer in range(v8.n_dec_layers):
        s, d = f"blocks.{layer}.", f"dec_blocks.{layer}."
        ff = even(v7[s + "ff1.weight"].shape[0], target[d + "ff1.weight"].shape[0])
        for nm in ("norm1", "norm2"):
            put(d + nm + ".weight", v7[s + nm + ".weight"][res])
            put(d + nm + ".bias", v7[s + nm + ".bias"][res])
        for nm in ("q_proj", "k_proj", "v_proj"):
            put(d + nm + ".weight", v7[s + nm + ".weight"][head][:, res])
            put(d + nm + ".bias", v7[s + nm + ".bias"][head])
        put(d + "o_proj.weight", v7[s + "o_proj.weight"][res][:, head])
        put(d + "o_proj.bias", v7[s + "o_proj.bias"][res])
        put(d + "ff1.weight", v7[s + "ff1.weight"][ff][:, res])
        put(d + "ff1.bias", v7[s + "ff1.bias"][ff])
        put(d + "ff2.weight", v7[s + "ff2.weight"][res][:, ff])
        put(d + "ff2.bias", v7[s + "ff2.bias"][res])
        put(d + "edge_proj.weight", torch.zeros_like(target[d + "edge_proj.weight"]))
        put(d + "edge_proj.bias", torch.zeros_like(target[d + "edge_proj.bias"]))
    put("dec_final_norm.weight", v7["final_norm.weight"][res])
    put("dec_final_norm.bias", v7["final_norm.bias"][res])
    for nm in ZERO_HEADS:
        for part in ("weight", "bias"):
            put(f"{nm}.{part}", torch.zeros_like(target[f"{nm}.{part}"]))
    if with_delta_w:
        put("delta_W.weight", v7["delta_W.weight"][:, res])
        put("delta_W.bias", v7["delta_W.bias"].clone())
    zero = {f"{nm}.{p}" for nm in ZERO_HEADS for p in ("weight", "bias")} - ({"delta_W.weight", "delta_W.bias"} if with_delta_w else set())
    zero |= {k for k in out if ".edge_proj." in k}
    carried = sum(out[k].numel() for k in out if k not in zero)
    report = {
        "schema": "dope.v8-surgery/v1", "source_width": d7, "target_width": d8, "source_layers": n7,
        "layers_used": list(range(v8.n_dec_layers)), "selection": "consistent evenly spaced (residual, per head, ffn)",
        "col_map": {str(k): v for k, v in V7_TO_V8_COL.items()}, "glob_map": {str(k): v for k, v in V7_TO_V8_GLOB.items()},
        "glob_scale": {str(k): v for k, v in GLOB_SCALE.items()}, "with_delta_w": with_delta_w,
        "tensors_written": len(out), "params_carried_from_v7": carried,
        "decoder_params": sum(t.numel() for k, t in target.items() if not k.startswith(("enc_", "aux_"))),
        "v8_params": sum(t.numel() for t in target.values()),
    }
    return out, report


def main(argv: Sequence[str] | None = None) -> int:
    from safetensors.torch import load_file, save_file

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v7", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--with-delta-w", action="store_true")
    a = ap.parse_args(argv)
    src = Path(a.v7).read_bytes()
    tensors, report = transplant(load_file(a.v7), SingleEncV8(), with_delta_w=a.with_delta_w)
    save_file({k: tensors[k] for k in sorted(tensors)}, a.out, metadata={"schema": "dope.v8-surgery/v1"})
    report.update(source=str(a.v7), source_sha256=hashlib.sha256(src).hexdigest(),
                  out_sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
    text = json.dumps(report, indent=1, sort_keys=True)
    if a.report:
        Path(a.report).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
