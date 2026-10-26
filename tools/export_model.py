#!/usr/bin/env python
"""Export the INT8 TFLite model as a plain C header the firmware can compile.

WHY THIS EXISTS INSTEAD OF SHIPPING THE .tflite FLATBUFFER
----------------------------------------------------------
The original plan was TensorFlow Lite Micro on the ESP32, which parses the
flatbuffer at boot and runs its own kernels. That is the normal answer and it is
what `model_data.h` (the raw flatbuffer bytes) was generated for.

It does not build. The only Arduino-library packaging of TFLite Micro for the
ESP32 is `TensorFlowLite_ESP32`, last released in 2021, and it fails three
different ways against the current platform (espressif32 8.x / arduino-esp32
3.3 / ESP-IDF 5.5):

  1. src/screen/.../st7796.c           -- calls LCD_IFACE_RELEASE, gone from IDF 5
  2. src/screen/.../scr_interface.c    -- calls gpio_pad_select_gpio, gone from IDF 5
  3. third_party/flatbuffers/stl_emulation.h:386
         error: assignment of read-only member 'span<T,Extent>::count_'

(1) and (2) are an LCD driver the library bundles and we never touch; (3) is in
the real inference path. The usual fix is to pin `platform = espressif32@6.x` and
drag the whole toolchain back two years, which makes the repo harder to build for
anyone cloning it and freezes us on an unmaintained library either way.

The model is three fully-connected layers and a softmax -- 1,157 parameters. The
int8 forward pass for that is about seventy lines of integer arithmetic. So we
emit the weights here and run them with our own kernel in `nn.h`, which:

  * builds on any toolchain, with no third-party dependency at all
  * is smaller and faster than a flatbuffer parser plus generic kernels
  * is verifiable -- tests/test_parity.py runs the same frames through this
    kernel and through TFLite's own int8 interpreter and requires the predicted
    class to agree on every frame

That last point matters. Hand-writing inference is only defensible if you then
prove it matches the reference, so the parity test is part of the deal, not an
afterthought.

WHAT THE NUMBERS IN THE HEADER MEAN
-----------------------------------
TFLite's int8 fully-connected layer computes, for output channel j:

    acc_j = sum_i (q_in[i] - zp_in) * q_w[j][i]        // int32
    acc_j = acc_j + bias_j                             // int32, already scaled
    q_out_j = clamp(zp_out + requantize(acc_j, M_j))   // back to int8

`acc_j` lives at scale (s_in * s_w[j]) -- the product of the two input scales --
which is exactly why the bias is stored as int32 at that same scale and can just
be added. Getting back to the output's own scale needs a multiply by

    M_j = (s_in * s_w[j]) / s_out

and M_j is a fraction less than 1, so it cannot be an integer. TFLite stores it
as a fixed-point pair (multiplier, shift) with multiplier in [2^30, 2^31):

    M_j ~= multiplier_j * 2^-31 * 2^shift_j

We precompute that pair here, on the host, in double precision, so the device
never does floating-point division. Weights are per-output-channel quantized
(each row of W has its own scale), hence one multiplier/shift per channel rather
than one per layer.

Run:
    PYTHONPATH=src python tools/export_model.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parent.parent
TFLITE = ROOT / "results" / "mlp_int8.tflite"
OUT = ROOT / "firmware" / "esp32" / "src" / "model_int8.h"

CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


def quantize_multiplier(real: float) -> tuple[int, int]:
    """Split a positive real multiplier into (int32 multiplier, shift).

    This is TFLite's QuantizeMultiplier, reimplemented so the exported constants
    are bit-identical to what the reference kernel would have used.

    frexp gives real = frac * 2^shift with frac in [0.5, 1). Scaling frac by
    2^31 lands it in [2^30, 2^31), which is the invariant the fixed-point
    multiply below relies on for its headroom.
    """
    if real == 0.0:
        return 0, 0
    frac, shift = math.frexp(real)
    q = int(round(frac * (1 << 31)))
    if q == (1 << 31):          # frac rounded up to 1.0, renormalise
        q //= 2
        shift += 1
    assert (1 << 30) <= q < (1 << 31), q
    return q, shift


def carr(name: str, values, dtype: str, per_line: int, fmt=str) -> str:
    """Format a C array initialiser, wrapped so the header stays readable."""
    vals = [fmt(v) for v in np.asarray(values).reshape(-1)]
    lines, cur = [], []
    for v in vals:
        cur.append(v)
        if len(cur) == per_line:
            lines.append("    " + ", ".join(cur) + ",")
            cur = []
    if cur:
        lines.append("    " + ", ".join(cur) + ",")
    body = "\n".join(lines)
    return f"constexpr {dtype} {name}[{len(vals)}] = {{\n{body}\n}};\n"


def main() -> int:
    if not TFLITE.exists():
        print(f"missing {TFLITE} -- run quantize.py first", file=sys.stderr)
        return 1

    interp = tf.lite.Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    td = {t["index"]: t for t in interp.get_tensor_details()}

    # Walk the ops rather than hard-coding tensor indices, so a retrain that
    # renumbers tensors does not silently export the wrong array.
    layers = []
    softmax_in = None
    for op in interp._get_ops_details():
        if op["op_name"] == "FULLY_CONNECTED":
            x, w, b = op["inputs"]
            layers.append({"in": x, "w": int(w), "b": int(b),
                           "out": int(op["outputs"][0])})
        elif op["op_name"] == "SOFTMAX":
            softmax_in = int(op["inputs"][0])

    if len(layers) != 3 or softmax_in is None:
        print(f"unexpected graph: {len(layers)} FC ops, softmax={softmax_in}",
              file=sys.stderr)
        return 1

    # The op list is not guaranteed to be in execution order, so chain the
    # layers by matching each one's input tensor to the previous one's output.
    first = [L for L in layers if L["in"] == 0]
    assert len(first) == 1, "could not find the layer reading the input tensor"
    ordered = [first[0]]
    while len(ordered) < 3:
        prev_out = ordered[-1]["out"]
        nxt = [L for L in layers if L["in"] == prev_out]
        assert len(nxt) == 1, f"broken chain after tensor {prev_out}"
        ordered.append(nxt[0])
    assert ordered[-1]["out"] == softmax_in, "last FC does not feed the softmax"
    layers = ordered

    in_q = td[0]["quantization_parameters"]
    in_scale = float(in_q["scales"][0])
    in_zero = int(in_q["zero_points"][0])

    parts: list[str] = []
    meta = {"layers": []}

    for li, L in enumerate(layers):
        w = interp.get_tensor(L["w"])            # [out_dim, in_dim] int8
        b = interp.get_tensor(L["b"])            # [out_dim] int32
        wq = td[L["w"]]["quantization_parameters"]
        s_in = float(td[L["in"]]["quantization_parameters"]["scales"][0])
        out_q = td[L["out"]]["quantization_parameters"]
        s_out = float(out_q["scales"][0])
        zp_out = int(out_q["zero_points"][0])
        zp_in = int(td[L["in"]]["quantization_parameters"]["zero_points"][0])

        out_dim, in_dim = w.shape
        w_scales = np.asarray(wq["scales"], dtype=np.float64)
        assert len(w_scales) == out_dim, "expected per-channel weight scales"
        assert np.all(np.asarray(wq["zero_points"]) == 0), \
            "weights must be symmetric (zero_point 0)"

        mults, shifts = [], []
        for j in range(out_dim):
            m, s = quantize_multiplier(s_in * float(w_scales[j]) / s_out)
            mults.append(m)
            shifts.append(s)

        # Layers 0 and 1 have ReLU fused. For int8 output whose zero_point is
        # -128, "clamp at 0.0 in float" is "clamp at -128 in int8", which is
        # already the bottom of the type -- so the ReLU costs literally nothing,
        # it is the saturation we would do anyway. The last layer produces
        # logits and has no activation. We export the bounds explicitly rather
        # than assume, so a model retrained with a different activation still
        # exports correctly.
        is_last = li == len(layers) - 1
        act_min = -128 if is_last else max(-128, zp_out)
        act_max = 127

        p = f"L{li}"
        parts.append(f"// ---- layer {li}: {in_dim} -> {out_dim}"
                     f"{'' if is_last else ' + ReLU'} ----\n")
        parts.append(f"#define SC_{p}_IN  {in_dim}\n")
        parts.append(f"#define SC_{p}_OUT {out_dim}\n")
        parts.append(f"constexpr int32_t SC_{p}_IN_ZERO  = {zp_in};\n")
        parts.append(f"constexpr int32_t SC_{p}_OUT_ZERO = {zp_out};\n")
        parts.append(f"constexpr int32_t SC_{p}_ACT_MIN  = {act_min};\n")
        parts.append(f"constexpr int32_t SC_{p}_ACT_MAX  = {act_max};\n")
        parts.append(carr(f"SC_{p}_W", w, "int8_t", 16, lambda v: f"{int(v):4d}"))
        parts.append(carr(f"SC_{p}_B", b, "int32_t", 8, lambda v: f"{int(v):7d}"))
        parts.append(carr(f"SC_{p}_M", mults, "int32_t", 4,
                          lambda v: f"{int(v):11d}"))
        parts.append(carr(f"SC_{p}_SHIFT", shifts, "int32_t", 12,
                          lambda v: f"{int(v):3d}"))
        parts.append("\n")

        meta["layers"].append({
            "in": int(in_dim), "out": int(out_dim),
            "params": int(w.size + b.size),
            "relu": not is_last,
        })

    logit_q = td[softmax_in]["quantization_parameters"]
    n_params = sum(L["params"] for L in meta["layers"])

    header = f'''// Generated by tools/export_model.py -- do not edit by hand.
//
// The INT8 weights of the 16-32-16-5 classifier, lifted out of
// results/mlp_int8.tflite so the firmware can run them without TensorFlow Lite
// Micro. See the docstring in tools/export_model.py for why we do not link
// TFLite Micro (short version: its only ESP32 Arduino packaging is from 2021
// and does not compile against ESP-IDF 5).
//
// {n_params} parameters total. nn.h consumes these; tests/test_parity.py checks the
// result against TFLite's own int8 interpreter on real frames.
#pragma once

#include <stdint.h>

#define SC_N_CLASSES {len(CLASS_NAMES)}

// Input quantisation. The 16 features are already min-max scaled to [0, 1] by
// features.h, and this maps that range onto the full int8 span:
//     q = round(f / {in_scale:.9f}) + ({in_zero})
constexpr float   SC_IN_SCALE = {in_scale:.9f}f;
constexpr int32_t SC_IN_ZERO  = {in_zero};

// Output logit quantisation, needed only to turn the winning logit into a
// human-readable confidence. The CLASS does not depend on it: softmax is
// monotonic, so argmax over the int8 logits is the same class as argmax over
// the float probabilities.
constexpr float   SC_LOGIT_SCALE = {float(logit_q["scales"][0]):.9f}f;
constexpr int32_t SC_LOGIT_ZERO  = {int(logit_q["zero_points"][0])};

'''

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + "".join(parts), encoding="utf-8")

    size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(ROOT)}  ({size:,} bytes of source)")
    print(f"  {n_params:,} parameters in {len(meta['layers'])} layers")
    for i, L in enumerate(meta["layers"]):
        print(f"    layer {i}: {L['in']:>2} -> {L['out']:<2} "
              f"{L['params']:>5,} params{'  + ReLU' if L['relu'] else ''}")
    weight_bytes = sum(L["in"] * L["out"] for L in meta["layers"])
    bias_bytes = sum(L["out"] * 4 for L in meta["layers"])
    req_bytes = sum(L["out"] * 8 for L in meta["layers"])
    print(f"  flash: {weight_bytes:,} B weights + {bias_bytes:,} B biases"
          f" + {req_bytes:,} B requant = {weight_bytes + bias_bytes + req_bytes:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
