#!/usr/bin/env python3
"""Shrink the MLP to INT8 and export it as a C array for the ESP32.

Why quantise at all? The float32 model is only 4.5 KB, so size is not really the
problem -- speed is. An ESP32 has no floating-point-friendly SIMD, but it does
have fast 8-bit multiply-accumulate. Doing the whole network in INT8 arithmetic
is several times faster per frame, and at 500 kbit/s we only have ~250 us per
frame to play with.

"Full integer" quantisation means input, weights, activations and output are ALL
int8. TensorFlow needs to know the real-world range of every tensor to pick the
scale factors, and it cannot work that out from the weights alone -- activations
depend on the data. So we hand it a REPRESENTATIVE DATASET: a few hundred real
feature rows. TF runs them through the float model, watches the min/max at every
layer, and sizes the quantisation accordingly.

Two things matter about that sample:

  1. It must come from TRAINING data. Using test rows would leak the test
     distribution into the shipped model.
  2. It must include attack frames. Sampling only normal traffic would make
     bus_rate's observed range far too narrow, and a DoS flood would then
     saturate at the top of the int8 range and lose all resolution exactly when
     we need it.

We then verify on the real test split that INT8 did not cost us accuracy, which
is the number the README quotes.

    python -m securecan.quantize
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import dataset, features as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
FIRMWARE = ROOT / "firmware" / "esp32" / "src"
CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def representative_dataset(X, n=500, seed=0):
    """Yield single feature rows for TF to profile the activation ranges with.

    Stratified across classes is not possible here (we only get X), so the caller
    passes an already-balanced sample. 500 rows is plenty: we are measuring
    min/max, not training anything.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    def gen():
        for i in idx:
            yield [X[i:i + 1].astype(np.float32)]

    return gen


def balanced_sample(X, y, per_class=150, seed=0):
    """Equal numbers of each class, so no class's feature range is missed."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in range(5):
        idx = np.flatnonzero(y == c)
        if not len(idx):
            continue
        picks.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    return X[sel]


def convert_int8(keras_path, rep_X):
    """Keras float32 -> fully-integer INT8 TFLite bytes."""
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)

    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(rep_X)

    # Refuse to fall back to float. Without these three lines TF will happily
    # leave an op in float32 if INT8 is inconvenient, and TFLite Micro on the
    # ESP32 would then need the float kernels linked in -- bigger binary, slower,
    # and the "fully integer" claim would be false.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    return conv.convert()


def convert_float(keras_path):
    """Plain float32 TFLite, for the size/accuracy comparison only."""
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path)
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


# ---------------------------------------------------------------------------
# running the quantised model
# ---------------------------------------------------------------------------

def tflite_predict(tflite_bytes, X, batch_log=None):
    """Run every row through the interpreter and return predicted classes.

    TFLite Micro has no batching, so this mirrors the device exactly: one frame
    in, one answer out. It is slow in Python (~10k frames/s) which is why we
    evaluate on a subsample rather than all 2.6 M test rows.
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    # The int8 model stores its own scale/zero-point. Quantising the input by
    # hand with these exact numbers is what the firmware has to do too.
    in_scale, in_zero = inp["quantization"]
    is_int8 = inp["dtype"] == np.int8

    preds = np.empty(len(X), dtype=np.int64)
    for i, row in enumerate(X):
        if is_int8:
            q = np.round(row / in_scale + in_zero)
            q = np.clip(q, -128, 127).astype(np.int8)
        else:
            q = row.astype(np.float32)

        interp.set_tensor(inp["index"], q.reshape(1, -1))
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(out["index"])[0]))

        if batch_log and i and i % batch_log == 0:
            print(f"    {i:,} / {len(X):,}")

    return preds


# ---------------------------------------------------------------------------
# C export
# ---------------------------------------------------------------------------

def constants_header(const, path=None):
    """Export the learned constants as C, so the firmware's features match ours.

    The ID table is a sorted array of {id, period} pairs. Sorted so the firmware
    can binary-search it; an ID that is not found is by definition unknown, which
    is exactly the is_known_id feature.
    """
    ids = sorted(int(k) for k in const.id_periods)

    lines = [
        "// Generated by src/securecan/quantize.py -- do not edit by hand.",
        "//",
        "// Learned from the TRAINING split of the HCRL captures, normal frames",
        "// only. These must match results/constants.json exactly or the firmware",
        "// will compute different features than the model was trained on.",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        f"#define SC_N_KNOWN_IDS {len(ids)}",
        "",
        f"constexpr float SC_DT_PREV_CLIP      = {const.dt_prev_clip:.9f}f;",
        f"constexpr float SC_DT_SAME_CLIP      = {const.dt_same_clip:.9f}f;",
        f"constexpr float SC_PERIOD_RATIO_CLIP = {const.period_ratio_clip:.9f}f;",
        f"constexpr float SC_BUS_RATE_CLIP     = {const.bus_rate_clip:.9f}f;",
        "",
        "// Sorted by id so the firmware can binary-search. Not found == unknown id.",
        "struct ScKnownId { uint16_t id; float period; };",
        "",
        "constexpr ScKnownId SC_KNOWN_IDS[SC_N_KNOWN_IDS] = {",
    ]

    for cid in ids:
        p = const.id_periods[str(cid)]
        lines.append(f"    {{ 0x{cid:03X}, {p:.9f}f }},   // {p * 1e3:8.2f} ms")

    lines += ["};", ""]
    text = "\n".join(lines)

    if path:
        Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-rows", type=int, default=120_000,
                    help="test rows to check INT8 accuracy on (full 2.6M is slow)")
    args = ap.parse_args()

    FIRMWARE.mkdir(parents=True, exist_ok=True)
    const = F.Constants.load(RESULTS / "constants.json")

    print("\n=== loading splits ===")
    splits = dataset.load_or_build(const)
    Xt, yt = splits["train"]
    Xs, ys = splits["test"]

    rep_X = balanced_sample(Xt, yt)
    print(f"  representative sample: {len(rep_X)} rows from TRAIN, balanced")

    # --- convert -----------------------------------------------------------
    print("\n=== converting ===")
    float_blob = convert_float(RESULTS / "mlp.keras")
    int8_blob = convert_int8(RESULTS / "mlp.keras", rep_X)

    (RESULTS / "mlp_float32.tflite").write_bytes(float_blob)
    (RESULTS / "mlp_int8.tflite").write_bytes(int8_blob)

    kb = lambda b: len(b) / 1024  # noqa: E731
    print(f"  keras   {os.path.getsize(RESULTS / 'mlp.keras') / 1024:7.1f} KB")
    print(f"  float32 {kb(float_blob):7.1f} KB  (tflite)")
    print(f"  int8    {kb(int8_blob):7.1f} KB  "
          f"({kb(float_blob) / kb(int8_blob):.2f}x smaller than float32)")

    # --- accuracy check ----------------------------------------------------
    # Subsample the test split, but keep it a RANDOM subsample so the class
    # balance -- and therefore the FPR -- stays representative.
    rng = np.random.default_rng(0)
    n = min(args.eval_rows, len(ys))
    sel = rng.choice(len(ys), size=n, replace=False)
    Xe, ye = Xs[sel], ys[sel]

    print(f"\n=== accuracy on {n:,} random test rows ===")
    keras_pred = np.load(RESULTS / "_mlp_test_pred.npy")[sel]

    print("  running int8 interpreter (one frame at a time, like the device)")
    int8_pred = tflite_predict(int8_blob, Xe, batch_log=40_000)

    acc_keras = float((keras_pred == ye).mean())
    acc_int8 = float((int8_pred == ye).mean())
    agree = float((int8_pred == keras_pred).mean())

    def fpr(pred):
        norm = ye == 0
        return float((pred[norm] != 0).mean()) if norm.any() else 0.0

    def attack_f1(pred):
        from sklearn.metrics import f1_score
        return float(f1_score((ye != 0).astype(int), (pred != 0).astype(int),
                              zero_division=0))

    print(f"\n  {'':<12}{'accuracy':>11}{'attack F1':>11}{'FPR %':>9}")
    print(f"  {'float32':<12}{acc_keras:>11.5f}{attack_f1(keras_pred):>11.5f}"
          f"{fpr(keras_pred) * 100:>9.4f}")
    print(f"  {'int8':<12}{acc_int8:>11.5f}{attack_f1(int8_pred):>11.5f}"
          f"{fpr(int8_pred) * 100:>9.4f}")
    print(f"\n  accuracy change: {(acc_int8 - acc_keras) * 100:+.4f} pp")
    print(f"  int8 agrees with float32 on {agree * 100:.3f}% of frames")

    # --- per-class, so we can see if one attack took the hit ----------------
    print(f"\n  {'class':<10}{'float32 recall':>16}{'int8 recall':>14}")
    per_class = {}
    for c in range(5):
        m = ye == c
        if not m.any():
            continue
        rk = float((keras_pred[m] == c).mean())
        ri = float((int8_pred[m] == c).mean())
        per_class[CLASS_NAMES[c]] = {"float32_recall": round(rk, 5),
                                     "int8_recall": round(ri, 5)}
        print(f"  {CLASS_NAMES[c]:<10}{rk:>16.5f}{ri:>14.5f}")

    # --- export to firmware -------------------------------------------------
    #
    # The feature constants go out as a C header here. The WEIGHTS do not: the
    # firmware does not parse the flatbuffer, because the only ESP32 Arduino
    # packaging of TFLite Micro is a 2021 snapshot that will not build against
    # ESP-IDF 5. tools/export_model.py reads results/mlp_int8.tflite and emits
    # the weights as plain arrays for firmware/esp32/src/nn.h instead.
    #
    # Keeping that in a separate tool is deliberate -- it means the .tflite in
    # results/ stays the single source of the model, and the firmware header is
    # derived from it rather than being a second thing to keep in sync.
    print("\n=== exporting C headers ===")
    constants_header(const, FIRMWARE / "constants_data.h")
    print(f"  firmware/esp32/src/constants_data.h  "
          f"{len(const.id_periods)} known IDs")
    print( "  weights: run `python tools/export_model.py` -> src/nn.h's arrays")

    report = {
        "keras_bytes": os.path.getsize(RESULTS / "mlp.keras"),
        "tflite_float32_bytes": len(float_blob),
        "tflite_int8_bytes": len(int8_blob),
        "int8_kb": round(kb(int8_blob), 2),
        "shrink_vs_float32": round(kb(float_blob) / kb(int8_blob), 3),
        "eval_rows": n,
        "float32_accuracy": round(acc_keras, 6),
        "int8_accuracy": round(acc_int8, 6),
        "accuracy_drop_pp": round((acc_keras - acc_int8) * 100, 4),
        "float32_attack_f1": round(attack_f1(keras_pred), 5),
        "int8_attack_f1": round(attack_f1(int8_pred), 5),
        "float32_fpr_pct": round(fpr(keras_pred) * 100, 5),
        "int8_fpr_pct": round(fpr(int8_pred) * 100, 5),
        "int8_vs_float32_agreement_pct": round(agree * 100, 4),
        "per_class": per_class,
    }
    (RESULTS / "quantization.json").write_text(json.dumps(report, indent=2))
    print("\n-> results/mlp_int8.tflite, results/quantization.json")


if __name__ == "__main__":
    main()
