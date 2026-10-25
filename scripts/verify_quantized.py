#!/usr/bin/env python3
"""Verify the exported INT8 forest against the float sklearn model.

Re-implements, in Python, the exact integer path that src/detector.cpp walks:
  quantize(feature) -> int8, then NODE_* array traversal + majority vote.
Any disagreement means the C++ header or the quantization math is wrong.
"""

import re
import json
import numpy as np
import joblib
from pathlib import Path

from pipeline import extract_features, generate_can_traffic, ATTACK_LABELS

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "model"


def parse_header(path):
    """Pull the static arrays out of detector_quantized.h."""
    text = path.read_text()
    out = {}
    for name in ["FEAT_LO", "FEAT_SCALE", "NODE_FEATURE", "NODE_THRESH",
                 "NODE_LEFT", "NODE_RIGHT", "NODE_CLASS", "TREE_OFFSET"]:
        m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}", text)
        if not m:
            raise ValueError(f"{name} missing from header")
        vals = [v.strip().rstrip("f") for v in m.group(1).split(",") if v.strip()]
        out[name] = np.array([float(v) for v in vals])
    for name in ["N_FEATURES", "N_TREES", "N_NODES", "N_CLASSES"]:
        out[name] = int(re.search(rf"#define {name} (\d+)", text).group(1))
    return out


def classify_int8(h, feats):
    """Mirror of classify() in src/detector.cpp."""
    q = np.clip((feats - h["FEAT_LO"]) * h["FEAT_SCALE"] - 127.0, -128, 127).astype(np.int8)

    votes = np.zeros(h["N_CLASSES"], dtype=int)
    for t in range(h["N_TREES"]):
        base = int(h["TREE_OFFSET"][t])
        node = 0
        while h["NODE_FEATURE"][base + node] >= 0:
            f = int(h["NODE_FEATURE"][base + node])
            thr = h["NODE_THRESH"][base + node]
            node = int(h["NODE_LEFT"][base + node] if q[f] <= thr
                       else h["NODE_RIGHT"][base + node])
        votes[int(h["NODE_CLASS"][base + node])] += 1
    return int(votes.argmax())


if __name__ == "__main__":
    header = parse_header(MODEL / "detector_quantized.h")
    clf = joblib.load(MODEL / "detector.joblib")

    df = generate_can_traffic()
    X, y = extract_features(df)

    float_pred = clf.predict(X)
    int8_pred = np.array([classify_int8(header, x.astype(np.float64)) for x in X])

    agree = (float_pred == int8_pred).mean()
    acc_f = (float_pred == y).mean()
    acc_q = (int8_pred == y).mean()

    print(f"\nwindows evaluated      : {len(X)}")
    print(f"float32 vs int8 agree  : {agree * 100:.2f}%")
    print(f"accuracy  float32      : {acc_f * 100:.2f}%")
    print(f"accuracy  int8         : {acc_q * 100:.2f}%")
    print(f"accuracy  delta        : {(acc_q - acc_f) * 100:+.2f} pp")

    print("\nper-class int8 recall:")
    for c in sorted(set(y)):
        m = y == c
        print(f"  {ATTACK_LABELS[c]:<10} {(int8_pred[m] == c).mean() * 100:6.2f}%  (n={m.sum()})")

    arrays_bytes = header["N_NODES"] * (1 * 3 + 2 * 2)   # 3x int8 + 2x int16
    print(f"\nforest  : {header['N_TREES']} trees / {header['N_NODES']} nodes")
    print(f"runtime : {arrays_bytes} B ({arrays_bytes / 1024:.1f} KB) of int8/int16 arrays")

    json.dump({
        "windows": int(len(X)),
        "agreement_pct": round(agree * 100, 2),
        "acc_float32_pct": round(acc_f * 100, 2),
        "acc_int8_pct": round(acc_q * 100, 2),
        "delta_pp": round((acc_q - acc_f) * 100, 2),
        "n_trees": header["N_TREES"],
        "n_nodes": header["N_NODES"],
        "runtime_bytes": int(arrays_bytes),
        "per_class_recall_pct": {
            ATTACK_LABELS[c]: round(float((int8_pred[y == c] == c).mean() * 100), 2)
            for c in sorted(set(y))
        },
    }, open(MODEL / "quantization_report.json", "w"), indent=2)
    print("\n-> model/quantization_report.json")
