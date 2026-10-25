#!/usr/bin/env python3
"""End-to-end check: does the C++ device code reproduce the trained model?

verify_quantized.py re-implements classify()'s integer path *in Python*, so it
validates the exported arrays but not the C++ itself. This compares the actual
compiled binary's output against sklearn, feature by feature and window by
window:

    1. features  — C++ extract_features() vs pipeline.extract_features()
    2. decisions — C++ INT8 forest        vs sklearn float32 forest

Run tests/host_eval.cpp first to produce tests/host_predictions.csv.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import FEATURES, extract_features  # noqa: E402

host = pd.read_csv(ROOT / "tests" / "host_predictions.csv")
traffic = pd.read_csv(ROOT / "data" / "can_traffic.csv")
clf = joblib.load(ROOT / "model" / "detector.joblib")

X_py, y_py = extract_features(traffic)
X_cpp = host[[f"f{i}" for i in range(len(FEATURES))]].to_numpy(dtype=np.float64)

if len(X_py) != len(X_cpp):
    sys.exit(f"window count mismatch: python {len(X_py)} vs c++ {len(X_cpp)}")

print(f"[compare] {len(X_cpp)} windows x {len(FEATURES)} features\n")

# ---- 1. features -----------------------------------------------------------
# C++ reads integer-microsecond timestamps (what micros() returns on device);
# Python reads full-precision float seconds. Timing features therefore differ
# by the rounding, which is a property of the hardware clock, not a bug.
print(f"{'feature':<20} {'max abs diff':>14} {'max rel diff':>14}")
print("-" * 50)
worst_rel = 0.0
for i, name in enumerate(FEATURES):
    a, b = X_py[:, i].astype(np.float64), X_cpp[:, i]
    abs_d = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    rel_d = abs_d / denom
    worst_rel = max(worst_rel, rel_d.max())
    print(f"{name:<20} {abs_d.max():>14.3e} {rel_d.max():>14.3e}")

# ---- 2. decisions ----------------------------------------------------------
pred_sklearn = clf.predict(X_py.astype(np.float32))
pred_cpp = host["pred"].to_numpy()
truth = host["true"].to_numpy()

agree = (pred_sklearn == pred_cpp).mean()
acc_sklearn = (pred_sklearn == truth).mean()
acc_cpp = (pred_cpp == truth).mean()

print(f"\n{'':<34}{'all 2160 windows':>16}")
print("-" * 50)
print(f"{'sklearn float32 vs ground truth':<34}{acc_sklearn:>15.4%}")
print(f"{'C++ INT8      vs ground truth':<34}{acc_cpp:>15.4%}")
print(f"{'C++ INT8      vs sklearn float32':<34}{agree:>15.4%}")

mismatch = np.flatnonzero(pred_sklearn != pred_cpp)
if mismatch.size:
    print(f"\n{mismatch.size} disagreement(s):")
    names = ["NORMAL", "DOS", "FUZZY", "FAKE_RPM", "FAKE_GEAR"]
    for w in mismatch[:10]:
        print(f"  window {w:5d}  sklearn={names[pred_sklearn[w]]:<10}"
              f" c++={names[pred_cpp[w]]:<10} truth={names[truth[w]]}")

print(f"\nworst relative feature error: {worst_rel:.3e}")
sys.exit(0 if agree == 1.0 else 1)
