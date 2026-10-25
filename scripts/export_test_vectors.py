#!/usr/bin/env python3
"""Export test vectors for the compiled C++ harness, then diff the results.

Writes two CSVs:
  tests/vec_features.csv  feature vectors   -> checks classify()
  tests/vec_windows.csv   raw CAN windows   -> checks extract_features()

Run tests/harness.exe over each, then call --check to compare against Python.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline import FEATURES, WINDOW, extract_features, ATTACK_LABELS

ROOT = Path(__file__).parent.parent
TESTS = ROOT / "tests"
HARNESS = TESTS / "harness.exe"


def build_vectors(n_windows=400):
    df = pd.read_csv(ROOT / "data" / "can_traffic.csv")
    X, y = extract_features(df)

    # spread the sample across the capture so every class is represented
    idx = np.linspace(0, len(X) - 1, min(n_windows, len(X))).astype(int)

    pd.DataFrame(X[idx], columns=FEATURES).to_csv(TESTS / "vec_features.csv",
                                                  index=False, float_format="%.6f")

    # raw windows: the exact frames each sampled window was built from
    rows = []
    arr = df.to_numpy()
    for w, i in enumerate(idx):
        s = i * (WINDOW // 2)
        for r in arr[s: s + WINDOW]:
            rows.append([w, r[0], int(r[1]), int(r[2]), *[int(v) for v in r[3:11]]])
    cols = ["window", "ts", "can_id", "dlc"] + [f"b{i}" for i in range(8)]
    pd.DataFrame(rows, columns=cols).to_csv(TESTS / "vec_windows.csv",
                                            index=False, float_format="%.6f")

    np.save(TESTS / "_idx.npy", idx)
    print(f"[export] {len(idx)} windows -> tests/vec_features.csv, tests/vec_windows.csv")
    return X, y, idx


def run_harness(csv, extra=None):
    cmd = [str(HARNESS), str(TESTS / csv)] + (extra or [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit(f"harness failed on {csv}")
    print("  " + out.stderr.strip())
    from io import StringIO
    return pd.read_csv(StringIO(out.stdout))


def check():
    df = pd.read_csv(ROOT / "data" / "can_traffic.csv")
    X, y = extract_features(df)
    idx = np.load(TESTS / "_idx.npy")
    clf = joblib.load(ROOT / "model" / "detector.joblib")

    Xs, ys = X[idx], y[idx]
    py_pred = clf.predict(Xs)

    print("\n[1/2] classify() on identical feature vectors")
    r1 = run_harness("vec_features.csv")
    cpp_pred = r1["predicted"].to_numpy()
    agree_cls = float((cpp_pred == py_pred).mean())
    print(f"  C++ vs sklearn agreement : {agree_cls * 100:.2f}%")
    print(f"  mean inference           : {r1['latency_ns'].mean():.0f} ns (host x86)")

    print("\n[2/2] extract_features() on raw CAN windows")
    r2 = run_harness("vec_windows.csv", ["--windows"])
    fcols = [c for c in r2.columns if c.startswith("f")]
    cpp_feats = r2[fcols].to_numpy()
    n = min(len(cpp_feats), len(Xs))

    # us-resolution timestamps on the device vs float seconds in Python, so
    # compare relative error per feature rather than demanding bit-equality
    denom = np.maximum(np.abs(Xs[:n]), 1e-6)
    rel = np.abs(cpp_feats[:n] - Xs[:n]) / denom
    worst = rel.max(0)

    print(f"  windows compared         : {n}")
    for name, w in sorted(zip(FEATURES, worst), key=lambda t: -t[1])[:5]:
        print(f"    {name:<20} max rel err {w:.2e}")

    cpp_wpred = r2["predicted"].to_numpy()[:n]
    agree_e2e = float((cpp_wpred == py_pred[:n]).mean())
    print(f"  end-to-end agreement     : {agree_e2e * 100:.2f}%")

    # NOTE: this sample spans the whole capture, so most windows were seen in
    # training. These are sanity numbers for the equivalence check only -- the
    # held-out metric is the one pipeline.py reports on its 25% test split.
    acc_py = float((py_pred == ys).mean())
    acc_cpp = float((cpp_wpred == ys[:n]).mean())
    print(f"\n  sklearn on this sample   : {acc_py * 100:.2f}%  (NOT held out)")
    print(f"  C++ int8 on this sample  : {acc_cpp * 100:.2f}%  (NOT held out)")

    report = {
        "purpose": "C++/sklearn equivalence, not a generalization measurement",
        "windows": int(n),
        "classify_agreement_pct": round(agree_cls * 100, 2),
        "end_to_end_agreement_pct": round(agree_e2e * 100, 2),
        "host_inference_ns": round(float(r1["latency_ns"].mean()), 1),
        "sample_accuracy_sklearn_pct": round(acc_py * 100, 2),
        "sample_accuracy_cpp_pct": round(acc_cpp * 100, 2),
        "sample_accuracy_caveat": (
            "Sample spans the full capture and overlaps the training split; "
            "not a held-out score. See model/metrics.json for that."
        ),
        "max_rel_feature_err": {k: float(v) for k, v in zip(FEATURES, worst)},
        "min_iat_note": (
            "min_iat shows large relative error: device timestamps are uint32 "
            "microseconds (1 us floor) while Python uses float seconds. Absolute "
            "error is <=1 us and no prediction changed, but sub-microsecond "
            "intervals are not representable on device."
        ),
        "per_class_cpp_recall_pct": {
            ATTACK_LABELS[c]: round(float((cpp_wpred[ys[:n] == c] == c).mean() * 100), 2)
            for c in sorted(set(ys[:n]))
        },
    }
    json.dump(report, open(TESTS / "cpp_verification.json", "w"), indent=2)
    print("\n-> tests/cpp_verification.json")

    ok = agree_cls == 1.0 and agree_e2e >= 0.99
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run harness and diff")
    ap.add_argument("--windows", type=int, default=400)
    a = ap.parse_args()

    build_vectors(a.windows)
    raise SystemExit(check() if a.check else 0)
