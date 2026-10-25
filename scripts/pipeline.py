#!/usr/bin/env python3
"""SecureCAN AI - Full pipeline: data -> features -> model -> quantized C++ export."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
MODEL = ROOT / "model"
DATA.mkdir(exist_ok=True)
MODEL.mkdir(exist_ok=True)

ATTACK_LABELS = {0: "NORMAL", 1: "DOS", 2: "FUZZY", 3: "FAKE_RPM", 4: "FAKE_GEAR"}
WINDOW = 50  # messages per detection window


ECU_IDS = [0x0A0, 0x130, 0x18F, 0x260, 0x2C0, 0x316, 0x43F, 0x545]


def _normal_msg(rng, t, i):
    """One benign message. Each ECU has its own signal ranges + drift."""
    cid = ECU_IDS[i % len(ECU_IDS)]
    base = (cid & 0xFF) % 90
    # slow signal drift so payloads are not stationary
    drift = 25 * np.sin(t / 7.0 + (cid % 13))
    data = np.clip(rng.normal(base + drift, 22, 8), 0, 255).astype(int)
    return [t, cid, int(rng.choice([6, 7, 8], p=[.1, .2, .7])), *data, 0]


def generate_can_traffic():
    """HCRL-style capture: continuous benign stream with attacks *injected into*
    it in bursts, so attack windows always contain real background traffic."""
    rng = np.random.default_rng(42)
    rows = []
    t = 0.0

    # schedule attack bursts across the capture; gaps stay purely benign
    bursts = []
    cursor = 8.0
    for _ in range(60):
        kind = int(rng.integers(1, 5))
        dur = float(rng.uniform(0.4, 2.2))
        bursts.append((cursor, cursor + dur, kind))
        cursor += dur + float(rng.uniform(1.5, 5.0))
    burst_end = cursor

    i = 0
    while t < burst_end + 8.0:
        t += float(rng.normal(0.010, 0.0015))
        i += 1
        rows.append(_normal_msg(rng, t, i))

        active = [b for b in bursts if b[0] <= t <= b[1]]
        if not active:
            continue
        _, _, kind = active[0]

        if kind == 1:  # DoS: flood, but period jitters and ID varies run to run
            period = float(rng.uniform(0.0003, 0.0012))
            n = int(0.010 / period)
            flood_id = int(rng.choice([0x000, 0x001, 0x7FF]))
            tt = t
            for _ in range(n):
                tt += period * float(rng.uniform(0.7, 1.3))
                pay = rng.integers(0, 40, 8) if rng.random() < .5 else np.zeros(8, int)
                rows.append([tt, flood_id, 8, *pay, 1])

        elif kind == 2:  # Fuzzy: random, but only ~40% of bytes are wild
            for _ in range(int(rng.integers(2, 6))):
                t += float(rng.normal(0.010, 0.004))
                cid = int(rng.integers(0, 0x800))
                pay = np.clip(rng.normal(90, 30, 8), 0, 255).astype(int)
                mask = rng.random(8) < 0.4
                pay[mask] = rng.integers(0, 256, mask.sum())
                rows.append([t, cid, int(rng.integers(0, 9)), *pay, 2])

        elif kind == 3:  # Fake RPM on 0x316: plausible-looking but inconsistent
            for _ in range(int(rng.integers(2, 5))):
                t += float(rng.normal(0.010, 0.0015))
                pay = np.clip(rng.normal(70, 20, 8), 0, 255).astype(int)
                rpm = int(rng.choice([rng.integers(200, 255), rng.integers(0, 20)]))
                pay[0], pay[1] = rpm, int(np.clip(rpm + rng.normal(0, 4), 0, 255))
                rows.append([t, 0x316, 8, *pay, 3])

        elif kind == 4:  # Fake Gear on 0x43F: invalid nibble, rest looks normal
            for _ in range(int(rng.integers(2, 5))):
                t += float(rng.normal(0.010, 0.0015))
                pay = np.clip(rng.normal(63, 20, 8), 0, 255).astype(int)
                pay[2] = int(rng.choice([0x0D, 0x0E, 0x0F]))
                pay[3] = int(rng.choice([0xF0, 0xE0, 0xD0]))
                rows.append([t, 0x43F, 8, *pay, 4])

    cols = ["ts", "can_id", "dlc"] + [f"b{i}" for i in range(8)] + ["label"]
    df = pd.DataFrame(rows, columns=cols).sort_values("ts").reset_index(drop=True)
    df.to_csv(DATA / "can_traffic.csv", index=False)
    print(f"[data] {len(df)} messages -> data/can_traffic.csv")
    print(df.label.value_counts().rename(ATTACK_LABELS).to_string())
    return df


FEATURES = [
    "msg_rate", "mean_iat", "std_iat", "min_iat",
    "unique_ids", "id_entropy", "dominant_id_ratio",
    "mean_dlc", "mean_byte", "std_byte", "max_byte", "byte_entropy",
    "saturated_ratio", "zero_ratio",
]


def extract_features(df):
    """Sliding-window feature extraction. Returns (X, y)."""
    X, y = [], []
    arr = df.to_numpy()
    ts, cid, dlc = arr[:, 0], arr[:, 1].astype(int), arr[:, 2]
    bytes_ = arr[:, 3:11]
    labels = arr[:, 11].astype(int)

    for s in range(0, len(df) - WINDOW, WINDOW // 2):
        e = s + WINDOW
        w_ts, w_cid, w_dlc = ts[s:e], cid[s:e], dlc[s:e]
        w_b = bytes_[s:e]

        iat = np.diff(w_ts)
        iat = iat[iat > 0] if (iat > 0).any() else np.array([1e-6])
        span = max(w_ts[-1] - w_ts[0], 1e-6)

        _, counts = np.unique(w_cid, return_counts=True)
        p = counts / counts.sum()

        bflat = w_b.ravel()
        bhist = np.bincount(bflat.astype(int), minlength=256) / bflat.size
        bp = bhist[bhist > 0]

        X.append([
            WINDOW / span,
            iat.mean(), iat.std(), iat.min(),
            len(counts), -(p * np.log2(p)).sum(), counts.max() / WINDOW,
            w_dlc.mean(), bflat.mean(), bflat.std(), bflat.max(),
            -(bp * np.log2(bp)).sum(),
            (bflat >= 250).mean(), (bflat == 0).mean(),
        ])
        # window label = majority attack class if any attack present
        wl = labels[s:e]
        atk = wl[wl > 0]
        y.append(np.bincount(atk).argmax() if len(atk) > WINDOW * 0.3 else 0)

    return np.array(X, dtype=np.float32), np.array(y)


def train(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    names = [ATTACK_LABELS[i] for i in sorted(set(y))]
    report = classification_report(yte, pred, target_names=names, digits=4)
    cm = confusion_matrix(yte, pred)

    print(f"\n[model] test accuracy: {acc:.4f}")
    print(report)

    joblib.dump(clf, MODEL / "detector.joblib")
    (MODEL / "feature_names.txt").write_text("\n".join(FEATURES))
    return clf, acc, report, cm, (Xte, yte)


def export_cpp(clf, X_ref):
    """Flatten forest to INT8-quantized arrays in a C++ header."""
    # per-feature quantization scale: map observed range to int8
    lo, hi = X_ref.min(0), X_ref.max(0)
    scale = np.where(hi - lo > 0, 254.0 / (hi - lo), 1.0)

    nodes = []  # (feature, threshold_q, left, right, class)
    offsets = []
    for est in clf.estimators_:
        t = est.tree_
        offsets.append(len(nodes))
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                nodes.append((-1, 0, -1, -1, int(np.argmax(t.value[i][0]))))
            else:
                f = int(t.feature[i])
                thr_q = int(np.clip((t.threshold[i] - lo[f]) * scale[f] - 127, -128, 127))
                nodes.append((f, thr_q, int(t.children_left[i]), int(t.children_right[i]), -1))

    h = ["#pragma once", "#include <stdint.h>", "",
         f"#define N_FEATURES {len(FEATURES)}",
         f"#define N_TREES {len(clf.estimators_)}",
         f"#define N_NODES {len(nodes)}",
         f"#define N_CLASSES {len(clf.classes_)}", "",
         "// per-feature dequant: q = (x - lo) * scale - 127",
         "static const float FEAT_LO[N_FEATURES] = {" + ",".join(f"{v:.6g}f" for v in lo) + "};",
         "static const float FEAT_SCALE[N_FEATURES] = {" + ",".join(f"{v:.6g}f" for v in scale) + "};", "",
         "static const int8_t  NODE_FEATURE[N_NODES]  = {" + ",".join(str(n[0]) for n in nodes) + "};",
         "static const int8_t  NODE_THRESH[N_NODES]   = {" + ",".join(str(n[1]) for n in nodes) + "};",
         "static const int16_t NODE_LEFT[N_NODES]     = {" + ",".join(str(n[2]) for n in nodes) + "};",
         "static const int16_t NODE_RIGHT[N_NODES]    = {" + ",".join(str(n[3]) for n in nodes) + "};",
         "static const int8_t  NODE_CLASS[N_NODES]    = {" + ",".join(str(n[4]) for n in nodes) + "};",
         "static const int16_t TREE_OFFSET[N_TREES]   = {" + ",".join(str(o) for o in offsets) + "};", ""]

    path = MODEL / "detector_quantized.h"
    path.write_text("\n".join(h))

    orig = (MODEL / "detector.joblib").stat().st_size
    quant = path.stat().st_size
    print(f"[quant] {len(nodes)} nodes | joblib {orig/1024:.1f} KB -> header {quant/1024:.1f} KB"
          f" | runtime arrays ~{len(nodes)*7/1024:.1f} KB")
    return orig, quant, len(nodes)


if __name__ == "__main__":
    df = generate_can_traffic()
    X, y = extract_features(df)
    print(f"[features] {X.shape[0]} windows x {X.shape[1]} features")
    clf, acc, report, cm, (Xte, yte) = train(X, y)
    orig, quant, n_nodes = export_cpp(clf, X)

    json.dump({
        "accuracy": float(acc),
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_nodes": int(n_nodes),
        "joblib_kb": round(orig / 1024, 1),
        "header_kb": round(quant / 1024, 1),
        "runtime_kb": round(n_nodes * 7 / 1024, 1),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }, open(MODEL / "metrics.json", "w"), indent=2)
    print("\n[done] model/ populated")
