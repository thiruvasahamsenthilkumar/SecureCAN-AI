#!/usr/bin/env python3
"""Replay CAN traffic over a virtual bus and detect intrusions in real time.

A producer thread writes the generated capture onto a python-can `virtual` bus
(cross-platform, unlike socketcan) at its recorded timing. A consumer thread
reads frames back off the bus, maintains a sliding window, and classifies each
window with the trained forest -- so detection runs on frames that actually
travelled the bus rather than on a dataframe slice.

Writes demo/detections.jsonl (one record per window) and demo/summary.json.
"""

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import can
import joblib
import numpy as np
import pandas as pd

from pipeline import ATTACK_LABELS, FEATURES, WINDOW

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "demo"


def window_features(win, speed=1.0):
    """Feature vector for one window. Mirrors extract_features in pipeline.py.

    Timestamps are the consumer's own receive clock, because python-can's virtual
    bus overwrites Message.timestamp with wall-clock time on receive. Replaying at
    `speed`x compresses every interval by that factor, so timings are scaled back
    up to the capture's timeline -- otherwise a 40x replay reports 40x the message
    rate and the model, trained on real-time rates, calls everything DoS.
    """
    ts = np.array([m[0] for m in win]) * speed
    cid = np.array([m[1] for m in win])
    dlc = np.array([m[2] for m in win], dtype=float)
    b = np.array([m[3] for m in win], dtype=float)

    iat = np.diff(ts)
    iat = iat[iat > 0] if (iat > 0).any() else np.array([1e-6])
    span = max(ts[-1] - ts[0], 1e-6)

    _, counts = np.unique(cid, return_counts=True)
    p = counts / counts.sum()

    bflat = b.ravel()
    bhist = np.bincount(bflat.astype(int), minlength=256) / bflat.size
    bp = bhist[bhist > 0]

    return np.array([
        WINDOW / span,
        iat.mean(), iat.std(), iat.min(),
        len(counts), -(p * np.log2(p)).sum(), counts.max() / WINDOW,
        dlc.mean(), bflat.mean(), bflat.std(), bflat.max(),
        -(bp * np.log2(bp)).sum(),
        (bflat >= 250).mean(), (bflat == 0).mean(),
    ], dtype=np.float32)


def producer(bus, df, speed, stop):
    """Write frames onto the bus, preserving relative inter-frame timing."""
    ts = df["ts"].to_numpy()
    base = time.perf_counter()
    t0 = ts[0]

    for row in df.itertuples(index=False):
        if stop.is_set():
            break
        # pace to the capture's own timeline, compressed by `speed`
        target = (row.ts - t0) / speed
        lag = target - (time.perf_counter() - base)
        if lag > 0:
            time.sleep(lag)

        data = bytes(int(getattr(row, f"b{i}")) & 0xFF for i in range(8))
        try:
            bus.send(can.Message(arbitration_id=int(row.can_id) & 0x7FF,
                                 data=data[: int(row.dlc)] or b"",
                                 is_extended_id=False,
                                 timestamp=row.ts))
        except can.CanError:
            break
    stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=25.0,
                    help="replay speed multiplier (25 = 25x faster than real time)")
    ap.add_argument("--limit", type=int, default=20000, help="frames to replay")
    args = ap.parse_args()

    DEMO.mkdir(exist_ok=True)
    clf = joblib.load(ROOT / "model" / "detector.joblib")
    df = pd.read_csv(ROOT / "data" / "can_traffic.csv").head(args.limit)

    truth = df["label"].to_numpy()
    print(f"replaying {len(df)} frames at {args.speed:.0f}x over a virtual CAN bus")

    bus = can.interface.Bus(channel="securecan", interface="virtual", receive_own_messages=True)
    stop = threading.Event()
    th = threading.Thread(target=producer, args=(bus, df, args.speed, stop), daemon=True)
    th.start()

    win = deque(maxlen=WINDOW)
    records, latencies = [], []
    n_rx = 0
    wall0 = time.perf_counter()

    while not (stop.is_set() and n_rx >= len(df)):
        msg = bus.recv(timeout=0.5)
        if msg is None:
            if stop.is_set():
                break
            continue

        n_rx += 1
        rx = time.perf_counter() - wall0
        payload = list(msg.data) + [0] * (8 - len(msg.data))
        win.append((rx, msg.arbitration_id, msg.dlc, payload))

        # classify on a half-window stride, matching training
        if len(win) == WINDOW and n_rx % (WINDOW // 2) == 0:
            t0 = time.perf_counter()
            feats = window_features(list(win), args.speed)
            # one forest walk, not two: predict() would re-traverse all 40 trees
            proba = clf.predict_proba(feats.reshape(1, -1))[0]
            pred = int(proba.argmax())
            dt_ms = (time.perf_counter() - t0) * 1e3
            latencies.append(dt_ms)

            seg = truth[max(0, n_rx - WINDOW): n_rx]
            atk = seg[seg > 0]
            actual = int(np.bincount(atk).argmax()) if len(atk) > WINDOW * 0.3 else 0

            rec = {
                "frame": n_rx,
                "bus_time": round(rx * args.speed, 4),
                "predicted": ATTACK_LABELS[pred],
                "actual": ATTACK_LABELS[actual],
                "correct": pred == actual,
                "confidence": round(float(proba[pred]), 4),
                "latency_ms": round(dt_ms, 3),
                "msg_rate": round(float(feats[0]), 1),
                "features": {k: round(float(v), 4) for k, v in zip(FEATURES, feats)},
            }
            records.append(rec)

            if pred != 0:
                flag = "" if rec["correct"] else f"  [wrong: actually {rec['actual']}]"
                print(f"  [{rec['bus_time']:8.2f}s] {ATTACK_LABELS[pred]:<10} "
                      f"conf {proba[pred]:.2f}  {feats[0]:7.0f} msg/s  "
                      f"{dt_ms:.2f} ms{flag}")

    wall = time.perf_counter() - wall0
    bus.shutdown()

    correct = sum(r["correct"] for r in records)
    n = len(records) or 1
    alerts = [r for r in records if r["predicted"] != "NORMAL"]
    attack_windows = [r for r in records if r["actual"] != "NORMAL"]
    benign = [r for r in records if r["actual"] == "NORMAL"]
    false_pos = [r for r in benign if r["predicted"] != "NORMAL"]

    per_class = {}
    for cid, name in ATTACK_LABELS.items():
        rel = [r for r in records if r["actual"] == name]
        if rel:
            per_class[name] = {
                "windows": len(rel),
                "recall_pct": round(100 * sum(r["correct"] for r in rel) / len(rel), 2),
            }

    summary = {
        "frames_replayed": n_rx,
        "wall_seconds": round(wall, 2),
        "throughput_fps": round(n_rx / wall, 1),
        "windows_classified": len(records),
        "accuracy_pct": round(100 * correct / n, 2),
        "alerts_raised": len(alerts),
        "attack_windows": len(attack_windows),
        "false_positives": len(false_pos),
        "false_positive_rate_pct": round(100 * len(false_pos) / max(len(benign), 1), 2),
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 3),
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "max": round(float(np.max(latencies)), 3),
        },
        "per_class": per_class,
        "note": "Latency is host Python/sklearn, not ESP32. See wokwi/ for on-device timing.",
    }

    with open(DEMO / "detections.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    json.dump(summary, open(DEMO / "summary.json", "w"), indent=2)

    # Bake results into a JS global. The dashboard is opened over file://, where
    # fetch() of a sibling JSON is blocked by CORS, so a script tag is the only
    # way to hand it real data without running a server.
    model_metrics = json.load(open(ROOT / "model" / "metrics.json"))
    cpp = ROOT / "tests" / "cpp_verification.json"

    # Only the fields the dashboard actually reads. Shipping all 14 features per
    # record made data.js ~350 KB and stalled the renderer.
    slim = [{
        "frame": r["frame"],
        "bus_time": r["bus_time"],
        "predicted": r["predicted"],
        "actual": r["actual"],
        "correct": r["correct"],
        "confidence": r["confidence"],
        "latency_ms": r["latency_ms"],
        "msg_rate": r["msg_rate"],
        "byte_entropy": r["features"]["byte_entropy"],
    } for r in records]

    payload = {
        "summary": summary,
        "detections": slim,
        "model": {
            "heldout_accuracy_pct": round(model_metrics["accuracy"] * 100, 2),
            "n_trees": 40,
            "n_nodes": model_metrics["n_nodes"],
            "runtime_kb": model_metrics["runtime_kb"],
        },
        "cpp": json.load(open(cpp)) if cpp.exists() else None,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ROOT / "frontend" / "data.js").write_text(
        "// Generated by scripts/live_demo.py -- do not edit.\n"
        "window.SECURECAN = " + json.dumps(payload, indent=1) + ";\n")

    print(f"\n  frames        {n_rx} in {wall:.1f}s ({summary['throughput_fps']:.0f} f/s)")
    print(f"  windows       {len(records)}  accuracy {summary['accuracy_pct']}%")
    print(f"  alerts        {len(alerts)}  on {len(attack_windows)} attack windows")
    print(f"  false pos     {len(false_pos)}/{len(benign)} benign "
          f"({summary['false_positive_rate_pct']}%)")
    print(f"  latency       mean {summary['latency_ms']['mean']} ms  "
          f"p95 {summary['latency_ms']['p95']} ms")
    for name, s in per_class.items():
        print(f"    {name:<10} {s['recall_pct']:6.2f}%  (n={s['windows']})")
    print("\n-> demo/detections.jsonl, demo/summary.json")


if __name__ == "__main__":
    main()
