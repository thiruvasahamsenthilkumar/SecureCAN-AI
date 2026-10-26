#!/usr/bin/env python3
"""Live intrusion detection on a virtual CAN bus.

This is the part that makes the project feel real. Two threads share a
python-can `virtual` bus -- an in-process software bus, so it works on Windows
without SocketCAN or any hardware:

    REPLAYER  reads real HCRL frames and sends them onto the bus, pacing itself
              by the original timestamps so 10 ms really is 10 ms
    IDS       receives every frame, extracts the 16 features with the SAME
              StreamingExtractor the ESP32 mirrors, runs the model, prints

They are deliberately separate threads that only share the bus, never a Python
object. The IDS gets frames the same way the firmware does: one at a time, in
arrival order, with no idea what is coming next. Nothing in this script can see
the ground-truth label while deciding -- labels are only used afterwards, to
colour the line and count the score.

    python -m securecan.demo                      # replay a real attack capture
    python -m securecan.demo --inject dos         # clean traffic + synthetic DoS
    python -m securecan.demo --inject gear --seconds 20

--inject is the interesting mode for a demo video: it starts from the clean
normal capture and injects an attack we generate ourselves, so you can see the
exact moment the bus changes and how fast the alert follows.
"""

import argparse
import os
import queue
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import features as F  # noqa: E402
from .evaluate import BURST_GAP, detection_delay  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]

# The four attacks exactly as HCRL describes them. Used by --inject.
#   period  = how often the attacker sends, in seconds
#   can_id  = the ID it uses (None = random, i.e. Fuzzy)
INJECTORS = {
    "dos":   {"cls": 1, "can_id": 0x000, "period": 0.0003, "capture": "DoS_dataset"},
    "fuzzy": {"cls": 2, "can_id": None,  "period": 0.0005, "capture": None},
    "gear":  {"cls": 3, "can_id": 0x43F, "period": 0.0010, "capture": "gear_dataset"},
    "rpm":   {"cls": 4, "can_id": 0x316, "period": 0.0010, "capture": "RPM_dataset"},
}


def hcrl_attack_payload(kind):
    """The exact 8 bytes HCRL injected for this attack, read from the capture.

    Each HCRL spoofing campaign used ONE constant payload for every injected
    frame -- 654,897 identical frames for RPM, 597,252 for Gear. That single
    constant is what the model was trained on, so it is what the model knows.

    This is read from the dataset rather than hardcoded, because a hardcoded
    constant would quietly become a lie if the capture were ever re-parsed.
    Returns None if the capture is not available.
    """
    cap = INJECTORS[kind]["capture"]
    if cap is None:
        return None
    p = ROOT / "data" / "processed" / f"{cap}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    m = d["label"] != 0
    if not m.any():
        return None
    # the modal injected payload -- for HCRL there is only one
    vals, counts = np.unique(d["data"][m], axis=0, return_counts=True)
    return np.asarray(vals[int(counts.argmax())], dtype=np.uint8)


# ---------------------------------------------------------------------------
# terminal colour
# ---------------------------------------------------------------------------

class C:
    """ANSI codes. colorama makes these work in the Windows console too."""
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GREY = "\033[90m"


CLASS_COLOUR = [C.GREEN, C.RED, C.MAGENTA, C.YELLOW, C.CYAN]


def init_colour():
    try:
        import colorama
        colorama.just_fix_windows_console()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# where the frames come from
# ---------------------------------------------------------------------------

def load_capture(name):
    """Load one parsed .npz capture."""
    p = PROCESSED / f"{name}.npz"
    if not p.exists():
        avail = ", ".join(sorted(q.stem for q in PROCESSED.glob("*.npz")))
        raise SystemExit(f"no such capture: {p.name}\navailable: {avail}")
    d = np.load(p)
    return d["ts"], d["can_id"], d["dlc"], d["data"], d["label"]


def burst_windows(ts, label, gap=BURST_GAP):
    """Group injected frames into bursts by TIME GAP, not by label runs.

    Returns [(t_start, t_stop), ...]. The obvious implementation -- contiguous
    runs of the attack label -- is wrong, and wrong in a way that looks fine:
    during a burst the real ECU keeps transmitting, so the label alternates
    fake/real/fake frame by frame and every flip looks like a new burst. On this
    data that turns ~20 bursts into ~6,000. Injected frames arrive sub-millisecond
    apart inside a burst and seconds apart between bursts, so a 0.5 s gap splits
    them cleanly with nothing anywhere near the threshold.
    """
    atk = np.flatnonzero(label != 0)
    if not len(atk):
        return []
    breaks = np.flatnonzero(np.diff(ts[atk]) > gap) + 1
    return [(float(ts[g[0]]), float(ts[g[-1]]))
            for g in np.split(atk, breaks) if len(g)]


def pick_window(ts, label, seconds, want_attack, seed=0):
    """Choose a slice of the capture that actually shows something.

    For a replay demo we want a window containing at least one attack burst,
    otherwise the viewer watches a green wall for 15 seconds. Picks a random
    burst and starts a little before it so the transition is visible.
    """
    rng = np.random.default_rng(seed)

    if not want_attack:
        start = int(rng.integers(0, max(1, len(ts) - 1)))
    else:
        bursts = burst_windows(ts, label)
        if not bursts:
            start = 0
        else:
            t0 = bursts[int(rng.integers(0, len(bursts)))][0] - 1.5
            start = int(np.searchsorted(ts, t0, side="left"))

    stop = int(np.searchsorted(ts, ts[start] + seconds, side="right"))
    return start, min(stop, len(ts))


def real_normal(seconds, seed=0):
    """A window of REAL clean traffic, used as the baseline for --inject.

    An earlier version generated this synthetically from the learned per-ID
    periods. The timing was right but the payloads were random bytes, and the
    detector flagged 19.8% of them -- every one as Fuzzy. That was the model
    being correct, not broken: real CAN payloads are mostly constant bytes,
    counters and a couple of slowly-moving signal bytes, so uniform random bytes
    on a known ID genuinely is an anomaly.

    Rather than build a payload model good enough to fool our own detector, use
    the real thing. The frames we SYNTHESISE are the injected ones, which is the
    whole point of --inject; the baseline it is injected into should be traffic
    the car actually produced.
    """
    p = PROCESSED / "normal_run_data.npz"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run python -m securecan.parse first")

    d = np.load(p)
    ts, cid, dlc, data = d["ts"], d["can_id"], d["dlc"], d["data"]

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(ts) - 1)))
    stop = int(np.searchsorted(ts, ts[start] + seconds, side="right"))
    stop = min(stop, len(ts))

    t0 = ts[start]
    return [(float(ts[i] - t0), int(cid[i]), int(dlc[i]), data[i], 0)
            for i in range(start, stop)]


def synth_attack(const, kind, seconds, seed=0, burst=(3.0, 5.0),
                 novel_payload=False):
    """Real clean traffic plus injected frames, in HCRL-style 3-5 second bursts.

    `novel_payload` is the interesting switch, and it exists because of what
    evaluate.py found.

    HCRL's spoofing attacks each used ONE constant 8-byte payload for the whole
    campaign. A per-frame classifier trained on that can memorise the constant
    instead of learning the timing anomaly, and ours did: substituting a
    different fake value, with the timing and ids untouched, drops recall from
    100% to 0% on RPM and to 16% on Gear.

    So this flag picks which attack we are actually demonstrating:

      default           inject HCRL's own constant -- the attack the model was
                        trained on. Recall is high, and that is a fair
                        demonstration of the shipped detector against the
                        threat in its training set.

      --novel-payload   inject a different fake value. Same ids, same timing,
                        same burst structure -- only the lie is different. The
                        detector largely misses it. This is the honest limit of
                        the model, and it is far more useful to be able to show
                        and explain than to hide.
    """
    spec = INJECTORS[kind]
    rng = np.random.default_rng(seed + 1)
    rows = real_normal(seconds, seed=seed)

    # The payload the attacker sends. Two very different choices -- see above.
    base = np.zeros(8, dtype=np.uint8)
    if spec["can_id"] is not None:
        if novel_payload:
            # A real payload of this id with one byte changed. That is what a
            # plausible unseen spoof looks like: the attacker lies about the
            # value, but not with the exact bytes HCRL happened to use.
            same = [r[3] for r in rows if r[1] == spec["can_id"]]
            if same:
                base = np.asarray(same[0], dtype=np.uint8).copy()
                base[0] = 0x00 if kind == "gear" else 0xFF
        else:
            hcrl = hcrl_attack_payload(kind)
            if hcrl is not None:
                base = hcrl.copy()

    # bursts: start around 1/3 in, 3-5 s long, then a gap, repeating
    t = seconds / 3.0
    bursts = []
    while t < seconds - 1.0:
        length = rng.uniform(*burst)
        bursts.append((t, min(t + length, seconds)))
        t += length + rng.uniform(2.0, 4.0)

    for b0, b1 in bursts:
        t = b0
        while t < b1:
            t += spec["period"] * rng.uniform(0.9, 1.1)
            if spec["can_id"] is None:
                cid = int(rng.integers(0, 0x800))         # Fuzzy: anything
                payload = rng.integers(0, 256, 8).astype(np.uint8)
                dlc = int(rng.integers(1, 9))
            else:
                # DoS floods 0x000 with all-zero payload; spoofing sends `base`,
                # which is either HCRL's constant or a novel value (see above).
                cid = spec["can_id"]
                payload = base.copy()
                dlc = 8
            rows.append((t, cid, dlc, payload, spec["cls"]))

    rows.sort(key=lambda r: r[0])
    n_inj = sum(1 for r in rows if r[4] != 0)
    print(f"  {len(rows) - n_inj:,} real normal frames + {n_inj:,} injected "
          f"{kind} frames in {len(bursts)} burst(s) over {seconds:.0f} s")
    if spec["can_id"] is not None:
        hexs = " ".join(f"{b:02X}" for b in base)
        which = ("a NOVEL fake value (not the one in the training data)"
                 if novel_payload else "HCRL's own injected constant")
        print(f"  payload: {hexs}  -- {which}")
    return rows


# ---------------------------------------------------------------------------
# the two threads
# ---------------------------------------------------------------------------

def replayer(bus, rows, speed, stop_evt, meta):
    """Send frames onto the virtual bus, paced by the original timestamps.

    The sleep is relative to a single wall-clock origin rather than cumulative,
    so a slow iteration does not make the whole replay drift late.
    """
    import can

    t_origin = rows[0][0]
    wall0 = time.perf_counter()
    sent = 0

    for ts, cid, dlc, payload, lab in rows:
        if stop_evt.is_set():
            break

        target = (ts - t_origin) / speed
        lag = target - (time.perf_counter() - wall0)
        if lag > 0:
            time.sleep(lag)

        msg = can.Message(arbitration_id=int(cid),
                          data=bytes(payload[:max(1, min(int(dlc), 8))]),
                          is_extended_id=False)
        # The label rides along out-of-band so the IDS never sees it. We stash it
        # in a dict keyed by send order; the IDS looks it up only to score.
        meta.append(int(lab))
        try:
            bus.send(msg, timeout=0.5)
        except Exception:
            break
        sent += 1

    stop_evt.set()
    return sent


def ids_thread(bus, const, model, stop_evt, out_q, scale, zero_point):
    """Receive, extract features, classify. One frame at a time, like the ESP32."""
    ex = F.StreamingExtractor(const)
    seen = 0

    while not stop_evt.is_set() or True:
        msg = bus.recv(timeout=0.25)
        if msg is None:
            if stop_evt.is_set():
                break
            continue

        data = np.zeros(8, dtype=np.uint8)
        d = bytes(msg.data)[:8]
        data[:len(d)] = np.frombuffer(d, dtype=np.uint8)

        t0 = time.perf_counter()
        feats = ex.push(msg.timestamp, msg.arbitration_id, len(d), data)

        if scale is not None:
            q = np.clip(np.round(feats / scale + zero_point), -128, 127)
            probs = model(q.astype(np.int8))
        else:
            probs = model(feats)

        us = (time.perf_counter() - t0) * 1e6
        cls = int(np.argmax(probs))

        out_q.put((seen, msg.timestamp, msg.arbitration_id, len(d), data,
                   cls, float(probs[cls]), us, feats))
        seen += 1

    out_q.put(None)


# ---------------------------------------------------------------------------
# model wrappers -- keras or the actual int8 tflite
# ---------------------------------------------------------------------------

def load_model(use_int8):
    """Return (predict_one, in_scale, in_zero).

    predict_one takes one feature row and returns 5 probabilities. For --int8 it
    runs the exact FlatBuffer that is compiled into the firmware, so the demo and
    the ESP32 are provably answering with the same weights.
    """
    import tensorflow as tf

    if use_int8:
        blob = (RESULTS / "mlp_int8.tflite").read_bytes()
        interp = tf.lite.Interpreter(model_content=blob)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        in_scale, in_zero = inp["quantization"]
        o_scale, o_zero = out["quantization"]

        def predict_one(q):
            interp.set_tensor(inp["index"], q.reshape(1, -1))
            interp.invoke()
            raw = interp.get_tensor(out["index"])[0].astype(np.float32)
            return (raw - o_zero) * o_scale

        predict_one(np.zeros(F.N_FEATURES, dtype=np.int8))   # warm up
        return predict_one, in_scale, in_zero

    keras = tf.keras.models.load_model(RESULTS / "mlp.keras")
    # Build a concrete tf.function once: calling model() per frame via the Keras
    # __call__ path rebuilds a graph every time and is ~50x slower.
    fn = tf.function(
        lambda x: keras(x, training=False),
        input_signature=[tf.TensorSpec([1, F.N_FEATURES], tf.float32)],
    )

    def predict_one(feats):
        return fn(feats.reshape(1, -1)).numpy()[0]

    # Trace the graph before timing starts. The very first call costs ~70 ms of
    # one-off tracing, which would otherwise land on frame 0 and skew the
    # latency figures the demo prints.
    predict_one(np.zeros(F.N_FEATURES, dtype=np.float32))
    return predict_one, None, None


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="RPM_dataset",
                    help="parsed capture to replay (ignored with --inject)")
    ap.add_argument("--inject", choices=sorted(INJECTORS),
                    help="synthesise clean traffic plus this attack instead")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="how much bus time to play")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier (2.0 = twice real time)")
    ap.add_argument("--int8", action="store_true",
                    help="run the quantised model that ships to the ESP32")
    ap.add_argument("--quiet-normal", action="store_true",
                    help="only print alerts, not every normal frame")
    ap.add_argument("--max-lines", type=int, default=400,
                    help="stop printing after this many lines (still scores all)")
    ap.add_argument("--novel-payload", action="store_true",
                    help="with --inject: send a fake value the model was NOT "
                         "trained on. Demonstrates the payload-memorisation "
                         "limit -- recall collapses, which is the honest result")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    init_colour()
    import can

    const = F.Constants.load(RESULTS / "constants.json")

    # --- build the frame list ----------------------------------------------
    print(f"\n{C.BOLD}SecureCAN AI -- live detection on a virtual CAN bus{C.RESET}")
    print(f"{C.DIM}{'-' * 74}{C.RESET}")

    if args.inject:
        print(f"  source : synthetic clean traffic + injected {args.inject.upper()}")
        rows = synth_attack(const, args.inject, args.seconds, seed=args.seed,
                            novel_payload=args.novel_payload)
    else:
        ts, cid, dlc, data, label = load_capture(args.capture)
        start, stop = pick_window(ts, label, args.seconds, True, seed=args.seed)
        rows = [(float(ts[i]), int(cid[i]), int(dlc[i]), data[i], int(label[i]))
                for i in range(start, stop)]
        print(f"  source : {args.capture} frames {start:,}-{stop:,}  "
              f"({len(rows):,} frames, {(ts[stop - 1] - ts[start]):.1f} s)")

    if not rows:
        raise SystemExit("no frames to replay")

    model, scale, zero = load_model(args.int8)
    print(f"  model  : {'INT8 TFLite (the ESP32 build)' if args.int8 else 'float32 Keras'}")
    print(f"  speed  : {args.speed}x real time")
    print(f"{C.DIM}{'-' * 74}{C.RESET}\n")

    # --- wire up the bus ---------------------------------------------------
    # Two separate Bus objects on the same channel: that is how python-can's
    # virtual interface links a sender and a receiver.
    tx = can.interface.Bus(channel="securecan", interface="virtual",
                           receive_own_messages=False)
    rx = can.interface.Bus(channel="securecan", interface="virtual")

    stop_evt = threading.Event()
    out_q = queue.Queue()
    meta = []

    t_rep = threading.Thread(target=replayer,
                             args=(tx, rows, args.speed, stop_evt, meta),
                             daemon=True)
    t_ids = threading.Thread(target=ids_thread,
                             args=(rx, const, model, stop_evt, out_q, scale, zero),
                             daemon=True)

    hdr = (f"  {'bus t':>8}  {'ID':>5} {'DLC':>3}  {'payload':<24}"
           f"{'verdict':<9}{'conf':>6}{'us':>8}")
    print(C.BOLD + hdr + C.RESET)
    print(f"{C.DIM}  {'-' * 72}{C.RESET}")

    wall0 = time.perf_counter()
    t_ids.start()
    t_rep.start()

    # --- consume the IDS output -------------------------------------------
    truth, pred, bus_t = [], [], []
    lines = 0
    alerts = Counter()
    lat = []
    t_origin = rows[0][0]

    while True:
        item = out_q.get()
        if item is None:
            break

        i, ts_f, cid, dlc, data, cls, conf, us, feats = item
        lab = meta[i] if i < len(meta) else 0
        truth.append(lab)
        pred.append(cls)
        # The ORIGINAL bus timestamp, not the arrival time: burst grouping uses a
        # 0.5 s gap, and with --speed 2 the arrival gaps would be half as wide.
        bus_t.append(rows[i][0] if i < len(rows) else 0.0)
        lat.append(us)
        if cls != 0:
            alerts[cls] += 1

        if lines >= args.max_lines:
            continue
        if args.quiet_normal and cls == 0 and lab == 0:
            continue

        col = CLASS_COLOUR[cls]
        pay = " ".join(f"{b:02X}" for b in data[:max(1, min(dlc, 8))])

        # a small marker when our verdict and the ground truth disagree
        if cls == 0 and lab != 0:
            mark = f"{C.RED} MISS{C.RESET}"
        elif cls != 0 and lab == 0:
            mark = f"{C.YELLOW} FP{C.RESET}"
        elif cls != 0 and cls != lab:
            mark = f"{C.DIM} (was {CLASS_NAMES[lab]}){C.RESET}"
        else:
            mark = ""

        tag = "OK " if cls == 0 else "ALERT"
        print(f"  {ts_f - t_origin:>8.3f}  {cid:>#05X} {dlc:>3}  {pay:<24}"
              f"{col}{tag:<9}{C.RESET}{conf:>6.2f}{us:>8.0f}{mark}")
        lines += 1

    t_rep.join(timeout=2)
    elapsed = time.perf_counter() - wall0
    tx.shutdown()
    rx.shutdown()

    # --- score -------------------------------------------------------------
    truth = np.array(truth)
    pred = np.array(pred)
    n = len(truth)

    print(f"\n{C.DIM}{'-' * 74}{C.RESET}")
    print(f"{C.BOLD}  summary{C.RESET}   {n:,} frames in {elapsed:.1f} s wall "
          f"({n / max(elapsed, 1e-9):,.0f} frames/s)")

    if n:
        atk_t = truth != 0
        atk_p = pred != 0
        tp = int((atk_t & atk_p).sum())
        fn_ = int((atk_t & ~atk_p).sum())
        fp = int((~atk_t & atk_p).sum())
        tn = int((~atk_t & ~atk_p).sum())

        recall = tp / max(tp + fn_, 1)
        prec = tp / max(tp + fp, 1)
        fpr = fp / max(fp + tn, 1)

        print(f"    ground truth : {int(atk_t.sum()):,} attack / "
              f"{int((~atk_t).sum()):,} normal")
        print(f"    detected     : {tp:,} true positives, {fn_:,} missed")
        print(f"    false alarms : {fp:,} of {int((~atk_t).sum()):,} normal frames "
              f"({fpr * 100:.3f}%)")
        print(f"    recall {recall * 100:6.2f}%   precision {prec * 100:6.2f}%")

        if alerts:
            named = "  ".join(f"{CLASS_NAMES[c]}={v:,}" for c, v in
                              sorted(alerts.items()))
            print(f"    alerts by class: {named}")

        # Per-burst detection delay, scored by the SAME function that produces
        # the numbers in results/metrics.json -- one definition, so the demo
        # cannot quietly disagree with the reported metrics.
        if atk_t.any():
            dd = detection_delay(truth, pred, np.array(bus_t))
            print(f"    bursts       : {dd['bursts']} injected burst(s), "
                  f"{dd['detected']} detected, {dd['missed']} missed")
            print(f"    alert delay  : median {dd['frames_to_alert_median']} "
                  f"frame(s) after the first injected frame "
                  f"(max {dd['frames_to_alert_max']}), "
                  f"{dd['median_ms']:.3f} ms median")

    if lat:
        lat = np.array(lat)
        print(f"    inference    : median {np.median(lat):.0f} us, "
              f"p95 {np.percentile(lat, 95):.0f} us  "
              f"{C.DIM}(Python on a laptop; ESP32 figure is in the README){C.RESET}")

    print()


if __name__ == "__main__":
    main()
