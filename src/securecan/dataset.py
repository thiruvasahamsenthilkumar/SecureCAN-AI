#!/usr/bin/env python3
"""Assemble the parsed captures into train / validation / test matrices.

Two rules drive everything here, and both exist to stop the score from being a lie:

1. SPLIT BY TIME, NEVER SHUFFLE. Consecutive CAN frames are strongly correlated --
   the frame 2 ms after an injected frame is almost certainly also injected. A
   shuffled split puts near-identical neighbours on both sides of the boundary, and
   the model gets to memorise rather than generalise. The published accuracy would
   look excellent and mean nothing.

2. FIT ON TRAINING DATA ONLY. The per-ID periods and the normalisation ceilings are
   learned from the training portion, and only from frames labelled normal.
   Validation and test frames never influence them.

Feature extraction runs over each file *whole*, in recorded order, and the split is
applied afterwards by index. That matters because the features are stateful: cutting
the file first would leave the first frames of validation with no per-ID history,
which is an artefact of our slicing rather than anything real.

THE CLEAN-TAIL TRAP
-------------------
HCRL stopped injecting before the end of every attack recording. The last 20-27% of
each attack file is pure normal traffic:

    DoS   last injected frame at 73.0% of the file
    Fuzzy                         76.3%
    gear                          79.4%
    RPM                           80.3%

A single 70/15/15 index split over the whole file therefore lands the ENTIRE test
split inside that clean tail. The first time this ran, the test split was 2,633,754
frames with zero attacks in it -- the confusion matrix had one non-empty row and
"99.996% accuracy" meant nothing at all.

The fix is to treat each file as two contiguous regions, the attack-bearing part and
the clean tail, and split each one 70/15/15 by time. Every split then sees the same
class balance, the ordering inside each region is still chronological, and the clean
tail becomes what it actually is: extra genuine normal traffic, which makes the
false-positive-rate estimate better rather than worse.
"""

from pathlib import Path

import numpy as np

from . import features as F

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

TRAIN_END, VAL_END = 0.70, 0.85

F_NORMAL = 0
CACHE = PROCESSED / "splits"


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def _segments(label):
    """Contiguous regions of a file that get split independently.

    Returns [(start, stop), ...]. For an attack file that is the injected region
    followed by the clean tail; for the normal capture it is the whole file.
    """
    atk = np.flatnonzero(label != F_NORMAL)
    if not len(atk):
        return [(0, len(label))]

    end = int(atk[-1]) + 1
    if end >= len(label):
        return [(0, len(label))]
    return [(0, end), (end, len(label))]


def split_slices(label):
    """name -> list of slices, chronological inside each region.

    One list per split because a file contributes two disjoint ranges to each
    (one from the injected region, one from the clean tail).
    """
    out = {"train": [], "val": [], "test": []}

    for lo, hi in _segments(label):
        n = hi - lo
        a = lo + int(n * TRAIN_END)
        b = lo + int(n * VAL_END)
        out["train"].append(slice(lo, a))
        out["val"].append(slice(a, b))
        out["test"].append(slice(b, hi))

    return out


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

def fit_constants(files=None):
    """Learn feature constants from the training portion of every capture.

    Fitted on the first training slice only -- the start of the attack-bearing
    region. That slice is contiguous, which matters: np.diff over two ranges
    joined across a gap would invent one bogus multi-minute interval. It also
    contains the overwhelming majority of the normal frames, so nothing is lost.
    """
    files = files or sorted(PROCESSED.glob("*.npz"))
    per_file = []

    for p in files:
        d = np.load(p)
        sl = split_slices(d["label"])["train"][0]
        c = F.fit(d["ts"][sl], d["can_id"][sl], d["label"][sl])
        print(f"  {p.stem:<22} {len(c.known_ids):>3} known IDs  "
              f"dt_prev_p99 {c.dt_prev_clip * 1e3:6.2f} ms  "
              f"dt_same_p99 {c.dt_same_clip * 1e3:7.2f} ms  "
              f"bus_rate_p99 {c.bus_rate_clip:5.0f}")
        per_file.append(c)

    const = F.merge(per_file)
    print(f"  merged                 {len(const.known_ids):>3} known IDs")
    return const


# ---------------------------------------------------------------------------
# building the matrices
# ---------------------------------------------------------------------------

def build(const, normal_keep=0.15, seed=0, files=None):
    """Extract features for every capture and return the three splits.

    Normal frames are subsampled in train and validation only: they outnumber
    attack frames ~6:1 and keeping all 14 million would cost memory without adding
    information. The TEST SPLIT IS KEPT COMPLETE -- subsampling it would change the
    class balance and make the false-positive rate meaningless, since FPR is
    measured per normal frame.
    """
    rng = np.random.default_rng(seed)
    files = files or sorted(PROCESSED.glob("*.npz"))
    out = {k: ([], []) for k in ("train", "val", "test")}

    for p in files:
        d = np.load(p)
        ts, can_id, dlc, data, label = (d["ts"], d["can_id"], d["dlc"],
                                        d["data"], d["label"])

        # features over the whole file, in order, so per-ID history is continuous
        X = F.transform(ts, can_id, dlc, data, const)
        slices = split_slices(label)

        shown = []
        for name in ("train", "val", "test"):
            for sl in slices[name]:
                Xs, ys = X[sl], label[sl]

                if name != "test" and normal_keep < 1.0:
                    normal = ys == F_NORMAL
                    keep = ~normal | (rng.random(len(ys)) < normal_keep)
                    Xs, ys = Xs[keep], ys[keep]

                out[name][0].append(Xs)
                out[name][1].append(ys)

            shown.append(
                f"{name}={sum(s.stop - s.start for s in slices[name]):,}")

        print(f"  {p.stem:<22} {len(ts):>9,} frames  "
              f"{len(slices['train'])} region(s)  " + "  ".join(shown))

    splits = {}
    for name, (xs, ys) in out.items():
        X = np.concatenate(xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        splits[name] = (X, y)
        counts = np.bincount(y, minlength=5)
        print(f"  {name:<6} {len(y):>9,}  " +
              "  ".join(f"{n}={c:,}" for n, c in
                        zip(["norm", "dos", "fuzz", "gear", "rpm"], counts)) +
              f"   attack {(y != 0).mean() * 100:5.2f}%")

    return splits


def load_or_build(const, rebuild=False, **kw):
    """Build the splits once and cache them.

    Feature extraction over 17.5 M frames takes a few minutes, and train.py,
    evaluate.py and quantize.py all need the same matrices. The cache is keyed to
    nothing -- delete it or pass rebuild=True if the features change.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    names = ("train", "val", "test")

    if not rebuild and all((CACHE / f"{n}_X.npy").exists() for n in names):
        print("[cache] loading data/processed/splits/")
        return {n: (np.load(CACHE / f"{n}_X.npy"),
                    np.load(CACHE / f"{n}_y.npy")) for n in names}

    splits = build(const, **kw)
    for n, (X, y) in splits.items():
        np.save(CACHE / f"{n}_X.npy", X)
        np.save(CACHE / f"{n}_y.npy", y)
    print(f"[cache] wrote {CACHE.relative_to(ROOT)}")
    return splits


def test_frames_for_delay(const, path, n=250_000):
    """Raw test-split frames for the detection-delay metric.

    Delay is measured in wall-clock time from the first injected frame of a burst
    to the first frame the model flags, so it needs the original timestamps
    alongside the features -- which build() throws away.

    Takes the test slice of the ATTACK-BEARING region specifically. The clean
    tail's test slice has no bursts in it, so there would be nothing to time.
    """
    d = np.load(path)
    sl = split_slices(d["label"])["test"][0]

    # cap the length so this stays quick; bursts are only seconds apart
    stop = min(sl.stop, sl.start + n)
    sl = slice(sl.start, stop)

    X = F.transform(d["ts"], d["can_id"], d["dlc"], d["data"], const)[sl]
    return X, d["label"][sl], d["ts"][sl]
