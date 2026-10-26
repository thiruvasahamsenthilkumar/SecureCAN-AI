#!/usr/bin/env python3
"""Turn a stream of CAN frames into 16 features per frame.

THIS FILE IS THE SINGLE SOURCE OF TRUTH. firmware/esp32/src/features.h must produce
bit-identical output; tests/test_parity.py checks that.

Design constraint: everything here has to be computable on a microcontroller that
sees frames one at a time and can never look ahead. So every feature depends only
on the current frame plus a small fixed-size memory of the past:

  - a per-ID state table (last timestamp, last payload, learned normal period)
  - a ring buffer of recent timestamps for the bus-rate count

The table is fixed-size on purpose. An attacker spraying random IDs (the Fuzzy
attack) must not be able to grow our memory, so unknown IDs are never inserted --
they get fixed default values instead. That "no entry for you" fact is itself a
useful signal, exposed as `is_known_id`.

The 16 features, in order:

   0  can_id_norm      CAN ID / 0x7FF
   1  is_known_id      1.0 if this ID appears in normal training traffic
   2  dlc_norm         DLC / 8
   3-10 byte0..byte7   payload bytes / 255, missing bytes = 0
  11  dt_prev          time since the previous frame on the bus (normalised)
  12  dt_same_id       time since the previous frame with this ID (normalised)
  13  period_ratio     dt_same_id / this ID's learned normal period (normalised)
  14  payload_change   bits changed vs the last payload of this ID, / 64
  15  bus_rate         frames seen in the last 10 ms (normalised)

Why these catch the attacks (see docs/can_basics.md section 6):

  DoS    -> bus_rate spikes, dt_prev collapses, and ID 0x000 has is_known_id = 0
  Fuzzy  -> is_known_id = 0 for most frames, payload bytes look uniformly random
  Gear   -> known ID arriving ~10x too often, so period_ratio drops far below 1,
  /RPM      and payload_change spikes as real and fake values interleave

Those last two are the whole reason this project exists: a gear-spoof frame is a
perfectly legal CAN frame, so nothing about the frame *by itself* is wrong. Only
its timing and its relationship to the previous value give it away.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "can_id_norm", "is_known_id", "dlc_norm",
    "byte0", "byte1", "byte2", "byte3", "byte4", "byte5", "byte6", "byte7",
    "dt_prev", "dt_same_id", "period_ratio", "payload_change", "bus_rate",
]
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 16

CAN_ID_MAX = 0x7FF          # 11-bit identifier
BUS_RATE_WINDOW = 0.010     # seconds; "frames in the last 10 ms"

# Why an epsilon is needed on the bus-rate window.
#
# The firmware keeps time as uint32 microseconds and evicts with
#     (uint32_t)(now_us - old_us) > 10000
# which is exact: a frame sitting exactly 10 ms back is kept.
#
# Here time is float64 seconds, and `ts - 0.010` is not exact. The parity test
# caught a real case:
#
#     ts = 0.024,  ts - 0.010 = 0.014,  stored old ts = 0.013999999999999999
#
# The stored value is a hair BELOW the computed cutoff, so a strict `<` threw
# away a frame the device kept -- bus_rate came out one frame short, a 1/30
# difference on a feature clipped at 30. One frame is a real difference, not
# rounding noise, so it had to be fixed rather than widened away by tolerance.
#
# 1 ns is 1000x finer than the 1 us resolution of the data, so widening the
# window by that much can never include or exclude a frame that integer
# microsecond arithmetic would have decided differently. It only absorbs the
# float64 representation error at the boundary.
_EVICT_EPS = 1e-9

# Defaults used when we have no history for an ID -- either it is the first frame
# with that ID, or the ID is unknown and deliberately not stored. Chosen to look
# "unremarkable" so the model leans on is_known_id instead of a magic number.
DEFAULT_DT_SAME_ID = 1.0    # seconds: a long time, i.e. no recent history
DEFAULT_PERIOD_RATIO = 1.0  # exactly nominal
DEFAULT_PAYLOAD_CHANGE = 0  # no observed change

# Bit-count lookup, 0..255. Mirrors the POPCOUNT8 table in the firmware.
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass
class Constants:
    """Everything learned from the training split. Exported to C as a header.

    Keeping these in one object makes the leakage rule easy to enforce and easy to
    audit: fit() only ever sees training frames, and transform() only ever reads
    this object.
    """
    known_ids: list          # sorted CAN IDs seen in normal training traffic
    id_periods: dict         # can_id (as str) -> median seconds between frames
    dt_prev_clip: float      # normalisation ceilings, from training percentiles
    dt_same_clip: float
    period_ratio_clip: float
    bus_rate_clip: float

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path):
        return Constants(**json.loads(Path(path).read_text()))

    def period_lookup(self):
        """dict with int keys, for fast access during transform()."""
        return {int(k): v for k, v in self.id_periods.items()}


def fit(ts, can_id, label, normal_class=0):
    """Learn the constants from TRAINING frames only.

    Critically this uses only rows labelled normal. If we learned each ID's period
    from attack frames too, the spoofed 1 ms interval would drag the median down
    and period_ratio would stop being able to flag it -- we would have taught the
    detector that the attack is normal.
    """
    mask = label == normal_class
    ts_n, id_n = ts[mask], can_id[mask]

    known = np.unique(id_n)
    periods = {}
    same_id_gaps = []
    for cid in known:
        t = ts_n[id_n == cid]
        if len(t) < 3:
            continue
        d = np.diff(t)
        d = d[d > 0]
        if len(d):
            periods[int(cid)] = float(np.median(d))
            same_id_gaps.append(d)

    # Only keep IDs we actually have a period for; an ID with no usable period
    # would make period_ratio undefined on device.
    known_ids = sorted(periods.keys())

    # Normalisation ceilings from the training distribution. p99 rather than max,
    # so one outlier frame doesn't squash every other value towards zero.
    dt_prev = np.diff(ts)
    dt_prev = dt_prev[dt_prev > 0]
    bus_rate = _bus_rate(ts)

    # dt_same_id must be scaled by the spread of PER-ID gaps, not bus-wide gaps.
    # Bus-wide gaps are ~0.4 ms (38 IDs interleaved) while a single ID's gap is
    # ~10 ms, so scaling by the bus-wide figure would clip almost every normal
    # frame to 1.0 and throw away the resolution this feature exists to provide.
    # Pooling the per-ID gaps weights by frame count automatically, so the busy
    # 10 ms IDs dominate and a once-a-minute diagnostic ID doesn't skew the scale.
    gaps = np.concatenate(same_id_gaps) if same_id_gaps else np.array([0.1])

    return Constants(
        known_ids=known_ids,
        id_periods={str(k): v for k, v in periods.items()},
        dt_prev_clip=float(np.percentile(dt_prev, 99)) if len(dt_prev) else 0.01,
        dt_same_clip=float(np.percentile(gaps, 99)),
        period_ratio_clip=4.0,   # ratio of 1.0 is nominal; 4x nominal is plenty
        bus_rate_clip=float(np.percentile(bus_rate, 99)) if len(bus_rate) else 100.0,
    )


def merge(consts):
    """Combine per-file Constants into one set for the final model.

    Each capture is a separate recording session with its own absolute timeline, so
    we cannot just concatenate the files and call fit() -- np.diff would stride
    across a session boundary and invent a bogus multi-second interval.

    Instead each file is fitted on its own, and here we take the median across
    files for every quantity. An ID's period is a property of the car, so the
    per-file estimates should agree closely; if they don't, that is worth knowing
    and the median is the robust answer either way.
    """
    all_ids = sorted({cid for c in consts for cid in c.known_ids})

    periods = {}
    for cid in all_ids:
        vals = [c.id_periods[str(cid)] for c in consts if str(cid) in c.id_periods]
        if vals:
            periods[str(cid)] = float(np.median(vals))

    med = lambda attr: float(np.median([getattr(c, attr) for c in consts]))  # noqa: E731

    return Constants(
        known_ids=[cid for cid in all_ids if str(cid) in periods],
        id_periods=periods,
        dt_prev_clip=med("dt_prev_clip"),
        dt_same_clip=med("dt_same_clip"),
        period_ratio_clip=consts[0].period_ratio_clip,   # a fixed choice, not fitted
        bus_rate_clip=med("bus_rate_clip"),
    )


def _bus_rate(ts):
    """Frames seen in the preceding BUS_RATE_WINDOW seconds, per frame.

    On device this is a ring buffer: push the new timestamp, pop anything older
    than 10 ms, report the count. Here the same answer comes from a binary search,
    which is identical as long as timestamps are non-decreasing.

    The cutoff is nudged down by _EVICT_EPS so a frame exactly on the 10 ms
    boundary is kept, matching the firmware's exact integer-microsecond test.
    See the _EVICT_EPS comment above.
    """
    start = np.searchsorted(ts, ts - BUS_RATE_WINDOW - _EVICT_EPS, side="left")
    return (np.arange(len(ts)) - start).astype(np.float32)


def _per_id_history(ts, can_id, data, known_set):
    """dt_same_id and payload_change, computed per ID.

    Vectorised here for speed over millions of frames, but the semantics are
    exactly what a streaming implementation produces: for each frame, look at the
    previous frame carrying the same ID. Unknown IDs are skipped entirely, so they
    keep the defaults -- same as the device refusing to allocate a table slot.
    """
    n = len(ts)
    dt_same = np.full(n, DEFAULT_DT_SAME_ID, dtype=np.float64)
    pay_change = np.full(n, DEFAULT_PAYLOAD_CHANGE, dtype=np.float32)

    for cid in known_set:
        idx = np.flatnonzero(can_id == cid)
        if len(idx) < 2:
            continue

        # time since the previous frame with this ID (first occurrence keeps default)
        dt_same[idx[1:]] = np.diff(ts[idx])

        # bits changed vs the previous payload of this ID: XOR then popcount
        prev = data[idx[:-1]]
        cur = data[idx[1:]]
        pay_change[idx[1:]] = _POPCOUNT8[np.bitwise_xor(prev, cur)].sum(axis=1)

    return dt_same, pay_change


def transform(ts, can_id, dlc, data, const):
    """Build the (n, 16) feature matrix. `const` must come from training data."""
    n = len(ts)
    known_set = set(const.known_ids)
    periods = const.period_lookup()

    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # --- per-frame fields, no history needed ---------------------------------
    X[:, 0] = can_id / CAN_ID_MAX
    is_known = np.isin(can_id, const.known_ids)
    X[:, 1] = is_known.astype(np.float32)
    X[:, 2] = np.clip(dlc, 0, 8) / 8.0
    X[:, 3:11] = data / 255.0

    # --- time since the previous frame on the bus ----------------------------
    dt_prev = np.empty(n, dtype=np.float64)
    dt_prev[0] = const.dt_prev_clip          # no predecessor for the first frame
    dt_prev[1:] = np.diff(ts)
    X[:, 11] = np.clip(dt_prev, 0, const.dt_prev_clip) / const.dt_prev_clip

    # --- per-ID history ------------------------------------------------------
    dt_same, pay_change = _per_id_history(ts, can_id, data, known_set)
    X[:, 12] = np.clip(dt_same, 0, const.dt_same_clip) / const.dt_same_clip

    # period_ratio: how does this gap compare with how often this ID *should*
    # arrive? A spoofed ID arrives ~10x too often, pushing the ratio towards 0.
    nominal = np.array([periods.get(int(c), 0.0) for c in can_id], dtype=np.float64)
    ratio = np.full(n, DEFAULT_PERIOD_RATIO, dtype=np.float64)
    ok = nominal > 0
    ratio[ok] = dt_same[ok] / nominal[ok]
    X[:, 13] = np.clip(ratio, 0, const.period_ratio_clip) / const.period_ratio_clip

    X[:, 14] = pay_change / 64.0             # 8 bytes = 64 bits maximum

    # --- bus load ------------------------------------------------------------
    X[:, 15] = np.clip(_bus_rate(ts), 0, const.bus_rate_clip) / const.bus_rate_clip

    return X


class StreamingExtractor:
    """One-frame-at-a-time reference implementation.

    transform() above is vectorised so it can chew through 17 million frames in
    reasonable time, but the firmware cannot look ahead. This class is the honest
    streaming version: it is what the C++ mirrors, it is what demo.py uses on the
    live bus, and tests/test_features.py asserts it agrees with transform().

    If these two ever disagree, transform() has accidentally used the future.
    """

    def __init__(self, const):
        self.const = const
        self.periods = const.period_lookup()
        self.known = set(const.known_ids)
        self.last_ts_any = None
        self.last_ts_id = {}        # can_id -> timestamp of its previous frame
        self.last_payload = {}      # can_id -> its previous 8 bytes
        self.recent = []            # timestamps within the bus-rate window

    def push(self, ts, can_id, dlc, data):
        """Feed one frame, get back its 16 features."""
        c = self.const
        f = np.zeros(N_FEATURES, dtype=np.float32)
        data = np.asarray(data, dtype=np.uint8)

        f[0] = can_id / CAN_ID_MAX
        known = can_id in self.known
        f[1] = 1.0 if known else 0.0
        f[2] = min(max(dlc, 0), 8) / 8.0
        f[3:11] = data / 255.0

        dt_prev = c.dt_prev_clip if self.last_ts_any is None else ts - self.last_ts_any
        f[11] = min(max(dt_prev, 0.0), c.dt_prev_clip) / c.dt_prev_clip

        # Unknown IDs never get a table slot -- fixed memory, and an attacker
        # spraying random IDs cannot exhaust it.
        if known and can_id in self.last_ts_id:
            dt_same = ts - self.last_ts_id[can_id]
            change = int(_POPCOUNT8[np.bitwise_xor(self.last_payload[can_id],
                                                   data)].sum())
        else:
            dt_same = DEFAULT_DT_SAME_ID
            change = DEFAULT_PAYLOAD_CHANGE

        f[12] = min(max(dt_same, 0.0), c.dt_same_clip) / c.dt_same_clip

        nominal = self.periods.get(can_id, 0.0)
        ratio = (dt_same / nominal) if nominal > 0 else DEFAULT_PERIOD_RATIO
        f[13] = min(max(ratio, 0.0), c.period_ratio_clip) / c.period_ratio_clip

        f[14] = change / 64.0

        # ring buffer: drop anything older than the window, then count.
        # _EVICT_EPS keeps a frame that sits exactly on the boundary, which is
        # what the firmware's integer-microsecond compare does.
        self.recent.append(ts)
        cutoff = ts - BUS_RATE_WINDOW - _EVICT_EPS
        while self.recent and self.recent[0] < cutoff:
            self.recent.pop(0)
        f[15] = min(len(self.recent) - 1, c.bus_rate_clip) / c.bus_rate_clip

        # update state AFTER computing features, so a frame never sees itself
        self.last_ts_any = ts
        if known:
            self.last_ts_id[can_id] = ts
            self.last_payload[can_id] = data.copy()

        return f
