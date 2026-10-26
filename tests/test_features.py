"""Tests for feature extraction.

The most important test here is test_streaming_matches_vectorised. transform() is
vectorised so it can process 17 million frames in reasonable time, but a vectorised
implementation can accidentally look into the future -- one off-by-one in a shift
and a feature starts encoding the answer. StreamingExtractor physically cannot do
that, so making the two agree exactly is what proves transform() is honest.

It is also the contract the C++ firmware implements, so this test is the Python
half of the Python/C++ parity story.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from securecan.features import (  # noqa: E402
    BUS_RATE_WINDOW, DEFAULT_PAYLOAD_CHANGE, FEATURE_NAMES, N_FEATURES,
    Constants, StreamingExtractor, fit, transform,
)

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def make_frames(n=400, seed=0):
    """Synthetic but realistic traffic: three IDs each on their own period."""
    rng = np.random.default_rng(seed)
    specs = [(0x316, 0.010), (0x153, 0.020), (0x2B0, 0.050)]

    rows = []
    for cid, period in specs:
        t = 0.0
        payload = rng.integers(0, 256, 8)
        for _ in range(n // len(specs)):
            t += period * rng.uniform(0.95, 1.05)   # a little jitter, like a real ECU
            payload = (payload + rng.integers(0, 3, 8)) % 256
            rows.append((t, cid, 8, payload.copy()))

    rows.sort(key=lambda r: r[0])
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    can_id = np.array([r[1] for r in rows], dtype=np.int32)
    dlc = np.array([r[2] for r in rows], dtype=np.int8)
    data = np.array([r[3] for r in rows], dtype=np.uint8)
    label = np.zeros(len(rows), dtype=np.int8)
    return ts, can_id, dlc, data, label


@pytest.fixture
def frames():
    return make_frames()


@pytest.fixture
def const(frames):
    ts, can_id, _, _, label = frames
    return fit(ts, can_id, label)


# --------------------------------------------------------------------------
# the contract test
# --------------------------------------------------------------------------

def test_streaming_matches_vectorised(frames, const):
    """One frame at a time must give the same answer as the whole-array path.

    If this fails, transform() is using information the device could not have.
    """
    ts, can_id, dlc, data, _ = frames
    X = transform(ts, can_id, dlc, data, const)

    ex = StreamingExtractor(const)
    for i in range(len(ts)):
        got = ex.push(ts[i], int(can_id[i]), int(dlc[i]), data[i])
        np.testing.assert_allclose(
            got, X[i], rtol=1e-5, atol=1e-6,
            err_msg=f"frame {i} (id 0x{can_id[i]:03X}) diverged",
        )


# --------------------------------------------------------------------------
# no-leakage guarantees
# --------------------------------------------------------------------------

def test_periods_learned_from_normal_frames_only():
    """Attack frames must not drag an ID's learned period down.

    This is the subtle one. RPM spoofing floods a legitimate ID at ~1 ms. If fit()
    averaged that in, the 'normal' period would shrink towards 1 ms and
    period_ratio would stop flagging the attack -- we would have trained the
    detector to accept it.
    """
    # ID 0x316 genuinely every 10 ms, plus injected frames every 1 ms
    normal_t = np.arange(0, 1.0, 0.010)
    attack_t = np.arange(0, 1.0, 0.001)
    ts = np.concatenate([normal_t, attack_t])
    can_id = np.full(len(ts), 0x316, dtype=np.int32)
    label = np.concatenate([
        np.zeros(len(normal_t), dtype=np.int8),
        np.full(len(attack_t), 4, dtype=np.int8),      # RPM spoof
    ])
    order = np.argsort(ts, kind="stable")

    const = fit(ts[order], can_id[order], label[order])
    assert const.id_periods["790"] == pytest.approx(0.010, abs=1e-4)


def test_unknown_ids_are_not_learned(frames):
    ts, can_id, _, _, label = frames
    const = fit(ts, can_id, label)
    assert 0x000 not in const.known_ids          # the DoS flood ID was never seen
    assert set(const.known_ids) == {0x153, 0x2B0, 0x316}


# --------------------------------------------------------------------------
# individual features
# --------------------------------------------------------------------------

def test_is_known_id_flags_unseen_ids(frames, const):
    """Fuzzy's giveaway: IDs that never appear in normal traffic."""
    ts, can_id, dlc, data, _ = frames
    # splice in a frame with an ID the model has never seen
    can_id = can_id.copy()
    can_id[10] = 0x7AA

    X = transform(ts, can_id, dlc, data, const)
    assert X[10, F["is_known_id"]] == 0.0
    assert X[11, F["is_known_id"]] == 1.0


def test_unknown_id_gets_defaults_not_history(frames, const):
    """An unknown ID must never get a table slot -- fixed memory on device.

    Two frames with the same unknown ID still both get defaults, because nothing
    was stored between them.
    """
    ts, can_id, dlc, data, _ = frames
    can_id = can_id.copy()
    can_id[10] = can_id[11] = 0x7AA

    X = transform(ts, can_id, dlc, data, const)
    assert X[10, F["payload_change"]] == DEFAULT_PAYLOAD_CHANGE
    assert X[11, F["payload_change"]] == DEFAULT_PAYLOAD_CHANGE
    assert X[10, F["dt_same_id"]] == X[11, F["dt_same_id"]]


def test_payload_change_counts_bits_not_bytes(const):
    """payload_change is a popcount of the XOR, normalised by 64 bits."""
    ts = np.array([0.0, 0.010], dtype=np.float64)
    can_id = np.array([0x316, 0x316], dtype=np.int32)
    dlc = np.array([8, 8], dtype=np.int8)
    # 0x00 -> 0xFF in byte 0 is 8 bits; 0x00 -> 0x01 in byte 1 is 1 bit. 9 total.
    data = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [0xFF, 0x01, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    X = transform(ts, can_id, dlc, data, const)
    assert X[1, F["payload_change"]] == pytest.approx(9 / 64)


def test_period_ratio_drops_when_id_arrives_too_fast(const):
    """The gear/RPM spoof signature, in miniature.

    0x316's learned period is 10 ms. A frame arriving 1 ms after the last one has
    a ratio of 0.1 -- ten times too soon.
    """
    ts = np.array([0.0, 0.010, 0.011], dtype=np.float64)
    can_id = np.full(3, 0x316, dtype=np.int32)
    dlc = np.full(3, 8, dtype=np.int8)
    data = np.zeros((3, 8), dtype=np.uint8)

    X = transform(ts, can_id, dlc, data, const)
    nominal = const.id_periods[str(0x316)]
    scale = const.period_ratio_clip

    assert X[1, F["period_ratio"]] == pytest.approx((0.010 / nominal) / scale, rel=0.1)
    assert X[2, F["period_ratio"]] < X[1, F["period_ratio"]]


def test_bus_rate_counts_only_the_window(const):
    """DoS's giveaway. Frames older than 10 ms must fall out of the count."""
    # The fixture's own bus_rate_clip is tiny (its synthetic traffic is slow), and
    # clipping would hide the behaviour under test. Use a headroom-y ceiling so we
    # read the raw count back out.
    const = replace(const, bus_rate_clip=100.0)

    # 20 frames 1 ms apart, then a 1-second gap, then one more
    ts = np.concatenate([np.arange(20) * 0.001, [1.5]])
    can_id = np.full(len(ts), 0x316, dtype=np.int32)
    dlc = np.full(len(ts), 8, dtype=np.int8)
    data = np.zeros((len(ts), 8), dtype=np.uint8)

    X = transform(ts, can_id, dlc, data, const)
    raw = X[:, F["bus_rate"]] * const.bus_rate_clip

    assert raw[0] == 0                       # nothing precedes the first frame
    assert raw[15] == pytest.approx(10, abs=1)   # ~10 frames in the last 10 ms
    assert raw[-1] == 0                      # the gap emptied the ring buffer


def test_dt_prev_shrinks_under_flood(const):
    """A DoS flood collapses the inter-frame gap towards zero."""
    normal = np.arange(0, 0.1, 0.010)
    flood = np.arange(0.1, 0.11, 0.0003)     # ID 0x000 every 0.3 ms
    ts = np.concatenate([normal, flood])
    can_id = np.concatenate([
        np.full(len(normal), 0x316), np.full(len(flood), 0x000)
    ]).astype(np.int32)
    dlc = np.full(len(ts), 8, dtype=np.int8)
    data = np.zeros((len(ts), 8), dtype=np.uint8)

    X = transform(ts, can_id, dlc, data, const)
    assert X[len(normal) + 5, F["dt_prev"]] < X[5, F["dt_prev"]]


# --------------------------------------------------------------------------
# shape and range sanity
# --------------------------------------------------------------------------

def test_shape_and_names_agree(frames, const):
    ts, can_id, dlc, data, _ = frames
    X = transform(ts, can_id, dlc, data, const)
    assert X.shape == (len(ts), N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES == 16


def test_all_features_in_unit_range(frames, const):
    """Everything is normalised to [0, 1] so INT8 quantisation keeps resolution."""
    ts, can_id, dlc, data, _ = frames
    X = transform(ts, can_id, dlc, data, const)
    assert X.min() >= 0.0
    assert X.max() <= 1.0 + 1e-6
    assert X.dtype == np.float32


def test_constants_round_trip(const, tmp_path):
    """Constants go to the firmware as a C header; JSON is the intermediate."""
    p = tmp_path / "constants.json"
    const.save(p)
    back = Constants.load(p)
    assert back.known_ids == const.known_ids
    assert back.id_periods == const.id_periods
    assert back.dt_prev_clip == const.dt_prev_clip
