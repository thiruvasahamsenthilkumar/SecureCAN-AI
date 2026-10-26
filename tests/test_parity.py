"""Python <-> C++ feature parity.

The claim this project makes is that the ESP32 computes the SAME 16 features the
model was trained on. If it does not, nothing else in the repo means anything:
the firmware would keep producing confident answers, just from a model trained on
different numbers than it is being fed. That failure is completely silent, which
is exactly why it needs a test.

So this compiles the real firmware/esp32/src/features.h for the host, pipes real
HCRL frames through it, and compares against StreamingExtractor in
src/securecan/features.py frame by frame.

Three implementations have to agree, and each guards a different mistake:

    transform()           vectorised, fast, used for training
      | tests/test_features.py::test_streaming_matches_vectorised
      |   guards against lookahead -- a vectorised shift that peeks at the future
    StreamingExtractor    one frame at a time, the reference semantics
      | THIS FILE
      |   guards against the port drifting from the reference
    features.h            C++, fixed memory, uint32 microsecond clock

Skipped automatically if no C++ compiler is available, so `pytest` still works on
a machine that only has Python. CI should make sure the compiler is there.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from securecan import features as F  # noqa: E402

FW = ROOT / "firmware" / "esp32" / "src"
HOST_SRC = ROOT / "tools" / "parity_host.cpp"

# Where g++ ends up on Windows when installed via winget/MSYS2. Checked in order
# after the plain PATH lookup.
EXTRA_COMPILER_PATHS = [
    r"C:\mingw64\bin\g++.exe",
    r"C:\msys64\mingw64\bin\g++.exe",
    r"C:\ProgramData\chocolatey\bin\g++.exe",
]


def find_compiler():
    for name in ("g++", "clang++"):
        p = shutil.which(name)
        if p:
            return p
    for p in EXTRA_COMPILER_PATHS:
        if Path(p).exists():
            return p
    return None


COMPILER = find_compiler()

needs_cxx = pytest.mark.skipif(
    COMPILER is None, reason="no C++ compiler found; parity not checked")
needs_headers = pytest.mark.skipif(
    not (FW / "constants_data.h").exists(),
    reason="constants_data.h not generated -- run python -m securecan.quantize")


# ---------------------------------------------------------------------------
# build + run the C++ side
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def host_binary(tmp_path_factory):
    """Compile tools/parity_host.cpp against the real firmware header."""
    out_dir = tmp_path_factory.mktemp("parity")
    exe = out_dir / ("parity_host.exe" if os.name == "nt" else "parity_host")

    cmd = [COMPILER, "-O2", "-std=c++17", "-Wall",
           "-I", str(FW), "-o", str(exe), str(HOST_SRC)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail("features.h does not compile:\n"
                    f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return exe


def run_cpp(exe, t_us, can_id, dlc, data):
    """Feed frames to the C++ binary, return its (n, 16) feature matrix."""
    lines = []
    for i in range(len(t_us)):
        row = " ".join(str(int(b)) for b in data[i])
        lines.append(f"{int(t_us[i])} {int(can_id[i])} {int(dlc[i])} {row}")
    stdin = "\n".join(lines) + "\n"

    r = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, f"parity_host crashed: {r.stderr}"

    out = [l for l in r.stdout.strip().splitlines() if l.strip()]
    assert len(out) == len(t_us), \
        f"C++ returned {len(out)} rows for {len(t_us)} frames"

    return np.array([[float(v) for v in l.split()] for l in out],
                    dtype=np.float64)


def run_python(const, t_us, can_id, dlc, data):
    """Same frames through StreamingExtractor.

    Timestamps are fed as seconds derived from the SAME integer microseconds the
    C++ gets, so the comparison tests the feature logic rather than a difference
    in how the two sides happen to round time.
    """
    ex = F.StreamingExtractor(const)
    rows = []
    for i in range(len(t_us)):
        rows.append(ex.push(float(t_us[i]) * 1e-6, int(can_id[i]),
                            int(dlc[i]), data[i]))
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# frames to test with
# ---------------------------------------------------------------------------

def real_frames(n=4000):
    """A slice of a real capture that includes an attack burst.

    Synthetic frames would miss the cases that actually differ: unknown IDs, DLC
    under 8, a bus rate high enough to exercise the ring buffer, and the first
    frame of each ID where the defaults kick in.
    """
    p = ROOT / "data" / "processed" / "RPM_dataset.npz"
    if not p.exists():
        pytest.skip(f"{p.name} not present -- run tools/fetch_dataset.py and parse.py")

    d = np.load(p)
    label = d["label"]

    # start a little before an attack burst so both regimes are covered
    atk = np.flatnonzero(label != 0)
    start = max(0, int(atk[0]) - n // 4) if len(atk) else 0
    sl = slice(start, start + n)

    ts = d["ts"][sl]
    t_us = np.round((ts - ts[0]) * 1e6).astype(np.uint32)
    return t_us, d["can_id"][sl], d["dlc"][sl], d["data"][sl], label[sl]


@pytest.fixture(scope="session")
def const():
    p = ROOT / "results" / "constants.json"
    if not p.exists():
        pytest.skip("results/constants.json missing -- run securecan.train")
    return F.Constants.load(p)


# ---------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------

@needs_cxx
@needs_headers
def test_cpp_matches_python_on_real_frames(host_binary, const):
    """The headline test: 4,000 real frames, all 16 features, both regimes.

    Tolerance is 1e-5 absolute. It is not exactly zero because Python computes in
    float64 and the device in float32, which is a deliberate choice -- float64 on
    an ESP32 is software-emulated and far too slow for a 250 us budget. 1e-5 on a
    feature scaled to [0, 1] is ~100x smaller than one INT8 quantisation step
    (1/255), so a difference this size cannot change the model's answer.
    """
    t_us, can_id, dlc, data, label = real_frames()

    got = run_cpp(host_binary, t_us, can_id, dlc, data)
    want = run_python(const, t_us, can_id, dlc, data)

    assert got.shape == want.shape == (len(t_us), F.N_FEATURES)

    bad = np.abs(got - want) > 1e-5
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise AssertionError(
            f"{int(bad.sum())} of {bad.size} values differ.\n"
            f"first at frame {i} (id 0x{int(can_id[i]):03X}, "
            f"label {int(label[i])}), feature {j} "
            f"({F.FEATURE_NAMES[j]}):\n"
            f"  C++    {got[i, j]!r}\n"
            f"  Python {want[i, j]!r}\n"
            f"  diff   {abs(got[i, j] - want[i, j]):.3e}"
        )


@needs_cxx
@needs_headers
def test_parity_holds_per_feature(host_binary, const):
    """Report the worst disagreement per feature, so a failure localises itself.

    Without this, a parity break just says "something differs" and you go
    hunting. With it, the failure names the feature.
    """
    t_us, can_id, dlc, data, _ = real_frames(2000)

    got = run_cpp(host_binary, t_us, can_id, dlc, data)
    want = run_python(const, t_us, can_id, dlc, data)

    worst = np.abs(got - want).max(axis=0)
    report = "\n".join(f"  {F.FEATURE_NAMES[j]:<16} {worst[j]:.3e}"
                       for j in range(F.N_FEATURES))
    assert worst.max() <= 1e-5, f"per-feature max abs diff:\n{report}"


@needs_cxx
@needs_headers
def test_unknown_ids_get_no_table_slot(host_binary, const):
    """An ID absent from the known list must score is_known_id = 0 in C++ too.

    This is the Fuzzy signal AND the fixed-memory guarantee: the firmware refuses
    to allocate for an ID it has not learned, so spraying random IDs cannot
    exhaust its memory.
    """
    known = sorted(const.known_ids)
    unknown = next(i for i in range(0x800) if i not in set(known))

    t_us = np.array([0, 10_000, 20_000], dtype=np.uint32)
    can_id = np.array([known[0], unknown, known[0]], dtype=np.int32)
    dlc = np.array([8, 8, 8], dtype=np.int8)
    data = np.zeros((3, 8), dtype=np.uint8)

    got = run_cpp(host_binary, t_us, can_id, dlc, data)
    k = F.FEATURE_NAMES.index("is_known_id")

    assert got[0, k] == 1.0
    assert got[1, k] == 0.0, "unknown ID was treated as known"
    assert got[2, k] == 1.0


@needs_cxx
@needs_headers
def test_short_dlc_frames_match(host_binary, const):
    """DLC < 8 is where the HCRL parser had its trap; check the C++ agrees.

    Bytes past the DLC are zero-filled on both sides, so the feature vectors must
    still line up exactly.
    """
    known = sorted(const.known_ids)[:1] * 4
    t_us = np.array([0, 10_000, 20_000, 30_000], dtype=np.uint32)
    can_id = np.array(known, dtype=np.int32)
    dlc = np.array([2, 5, 8, 3], dtype=np.int8)
    data = np.array([
        [0xDE, 0xAD, 0, 0, 0, 0, 0, 0],
        [0x01, 0x02, 0x03, 0x04, 0x05, 0, 0, 0],
        [0xFF] * 8,
        [0x11, 0x22, 0x33, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    got = run_cpp(host_binary, t_us, can_id, dlc, data)
    want = run_python(const, t_us, can_id, dlc, data)
    np.testing.assert_allclose(got, want, atol=1e-5)


@needs_cxx
@needs_headers
def test_ring_buffer_evicts_like_python(host_binary, const):
    """bus_rate: frames older than 10 ms must leave the window on both sides.

    The C++ uses a fixed 128-slot ring and unsigned microsecond arithmetic;
    Python uses an unbounded list of float seconds. Different mechanisms, so this
    is worth pinning: 30 frames 1 ms apart (window fills), then a 1 s gap (window
    must empty completely).
    """
    cid = sorted(const.known_ids)[0]
    t_us = np.concatenate([np.arange(30) * 1000, [1_030_000]]).astype(np.uint32)
    can_id = np.full(len(t_us), cid, dtype=np.int32)
    dlc = np.full(len(t_us), 8, dtype=np.int8)
    data = np.zeros((len(t_us), 8), dtype=np.uint8)

    got = run_cpp(host_binary, t_us, can_id, dlc, data)
    want = run_python(const, t_us, can_id, dlc, data)

    b = F.FEATURE_NAMES.index("bus_rate")
    np.testing.assert_allclose(got[:, b], want[:, b], atol=1e-5)
    assert got[0, b] == 0.0, "nothing precedes the first frame"
    assert got[-1, b] == 0.0, "the 1 s gap should have emptied the window"


BANNED_IN_FIRMWARE = ("malloc", "calloc", "realloc", "new ", "std::vector",
                      "std::map", "std::string")


def strip_comments(text):
    """Remove // and /* */ comments.

    Without this the check greps its own documentation: the comment in nn.h
    explaining that there is no malloc in the firmware contains the word
    "malloc", and a raw substring search fails on it. A test that cannot
    distinguish code from prose about the code is not checking what it claims.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


@needs_cxx
@pytest.mark.parametrize("header", ["features.h", "nn.h"])
def test_firmware_headers_have_no_dynamic_allocation(header):
    """No malloc/new on the device path -- fixed memory is a design guarantee.

    A detector that can fail to allocate while driving is worse than no
    detector. Both the feature extractor and the classifier hold only
    compile-time sized buffers, so the worst-case RAM use comes off the linker
    map instead of being hoped for at runtime.
    """
    code = strip_comments((FW / header).read_text())
    for bad in BANNED_IN_FIRMWARE:
        assert bad not in code, f"{header} uses {bad!r}; must be fixed-memory"


def test_strip_comments_actually_strips():
    """The helper above is load-bearing, so check it on the real case."""
    assert "malloc" not in strip_comments("int x; // no malloc here\nint y;")
    assert "malloc" not in strip_comments("/* malloc */ int z;")
    assert "malloc" in strip_comments("void* p = malloc(4); // allocate")


# ---------------------------------------------------------------------------
# inference parity: our hand-written int8 kernel vs TFLite's own
#
# The firmware does not link TensorFlow Lite Micro -- its only ESP32 Arduino
# packaging is a 2021 snapshot that does not compile against ESP-IDF 5, so
# nn.h implements the int8 forward pass directly (see tools/export_model.py).
#
# That is only a defensible choice if the hand-written kernel demonstrably
# matches the reference. These tests are that proof. They feed the same real
# frames to both and require the predicted CLASS to be identical on every
# frame -- not close on average, identical -- because a classifier that
# disagrees on one frame in a thousand disagrees 4 times a second on a 500
# kbit/s bus.
# ---------------------------------------------------------------------------

TFLITE_INT8 = ROOT / "results" / "mlp_int8.tflite"

needs_tflite = pytest.mark.skipif(
    not TFLITE_INT8.exists(),
    reason="results/mlp_int8.tflite missing -- run python -m securecan.quantize")


def run_cpp_nn(exe, t_us, can_id, dlc, data):
    """Feed frames to the C++ binary in --nn mode.

    Returns (classes, confidences). This is the full device chain: features.h
    computes the features and nn.h classifies them, exactly as on the ESP32.
    """
    lines = []
    for i in range(len(t_us)):
        row = " ".join(str(int(b)) for b in data[i])
        lines.append(f"{int(t_us[i])} {int(can_id[i])} {int(dlc[i])} {row}")

    r = subprocess.run([str(exe), "--nn"], input="\n".join(lines) + "\n",
                       capture_output=True, text=True)
    assert r.returncode == 0, f"parity_host --nn crashed: {r.stderr}"

    out = [l for l in r.stdout.strip().splitlines() if l.strip()]
    assert len(out) == len(t_us), \
        f"C++ returned {len(out)} rows for {len(t_us)} frames"

    cls = np.array([int(l.split()[0]) for l in out], dtype=np.int32)
    conf = np.array([float(l.split()[1]) for l in out], dtype=np.float64)
    return cls, conf


def run_tflite_int8(feats):
    """The reference: TFLite's own int8 interpreter on the given features.

    Quantise the input the same way the device does, invoke, and take the argmax.
    Driving the interpreter with the quantised int8 input rather than letting it
    convert from float keeps the comparison about the arithmetic instead of
    about who rounds the input.
    """
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(TFLITE_INT8))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zero = inp["quantization"]

    preds = np.empty(len(feats), dtype=np.int32)
    for i, row in enumerate(feats):
        q = np.clip(np.rint(row / scale) + zero, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, -1))
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(out["index"])[0]))
    return preds


@needs_cxx
@needs_headers
@needs_tflite
def test_nn_matches_tflite_on_real_frames(host_binary, const):
    """Our int8 kernel and TFLite's agree on the class for every real frame."""
    t_us, can_id, dlc, data, _label = real_frames(2000)

    got, conf = run_cpp_nn(host_binary, t_us, can_id, dlc, data)
    feats = run_python(const, t_us, can_id, dlc, data)
    want = run_tflite_int8(feats)

    disagree = np.flatnonzero(got != want)
    if len(disagree):
        i = int(disagree[0])
        raise AssertionError(
            f"{len(disagree)} of {len(got)} frames disagree with TFLite.\n"
            f"first at frame {i}: nn.h says {got[i]}, TFLite says {want[i]}\n"
            f"features: {np.array2string(feats[i], precision=6)}")

    # Confidence is a float softmax over dequantised logits, so it is not
    # expected to be bit-identical to anything -- but it must be a probability.
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)
    assert np.all(conf >= 1.0 / 5 - 1e-6), \
        "the winning class cannot have less than uniform probability"


@needs_cxx
@needs_headers
@needs_tflite
def test_nn_detects_the_attack_frames(host_binary, const):
    """Sanity check that parity is not two implementations agreeing on nothing.

    A kernel that returned 0 for everything would pass the test above if TFLite
    somehow did too. This pins the actual behaviour: the injected frames in this
    slice are flagged, and the model names the right attack.
    """
    t_us, can_id, dlc, data, label = real_frames(2000)
    got, _ = run_cpp_nn(host_binary, t_us, can_id, dlc, data)

    atk = label != 0
    assert atk.sum() > 50, "test slice should contain a decent number of attacks"

    recall = (got[atk] != 0).mean()
    assert recall > 0.95, f"only {recall:.1%} of injected frames flagged"

    # and it should be the right class, not just "something is wrong"
    expected = int(np.bincount(label[atk]).argmax())
    assert (got[atk] == expected).mean() > 0.95, \
        f"flagged, but not as class {expected}"
