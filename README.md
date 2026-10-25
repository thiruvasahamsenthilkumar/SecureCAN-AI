# SecureCAN AI

Real-time intrusion detection for a vehicle's CAN bus, running as an INT8-quantized
random forest on an ESP32.

A car's ECUs (engine, transmission, ABS, ...) talk over a CAN bus that has **no
authentication and no encryption** — any device physically on the bus can transmit any
message with any ID. The 2015 Jeep Cherokee remote hack exploited exactly this and led
Fiat Chrysler to recall ~1.4 million vehicles. SecureCAN AI watches the bus and flags
injected frames on the vehicle's own microcontroller, with no cloud round-trip.

---

## Results

Random forest (40 trees, depth 8) over 14 sliding-window features, 50 messages per
window, 25% held-out test split.

| Metric | Value |
|---|---|
| Test accuracy (held out) | **99.44%** |
| Macro F1 | 0.983 |
| float32 → INT8 agreement | **100.00%** |
| Accuracy lost to quantization | **0.00 pp** |
| Forest size | 40 trees / 1024 nodes |
| Model arrays on device | **7.0 KB** (int8 + int16) |
| Feature vector | 14 floats |

### Per-class performance (held-out test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.990 | 0.995 | 0.993 | 200 |
| DoS | 1.000 | 1.000 | 1.000 | 277 |
| Fuzzy | 1.000 | 0.957 | 0.978 | 23 |
| Fake RPM | 0.933 | 1.000 | 0.966 | 14 |
| Fake Gear | 1.000 | 0.962 | 0.980 | 26 |

DoS is trivially separable — flooding the bus every 0.3 ms shows up directly in the
message-rate feature. **Fake RPM is the hard class**: it uses a legitimate ECU ID
(`0x316`) with a plausible-looking payload, so the only signal is that the RPM bytes sit
outside the band that ID normally transmits. It carries the lowest precision (0.933),
i.e. it accounts for most of the false positives.

Reproduce:

```bash
python scripts/pipeline.py          # generate data, train, export C++ header
python scripts/verify_quantized.py  # check INT8 forest == float32 forest
```

---

## How it works

```
CAN frames ──▶ 50-message sliding window ──▶ 14 features ──▶ INT8 forest ──▶ class
                                                                              │
                           NORMAL / DoS / Fuzzy / Fake RPM / Fake Gear ◀──────┘
```

### Features (14)

Single frames are not enough — an injected frame is often individually valid and only
looks wrong relative to its neighbours. Detection therefore runs per window:

- **Timing** — `msg_rate`, `mean_iat`, `std_iat`, `min_iat`
  Inter-arrival time. This is what catches DoS.
- **Identity** — `unique_ids`, `id_entropy`, `dominant_id_ratio`
  A flood collapses ID entropy; fuzzing inflates it.
- **Payload** — `mean_dlc`, `mean_byte`, `std_byte`, `max_byte`, `byte_entropy`,
  `saturated_ratio`, `zero_ratio`
  Byte-level distribution. This is what catches spoofed signal values.

The same 14 features are computed twice — once in `scripts/pipeline.py` for training and
once in `src/detector.cpp` for the device. **Their order must stay identical**, otherwise
the device silently reads the wrong feature into each tree comparison.

### Quantization

Each feature has its own affine scale fitted to the observed training range:

```
q = clamp((x - FEAT_LO[i]) * FEAT_SCALE[i] - 127, -128, 127)
```

Thresholds are quantized the same way at export time, so a tree comparison becomes a
single `int8` compare. For a decision tree this is lossless as long as quantization never
moves a value across a threshold — measured agreement with the float model is 100.00% over
2160 windows, so no sample crosses a boundary here.

The device stores 5 arrays (`NODE_FEATURE`, `NODE_THRESH`, `NODE_LEFT`, `NODE_RIGHT`,
`NODE_CLASS`) plus per-tree offsets: 7 bytes × 1024 nodes = **7.0 KB**. Inference is a
pointer-chase per tree and a majority vote — no multiplies.

---

## Layout

```
scripts/pipeline.py          data generation, features, training, C++ export
scripts/verify_quantized.py  INT8 vs float32 equivalence check
src/detector.h / .cpp        on-device feature extraction + forest traversal
model/detector_quantized.h   generated: INT8 forest as C arrays
model/detector.joblib        generated: trained sklearn model
wokwi/sketch.ino             ESP32 simulation entry point
frontend/                    monitoring dashboard
tests/test_cases.md          50 test scenarios across the 5 classes
```

---

## Dataset

Designed against the structure of the **Korea University HCRL Car-Hacking dataset**
(DoS / fuzzy / RPM-spoof / gear-spoof, frames labelled normal or injected) — the standard
benchmark in this area, which makes published results comparable.

**This repo ships a synthetic generator, not HCRL itself.** `scripts/pipeline.py` builds a
continuous benign stream and injects attack bursts *into* it, so every attack window also
contains real background traffic and the classes genuinely overlap. An earlier version
generated each class as a separate clean block and scored 100.00% — that was an artifact of
separable synthetic data, not a real result. See [DATASET.md](DATASET.md) to swap in the
real HCRL capture; the feature pipeline takes the same columns.

---

## Running on ESP32

`wokwi/sketch.ino` replays scripted traffic — two benign windows, then DoS, fuzzy,
fake-RPM and fake-gear bursts — and prints the predicted class plus per-window latency
over serial at 115200 baud.

Open it at [wokwi.com](https://wokwi.com) with an ESP32 board, or build with PlatformIO.
`src/` has no Arduino dependency beyond `Serial`/`micros`, so it ports to any MCU with a
C++ toolchain.

### Status

- [x] Feature pipeline, training, INT8 export
- [x] INT8 forest verified equivalent to float32 (100.00% agreement, 2160 windows)
- [x] On-device feature extraction and forest traversal written
- [x] Monitoring dashboard
- [ ] **Not yet compiled or run on ESP32 hardware/Wokwi** — no toolchain on the dev
      machine yet, so on-device latency and flash/RAM figures are unmeasured. The 7.0 KB
      above is the computed array size, not a linker report.
- [ ] Validation against the real HCRL capture

The quantized forest is verified by re-implementing `classify()`'s exact integer path in
`scripts/verify_quantized.py` and comparing against sklearn. That validates the exported
arrays and the quantization math — it does not prove the C++ compiles.

---

## Stack

Python 3.14 · scikit-learn · NumPy · pandas · C++ (ESP32) · Wokwi
