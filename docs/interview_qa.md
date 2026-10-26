# SecureCAN AI — Interview Questions and Answers

Three questions per milestone, with the kind of answer that holds up to a follow-up.
Every number here comes from a real run; the command that produced it is named so you
can re-run it in front of someone.

The hardest questions in this file are the ones in M3 about what the model is actually
keying on. If you can only prepare one section, prepare that one — it is where a good
interviewer will push, and it is where this project has a real answer.

---

## M1 — CAN bus and the threat model

**Q1. Why is a CAN bus insecure by design? Isn't that a bug someone should fix?**

It isn't a bug, it's a 1986 design goal. CAN was built for reliability and determinism
inside a sealed metal box, back when nothing on the bus could be reached from outside
the car. A frame carries an 11-bit ID, up to 8 data bytes, and a CRC — and no sender
field, no signature, no encryption. Every node sees every frame and decides for itself
whether to act on it. So any node that can transmit can impersonate any other node,
and there is nowhere in the format to put a credential even if you wanted to.

Retrofitting authentication is genuinely hard: 8 data bytes leaves no room for a MAC,
the ECUs are cost-optimised 8- and 16-bit parts, and a 500 kbit/s bus running at 50%
load has no spare bandwidth for extra frames. That's why detection is the practical
answer for existing vehicles — you can add a listener without touching the protocol or
any existing ECU.

**Q2. Why does the DoS attack use CAN ID 0x000 specifically?**

Because arbitration is bitwise and dominant-0 wins. When two nodes transmit at once,
each watches the bus while sending its ID; a node that sends a recessive 1 and sees a
dominant 0 knows it lost and backs off immediately. ID 0x000 is eleven dominant bits,
so it wins against every other ID on the bus, every time. Flooding 0x000 doesn't just
add traffic — it structurally starves every legitimate frame, including brakes.

That also makes DoS the easy attack to detect, and I'm explicit about that in my
results: 0x000 is an ID this car never sends, so `is_known_id` alone catches it. The
100% recall on DoS is not evidence the model is clever.

**Q3. Why detect instead of block?**

A false positive that suppressed a real brake or airbag frame would be more dangerous
than the attack. The failure modes are asymmetric: a missed detection leaves you where
you already were, a wrongly-dropped safety frame actively causes a crash. So the
firmware never blocks or drops — it classifies and raises an alert, and a
higher-level system or a human decides. That's also why I report false-positive rate
per CAN ID rather than only as an average: an average hides the fact that the false
alarms concentrate on one ID.

---

## M2 — Parsing and features

**Q1. What went wrong with the raw data, and how did you catch it?**

Three things, and all three would have silently inflated my results.

The captures have no header row and the attack-free file uses a different column layout
from the four attack files. Worse: rows with DLC < 8 shift the flag column left by
(8 − DLC), so a fixed-column read picks up a data byte where the label should be and
labels injected frames as normal. I pinned all three in `tests/test_parse.py`.

The one I'm most glad I caught is the clean tail. HCRL stopped injecting before each
recording ended, so 20–27% of every attack capture is attack-free — at the end. I split
by time, so without noticing that, a mostly-clean block would have landed in the test
set and flattered every number downstream. `notebooks/01_explore.ipynb` has the density
plot that shows it.

**Q2. You split by time and never shuffle. Why does that matter so much here?**

Consecutive CAN frames are strongly correlated — the same IDs on the same periodic
schedule, and attacks arrive in multi-second bursts. If I shuffle, frames from the
middle of a single attack burst end up in both train and test, so the model can
memorise that specific burst and I'd measure recall against data it has effectively
seen. A time split means the test set is a later stretch of driving, which is the
question I actually care about: does this work on traffic that hasn't happened yet.

Same reasoning for the scalers and the learned per-ID periods: fit on training-split
normal frames only. Fitting the scaler on everything leaks the test set's distribution
into the features.

**Q3. Why is the per-ID state table fixed size and never allocated on demand?**

Because the Fuzzy attack sprays random CAN IDs. A table that grew when it saw a new ID
would be a remote memory-exhaustion bug in the thing that is supposed to be the
defence — an attacker could OOM the detector by sending 2,048 distinct IDs. So the
table has one slot per ID learned from normal traffic (38 for this car), unknown IDs
never get one, and the *absence* of a slot is exactly what the `is_known_id` feature
reports. The security property and the feature are the same mechanism.

More generally there is no `malloc` anywhere on the device path, which
`tests/test_parity.py` enforces. A detector that can fail to allocate mid-drive is
worse than no detector.

---

## M3 — Model, and what it is really learning

This is the section to know cold.

**Q1. You report attack F1 of 0.9994. Why don't you believe it?**

Because I ran leave-one-attack-out: train on three attacks, test on the fourth unseen.

| held-out attack | recall on it |
|---|---|
| DoS | 100.00% |
| Fuzzy | 0.20% |
| Gear | 0.00% |
| RPM | 0.00% |

If the model had learned "this ID is arriving far too often for its schedule," that
would transfer between spoofing attacks, because it's the same anomaly with a different
target. Zero transfer means it learned something else.

My first hypothesis was that it keyed on `can_id` — memorising "0x316 means RPM
attack." I tested that by ablating the feature, and recall stayed 100.00 / 0.24 / 0.00 /
0.00. So `can_id` was *not* the cause, and I had to keep looking.

**Q2. So what is it keying on, and how did you prove it?**

The payload. Each HCRL spoofing campaign injected **one single constant 8-byte payload**
for every frame of the attack:

| attack | injected payload | injected frames | distinct injected values | distinct legitimate values, same ID |
|---|---|---|---|---|
| RPM, 0x316 | `45 29 24 FF 29 24 00 FF` | 654,897 | **1** | 2,297 |
| Gear, 0x43F | `01 45 60 FF 6B 00 00 00` | 597,252 | **1** | 8,093 |

Memorising one 8-byte constant is a much easier function to fit than a timing model, so
that is what the network did. Nothing about a memorised constant transfers to an attack
that uses a different constant — which is exactly the 0% above.

The experiment that settles it: take the real attack frames and change **only the fake
value**, leaving the IDs, the timing, the burst structure and every other feature
untouched. If the model were keying on timing, recall should barely move.

| attack | recall, HCRL's constant | recall, substituted value |
|---|---|---|
| RPM, 0x316 | 100.00% | **0.00%** |
| Gear, 0x43F | 100.00% | **16.28%** |

That is `payload_shortcut()` in `evaluate.py`, and you can watch it happen live with
`python -m securecan.demo --inject rpm --novel-payload`.

The crucial point: **this is a property of the dataset, not of my model.** Any
per-frame classifier trained on these captures can take the same shortcut. Published
99%-plus numbers on this dataset should be read with that in mind — including mine.

**Q3. Does that shortcut cost anything on normal traffic?**

Yes, and it shows up exactly where the theory predicts. On the attack-free capture
(988,871 frames) the overall false-alarm rate is 0.0675%, but it is not spread evenly:

| CAN ID | frames | false-alarm rate |
|---|---|---|
| 0x4B1 | 25,324 | 1.042% |
| **0x43F** | 50,641 | **0.579%** |
| 0x0A0 | 5,068 | 0.237% |

0x43F is the Gear attack's target, and 3,681 genuine 0x43F frames in normal traffic sit
**one byte** away from the memorised Gear constant. So the model false-alarms on real
frames for being too close to a value it memorised. The demo shows this plainly: nearly
every false positive is 0x43F carrying `01 45 60 FF 6A 00 00 00`, one byte from the
attack constant `01 45 60 FF 6B 00 00 00`.

That also explains the asymmetry in the table above — why Gear only fell to 16% while
RPM fell to 0%. The Gear constant sits *inside* the cloud of genuine 0x43F payloads, so
some novel fake values near it still land in the memorised region. The RPM constant is
far from genuine 0x316 traffic, so a substituted value falls straight out.

**Follow-up you should expect: "then how would you fix it?"** Train on the timing
features alone and accept lower headline numbers; or treat it as per-ID anomaly
detection against the learned period rather than per-frame classification; or evaluate
on a dataset where the injected payloads vary, such as ROAD. I'd want the second one,
because the timing anomaly is the part that generalises. I didn't do it here because the
honest finding was worth more than a bigger number.

---

## M4 — Live demo

**Q1. Your first demo reported a 19.8% false-alarm rate. What was wrong?**

Not the model — the fake traffic. I was generating normal frames with uniform-random
payload bytes, and real CAN payloads are nothing like that: they're mostly constant
bytes and slow counters. The detector was right to flag them. Replaying the real
attack-free capture instead gives 0.119%. The lesson I took is that when a number looks
bad, the test harness is as likely to be wrong as the thing being tested, and it is
worth a few minutes to find out which.

**Q2. It also reported 5,939 bursts for one injected burst. Why?**

I was grouping frames into bursts by label contiguity. But legitimate frames keep
arriving during an attack, so the label alternates attack/normal/attack frame by frame,
and every single frame became its own "burst." Grouping by a time gap instead (0.5 s)
gives the right answer. As a check that the threshold is sane, the burst durations it
recovers are 3.465–5.258 s, which independently matches HCRL's documented 3–5 second
injection windows — I didn't tune to that, it fell out.

**Q3. What does the demo actually measure, and what doesn't it prove?**

Two threads over a `python-can` virtual bus: a replayer pacing frames by their original
timestamps, and an IDS that gets one frame at a time with no lookahead, the same way the
firmware does. Nothing in the detector path can see the label while deciding.

Measured over 12 s of bus time each: DoS 100.00% recall at 0.107% FPR, Fuzzy 99.03% at
0.188%, Gear 100.00% at 0.205%, RPM 100.00% at 0.094%, every burst detected with a
median delay of 0 frames.

What it does not prove is embedded latency. Those are Python timings on a laptop
(median 378 µs float32, 74 µs INT8) — useful for the ~5× INT8 speedup, useless as an
ESP32 number. The device figure has to come from the device.

---

## M5 — Quantisation and the firmware

**Q1. Walk me through what INT8 quantisation actually does to a layer.**

A quantised tensor stores small integers plus a rule for reading them as reals:
`real = scale × (quantized − zero_point)`. The scale comes from the observed range of
that tensor; the zero_point is the integer that means exactly 0.0, which has to be
exact because padding and ReLU depend on it.

Substitute that into a dense layer's `y = Σ x·w + b`, with symmetric weights
(zero_point 0):

```
y_j = s_x · s_wj · Σ (q_x[i] − z_x) · q_w[j][i]  +  b_j
      \__________/   \_________________________/
      float constant      integer dot product
```

So the layer becomes an integer dot product and all the floats collapse into one
per-channel constant. The accumulator has to be int32 — 16 terms of up to 255×127 needs
23 bits — and the bias is pre-divided by `s_x · s_wj` at export time so it can just be
added. To hand the result to the next layer you rescale by
`M_j = (s_x · s_wj) / s_out`, a fraction below 1, which is split on the host into an
int32 multiplier and a shift so the device does integer arithmetic only.

One detail I like: my ReLU layers quantise with zero_point −128, so "clamp at 0.0" *is*
"clamp at −128," which is already the bottom of int8. The ReLU costs nothing — it's the
saturation you'd do anyway.

Measured cost on this model: 4,968 bytes instead of 6.6 KB float32, accuracy change
**+0.0017 pp** (it very slightly improved, which is noise at this scale), and int8
agrees with float32 on 99.997% of test frames.

**Q2. You dropped TensorFlow Lite Micro. Defend that.**

The plan was TFLite Micro. Its only ESP32 Arduino packaging is
`tanakamasayuki/TensorFlowLite_ESP32`, last released in 2021, and it does not compile
against the current platform (espressif32 8.x / ESP-IDF 5.5). Three distinct errors:
two bundled LCD drivers call IDF functions that were removed, and its vendored
flatbuffers assigns to a `const` member in `span`. The standard workaround is pinning
`platform = espressif32@6.x`, which drags the whole toolchain back two years and still
leaves the project on an abandoned library.

The model is three dense layers and 1,157 parameters. The int8 forward pass for that is
~70 lines, so I wrote it: 1,740 bytes of weights instead of a 4,968-byte flatbuffer
plus a parser plus generic kernels, no dependency, and it builds on any toolchain.

The part that makes this defensible rather than reckless is that I then proved it
matches. `tests/test_parity.py` runs the same real frames through my kernel and through
TFLite's own int8 interpreter and requires the predicted class to be **identical on
every frame** — not close on average. One disagreement per thousand frames is four
wrong answers a second on a 500 kbit/s bus. I also copied TFLite's
`MultiplyByQuantizedMultiplier` rounding faithfully instead of simplifying it, because
the exported multipliers were computed to match it.

**Q3. How do you know the firmware computes the same features as your Python?**

I don't trust reading them side by side — that's how off-by-one bugs ship, and a feature
mismatch would be silent, because the firmware would still produce confident answers
from a model trained on different numbers. So `tools/parity_host.cpp` compiles the real
`features.h` for the host, feeds it the same frames, and the test compares all 16
outputs against `features.py`.

It caught a genuine bug. `bus_rate` differed by exactly 1/30 — one frame on a feature
clipped at 30. The cause: the device keeps time as uint32 microseconds and evicts with
`(uint32_t)(now − old) > 10000`, which is exact, while Python used float64 seconds where
`0.024 − 0.010` is `0.014` but the stored timestamp was `0.013999999999999999`. A hair
below the cutoff, so Python dropped a frame the device kept. I fixed the *Python* side
with a 1 ns epsilon — the C++ was the correct side, and 1 ns is 1000× finer than the
data's 1 µs resolution, so it can't change any decision integer arithmetic would make.

Related: time on the device is a relative microsecond counter, not Unix seconds. A
float32 at 1.478e9 has ~128 s spacing, so every inter-frame gap would round to zero and
half the features would be dead. Using uint32 µs also makes the ~71 minute wraparound
free, because unsigned subtraction gives the true elapsed time across the rollover.

---

## M6 — Judgement and limitations

**Q1. What's the single biggest weakness of this project?**

The model memorises one 8-byte constant per spoofing attack rather than learning the
timing anomaly, so it does not detect unseen spoofing. I can quantify it: substituting
only the fake value drops RPM recall from 100% to 0%. The second-biggest is that it's
one car (Hyundai YF Sonata), one dataset, so the learned ID periods and the 38-ID table
don't transfer to another vehicle without re-learning.

I'd rather lead with that than be caught by it. The detection pipeline, the quantisation
and the parity testing are solid and reusable; the generalisation claim is not, and I
know precisely why.

**Q2. What would you do next, with another week?**

Train on timing features only and report the honest, lower number — that tells me how
much of the performance is the part that generalises. Then evaluate on ROAD (Oak Ridge),
which has varied injected payloads, so the shortcut isn't available. And measure on real
hardware with an SN65HVD230 transceiver instead of a simulator, because my device
timings are from Wokwi.

**Q3. Which numbers in your README are measured and which are estimated?**

All of them are measured, and the README says which command produced each. Where a
number is from the Wokwi simulator rather than silicon, it says "measured in Wokwi
simulation" — Wokwi runs the same ELF on a cycle-level model of the chip, but it doesn't
model flash-cache misses, so expect the same order of magnitude on real hardware, not
the same three digits. Where something isn't measured yet, it's marked as not measured
rather than estimated.

That discipline is deliberate. The one number I was tempted to state without measuring
was the leave-one-attack-out result, because 0% looked like a bug in my code. It wasn't.
