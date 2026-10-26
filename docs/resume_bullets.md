# Resume bullets

Every number is from a real run and is reproducible from the repo. Nothing here is
rounded in my favour.

## The three to use

> **SecureCAN AI — on-device CAN bus intrusion detection** · Python, TensorFlow, C++, ESP32
>
> - Trained a 1,157-parameter neural network on **17.5M real CAN frames** (HCRL
>   Car-Hacking, Hyundai YF Sonata) to classify four injection attacks, reaching
>   **0.9994 attack F1 at a 0.0189% false-positive rate** on a complete 2.6M-frame
>   time-based held-out split, detecting **every one of 101 attack bursts on its first
>   injected frame**.
> - Quantised the model to **INT8 (4.9 KB, +0.002 pp accuracy change)** and deployed it
>   to an ESP32 in **hand-written integer inference** — int32 accumulators with
>   fixed-point requantisation — after the only Arduino build of TensorFlow Lite Micro
>   proved incompatible with ESP-IDF 5; **validated class-for-class against TFLite's own
>   INT8 interpreter on every test frame**, using 7.7% of RAM.
> - Profiled the firmware on-device and found two latency bugs the compiler had hidden:
>   float division expanding to Newton–Raphson on an FPU with **no divide instruction**
>   (**13.5× on that stage**), and TFLite's reference rescale compiling to `__divdi3`, a
>   **software 64-bit division loop**, on 32-bit Xtensa.
> - Found and quantified a **shortcut in the benchmark dataset itself**: because each
>   attack campaign injects one constant payload, substituting only the fake value drops
>   recall from **100% to 0%**, so published 99%-plus results on this dataset do not
>   reflect unseen-attack performance.

## Shorter, if you only get one line

> Built an INT8 neural-net CAN-bus intrusion detector for ESP32 (4.9 KB, 0.9994 attack
> F1, 0.0189% FPR over 17.5M real frames); hand-wrote the integer inference kernel and
> validated it against TFLite, and identified a dataset shortcut that collapses recall
> from 100% to 0% on unseen payloads.

## Elevator pitch (spoken, ~30 s)

> Cars run their brakes, engine and dashboard on one network with no authentication —
> that's how the Jeep hack worked. I trained a tiny neural network to spot injected
> messages and squeezed it to 4.9 kilobytes so it runs on a microcontroller in the car,
> because at 4,000 frames a second you have 250 microseconds to decide and no time to
> ask a server. It gets 0.9994 F1 on seventeen and a half million real frames from a
> Hyundai Sonata.
>
> The more interesting result is that I didn't trust that number. I held each attack out
> and tested unseen, and recall was zero. It turned out the dataset injects the same
> eight bytes for an entire attack, so the model memorised a constant instead of
> learning the timing anomaly. I proved it by changing only the fake value — recall went
> from 100% to 0%. That's a property of the dataset, so it applies to the published
> results on it too.

## Why the third bullet is the one that works

Interviewers see "99% accuracy on a public dataset" constantly, and it is close to
worthless as a signal — it usually means the candidate ran a tutorial. A candidate who
*distrusted their own good result, designed an experiment to attack it, and found a flaw
in the benchmark* is demonstrating the thing the job actually needs.

It also pre-empts the strongest question they could ask. If they probe the 0.9994, you
are not defending — you already have the answer, with numbers.

**Have these ready as follow-ups:**

- *"How did you find it?"* → Leave-one-attack-out: 100% / 0.20% / 0.00% / 0.00%. Zero
  transfer between two spoofing attacks that share the same anomaly meant it had learned
  something attack-specific.
- *"What did you think it was first?"* → `can_id`. I ablated the feature and recall
  barely moved, so that guess was wrong and I kept looking.
- *"How would you fix it?"* → Timing features only, and accept a lower honest number;
  or reframe as per-ID anomaly detection against the learned period; or evaluate on ROAD,
  whose payloads vary. The timing anomaly is the part that generalises.
- *"What does it cost you?"* → `0x43F` false-alarms at 0.579% on genuine traffic, because
  3,681 real frames of that ID sit one byte from the memorised constant.

## Skills this project actually evidences

Worth naming explicitly on the CV, because they are the embedded-AI ones:

- **Quantisation internals** — not "I called the converter," but per-channel scales,
  int32 accumulators, fixed-point requantisation multipliers, ReLU folded into the int8
  clamp.
- **Cross-implementation validation** — a Python↔C++ parity harness that caught a real
  float64-boundary bug in `bus_rate` (one frame, 1/30 of a clipped feature).
- **Fixed-memory embedded design** — no `malloc` on the device path, enforced by test;
  the per-ID table deliberately cannot grow, because the Fuzzy attack would otherwise
  exhaust it.
- **Real-time budgeting** — 250 µs/frame derived from 500 kbit/s and CAN 2.0A frame
  length, not asserted.
- **Instrument calibration** — proved the ESP32 simulator wasn't instruction-accurate
  (105 cycles for a 4–6 cycle loop) instead of trusting or dismissing its numbers, and
  published the resulting gap rather than a figure that couldn't be defended.
- **Reading compiler output** — diagnosed `__divdi3` and `__divsf3` from the
  disassembly; knowing that 32-bit Xtensa has no 64-bit divide and the LX6 FPU has no
  divide instruction is the kind of detail an embedded-AI interviewer probes for.
- **Evaluation discipline** — time-based splits, scalers fit on training normals only,
  leave-one-attack-out, per-ID false-alarm breakdown.
