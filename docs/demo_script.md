# 90-second demo video script

Record at 1920×1080, terminal font ≥ 16 pt so it survives compression. Keep the whole
thing in one take if you can — a visible cut invites the question "what did you skip?"

**Before you start:**

```bash
export PYTHONPATH=src          # $env:PYTHONPATH="src" on PowerShell
```

Have three terminal tabs ready so you are never typing a long command on camera.

---

## 0:00–0:12 — The problem

> "A car's brakes, engine and dashboard all talk on one network called the CAN bus.
> It has no sender field and no authentication, so any device that can transmit can
> impersonate the brakes. That's how the 2015 Jeep hack worked — 1.4 million cars
> recalled."

*On screen: the README's problem section, or just your face. Don't show code yet.*

---

## 0:12–0:25 — What the thing is

> "So I trained a neural network to spot injected messages, and shrank it to 4.9
> kilobytes so it runs on the car instead of in the cloud. It has to be on the car: a
> 500 kilobit bus is 4,000 frames a second, which is 250 microseconds per frame. A
> round trip to a server is tens of milliseconds. Either the decision is local or it's
> too late."

*On screen: the Mermaid architecture diagram from the README.*

---

## 0:25–0:50 — The live demo (the money shot)

Run it:

```bash
python -m securecan.demo --inject rpm --quiet-normal
```

> "This is a real recording of a Hyundai Sonata's bus, replayed at real speed. Green is
> normal. Now an attacker starts injecting fake engine-RPM messages…"

*Let the red ALERT lines fill the screen. Don't talk over them — let it run 4–5 seconds.*

> "Caught on the first injected frame. 100% recall, 0.09% false alarms, and the alert
> fires zero frames after the attack starts."

*On screen: let the summary block render. Point at the recall and FPR lines.*

---

## 0:50–1:10 — The honest part (this is what gets you hired)

> "But an F1 of 0.9994 on a security dataset should make you suspicious, so I went
> looking. I held out each attack and tested on it unseen — and recall on unseen
> spoofing was zero percent."

*On screen: the leave-one-attack-out table.*

> "It turned out the dataset injects the same eight bytes for the entire attack —
> 654,897 identical payloads, against 2,297 distinct legitimate ones on that same ID. So
> the model memorised a constant instead of learning the timing anomaly. Watch: same
> IDs, same timing, same bursts, I change only the fake value—"

```bash
python -m securecan.demo --inject rpm --novel-payload --quiet-normal
```

> "—and it misses everything."

*On screen: the 0.00% recall summary.*

> "That's a property of the dataset, not my model. Any per-frame classifier trained on
> it can take the same shortcut, which means published 99-percent results on this
> dataset — including mine — need that caveat."

---

## 1:10–1:25 — On the device

```bash
cd firmware/esp32 && pio run
```

> "The firmware is INT8 inference I wrote by hand — int32 accumulators, fixed-point
> requantisation — because the only Arduino build of TensorFlow Lite Micro for the
> ESP32 is from 2021 and doesn't compile any more. 1,740 bytes of weights, no
> dependencies, 7% of RAM."

*On screen: the RAM/Flash summary from the build output.*

> "And because hand-writing inference is only defensible if you prove it matches, there
> is a test that runs the same frames through my kernel and through TensorFlow's own
> INT8 interpreter and requires the class to be identical on every single frame."

```bash
pytest -q
```

*On screen: `32 passed`.*

---

## 1:25–1:30 — Close

> "Full write-up, every number reproducible, and an honest limitations section — link in
> the description."

---

## Notes

- **Do not** skip the failure section to make the demo look cleaner. It is the strongest
  30 seconds in the video; it is the part that shows judgement rather than tutorial
  following.
- If a run is slow on camera, use `--seconds 8`.
- If you are asked to shorten to 60 s, cut 0:12–0:25 (the architecture) and keep the
  failure section. Never the other way round.
