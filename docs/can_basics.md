# CAN Bus Basics — and Why It Can Be Hacked

**Milestone 1 of SecureCAN AI.** Everything here is background you need before touching
the data. Read it once, then take the quiz at the bottom. If you can answer those ten
questions out loud, you can handle the CAN half of an interview.

---

## 1. Why this project exists

In July 2015, two researchers (Charlie Miller and Chris Valasek) sat on a sofa with a
laptop and took control of a Jeep Cherokee being driven on a highway ten miles away.

They got in through the **Uconnect infotainment system** — the radio/nav unit — which
was reachable over the cellular network. From there they rewrote the firmware of a chip
that sat between the entertainment system and the car's internal network. Once they
could put messages on that internal network, they could operate the air conditioning,
the wipers, the transmission, and the brakes.

Chrysler recalled **1.4 million vehicles**.

The internal network they reached is the **CAN bus**. Nothing on it checked who was
sending those messages, because nothing on it *can*. That is the hole this project
tries to watch.

Two constraints shape the whole design:

- **No internet.** A car in a tunnel, in a car park, or in a dead zone still has to
  defend itself. Detection has to happen on a chip inside the car.
- **No time.** If a fake brake message is on the bus, an alert five seconds later is
  useless. We need an answer in milliseconds.

That is why this is an *embedded* AI project and not a cloud one.

---

## 2. What a CAN bus physically is

Forget computers for a second. A CAN bus is **two twisted copper wires** running
through the car, called CAN High and CAN Low, with a 120 Ω resistor at each end.

Every ECU (Engine Control Module, transmission, ABS, airbags, doors, dashboard...)
taps onto those same two wires. There is **no central hub, no router, no master**.
It is one long shared party line.

Bits are sent as a **voltage difference** between the two wires:

| State | CAN High | CAN Low | Difference | Logical value |
|---|---|---|---|---|
| **Recessive** | ~2.5 V | ~2.5 V | ~0 V | 1 (also the idle state) |
| **Dominant** | ~3.5 V | ~1.5 V | ~2 V | 0 |

Two details here matter enormously later, so slow down:

**(a) Difference, not absolute voltage.** Electrical noise from the spark plugs and
alternator hits both wires equally, so it cancels out when you subtract. This is why a
1980s design still works next to a running engine.

**(b) Dominant always beats recessive.** If one ECU sends a dominant 0 and another
sends a recessive 1 at the same instant, the bus reads **0**. The dominant node
physically drags the voltage. Think of a room where "recessive" is staying silent and
"dominant" is shouting — if anyone shouts, the room is loud.

That second rule is the clever heart of CAN, and also the thing an attacker abuses.

---

## 3. The CAN frame, field by field

A classic CAN frame (CAN 2.0A, 11-bit ID) looks like this on the wire:

```
 SOF   IDENTIFIER   RTR IDE r0   DLC    DATA        CRC      ACK    EOF   IFS
 [1]      [11]      [1] [1] [1]  [4]   [0..64]   [15+1]    [1+1]   [7]   [3]
  |         |        |           |       |          |         |      |
 start   who/      data or     how    the actual  error    "someone  gap
 of      priority  request?   many    payload     check     got it"
 frame                        bytes
```

| Field | Size | What it does |
|---|---|---|
| **SOF** | 1 bit | A dominant bit that says "frame starting" and syncs everyone's clocks. |
| **Identifier** | 11 bits | **Both the message's name and its priority.** More on this below. |
| **RTR** | 1 bit | 0 = here is data. 1 = I'm *asking* someone else to send this ID. |
| **IDE** | 1 bit | 0 = 11-bit ID (what we use). 1 = 29-bit extended ID. |
| **r0** | 1 bit | Reserved. |
| **DLC** | 4 bits | Data Length Code: how many payload bytes follow, 0 to 8. |
| **Data** | 0–8 bytes | The payload. This is all you get. |
| **CRC** | 15 bits | Checksum so receivers can spot corrupted bits. |
| **ACK** | 2 bits | Sender transmits recessive here; any receiver that got the frame cleanly pulls it dominant. |
| **EOF** | 7 bits | End of frame. |
| **IFS** | 3 bits | Required gap before the next frame. |

So an 8-byte frame is about **111 bits** on the wire (a bit more with stuffing, below).
At the common powertrain speed of **500 kbit/s**, that works out to roughly
**4,000 frames per second** maximum. Remember that number — our firmware has to keep up
with it, which is where the "under 250 microseconds per frame" target comes from.

### Two things that surprise people

**The identifier is not an address.** `0x316` does not mean "send this to ECU number
316". It means "this frame *is* engine RPM". Every ECU on the bus hears every single
frame and decides for itself, based on the ID, whether it cares. It's broadcast radio,
not postal mail.

**There is no "from" field anywhere in the frame.** Look at the table again. There is
nowhere to write who sent it. A frame carries *what* it is, never *who* said it.
This single omission is the root of everything in section 5.

**Bit stuffing (minor, but explains odd frame lengths):** after five identical bits in
a row, the transmitter inserts one opposite bit so receivers can keep their clocks
locked. The receiver strips them out. It's why a frame is "about" 111 bits and not
exactly 111.

---

## 4. Arbitration: how two ECUs talk at once without a collision

Ethernet handles collisions by detecting the crash, backing off a random amount, and
retrying. CAN does something smarter: **collisions are resolved while they happen, and
the winner never even pauses.**

Here is the mechanism:

1. Any ECU may start transmitting when the bus is idle.
2. While sending the 11-bit identifier, each transmitter **reads the bus back, bit by
   bit**, and compares it to what it just sent.
3. If a node sends **recessive (1)** but reads back **dominant (0)**, it knows somebody
   else is transmitting a lower ID. It **stops immediately** and switches to listening.
4. The winner never notices. Its frame continues uninterrupted — no corruption, no
   retry, no lost time.

Walk through two ECUs starting together, one sending `0x100`, one sending `0x300`:

```
bit position :  10  9  8  7  6  5  4  3  2  1  0
ID 0x100     :   0  0  1  0  0  0  0  0  0  0  0
ID 0x300     :   0  1  1  0  0  0  0  0  0  0  0
bus (wired-AND):  0  0  ...
                     ^
                     0x300 sent recessive 1, read back dominant 0 --> it loses, backs off.
                     0x100 continues and transmits its whole frame.
```

**The rule in one line: the lower the numeric ID, the higher the priority.**

That is good engineering. Brake messages get a low ID and always jump the queue ahead
of the "door is ajar" lamp. Priority is built into the addressing scheme for free.

Now flip it around, which is the whole point of this section:

> **ID `0x000` is eleven dominant bits. It wins every arbitration against every
> other message on the bus, every single time, by design.**

An attacker who floods `0x000` isn't exploiting a bug. They are using the arbitration
rule exactly as specified, and the rule obediently starves every real message. That is
the DoS attack in section 6, and now you know *why* it uses that specific ID — which is
a very good thing to be able to explain in an interview.

---

## 5. Why CAN has no security

The honest answer is: **nobody was attacking cars in 1986.**

Bosch designed CAN for a closed, physically sealed, wired network where every node was
installed by the manufacturer. The threat model was *electrical noise and broken
wires*, not adversaries. On those terms the design is excellent. It simply has no
answer to a question nobody thought to ask.

Concretely, what's missing:

| Missing | Consequence |
|---|---|
| **No sender identity** | A frame never says who sent it. Any node can send any ID, and it is indistinguishable from the real sender. |
| **No authentication** | No signature, no MAC, no key. The CRC catches *accidental* bit flips; it does nothing against a deliberately well-formed forgery — the attacker just computes the correct CRC. |
| **No encryption** | Every frame is plaintext. Plug into the port and read the whole car. |
| **Broadcast trust** | Every ECU hears everything and trusts whatever arrives with an ID it cares about. |
| **No room** | 8 data bytes. There is physically no space for a signature, and 1980s microcontrollers couldn't have computed one at bus speed anyway. |

And a subtle one worth knowing: **the ACK bit does not help.** It tells the sender that
*somebody* received the frame correctly. It never says who, and it says nothing about
whether the sender was legitimate.

### So how does an attacker get on the bus?

They need to reach those two wires, physically or through something already connected:

- **The OBD-II port** — legally mandatory, sitting under the dashboard of every car
  since the mid-90s, wired straight to the bus.
- **The infotainment unit** — has cellular, WiFi and Bluetooth, and a path to the bus.
  This is the Jeep route, and it is the scary one because it is remote.
- **Aftermarket dongles** — insurance trackers and fleet devices plugged into OBD-II,
  often with weak security of their own.
- **Any replaceable module** — a compromised aftermarket part is a node on the bus.

Once they're on, CAN gives them no resistance at all. Hence: watch the traffic and
raise an alarm, because we cannot fix the protocol itself.

---

## 6. The four attacks in our dataset

The HCRL Car-Hacking Dataset was recorded on a real **Hyundai YF Sonata** by plugging a
logger into the OBD-II port and injecting attacks while normal traffic kept flowing.
Each row is one frame:

```
Timestamp,          CAN ID,  DLC,  DATA[0] .. DATA[7],        Flag
1478198376.389427,  0316,    8,    05 21 68 09 21 21 00 6f,   R
```

`Flag` is the label: **R = regular** (real traffic), **T = injected** (the attack).
Important: the attack files contain *both* — the injected frames are mixed into live
normal traffic, which is exactly the realistic setting we want.

### 6.1 DoS — Denial of Service

**What:** flood ID `0x000` roughly **every 0.3 ms**.

**Why it works:** section 4. `0x000` wins every arbitration, so genuine messages get
pushed further and further back and the bus saturates. ECUs stop receiving the data
they need on time.

**What it looks like in the data:** the frame rate explodes, the gap between frames
collapses, and a single ID that normal traffic never uses suddenly dominates the log.

**Difficulty: easy.** It is extremely loud. A simple rate threshold catches it. We
should expect near-perfect scores here and should *not* brag about them — an
interviewer who knows the field will immediately ask about the other three.

### 6.2 Fuzzy

**What:** inject **randomly chosen IDs with random payloads**, roughly every 0.5 ms.

**Why:** it's reconnaissance, a shotgun. The attacker doesn't yet know which IDs do
anything interesting, so they spray the space and watch what the car does.

**What it looks like:** IDs that never appear in normal traffic start showing up, and
the payload bytes look uniformly random rather than like a real sensor reading.

**Difficulty: moderate.** "Have I ever seen this ID before?" is a strong signal.

### 6.3 Gear spoofing

**What:** send messages on the **real gear-status ID** carrying a **fake gear value**,
roughly every 1 ms.

**What it looks like — and this is the interesting part:** every injected frame is a
*perfectly valid CAN frame*. Legitimate ID. Legal DLC. Correct CRC. Examine any single
frame in isolation and there is **nothing wrong with it**.

It is only wrong **in context**:

- That ID normally arrives every ~10 ms. Now it's arriving every ~1 ms — roughly ten
  times too often, because the real ECU is *still sending too*. The genuine and fake
  messages interleave.
- The payload now flips back and forth between the real gear and the fake one, so
  consecutive values for that ID jump around in a way a physical gearbox never does.

**Difficulty: hard**, and this is the attack that justifies the entire project. A
"malformed frame" filter sees nothing. You need **per-ID timing history** and **payload
change history** — which is precisely why our feature list in M2 has `dt_same_id`,
`period_ratio`, and `payload_change`.

### 6.4 RPM spoofing

**What:** identical mechanism to gear spoofing, aimed at the **RPM ID** — fake engine
speed, roughly every 1 ms.

**What it looks like:** same fingerprint. Known ID arriving far too often, payload
oscillating between truth and lie.

**Difficulty: hard**, same reasons.

### Summary table

| Attack | Injected ID | Rate | Giveaway | Hard? |
|---|---|---|---|---|
| DoS | `0x000` | ~0.3 ms | Bus rate explodes, lowest-priority-wins flood | Easy |
| Fuzzy | random | ~0.5 ms | Unknown IDs, random-looking payloads | Moderate |
| Gear | real gear ID | ~1 ms | Known ID, ~10× too frequent, payload flip-flops | Hard |
| RPM | real RPM ID | ~1 ms | Same as gear | Hard |

---

## 7. Why a machine learning model (and why we still build a rule baseline)

Look at the summary table again. For DoS, a rule is enough: *if frames-per-10 ms goes
above X, alarm.* You don't need a neural network for that, and claiming you do is a bad
look.

Spoofing is where rules get painful. You'd have to hand-write a threshold **per ID**,
for hundreds of IDs, and then retune them because a cold engine, a hard acceleration
and a traffic jam all produce legitimately different timing. The thresholds become a
maintenance swamp.

A model learns the normal envelope straight from the data instead, and generalises
across IDs it was never specifically tuned for.

**But we build the rule baseline anyway**, deliberately, for two reasons:

1. If the rule already gets F1 0.99 on DoS, then the model's DoS score proves nothing
   and the honest story is "the model earns its keep on spoofing."
2. An interviewer **will** ask "did you try something simpler first?" Having a real
   number for the simple thing is the difference between sounding like an engineer and
   sounding like someone who reached for a neural network by reflex.

---

## 8. Where our IDS sits, and what it deliberately does not do

```
CAN frame  ->  feature extractor  ->  INT8 neural net  ->  class  ->  alert
               (small per-ID               (TFLite         (Normal/     (LED +
                state table)                Micro)          DoS/Fuzzy/   serial
                                                            Gear/RPM)    log)
```

It is a **passive monitor**. It reads every frame and raises an alarm. It **never
blocks a message**.

That restraint is a design decision, and you should be ready to defend it:

> If we block on a false positive, we could suppress a *genuine* brake, airbag or
> steering message. A missed alert is bad. Disabling a safety function because a model
> was 51% confident is catastrophic. Detection-only is also what shipping automotive
> IDS products do — they report to a gateway or fleet backend rather than filtering the
> bus.

Note too that the detector sits on the bus and sees *everything* — it does not need to
be built into each ECU. One cheap chip watches the whole network.

---

## 9. Vocabulary check

| Term | Plain meaning |
|---|---|
| **ECU** | Electronic Control Unit. One of the small computers in the car. |
| **CAN ID / arbitration ID** | 11-bit number that names the message *and* sets its priority. Lower = higher priority. |
| **DLC** | Data Length Code. How many payload bytes, 0–8. |
| **Dominant / recessive** | Logical 0 / logical 1. Dominant always overrides recessive on the shared wires. |
| **Arbitration** | The bit-by-bit contest that decides who transmits when two nodes start together. |
| **Bit stuffing** | Inserting an opposite bit after 5 identical ones to keep clocks synced. |
| **OBD-II** | The mandatory diagnostic port under the dashboard, wired to the bus. |
| **TWAI** | Espressif's name for the ESP32's built-in CAN controller (trademark reasons). |
| **IDS** | Intrusion Detection System. Watches and alerts; does not block. |
| **Injected frame** | A frame put on the bus by the attacker. `Flag = T` in our dataset. |

---

## 10. Quiz — ten questions

Answer these from memory first. Answers are in the next section; don't peek until
you've tried all ten.

1. The CAN frame has no "sender" field. Why does that matter for security, and what
   exactly can't you do because of it?

2. Two ECUs start transmitting at the same instant, one with ID `0x200` and one with
   ID `0x180`. Which one wins, how does the loser find out, and how much time does the
   winner lose to the collision?

3. Why does the DoS attack specifically use ID `0x000` instead of any other ID?

4. A gear-spoofing frame has a legitimate ID, a legal DLC, and a correct CRC. So what
   is actually wrong with it, and which two features from our plan would expose it?

5. CAN has a 15-bit CRC on every frame. Why doesn't that stop an attacker?

6. Classic CAN gives you 8 bytes of payload. Give two separate reasons that makes
   adding a cryptographic signature difficult.

7. At 500 kbit/s the bus carries roughly 4,000 frames per second. What per-frame
   processing budget does that give our ESP32 firmware, and where does that number
   come from?

8. Name three realistic ways an attacker gets access to a car's CAN bus. Which is the
   most dangerous, and why?

9. Our IDS detects but never blocks messages. Give the safety argument for that choice.

10. Of the four attacks, which should be easiest to detect and which hardest? Explain
    the reasoning — not just the names.

---

## 11. Answers

<details>
<summary>Click to expand (try the quiz first)</summary>

**1.** A frame states *what* it is (the ID names the signal), never *who* sent it. So
a receiving ECU has no way to tell a frame from the genuine engine controller apart
from an identical frame from an attacker — they are byte-for-byte the same. You cannot
do source authentication, and therefore you cannot do access control either: there's no
notion of "only the ECM may send `0x316`."

**2.** `0x180` wins — it's numerically lower, so it has a dominant bit where `0x200`
has a recessive one. The loser finds out by reading the bus back as it transmits: it
sends recessive, reads dominant, and immediately knows a higher-priority frame is in
flight, so it stops and becomes a receiver. The winner loses **no time at all** —
arbitration is non-destructive, its frame continues uninterrupted with no retry.

**3.** `0x000` is eleven dominant bits, the lowest possible ID, so it wins arbitration
against *every* other message on the bus unconditionally. Flooding it starves all real
traffic. The attacker isn't exploiting a bug — they're using the priority rule exactly
as designed.

**4.** Nothing is wrong with the frame *in isolation* — it's a valid CAN frame. It's
wrong **in context**: (a) that ID now arrives ~10× more often than its normal period,
because the real ECU is still transmitting and the fake messages interleave with it;
and (b) the payload flip-flops between the true gear and the fake one. The two features
that catch it are **`period_ratio`** (`dt_same_id` ÷ that ID's learned normal period,
which drops far below 1) and **`payload_change`** (bits changed versus the previous
payload on the same ID, which spikes).

**5.** The CRC is an *error-detecting* code, not a *cryptographic* one. There's no
secret key involved, so anyone can compute the correct CRC for any payload they like —
the attacker's hardware does it automatically. It protects against accidental
corruption from noise, not against a deliberate, well-formed forgery.

**6.** (a) **Space:** a meaningful MAC is 8–16 bytes, but the entire payload is only 8
bytes, so a signature would leave no room for the actual data — you'd need to split
every message across multiple frames and roughly double bus load. (b) **Compute and
latency:** 1980s–90s automotive microcontrollers couldn't compute a MAC at 4,000
frames/second, and even today the extra latency eats into hard real-time deadlines for
safety messages. (Also valid: no key distribution or key storage scheme exists across
dozens of ECUs from different suppliers.)

**7.** 1 second ÷ 4,000 frames = **250 microseconds per frame**. That's the ceiling: if
a frame takes longer than that to process, we fall behind a saturated bus and start
dropping frames. The 4,000 comes from the bit rate divided by frame length — about 111
bits for an 8-byte standard frame, so 500,000 ÷ ~125 ≈ 4,000.

**8.** (a) The **OBD-II port** — mandatory, under the dash, wired straight to the bus,
but needs physical access. (b) The **infotainment unit** — has cellular/WiFi/Bluetooth
and a path through to the bus. (c) **Aftermarket dongles** (insurance or fleet
trackers) plugged into OBD-II with weak security of their own. The infotainment route
is the most dangerous because it is **remote** — it scales to every car of that model
simultaneously, with no physical access, which is exactly what made the Jeep hack a
1.4-million-vehicle recall instead of one stolen car.

**9.** Blocking means acting on the model's output, and no model is perfect. A false
positive that blocks a genuine brake, airbag or steering message could disable a safety
function and directly cause a crash. A missed detection is bad, but it leaves the car
exactly as safe as it is today with no IDS at all. The asymmetry is enormous, so the
safe failure mode is to alert and let a human or a higher-level system decide. Shipping
automotive IDS products make the same call.

**10. Easiest: DoS.** It is extremely loud — the frame rate explodes and an ID that
never normally appears saturates the bus. A single rate threshold finds it, no ML
needed. **Hardest: gear and RPM spoofing.** Every injected frame is individually
valid — real ID, legal DLC, correct CRC — so nothing is detectable frame-by-frame. You
can only catch them by comparing against *history*: how often that ID normally arrives,
and how its payload normally evolves. (Fuzzy sits in between: unknown IDs give it
away.)

</details>

---

## What's next

**M2 — Data and features.** We download the real HCRL files, write a parser that
survives their formatting quirks, explore the traffic in a notebook (which IDs exist,
how often each one really arrives, what each attack looks like), and implement
`features.py` with tests.

Every feature in M2 traces back to something in this document. When you build
`period_ratio`, remember it exists because of section 6.3.
