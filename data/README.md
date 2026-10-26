# Dataset — HCRL Car-Hacking

The raw files are **not committed** (several hundred MB). This file tells you how to
get them and what they contain.

## Source

**Car-Hacking Dataset**, Hacking and Countermeasure Research Lab (HCRL),
Korea University.

- Landing page: <https://ocslab.hksecurity.net/Datasets/car-hacking-dataset>
- The page links to a public Dropbox folder — **no registration or account needed**.

To fetch it:

```bash
python tools/fetch_dataset.py
```

That downloads the zip into `data/raw/` and extracts it. Roughly 500 MB zipped.

## How it was recorded

CAN traffic was logged through the **OBD-II port of a real Hyundai YF Sonata** while
attacks were injected onto the live bus. Each attack ran in bursts of **3–5 seconds**,
and each file holds **30–40 minutes** of traffic in total.

Because the attacks were injected into traffic that kept flowing, every attack file
contains **both** normal and injected frames. That is what makes the dataset usable:
we get the attack and its realistic background in the same capture.

## Files

| File | Contents |
|---|---|
| `DoS_dataset.csv` | Normal traffic + DoS injection (ID `0x000`, every ~0.3 ms) |
| `Fuzzy_dataset.csv` | Normal traffic + random ID/payload injection (every ~0.5 ms) |
| `gear_dataset.csv` | Normal traffic + gear-signal spoofing (every ~1 ms) |
| `RPM_dataset.csv` | Normal traffic + RPM-signal spoofing (every ~1 ms) |
| `normal_run_data.txt` | Attack-free traffic only — **different format**, see below |

## Row format (the four attack CSVs)

No header row. Columns are:

```
Timestamp, CAN ID (hex), DLC, DATA[0] ... DATA[DLC-1], Flag
```

Example:

```
1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
```

`Flag` is the label: **`R` = regular** (genuine traffic), **`T` = injected** (attack).

### The parsing trap

**The number of columns varies per row.** `DLC` says how many data bytes follow, and
rows with `DLC < 8` have *fewer* data fields — which means the `Flag` column shifts
left.

```
1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R   <- flag at index 11
1478198376.389800,0153,5,00,00,00,00,20,R            <- flag at index 8
```

Reading this with a fixed column index silently puts data bytes into the label column
and produces garbage labels. `src/securecan/parse.py` reads each row by length:
the flag is always the **last** field, and the data bytes are fields `3 .. 3+DLC`.
Missing bytes are zero-filled.

### The normal-traffic file

`normal_run_data.txt` is **not** CSV. It is a candump-style log, roughly:

```
Timestamp: 1479121434.850202        ID: 0430    000    DLC: 8    some bytes here
```

It also has no `Flag` column at all — every row is normal by definition. The parser
detects the format by extension and handles it separately.

## Labels used in this project

| Label | Class | Source file |
|---|---|---|
| 0 | Normal | all files, rows with `R` |
| 1 | DoS | `DoS_dataset.csv`, rows with `T` |
| 2 | Fuzzy | `Fuzzy_dataset.csv`, rows with `T` |
| 3 | Gear spoof | `gear_dataset.csv`, rows with `T` |
| 4 | RPM spoof | `RPM_dataset.csv`, rows with `T` |

## Splitting

Split **by time**, never shuffled: the first 70% of each file is train, the next 15%
validation, the last 15% test.

Shuffling would leak information. Consecutive CAN frames are strongly correlated — the
frame 2 ms after an injected frame is probably also injected. A shuffled split puts
near-identical neighbours on both sides of the boundary and the score looks far better
than it is. Scaler statistics and the learned per-ID periods are also fit on the
**training portion only**, for the same reason.

## Citation

> H. Lee, S. H. Jeong and H. K. Kim, "OTIDS: A Novel Intrusion Detection System for
> In-vehicle Network by Using Remote Frame," PST (Privacy, Security and Trust), 2017.

> E. Seo, H. M. Song and H. K. Kim, "GIDS: GAN based Intrusion Detection System for
> In-Vehicle Network," PST, 2018.
