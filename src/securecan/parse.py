#!/usr/bin/env python3
"""Parse the HCRL Car-Hacking CSV/TXT logs into clean numpy arrays.

The four attack CSVs have no header and a *variable* number of columns: DLC says
how many data bytes follow, so a DLC-5 row is three fields shorter than a DLC-8 row
and the Flag column shifts left with it. Reading at a fixed index silently puts a
data byte where the label should be, so we parse each row by its own length.

The attack-free file is a different format entirely (candump-style text, no flag
column), handled separately.

Output per file: a dict of numpy arrays, one row per CAN frame.

    python -m securecan.parse            # parse everything, write data/processed/
    python -m securecan.parse --peek     # just show the first lines of each raw file
"""

import argparse
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Class labels. 0 is normal; each attack file contributes exactly one attack class.
NORMAL, DOS, FUZZY, GEAR, RPM = 0, 1, 2, 3, 4
CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]

# Which attack class the injected ("T") rows of each file represent. Matched as a
# substring against the filename, case-insensitively, because mirrors rename things.
ATTACK_OF_FILE = {
    "dos": DOS,
    "fuzzy": FUZZY,
    "gear": GEAR,
    "rpm": RPM,
}


def attack_class_for(path):
    """Which attack class does this file's injected traffic represent?"""
    name = path.name.lower()
    for key, cls in ATTACK_OF_FILE.items():
        if key in name:
            return cls
    return None  # attack-free capture


def parse_attack_csv(path, attack_class):
    """Parse one of the four attack CSVs.

    Row layout, no header:
        timestamp, can_id(hex), dlc, data[0..dlc-1], flag

    `flag` is 'R' (regular) or 'T' (injected). Everything between index 3 and the
    flag is payload, so we slice by DLC rather than trusting a fixed width.
    """
    ts, ids, dlcs, data, labels = [], [], [], [], []
    bad = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                bad += 1
                continue
            try:
                t = float(parts[0])
                cid = int(parts[1], 16)
                dlc = int(parts[2])
            except ValueError:
                bad += 1
                continue

            flag = parts[-1].strip().upper()
            if flag not in ("R", "T"):
                bad += 1
                continue

            # payload is whatever sits between the DLC and the flag; clamp to the
            # DLC so a malformed row can't overrun, and zero-fill short payloads
            payload = parts[3: 3 + min(dlc, 8)]
            row = [0] * 8
            try:
                for i, b in enumerate(payload):
                    row[i] = int(b, 16)
            except ValueError:
                bad += 1
                continue

            ts.append(t)
            ids.append(cid)
            dlcs.append(dlc)
            data.append(row)
            labels.append(attack_class if flag == "T" else NORMAL)

    if bad:
        print(f"       skipped {bad} malformed rows")

    return _pack(ts, ids, dlcs, data, labels)


# Timestamp: 1479121434.850202        ID: 0430    000    DLC: 8    05 28 ...
NORMAL_RE = re.compile(
    r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+).*?DLC:\s*(\d+)\s+(.*)$"
)


def parse_normal_txt(path):
    """Parse the attack-free capture. Different format, and every row is normal."""
    ts, ids, dlcs, data, labels = [], [], [], [], []
    bad = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = NORMAL_RE.search(line)
            if not m:
                bad += 1
                continue
            t, cid_hex, dlc_s, rest = m.groups()
            dlc = int(dlc_s)

            row = [0] * 8
            try:
                for i, b in enumerate(rest.split()[: min(dlc, 8)]):
                    row[i] = int(b, 16)
            except ValueError:
                bad += 1
                continue

            ts.append(float(t))
            ids.append(int(cid_hex, 16))
            dlcs.append(dlc)
            data.append(row)
            labels.append(NORMAL)

    if bad:
        print(f"       skipped {bad} unparsed lines")

    return _pack(ts, ids, dlcs, data, labels)


def _pack(ts, ids, dlcs, data, labels):
    return {
        "ts": np.asarray(ts, dtype=np.float64),      # seconds, needs the precision
        "can_id": np.asarray(ids, dtype=np.int32),
        "dlc": np.asarray(dlcs, dtype=np.int8),
        "data": np.asarray(data, dtype=np.uint8),    # (n, 8), zero-filled
        "label": np.asarray(labels, dtype=np.int8),
    }


def parse_file(path):
    cls = attack_class_for(path)
    print(f"[parse] {path.name}")
    out = (parse_normal_txt(path) if cls is None
           else parse_attack_csv(path, cls))

    n = len(out["ts"])
    atk = int((out["label"] != NORMAL).sum())
    print(f"        {n:>9,} frames   {atk:>8,} injected ({100 * atk / max(n, 1):.2f}%)")

    # Timestamps must be non-decreasing for the timing features to mean anything.
    d = np.diff(out["ts"])
    if (d < 0).any():
        print(f"        WARNING: {int((d < 0).sum())} backwards timestamps")

    return out


def peek(n=3):
    """Print the first few raw lines of each file, before writing any parser logic."""
    for p in sorted(RAW.iterdir()):
        if p.suffix not in (".csv", ".txt"):
            continue
        print(f"\n=== {p.name}  ({p.stat().st_size / 1e6:.1f} MB) ===")
        with open(p, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                print("  " + line.rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true",
                    help="show raw lines instead of parsing")
    args = ap.parse_args()

    if args.peek:
        peek()
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(RAW.iterdir()) if p.suffix in (".csv", ".txt")]
    if not files:
        raise SystemExit("no data in data/raw/ -- run tools/fetch_dataset.py first")

    for p in files:
        out = parse_file(p)
        dest = PROCESSED / (p.stem + ".npz")
        np.savez_compressed(dest, **out)
        print(f"        -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
