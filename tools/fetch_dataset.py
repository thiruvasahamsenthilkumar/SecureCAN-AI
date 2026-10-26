#!/usr/bin/env python3
"""Download and extract the HCRL Car-Hacking dataset into data/raw/.

The HCRL landing page (ocslab.hksecurity.net/Datasets/car-hacking-dataset) links to
a public Dropbox folder, so no account or registration is needed. Appending `dl=1`
to the folder URL makes Dropbox serve the whole folder as a single zip.

Usage:
    python tools/fetch_dataset.py          # download + extract, skip if present
    python tools/fetch_dataset.py --force  # re-download even if present
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

DROPBOX_URL = (
    "https://www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom/"
    "AF7JeRW893grZkigkulkAHk?rlkey=3h6zamu3kc262lrnipu5qden8&dl=1"
)
ZIP_PATH = RAW / "hcrl_car_hacking.zip"

# What we expect to find after extraction. Names vary slightly between mirrors, so
# the check is a substring match rather than an exact filename.
EXPECTED = ["DoS", "Fuzzy", "gear", "RPM", "normal"]


def download(force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        mb = ZIP_PATH.stat().st_size / 1e6
        print(f"[skip] {ZIP_PATH.name} already present ({mb:.0f} MB)")
        return

    print(f"[get ] {DROPBOX_URL[:60]}...")
    # Dropbox rejects the default urllib user-agent with a 403.
    req = urllib.request.Request(DROPBOX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(ZIP_PATH, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r       {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="")
            else:
                print(f"\r       {done / 1e6:6.0f} MB", end="")
    print()


def extract():
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        print(f"[zip ] {len(members)} files")
        for m in members:
            out = RAW / Path(m).name          # flatten any nested folders
            if out.exists():
                continue
            print(f"       -> {out.name}")
            with z.open(m) as src, open(out, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)


def verify():
    files = [p for p in RAW.iterdir() if p.suffix in (".csv", ".txt")]
    print(f"\n[ok  ] {len(files)} data files in data/raw/")
    for p in sorted(files):
        print(f"       {p.name:<32} {p.stat().st_size / 1e6:8.1f} MB")

    missing = [k for k in EXPECTED
               if not any(k.lower() in p.name.lower() for p in files)]
    if missing:
        print(f"\n[warn] no file matching: {', '.join(missing)}")
        print("       check data/raw/ by hand before running the parser")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    download(args.force)
    extract()
    sys.exit(verify())
