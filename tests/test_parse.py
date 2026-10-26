"""Tests for the HCRL log parser.

The parser's one genuinely tricky job is the variable-width rows: DLC says how many
data bytes follow, so the Flag column is not at a fixed index. Most of these tests
exist to pin that down, because getting it wrong corrupts the labels silently --
the code still runs, the model still trains, the accuracy is just quietly wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from securecan.parse import (  # noqa: E402
    DOS, FUZZY, GEAR, NORMAL, RPM,
    attack_class_for, parse_attack_csv, parse_normal_txt,
)


@pytest.fixture
def csv_file(tmp_path):
    """A CSV exercising full-width rows, short rows, and both flags."""
    p = tmp_path / "DoS_dataset.csv"
    p.write_text(
        # full 8-byte row, normal
        "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
        # short 5-byte row, normal -- flag sits three columns to the left
        "1478198376.389800,0153,5,00,11,22,33,44,R\n"
        # full-width injected row
        "1478198376.390000,0000,8,ff,ff,ff,ff,ff,ff,ff,ff,T\n"
        # DLC 2, injected
        "1478198376.390200,02b0,2,ab,cd,T\n"
    )
    return p


def test_short_rows_keep_correct_label(csv_file):
    """The whole point: a DLC-5 row must still be labelled from its last field."""
    out = parse_attack_csv(csv_file, DOS)
    assert out["label"].tolist() == [NORMAL, NORMAL, DOS, DOS]


def test_short_rows_zero_fill_payload(csv_file):
    out = parse_attack_csv(csv_file, DOS)
    # row 1 has DLC 5, so bytes 5..7 must be zero and not borrowed from elsewhere
    assert out["data"][1].tolist() == [0x00, 0x11, 0x22, 0x33, 0x44, 0, 0, 0]
    assert out["data"][3].tolist() == [0xAB, 0xCD, 0, 0, 0, 0, 0, 0]


def test_hex_fields_decoded(csv_file):
    out = parse_attack_csv(csv_file, DOS)
    assert out["can_id"].tolist() == [0x316, 0x153, 0x000, 0x2B0]
    assert out["data"][0].tolist() == [0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6F]


def test_dlc_preserved(csv_file):
    out = parse_attack_csv(csv_file, DOS)
    assert out["dlc"].tolist() == [8, 5, 8, 2]


def test_timestamps_keep_microsecond_precision(csv_file):
    """float32 would round these away; the timing features depend on them.

    These are Unix timestamps around 1.478e9, where float64 spacing is ~2.4e-7 s.
    So microsecond resolution survives but only just -- which is exactly why the
    parser must not downcast. float32 spacing up there is ~128 s, which would
    flatten every inter-frame gap to zero.
    """
    out = parse_attack_csv(csv_file, DOS)
    assert out["ts"].dtype == np.float64
    assert out["ts"][1] - out["ts"][0] == pytest.approx(0.000373, abs=1e-6)


def test_malformed_rows_skipped(tmp_path):
    p = tmp_path / "DoS_dataset.csv"
    p.write_text(
        "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
        "garbage line with no commas\n"
        "1478198376.3,ZZZZ,8,00,00,00,00,00,00,00,00,R\n"   # bad hex ID
        "1478198376.4,0316,8,05,21,68,09,21,21,00,6f,X\n"   # unknown flag
        "1478198376.5,0316,8,05,21,68,09,21,21,00,6f,T\n"
    )
    out = parse_attack_csv(p, DOS)
    assert len(out["ts"]) == 2
    assert out["label"].tolist() == [NORMAL, DOS]


def test_attack_class_from_filename():
    assert attack_class_for(Path("DoS_dataset.csv")) == DOS
    assert attack_class_for(Path("Fuzzy_dataset.csv")) == FUZZY
    assert attack_class_for(Path("gear_dataset.csv")) == GEAR
    assert attack_class_for(Path("RPM_dataset.csv")) == RPM
    # the attack-free capture has no injected class
    assert attack_class_for(Path("normal_run_data.txt")) is None


def test_attack_class_is_case_insensitive():
    """Mirrors rename these files; matching must not be brittle."""
    assert attack_class_for(Path("dos_DATASET.csv")) == DOS
    assert attack_class_for(Path("RPM_spoof.CSV")) == RPM


def test_normal_txt_format(tmp_path):
    """The attack-free file is candump-style text, not CSV, and has no flag."""
    p = tmp_path / "normal_run_data.txt"
    p.write_text(
        "Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    "
        "05 28 84 66 6d 00 00 a2\n"
        "Timestamp: 1479121434.850423        ID: 02c0    000    DLC: 4    "
        "14 00 00 01\n"
    )
    out = parse_normal_txt(p)

    assert out["can_id"].tolist() == [0x350, 0x2C0]
    assert out["dlc"].tolist() == [8, 4]
    assert out["data"][0].tolist() == [0x05, 0x28, 0x84, 0x66, 0x6D, 0x00, 0x00, 0xA2]
    assert out["data"][1].tolist() == [0x14, 0x00, 0x00, 0x01, 0, 0, 0, 0]
    # every frame in this file is normal by definition
    assert (out["label"] == NORMAL).all()


def test_output_dtypes_are_compact(csv_file):
    """These arrays are millions of rows; dtypes are not cosmetic."""
    out = parse_attack_csv(csv_file, DOS)
    assert out["can_id"].dtype == np.int32
    assert out["data"].dtype == np.uint8
    assert out["label"].dtype == np.int8
    assert out["data"].shape == (4, 8)
