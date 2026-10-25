# HCRL Dataset - Download Instructions

## Quick Download (Recommended)

The HCRL Car-Hacking dataset is available on **Kaggle**. Follow these steps:

### 1. Get Kaggle API Credentials
- Go to https://www.kaggle.com/settings/account
- Click "Create New API Token"
- This downloads `kaggle.json` — save it to `~/.kaggle/kaggle.json`
- On Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
- On Mac/Linux: `~/.kaggle/kaggle.json`

### 2. Install Kaggle CLI
```bash
pip install kaggle
```

### 3. Download the Dataset
```bash
# Navigate to the data directory
cd data/

# Download from Kaggle (replace with actual dataset name)
kaggle datasets download -d "dataset-name-here"

# Unzip
unzip dataset-name.zip
```

## Alternative: Manual Download

If Kaggle download doesn't work:
1. Visit the HCRL Lab GitHub or Kaggle page
2. Download the CSV files manually
3. Place them in this `data/` directory
4. Keep filenames consistent: `CAN_data_normal.csv`, `CAN_data_attack_*.csv`

## Dataset Files (After Download)

You should have these files in this directory:
```
data/
├── CAN_data_normal.csv           (~1M rows, normal vehicle traffic)
├── CAN_data_attack_DoS.csv       (DoS attack: message flooding)
├── CAN_data_attack_Fuzzy.csv     (Fuzzy attack: malformed messages)
├── CAN_data_attack_RPM.csv       (Fake RPM: spoofed engine signal)
├── CAN_data_attack_Gear.csv      (Fake Gear: spoofed transmission signal)
└── feature_names.txt             (generated during feature engineering)
```

## Verify Download

After downloading, verify with:
```bash
ls -lah *.csv
wc -l *.csv  # Count rows
```

Each file should have thousands to millions of rows.

## If Dataset is Unavailable

If the HCRL dataset is temporarily unavailable, we can:
1. **Generate synthetic data** for prototyping (same feature structure)
2. **Use a different public CAN dataset** (NSL-KDD has some CAN variants)
3. **Manually create test scenarios** with realistic CAN message patterns

The important part is the **feature engineering and model training logic** — we can always swap in the real data later.
