# HCRL Car-Hacking Dataset

## Overview
This project uses the **Korea University HCRL Car-Hacking Dataset**, one of the most widely cited datasets in automotive cybersecurity research.

**Dataset Name:** HCRL_CAN_Dataset  
**Source:** [Korea University HCRL Lab](https://sites.google.com/site/carhostedrldataset/)  
**Size:** ~2–3 million CAN messages  
**Scenarios:** 10 attack and normal operation scenarios  

## What's in the Dataset?

### CAN Bus Overview
A **CAN (Controller Area Network) bus** is a vehicle's internal communication network:
- Dozens of Electronic Control Units (ECUs) connected via CAN bus
- Examples: Engine Control Module (ECM), Transmission Control Module (TCM), Anti-lock Braking System (ABS)
- **Problem:** CAN bus has **NO built-in security** — any device can send any message
- **Real-world impact:** Jeep Cherokee hack (2015) — hackers took remote control by exploiting CAN access

### CAN Message Format
Each CAN message contains:
- **CAN_ID (11-bit):** Identifier (e.g., 0x123 = ECM message)
- **DLC:** Data Length Code (0–8 bytes)
- **Data:** 8-byte payload (content depends on message type)
- **Timestamp:** When the message was sent

### Attack Types in HCRL Dataset

| Attack Type | Description | How to Detect |
|------------|-------------|---------------|
| **DoS (Denial of Service)** | Flood bus with junk messages every 0.3 ms | Abnormally high message rate, specific ID patterns |
| **Fuzzy** | Send random, malformed messages | Unusual byte value distributions, out-of-range values |
| **Fake RPM** | Spoof engine RPM signal (tachometer) | RPM values don't match other signals, temporal anomalies |
| **Fake Gear** | Spoof transmission gear signal | Gear change inconsistent with vehicle state |

## Dataset Format (CSV)

Each row represents one CAN message:

```
Timestamp,CAN_ID,DLC,Data_Byte_1,Data_Byte_2,...,Data_Byte_8,Label
1349270610.701,0x001,8,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,R
1349270610.703,0x002,8,0x00,0x00,0xFF,0xFF,0x00,0x00,0x00,0x00,R
...
1349270610.801,0x123,8,0xAB,0xCD,0x00,0xFF,0x00,0x00,0x00,0x00,A
```

**Columns:**
- `Timestamp`: Unix timestamp (seconds since 1970-01-01)
- `CAN_ID`: Message identifier in hex (0x000 – 0x7FF)
- `DLC`: Data length (0–8)
- `Data_Byte_1` through `Data_Byte_8`: Payload bytes in hex
- `Label`: `R` (normal/benign) or `A` (attack/malicious)

## Download Instructions

### Option 1: Kaggle
The dataset is available on Kaggle:
```bash
# Install Kaggle CLI
pip install kaggle

# Authenticate (set up API key in ~/.kaggle/kaggle.json)
kaggle datasets download -d "dataset_name_here"

# Unzip
unzip dataset_name.zip -d data/
```

### Option 2: Direct Download from HCRL Lab
Visit [Korea University HCRL Lab](https://sites.google.com/site/carhostedrldataset/) and download the CSV files manually.

### Option 3: Use Sample Data for Testing
For quick prototyping, we can generate synthetic CAN data or use a small sample. The real dataset will be downloaded separately.

## Data Characteristics

### Class Distribution
- **Normal traffic:** ~85%
- **Attack messages:** ~15% (distributed across DoS, Fuzzy, Fake RPM, Fake Gear)
- **Imbalance:** Minority classes (especially Fake RPM/Gear) will need special handling in training

### Message Rate
- **Normal operation:** ~100–300 messages/second
- **DoS attack:** ~3,000+ messages/second (obvious anomaly)
- **Other attacks:** ~100–300 messages/second (harder to detect — requires feature analysis)

### Key Insights
1. **DoS is easy to detect:** Abnormally high message rate is the primary signal
2. **Fuzzy/Fake* attacks are harder:** Require statistical analysis of payload content
3. **Temporal patterns matter:** Time between consecutive messages from same ID
4. **Domain knowledge helps:** Understanding what RPM/Gear values are realistic

## Files in This Project

Once downloaded, the raw dataset will be stored here:
```
data/
├── README.md                           (this file)
├── CAN_data_normal.csv                (normal vehicle operation, ~1M messages)
├── CAN_data_attack_DoS.csv            (DoS attack scenario)
├── CAN_data_attack_Fuzzy.csv          (Fuzzy attack scenario)
├── CAN_data_attack_RPM.csv            (Fake RPM scenario)
├── CAN_data_attack_Gear.csv           (Fake Gear scenario)
└── features_train.csv / features_val.csv / features_test.csv (after preprocessing)
```

## References

**Papers using this dataset:**
- "Anomaly Detection in CAN Bus Networks" — HCRL Lab
- "In-Vehicle Network Intrusion Detection" — IEEE/ACM conferences
- Multiple other published works (shows dataset is credible)

**Related resources:**
- [CAN Protocol Specification](https://en.wikipedia.org/wiki/CAN_bus)
- [Jeep Cherokee Hack Details](https://www.wired.com/2015/07/hackers-remotely-kill-jeep-highway/)
- [NHTSA Vehicle Safety Recalls](https://www.nhtsa.gov/)
