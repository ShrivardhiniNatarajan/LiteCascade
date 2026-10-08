# Dataset Acquisition Guide

This document provides instructions on how to manually download and organize the three target IoT intrusion/malware datasets for LiteCascade.

## 1. N-BaIoT (UCI)
* **Description:** Network traffic from 9 real IoT devices infected by Mirai and BASHLITE.
* **URL:** [UCI Machine Learning Repository - N-BaIoT](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)
* **License:** Open Access (Attribution).
* **Expected Size:** ~7 GB (CSV files).
* **Download Steps:**
  1. Download all device zip files (e.g., `Danmini_Doorbell.zip`, `Provision_PT_737E_Security_Camera.zip`, etc.).
  2. Extract each zip file into its own folder inside `data/raw/nbaiot/`.
* **Expected Structure:**
  ```text
  data/raw/nbaiot/
  ├── Danmini_Doorbell/
  │   ├── benign.csv
  │   ├── gafgyt_combo.csv
  │   └── ...
  ├── Provision_PT_737E/
  └── ...
  ```

## 2. CICIoT2023 (UNB)
* **Description:** IoT dataset with 33 attacks performed on 105 IoT devices.
* **URL:** [UNB CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
* **License:** Open Access for research (Citation required).
* **Expected Size:** ~20-30 GB.
* **Download Steps:**
  1. Register/request access at the UNB website.
  2. Download the PCAP or CSV versions (CSV preferred).
  3. Place the CSV files directly in `data/raw/ciciot2023/`.
* **Expected Structure:**
  ```text
  data/raw/ciciot2023/
  ├── part-00000-xxxx.csv
  ├── part-00001-xxxx.csv
  └── ...
  ```

## 3. IoT-23 (Stratosphere)
* **Description:** Network traffic from IoT devices (3 benign captures, 20 malware captures).
* **URL:** [Stratosphere Laboratory - IoT-23](https://www.stratosphereips.org/datasets-iot23)
* **License:** CC BY 4.0.
* **Expected Size:** ~21 GB (Zeek logs).
* **Download Steps:**
  1. Download the `conn.log.labeled` files for each scenario.
  2. Place them in their respective scenario folders.
* **Expected Structure:**
  ```text
  data/raw/iot23/
  ├── opt/Malware-Project/MacCDC/
  │   └── conn.log.labeled
  ├── opt/Malware-Project/IoTScenarios/
  │   └── conn.log.labeled
  └── ...
  ```

## Synthetic Data
If you don't have access to these datasets, you can generate a synthetic fixture for testing:
```bash
python scripts/make_fixture.py
```
This generates `data/processed/synthetic_{dataset}.csv` containing ~2k rows with the required schema.
