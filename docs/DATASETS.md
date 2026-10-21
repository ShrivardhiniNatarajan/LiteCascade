# Datasets

The real datasets are **not** included (size + licensing) and none of the numbers in this repository
come from them. Download them yourself, place them as below, and pass `--dataset nbaiot|ciciot2023|iot23`
(optionally `--data-root <folder>`). Loaders: `src/litecascade/data/{nbaiot,ciciot2023,iot23}.py`.

All loaders return the same schema: numeric feature columns + `label_binary` (0 benign / 1 malicious),
`label_family`, `group` (split unit), `segment` (contiguous record stream; windows never cross it) and
`timestamp` (ordering). Cleaning (`data/clean.py`): ±inf → NaN, drop NaN rows, drop exact duplicate
records, drop constant feature columns.

## 1. N-BaIoT (Meidan et al., 2018)
* 9 devices, 115 features, benign + Mirai + BASHLITE (gafgyt). UCI: "detection_of_IoT_botnet_attacks_N_BaIoT"; also mirrored on Kaggle.
* Two layouts are recognised under `data/raw/nbaiot/`:
  ```text
  # UCI (extract the per-device .rar archives)
  Danmini_Doorbell/benign_traffic.csv
  Danmini_Doorbell/gafgyt_attacks/{combo,junk,scan,tcp,udp}.csv
  Danmini_Doorbell/mirai_attacks/{ack,scan,syn,udp,udpplain}.csv
  ...
  # Kaggle flat
  1.benign.csv  1.gafgyt.combo.csv  1.mirai.ack.csv  ...  9.mirai.udpplain.csv
  ```
* group = device, segment = file, order = file order (no timestamps in the CSVs).
* ~7M rows: use `--max-rows-per-file 20000` (first N rows of each file, keeps temporal contiguity) and/or
  `--max-train-windows` on a laptop. With 9 devices the default group split is 4/1/2/2 devices
  (train/val/calib/test); every split must contain malicious windows (≥ 598 malicious calibration
  windows for ε = 0.5 %, α = 0.05).

## 2. CICIoT2023 (Neto et al., 2023)
* 46 features + `label` (33 attacks + benign), UNB CIC website. Put any number of CSVs (`part-*.csv` or
  `Merged*.csv`) anywhere under `data/raw/ciciot2023/`.
* group = segment = file; `label_family` = one of 8 categories (benign, ddos, dos, mirai, recon, spoofing,
  web, bruteforce); `--task family` trains a multi-class model on them.
* The public CSV rows carry no timestamps and are not guaranteed to be temporally ordered, so windows with
  T > 1 may not contain real temporal structure: prefer `--window 1`. With few files use `--split-mode time`.

## 3. IoT-23 (Stratosphere Lab, 2020)
* Zeek `conn.log.labeled` files (the "lighter" release suffices):
  ```text
  data/raw/iot23/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled
  data/raw/iot23/CTU-Honeypot-Capture-4-1/bro/conn.log.labeled
  ...
  ```
* The last three fields (`tunnel_parents label detailed-label`) are space-separated in IoT-23; the parser
  handles that. Features: 9 numeric Zeek fields + one-hot `proto`, `conn_state`, `service` (fixed vocabularies).
* group = segment = capture, ordered by `ts`. Most captures are single-class; if a group split cannot give
  every split both classes use `--split-mode time` (chronological blocks per capture with a purge gap of T
  records). Some captures are multi-GB: use `--max-rows-per-file`.

## Synthetic fixture (what this repository actually runs on)
```bash
python scripts/make_fixture.py        # writes data/fixture/{nbaiot,ciciot2023,iot23}
```
Deterministic synthetic records written **in the native layouts above** so the real parsers are exercised
(`--dataset fixture` = N-BaIoT layout, 10 synthetic devices, 16,000 records; also `fixture-ciciot2023`,
`fixture-iot23`). It is generated from a latent AR(1) process with attack-specific mean shifts and an
alternating (temporal-only) "scan" pattern — see `src/litecascade/data/synthetic.py`. Results on it only show
that the pipeline works; they are not evidence of real-world detection performance.
