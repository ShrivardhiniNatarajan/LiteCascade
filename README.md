# LiteCascade: Lightweight Early-Exit Cascade for IoT Malware / Intrusion Detection

LiteCascade is a two-exit neural cascade for resource-constrained IoT devices: a tiny **Sentinel** classifies
every traffic window and a calibrated, risk-controlled **gate** lets confident windows leave early; only
uncertain windows run the larger **Analyst**. Both exits share a PCA-initialised projection (PIP) and a
depthwise-separable stem, are trained jointly with knowledge distillation from a CNN-BiLSTM teacher, and are
compressed with structured pruning and INT8 quantization-aware training followed by gate re-calibration.

This repository contains the full, runnable pipeline (CPU only): dataset loaders, leakage-safe preprocessing,
models, training, calibration, compression, evaluation, profiling, a CLI predictor, and a web dashboard.

> **Honesty note.** The real datasets (N-BaIoT, CICIoT2023, IoT-23) are not in this repository and **no result
> here was computed on them.** Every number in `reports/` comes from a small **synthetic** fixture and only shows
> that the pipeline works end-to-end. Latencies are wall-clock on the development PC, not on IoT hardware.

---

## 1. Setup (Windows / Linux / macOS, Python 3.10+, CPU only)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # CPU torch from PyPI is fine
pip install -e . --no-deps        # makes `litecascade` importable from scripts/
```

Tested with Python 3.11.4, torch 2.14.0+cpu, scikit-learn 1.9.1, pandas 2.2.3, FastAPI 0.111.1 on Windows 11.
INT8 uses PyTorch's eager quantization (`torch.ao.quantization`, deprecated upstream but present in this torch);
the backend is picked automatically from `x86 / fbgemm / onednn / qnnpack` (this machine only has `onednn`).

## 2. Quick start (synthetic fixture, ~15-25 min on a laptop CPU)

```bash
python scripts/run_demo.py --ablations
python -m litecascade.train.run --model dl --dataset fixture --epochs 8 --bundles artifacts
cd dashboard && npm ci && npm run build && cd ..
python -m litecascade.web.app
```

1. `run_demo.py` writes the synthetic fixture to `data/fixture/`, trains/profiles the teacher and baselines, trains
   LiteCascade-R and -T through the full chain (joint KD training → calibration → 30 % structured pruning +
   fine-tuning → INT8 QAT → re-calibration), runs the ablations and writes the report.
2. `--model dl --bundles artifacts` trains every deep-learning model (teacher, B1, B3-B11) and saves each one as a
   bundle so the dashboard can run them side by side.
3. The React dashboard is built into `src/litecascade/web/dashboard_dist/` (Node 18+; only needed once).
4. Open <http://127.0.0.1:8000>.

| Output | What |
|---|---|
| `artifacts/fixture_R_int8/`, `fixture_T_int8/`, `fixture_<model>/` | deployable bundles (weights, train-fitted scaler/PCA, calibrated gate, metrics) |
| `reports/fixture.md`, `reports/fixture.json` | comparison table, compression-stage table, gate details, ablations |
| `docs/PROJECT_OVERVIEW.md` | plain-language account of everything that was built and verified |
| `results/<run_id>/` | per-run `report.json` / `metrics.json` + checkpoints |
| `examples/fixture_sample_labeled.csv`, `..._unlabeled.csv` | raw records of unseen test devices for the dashboard |

`python scripts/run_demo.py --quick` is a ~3 min smoke version (fewer epochs, no ablations).

## 3. Dashboard ("LiteCascade Control Room")

```bash
python -m litecascade.web.app            # serves the React app + API on http://127.0.0.1:8000
cd dashboard && npm run dev              # optional: hot-reload frontend on :5173 (proxies /api to :8000)
```

| Page | What it does |
|---|---|
| **Live cascade** | Replays held-out test windows in real time. Animated packets travel PIP → stem → Sentinel → gate and either exit early or run the Analyst; live exit rate, compute saved, alerts, F1 and latency; score stream; event feed; one-click injection of real held-out attack bursts |
| **Model arena** | All 14 models get the same window at the same moment: verdict LEDs, running F1 rings, latency sparklines, confusion counts, animated leaderboard |
| **Benchmarks** | Stored test metrics: animated Pareto chart (F1 vs. MACs / params / size / latency), size and latency bars, sortable table |
| **Gate lab** | ε / α slider: τb, exit rate, exit-induced misses and F1 recomputed from the model's real calibration outputs |
| **Architecture** | Parameter budget per block, MAC split, the implemented equations, the model zoo |
| **Analyze CSV** | Upload traffic, compare any set of models, download every model's predictions |
| **Documentation** | `docs/*.md` and `reports/*.md` rendered in the app |

Every prediction is produced by the model itself; the replay's ground-truth labels are used only to score
predictions. Models trained on the fixture are flagged "synthetic". CSV format for uploads: one row per record with
the model's feature columns (the 115 N-BaIoT names for fixture models); optional `segment`/`group`, `timestamp`,
and `label` / `label_binary` (`benign*`/`0` = benign).

## 4. Command reference

```bash
# synthetic fixture (N-BaIoT / CICIoT2023 / IoT-23 native layouts)
python scripts/make_fixture.py

# one baseline / all models / all deep-learning models as dashboard bundles
python -m litecascade.train.run --model b6 --dataset fixture --epochs 8
python -m litecascade.train.run --model all --dataset fixture --epochs 8
python -m litecascade.train.run --model dl --dataset fixture --epochs 8 --bundles artifacts

# full LiteCascade chain -> results/<run>/report.json and a bundle
python -m litecascade.train.run_cascade --dataset fixture --variant R --epochs 15 --prune 0.3 --quantize qat --bundle my_model

# ablation switches (paper Table IV subset)
python -m litecascade.train.run_cascade --dataset fixture --no-kd --no-self-kd --no-pip-init --no-temp-scaling
python -m litecascade.train.run_cascade --dataset fixture --naive-gate | --verify-alerts | --conservative-gate | --variant T
python scripts/run_ablations.py --dataset fixture --epochs 15

# report from everything in results/
python -m litecascade.eval.report --dataset fixture --ablations results/ablations_fixture.json --out reports/fixture

# batch prediction without the UI
python -m litecascade.deploy.predict --bundle artifacts/fixture_R_int8 --csv examples/fixture_sample_labeled.csv --out preds.csv

# re-calibrate a bundle's gate with another epsilon / alpha
python scripts/calibrate_gate.py --bundle artifacts/fixture_R_int8 --epsilon 0.01 --write

# model size / MAC breakdown vs. the paper's design estimates
python scripts/print_model_summary.py --variant R --F 115 --K 2

# tests and lint
python -m pytest tests -q
python -m ruff check src tests scripts
```

### Real datasets

Place the data as described in [docs/DATASETS.md](docs/DATASETS.md), then e.g.

```bash
python -m litecascade.train.run_cascade --dataset nbaiot --max-rows-per-file 20000 --variant R --prune 0.3 --quantize qat --bundle nbaiot_R_int8
python -m litecascade.train.run --model all --dataset nbaiot --max-rows-per-file 20000
python -m litecascade.train.run_cascade --dataset ciciot2023 --window 1 --split-mode time --task family
python -m litecascade.train.run_cascade --dataset iot23 --split-mode time --max-rows-per-file 50000
```

These commands are implemented and the parsers are tested on synthetic files in the exact native layouts, but they
have **not** been run on the real data here.

## 5. What is implemented (and how it maps to the paper)

| Component | Where | Notes |
|---|---|---|
| Loaders N-BaIoT (UCI + Kaggle layouts), CICIoT2023, IoT-23 (Zeek) | `data/nbaiot.py`, `data/ciciot2023.py`, `data/iot23.py` | common schema, cleaning in `data/clean.py` |
| Leakage-safe splits | `data/splits.py` | whole groups per split **with a disjoint calibration split**; or chronological blocks with a T-record purge gap; LOFO split helper |
| Windows | `data/windows.py` | T consecutive records, never crossing a segment, labelled by the last record (causal) |
| Preprocessing | `data/preprocess.py`, `data/pipeline.py` | RobustScaler + PCA fitted on **train records only** |
| PIP layer (eq. 1) | `models/litecascade.py` | initialised with W = PCA components, b = −Wμ ⇒ z = W(x−μ); frozen for 3 warm-up epochs, then fine-tuned |
| Stem, Sentinel (eq. 2), Analyst, R/T variants | `models/litecascade.py` | d=32→c=24 DS stem; Sentinel DS block + GAP + linear; Analyst 2 residual inverted-bottleneck DS-TCN blocks (dil. 1, 2, exp. 2) + BiGRU(24) [R only] + attention pooling |
| Teacher CNN-BiLSTM | `models/teacher.py` | 2×Conv1d + 2-layer BiLSTM(128) + attention |
| Joint loss (eq. 5) | `train/loss.py` | CE(p1)+CE(p2)+γ·τ²KL(σ(t/τ)‖σ(s2/τ))+δ·KL(p2‖p1); self-KD gradient flows into the Sentinel only |
| Temperature scaling | `eval/gate.py` | T1, T2 fitted by NLL on the calibration split (log-parametrised, T>0 ⇒ argmax preserved) |
| Risk-controlled gate (eqs. 3-4) | `eval/gate.py` | τb = min{τ : U_CP(m(τ), N_m; α) ≤ ε}, m(τ) = malicious windows exiting benign **that the Analyst would have flagged**; exact, tie-safe float32 thresholds; infeasible ⇒ benign exit disabled. τm = empirical target-FPR threshold (or `--verify-alerts`) |
| Genuine cascade inference | `LiteCascade.cascade_predict` | Analyst executed only on the windows the gate did not release (tested with a forward hook) |
| Structured pruning | `compress/prune.py` | L1-ranked channel removal (Analyst expansion channels, Sentinel channels), layers physically rebuilt; then fine-tuning |
| INT8 | `compress/quantize.py` | eager QAT of PIP+stem+conv blocks (Conv+BN fused) → INT8, dynamic INT8 for Linear/GRU; or dynamic-only; **gate re-calibrated on the INT8 model** |
| Baselines B1-B11 | `models/baselines.py`, `train/run.py` | same data path; B5 = PCA inputs + dynamic INT8; new deep models B9 BiGRU, B10 TCN, B11 ResNet-1D |
| Metrics | `eval/metrics.py` | accuracy, precision, malicious recall, macro-F1, FPR, MCC, bootstrap 95 % CI |
| Efficiency | `eval/profiler.py` | params, MACs (fvcore + custom LSTM/GRU/attention handles), serialized size, host-CPU latency, E[C] and S (eq. 6), measured cascade vs. always-full latency |
| Bundles, predictor, live replay | `deploy/bundle.py`, `deploy/predict.py`, `web/live.py` | one `predict_windows` API for cascade and baseline bundles; replay engine; gate lab |
| Dashboard | `web/app.py`, `dashboard/` | FastAPI + Server-Sent Events backend; React 18 + TypeScript + framer-motion frontend |

Verified against the paper's design estimates (F = 115, T = 10, d = 32, c = 24, K = 2):
Sentinel exit-only 746 params (< 2 K), Analyst exit 12,722 (≈ 14 K), total 18,092 (≈ 20 K);
0.053 M / 0.176 M MACs for the Sentinel / full path (paper: ≈ 0.06 M / 0.18 M). The progress-notes figure of 3,308
Sentinel parameters came from the earlier d = 16 configuration *including* PIP + stem; with d = 32 the Sentinel
path including PIP + stem is 5,370.

## 6. Results on the synthetic fixture

Test split = 2 synthetic devices never seen in training (3,110 windows). Full tables: [reports/fixture.md](reports/fixture.md);
explanation: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).

| Model | macro-F1 | Recall | Params | Size KB |
|---|---|---|---|---|
| Teacher CNN-BiLSTM | 0.936 | 0.891 | 739,843 | 2,898 |
| Best baseline on this fixture: B1 L1-MLP | 0.948 | 0.914 | 75,810 | 299 |
| B10 TCN | 0.941 | 0.899 | 22,402 | 93 |
| **LiteCascade-R INT8** | **0.937** | **0.897** | **16,356** | **50.3** |

LiteCascade-R INT8 lets 86.4 % of test windows exit at the Sentinel with a measured exit-induced miss rate of 0 %
(bound ε = 0.5 %, α = 0.05), i.e. 2.76× faster per window than always running both exits on this PC. On this easy
synthetic data several small baselines are as accurate, so it does not show an accuracy advantage.

## 7. Limitations / not implemented

- **No real-dataset results.** Loaders/commands exist, but nothing was run on N-BaIoT, CICIoT2023 or IoT-23.
- **No on-device measurements** (Raspberry Pi / ESP32 / Cortex-M), no energy measurement; latencies are host-CPU PyTorch eager.
- **No TFLite / TFLite-Micro export** (TensorFlow was removed from the dependencies because it was unused); INT8
  models are PyTorch quantized modules. Sizes are serialized PyTorch state_dicts (include pickle overhead).
- The Malimg byteplot adapter (Sec. IV.I), the "wrong order (QAT→prune)" ablation, LOFO zero-day and perturbation
  robustness experiments (Table VI) are not implemented as experiments (a LOFO split helper exists).
- τm has no statistical guarantee (empirical FPR target). The τb guarantee assumes exchangeable calibration/test
  windows; it bounds the *extra* misses caused by early exit relative to the Analyst, not the total miss rate.
- Structured pruning leaves the shared stem/residual width c untouched.
- CICIoT2023 CSV rows have no timestamps, so windows with T > 1 may not reflect real temporal order there.
- Bundles are pickles (torch / joblib): only load bundles you created.

See [docs/PROGRESS.md](docs/PROGRESS.md) for the stage log, [docs/DATASETS.md](docs/DATASETS.md) and
[docs/BASELINES.md](docs/BASELINES.md) for details.
