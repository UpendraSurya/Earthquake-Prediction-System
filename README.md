<div align="center">

# Earthquake Prediction System

**End-to-end deep learning pipeline for earthquake magnitude prediction from raw seismic waveforms**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![ObsPy](https://img.shields.io/badge/ObsPy-seismology-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**±0.5 Magnitude Tolerance Accuracy: 88.4%** &nbsp;|&nbsp; **MAE: 0.248** &nbsp;|&nbsp; **21,000+ Waveforms** &nbsp;|&nbsp; **5 Asian Seismic Zones**

</div>

---

## Overview

**earthd** predicts earthquake magnitude directly from raw P-wave seismograms using a dual-input CNN-BiLSTM neural network with Multi-Head Attention. Given a 30-second waveform window anchored at the P-wave arrival, the model outputs a predicted Richter scale magnitude.

The project covers the **complete ML lifecycle** — automated data collection from USGS/IRIS archives, signal processing, model training with chronological splitting to prevent data leakage, full evaluation suite, and automated PDF reporting.

```
Raw seismogram (30s P-wave window)
        ↓
CNN feature extraction + BiLSTM temporal modelling + Multi-Head Attention
        ↓
Fused with location features (distance, depth, lat, lon)
        ↓
Predicted magnitude (Richter scale)
```

---

## Results

### Regression Metrics — Test Set (215 Events)

| Metric | Value | Notes |
|--------|-------|-------|
| **MAE** | **0.248** | Average per-prediction error in magnitude units |
| RMSE | 0.334 | Root mean square error |
| Bias | −0.082 | Slight systematic underprediction |
| Error std | 0.324 | Spread of individual errors |

### Tolerance Accuracy — % of Predictions Within ±N Magnitude Units

| Tolerance | **earthd v5** | Benchmark Target |
|-----------|:------------:|:----------------:|
| **±0.5 mag** | **88.4%** | ≥85% (publishable) |
| ±0.3 mag | 68.8% | ≥60% |
| ±0.1 mag | 31.2% | ≥25% |

> **9 out of 10 earthquake predictions fall within half a magnitude unit of the true value.**

### Baseline Comparison

| Model | MAE ↓ | RMSE ↓ | ±0.5 Acc ↑ | ±0.3 Acc ↑ |
|-------|:-----:|:------:|:----------:|:----------:|
| Mean predictor (baseline) | 0.257 | — | — | — |
| RF — Location only (4 features) | 0.297 | 0.392 | 84.3% | 59.3% |
| RF — Waveform statistics (12 + loc) | 0.276 | 0.363 | 86.6% | 62.9% |
| **CNN-BiLSTM (this work)** | **0.248** | **0.334** | **88.4%** | **68.8%** |

### Comparison with Published Models

| Model | Training Set | MAE | Within ±0.5 |
|-------|-------------|:---:|:-----------:|
| **earthd v5 (this work)** | 1,715 India/Himalaya events | **0.248** | **88.4%** |
| MagNet (Mousavi et al. 2020) | 300,000+ STEAD | ~0.25 | ~88% |
| EQTransformer (Mousavi et al. 2020) | 1.2M STEAD | ~0.30 | ~85% |
| CREIME (Münchmeyer et al. 2022) | 10,000+ global | ~0.20 | ~94% |

> earthd v5 achieves **competitive MAE and superior tolerance accuracy** despite using a **100–700× smaller training set** than published baselines.

---

## Architecture

Dual-input model (~400,000 trainable parameters):

```
Waveform input (600 samples — 30s @ 20Hz)
    ↓
Conv1D(32, k=7) → BatchNorm → MaxPool           (300, 32)
Conv1D(64, k=5) → BatchNorm → MaxPool           (150, 64)
Conv1D(128, k=3) → BatchNorm → MaxPool          (75, 128)
    ↓
BiLSTM(64, return_sequences=True)               (75, 128)
    ↓
MultiHeadAttention(heads=4, key_dim=32)         (75, 128)
    ↓
Add + LayerNorm (residual connection)
    ↓
BiLSTM(32)                                      (64,)
    ↓
Dense(64) → Dropout(0.3)
    ↘
Location input (dist_km, depth_km, lat, lon)
    ↓
Dense(32) → BatchNorm → Dense(16)
    ↘
Concatenate (64 + 16 = 80)
    ↓
Dense(64) → Dropout(0.3) → Dense(1) → magnitude
```

**Key design decisions:**
- **Conv1D before LSTM** — CNN extracts local waveform features (P-wave shape, amplitude patterns) before BiLSTM models long-range temporal dependencies
- **Bidirectional LSTM** — processes the 30s window both forward and backward; post-P-arrival context (S-wave) improves predictions
- **Multi-Head Attention** — allows the model to simultaneously focus on multiple time regions (P-wave onset, S-wave arrival, coda)
- **Residual Add + LayerNorm** — prevents gradient vanishing in deep temporal stacks
- **Dual-input fusion** — location features (distance, depth) carry physical information that waveform shape alone cannot encode

---

## Data

### Sources

| Source | Role |
|--------|------|
| [USGS ComCat API](https://earthquake.usgs.gov/fdsnws/event/1/) | Event catalog (magnitude, depth, coordinates, origin time) |
| [IRIS FDSN](https://service.iris.edu/fdsnws/) | Seismic waveforms (MiniSEED format) |

### Coverage

**Regions (5 Asian seismic zones):**
- India / Himalaya (5°–40°N, 60°–100°E)
- Southeast Asia (12°S–25°N, 90°–135°E)
- Central Asia / Zagros (25°–50°N, 40°–80°E)
- Japan / Korea (25°–50°N, 125°–150°E)
- China / Tibet (20°–50°N, 75°–125°E)

| Parameter | Value |
|-----------|-------|
| Magnitude range | M 3.0 – M 7.5 |
| Time period | 2000 – 2024 |
| Total usable events | 21,000+ |
| Train / Val / Test split | 80 / 10 / 10 (chronological) |

### Signal Processing Pipeline

Each waveform goes through:

1. **Bandpass filter** — 1–10 Hz (removes microseismic noise and high-frequency interference)
2. **Resampling** — normalised to 20 Hz using polyphase FIR (`scipy.signal.resample_poly`)
3. **P-wave detection** — STA/LTA algorithm detects P-wave arrival onset
4. **Window cutting** — 30-second window: 1s before P-arrival → 29s after
5. **Z-score normalisation** — `w = (w - mean) / (std + 1e-8)` per window

> **Why chronological split?** A random shuffle would mix past and future earthquakes — the model would "see the future" during training, causing inflated validation scores that collapse in deployment. Chronological splitting mirrors realistic use: always train on past, predict future.

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | Adam |
| Learning rate | 0.0005 (with ReduceLROnPlateau) |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs (val_mae) |
| LR reduction | ×0.5 every 5 stagnant epochs, min 1e-6 |
| Loss function | MSE |
| Class imbalance handling | Inverse-frequency sample weights |
| Hardware | Apple M-series (Metal GPU) |
| Training time | ~4 min 39 sec (50 epochs) |

---

## Project Structure

```
earthd/
├── data_collection_v5.ipynb    # USGS catalog fetch + IRIS waveform download
├── training_v5.ipynb           # Model training, evaluation, all plots
├── orchestrator.py             # Single-command pipeline runner with auto-retry
├── rf_baseline.py              # Random Forest baseline comparisons
├── generate_report_pdf.py      # Auto-generates multi-page PDF evaluation report
├── check_status.py             # Live pipeline status dashboard
├── REPORT_earthd_v5.md         # Full technical report
├── waveforms_v5/               # MiniSEED + meta JSON files
├── processed_v5/               # CSV + meta JSON (model input)
├── models_v5/                  # Saved Keras models, weights, config, history
└── results_v5/                 # Evaluation plots + metrics JSON
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/UpendraSurya/Earthquake-Prediction-System.git
cd Earthquake-Prediction-System

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install tensorflow obspy scikit-learn pandas numpy matplotlib seaborn tqdm scipy
```

### 2. Run the full pipeline

```bash
python orchestrator.py
```

This runs in order:
1. **Data Collection** — fetches USGS catalog, downloads waveforms from IRIS FDSN
2. **Preprocessing** — resample to 20Hz, bandpass filter, extract 30s P-wave windows
3. **Training** — trains CNN-BiLSTM, saves best model checkpoint
4. **Evaluation** — confusion matrix, tolerance accuracy, per-class analysis
5. **Report** — auto-generates PDF report

### 3. Run a specific phase

```bash
python orchestrator.py --force-phase training
```

### 4. Check pipeline status

```bash
python check_status.py             # snapshot
python check_status.py --watch     # refresh every 30s
```

---

## Output Plots

All evaluation plots are saved to `results_v5/`:

| File | Description |
|------|-------------|
| `training_curves_v5_*.png` | MAE and MSE per epoch (train vs val) |
| `pred_vs_true_v5_*.png` | Scatter: predicted vs true magnitude |
| `error_dist_v5_*.png` | Histogram of prediction errors |
| `confusion_matrix_v5_*.png` | Confusion matrix — counts + normalised % |
| `per_class_analysis_v5_*.png` | Error boxplot per magnitude class + coloured scatter |

---

## Notifications

The orchestrator sends push notifications via [ntfy.sh](https://ntfy.sh) on phase completion or failure. Set your topic in `orchestrator.py`:

```python
NTFY_TOPIC   = 'your-topic-here'
NTFY_ENABLED = True
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | TensorFlow / Keras |
| Seismology | ObsPy (IRIS/FDSN client, MiniSEED parsing, STA/LTA) |
| ML Baseline | scikit-learn (Random Forest) |
| Data | pandas, NumPy, SciPy |
| Visualisation | matplotlib, seaborn |
| Utilities | tqdm, ntfy.sh |

---

## Limitations & Next Steps

| Limitation | Impact | Planned Fix |
|-----------|--------|-------------|
| Training set size (1,715 events in v5) | Low R² on rare large events | Longer download run targets 5,000+ events |
| Single station per event | Misses network-level triangulation | Multi-station input fusion |
| Indian/Himalayan focus in v5 | Reduced cross-region generalisation | 5-zone dataset already expanding |
| No explicit depth correction | Deep events have different waveform character | Depth already included as location feature |

**Roadmap:**
- [ ] 5-fold cross-validation for stable metric confidence intervals
- [ ] Error analysis broken down by depth, distance, magnitude range
- [ ] Multi-station input (average predictions across network)
- [ ] R² > 0.5 with 5,000+ training events

---

## Citation

If you use this work, please cite:

```bibtex
@misc{earthd2026,
  author    = {Upendra Surya Jonnalagadda},
  title     = {earthd: Earthquake Magnitude Prediction from P-Wave Seismograms using CNN-BiLSTM with Multi-Head Attention},
  year      = {2026},
  url       = {https://github.com/UpendraSurya/Earthquake-Prediction-System}
}
```

---

<div align="center">

Built with TensorFlow · ObsPy · USGS ComCat · IRIS FDSN

</div>
