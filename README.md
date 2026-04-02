# earthd — Earthquake Magnitude Prediction

End-to-end deep learning pipeline for predicting earthquake magnitude from raw seismic waveforms using a dual-input CNN-BiLSTM with Multi-Head Attention.

## Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ | ±0.5 acc ↑ | ±0.3 acc ↑ |
|---|---|---|---|---|---|
| Mean predictor | 0.257 | — | -0.016 | — | — |
| RF-Location (4 features) | 0.297 | 0.392 | -0.196 | 84.3% | 59.3% |
| RF-Waveform (12 stats + loc) | 0.276 | 0.363 | -0.030 | 86.6% | 62.9% |
| **CNN-BiLSTM (this work)** | **0.271** | **0.376** | -0.035 | **87.5%** | **66.0%** |

Trained on 21,000+ real earthquake waveforms across 5 Asian seismic zones.

---

## Architecture

```
Waveform input (600 samples, 30s @ 20Hz)
    → Conv1D(32, k=7) → BN → MaxPool
    → Conv1D(64, k=5) → BN → MaxPool
    → Conv1D(128, k=3) → BN → MaxPool → ResBlock(256)
    → BiLSTM(64, return_sequences=True)
    → MultiHeadAttention(heads=4, key_dim=32)
    → BiLSTM(32)
    → Dense(64) → Dropout(0.3)
          ↘
Location input (dist_km, depth_km, lat, lon)
    → Dense(32) → BN → Dense(16)
          ↘
    Concatenate → Dense(64) → Dropout(0.3) → Dense(1) → magnitude
```

~400k trainable parameters.

---

## Project Structure

```
earthd/
├── data_collection_v5.ipynb   # USGS catalog fetch + IRIS waveform download
├── training_v5.ipynb          # Model training, evaluation, plots
├── orchestrator.py            # Single-command pipeline runner with auto-retry
├── rf_baseline.py             # Random Forest baseline comparison
├── generate_report_pdf.py     # Auto-generates multi-page PDF evaluation report
├── check_status.py            # Pipeline status checker
└── REPORT_earthd_v5.md        # Full technical report
```

---

## Pipeline

The full pipeline runs as a single command:

```bash
python orchestrator.py
```

This runs in order:
1. **Data Collection** — fetches USGS catalog, downloads waveforms from IRIS FDSN
2. **Preprocessing** — resample to 20Hz, bandpass filter, extract 30s P-wave windows
3. **Training** — trains CNN-BiLSTM, saves best model checkpoint
4. **Evaluation** — confusion matrix, tolerance accuracy, per-class analysis
5. **Report** — generates PDF report automatically

Force a specific phase:
```bash
python orchestrator.py --force-phase training
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow obspy scikit-learn pandas numpy matplotlib seaborn tqdm scipy
```

---

## Data

Waveforms are downloaded automatically from the [IRIS FDSN](https://service.iris.edu/fdsnws/) archive. The pipeline queries the [USGS ComCat API](https://earthquake.usgs.gov/fdsnws/event/1/) for event catalogs.

**Regions covered:**
- India / Himalaya (5°–40°N, 60°–100°E)
- Southeast Asia (12°S–25°N, 90°–135°E)
- Central Asia / Zagros (25°–50°N, 40°–80°E)
- Japan / Korea (25°–50°N, 125°–150°E)
- China / Tibet (20°–50°N, 75°–125°E)

**Magnitude range:** M3.0 – M7.5
**Time range:** 2000 – 2024
**Sample rate:** 20 Hz (resampled)
**Window:** 30 seconds starting 1s before P-wave

---

## Notifications

The orchestrator sends push notifications via [ntfy.sh](https://ntfy.sh) on phase completion/failure. Set your topic in `orchestrator.py`:

```python
NTFY_TOPIC   = 'your-topic-here'
NTFY_ENABLED = True
```

---

## Key Concepts

- **Chronological split**: events sorted by origin time, 80/10/10 split — prevents temporal data leakage
- **Dual-input fusion**: separate branches for waveform (CNN+BiLSTM) and location (dense), merged before output
- **Class imbalance**: dataset has ~13,000 Minor vs ~90 Major events — addressed with inverse-frequency sample weights
- **Baseline comparison**: two RF variants (location-only and waveform statistics) establish that CNN-BiLSTM adds genuine value

---

## Tech Stack

`Python` `TensorFlow/Keras` `ObsPy` `scikit-learn` `pandas` `NumPy` `SciPy` `matplotlib` `seaborn` `tqdm`
