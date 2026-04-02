"""
rf_baseline.py — Random Forest baseline comparison for earthd v5

Loads the same train/val/test splits used by training_v5.ipynb and trains
two Random Forest regressors:
  1. RF-Location : 4 location features only (dist_km, depth_km, lat, lon)
  2. RF-Waveform : 12 hand-crafted waveform statistics + 4 location features

Prints a comparison table against the CNN-BiLSTM result.

Usage:
    python rf_baseline.py

Requirements: numpy, scikit-learn, obspy, tqdm
"""

import json
import math
import os
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("processed_v5")
TARGET_SR     = 20          # Hz
WINDOW_SEC    = 30          # seconds
N_SAMPLES     = TARGET_SR * WINDOW_SEC   # 600 samples per waveform

TRAIN_FRACTION = 0.80
VAL_FRACTION   = 0.10
# test = remaining 10%

MAG_BINS  = [0.0, 4.5, 5.5, 6.5, 99.0]
BIN_NAMES = ['Minor', 'Moderate', 'Strong', 'Major']

# If you already have a saved CNN-BiLSTM result, paste metrics here for the table
CNN_MAE = None   # e.g. 0.267
CNN_R2  = None   # e.g. 0.028

# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset():
    """Load all processed waveforms and location features, sorted chronologically.

    CSVs have columns: time_sec, amplitude, label, dist_km_n, depth_km_n, lat_n, lon_n, split
    Rows are at 100Hz (0.01s steps). We resample to 600 samples (20Hz, 30s).
    Meta JSON has: eq_lat, eq_lon, eq_depth_km, magnitude, dist_km, origin_time
    """
    import pandas as pd
    from obspy import UTCDateTime
    from scipy.signal import resample

    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not len(csv_files):
        raise FileNotFoundError(f"No .csv files found in {PROCESSED_DIR}")

    waveforms, locations, magnitudes, origin_times = [], [], [], []

    for csv_path in tqdm(csv_files, desc="Loading files"):
        meta_path = PROCESSED_DIR / (csv_path.stem + "_meta.json")
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            df  = pd.read_csv(csv_path, usecols=['amplitude'])
            amp = df['amplitude'].values.astype(np.float32)
        except Exception:
            continue

        # Resample to exactly N_SAMPLES (600) regardless of original length
        if len(amp) < 100:
            continue
        if len(amp) != N_SAMPLES:
            amp = resample(amp, N_SAMPLES).astype(np.float32)

        mag      = float(meta.get("magnitude", 0.0))
        dist_km  = float(meta.get("dist_km",   0.0))
        depth_km = float(meta.get("eq_depth_km", meta.get("depth_km", 0.0)))
        lat      = float(meta.get("eq_lat",  meta.get("latitude",  0.0)))
        lon      = float(meta.get("eq_lon",  meta.get("longitude", 0.0)))
        ot_str   = meta.get("origin_time", "1970-01-01T00:00:00")
        try:
            ot = UTCDateTime(ot_str)
            ts_val = ot.timestamp
        except Exception:
            # fallback: parse with pandas
            import pandas as pd
            ts_val = pd.Timestamp(ot_str).timestamp()

        waveforms.append(amp)
        locations.append([dist_km, depth_km, lat, lon])
        magnitudes.append(mag)
        origin_times.append(ts_val)

    print(f"\nLoaded {len(waveforms)} valid events")
    Xw = np.array(waveforms, dtype=np.float32).reshape(len(waveforms), N_SAMPLES, 1)
    Xl = np.array(locations,  dtype=np.float32)
    y  = np.array(magnitudes, dtype=np.float32)
    ts = np.array(origin_times, dtype=np.float64)

    return Xw, Xl, y, ts


def chronological_split(Xw, Xl, y, ts):
    """Sort by origin time and split 80/10/10."""
    idx = np.argsort(ts)
    Xw, Xl, y = Xw[idx], Xl[idx], y[idx]

    n = len(y)
    n_tr  = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)

    splits = {
        'train': (Xw[:n_tr],           Xl[:n_tr],           y[:n_tr]),
        'val':   (Xw[n_tr:n_tr+n_val], Xl[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val]),
        'test':  (Xw[n_tr+n_val:],     Xl[n_tr+n_val:],     y[n_tr+n_val:]),
    }
    print(f"Split sizes → train: {n_tr}  val: {n_val}  test: {n - n_tr - n_val}")
    return splits


# ── Feature engineering ───────────────────────────────────────────────────────

def extract_waveform_features(Xw):
    """
    12 hand-crafted features from each waveform window.
    Xw shape: (N, 600, 1)  →  output shape: (N, 12)
    """
    Xw2d = Xw[:, :, 0]   # (N, 600)
    abs_  = np.abs(Xw2d)

    feats = np.column_stack([
        Xw2d.max(axis=1),                          # max amplitude
        Xw2d.min(axis=1),                          # min amplitude
        abs_.max(axis=1),                          # peak absolute amplitude
        Xw2d.mean(axis=1),                         # mean
        Xw2d.std(axis=1),                          # std
        (Xw2d ** 2).mean(axis=1),                  # RMS energy
        np.percentile(abs_, 90, axis=1),           # 90th percentile abs
        np.percentile(abs_, 75, axis=1),           # 75th percentile abs
        np.percentile(abs_, 50, axis=1),           # median abs
        np.diff(Xw2d, axis=1).std(axis=1),         # zero-crossing proxy (std of diff)
        abs_.sum(axis=1),                          # total energy
        (abs_ > abs_.mean(axis=1, keepdims=True)).sum(axis=1).astype(float),  # % above mean
    ])
    return feats.astype(np.float32)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def tolerance_accuracy(y_true, y_pred, tol):
    return float((np.abs(y_pred - y_true) <= tol).mean())


def bin_index(mag_arr):
    return np.digitize(mag_arr, MAG_BINS[1:-1])   # 0=Minor,1=Moderate,2=Strong,3=Major


def class_accuracy(y_true, y_pred):
    return float((bin_index(y_true) == bin_index(y_pred)).mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset …")
    Xw, Xl, y, ts = load_dataset()
    splits = chronological_split(Xw, Xl, y, ts)

    Xw_train, Xl_train, y_train = splits['train']
    Xw_test,  Xl_test,  y_test  = splits['test']

    # ── Mean-predictor baseline ────────────────────────────────────────────────
    y_mean_pred = np.full_like(y_test, y_train.mean())
    mae_mean    = mean_absolute_error(y_test, y_mean_pred)
    r2_mean     = r2_score(y_test, y_mean_pred)

    # ── RF-Location ────────────────────────────────────────────────────────────
    print("Training RF-Location (4 features) …")
    rf_loc = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf_loc.fit(Xl_train, y_train)
    y_pred_loc  = rf_loc.predict(Xl_test)
    mae_loc     = mean_absolute_error(y_test, y_pred_loc)
    rmse_loc    = math.sqrt(mean_squared_error(y_test, y_pred_loc))
    r2_loc      = r2_score(y_test, y_pred_loc)
    tol05_loc   = tolerance_accuracy(y_test, y_pred_loc, 0.5)
    tol03_loc   = tolerance_accuracy(y_test, y_pred_loc, 0.3)
    cls_loc     = class_accuracy(y_test, y_pred_loc)

    # ── RF-Waveform ────────────────────────────────────────────────────────────
    print("Extracting waveform features …")
    Xw_train_feats = extract_waveform_features(Xw_train)
    Xw_test_feats  = extract_waveform_features(Xw_test)
    X_train_comb   = np.hstack([Xw_train_feats, Xl_train])
    X_test_comb    = np.hstack([Xw_test_feats,  Xl_test])

    print("Training RF-Waveform (12 stats + 4 location features) …")
    rf_wav = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf_wav.fit(X_train_comb, y_train)
    y_pred_wav  = rf_wav.predict(X_test_comb)
    mae_wav     = mean_absolute_error(y_test, y_pred_wav)
    rmse_wav    = math.sqrt(mean_squared_error(y_test, y_pred_wav))
    r2_wav      = r2_score(y_test, y_pred_wav)
    tol05_wav   = tolerance_accuracy(y_test, y_pred_wav, 0.5)
    tol03_wav   = tolerance_accuracy(y_test, y_pred_wav, 0.3)
    cls_wav     = class_accuracy(y_test, y_pred_wav)

    # ── Print results ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"{'Model':<35} {'MAE':>6} {'RMSE':>6} {'R²':>7} {'±0.5':>6} {'±0.3':>6} {'ClassAcc':>9}")
    print("-" * 72)
    print(f"{'Mean predictor':<35} {mae_mean:>6.4f}   {'—':>4}  {r2_mean:>7.4f}   {'—':>4}   {'—':>4}         {'—':>4}")
    print(f"{'RF-Location (4 features)':<35} {mae_loc:>6.4f} {rmse_loc:>6.4f}  {r2_loc:>7.4f} {tol05_loc:>6.2%} {tol03_loc:>6.2%}  {cls_loc:>8.2%}")
    print(f"{'RF-Waveform (12 stats+loc)':<35} {mae_wav:>6.4f} {rmse_wav:>6.4f}  {r2_wav:>7.4f} {tol05_wav:>6.2%} {tol03_wav:>6.2%}  {cls_wav:>8.2%}")
    if CNN_MAE is not None:
        print(f"{'CNN-BiLSTM (this work)':<35} {CNN_MAE:>6.4f}   {'—':>4}  {CNN_R2:>7.4f}   {'—':>4}   {'—':>4}         {'—':>4}")
    print("=" * 72)

    # ── Feature importance (RF-Location) ──────────────────────────────────────
    print()
    feat_names = ['dist_km', 'depth_km', 'lat', 'lon']
    imps = rf_loc.feature_importances_
    print("RF-Location feature importances:")
    for name, imp in sorted(zip(feat_names, imps), key=lambda x: -x[1]):
        print(f"  {name:<12} {imp:.4f}")

    # ── Save predictions for further analysis ─────────────────────────────────
    np.save("rf_loc_predictions.npy",  y_pred_loc)
    np.save("rf_wav_predictions.npy",  y_pred_wav)
    np.save("rf_baseline_y_test.npy",  y_test)
    print()
    print("Predictions saved: rf_loc_predictions.npy, rf_wav_predictions.npy, rf_baseline_y_test.npy")


if __name__ == "__main__":
    main()
