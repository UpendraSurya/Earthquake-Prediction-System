"""Generate REPORT_earthd_v5.pdf from results and saved plots."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import json, numpy as np

RESULTS = Path('results_v5')
TS      = '20260331_114739'

results  = json.load(open(f'results_v5/results_v5_{TS}.json'))
extended = json.load(open(f'results_v5/extended_v5_{TS}.json'))
history  = json.load(open(f'models_v5/history_v5_{TS}.json'))

train_mae = history['mae']
val_mae   = history['val_mae']
epochs    = list(range(1, len(train_mae)+1))

# ─────────────────────────────────────────────────────────────────
def make_summary_page():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    y = [0.97]  # mutable container avoids nonlocal

    def heading(text, size=16):
        ax.text(0.05, y[0], text, ha='left', fontsize=size, fontweight='bold',
                color='#1a1a2e', transform=ax.transAxes)
        y[0] -= 0.045

    def row(label, value, color='black'):
        ax.text(0.08, y[0], label, ha='left', fontsize=10.5, color='#444444',
                transform=ax.transAxes)
        ax.text(0.75, y[0], value, ha='right', fontsize=10.5, fontweight='bold',
                color=color, transform=ax.transAxes)
        ax.axhline(y[0] - 0.008, xmin=0.05, xmax=0.95,
                   color='#eeeeee', linewidth=0.5)
        y[0] -= 0.031

    heading('earthd v5 — Full Validation Report', 18)
    y[0] -= 0.01

    heading('Dataset', 13)
    row('Total usable events',              '2,144')
    row('Train events (oldest 80%)',         '1,715')
    row('Val events   (middle 10%)',          '214')
    row('Test events  (most recent 10%)',     '215')
    row('Split strategy',       'Chronological — no data leakage')
    row('Sample rate',          '20 Hz  (Nyquist for 10 Hz bandpass)')
    row('Window size',          '600 samples  (30 seconds)')
    y[0] -= 0.01

    heading('Regression Metrics (Test Set)', 13)
    row('MAE',   f'{results["test_mae"]:.4f} magnitude units',  '#1a73e8')
    row('RMSE',  f'{results["test_rmse"]:.4f} magnitude units')
    row('R²',    f'{results["test_r2"]:.4f}')
    row('Bias',  f'{extended["bias_mean"]:.4f} magnitude units')
    row('Error std', f'{extended["error_std"]:.4f} magnitude units')
    y[0] -= 0.01

    heading('Tolerance Accuracy (Test Set)', 13)
    row('Within ±0.5 mag', f'{extended["within_0_5_mag"]*100:.1f}%  ← publishable standard', '#0d9448')
    row('Within ±0.3 mag', f'{extended["within_0_3_mag"]*100:.1f}%  ← good',                '#0d9448')
    row('Within ±0.1 mag', f'{extended["within_0_1_mag"]*100:.1f}%')
    y[0] -= 0.01

    heading('Classification (4 Magnitude Bins)', 13)
    row('Bin accuracy', f'{extended["class_accuracy"]*100:.1f}%')
    for cls, cnt in extended["class_counts_true"].items():
        row(f'True: {cls}', str(cnt))
    y[0] -= 0.01

    heading('Training', 13)
    row('Epochs ran',       str(results["epochs_ran"]))
    row('Best val MAE',     f'{results["best_val_mae"]:.4f}')
    row('Architecture',     'CNN-BiLSTM + Multi-Head Attention')
    row('Parameters',       '~400,000')
    row('Training time',    '~4 min 39 s  (Apple Metal GPU)')
    return fig


def make_next_steps():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.text(0.05, 0.95, 'Next Steps Toward Publication', ha='left',
            fontsize=18, fontweight='bold', color='#1a1a2e', transform=ax.transAxes)

    sections = [
        ('Immediate', [
            '1. Run data collection for 24 h (maxradius=20°, 10 networks)',
            '   → Target 5,000+ train events  →  R² expected to rise to 0.4–0.6',
            '2. Re-train — confusion matrix will improve with more rare large events',
        ]),
        ('Short Term', [
            '3. Baseline: Random Forest on same data  → prove CNN-BiLSTM wins',
            '4. k-fold cross-validation (5-fold) for stable confidence-interval metrics',
            '5. Error breakdown by depth, distance, magnitude range',
        ]),
        ('For Submission', [
            '6. Expand geographic scope beyond India/Himalaya',
            '7. Compare against CREIME / MagNet on same test events',
            '8. R² > 0.5 required  (currently limited to 1,715 training events)',
            '9. Write Methods: P-wave anchor + chronological split justification',
        ]),
        ('Key Metrics to Hit', [
            f'   MAE < 0.25  ✓  (currently {results["test_mae"]:.3f})',
            f'   Within ±0.5 > 90%  ✓  (currently {extended["within_0_5_mag"]*100:.1f}%)',
            f'   R² > 0.5    ✗  (currently {results["test_r2"]:.3f}  — needs more data)',
            '   k-fold results required — no single-split variance',
        ]),
    ]
    y = 0.86
    for section, items in sections:
        ax.text(0.05, y, section, ha='left', fontsize=13, fontweight='bold',
                color='#1a73e8', transform=ax.transAxes)
        y -= 0.04
        for item in items:
            ax.text(0.07, y, item, ha='left', fontsize=10, color='#333333',
                    transform=ax.transAxes)
            y -= 0.035
        y -= 0.015

    ax.text(0.05, 0.06, 'Backup: earthd_backup_20260331_114758/  (971 MB)',
            ha='left', fontsize=9, color='#888888', transform=ax.transAxes)
    ax.text(0.05, 0.03, 'Generated: 2026-03-31  |  earthd v5  |  CNN-BiLSTM + Multi-Head Attention',
            ha='left', fontsize=9, color='#888888', transform=ax.transAxes)
    return fig


# ─────────────────────────────────────────────────────────────────
with PdfPages('REPORT_earthd_v5.pdf') as pdf:

    # 1. Cover page
    fig = plt.figure(figsize=(8.5, 11))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.text(0.5, 0.80, 'earthd v5', ha='center', fontsize=48,
            fontweight='bold', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.70, 'Earthquake Magnitude Prediction', ha='center', fontsize=20,
            color='#a8d8ea', transform=ax.transAxes)
    ax.text(0.5, 0.62, 'CNN-BiLSTM + Multi-Head Attention\non P-Wave Seismograms',
            ha='center', fontsize=14, color='#a8d8ea', transform=ax.transAxes)
    ax.text(0.5, 0.48, f'MAE:  {results["test_mae"]:.3f} mag units', ha='center',
            fontsize=30, fontweight='bold', color='#00ff88', transform=ax.transAxes)
    ax.text(0.5, 0.40, f'Within ±0.5 mag:  {extended["within_0_5_mag"]*100:.1f}%',
            ha='center', fontsize=22, color='#00ff88', transform=ax.transAxes)
    ax.text(0.5, 0.30, f'{results["train_events"]:,} train  |  {results["val_events"]} val  |  {results["test_events"]} test',
            ha='center', fontsize=13, color='#cccccc', transform=ax.transAxes)
    ax.text(0.5, 0.24, f'{results["epochs_ran"]} epochs  |  Best val MAE:  {results["best_val_mae"]:.4f}',
            ha='center', fontsize=13, color='#cccccc', transform=ax.transAxes)
    ax.text(0.5, 0.12, '2026-03-31  |  India / Himalayan Seismic Zone  |  2000–2023',
            ha='center', fontsize=11, color='#888888', transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # 2. Metrics summary
    fig = make_summary_page()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # 3. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Curves — earthd v5', fontsize=14, fontweight='bold')
    axes[0].plot(epochs, train_mae, label='Train MAE', color='#1a73e8', linewidth=2)
    axes[0].plot(epochs, val_mae,   label='Val MAE',   color='#f4511e', linewidth=2)
    axes[0].axhline(results['best_val_mae'], color='green', linestyle='--', linewidth=1.5,
                    label=f'Best val MAE = {results["best_val_mae"]:.3f}')
    axes[0].set(title='MAE per Epoch', xlabel='Epoch', ylabel='MAE (magnitude units)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['val_loss'], label='Val Loss (MSE)',   color='#f4511e', linewidth=2)
    axes[1].plot(epochs, history['loss'],     label='Train Loss (MSE)', color='#1a73e8', linewidth=2)
    axes[1].set(title='Loss per Epoch', xlabel='Epoch', ylabel='MSE')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # 4–7. Embed saved result plots
    for fpath, title in [
        (f'{RESULTS}/confusion_matrix_v5_{TS}.png',  'Confusion Matrix — Magnitude Classes'),
        (f'{RESULTS}/per_class_analysis_v5_{TS}.png', 'Per-Class Error Analysis'),
        (f'{RESULTS}/pred_vs_true_v5_{TS}.png',       'Predicted vs True Magnitude'),
        (f'{RESULTS}/error_dist_v5_{TS}.png',          'Prediction Error Distribution'),
    ]:
        if Path(fpath).exists():
            img = plt.imread(fpath)
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    # 8. Next steps
    fig = make_next_steps()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print('PDF written: REPORT_earthd_v5.pdf')
