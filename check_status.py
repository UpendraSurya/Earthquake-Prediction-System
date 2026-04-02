#!/usr/bin/env python3
"""
Pipeline Status Checker — earthd project
=========================================
Run at any time to see where the pipeline is:

    python check_status.py          # pretty summary
    python check_status.py --json   # raw JSON
    python check_status.py --watch  # refresh every 30s
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR    = Path(__file__).parent.resolve()
STATUS_FILE = BASE_DIR / 'pipeline_status.json'
LOG_FILE    = BASE_DIR / 'pipeline.log'

# ── colour helpers (graceful fallback if terminal doesn't support ANSI) ──────

RESET  = '\033[0m'
BOLD   = '\033[1m'
GREEN  = '\033[32m'
YELLOW = '\033[33m'
RED    = '\033[31m'
CYAN   = '\033[36m'
DIM    = '\033[2m'

def _c(text, *codes):
    if not sys.stdout.isatty():
        return text
    return ''.join(codes) + str(text) + RESET

def _status_colour(s):
    return {
        'success' : _c(s, GREEN,  BOLD),
        'running' : _c(s, YELLOW, BOLD),
        'failed'  : _c(s, RED,    BOLD),
        'warning' : _c(s, YELLOW),
        'skipped' : _c(s, DIM),
    }.get(s, s)

# ── time helpers ──────────────────────────────────────────────────────────────

def _parse_iso(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _fmt_duration(seconds):
    if seconds is None:
        return '—'
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m {seconds % 60}s'
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f'{h}h {m}m'

def _elapsed(started_at):
    dt = _parse_iso(started_at)
    if not dt:
        return None
    now = datetime.now(tz=dt.tzinfo)
    return (now - dt).total_seconds()

def _eta(started_at, estimate_sec):
    if not started_at or not estimate_sec:
        return None, None, None
    elapsed   = _elapsed(started_at)
    remaining = max(0, estimate_sec - elapsed)
    pct       = min(100, int(elapsed / estimate_sec * 100))
    return elapsed, remaining, pct

def _progress_bar(pct, width=30):
    filled = int(width * pct / 100)
    bar    = '█' * filled + '░' * (width - filled)
    return f'[{bar}] {pct:3d}%'

# ── process check ─────────────────────────────────────────────────────────────

def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False

# ── waveform / processed count ───────────────────────────────────────────────

def _counts():
    wav   = len(list((BASE_DIR / 'waveforms_v5').glob('*.mseed')))  if (BASE_DIR / 'waveforms_v5').exists()  else 0
    proc  = len(list((BASE_DIR / 'processed_v5').glob('*.csv')))    if (BASE_DIR / 'processed_v5').exists()  else 0
    mdls  = len(list((BASE_DIR / 'models_v5').glob('*.keras')))     if (BASE_DIR / 'models_v5').exists()     else 0
    return wav, proc, mdls

# ── last N log lines ──────────────────────────────────────────────────────────

def _tail_log(n=10):
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(errors='replace').splitlines()
    return lines[-n:]

# ── main display ──────────────────────────────────────────────────────────────

def print_status(raw=False, tail=10):
    if not STATUS_FILE.exists():
        print(_c('No status file found.', DIM))
        print(f'Start the pipeline first:  nohup .venv/bin/python orchestrator.py > pipeline.log 2>&1 &')
        return

    data = json.loads(STATUS_FILE.read_text())

    if raw:
        print(json.dumps(data, indent=2))
        return

    pipeline_state = data.get('pipeline', 'unknown')
    pid            = data.get('pid')
    started_at     = data.get('pipeline_started_at')
    ended_at       = data.get('pipeline_ended_at')
    current_phase  = data.get('current_phase', '—')
    phases         = data.get('phases', {})

    wav_count, proc_count, model_count = _counts()
    pid_alive = _pid_alive(pid)

    # ── header ──────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print()
    print(_c('═' * 62, BOLD))
    print(_c(f'  earthd ML Pipeline — Status Report   {now_str}', BOLD))
    print(_c('═' * 62, BOLD))
    print()

    # ── overall status ───────────────────────────────────────────────────────
    print(f'  Pipeline  :  {_status_colour(pipeline_state)}')
    print(f'  PID       :  {pid or "—"}  '
          + (_c('(running)', GREEN) if pid_alive else _c('(not running)', DIM)))
    if started_at:
        elapsed_total = _elapsed(started_at)
        print(f'  Started   :  {started_at}  ({_fmt_duration(elapsed_total)} ago)')
    if ended_at:
        print(f'  Ended     :  {ended_at}')
    print()

    # ── artefact counts ──────────────────────────────────────────────────────
    print(f'  MSEED waveforms  :  {wav_count:,}   (waveforms_v5/)')
    print(f'  Processed CSVs   :  {proc_count:,}   (processed_v5/)')
    print(f'  Saved models     :  {model_count:,}   (models_v5/)')
    print()

    # ── per-phase breakdown ──────────────────────────────────────────────────
    print(_c('  Phase Details', BOLD))
    print('  ' + '─' * 58)

    PHASE_LABELS = {
        'data_collection' : 'Data Collection (IRIS waveforms → processed_v5/)',
        'training'        : 'Model Training (CNN-BiLSTM → models_v5/)',
    }

    for phase_name, label in PHASE_LABELS.items():
        pdata  = phases.get(phase_name, {})
        pstate = pdata.get('status', 'not_started')
        icon   = {'success': '✓', 'running': '▶', 'failed': '✗',
                  'warning': '!', 'skipped': '–'}.get(pstate, '○')
        print(f'  {icon}  {label}')
        print(f'       status  :  {_status_colour(pstate)}')

        if pdata.get('started_at'):
            if pstate == 'running':
                elapsed, remaining, pct = _eta(
                    pdata['started_at'], pdata.get('estimate_sec', 0))
                if pct is not None:
                    print(f'       progress:  {_progress_bar(pct)}')
                    print(f'       elapsed  :  {_fmt_duration(elapsed)}')
                    print(f'       remaining:  ≈{_fmt_duration(remaining)}')
            else:
                t_start = _parse_iso(pdata['started_at'])
                t_end   = _parse_iso(pdata.get('ended_at'))
                if t_start and t_end:
                    took = (t_end - t_start).total_seconds()
                    print(f'       duration :  {_fmt_duration(took)}')

        if pdata.get('error'):
            print(f'       error    :  {_c(pdata["error"], RED)}')
        print()

    # ── recent log lines ─────────────────────────────────────────────────────
    if tail > 0:
        lines = _tail_log(tail)
        if lines:
            print(_c('  Recent log lines', BOLD))
            print('  ' + '─' * 58)
            for line in lines:
                print('  ' + _c(line, DIM))
            print()

    # ── next action hint ─────────────────────────────────────────────────────
    print(_c('  Actions', BOLD))
    print('  ' + '─' * 58)
    if pipeline_state in ('not_started', 'starting'):
        print('  Start  :  nohup .venv/bin/python orchestrator.py > pipeline.log 2>&1 &')
    elif pipeline_state == 'running':
        print('  Watch log  :  tail -f pipeline.log')
        print('  Full log   :  less pipeline.log')
    elif pipeline_state == 'failed':
        print('  Fix the error above, then resume:')
        print('  Resume :  nohup .venv/bin/python orchestrator.py --resume > pipeline.log 2>&1 &')
        print('  Or force a specific phase:')
        print('  Force  :  nohup .venv/bin/python orchestrator.py --force-phase data_collection > pipeline.log 2>&1 &')
    elif pipeline_state == 'success':
        print('  Pipeline finished successfully.')
        print('  Re-run :  nohup .venv/bin/python orchestrator.py > pipeline.log 2>&1 &')
    print()
    print(_c('═' * 62, BOLD))
    print()


def watch_mode(interval=30, tail=5):
    try:
        while True:
            os.system('clear')
            print_status(tail=tail)
            print(f'  (refreshing every {interval}s — Ctrl+C to stop)')
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\nStopped.')


def main():
    parser = argparse.ArgumentParser(description='Check earthd pipeline status')
    parser.add_argument('--json',  action='store_true', help='Print raw JSON')
    parser.add_argument('--watch', action='store_true', help='Refresh every 30 seconds')
    parser.add_argument('--interval', type=int, default=30,
                        help='Watch refresh interval in seconds (default 30)')
    parser.add_argument('--tail', type=int, default=10,
                        help='Number of log lines to show (default 10, 0 to hide)')
    args = parser.parse_args()

    if args.watch:
        watch_mode(interval=args.interval, tail=args.tail)
    else:
        print_status(raw=args.json, tail=args.tail)


if __name__ == '__main__':
    main()
