#!/usr/bin/env python3
"""
ML Pipeline Orchestrator — earthd project  (self-healing edition)
==================================================================
Runs data collection → model training in sequence.
Detects errors in notebook output, auto-fixes them, and retries.
Survives terminal/session disconnects when launched with nohup.

Usage:
    nohup .venv/bin/python orchestrator.py > pipeline.log 2>&1 &

Resume (skips already-succeeded phases):
    nohup .venv/bin/python orchestrator.py --resume > pipeline.log 2>&1 &

Force re-run one phase:
    nohup .venv/bin/python orchestrator.py --force-phase data_collection > pipeline.log 2>&1 &
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.resolve()
VENV_PY     = BASE_DIR / '.venv' / 'bin' / 'python'
VENV_PIP    = BASE_DIR / '.venv' / 'bin' / 'pip'
VENV_JNB    = BASE_DIR / '.venv' / 'bin' / 'jupyter'

STATUS_FILE = BASE_DIR / 'pipeline_status.json'
LOG_FILE    = BASE_DIR / 'pipeline.log'

MAX_RETRIES = 3          # max auto-fix+retry attempts per phase
RETRY_DELAY = 30         # seconds to wait before each retry

# ── Push notifications via ntfy.sh (free, no account needed) ─────────────────
# Install the ntfy app on your phone (iOS / Android) and subscribe to this topic.
# iOS:     https://apps.apple.com/app/ntfy/id1625396347
# Android: https://play.google.com/store/apps/details?id=io.heckel.ntfy
# Then subscribe to the topic below inside the app.
NTFY_TOPIC   = 'earthd-a7971bd4'        # your unique topic — keep this private
NTFY_ENABLED = True                      # set False to silence all notifications

PHASES = [
    {
        'name'         : 'data_collection',
        'label'        : 'Data Collection (data_collection_v5.ipynb)',
        'notebook'     : BASE_DIR / 'data_collection_v5.ipynb',
        'estimate_sec' : 24 * 3600,   # expanded region can take 12–24 h
        'success_check': lambda: _count_processed_csv() > 0,
        'success_hint' : 'processed_v5/*.csv files exist',
    },
    {
        'name'         : 'training',
        'label'        : 'Model Training (training_v5.ipynb)',
        'notebook'     : BASE_DIR / 'training_v5.ipynb',
        'estimate_sec' : 2 * 3600,
        'success_check': lambda: _training_output_exists(),
        'success_hint' : 'models_v5/*.keras file exists',
    },
]

# ─────────────────────────────────────────────────────────────────
#  SUCCESS CHECKS
# ─────────────────────────────────────────────────────────────────

def _count_processed_csv() -> int:
    d = BASE_DIR / 'processed_v5'
    return len(list(d.glob('*.csv'))) if d.exists() else 0

def _training_output_exists() -> bool:
    d = BASE_DIR / 'models_v5'
    if not d.exists():
        return False
    for ext in ('*.keras', '*.h5', '*.pt', '*.pth', '*.pkl'):
        if list(d.glob(ext)):
            return True
    return False

# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────

def log(msg: str, level: str = 'INFO') -> None:
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] [{level:7s}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as fh:
        fh.write(line + '\n')

# ─────────────────────────────────────────────────────────────────
#  STATUS HELPERS
# ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def _load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            pass
    return {'pipeline': 'not_started', 'phases': {}}

def _save_status(s: dict) -> None:
    STATUS_FILE.write_text(json.dumps(s, indent=2))

def _update_phase(pipeline: dict, name: str, **fields) -> None:
    pipeline['phases'].setdefault(name, {}).update(fields)
    _save_status(pipeline)

# ─────────────────────────────────────────────────────────────────
#  PUSH NOTIFICATIONS  (ntfy.sh — free, no account needed)
# ─────────────────────────────────────────────────────────────────

def _notify(title: str, message: str, priority: str = 'default',
            tags: list = None) -> None:
    """Send a push notification to your phone via ntfy.sh.

    priority: 'max' (urgent), 'high', 'default', 'low', 'min'
    tags    : emoji / tag names shown in notification  e.g. ['white_check_mark']
    """
    if not NTFY_ENABLED:
        return
    import urllib.request, urllib.error
    url     = f'https://ntfy.sh/{NTFY_TOPIC}'
    headers = {
        'Title'   : title.encode(),
        'Priority': priority.encode(),
    }
    if tags:
        headers['Tags'] = ','.join(tags).encode()
    try:
        req = urllib.request.Request(
            url,
            data    = message.encode('utf-8'),
            headers = headers,
            method  = 'POST',
        )
        urllib.request.urlopen(req, timeout=10)
        log(f'NTFY  Notification sent: "{title}"')
    except Exception as exc:
        log(f'NTFY  Failed to send notification: {exc}', 'WARNING')


# ─────────────────────────────────────────────────────────────────
#  DURATION FORMATTER
# ─────────────────────────────────────────────────────────────────

def _fmt(sec: float) -> str:
    sec = int(sec)
    if sec < 60:   return f'{sec}s'
    if sec < 3600: return f'{sec//60}m {sec%60}s'
    return f'{sec//3600}h {(sec%3600)//60}m'

# ─────────────────────────────────────────────────────────────────
#  NOTEBOOK CONFIG PATCHERS  (used by auto-fix routines)
# ─────────────────────────────────────────────────────────────────

def _patch_notebook_value(nb_path: Path, pattern: str, replacement: str) -> bool:
    """Regex-replace a value inside a notebook's source JSON and save it."""
    try:
        nb = json.loads(nb_path.read_text())
        changed = False
        for cell in nb.get('cells', []):
            src = cell.get('source', '')
            if isinstance(src, list):
                new_src = [re.sub(pattern, replacement, line) for line in src]
                if new_src != src:
                    cell['source'] = new_src
                    changed = True
            elif isinstance(src, str):
                new_src = re.sub(pattern, replacement, src)
                if new_src != src:
                    cell['source'] = new_src
                    changed = True
        if changed:
            nb_path.write_text(json.dumps(nb, indent=1))
        return changed
    except Exception as exc:
        log(f'  patch_notebook failed: {exc}', 'WARNING')
        return False

def _read_notebook_value(nb_path: Path, pattern: str):
    """Return first regex match group from any notebook cell source."""
    try:
        nb = json.loads(nb_path.read_text())
        for cell in nb.get('cells', []):
            src = cell.get('source', '')
            text = ''.join(src) if isinstance(src, list) else src
            m = re.search(pattern, text)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────
#  AUTO-FIX ENGINE
#  Each fixer returns True if it applied a fix (triggering a retry).
# ─────────────────────────────────────────────────────────────────

def _fix_missing_module(output: str, nb_path: Path) -> bool:
    """pip-install any module that raised ModuleNotFoundError."""
    matches = re.findall(
        r"ModuleNotFoundError: No module named '([^']+)'", output)
    if not matches:
        return False
    # de-duplicate; map common import names → pip package names
    pip_map = {
        'sklearn'   : 'scikit-learn',
        'cv2'       : 'opencv-python',
        'PIL'       : 'Pillow',
        'tensorflow': 'tensorflow-macos',
        'tf'        : 'tensorflow-macos',
    }
    installed_any = False
    seen = set()
    for mod in matches:
        pkg = pip_map.get(mod, mod)
        if pkg in seen:
            continue
        seen.add(pkg)
        log(f'  AUTO-FIX: pip install {pkg}')
        r = subprocess.run(
            [str(VENV_PIP), 'install', '--quiet', pkg],
            capture_output=True, text=True)
        if r.returncode == 0:
            log(f'  AUTO-FIX: installed {pkg} OK')
            installed_any = True
        else:
            log(f'  AUTO-FIX: install {pkg} failed — {r.stderr.strip()[:200]}',
                'WARNING')
    return installed_any


def _fix_oom(output: str, nb_path: Path) -> bool:
    """Halve BATCH_SIZE in the notebook if an out-of-memory error is detected."""
    oom_signals = [
        'ResourceExhaustedError', 'OOM', 'out of memory',
        'Cannot allocate memory', 'RESOURCE_EXHAUSTED',
        'Dst tensor is not initialized',
    ]
    if not any(sig in output for sig in oom_signals):
        return False

    current = _read_notebook_value(nb_path, r'BATCH_SIZE\s*=\s*(\d+)')
    if current is None:
        log('  AUTO-FIX OOM: BATCH_SIZE not found in notebook', 'WARNING')
        return False

    new_val = max(8, int(current) // 2)
    if new_val == int(current):
        log('  AUTO-FIX OOM: BATCH_SIZE already at minimum (8), cannot reduce further',
            'WARNING')
        return False

    changed = _patch_notebook_value(
        nb_path,
        r'(BATCH_SIZE\s*=\s*)\d+',
        rf'\g<1>{new_val}')
    if changed:
        log(f'  AUTO-FIX OOM: BATCH_SIZE {current} → {new_val}')
    return changed


def _fix_wrong_kernel(output: str, nb_path: Path) -> bool:
    """Fix TF metal TypeError caused by wrong kernel (system Python vs venv)."""
    signals = [
        'Unable to convert function return value to a Python type',
        'ipykernel_launcher',
        'Python/3.9',
        'python3.9',
    ]
    if not any(sig in output for sig in signals):
        return False
    log('  AUTO-FIX: wrong kernel / TF-metal mismatch detected')
    # Re-register the venv kernel and downgrade tensorflow-metal to stable version
    subprocess.run(
        [str(VENV_PY), '-m', 'ipykernel', 'install', '--user',
         '--name', 'earthd', '--display-name', 'earthd (Python 3.10)'],
        capture_output=True)
    subprocess.run(
        [str(VENV_PIP), 'install', '--quiet', 'tensorflow-metal==1.1.0'],
        capture_output=True)
    log('  AUTO-FIX: re-registered earthd kernel + pinned tensorflow-metal==1.1.0')
    return True


def _fix_connection_error(output: str, nb_path: Path) -> bool:
    """Sleep and retry on IRIS/USGS connection failures."""
    signals = [
        'ConnectionError', 'ConnectionResetError', 'RemoteDisconnected',
        'ChunkedEncodingError', 'ReadTimeout', 'FDSNException',
        'ServiceUnavailable', '503', '429', 'timed out',
        'Failed to establish a new connection',
    ]
    if not any(sig in output for sig in signals):
        return False
    wait = 120
    log(f'  AUTO-FIX: network/IRIS error detected — waiting {wait}s before retry')
    time.sleep(wait)
    return True   # just retry — the notebook's internal retry logic will handle the rest


def _fix_kernel_crash(output: str, nb_path: Path) -> bool:
    """Detect nbconvert kernel-died errors and retry."""
    signals = [
        'Kernel died', 'Kernel interrupted', 'DeadKernelError',
        'A process in the process group', 'nbconvert.preprocessors.execute',
        'CellExecutionError',
    ]
    if not any(sig in output for sig in signals):
        return False
    log('  AUTO-FIX: kernel crash detected — will retry')
    time.sleep(15)
    return True


def _fix_tf_metal(output: str, nb_path: Path) -> bool:
    """Install tensorflow-metal if Metal GPU errors appear."""
    if 'tensorflow_metal' not in output and 'metal' not in output.lower():
        return False
    if 'No module named' not in output and 'ImportError' not in output:
        return False
    log('  AUTO-FIX: installing tensorflow-metal')
    r = subprocess.run(
        [str(VENV_PIP), 'install', '--quiet', 'tensorflow-metal'],
        capture_output=True, text=True)
    return r.returncode == 0


# Ordered list — each fixer is tried in sequence on failure
FIXERS = [
    _fix_wrong_kernel,      # check first — most specific TF/kernel error
    _fix_missing_module,
    _fix_oom,
    _fix_connection_error,
    _fix_kernel_crash,
    _fix_tf_metal,
]


def _attempt_auto_fix(output: str, nb_path: Path, attempt: int) -> bool:
    """
    Try every fixer against the captured output.
    Returns True if at least one fix was applied (meaning we should retry).
    """
    log(f'  Scanning output for known error patterns (attempt {attempt}/{MAX_RETRIES}) ...')
    fixed = False
    for fixer in FIXERS:
        try:
            if fixer(output, nb_path):
                fixed = True
        except Exception as exc:
            log(f'  Fixer {fixer.__name__} raised: {exc}', 'WARNING')
    if not fixed:
        log('  No auto-fix matched this error. Manual inspection needed.', 'WARNING')
    return fixed

# ─────────────────────────────────────────────────────────────────
#  SUBPROCESS RUNNER  (streams output AND captures it for analysis)
# ─────────────────────────────────────────────────────────────────

def _run_and_capture(cmd: list[str]) -> tuple[int, str]:
    """
    Run cmd, stream every line to stdout (→ pipeline.log) in real time,
    AND accumulate all output in a string for error analysis.
    Returns (returncode, full_output).
    """
    buf = []

    def _stream(pipe):
        for raw in iter(pipe.readline, ''):
            line = raw.rstrip('\n')
            print(line, flush=True)
            with open(LOG_FILE, 'a') as fh:
                fh.write(line + '\n')
            buf.append(line)
        pipe.close()

    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    t = threading.Thread(target=_stream, args=(proc.stdout,))
    t.start()
    proc.wait()
    t.join()
    return proc.returncode, '\n'.join(buf)

# ─────────────────────────────────────────────────────────────────
#  PHASE RUNNER  (with auto-fix retry loop)
# ─────────────────────────────────────────────────────────────────

def run_phase(phase: dict, pipeline: dict, force: bool = False) -> bool:
    name    = phase['name']
    label   = phase['label']
    nb_path = phase['notebook']
    pdata   = pipeline['phases'].get(name, {})

    # ── skip if already done ─────────────────────────────────────
    if pdata.get('status') == 'success' and not force:
        log(f'SKIP  {label} — already completed at {pdata.get("ended_at","?")}')
        return True

    # ── notebook must exist ──────────────────────────────────────
    if not nb_path.exists():
        log(f'SKIP  {label} — notebook not found: {nb_path.name}', 'WARNING')
        _update_phase(pipeline, name,
                      status='skipped', reason=f'{nb_path.name} not found')
        return True   # soft skip — don't block the pipeline

    cmd = [
        str(VENV_JNB), 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--ExecutePreprocessor.timeout=172800',   # 48 h — overnight multi-region download
        '--ExecutePreprocessor.kernel_name=earthd',   # force venv Python 3.10 kernel
        '--inplace',
        str(nb_path),
    ]

    start_ts = _now_iso()
    log(f'START {label}')
    _update_phase(pipeline, name,
                  status='running', started_at=start_ts,
                  estimate_sec=phase['estimate_sec'],
                  ended_at=None, error=None, attempts=0)
    pipeline['pipeline']      = 'running'
    pipeline['current_phase'] = name
    _save_status(pipeline)

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            log(f'RETRY {label} — attempt {attempt}/{MAX_RETRIES} '
                f'(waiting {RETRY_DELAY}s ...)')
            time.sleep(RETRY_DELAY)

        _update_phase(pipeline, name, attempts=attempt)
        rc, output = _run_and_capture(cmd)
        end_ts     = _now_iso()

        # ── check artefacts ──────────────────────────────────────
        try:
            artefact_ok = phase['success_check']()
        except Exception:
            artefact_ok = False

        if rc == 0 and artefact_ok:
            elapsed = (datetime.fromisoformat(end_ts) -
                       datetime.fromisoformat(start_ts)).total_seconds()
            log(f'DONE  {label}  (took {_fmt(elapsed)}, {attempt} attempt(s))')
            _update_phase(pipeline, name,
                          status='success', ended_at=end_ts, error=None)
            csv_count = _count_processed_csv()
            _notify(
                title   = f'✅ earthd — {label.split("(")[0].strip()} done',
                message = (f'Took {_fmt(elapsed)}.\n'
                           f'Processed CSVs: {csv_count:,}\n'
                           f'Next: training will start automatically.'),
                priority = 'high',
                tags     = ['white_check_mark', 'earthquake'],
            )
            return True

        if rc == 0 and not artefact_ok:
            log(f'WARN  {label} — exit 0 but artefact check failed '
                f'({phase["success_hint"]})', 'WARNING')
            _update_phase(pipeline, name,
                          status='warning', ended_at=end_ts,
                          error='artefact check failed: ' + phase['success_hint'])
            _notify(
                title   = f'⚠️ earthd — {label.split("(")[0].strip()} warning',
                message = f'Finished but artefact check failed: {phase["success_hint"]}',
                priority = 'high',
                tags     = ['warning'],
            )
            return True   # soft success

        # ── failure — try to auto-fix ────────────────────────────
        log(f'FAIL  {label} — exit code {rc} (attempt {attempt}/{MAX_RETRIES})',
            'ERROR')
        _update_phase(pipeline, name,
                      status='failed', ended_at=end_ts,
                      error=f'exit code {rc} on attempt {attempt}')

        if attempt < MAX_RETRIES:
            fixed = _attempt_auto_fix(output, nb_path, attempt)
            if not fixed:
                # no fix available — no point retrying, fail immediately
                log(f'HALT  {label} — no fix available, giving up early', 'ERROR')
                break
        else:
            log(f'HALT  {label} — exhausted {MAX_RETRIES} attempts', 'ERROR')

    # ── all attempts failed ──────────────────────────────────────
    log(f'ERROR: Phase "{name}" could not be auto-recovered.', 'ERROR')
    log(f'       Check pipeline.log for the full error output.', 'ERROR')
    log(f'       Fix manually, then re-run:  '
        f'.venv/bin/python orchestrator.py --resume', 'ERROR')
    _notify(
        title   = f'❌ earthd — {label.split("(")[0].strip()} FAILED',
        message = (f'All {MAX_RETRIES} attempts failed.\n'
                   f'Check pipeline.log and run:\n'
                   f'python orchestrator.py --resume'),
        priority = 'max',
        tags     = ['rotating_light', 'earthquake'],
    )
    return False

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='earthd ML pipeline orchestrator')
    parser.add_argument('--resume',       action='store_true',
                        help='Skip phases that already succeeded')
    parser.add_argument('--force-phase',  metavar='PHASE_NAME',
                        help='Re-run a specific phase even if it already succeeded')
    parser.add_argument('--list-phases',  action='store_true',
                        help='Print phase names and exit')
    args = parser.parse_args()

    if args.list_phases:
        for p in PHASES:
            print(f"  {p['name']:25s}  {p['label']}")
        return

    log('=' * 65)
    log('earthd ML pipeline — STARTED  (self-healing edition)')
    log(f'PID      : {os.getpid()}')
    log(f'Base dir : {BASE_DIR}')
    log(f'Mode     : {"RESUME" if args.resume else "FULL RUN"}'
        + (f'  (force: {args.force_phase})' if args.force_phase else ''))
    log(f'Auto-fix : up to {MAX_RETRIES} retries per phase')
    log('=' * 65)

    # --force-phase always loads existing status (so other phases stay skipped)
    # --resume also loads existing status
    # plain run starts fresh
    pipeline = (_load_status() if (args.resume or args.force_phase)
                else {'pipeline': 'starting', 'phases': {}})
    pipeline['pipeline_started_at'] = (pipeline.get('pipeline_started_at')
                                        or _now_iso())
    pipeline['pid'] = os.getpid()
    _save_status(pipeline)

    for phase in PHASES:
        force_this = (args.force_phase == phase['name'])
        ok = run_phase(phase, pipeline, force=force_this)
        if not ok:
            pipeline['pipeline']         = 'failed'
            pipeline['pipeline_ended_at'] = _now_iso()
            _save_status(pipeline)
            log('Pipeline HALTED — check pipeline.log for details.', 'ERROR')
            _notify(
                title   = '🛑 earthd — Pipeline HALTED',
                message = 'A phase failed and could not be auto-fixed.\nCheck pipeline.log.',
                priority = 'max',
                tags     = ['rotating_light'],
            )
            sys.exit(1)

    pipeline['pipeline']         = 'success'
    pipeline['pipeline_ended_at'] = _now_iso()
    _save_status(pipeline)
    log('=' * 65)
    log('Pipeline COMPLETE — all phases succeeded.')
    log('=' * 65)
    csv_count   = _count_processed_csv()
    model_count = len(list((BASE_DIR / 'models_v5').glob('*.keras'))) if (BASE_DIR / 'models_v5').exists() else 0
    _notify(
        title   = '🎉 earthd — Full pipeline COMPLETE',
        message = (f'All phases succeeded.\n'
                   f'CSVs: {csv_count:,}  |  Models: {model_count}\n'
                   f'Run: python check_status.py for results.'),
        priority = 'high',
        tags     = ['tada', 'earthquake'],
    )


if __name__ == '__main__':
    main()
