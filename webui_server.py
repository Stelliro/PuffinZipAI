from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_cors import CORS
import threading
import queue
import multiprocessing
import os
import sys
import time
import traceback
import json
import glob
import logging
import gc
import math
import secrets
import hashlib
from functools import wraps
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from puffinzip_ai.evolution_core.evolutionary_optimizer import EvolutionaryOptimizer

# --- RESOURCE DETECTION ---
def _detect_system_limits():
    """Auto-detect safe defaults based on system resources."""
    ram_gb = 8.0
    cpu_cores = 4
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        cpu_cores = psutil.cpu_count(logical=False) or 2
    except ImportError:
        try:
            cpu_cores = os.cpu_count() or 2
        except Exception:
            pass
    
    # Scale defaults to system resources
    if ram_gb >= 16 and cpu_cores >= 8:
        default_pop, default_gens, max_pop, max_gens = 100, 200, 500, 10000
    elif ram_gb >= 8 and cpu_cores >= 4:
        default_pop, default_gens, max_pop, max_gens = 50, 100, 200, 5000
    elif ram_gb >= 4:
        default_pop, default_gens, max_pop, max_gens = 30, 50, 100, 2000
    else:
        default_pop, default_gens, max_pop, max_gens = 20, 30, 50, 500
    
    return {
        'ram_gb': round(ram_gb, 1),
        'cpu_cores': cpu_cores,
        'default_pop': default_pop,
        'default_gens': default_gens,
        'max_pop': max_pop,
        'max_gens': max_gens,
        'default_workers': int(os.environ.get('PUFFIN_DEFAULT_WORKERS', max(1, cpu_cores - 1))),
    }

SYSTEM_LIMITS = _detect_system_limits()
_debug_mode = False  # Set True by --debug flag

# --- HARDWARE PROFILE & RUN PRESETS ---
def _build_hardware_profile():
    """Build a hardware profile dict from env vars set by start.sh / start.bat
    and from runtime detection via psutil / torch."""
    gpu_count = int(os.environ.get('PUFFIN_HW_GPU_COUNT', 0))
    gpu_name = os.environ.get('PUFFIN_HW_GPU_NAME', 'None')
    gpu_vram_mb = int(os.environ.get('PUFFIN_HW_GPU_VRAM_MB', 0))
    cpu_cores = int(os.environ.get('PUFFIN_HW_CPU_CORES', SYSTEM_LIMITS['cpu_cores']))
    ram_mb = int(os.environ.get('PUFFIN_HW_RAM_MB', int(SYSTEM_LIMITS['ram_gb'] * 1024)))
    ram_gb = round(ram_mb / 1024, 1)

    # If env vars weren't set (e.g. running webui_server.py directly), detect at runtime
    if gpu_count == 0 and gpu_name == 'None':
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            pass

    return {
        'gpu_count': gpu_count,
        'gpu_name': gpu_name,
        'gpu_vram_mb': gpu_vram_mb,
        'gpu_vram_gb': round(gpu_vram_mb / 1024, 1) if gpu_vram_mb > 0 else 0,
        'cpu_cores': cpu_cores,
        'ram_mb': ram_mb,
        'ram_gb': ram_gb,
        'has_gpu': gpu_count > 0,
    }


def _compute_run_presets(hw):
    """Return test / medium / max presets calibrated to the detected hardware."""
    cpu = hw['cpu_cores']
    ram = hw['ram_gb']
    vram = hw['gpu_vram_mb']
    has_gpu = hw['has_gpu']

    # Workers: leave 1 core free, min 2
    safe_workers = max(2, cpu - 1)

    # ── TEST preset — quick smoke test (< 5 min) ────────────────────────
    test = {
        'label': 'Test Run',
        'description': 'Quick smoke test (~2-5 min). Small population, few generations.',
        'population_size': 12,
        'num_generations': 10,
        'batch_size': 6,
        'cpu_workers': min(safe_workers, 4),
        'target_device': 'GPU_AUTO' if has_gpu else 'CPU',
        'infinite': False,
    }

    # ── MEDIUM preset — balanced run ─────────────────────────────────────
    if ram >= 16 and cpu >= 8:
        med_pop, med_gens, med_batch = 50, 100, 10
    elif ram >= 8 and cpu >= 4:
        med_pop, med_gens, med_batch = 30, 60, 8
    else:
        med_pop, med_gens, med_batch = 20, 40, 6

    # GPU VRAM scaling for medium
    if has_gpu and vram >= 40000:      # A100 / A40 class
        med_pop, med_batch = 80, 16
    elif has_gpu and vram >= 20000:    # RTX 3090 / 4090 class
        med_pop, med_batch = 60, 12
    elif has_gpu and vram >= 8000:     # RTX 3070 / 4060 class
        med_pop = max(med_pop, 40)

    medium = {
        'label': 'Medium Run',
        'description': 'Balanced training run (~30-60 min). Good for exploration.',
        'population_size': med_pop,
        'num_generations': med_gens,
        'batch_size': med_batch,
        'cpu_workers': safe_workers,
        'target_device': 'GPU_AUTO' if has_gpu else 'CPU',
        'infinite': False,
    }

    # ── MAX preset — full power, infinite mode ───────────────────────────
    if ram >= 32 and cpu >= 16:
        max_pop, max_gens, max_batch = 200, 500, 20
    elif ram >= 16 and cpu >= 8:
        max_pop, max_gens, max_batch = 100, 300, 16
    elif ram >= 8 and cpu >= 4:
        max_pop, max_gens, max_batch = 60, 200, 10
    else:
        max_pop, max_gens, max_batch = 40, 100, 8

    # GPU VRAM scaling for max
    if has_gpu and vram >= 80000:      # H100 class
        max_pop, max_batch = 500, 32
    elif has_gpu and vram >= 40000:    # A100 / A40 class
        max_pop, max_batch = 300, 24
    elif has_gpu and vram >= 20000:    # RTX 3090 / 4090 class
        max_pop, max_batch = 150, 16
    elif has_gpu and vram >= 8000:
        max_pop = max(max_pop, 80)

    maximum = {
        'label': 'Max Run',
        'description': 'Full-power run (infinite mode). Uses all available resources.',
        'population_size': max_pop,
        'num_generations': max_gens,
        'batch_size': max_batch,
        'cpu_workers': safe_workers,
        'target_device': 'GPU_AUTO' if has_gpu else 'CPU',
        'infinite': True,
    }

    return {'test': test, 'medium': medium, 'max': maximum}


HARDWARE_PROFILE = _build_hardware_profile()
RUN_PRESETS = _compute_run_presets(HARDWARE_PROFILE)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- IMPORTS ---
_EvolutionaryOptimizerClass: Optional[type] = None
try:
    import puffinzip_ai
    from puffinzip_ai.evolution_core.evolutionary_optimizer import EvolutionaryOptimizer as _EvOptImported
    _EvolutionaryOptimizerClass = _EvOptImported
    from puffinzip_ai.config import APP_VERSION, ELS_LOG_PREFIX
    from puffinzip_ai.hybrid_compression_engine import get_hybrid_engine
    print(f">>> [SUCCESS] PuffinZipAI {APP_VERSION} loaded correctly.")
except Exception as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    traceback.print_exc()
    APP_VERSION = "DEV_MODE"
    ELS_LOG_PREFIX = "[ELS]"
    _EvolutionaryOptimizerClass = None
    get_hybrid_engine = None

app = Flask(__name__, static_folder='webui_static', template_folder='webui_templates')
CORS(app)

# --- AUTHENTICATION (credentials file + env-var override) ---
from datetime import timedelta as _timedelta
from webui_credentials_manager import load_or_create_credentials, _CREDENTIALS_FILE

_credentials = load_or_create_credentials()
_AUTH_USERNAME = _credentials['username']
_AUTH_PASSWORD = _credentials['password']
app.secret_key = _credentials['secret_key']
_PUBLIC_ACCESS = _credentials.get('public_access', False)
_AUTH_ENABLED = True  # Always enabled — credentials are guaranteed non-empty

# Admin / remote-access credentials (optional secondary login)
_ADMIN_USERNAME = _credentials.get('admin_username', '').strip()
_ADMIN_PASSWORD = _credentials.get('admin_password', '').strip()
_ADMIN_AUTH_ENABLED = bool(_ADMIN_USERNAME and _ADMIN_PASSWORD)

# Custom URL for remote access (informational — shown in banner)
_CUSTOM_URL = _credentials.get('custom_url', '').strip()

# Session hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,       # JS cannot read the session cookie
    SESSION_COOKIE_SAMESITE='Lax',      # CSRF mitigation
    PERMANENT_SESSION_LIFETIME=_timedelta(hours=12),  # Auto-expire after 12 h
)

# Pre-hash credentials so we never compare in plain text after startup.
# Both username and password are hashed so all comparisons are constant-time.
_AUTH_USERNAME_HASH = hashlib.sha256(_AUTH_USERNAME.encode()).hexdigest()
_AUTH_PASSWORD_HASH = hashlib.sha256(_AUTH_PASSWORD.encode()).hexdigest() if _AUTH_PASSWORD else ''

# Pre-hash admin credentials (empty hash if admin auth is disabled)
_ADMIN_USERNAME_HASH = hashlib.sha256(_ADMIN_USERNAME.encode()).hexdigest() if _ADMIN_AUTH_ENABLED else ''
_ADMIN_PASSWORD_HASH = hashlib.sha256(_ADMIN_PASSWORD.encode()).hexdigest() if _ADMIN_AUTH_ENABLED else ''

# Routes that do NOT require authentication (health check for scripts)
_PUBLIC_ROUTES = frozenset(['login', 'logout', 'health', 'static'])

# --- Brute-force rate limiting ---
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}  # ip → list of timestamps
_MAX_LOGIN_ATTEMPTS = 5        # max failures per window
_LOGIN_WINDOW_SECONDS = 300    # 5-minute sliding window


def _is_rate_limited(ip: str) -> bool:
    """Return True if *ip* has exceeded the login attempt limit."""
    now = time.time()
    attempts = _LOGIN_ATTEMPTS.get(ip, [])
    # Prune old entries outside the window
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[ip] = attempts
    return len(attempts) >= _MAX_LOGIN_ATTEMPTS


def _record_failed_attempt(ip: str) -> None:
    """Record a failed login attempt for *ip*."""
    _LOGIN_ATTEMPTS.setdefault(ip, []).append(time.time())


def _check_username(candidate: str) -> bool:
    """Constant-time username comparison via SHA-256."""
    return secrets.compare_digest(
        hashlib.sha256(candidate.encode()).hexdigest(),
        _AUTH_USERNAME_HASH,
    )


def _check_password(candidate: str) -> bool:
    """Constant-time password comparison via SHA-256."""
    return secrets.compare_digest(
        hashlib.sha256(candidate.encode()).hexdigest(),
        _AUTH_PASSWORD_HASH,
    )


def _check_admin_username(candidate: str) -> bool:
    """Constant-time admin username comparison via SHA-256."""
    if not _ADMIN_AUTH_ENABLED:
        return False
    return secrets.compare_digest(
        hashlib.sha256(candidate.encode()).hexdigest(),
        _ADMIN_USERNAME_HASH,
    )


def _check_admin_password(candidate: str) -> bool:
    """Constant-time admin password comparison via SHA-256."""
    if not _ADMIN_AUTH_ENABLED:
        return False
    return secrets.compare_digest(
        hashlib.sha256(candidate.encode()).hexdigest(),
        _ADMIN_PASSWORD_HASH,
    )


@app.before_request
def _require_login():
    """Redirect unauthenticated users to the login page."""
    if not _AUTH_ENABLED:
        return  # Auth disabled — allow everything
    if request.endpoint in _PUBLIC_ROUTES:
        return  # Public route
    if session.get('authenticated'):
        return  # Already logged in
    # For API routes return 401 so JS callers can detect auth failure
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Authentication required'}), 401
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        client_ip = request.remote_addr or '0.0.0.0'
        if _is_rate_limited(client_ip):
            error = 'Too many failed attempts. Please wait a few minutes.'
        else:
            username = request.form.get('username', '')
            password = request.form.get('password', '')
            if (_check_username(username) and _check_password(password)) or \
               (_check_admin_username(username) and _check_admin_password(password)):
                session['authenticated'] = True
                session.permanent = True
                # Clear rate-limit history on successful login
                _LOGIN_ATTEMPTS.pop(client_ip, None)
                return redirect(url_for('index'))
            _record_failed_attempt(client_ip)
            error = 'Invalid username or password.'
    return render_template('login.html', version=APP_VERSION, error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/health')
def health():
    """Unauthenticated health-check endpoint for launcher scripts."""
    return jsonify({'status': 'ok'})

# --- WebUI metrics cache file (persists across restarts) ---
_WEBUI_CACHE_PATH = os.path.join(BASE_DIR, "webui_metrics_cache.json")
# Flush in-memory metrics to disk every N generations
_CACHE_FLUSH_INTERVAL = 5

# Maximum metrics history entries kept in memory / on disk.
# At one entry per generation this covers ~10,000 gens.
# Older entries are evicted (FIFO) on append.
_MAX_METRICS_HISTORY = 10_000

def _json_safe(obj):
    """Fallback serializer for json.dump — handles numpy / non-standard types."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return str(obj)

class AppState:
    def __init__(self):
        self.metrics_history = []
        self.is_training = False
        self.current_generation = 0
        self.current_fitness = 0.0
        self.current_compression_ratio = 0.0
        self.current_file_size = 0.0
        self.current_complexity_tier = 'UNKNOWN'
        self.current_complexity_value = 0
        self.current_tier_budget_mb = 0.0
        self.current_tier_ceiling_kb = 0.0
        self.current_best_robustness = 0.0
        self.current_training_phase = ''
        self.current_corruption_level = 0.0
        self.current_decomp_mismatches = 0
        self.current_items_evaluated = 0
        self.current_successful_compressions = 0
        self.current_method_stats = {}
        self.current_novel_pipeline = 'none'
        self.current_gold_standard_win_rate = -1.0
        self.start_time = None
        self.log_queue = queue.Queue(maxsize=2000)
        self.optimizer: Optional[EvolutionaryOptimizer] = None 
        self.stop_event: Optional[threading.Event] = None
        self._cached_snapshots: list = []  # Restored from disk cache
        # --- Run history ---
        self.run_number = 0              # Current run number (incremented on fresh start)
        self.run_history = []            # List of archived run summaries
        self.completed_naturally = False  # True when training ended without stop

    def reset(self):
        self.metrics_history = []
        self.current_generation = 0
        self.current_fitness = 0.0
        self.current_compression_ratio = 0.0
        self.current_complexity_tier = 'UNKNOWN'
        self.current_complexity_value = 0
        self.current_tier_budget_mb = 0.0
        self.current_tier_ceiling_kb = 0.0
        self.current_best_robustness = 0.0
        self.current_training_phase = ''
        self.current_corruption_level = 0.0
        self.current_decomp_mismatches = 0
        self.current_items_evaluated = 0
        self.current_successful_compressions = 0
        self.current_method_stats = {}
        self.current_novel_pipeline = 'none'
        self.current_gold_standard_win_rate = -1.0
        self.start_time = time.time()
        self.completed_naturally = False
        with self.log_queue.mutex:
            self.log_queue.queue.clear()

    def archive_run(self):
        """Archive current run's metrics before starting a fresh run."""
        if not self.metrics_history:
            return
        summary = {
            'run_number': self.run_number,
            'generations': len(self.metrics_history),
            'best_fitness': self.current_fitness,
            'best_ratio': self.current_compression_ratio,
            'final_tier': self.current_complexity_tier,
            'metrics': list(self.metrics_history),
        }
        self.run_history.append(summary)
        # Keep at most 10 archived runs to prevent memory bloat
        if len(self.run_history) > 10:
            self.run_history = self.run_history[-10:]

    # --- Disk cache for metrics & snapshots ---

    def save_cache(self):
        """Persist metrics_history and generation_snapshots to disk."""
        tmp_path = _WEBUI_CACHE_PATH + ".tmp"
        try:
            snapshots = []
            if self.optimizer:
                snapshots = getattr(self.optimizer, 'generation_snapshots', [])
            elif self._cached_snapshots:
                snapshots = self._cached_snapshots
            payload = {
                'metrics_history': self.metrics_history,
                'generation_snapshots': snapshots,
                'current_generation': self.current_generation,
                'current_fitness': self.current_fitness,
                'current_compression_ratio': self.current_compression_ratio,
            }
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, default=_json_safe)
            # Atomic replace (best effort on Windows)
            if os.path.exists(_WEBUI_CACHE_PATH):
                os.replace(tmp_path, _WEBUI_CACHE_PATH)
            else:
                os.rename(tmp_path, _WEBUI_CACHE_PATH)
        except Exception as exc:
            # Non-fatal — caching is best-effort
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_cache(self):
        """Restore cached metrics from a prior session, if available."""
        if not os.path.isfile(_WEBUI_CACHE_PATH):
            return
        try:
            with open(_WEBUI_CACHE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cached_metrics = data.get('metrics_history', [])
            cached_snapshots = data.get('generation_snapshots', [])
            if cached_metrics:
                self.metrics_history = cached_metrics
                self.current_generation = data.get('current_generation', 0)
                self.current_fitness = data.get('current_fitness', 0.0)
                self.current_compression_ratio = data.get('current_compression_ratio', 0.0)
            if cached_snapshots:
                self._cached_snapshots = cached_snapshots
                if self.optimizer:
                    self.optimizer.generation_snapshots = cached_snapshots
        except Exception:
            pass  # Corrupted cache — start fresh

app_state = AppState()
# Restore cached metrics from a prior session (if any)
app_state.load_cache()

@app.route('/')
def index(): 
    return render_template('index.html', version=APP_VERSION)

@app.route('/api/status')
def status():
    elapsed = time.time() - app_state.start_time if app_state.is_training and app_state.start_time else 0.0
    return jsonify({
        'is_training': app_state.is_training,
        'current_fitness': app_state.current_fitness,
        'current_generation': app_state.current_generation,
        'evolution_time': elapsed,
        'system_limits': SYSTEM_LIMITS,
        'run_number': app_state.run_number,
        'completed_naturally': app_state.completed_naturally,
        'can_continue': (not app_state.is_training 
                         and app_state.optimizer is not None),
        'run_history_count': len(app_state.run_history),
    })


@app.route('/api/hardware-profile')
def hardware_profile():
    """Return detected hardware profile and run presets for the dashboard."""
    return jsonify({
        'hardware': HARDWARE_PROFILE,
        'presets': RUN_PRESETS,
    })

@app.route('/api/metrics')
def metrics():
    if not app_state.metrics_history:
        return jsonify({
            'metrics': [], 'history': [], 
            'generation': 0, 'best_fitness': 0.0, 
            'compression_ratio': 0.0, 'benchmark_size': "0.0 MB",
            'complexity_tier': 'UNKNOWN', 'complexity_value': 0,
            'tier_budget_mb': 0.0, 'tier_ceiling_kb': 0.0
        })
    
    history_data = [[m['generation'], m.get('ratio', 0.0)] for m in app_state.metrics_history]
    bench_mb = (app_state.current_file_size / (1024*1024)) if app_state.current_file_size else 0.0
    
    return jsonify({
        'metrics': app_state.metrics_history,
        'history': history_data,
        'generation': app_state.current_generation,
        'best_fitness': app_state.current_fitness,
        'compression_ratio': app_state.current_compression_ratio,
        'benchmark_size': f"{bench_mb:.2f} MB",
        'complexity_tier': app_state.current_complexity_tier,
        'complexity_value': app_state.current_complexity_value,
        'tier_budget_mb': app_state.current_tier_budget_mb,
        'tier_ceiling_kb': app_state.current_tier_ceiling_kb,
        'best_robustness': app_state.current_best_robustness,
        'training_phase': app_state.current_training_phase,
        'corruption_level': app_state.current_corruption_level,
        'decomp_mismatches': app_state.current_decomp_mismatches,
        'items_evaluated': app_state.current_items_evaluated,
        'successful_compressions': app_state.current_successful_compressions,
        'method_stats': app_state.current_method_stats,
        'novel_pipeline': app_state.current_novel_pipeline,
        'gold_standard_win_rate': app_state.current_gold_standard_win_rate,
    })

@app.route('/api/population')
def population():
    if not app_state.optimizer:
        return jsonify({'population': []})
    
    try:
        raw_pop = getattr(app_state.optimizer, 'population', [])
        pop_data = []
        sorted_pop = sorted(
            [a for a in raw_pop if a is not None], 
            key=lambda x: (getattr(x, 'fitness', None) or -9999.0), 
            reverse=True
        )

        for agent in sorted_pop[:50]: 
            try:
                ai = getattr(agent, 'puffin_ai', None)
                fit = getattr(agent, 'fitness', None)
                thresholds = "N/A"
                if ai and hasattr(ai, 'len_thresholds'):
                    t = ai.len_thresholds
                    if len(t) > 8: thresholds = f"{len(t)} thresholds"
                    else: thresholds = ", ".join(map(str, t))

                pop_data.append({
                    'id': getattr(agent, 'agent_id', 'Unknown'),
                    'fitness': fit if fit is not None and fit > -999 else "Pending...",
                    'gen_born': getattr(agent, 'generation_born', 0),
                    'thresholds': thresholds
                })
            except: continue
        return jsonify({'population': pop_data})
    except Exception as e:
        return jsonify({'population': [], 'error': str(e)})

@app.route('/api/population/history')
def population_history():
    """Return paginated generation snapshots for the population viewer.
    
    Query params:
        page (int): 1-based page number (default 1)
        per_page (int): generations per page (default 20)
    """
    if not app_state.optimizer:
        # Fall back to cached snapshots from a prior session
        snapshots = app_state._cached_snapshots or []
        if not snapshots:
            return jsonify({'generations': [], 'total_gens': 0, 'page': 1, 'total_pages': 1})
    else:
        snapshots = getattr(app_state.optimizer, 'generation_snapshots', [])
    
    total = len(snapshots)
    
    per_page = min(100, max(1, int(request.args.get('per_page', 20))))
    total_pages = max(1, math.ceil(total / per_page))
    page = max(1, min(int(request.args.get('page', 1)), total_pages))
    
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    page_snapshots = snapshots[start:end]
    
    # Also include the live population as the current generation (if training)
    live_gen = None
    if app_state.is_training:
        try:
            raw_pop = getattr(app_state.optimizer, 'population', [])
            sorted_pop = sorted(
                [a for a in raw_pop if a is not None],
                key=lambda x: (getattr(x, 'fitness', None) or -9999.0),
                reverse=True
            )
            live_agents = []
            for agent in sorted_pop[:50]:
                ai = getattr(agent, 'puffin_ai', None)
                fit = getattr(agent, 'fitness', None)
                thresholds = "N/A"
                if ai and hasattr(ai, 'len_thresholds'):
                    t = ai.len_thresholds
                    if len(t) > 8: thresholds = f"{len(t)} thresholds"
                    else: thresholds = ", ".join(map(str, t))
                live_agents.append({
                    'agent_id': getattr(agent, 'agent_id', 'Unknown'),
                    'fitness': fit if fit is not None and fit > -999 else None,
                    'generation_born': getattr(agent, 'generation_born', 0),
                    'thresholds_str': thresholds,
                    'evaluation_stats': dict(getattr(agent, 'evaluation_stats', {}) or {}),
                })
            # Compute live avg_fitness excluding catastrophic failures (< -50)
            live_fitnesses = [a['fitness'] for a in live_agents
                              if a['fitness'] is not None and a['fitness'] > -50.0]
            live_avg = (sum(live_fitnesses) / len(live_fitnesses)) if live_fitnesses else 0.0
            # The live population is being worked on for the NEXT
            # generation (current_generation was the last completed one).
            live_gen_num = (app_state.current_generation or 0) + 1
            live_gen = {
                'generation': live_gen_num,
                'is_live': True,
                'agent_count': len(live_agents),
                'best_fitness': app_state.current_fitness,
                'avg_fitness': live_avg,
                'batches': [{'batch_idx': 0, 'agents': live_agents}]
            }
        except Exception:
            pass
    
    return jsonify({
        'generations': page_snapshots,
        'live_generation': live_gen,
        'total_gens': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages
    })

@app.route('/api/logs')
def logs():
    l = []
    try:
        for _ in range(50):
            if app_state.log_queue.empty(): break
            l.append(app_state.log_queue.get_nowait())
    except: pass
    return jsonify(l)

@app.route('/api/compression-methods')
def methods():
    data = []
    if get_hybrid_engine:
        try:
            for n in get_hybrid_engine().registry.methods.keys():
                data.append(n.replace('_',' ').title())
        except: pass
    return jsonify(data if data else ["Standard Methods"])

@app.route('/api/training/start', methods=['POST'])
def start():
    if app_state.is_training: return jsonify({'success': False})
    
    # Archive previous run before resetting (if there was one)
    app_state.archive_run()
    app_state.run_number += 1
    
    config = request.json or {}
    pop_size = min(int(config.get('population_size', SYSTEM_LIMITS['default_pop'])), SYSTEM_LIMITS['max_pop'])
    num_gens = min(int(config.get('num_generations', SYSTEM_LIMITS['default_gens'])), SYSTEM_LIMITS['max_gens'])
    batch_size = max(1, min(int(config.get('batch_size', 10)), pop_size))
    infinite_mode = bool(config.get('infinite', False))
    target_device = config.get('target_device', 'GPU_AUTO')
    cpu_workers = max(1, min(int(config.get('cpu_workers', 4)), 32))

    def run_thread(p, g, batch_sz=10, infinite=False, device='GPU_AUTO', workers=4):
        app_state.is_training = True
        app_state.reset()
        mode_str = "INFINITE" if infinite else f"{g}"
        if _debug_mode:
            print(f"DEBUG: Thread started. Pop: {p}, Gens: {mode_str}, Batch: {batch_sz}")
        app_state.log_queue.put({'level': 'INFO', 'message': f'Initializing (Pop: {p}, Gens: {mode_str}, Batch: {batch_sz})...'})

        class Bridge:
            def put_nowait(self, item):
                if not isinstance(item, str): return
                if item.startswith("METRICS_JSON:"):
                    try:
                        data = json.loads(item.replace("METRICS_JSON:", "", 1))
                        app_state.current_generation = data.get('generation')
                        app_state.current_fitness = data.get('fitness')
                        app_state.current_compression_ratio = data.get('ratio')
                        app_state.current_file_size = data.get('benchmark_size')
                        app_state.current_complexity_tier = data.get('complexity_tier', 'UNKNOWN')
                        app_state.current_complexity_value = data.get('complexity_value', 0)
                        app_state.current_tier_budget_mb = data.get('tier_budget_mb', 0.0)
                        app_state.current_tier_ceiling_kb = data.get('tier_ceiling_kb', 0.0)
                        app_state.current_best_robustness = data.get('best_robustness', 0.0)
                        app_state.current_training_phase = data.get('training_phase', '')
                        app_state.current_corruption_level = data.get('corruption_level', 0.0)
                        app_state.current_decomp_mismatches = data.get('decomp_mismatches', 0)
                        app_state.current_items_evaluated = data.get('items_evaluated', 0)
                        app_state.current_successful_compressions = data.get('successful_compressions', 0)
                        app_state.current_method_stats = data.get('method_stats', {})
                        app_state.current_novel_pipeline = data.get('novel_pipeline', 'none')
                        app_state.current_gold_standard_win_rate = data.get('gold_standard_win_rate', -1.0)
                        app_state.metrics_history.append({
                            'generation': app_state.current_generation,
                            'fitness': app_state.current_fitness,
                            'ratio': app_state.current_compression_ratio,
                            'benchmark_size': app_state.current_file_size,
                            'complexity_tier': app_state.current_complexity_tier,
                            'complexity_value': app_state.current_complexity_value,
                            'tier_budget_mb': app_state.current_tier_budget_mb,
                            'tier_ceiling_kb': app_state.current_tier_ceiling_kb,
                            'best_robustness': app_state.current_best_robustness,
                            'training_phase': app_state.current_training_phase,
                            'corruption_level': app_state.current_corruption_level,
                            'decomp_mismatches': app_state.current_decomp_mismatches,
                            'items_evaluated': app_state.current_items_evaluated,
                            'successful_compressions': app_state.current_successful_compressions,
                            'method_stats': app_state.current_method_stats,
                            'novel_pipeline': app_state.current_novel_pipeline,
                            'gold_standard_win_rate': app_state.current_gold_standard_win_rate,
                        })
                        # Cap history length so infinite runs don't exhaust RAM.
                        if len(app_state.metrics_history) > _MAX_METRICS_HISTORY:
                            app_state.metrics_history = app_state.metrics_history[-_MAX_METRICS_HISTORY:]
                        # Periodic flush to disk so data survives restarts
                        gen = app_state.current_generation or 0
                        if gen > 0 and gen % _CACHE_FLUSH_INTERVAL == 0:
                            app_state.save_cache()
                    except: pass
                    return
                clean = item.replace(ELS_LOG_PREFIX, "").strip()
                if not clean: return
                level = 'ERROR' if 'Error' in clean else 'INFO'
                try: app_state.log_queue.put_nowait({'level': level, 'message': clean})
                except: pass

        try:
            if _EvolutionaryOptimizerClass is None:
                raise ImportError("EvolutionaryOptimizer failed to load.")

            t_opt_init = time.perf_counter()
            opt = _EvolutionaryOptimizerClass(
                population_size=p,
                num_generations=g,
                gui_output_queue=Bridge(),
                gui_stop_event=threading.Event(),
                target_device=device,
                dynamic_benchmarking_active=True,
                infinite_mode=infinite,
                population_batch_size=batch_sz,
                cpu_eval_workers=workers
            )
            opt_init_ms = (time.perf_counter() - t_opt_init) * 1000
            if _debug_mode:
                print(f"DEBUG-TIMING: EvolutionaryOptimizer.__init__() took {opt_init_ms:.0f}ms")
                if opt_init_ms > 5000:
                    print(f"DEBUG-TIMING: *** SLOW *** Optimizer init > 5s ({opt_init_ms:.0f}ms) — check subsystem breakdown above")
            app_state.optimizer = opt
            app_state.stop_event = opt.gui_stop_event
            
            if _debug_mode:
                print("DEBUG: Calling start_evolution...")
            opt.start_evolution()
            if _debug_mode:
                print("DEBUG: start_evolution finished normally.")
            app_state.completed_naturally = not app_state.stop_event.is_set() if app_state.stop_event else True
            app_state.log_queue.put({'level': 'SUCCESS', 'message': 'Evolution Finished. Press Continue to extend, or Start for a new run.'})
        except Exception as e:
            err_msg = f"CRITICAL THREAD ERROR: {e}"
            print(err_msg)
            traceback.print_exc()
            app_state.log_queue.put({'level': 'ERROR', 'message': err_msg})
        finally:
            # Final flush of metrics to disk before thread exits
            app_state.save_cache()
            app_state.is_training = False
            print("DEBUG: Thread exiting, is_training set to False.")

    t = threading.Thread(target=run_thread, args=(pop_size, num_gens, batch_size, infinite_mode, target_device, cpu_workers), daemon=True)
    t.start()
    return jsonify({'success': True})

@app.route('/api/training/stop', methods=['POST'])
def stop():
    app_state.is_training = False
    app_state.completed_naturally = False
    if app_state.stop_event: app_state.stop_event.set()
    return jsonify({'success': True})

@app.route('/api/training/continue', methods=['POST'])
def continue_training():
    """Continue evolution from where it left off, reusing the existing optimizer and population.
    
    Extends initial_num_generations by the requested amount (default: 100).
    Optionally switches to infinite mode.
    Does NOT reset metrics or population — the chart continues from the last generation.
    """
    if app_state.is_training:
        return jsonify({'success': False, 'error': 'Training is already running'})
    if not app_state.optimizer:
        return jsonify({'success': False, 'error': 'No previous run to continue — use Start instead'})
    
    config = request.json or {}
    extra_gens = max(1, min(int(config.get('extra_generations', 100)), SYSTEM_LIMITS['max_gens']))
    switch_infinite = bool(config.get('infinite', False))
    
    opt = app_state.optimizer
    
    # Clear the stop event so the loop can resume
    if opt.gui_stop_event:
        opt.gui_stop_event.clear()
    app_state.stop_event = opt.gui_stop_event
    
    mode_str = "INFINITE" if switch_infinite else f"+{extra_gens} (total target {opt.total_generations_elapsed + extra_gens})"
    app_state.log_queue.put({'level': 'INFO', 'message': f'Continuing evolution: {mode_str} from gen {opt.total_generations_elapsed}...'})
    
    def continue_thread():
        app_state.is_training = True
        app_state.completed_naturally = False
        app_state.start_time = time.time()
        try:
            opt.continue_evolution(additional_gens=extra_gens, switch_infinite=switch_infinite)
            app_state.completed_naturally = not opt.gui_stop_event.is_set()
            app_state.log_queue.put({'level': 'SUCCESS', 'message': 'Evolution Finished. Press Continue to extend, or Start for a new run.'})
        except Exception as e:
            err_msg = f"CRITICAL CONTINUE ERROR: {e}"
            print(err_msg)
            traceback.print_exc()
            app_state.log_queue.put({'level': 'ERROR', 'message': err_msg})
        finally:
            app_state.save_cache()
            app_state.is_training = False
    
    t = threading.Thread(target=continue_thread, daemon=True)
    t.start()
    return jsonify({'success': True, 'mode': mode_str})

@app.route('/api/training/run-history')
def run_history():
    """Return archived run summaries (metrics only, not full history arrays for large runs)."""
    summaries = []
    for run in app_state.run_history:
        summaries.append({
            'run_number': run.get('run_number', 0),
            'generations': run.get('generations', 0),
            'best_fitness': run.get('best_fitness', 0),
            'best_ratio': run.get('best_ratio', 0),
            'final_tier': run.get('final_tier', 'UNKNOWN'),
        })
    return jsonify({'runs': summaries, 'current_run': app_state.run_number})

# --- CHECKPOINT API ---
def _get_checkpoint_manager():
    """Get checkpoint manager from optimizer, or create a standalone one for listing."""
    if app_state.optimizer and getattr(app_state.optimizer, 'checkpoint_manager', None):
        return app_state.optimizer.checkpoint_manager
    # Fallback: create a read-only manager to list checkpoints from disk
    try:
        from puffinzip_ai.checkpoint_manager import CheckpointManager
        from puffinzip_ai.config import LOGS_DIR_PATH
        cp_dir = os.path.join(os.path.dirname(LOGS_DIR_PATH), "checkpoints")
        return CheckpointManager(checkpoint_dir=cp_dir)
    except Exception:
        return None

@app.route('/api/evolution/deep-dive')
def evolution_deep_dive():
    """Return enriched generation data for the Neural Network & Evolution deep-dive tab.
    
    Includes per-generation agent data with parent lineage, gene-pool clustering,
    fitness distributions, and breeding relationships.
    
    Query params:
        page (int): 1-based page number (default 1)
        per_page (int): generations per page (default 20)
    """
    if not app_state.optimizer:
        # Fall back to cached snapshots from a prior session
        snapshots = app_state._cached_snapshots or []
        if not snapshots:
            return jsonify({'generations': [], 'gene_pools': {}, 'top_agents': [],
                            'total_gens': 0, 'page': 1, 'total_pages': 1,
                            'mutation_rate': 0.0, 'stagnation_counter': 0})
    else:
        snapshots = getattr(app_state.optimizer, 'generation_snapshots', [])
    
    total = len(snapshots)
    
    per_page = min(100, max(1, int(request.args.get('per_page', 20))))
    total_pages = max(1, math.ceil(total / per_page))
    page = max(1, min(int(request.args.get('page', 1)), total_pages))
    
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    page_snapshots = snapshots[start:end]
    
    # --- Build gene-pool clustering from parent lineage ---
    # Trace each agent back to its root ancestor to assign gene-pool membership.
    # Agents sharing the same root ancestor belong to the same gene pool.
    all_agents_by_id = {}  # agent_id -> agent summary dict
    parent_map = {}        # agent_id -> list of parent_ids
    
    for snap in snapshots:
        for batch in snap.get('batches', []):
            for agent in batch.get('agents', []):
                aid = agent.get('agent_id', '')
                all_agents_by_id[aid] = agent
                pids = agent.get('parent_ids', [])
                parent_map[aid] = pids
    
    # Also include live population if training
    if app_state.is_training:
        try:
            raw_pop = getattr(app_state.optimizer, 'population', [])
            for agent in raw_pop:
                if agent is None:
                    continue
                aid = getattr(agent, 'agent_id', '')
                pids = list(getattr(agent, 'parent_ids', []))
                parent_map[aid] = pids
        except Exception:
            pass
    
    # Find root ancestor for each agent (BFS up the parent tree)
    root_cache = {}
    def find_root(agent_id, visited=None):
        if agent_id in root_cache:
            return root_cache[agent_id]
        if visited is None:
            visited = set()
        if agent_id in visited:
            return agent_id  # cycle guard
        visited.add(agent_id)
        parents = parent_map.get(agent_id, [])
        if not parents:
            root_cache[agent_id] = agent_id
            return agent_id
        root = find_root(parents[0], visited)
        root_cache[agent_id] = root
        return root
    
    # Build gene_pools: root_ancestor_id -> list of member agent_ids
    gene_pools = {}
    for aid in parent_map:
        root = find_root(aid)
        gene_pools.setdefault(root, []).append(aid)
    
    # Assign stable color indices to gene pools (largest pools first)
    sorted_pools = sorted(gene_pools.items(), key=lambda x: -len(x[1]))
    pool_color_map = {}  # agent_id -> pool_index
    for idx, (root, members) in enumerate(sorted_pools):
        for mid in members:
            pool_color_map[mid] = idx
    
    # --- Build agent fitness lookup across all snapshots ---
    agent_fitness_map = {}  # agent_id -> fitness
    for snap in snapshots:
        for batch in snap.get('batches', []):
            for agent in batch.get('agents', []):
                aid = agent.get('agent_id', '')
                fit = agent.get('fitness')
                if aid and fit is not None:
                    agent_fitness_map[aid] = fit

    # --- Track how often each agent is selected as a parent (top breeders) ---
    breeder_frequency = {}  # agent_id -> count of times selected as parent

    # --- Enrich generation data ---
    enriched_gens = []
    for snap in page_snapshots:
        gen_agents = []
        breeding_pairs = []  # enriched with fitness & pool info
        mutation_count = 0
        # Pool-to-pool breeding matrix for this generation
        pool_breed_matrix = {}  # "poolA→poolB" -> count

        # Collect compression parameter values for method direction tracking
        rle_values = []
        threshold_counts = {}  # threshold_bucket -> count

        for batch in snap.get('batches', []):
            for agent in batch.get('agents', []):
                aid = agent.get('agent_id', '')
                pids = agent.get('parent_ids', [])

                enriched_agent = {
                    'agent_id': aid,
                    'fitness': agent.get('fitness'),
                    'generation_born': agent.get('generation_born', 0),
                    'parent_ids': pids,
                    'pool_index': pool_color_map.get(aid, 0),
                    'learning_rate': agent.get('learning_rate', 0.0),
                    'exploration_rate': agent.get('exploration_rate', 0.0),
                    'rle_min_run': agent.get('rle_min_run', 'N/A'),
                    'thresholds_str': agent.get('thresholds_str', 'N/A'),
                    'is_elite': 'elite' in aid.lower() if aid else False,
                    'has_novel_method': agent.get('has_novel_method', False),
                    'novel_pipeline': agent.get('novel_pipeline', 'none'),
                }
                gen_agents.append(enriched_agent)

                # Track compression parameters
                rle_val = agent.get('rle_min_run', 'N/A')
                if rle_val != 'N/A':
                    try:
                        rle_values.append(float(rle_val))
                    except (ValueError, TypeError):
                        pass
                thr_str = agent.get('thresholds_str', 'N/A')
                if thr_str and thr_str != 'N/A':
                    threshold_counts[thr_str] = threshold_counts.get(thr_str, 0) + 1

                if len(pids) >= 2:
                    p1_id, p2_id = pids[0], pids[1]
                    p1_fit = agent_fitness_map.get(p1_id)
                    p2_fit = agent_fitness_map.get(p2_id)
                    child_fit = agent.get('fitness')
                    p1_pool = pool_color_map.get(p1_id, 0)
                    p2_pool = pool_color_map.get(p2_id, 0)
                    child_pool = pool_color_map.get(aid, 0)

                    breeding_pairs.append({
                        'parent1': p1_id,
                        'parent2': p2_id,
                        'child': aid,
                        'parent1_fitness': p1_fit,
                        'parent2_fitness': p2_fit,
                        'child_fitness': child_fit,
                        'parent1_pool': p1_pool,
                        'parent2_pool': p2_pool,
                        'child_pool': child_pool,
                        'cross_pool': p1_pool != p2_pool,
                    })

                    # Pool breeding matrix
                    key = f"{min(p1_pool, p2_pool)}-{max(p1_pool, p2_pool)}"
                    pool_breed_matrix[key] = pool_breed_matrix.get(key, 0) + 1

                    # Breeder frequency
                    breeder_frequency[p1_id] = breeder_frequency.get(p1_id, 0) + 1
                    breeder_frequency[p2_id] = breeder_frequency.get(p2_id, 0) + 1

                if len(pids) == 1:
                    mutation_count += 1  # cloned (likely mutated)

        # Sort agents by fitness descending
        gen_agents.sort(key=lambda a: a.get('fitness') or -9999, reverse=True)

        # Cross-pool breeding stats
        cross_pool_count = sum(1 for bp in breeding_pairs if bp.get('cross_pool'))

        # Method direction: most common threshold config
        top_threshold = max(threshold_counts, key=lambda k: threshold_counts[k]) if threshold_counts else 'N/A'

        enriched_gens.append({
            'generation': snap.get('generation', 0),
            'best_fitness': snap.get('best_fitness', 0.0),
            'avg_fitness': snap.get('avg_fitness', 0.0),
            'min_fitness': snap.get('min_fitness', 0.0),
            'agent_count': snap.get('agent_count', 0),
            'agents': gen_agents,
            'breeding_pairs': breeding_pairs,
            'mutation_count': mutation_count,
            'crossover_count': len(breeding_pairs),
            'cross_pool_breeding': cross_pool_count,
            'pool_breed_matrix': pool_breed_matrix,
            'avg_rle_min_run': round(sum(rle_values) / len(rle_values), 3) if rle_values else None,
            'top_threshold_config': top_threshold,
            'novel_method_count': sum(1 for a in gen_agents if a.get('has_novel_method')),
        })
    
    # --- Top agents overall (across all generations) ---
    top_agents = sorted(
        all_agents_by_id.values(),
        key=lambda a: a.get('fitness') or -9999,
        reverse=True
    )[:20]
    for ta in top_agents:
        ta['pool_index'] = pool_color_map.get(ta.get('agent_id', ''), 0)
    
    # --- Top breeders (agents selected as parents most frequently) ---
    top_breeders = sorted(
        breeder_frequency.items(),
        key=lambda x: -x[1]
    )[:10]
    top_breeders_list = []
    for bid, count in top_breeders:
        top_breeders_list.append({
            'agent_id': bid,
            'breed_count': count,
            'fitness': agent_fitness_map.get(bid),
            'pool_index': pool_color_map.get(bid, 0),
        })

    # --- Summary stats ---
    mutation_rate = getattr(app_state.optimizer, 'base_mutation_rate', 0.0)
    stagnation = getattr(app_state.optimizer, '_stagnation_counter', 0)

    # Gene pool summary (top 12 pools by size)
    pool_summary = {}
    for idx, (root, members) in enumerate(sorted_pools[:12]):
        pool_summary[str(idx)] = {
            'root_ancestor': root,
            'size': len(members),
            'index': idx,
        }

    # --- Method direction summary across visible generations ---
    method_direction = []
    for eg in enriched_gens:
        method_direction.append({
            'generation': eg['generation'],
            'avg_rle_min_run': eg.get('avg_rle_min_run'),
            'novel_method_count': eg.get('novel_method_count', 0),
            'cross_pool_breeding': eg.get('cross_pool_breeding', 0),
            'top_threshold_config': eg.get('top_threshold_config', 'N/A'),
        })

    return jsonify({
        'generations': enriched_gens,
        'gene_pools': pool_summary,
        'top_agents': top_agents,
        'top_breeders': top_breeders_list,
        'method_direction': method_direction,
        'total_gens': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages,
        'mutation_rate': mutation_rate,
        'stagnation_counter': stagnation,
        'total_pool_count': len(gene_pools),
    })

@app.route('/api/checkpoint/save', methods=['POST'])
def save_checkpoint():
    if not app_state.optimizer:
        return jsonify({'success': False, 'error': 'No active optimizer. Start training first.'})
    name = (request.json or {}).get('name', '').strip()
    if not name:
        import datetime
        name = f"checkpoint_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        ok = app_state.optimizer.save_checkpoint(name)
        if ok:
            return jsonify({'success': True, 'name': name})
        else:
            return jsonify({'success': False, 'error': 'Checkpoint save failed (see server logs).'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/checkpoint/list')
def list_checkpoints():
    cm = _get_checkpoint_manager()
    if not cm:
        return jsonify({'checkpoints': []})
    try:
        items = cm.list_checkpoints()
        return jsonify({'checkpoints': items})
    except Exception as e:
        return jsonify({'checkpoints': [], 'error': str(e)})

@app.route('/api/checkpoint/test', methods=['POST'])
def test_checkpoint():
    """Test a saved checkpoint by compressing an uploaded file with the best agent."""
    checkpoint_name = request.form.get('checkpoint', '').strip()
    uploaded_file = request.files.get('file')

    if not checkpoint_name:
        return jsonify({'error': 'No checkpoint name provided.'})
    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({'error': 'No test file uploaded.'})

    cm = _get_checkpoint_manager()
    if not cm:
        return jsonify({'error': 'Checkpoint manager not available.'})

    # File size guard — reject uploads over 10 MB to avoid memory issues
    MAX_TEST_FILE_BYTES = 10 * 1024 * 1024
    uploaded_file.seek(0, 2)  # seek to end
    file_len = uploaded_file.tell()
    uploaded_file.seek(0)
    if file_len > MAX_TEST_FILE_BYTES:
        return jsonify({'error': f'File too large ({file_len / (1024*1024):.1f} MB). Max is 10 MB.'})

    try:
        # Find the matching checkpoint key (keys are name_timestamp format)
        matching_key = None
        for key in cm.checkpoints_metadata:
            meta = cm.checkpoints_metadata[key]
            if meta.name == checkpoint_name or key == checkpoint_name:
                matching_key = key
                break

        if not matching_key:
            return jsonify({'error': f"Checkpoint '{checkpoint_name}' not found."})

        # Load the checkpoint
        success, optimizer_state = cm.load_checkpoint(matching_key)
        if not success or not optimizer_state:
            return jsonify({'error': f"Failed to load checkpoint '{checkpoint_name}'."})

        # Read the uploaded file content
        file_bytes = uploaded_file.read()
        try:
            file_text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            file_text = file_bytes.decode('latin-1')

        original_size = len(file_text)
        if original_size == 0:
            return jsonify({'error': 'Uploaded file is empty.'})

        # Extract the best agent from the population and compress
        population = optimizer_state.get('population', [])
        if not population:
            return jsonify({'error': 'Checkpoint contains no population data.'})

        # Sort by fitness to find the best agent
        best_individual = max(population, key=lambda ind: getattr(ind, 'fitness', 0.0) or 0.0)
        agent_fitness = getattr(best_individual, 'fitness', 0.0) or 0.0
        ai_core = (getattr(best_individual, 'puffin_ai', None)
                   or getattr(best_individual, 'ai_core', None)
                   or getattr(best_individual, 'agent', None))

        if ai_core is None:
            return jsonify({'error': 'Could not extract AI agent from checkpoint.'})

        # Use the agent to decide a compression method and compress
        from puffinzip_ai.rle_utils import rle_compress, rle_decompress
        state_idx = ai_core._get_state_representation(file_text)
        action_idx = ai_core._choose_action(state_idx, use_exploration=False)
        action_name = ai_core.action_names[action_idx] if action_idx < len(ai_core.action_names) else f"action_{action_idx}"

        t_start = time.time()

        if action_idx == 0:
            compressed = rle_compress(file_text, method="simple",
                                      min_run_len_override=ai_core.rle_min_encodable_run_length)
            method_used = f"SimpleRLE (min_run={ai_core.rle_min_encodable_run_length})"
            decompressed = rle_decompress(compressed, method="simple",
                                          min_run_len_override=ai_core.rle_min_encodable_run_length)
        elif action_idx == 2:
            compressed = rle_compress(file_text, method="advanced")
            method_used = "AdvancedRLE"
            decompressed = rle_decompress(compressed, method="advanced")
        elif action_idx == 3 and hasattr(ai_core, '_novel_compress_fn') and ai_core._novel_compress_fn:
            compressed = ai_core._novel_compress_fn(file_text)
            method_used = "NovelCompression"
            if hasattr(ai_core, '_novel_decompress_fn') and ai_core._novel_decompress_fn:
                decompressed = ai_core._novel_decompress_fn(compressed)
            else:
                decompressed = None
        else:
            compressed = file_text
            method_used = "NoCompression"
            decompressed = file_text

        elapsed_ms = (time.time() - t_start) * 1000

        compressed_size = len(compressed)
        ratio_val = compressed_size / original_size if original_size > 0 else 1.0
        savings_pct = (1 - ratio_val) * 100

        # Round-trip integrity check
        if decompressed is not None:
            integrity = 'PASS' if decompressed == file_text else 'FAIL'
        else:
            integrity = 'SKIP (no decompressor)'

        # Checkpoint metadata for context
        cp_meta = cm.checkpoints_metadata.get(matching_key)
        cp_generation = cp_meta.generation if cp_meta else '?'

        return jsonify({
            'original_size': f"{original_size:,} bytes",
            'compressed_size': f"{compressed_size:,} bytes",
            'ratio': f"{ratio_val:.4f}",
            'savings_pct': f"{savings_pct:+.2f}%",
            'method': f"{action_name} \u2192 {method_used}",
            'time_ms': f"{elapsed_ms:.1f} ms",
            'integrity': integrity,
            'agent_fitness': f"{agent_fitness:.4f}",
            'checkpoint_generation': cp_generation,
        })

    except Exception as e:
        logging.getLogger(__name__).error(f"Checkpoint test failed: {e}", exc_info=True)
        return jsonify({'error': f'Test failed: {str(e)}'})

@app.route('/api/checkpoint/delete', methods=['POST'])
def delete_checkpoint():
    """Delete a saved checkpoint by key or name."""
    data = request.json or {}
    checkpoint_id = data.get('key', '').strip() or data.get('name', '').strip()
    if not checkpoint_id:
        return jsonify({'success': False, 'error': 'No checkpoint key or name provided.'})

    cm = _get_checkpoint_manager()
    if not cm:
        return jsonify({'success': False, 'error': 'Checkpoint manager not available.'})

    # Resolve by name if key not found directly
    if checkpoint_id not in cm.checkpoints_metadata:
        for key, meta in cm.checkpoints_metadata.items():
            if meta.name == checkpoint_id:
                checkpoint_id = key
                break

    try:
        ok = cm.delete_checkpoint(checkpoint_id)
        if ok:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': f"Checkpoint '{checkpoint_id}' not found or delete failed."})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    multiprocessing.freeze_support()
    import argparse
    parser = argparse.ArgumentParser(description='PuffinZipAI Web UI Server')
    _default_host = '0.0.0.0' if _PUBLIC_ACCESS else '127.0.0.1'
    parser.add_argument('--host', default=_default_host, help='Host to bind to (auto-set from credentials public_access)')
    parser.add_argument('--port', type=int, default=5001, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (verbose logging, Flask debugger)')
    args = parser.parse_args()

    if args.debug:
        _debug_mode = True
        # Propagate debug flag to puffinzip_ai config so optimizer/evaluator also print
        try:
            import puffinzip_ai.config as _cfg
            _cfg.DEBUG_LOG_CONSOLE_OUTPUT_ENABLED = True
        except Exception:
            pass
        # Debug mode: show all Flask/Werkzeug request logs + enable Flask debugger
        print("="*60)
        print("  DEBUG MODE ENABLED")
        print("  - Flask debugger: ON")
        print("  - Werkzeug request logs: ON")
        print("  - DEBUG-TIMING lines will appear below")
        print("="*60)
        logging.getLogger('werkzeug').setLevel(logging.DEBUG)
    else:
        # Normal mode: silence repetitive "GET /api/status 200" messages
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # Detect connect URL: RunPod proxy > public IP > local
    _connect_url = ''
    _platform = ''
    _runpod_id = os.environ.get('RUNPOD_POD_ID', '').strip()
    if _runpod_id:
        _platform = 'RunPod'
        _connect_url = f'https://{_runpod_id}-{args.port}.proxy.runpod.net'
    elif _PUBLIC_ACCESS:
        import urllib.request
        for _ip_url in ('https://api.ipify.org', 'https://ifconfig.me', 'https://icanhazip.com'):
            try:
                with urllib.request.urlopen(_ip_url, timeout=3) as _resp:
                    _pub = _resp.read().decode().strip()
                    if _pub:
                        _connect_url = f'http://{_pub}:{args.port}'
                        break
            except Exception:
                continue

    print(f">>> [SUCCESS] PuffinZipAI {APP_VERSION} loaded correctly.")
    if _CUSTOM_URL:
        print(f"--- CUSTOM URL: {_CUSTOM_URL} ---")
    if _connect_url:
        print(f"--- CONNECT URL: {_connect_url} ---")
    print(f"--- SERVER READY: http://{args.host}:{args.port} ---")
    print(f"--- Debug mode: {'ON' if args.debug else 'OFF'} ---")
    print(f"--- Auth: ENABLED ---")
    print(f"--- Public access: {'ON (0.0.0.0)' if _PUBLIC_ACCESS else 'OFF (127.0.0.1 — local only)'} ---")
    if _platform:
        print(f"--- Platform: {_platform} ---")
    print(f"--- Credentials file: {_CREDENTIALS_FILE} ---")
    print(f"--- Username: {_AUTH_USERNAME} ---")
    print(f"--- Password: {_AUTH_PASSWORD} ---")
    if _ADMIN_AUTH_ENABLED:
        print(f"--- Admin login: ENABLED (user: {_ADMIN_USERNAME}) ---")
    else:
        print(f"--- Admin login: DISABLED (set admin_username/admin_password in credentials file or .env) ---")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)