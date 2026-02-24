# PuffinZipAI — Complete Codebase Index

**Project**: PuffinZipAI v0.9.8  
**Description**: AI-driven compression system using Q-learning agents with evolutionary optimization  
**Last Updated**: 2026-02-25

---

## Directory Structure

```
PuffinZipAI/
├── main_cli.py                     # CLI entry point
├── run_gui.py                      # GUI launcher
├── webui_server.py                 # Flask web UI server
├── webui_credentials_manager.py    # Auto-generates & persists WebUI credentials (primary + admin)
├── webui_theme_manager.py          # Cross-platform theme system
├── .env.example                    # Template for local-only .env config (git-tracked)
├── requirements.txt
├── README.md                       # Project overview & quick start
├── LICENSE                         # PolyForm Noncommercial 1.0.0
├── PuffinZip.ico
│
├── puffinzip_ai/                   # Core AI/compression package
│   ├── __init__.py                 # Package init, auto-config, GPU/CPU selection
│   ├── config.py                   # Central configuration
│   ├── logger.py                   # Logging setup
│   ├── ai_core.py                  # Q-learning AI agent (PuffinZipAI)
│   ├── reward_system.py            # Multi-signal reward with progressive difficulty & competitive benchmarking
│   ├── rle_constants.py            # RLE error codes & safety constants
│   ├── rle_utils.py                # Simple RLE compression/decompression
│   ├── advanced_rle_methods.py     # Advanced RLE (lower min run)
│   ├── novel_compression_generator.py  # Invertible compression primitives & pipelines
│   ├── compression_scaffolding.py      # Reference method scaffolding (training wheels) with ban/penalty mechanics
│   ├── compression_method_registry.py  # Dynamic method registry
│   ├── hybrid_compression_engine.py    # Unified compression interface
│   ├── compression_benchmark.py    # A100-ready benchmark suite vs gzip/bz2/lzma/zlib/zstd + validation tiers
│   ├── gold_standard_benchmark.py  # Head-to-head benchmark after each gen; gold-standard checkpoints + failure diagnostics
│   ├── checkpoint_manager.py       # Save/load/compare checkpoints
│   ├── rust_compression_interface.py   # Rust interface with Python fallbacks
│   ├── rust_compression_stub.rs    # Rust stub source
│   │
│   ├── evolution_core/             # Evolutionary optimization
│   │   ├── evolutionary_optimizer.py   # Main evolution loop
│   │   ├── individual_agent.py     # EvolvingAgent wrapper with heritage lineage & agent types
│   │   ├── breeding_methods.py     # Advanced crossover
│   │   ├── crossover_methods.py    # Q-table & parameter crossover
│   │   ├── mutation_methods.py     # Mutation operators
│   │   └── selection_methods.py    # Parent selection strategies
│   │
│   ├── gpu_core/                   # GPU acceleration (CuPy/Numba)
│   │   ├── gpu_ai_agent.py         # GPU-accelerated AI agent
│   │   ├── gpu_model_utils.py      # GPU array transfer utilities
│   │   ├── gpu_rle_interface.py    # GPU RLE compression
│   │   ├── gpu_training_utils.py   # Batch Q-table ops on GPU
│   │   └── potential_cuda_kernels.cu
│   │
│   ├── nn_core/                    # Neural network (Dueling DQN) agents (PyTorch)
│   │   ├── __init__.py             # Package init, TORCH_AVAILABLE flag
│   │   ├── dqn_model.py            # Dueling DQN + NoisyNet + Attention + Feature Gate
│   │   ├── replay_buffer.py        # Prioritized Experience Replay (PER) with SumTree
│   │   └── nn_agent.py             # PuffinZipAI_NN agent (20-dim features, Double DQN, N-step)
│   │
│   └── utils/                      # Utility modules
│       ├── benchmark_evaluator.py  # Population fitness evaluation
│       ├── benchmark_generator.py  # Benchmark dataset generation
│       ├── github_file_fetcher.py  # GitHub real-world file fetcher for benchmark training
│       ├── hardware_detector.py    # CPU/GPU hardware detection
│       ├── performance_tuner.py    # Adaptive performance tuning
│       └── settings_manager.py     # Config.py read/write & GUI state
│
├── puffinzip_gui/                  # Tkinter desktop GUI
│   ├── primary_main_app.py         # Main app class (PuffinZipApp)
│   ├── secondary_main_app.py       # Evolution controls & changelog tabs
│   ├── settings_gui.py             # Settings editor panel
│   ├── chart_utils.py              # Matplotlib charting
│   ├── checkpoint_manager_panel.py # Checkpoint management panel
│   ├── generational_data_viewer.py # Generation deep-dive viewer (collapsible, lazy-loaded, paginated)
│   ├── gui_layout_setup.py         # Main layout builder
│   ├── gui_style_setup.py          # ttk style configuration
│   ├── gui_utils.py                # Font resolution & scroll helpers
│   └── gui_themes.json             # Theme presets
│
├── webui_static/                   # Web UI static assets
│   ├── css/ (style.css, themes.css)
│   └── js/  (app.js, charts.js, logger.js)
├── webui_templates/                # Web UI HTML templates
│   └── index.html\n│   └── login.html
│
├── start.sh                        # Universal Linux/macOS launcher (hardware auto-detect + run presets)
├── start.bat                       # Universal Windows launcher (hardware auto-detect + run presets)
│
├── scripts/                        # Build helpers & dev scripts
│   ├── package_a100.bat          # Windows: package project ZIP for pod deployment
│   ├── _package_a100_impl.ps1    # PowerShell implementation for package_a100.bat
│   ├── run_webui_windows.bat     # Developer's personal Windows launcher
│   ├── preflight_metrics_check.py
│   └── run_gui.spec
│
├── examples/                       # Demo & example scripts
│   ├── compression_discovery_example.py
│   └── hybrid_compression_demo.py
│
├── docs/                           # Documentation
│   ├── README.md
│   ├── HYBRID_COMPRESSION_GUIDE.md
│   ├── WEBUI_GUIDE.md
│   ├── CLOUDFLARE_TUNNEL_GUIDE.md  # Cloudflare Tunnel setup for custom domain hosting
│   └── changelog.md
│
├── data/                           # Training data & models
│   ├── benchmark_sets/
│   └── models/
├── logs/                           # Runtime logs
├── checkpoints/                    # Evolution checkpoints
├── gold_standard_results/          # Head-to-head failure diagnostics (gen_<N>/summary + artefacts) — EXCLUDED from index (runtime output)
└── CODEBASE_INDEX.md               # This file
```

---

## Table of Contents

1. [Root-Level Entry Points](#root-level-entry-points)
2. [puffinzip_ai/ — Core Package](#puffinzip_ai--core-package)
3. [puffinzip_ai/evolution_core/ — Evolutionary Optimization](#puffinzip_aievolution_core--evolutionary-optimization)
4. [puffinzip_ai/gpu_core/ — GPU Acceleration](#puffinzip_aigpu_core--gpu-acceleration)
5. [puffinzip_ai/nn_core/ — Neural Network (DQN) Agents](#puffinzip_ainn_core--neural-network-dqn-agents)
6. [puffinzip_ai/utils/ — Utility Modules](#puffinzip_aiutils--utility-modules)
7. [puffinzip_gui/ — Tkinter Desktop GUI](#puffinzip_gui--tkinter-desktop-gui)
8. [examples/ — Demo Scripts](#examples--demo-scripts)
9. [scripts/ — Launcher Scripts](#scripts--launcher-scripts)

---

## Root-Level Entry Points

---

### main_cli.py
**Lines**: 238  
**Purpose**: Command-line interface for PuffinZipAI training, compression, decompression, and model management.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 28 | `print_header()` | Prints CLI header banner |
| 32 | `get_user_input()` | Prompts user for input with validation |
| 63 | `list_str_parser()` | Parses comma-separated string lists |
| 67 | `list_int_parser()` | Parses comma-separated integer lists |
| 76 | `display_status()` | Shows current AI agent status |
| 79 | `train_random_cli()` | Trains AI on randomly generated data |
| 100 | `train_folder_cli()` | Trains AI on files from a folder |
| 117 | `batch_compress_cli()` | Batch-compresses files in a folder |
| 128 | `batch_decompress_cli()` | Batch-decompresses .pfz files |
| 139 | `single_item_cli()` | Compress/decompress a single text item |
| 162 | `configure_ai_cli()` | Configure AI hyperparameters |
| 183 | `manage_model_cli()` | Save/load model from CLI |
| 210 | `main_menu()` | Main interactive menu loop |

**Key Dependencies**: `puffinzip_ai` (PuffinZipAI, setup_logger), `puffinzip_ai.config`

---

### run_gui.py
**Lines**: 175  
**Purpose**: GUI launcher with dependency checking and performance auto-tuning.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 126 | `FallbackLogger` | Minimal logger when setup_logger unavailable |

**Constants**:
| Line | Name | Value |
|------|------|-------|
| 11 | `REQUIRED_PACKAGES` | List of required pip packages |
| 44 | `TUNED_THROTTLE_PARAMS` | Auto-tuned throttle parameters dict |
| 73 | `_default_throttles_fallback` | Fallback throttle dict |

**Key Dependencies**: `puffinzip_gui.primary_main_app` (PuffinZipApp), `puffinzip_ai.logger`, `puffinzip_ai.utils.performance_tuner`

---

### webui_server.py
**Lines**: ~1470  
**Purpose**: Flask web UI server for evolutionary optimization visualization (port 5001). Tracks complexity tier, complexity value, tier budget, tier ceiling, **robustness**, **training phase**, **corruption level**, **decompression stats** (mismatches, items evaluated, successful compressions) alongside fitness/generation metrics. Provides deep-dive API for gene pool visualization and breeding analysis. Live generation number in population history is `current_generation + 1` (the generation currently being worked on, not the last completed one). **Metrics caching**: `metrics_history` and `generation_snapshots` are periodically flushed to `webui_metrics_cache.json` (every 5 generations + on training completion) and restored on server startup so data survives restarts. **Run history**: When a new training run starts, the previous run's metrics are archived (up to 10 runs). The Continue endpoint reuses the existing optimizer and population to extend evolution without resetting. **Authentication**: Session-based login via `PUFFIN_USERNAME` / `PUFFIN_PASSWORD` env vars; when both are set all routes except `/login`, `/logout`, `/health`, and static files require an authenticated session. A `/health` endpoint is always public for launcher health-checks. **URL prefix / subpath hosting**: When `custom_url` in credentials has a path component (e.g. `stelliro.com/PuffinZipAI`), a `_PrefixMiddleware` WSGI wrapper strips the prefix before Flask sees the request and sets `SCRIPT_NAME` so `url_for()` generates correctly prefixed URLs. `ProxyFix` is also applied for X-Forwarded-* header support behind reverse proxies. The prefix is injected into the dashboard HTML as `window.PUFFIN_PREFIX` for JS fetch calls. When `custom_url` is a bare subdomain (e.g. `PuffinZipAI.Stelliro.com`) with no path, no prefix middleware is applied and the app runs at `/`. **Route structure**: `/` redirects to `/login` (public), `/dashboard` renders the main UI (requires auth), `/login` handles authentication, login success redirects to `/dashboard`.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 65 | `AppState` | Manages training state, metrics queue, log queue, **complexity tracking** (`current_complexity_tier`, `current_complexity_value`, `current_tier_budget_mb`, `current_tier_ceiling_kb`), **anti-corruption tracking** (`current_best_robustness`, `current_training_phase`, `current_corruption_level`, `current_decomp_mismatches`, `current_items_evaluated`, `current_successful_compressions`), **disk cache** (`save_cache()`, `load_cache()` for `webui_metrics_cache.json`), **run history** (`run_number`, `run_history[]`, `completed_naturally`, `archive_run()`) |
| ~253 | `_PrefixMiddleware` | WSGI middleware that strips a URL prefix from `PATH_INFO` and sets `SCRIPT_NAME` for subpath hosting (e.g. `/PuffinZipAI`). Non-prefixed requests return 404. Only active when `custom_url` has a path component. |

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 14 | `_detect_system_limits()` | Detects RAM, CPU cores, GPU memory |
| 83 | `index()` | Serves main HTML page (GET /) |
| 87 | `status()` | Returns training status JSON (GET /status) |
| 121 | `metrics()` | Returns fitness/generation/complexity/robustness/decompression metrics (GET /metrics) — includes `complexity_tier`, `complexity_value`, `tier_budget_mb`, `tier_ceiling_kb`, `best_robustness`, `training_phase`, `corruption_level`, `decomp_mismatches`, `items_evaluated`, `successful_compressions` |
| 149 | `population()` | Returns population details JSON (GET /population) |
| 184 | `population_history()` | Returns paginated generation snapshots (GET /population/history) — `page`, `per_page` params, includes live generation with LIVE badge; **live gen number = `current_generation + 1`** (the generation being worked on); live agents include `evaluation_stats` for robustness/decompression display |
| ~252 | `logs()` | Returns log messages (GET /logs) |
| ~260 | `methods()` | Returns available compression methods (GET /methods) |
| ~267 | `start()` | Starts evolution training (POST /start) — accepts `target_device` (GPU_AUTO/CPU) and `cpu_workers` (1-32) |
| ~350 | `stop()` | Stops evolution training (POST /stop) |
| ~357 | `continue_training()` | Continues evolution from where it left off (POST /api/training/continue) — reuses existing optimizer and population, accepts `extra_generations` (default 100) and `infinite` (bool). Calls `optimizer.continue_evolution()` which preserves population state and resumes the generation loop. |
| ~420 | `run_history()` | Returns archived run summaries (GET /api/training/run-history) — `run_number`, `generations`, `best_fitness`, `best_ratio`, `final_tier` per archived run |
| ~355 | `evolution_deep_dive()` | Returns enriched generation data for deep-dive tab (GET /api/evolution/deep-dive) — gene pool clustering, enriched breeding pairs (with parent/child fitness, pool indices, cross-pool flag), pool-to-pool breeding matrix, top breeders (most selected parents), method direction trends (avg RLE min run, novel method count, cross-pool stats per gen), mutation/crossover counts, top agents, per-agent lineage |
| ~373 | `_get_checkpoint_manager()` | Returns optimizer's checkpoint manager, or creates a standalone read-only one to list checkpoints from disk |
| ~415 | `test_checkpoint()` | Tests a saved checkpoint by compressing an uploaded file with the best agent (POST /api/checkpoint/test) — accepts FormData with `checkpoint` name and `file` upload; returns ratio, savings %, round-trip integrity, timing, agent fitness |
| ~510 | `delete_checkpoint()` | Deletes a checkpoint by key or name (POST /api/checkpoint/delete) — accepts JSON `{key}` or `{name}` |

**Constants**:
| Line | Name | Description |
|------|------|-------|
| 42 | `SYSTEM_LIMITS` | Dict of detected hardware limits |
| 63 | `app` | Flask application instance |
| 80 | `app_state` | Global AppState singleton |
| -- | `_WEBUI_CACHE_PATH` | `webui_metrics_cache.json` in project root — persisted metrics + snapshots |
| -- | `_CACHE_FLUSH_INTERVAL` | `5` — flush to disk every N generations |
| -- | `_MAX_METRICS_HISTORY` | `10_000` — cap on `metrics_history` entries in memory/disk; oldest evicted on overflow to bound RAM/cache file for infinite runs |

**Key Dependencies**: `flask`, `flask_cors`, `puffinzip_ai.evolution_core.evolutionary_optimizer`, `puffinzip_ai.hybrid_compression_engine`

### Web UI Frontend (webui_static/ + webui_templates/)
**Layout**: Horizontal dashboard design — KPI strip (6 cards), chart + sidebar main row, tabbed bottom section.

**Files**:
| Path | Purpose |
|------|---------|
| `webui_templates/index.html` | Two top-level tabs (Dashboard, Evolution Deep Dive). Dashboard: KPI strip, chart panel + complexity sidebar, sub-tabs (Population, Logs, Methods, Validation, Settings). Deep Dive: gene pool KPIs, fitness/mutation chart, gene pool legend, top agents leaderboard, coloured agent grid with click-to-trace lineage + tooltips, breeding network panel (SVG flow diagram + pool×pool matrix + top breeders), method direction panel (trend chart + summary cards), breeding pairs + generation detail cards with fitness comparison. |
| `webui_static/css/style.css` | Dashboard CSS grid layout + Deep Dive layout (`.top-tabs`, `.dd-*` classes for gene pool grid, agent cells, tooltips, lineage highlighting, breeding network SVG panel, pool matrix, top breeders, method direction, generation cards). Responsive breakpoints for all layouts. |
| `webui_static/css/themes.css` | 20 theme definitions (CSS custom properties) |
| `webui_static/js/app.js` | `PuffinZipAIApp` class — 3-dataset chart (compression score + benchmark + complexity), complexity polling, log filtering, export CSV/TXT, population history viewer, top-level tab switching. Compression score displayed as compression factor `orig/comp*100` (>100% = actual compression). **Continue training**: `continueTraining()` sends POST to `/api/training/continue`, `canContinue` flag tracked from `/api/status`, `updateUIState()` enables Continue button when a run completes naturally. Chart is archived via `archiveAndResetChart()` when Start is pressed after a previous run. |
| `webui_static/js/deep_dive.js` | `DeepDiveManager` class — Polls `/api/evolution/deep-dive`, renders: interactive gene pool grid (click agent for **cross-generation lineage highlighting** of parents/children/siblings), lineage detail panel (multi-generational ancestry: parents, grandparents, great-grandparents, children, siblings — with clickable navigation to their generation), fitness/mutation chart, method direction chart (RLE trend + cross-pool breeds + novel methods), gene pool legend, top agents leaderboard, breeding flow SVG (parent→child connection curves, cross-pool dashed lines), pool×pool breeding matrix (heatmap), top breeders list (frequency bars), method direction summary cards, per-generation expandable detail cards with enriched breeding pairs (fitness comparison, outcome indicators) and mini fitness bar charts. |
| `webui_static/js/charts.js` | Global helpers `addMetricToChart()` (3 datasets) + `resetChart()` + `archiveAndResetChart()` (stores current chart data in `window._chartRunHistory` before clearing, max 10 archived runs) |
| `webui_static/js/logger.js` | `logMessage()` + `escapeHtml()` helpers |

**Complexity Metrics Flow**: `evolutionary_optimizer._send_metrics_json()` → `METRICS_JSON:` string via queue → `Bridge.put_nowait()` in webui_server → `AppState` fields → `/api/metrics` JSON → `app.js pollMetrics()` → KPI cards + sidebar + chart dataset

---

### webui_credentials_manager.py
**Lines**: ~155  
**Purpose**: Auto-generates and persists `webui_credentials.json` in the project root on first WebUI launch. Produces a 20-character alphanumeric username, a 64-character alphanumeric password, a 64-character hex secret key, a `public_access` boolean (default `false`), optional `admin_username` / `admin_password` fields (default empty), and a `custom_url` field (default empty). When `public_access` is `true`, the server binds to `0.0.0.0` (network/internet-accessible); when `false`, it binds to `127.0.0.1` (local only). The `admin_username` / `admin_password` fields provide a secondary login for remote access (e.g. `stelliro.com/puffinzipai`); both must be non-empty to enable the admin login. `custom_url` is the public-facing URL shown in the startup banner. Env-var overrides: `PUFFIN_USERNAME`, `PUFFIN_PASSWORD`, `PUFFIN_SECRET_KEY`, `PUFFIN_PUBLIC_ACCESS`, `PUFFIN_ADMIN_USERNAME`, `PUFFIN_ADMIN_PASSWORD`, `PUFFIN_CUSTOM_URL`. The credentials file is `.gitignore`-ed and `chmod 600`-ed on Linux.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 55 | `_generate_random_string()` | Crypto-random alphanumeric string of a given length |
| 60 | `_generate_credentials()` | Creates a fresh `Credentials` dict |
| 67 | `_load_credentials_file()` | Reads `webui_credentials.json` from disk |
| 83 | `_save_credentials_file()` | Writes credentials JSON with restrictive permissions |
| 93 | `load_or_create_credentials()` | Main entry — env-vars → file → auto-generate (priority order) |

**Constants**:
| Line | Name | Description |
|------|------|-------------|
| 40 | `_CREDENTIALS_FILE` | Path to `webui_credentials.json` |
| 43 | `_USERNAME_LENGTH` | 20 |
| 44 | `_PASSWORD_LENGTH` | 64 |

---

### webui_theme_manager.py
**Lines**: 313  
**Purpose**: Cross-compatible theme system for GUI and web UI with 20 built-in themes.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 249 | `ThemeManager` | Theme management — `get_all_themes()`, `get_theme()`, `get_theme_names()`, `get_css_class()`, `validate_theme()` |

**Constants**:
| Line | Name | Description |
|------|------|-------|
| 11 | `THEMES_CONFIG` | Dict with 20 theme definitions (Nord, Dracula, Monokai, Solarized, etc.) |

---

## puffinzip_ai/ — Core Package

---

### puffinzip_ai/\_\_init\_\_.py
**Lines**: 313  
**Purpose**: Package init — auto-generates config.py if missing, selects GPU vs CPU core, exports all public symbols.

**Constants**:
| Line | Name | Description |
|------|------|-------|
| 22 | `ALL_CONFIG_DEFAULTS_INIT_TIME` | Massive dict of all configuration defaults; used to auto-generate config.py |
| 18 | `CONFIG_FILE_PATH_FOR_INIT` | Path to config.py |

**Key Logic**: Agent class resolution priority:
1. If `NN_ENABLED=True` in config → tries `nn_core.nn_agent.PuffinZipAI_NN` (DQN neural network)
2. If `ACCELERATION_TARGET_DEVICE` contains "GPU" → tries `gpu_core.PuffinZipAI_GPU` (CuPy Q-table)
3. Else falls back to `ai_core.PuffinZipAI` (CPU tabular Q-table)

**Exports**: `PuffinZipAI`, `setup_logger`, `rle_compress`, `rle_decompress`, `get_hybrid_engine`, `get_registry`, `get_generator`, `generate_novelty`, `evolve`, `calculate_reward`, `EvolutionaryOptimizer`, `EvolvingAgent`

---

### puffinzip_ai/config.py
**Lines**: ~200  
**Purpose**: Central configuration file with all constants and path definitions.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~150 | `ensure_dirs()` | Creates required directories (data, models, logs, benchmarks, github_cache) |

**Constants** (selected):
| Line | Name | Value |
|------|------|-------|
| — | `APP_VERSION` | `'0.9.8'` |
| — | `COMPRESSED_FILE_SUFFIX` | `'.pfz'` |
| — | `DEFAULT_LEN_THRESHOLDS` | `[50, 150, 500]` |
| — | `DEFAULT_LEARNING_RATE` | `0.1` |
| — | `DEFAULT_DISCOUNT_FACTOR` | `0.9` |
| — | `DEFAULT_EXPLORATION_RATE` | `1.0` |
| — | `DEFAULT_EXPLORATION_DECAY_RATE` | `0.9995` |
| — | `DEFAULT_MIN_EXPLORATION_RATE` | `0.01` |
| — | `DEFAULT_TRAIN_BATCH_SIZE` | `32` |
| — | `DEFAULT_POPULATION_SIZE` | `50` |
| — | `DEFAULT_NUM_GENERATIONS` | `100` |
| — | `DEFAULT_MUTATION_RATE` | `0.15` |
| — | `DEFAULT_ELITISM_COUNT` | `2` |
| — | `DEFAULT_SELECTION_STRATEGY` | `'tournament'` |
| — | `ACCELERATION_TARGET_DEVICE` | `'GPU_ID:0'` |
| — | `ELS_CONTINUOUS_RUN_ENABLED` | `True` |
| — | `GPU_RLE_TARGET_VRAM_USAGE_FRACTION` | Float (VRAM fraction for RLE workspace) |
| — | `NN_STATE_FEATURE_DIM` | `20` (was 7) |
| — | `NN_HIDDEN_SIZES` | `[256, 256]` (was [128,128]) |
| — | `NN_REPLAY_BUFFER_CAPACITY` | `50000` (was 10000) |
| — | `NN_TRAIN_BATCH_SIZE` | `128` (was 64) |
| — | `NN_LEARNING_RATE` | `3e-4` (was 1e-3, now AdamW) |
| — | `NN_PER_ALPHA` | `0.6` — PER prioritisation exponent |
| — | `NN_PER_BETA_START` | `0.4` — IS correction initial beta |
| — | `NN_PER_BETA_FRAMES` | `100000` — beta annealing horizon |
| — | `NN_COSINE_LR_T_MAX` | `5000` — CosineAnnealing restart period |
| — | `NN_COSINE_LR_ETA_MIN` | `1e-6` — min LR at cosine trough |
| — | `NN_SOFT_TARGET_TAU` | `0.005` — Polyak averaging coefficient |
| — | `NN_NSTEP_RETURNS` | `3` — N-step return horizon |
| — | `NN_DROPOUT` | `0.1` — dropout in residual blocks |
| — | `NN_ATTENTION_HEADS` | `4` — multi-head attention heads |
| — | `NN_NOISY_SIGMA` | `0.5` — NoisyNet initial sigma |

---

### puffinzip_ai/logger.py
**Lines**: 153  
**Purpose**: Configurable logging with RotatingFileHandler and optional console output.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 35 | `setup_logger()` | Creates logger with file + optional console handlers; respects `DEBUG_LOG_CONSOLE_OUTPUT_ENABLED` |

**Constants**:
| Line | Name | Description |
|------|------|-------|
| 31 | `LOG_LEVEL_ACTUAL_DEFAULT` | Default log level |

---

### puffinzip_ai/rle_constants.py
**Lines**: 25  
**Purpose**: RLE error codes and safety constants.

**Constants**:
| Line | Name | Value |
|------|------|-------|
| — | `RLE_ERROR_NO_COUNT` | Error string |
| — | `RLE_ERROR_BAD_COUNT` | Error string |
| — | `RLE_ERROR_NO_CHAR` | Error string |
| — | `RLE_ERROR_MALFORMED` | Error string |
| — | `RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY` | Error string |
| — | `RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED` | Error string |
| — | `RLE_ERROR_MEMORY_ON_CHUNK` | Error string |
| — | `RLE_ERROR_MEMORY_ON_JOIN` | Error string |
| — | `RLE_DECOMPRESSION_ERRORS` | Set of all error strings |
| — | `ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE` | `200 * 1024 * 1024` (200 MB) |
| — | `ABSOLUTE_MAX_PARSED_COUNT` | `100 * 1024 * 1024` (100M) |
| — | `MAX_COUNT_STRING_DIGITS` | `9` |

---

### puffinzip_ai/rle_utils.py
**Lines**: ~310  
**Purpose**: Simple RLE compression/decompression using marker-framed format (STX `\x02` control char).

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~130 | `_get_advanced_rle_module()` | Lazy-loads advanced_rle_methods |
| ~155 | `simple_rle_compress()` | Simple RLE compression with marker framing |
| ~198 | `simple_rle_decompress()` | Simple RLE decompression |
| ~280 | `rle_compress()` | Dispatcher — routes to simple or advanced |
| ~295 | `rle_decompress()` | Dispatcher — routes to simple or advanced |

**Constants**:
| Line | Name | Value |
|------|------|-------|
| — | `MIN_ENCODABLE_RUN_LENGTH` | `3` |
| — | `RLE_RUN_MARKER` | `'\x02'` (STX) |
| — | `RLE_DELIMITER` | `` '`' `` |
| — | `THROTTLE_RUN_LENGTH_THRESHOLD` | From performance_tuner |
| — | `THROTTLE_CHUNK_SIZE` | From performance_tuner |
| — | `THROTTLE_SLEEP_DURATION` | From performance_tuner |

**Key Dependencies**: `.rle_constants`, `.utils.performance_tuner`, `.advanced_rle_methods` (lazy)

---

### puffinzip_ai/advanced_rle_methods.py
**Lines**: ~400  
**Purpose**: Advanced RLE using SOH (`\x01`) delimiter-framed format with lower min run length.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~115 | `advanced_rle_compress()` | Advanced RLE compression |
| ~150 | `advanced_rle_decompress()` | Advanced RLE decompression |

**Constants**:
| Line | Name | Value |
|------|------|-------|
| — | `MIN_ENCODABLE_RUN_LENGTH_ADVANCED` | `2` |
| — | `ADV_RLE_DELIMITER` | `'\x01'` (SOH) |

**Key Dependencies**: `.rle_constants`

---

### puffinzip_ai/ai_core.py
**Lines**: 1177  
**Purpose**: Core Q-learning AI agent for compression strategy selection (TD(0) learning).

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 30 | `DummyLogger` | Minimal logging fallback |
| 38 | `PuffinZipAI` | Main AI class — Q-learning agent for compression |

**PuffinZipAI Key Methods**:
| Line | Name | Description |
|------|------|-------------|
| — | `__init__()` | Initializes Q-table, actions, hyperparameters |
| — | `_get_state_representation()` | Maps data to state index (length category × unique char ratio × run length) |
| — | `_choose_action()` | Epsilon-greedy action selection from Q-table |
| — | `_update_q_table()` | TD(0) Q-learning update |
| — | `_generate_random_item()` | Generates random training data using bulk chunk-based generation (random.choices + string multiply) |
| — | `_handle_item_processing_for_training()` | Processes one item through compress→decompress→reward pipeline |
| — | `_process_batch()` | Processes a training batch |
| — | `train()` | Trains on random data for N batches |
| — | `learn_from_folder()` | Trains on real files from a folder |
| — | `batch_compress_folder()` | Compresses all files in a folder |
| — | `batch_decompress_folder()` | Decompresses all .pfz files |
| — | `compress_user_item()` | Compresses a single user-provided item |
| — | `decompress_user_item_rle()` | Decompresses a single .pfz item |
| — | `display_q_table_summary()` | Prints Q-table statistics |
| — | `test_agent_on_random_items()` | Tests agent on random data |
| — | `clone_core_model()` | Deep-clones the AI agent |
| — | `save_model()` / `load_model()` | Pickle-based model persistence |
| — | `get_config_dict()` | Returns configuration as dict |
| — | `__getstate__()` / `__setstate__()` | Pickle support — excludes unpicklable closures (novel method fns), loggers, GUI refs; reconstructs novel methods from saved metadata on unpickle |
| — | `configure_data_categories()` | Updates length thresholds and reinitializes Q-table |
| — | `_send_to_gui()` | Sends messages to GUI output queue |

**Key Attributes**: `q_table` (numpy), `len_thresholds`, `action_names={0:"RLE", 1:"NoCompression", 2:"AdvancedRLE", 3:"NovelMethod", 4:"ReferenceMethod"}`, `novel_method`, `_novel_compress_fn`, `_novel_decompress_fn`, `_scaffolding_enabled`, `_preferred_reference`, `_reference_compress_fn`, `_reference_decompress_fn`

**Pickle Behavior**: `__getstate__` strips unpicklable closures (`_novel_compress_fn`, `_novel_decompress_fn`, `_reference_compress_fn`, `_reference_decompress_fn`, `novel_method`), logger, and GUI refs. Saves novel method reconstruction metadata (`pipeline`, `discovery_seed`, `rle_min_run`, `steps`, `author`). `__setstate__` rebuilds novel closures via `NovelCompressionGenerator._build_pipeline()`, reconstructs full `novel_method` metadata dict (pipeline, discovery_seed, rle_min_run, steps) and passes `author` to the rebuilt `CompressionMethod`, restores reference closures from `compression_scaffolding.get_reference_method()`, and restores the logger.

**Key Dependencies**: `.config`, `.logger`, `.reward_system`, `.rle_utils`, `.rle_constants`, `.compression_scaffolding`, `numpy`

---

### puffinzip_ai/reward_system.py
**Lines**: ~500  
**Purpose**: Multi-signal reward calculation for Q-learning with progressive difficulty scaling, data-awareness, competitive benchmarking, consistency tracking, and composite fitness.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~30 | `calculate_size_scaled_reward()` | Applies size-based bonus to reward (larger files = higher reward) |
| ~100 | `calculate_method_diversity_adjustment()` | Penalizes method monotony, rewards diverse strategy usage |
| ~150 | `calculate_population_novelty_scores()` | Computes novelty scores for a population based on strategy uniqueness |
| ~200 | `calculate_generation_repetition_penalty()` | Penalizes methods overused across generations |
| ~260 | `calculate_reward()` | Main reward function — considers ratio, correctness, errors, plus progressive difficulty, data-awareness, consistency. Accepts optional `generation` and `recent_results` |
| ~370 | `calculate_scaffolded_reward()` | Applies scaffolding multiplier for reference method usage or own-method bonus when agent beats reference |
| ~390 | `calculate_competitive_reward()` | Compares AI vs gzip/bz2/lzma/zlib baselines; bonuses for beating each |
| ~430 | `calculate_data_aware_adjustment()` | Evaluates method-data fitness based on entropy analysis |
| ~470 | `calculate_composite_fitness()` | Combines all reward signals into a single fitness scalar |

**Constants** (selected):
| Line | Name | Value |
|------|------|-------|
| — | `PENALTY_MISMATCH` | `-10.0` |
| — | `PENALTY_RLE_PROCESSING_ERROR` | `-12.0` |
| — | `PENALTY_CATASTROPHIC_EXPANSION` | `-15.0` |
| — | `PENALTY_WRONG_METHOD_FOR_DATA` | `-3.0` |
| — | `PENALTY_REPEATED_FAILURE` | `-5.0` |
| — | `PENALTY_REFERENCE_METHOD_BASE` | `-0.5` |
| — | `REWARD_SCALER_COMPRESSION_SUCCESS` | `10.0` |
| — | `REWARD_EXCEPTIONAL_COMPRESSION` | `15.0` (>50% ratio bonus) |
| — | `REWARD_SPEED_BONUS` | `2.0` |
| — | `REWARD_CORRECT_METHOD_CHOICE` | `3.0` |
| — | `REWARD_BEAT_BASELINE` | `5.0` |
| — | `REWARD_CONSISTENCY_BONUS` | `2.0` |
| — | `PROGRESSIVE_DIFFICULTY_START_GEN` | `20` |
| — | `PROGRESSIVE_DIFFICULTY_MAX_GEN` | `200` |

---

### puffinzip_ai/checkpoint_manager.py
**Lines**: ~450  
**Purpose**: Checkpoint save/load/compare system with metadata and scoring. Thread-safe, atomic-write (temp file + rename) to prevent 0-byte checkpoint files.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~16 | `CheckpointMetadata` | Dataclass for checkpoint metadata (generation, fitness, timestamp, etc.) |
| ~60 | `CompressionScoreCalculator` | Calculates compression performance scores |
| ~120 | `CheckpointManager` | Main checkpoint management — `save_checkpoint()` (thread-safe, atomic write via temp file + `os.replace`), `load_checkpoint()`, `list_checkpoints()`, `compare_checkpoints()`, `delete_checkpoint()`, `get_checkpoint_metadata()` |

**Key Dependencies**: `json`, `pickle`, `logging`, `tempfile`, `copy`, `threading`

---

### puffinzip_ai/compression_benchmark.py
**Lines**: ~520  
**Purpose**: Comprehensive A100-ready benchmark and validation suite. Compares AI compression against gzip, bz2, lzma, zlib (and zstd if available). Supports multi-dataset evaluation, throughput profiling, latency percentile analysis, GPU memory tracking, and tiered success criteria.

**Data Classes**:
| Name | Description |
|------|-------------|
| `CompressionResult` | Single compression operation result (ratio, savings, timings, throughput, verified) |
| `BenchmarkReport` | Full dataset benchmark report with per-method aggregates, AI vs baseline, tier |

**Classes**:
| Name | Description |
|------|-------------|
| `CompressionBenchmark` | Main benchmark engine — all baseline compressors, multi-dataset suite, GPU profiling |

**CompressionBenchmark Key Methods**:
| Method | Description |
|--------|-------------|
| `compress_with_gzip/bz2/lzma/zlib/zstd()` | Individual baseline compressors |
| `benchmark_single(data)` | Benchmark all baselines on one data item; returns `list[CompressionResult]` |
| `get_baseline_compression(data)` | Dict-based baseline results (backward-compatible) |
| `compare_compression(original, ai_size)` | AI vs best baseline comparison dict |
| `run_benchmark_suite(datasets, ai_fn, ...)` | Multi-dataset benchmark with n_runs statistical robustness |
| `validate_ai_success(orig, ai_compressed)` | Validates AI beats baselines; generates report |
| `format_comparison_report(comparison)` | Human-readable single comparison |
| `format_full_report(report)` | Full benchmark report with tables, percentiles, tier |
| `profile_gpu_performance(agent, data, ...)` | A100 GPU profiling: latency p50/p95/p99, throughput MB/s, VRAM usage |
| `generate_test_datasets()` | Static method — 6 synthetic datasets (repetitive, random, english, numeric, mixed, binary) |

**Success Tiers**:
| Tier | Criteria |
|------|----------|
| Platinum | AI beats baseline >50% of items AND avg ratio within 5% |
| Gold | AI beats baseline >30% of items OR avg ratio within 10% |
| Silver | AI achieves >0% compression on >70% of items |
| Bronze | AI achieves >0% compression on >40% of items |

---

### puffinzip_ai/gold_standard_benchmark.py
**Lines**: ~770  
**Purpose**: After each generation, pits the best agent head-to-head against gzip, bz2, lzma, zlib, and zstd. If the agent beats **all** baselines on **every** test item, a "gold standard" checkpoint is saved. On failure, saves compressed + decompressed artefacts and a summary to `gold_standard_results/gen_<N>/` for diagnosis.

**Artifact Compression & Size Management**: Eval directories are compressed into `.zip` archives (ZIP_DEFLATED, level 6) only for generations **older** than the keep-recent window (`KEEP_RECENT_GENS_UNCOMPRESSED = 100`). The most recent 100 generations' eval dirs remain uncompressed on disk so they can be browsed directly. On `__init__`, any uncompressed eval directories from prior runs that are outside the recent window are migrated to `.zip`. A 10 GB size limit (`MAX_ARTIFACTS_BYTES`) is enforced after each save — the oldest generation directories are pruned first, but the most recent 100 generations are protected from deletion.

**Size Measurement**: AI compressed output is measured via `_measure_compressed_size()` which counts
Latin-1 chars (0-255) as 1 byte each, and higher codepoints at their minimum UTF-8 byte width.
This gives a fair comparison against baselines that produce raw bytes.

**Data Classes**:
| Name | Description |
|------|-------------|
| `ItemResult` | One method's compression outcome for a single item (sizes, ratio, verified) |
| `HeadToHeadResult` | Per-item AI vs all baselines — includes `ai_beats_all` flag |
| `GenerationBenchmarkReport` | Full generation report: items, wins, losses, gold_standard flag, summary |

**Classes**:
| Name | Description |
|------|-------------|
| `GoldStandardBenchmark` | Main benchmark runner — initialised by `EvolutionaryOptimizer`, called each generation |

**GoldStandardBenchmark Key Methods**:
| Method | Description |
|--------|-------------|
| `benchmark_generation(gen, agent, items, ...)` | Run head-to-head; return report; auto-save gold checkpoint or failure artefacts |
| `_save_failure_artifacts(gen, report, items, agent)` | Write `summary.txt`, `summary.json`, per-item files; then compress old gens outside the keep-recent window |
| `_compress_eval_to_zip(eval_dir)` | Compress an eval directory to `.zip` and remove the loose tree; handles already-zipped case (interrupted prior runs) |
| `_compress_old_generations(current_gen)` | Walk `gen_*` dirs and zip eval dirs for gens older than `current_gen - KEEP_RECENT_GENS_UNCOMPRESSED` |
| `_migrate_uncompressed_evals()` | One-time `__init__`-time scan: zips any remaining loose eval dirs from prior runs (**respects** `KEEP_RECENT_GENS_UNCOMPRESSED`) |
| `_enforce_size_limit()` | Prunes oldest `gen_*` directories when total exceeds `MAX_ARTIFACTS_BYTES` (10 GB); **protects** the most recent 100 gens |
| `_get_artifacts_total_bytes()` | Walks the artifacts directory to compute total on-disk size |

**Constants**:
| Name | Value | Description |
|------|-------|-------------|
| `ARTIFACTS_DIR` | `"gold_standard_results"` | Directory name relative to project root |
| `MAX_ARTIFACTS_BYTES` | `10 * 1024³` (10 GB) | Size cap; triggers oldest-gen pruning |
| `KEEP_RECENT_GENS_UNCOMPRESSED` | `100` | Most recent N gens kept as loose dirs (not zipped); also protected from size-limit deletion |

**Artefact Layout (on failure)**:
```
gold_standard_results/              (runtime output — excluded from codebase index)
  gen_<N>/
    eval_<K>.zip            – compressed eval archive (ZIP_DEFLATED)
      eval_<K>/
        summary.txt           – human-readable report
        summary.json          – machine-readable report
        item_000/
          original.txt        – original test text
          ai_compressed.txt   – AI-compressed output (text or hex dump)
          ai_decompressed.txt – AI round-trip decompressed
          ai_action.txt       – which action the agent chose
          gzip_compressed.txt – gzip baseline hex dump
          bz2_compressed.txt  – bz2 baseline hex dump
          ...
```

**Integration**: Imported & initialised by `EvolutionaryOptimizer.__init__()`. Called in the evolution loop after evaluation + metrics, before auto-checkpoint & breeding (step 2b-ii).

**Key Dependencies**: `gzip`, `bz2`, `lzma`, `zlib`, `zstandard` (optional), `zipfile`, `shutil`, `.rle_utils`

---

### puffinzip_ai/compression_method_registry.py
**Lines**: ~200  
**Purpose**: Dynamic registry for compression methods from any language.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~10 | `CompressionLanguage` | Enum: `PYTHON`, `RUST`, `CPP`, `CUDA`, `HYBRID` |
| ~30 | `CompressionMetric` | Metrics dataclass (ratio, speed, correctness) |
| ~45 | `CompressionMethod` | Dataclass — name, language, compress_fn, decompress_fn, metrics |
| ~65 | `CompressionRegistry` | Registry singleton — register/list/get methods |

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~155 | `get_registry()` | Returns global registry singleton |
| ~160 | `register_method()` | Registers a CompressionMethod |
| ~165 | `register_python_method()` | Convenience for Python-only methods |

**Constants**:
| Line | Name | Description |
|------|------|-------|
| — | `_GLOBAL_REGISTRY` | Singleton CompressionRegistry |

---

### puffinzip_ai/hybrid_compression_engine.py
**Lines**: ~250  
**Purpose**: Unified compression interface for Python, Rust, and novelty methods.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~30 | `HybridCompressionEngine` | Unified engine — `compress()`, `decompress()`, `discover_novelty_method()`, `evolve_methods()`, `test_method()`, `get_best_method()`, `list_available_methods()` |

**Built-in Methods**: `"burst"` (Rust BURST), `"delta_rle"` (Python), `"frequency_codec"` (Python)

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~235 | `get_hybrid_engine()` | Returns singleton engine |

**Key Dependencies**: `.compression_method_registry`, `.novel_compression_generator`, `.rust_compression_interface`

---

### puffinzip_ai/novel_compression_generator.py
**Lines**: ~800  
**Purpose**: Generates invertible compression methods from composable primitives + random discovery transforms.

**Encoding Convention** (v2 — Latin-1):
Discovery transforms (XOR, permutation) and delta encoding now store binary data
using **Latin-1** (1 byte ↔ 1 char, zero overhead) instead of hex encoding.
The safe-RLE escape mechanism uses **suffix-based escapes** (ESC+'A' for marker,
ESC+'B' for escape) so that escape sequences cannot form compressible runs.
The benchmark size measurement in `gold_standard_benchmark.py` uses
`_measure_compressed_size()` which counts Latin-1 chars as 1 byte and higher
codepoints (e.g. BPE PUA chars) at their minimum UTF-8 byte width.

**Invertible Primitives** (top-level functions):
| Name | Description |
|------|-------------|
| `_rle_compress_safe()` / `_rle_decompress_safe()` | Safe RLE primitive (suffix-based escaping) |
| `_bwt_compress()` / `_bwt_decompress()` | Burrows-Wheeler Transform |
| `_mtf_compress()` / `_mtf_decompress()` | Move-to-Front |
| `_delta_compress()` / `_delta_decompress()` | Delta encoding (2-byte Latin-1 per value) |
| `_bpe_compress()` / `_bpe_decompress()` | Byte Pair Encoding |
| `_create_random_xor_transform()` | Random XOR-based transform (Latin-1 data encoding) |
| `_create_random_byte_permutation()` | Random byte permutation (Latin-1 data encoding) |
| `_create_random_block_shuffle()` | Random block shuffle |

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~345 | `CompressionPattern` | Dataclass — pattern name, steps, compress_fn, decompress_fn |
| ~355 | `DiscoveredTransform` | Dataclass — random discovery transform |
| ~370 | `NovelCompressionGenerator` | Pipeline builder — `generate_novelty_method()`, `generate_random_discovery()`, `evolve_methods()`, `_build_pipeline()`, `_verify_invertibility()` |

**Named Pipelines** (`PIPELINES` dict): `rle_only`, `bwt_rle`, `bwt_mtf_rle`, `delta_rle`, `bpe_rle`, `bpe_only`, `delta_bpe`, `bwt_bpe`

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `get_generator()` | Module-level generator singleton |
| — | `generate_novelty()` | Convenience wrapper |
| — | `evolve()` | Convenience wrapper for method evolution |

**Markers**: `_M_RLE='\x10'`, `_M_BWT='\x11'`, `_M_BWTC='\x12'`, `_M_MTF='\x13'`, etc.

---

### puffinzip_ai/compression_scaffolding.py
**Lines**: ~250  
**Purpose**: "Training wheels" system — gives agents access to known compression methods (gzip, zlib, bz2, lzma) as references, with progressive penalty scaling and temporary bans to encourage novel method discovery over reference reliance.

**Data Classes**:
| Name | Description |
|------|-------------|
| `ReferenceMethod` | Named reference compressor — `name`, `compress_fn`, `decompress_fn`, `description` |
| `AgentScaffoldState` | Per-agent scaffolding tracker — `usage_history` (rolling window), `is_banned`, `ban_remaining`, `ban_count`, `post_ban_cooldown`, `reliance_ratio`, `recent_reliance_ratio` |

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~80 | `ScaffoldingManager` | Manages per-agent scaffolding state. Key methods: `is_reference_allowed()`, `record_and_check()` (triggers ban when reliance ≥ threshold over window), `calculate_reward_multiplier()` (decays from 0.60 based on reliance + generation), `calculate_own_method_bonus()` (bonus when agent beats reference) |

**Key Constants**:
| Name | Value | Description |
|------|-------|-------------|
| `RELIANCE_BAN_THRESHOLD` | `0.50` | Ban agent if reliance ratio ≥ this |
| `BAN_DURATION_ITEMS` | `30` | Initial ban length; doubles each repeat |
| `BAN_MAX_DURATION_ITEMS` | `200` | Maximum ban duration cap |
| `SCAFFOLDING_GRACE_GENERATIONS` | `10` | Generations before penalties ramp |
| `SCAFFOLDING_RAMP_GENERATIONS` | `80` | Penalty ramp-up phase length |
| `SCAFFOLDING_MATURE_GENERATIONS` | `150` | Auto-ban reference at this generation |
| `BEAT_REFERENCE_BONUS` | `4.0` | Reward bonus when own method beats reference |
| `WEANING_BONUS` | `2.0` | Bonus when agent stops using reference |

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `get_reference_method()` | Returns a specific `ReferenceMethod` by name |
| — | `get_best_reference_ratio()` | Tries all reference methods, returns best compression ratio |
| — | `get_scaffolding_manager()` | Module-level singleton |

**Lifecycle**:
1. **Grace phase** (gens 0-10): Reference method available with mild reward multiplier (0.60)
2. **Ramp phase** (gens 10-80): Penalty increases progressively; ban triggered if reliance ≥ 50%
3. **Mature phase** (gen 150+): Reference method permanently auto-banned
4. Agents that beat the reference with their own methods receive `BEAT_REFERENCE_BONUS` (4.0)

**Key Dependencies**: `gzip`, `zlib`, `bz2`, `lzma` (standard library only)

---

### puffinzip_ai/rust_compression_interface.py
**Lines**: ~230  
**Purpose**: Interface to Rust compression implementations with Python fallbacks.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~22 | `RustCompressionInterface` | `compress_burst()`, `decompress_burst()`, `_python_burst_compress()`, `_python_burst_decompress()`, `_rle_pass()`, `_reverse_rle()`, `_calculate_entropy()`, `_recognize_tuples()` |

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `get_rust_interface()` | Returns singleton interface |

**Key Dependencies**: `subprocess` (for rustc check), `numpy` (for entropy calculation)

---

## puffinzip_ai/evolution_core/ — Evolutionary Optimization

---

### evolution_core/\_\_init\_\_.py
**Lines**: 27  
**Purpose**: Package init — exports `EvolutionaryOptimizer` and `EvolvingAgent`.

**Constants**:
| Line | Name | Value |
|------|------|-------|
| — | `__version__` | `"0.2.4"` |

---

### evolution_core/evolutionary_optimizer.py
**Lines**: ~1030  
**Purpose**: Main evolutionary optimization loop — resource-aware population management, evaluation, breeding, novelty tracking, and per-generation snapshots for the GDV.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 89 | `EvolutionaryOptimizer` | Main evolution engine |

**EvolutionaryOptimizer Key Methods**:
| Line | Name | Description |
|------|------|-------------|
| 89 | `__init__()` | Detects system resources, calculates safe pop/gen limits, initializes subsystems |
| 161 | `_detect_system_resources()` | Static — detects RAM (GB) + CPU cores via psutil |
| 173 | `_calculate_safe_population_size()` | RAM-based max population (~5MB/agent) |
| 182 | `_calculate_safe_generations()` | RAM-based max generations |
| 190 | `_detect_gpu_memory()` | Static — detects GPU VRAM via CuPy or nvidia-smi |
| 208 | `_calculate_agent_batch_size()` | Resource-aware batch size targeting 70% utilization |
| 237 | `_send_to_gui()` | Sends messages to GUI queue |
| 243 | `_send_metrics_json()` | Sends JSON metrics to GUI queue; **metrics reporting fix** (v0.9.7): now passes `previous_tier_index` from benchmark evaluator to `get_generation_size_limits()` so reported `tier_budget_mb` / `tier_ceiling_kb` match the actual active tier (previously defaulted to `-1`, always showing tier 0) |
| ~360 | `_snapshot_generation()` | Captures lightweight per-generation population snapshot (agent summaries, batch grouping) for GDV history; sends `GEN_SNAPSHOT:<gen>` via queue |
| 253 | `_enforce_gpu_safe_benchmark_size()` | Pads benchmark items below 4KB for GPU safety |
| 289 | `_sanitize_agent()` | Clamps agent parameters to safe bounds |
| 303 | `_create_initial_population()` | Parallel agent creation with deferred GPU init + bulk GPU finalization + **50/50 agent type split** (first half "compression", second half "anti_corruption"). Types survive checkpoint/restart via EvolvingAgent `__setstate__` migration |
| 358 | `_evaluate_population()` | Evaluates all agents in resource-aware batches; when `cpu_eval_workers > 1`, creates a `ProcessPoolExecutor` and uses `evaluate_population_pipelined()` for GPU+CPU parallel evaluation; falls back to sequential `evaluate_population_batch()` for single-worker mode. Applies novelty scoring; records batch agent IDs for snapshots. **Type-aware evaluation**: after clean eval, anti_corruption agents get a second pass on corrupted benchmark data; uses **central API** `benchmark_evaluator.get_anti_corruption_benchmark_items()` to obtain phase-appropriate corrupted items (eliminates duplicated phase logic). **Pipeline anti-corruption eval**: when `cpu_eval_workers > 1`, creates a **dedicated** `ProcessPoolExecutor` initialised with corruption items and uses `evaluate_population_pipelined()` for parallel anti-corruption evaluation; falls back to sequential on pipeline failure; the anti-corruption pool is separate from the clean-eval pool and is created/destroyed within the phased training block. `compression_fitness` and `robustness_fitness` are split; explicit `clean_items=` snapshot passed to prevent stale-data corruption. **Type-aware heritage recording**: compression agents record tricks based on clean fitness; anti_corruption agents only record tricks that survive corruption (based on `robustness_fitness`). **Phased training integration**: benchmark swap for anti-corruption eval is wrapped in try/finally so evaluator always restores original items; GitHub file fetcher guarded by `phased_enabled` flag + warning logging when unavailable or empty; `github_ratio` reset to 0 on GitHub failure so logs and MIX logic stay accurate; phase label appended with `[DEGRADED]` tag. **Type-aware ranking**: final `best_fit` returns best *compression* fitness only (drives stagnation detection and benchmark sizing); `_last_gen_best_robustness` stored for metrics JSON + GUI log; population sorted per-type with compression agents first (`group 1`) then anti-corruption (`group 0`) so `population[0]` is always the best compression agent |
| 524 | `start_evolution()` | Main evolution loop — evaluate → snapshot → breed → pre-generate next benchmark set → repeat; uses background threading to overlap benchmark data generation with evaluation/breeding. Accepts `_continue=True` to resume from `total_generations_elapsed` without recreating population. |
| ~501 | `continue_evolution()` | Resumes evolution from last generation — preserves population, extends `initial_num_generations` by `additional_gens`, optionally switches to infinite mode, then calls `start_evolution(_continue=True)` |
| — | `_should_refresh_at_gen()` | Centralised refresh-schedule check (every 3 gens + tier boundaries) |
| — | `_start_prefetch_benchmarks()` | Kicks off benchmark generation in a background thread via ThreadPoolExecutor; **captures live evaluator's `benchmark_items` (for growth-limiter prev_avg_size), `_current_complexity_tier`, and `_previous_tier_index`** before spawning; worker returns a dict `{items, complexity_tier, tier_index}` (not a bare list) |
| — | `_collect_prefetched_benchmarks()` | Non-blocking collection of pre-generated benchmark result; returns dict `{items, complexity_tier, tier_index}` or None if not ready; supports legacy bare-list format for safety |
| 622 | `_run_breeding_cycle()` | **Lineage-aware breeding** — Per-type elitism (top-1 from each type) → tournament selection → crossover producing TWO children per parent pair (both used, space permitting) → mutation + gradual novel-method evolution (gen 0-20: 5%→15% novel prob, gen 21-50: 15%→60%, gen 51+: 60%→80% cap) + heritage merging + heritage reuse (`HERITAGE_REUSE_PROB=0.65`) + strict budget-based 50/50 type balance (pre-computed `comp_budget`/`anti_budget` instead of weak post-hoc nudge) |
| 697 | `save_checkpoint()` | Saves evolution state + metrics (fitness, generation, pop_size) via CheckpointManager; returns True/False |

**Key State**: `self.generation_snapshots` — list of lightweight dicts per generation (no Q-tables); `self._last_eval_batch_agent_ids` — batch grouping from latest evaluation; `self._last_gen_best_robustness` — best robustness score from the most recent generation's anti-corruption agents (reset each gen, surfaced in metrics JSON + GUI log); `self._last_best_compression_ratio` — best agent's compression ratio (0-100%) from the most recent generation, used to gate complexity advancement via `COMPLEXITY_RATIO_GATES`; `self._last_gold_standard_win_rate` — fraction (0.0-1.0) of benchmark items where the AI beat ALL baseline compressors (gzip/bz2/lzma/zlib/zstd) in the most recent head-to-head benchmark; -1.0 means no data collected yet (gates bypassed); surfaced in METRICS_JSON; `self._last_training_phase` — human-readable training phase label (e.g. "Phase 1 (corruption-only)"), surfaced in WebUI; `self._last_corruption_level` — corruption fraction from the most recent anti-corruption evaluation, surfaced in WebUI. `_send_metrics_json()` **reads** `_current_complexity_tier` directly (no mutation) — only `generate_and_set_dynamic_benchmark_items()` can advance the tier. METRICS_JSON payload includes: `generation`, `fitness`, `ratio`, `benchmark_size`, `complexity_tier`, `complexity_value`, `tier_budget_mb`, `tier_ceiling_kb`, `best_robustness`, `training_phase`, `corruption_level`, `decomp_mismatches`, `items_evaluated`, `successful_compressions`, `gold_standard_win_rate`, `method_stats` (per-method bytes_saved/attempts/successes/profile), `novel_pipeline` (top agent's active pipeline name). Snapshot captures `novel_pipeline` and `has_novel_method` per agent for proper WebUI detection.

**Key Features**: Resource detection (RAM/CPU/GPU), 70% utilization targeting, stagnation detection with hypermutation, population-level novelty scoring, adaptive benchmark refresh using **current-benchmark fitness** (not stale all-time best — prevents runaway size growth), **background benchmark pre-generation** (next generation's data is produced in a background thread while the current generation evaluates/breeds, eliminating refresh wait time), **infinite mode** (`infinite_mode=True` bypasses generation limit — loop runs until stop event; `set_continuous_run_enabled(bool)` method allows toggling at runtime from GUI checkbox; GUI passes `infinite_mode` to constructor via checkbox state), per-generation snapshots for GDV (**capped at 500** via `_max_generation_snapshots` to bound memory on long/infinite runs), **GPU+CPU parallel pipeline** (ProcessPoolExecutor for CPU compression + batched GPU NN inference — true multi-core parallelism bypassing GIL), **auto-checkpointing** every 10 generations + final checkpoint at end of training run (thread-safe via `_checkpoint_lock`; **checkpoint rotation** via `_rotate_auto_checkpoints()` keeps only the 10 most recent auto-checkpoints, preventing disk bloat on infinite runs — only `auto_gen*` checkpoints are rotated; user-named and gold-standard checkpoints are preserved; rotation is guarded by `_checkpoint_lock` to prevent races with concurrent saves), **lineage-aware breeding** with "grandpapi" heritage merging and reuse, **gradual novel-method evolution** (scaffolding-heavy early → novel-heavy late), **agent specialization** with strict budget-based 50/50 compression/anti-corruption split + per-type elitism + both crossover children used, **type-aware corruption evaluation** (anti_corruption agents evaluated on corrupted benchmark data with generation-scaling corruption level; **pipeline anti-corruption eval** creates a dedicated ProcessPoolExecutor for parallel anti-corruption evaluation), **type-aware heritage recording** (compression agents record on clean fitness, anti_corruption agents record only corruption-surviving tricks via `robustness_fitness`), **phased training** (3-phase progression: corruption-only → blend corrupted + GitHub real-world files → mostly GitHub files; lazy-initialised GitHubFileFetcher with background download and in-memory caching; **centralised via `get_anti_corruption_benchmark_items()` API** — optimizer no longer duplicates phase logic), **printable-only corruption** (all injected chars restricted to ASCII 32-126; `_sanitize_to_printable()` post-processing catches control chars from RLE output), **progression-locked complexity** (complexity tracked via continuous `_complexity_pct` (0-100) that advances **1 % at a time** per benchmark refresh; advancement gated by piecewise-linear interpolated ratio gates (`_RATIO_GATE_KNOTS`) **AND gold standard win rate** gates (`_GS_GATE_KNOTS`) — the AI must prove compression ability before data gets harder; minimum dwell = 1 refresh per 1 % step; `DataComplexity` enum tier is derived from pct via `_tier_from_pct()` for backward compat; fitness is NOT used for gating; nudge logic removed — advancement handled entirely by `determine_target_complexity()`; **multi-step drops** — complexity can drop **multiple pct points in one refresh** when the ratio falls below the retention threshold (75 % of each pct's interpolated gate), ensuring difficulty always tracks real performance; data generation parameters (`run_likelihood`, `unique_focus`, `max_run_cap`) interpolate smoothly via `_interpolate_generation_params(pct)` — no hard jumps between tiers; `_send_metrics_json()` reads `_complexity_pct` directly as `complexity_value` (0-100 continuous); prefetch worker inherits `_complexity_pct`, `_current_complexity_tier`, `_refreshes_at_current_tier`, and `_previous_size_tier_refreshes` from live evaluator **AND syncs them all back** after swap-in — prevents display lag, double-advancement, and dwell-counter desync; ratio tracked via `_last_best_compression_ratio`; gold standard win rate tracked via `_last_gold_standard_win_rate`; **gold standard report** captured from `benchmark_generation()` return value every generation — win rate = fraction of items where AI beat ALL baselines; **gold standard immediate advancement** — when `gs_report.gold_standard == True` (AI beat ALL baselines on ALL items), immediately triggers a synchronous benchmark refresh to raise difficulty without waiting for the 3-gen schedule — ensures the AI keeps learning even at gen 500+ with easy data; sets `_gs_advanced_this_gen` flag to prevent double-advancement via the next-gen prefetch), **ratio-only difficulty scaling** (all size and complexity decisions driven by compression ratio (%) — fitness score is NOT used for scaling gates; `compute_continuous_benchmark_size()` uses ratio directly as interpolation parameter; `get_generation_size_limits()` uses generation threshold + ratio gate + **gold standard win rate gate** via `SIZE_TIER_GOLD_STANDARD_GATES` + **single-step advancement with 2-refresh dwell requirement** + **50 % hysteresis for tier drops**; size tier advancement/drop mirrors complexity dwell logic), **prefetch growth-limiter sync** (prefetch worker receives the live evaluator's `benchmark_items` so `prev_avg_size > 0` and the bidirectional `MAX_SIZE_GROWTH_FACTOR`/`MIN_SIZE_SHRINK_FACTOR` cap is enforced — without this the tmp evaluator had empty items, yielding `prev_avg_size = 0`, which bypassed the 2x growth cap and allowed item sizes to spike from floor to ceiling in one refresh), **full v0.9.7 checkpoint support** (EvolvingAgent `__getstate__`/`__setstate__` with backward-compatible migration from pre-v0.9.7 checkpoints)

**Key Dependencies**: `.individual_agent`, `.selection_methods`, `.mutation_methods`, `.crossover_methods`, `..reward_system`, `..checkpoint_manager`, `..utils.benchmark_evaluator`, `..utils.github_file_fetcher`, `..novel_compression_generator`

---

### evolution_core/individual_agent.py
**Lines**: ~300  
**Purpose**: Wrapper class for evolving AI agents in the genetic algorithm. Implements "grandpapi" lineage heritage — each agent carries a bounded list of ancestor tricks (proven pipelines) that are merged during breeding and reused by descendants.

**Constants**:
| Line | Name | Value |
|------|------|-------|
| 48 | `MAX_HERITAGE_ENTRIES` | `32` — max ancestor tricks kept per agent |

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 51 | `EvolvingAgent` | Wraps a `PuffinZipAI` instance with evolution metadata, heritage lineage, agent type, and dual fitness |

**EvolvingAgent Attributes**: `puffin_ai`, `agent_id` (uuid), `fitness`, `generation_born`, `parent_ids`, `evaluation_stats`, `heritage` (list of ancestor trick dicts), `agent_type` ("compression" or "anti_corruption"), `compression_fitness` (float), `robustness_fitness` (float)

**EvolvingAgent Methods**:
| Method | Description |
|--------|-------------|
| `clone()` | Deep-clones agent including heritage and agent_type |
| `get_puffin_ai()` | Returns wrapped PuffinZipAI instance |
| `set_fitness(f)` | Sets primary (clean/compression) fitness only — does NOT auto-update type-specific sub-scores |
| `set_compression_fitness(f)` | Sets `compression_fitness` sub-score only |
| `set_robustness_fitness(f)` | Sets `robustness_fitness` sub-score only — does NOT touch `self.fitness` so `get_fitness()` always returns clean fitness |
| `get_fitness()` | Returns overall fitness |
| `set_compression_fitness(score)` | Sets compression-specific fitness sub-score |
| `set_robustness_fitness(score)` | Sets robustness-specific fitness sub-score |
| `record_trick(trick_label, pipeline, discovery_seed, rle_min_run, fitness_when_learned, generation)` | Records a proven pipeline into heritage (bounded by MAX_HERITAGE_ENTRIES) |
| `merge_heritage(p1_heritage, p2_heritage)` | **Static** — Deduplicates two parents' heritage by `(ancestor_id, trick)` key, keeps top entries by fitness. Core of "grandpapi" inheritance |
| `get_best_heritage_pipeline()` | Returns highest-fitness heritage entry dict, or None |
| `__getstate__()` | Pickle serialization — ensures all v0.9.7 fields (heritage, agent_type, compression_fitness, robustness_fitness) are persisted |
| `__setstate__(state)` | Pickle deserialization — provides defaults for missing v0.9.7 fields when loading pre-v0.9.7 checkpoints |

**Pickle Behavior**: `__getstate__` explicitly includes all v0.9.7 fields with `setdefault`. `__setstate__` migrates pre-v0.9.7 checkpoints by defaulting missing fields: `heritage=[]`, `agent_type='compression'`, `compression_fitness=fitness`, `robustness_fitness=0.0`.

**Heritage Entry Dict Schema**: `{"trick": str, "pipeline": str, "discovery_seed": int, "rle_min_run": int, "fitness_when_learned": float, "ancestor_id": str, "generation": int}`

**Key Dependencies**: `..ai_core` (PuffinZipAI), `uuid`, `copy`, `traceback`

---

### evolution_core/breeding_methods.py
**Lines**: ~140  
**Purpose**: Advanced breeding/crossover for evolutionary optimization.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 7 | `complex_threshold_crossover()` | Crosses length thresholds between parents |
| ~65 | `fitness_weighted_parameter_crossover()` | Blends hyperparameters weighted by fitness |
| ~115 | `fitness_weighted_q_table_crossover()` | Blends Q-tables weighted by fitness |

**Key Dependencies**: `..config` (MAX_THRESHOLDS_COUNT_MERGED, MIN_THRESHOLDS_COUNT)

---

### evolution_core/crossover_methods.py
**Lines**: ~240  
**Purpose**: Q-table and parameter crossover operations.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 9 | `_resize_q_table()` | Resizes Q-table to match new state space |
| 23 | `q_table_single_point_crossover()` | Single-point Q-table crossover |
| 37 | `q_table_average_crossover()` | Element-wise Q-table averaging |
| 46 | `q_table_uniform_crossover()` | Uniform random Q-table crossover |
| 60 | `parameter_blend_crossover()` | Blends hyperparameters (LR, DR, ER) |
| 85 | `parameter_single_point_crossover()` | Single-point parameter crossover |
| 95 | `apply_crossover()` | Main dispatcher — selects and applies crossover strategy |

**Key Dependencies**: `..config`, `.breeding_methods`

---

### evolution_core/mutation_methods.py
**Lines**: ~340  
**Purpose**: Mutation operators for evolutionary optimization.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 5 | `mutate_parameter()` | Mutates a single numeric parameter with Gaussian noise |
| 22 | `mutate_thresholds()` | Mutates length thresholds (add/remove/shift) |
| 62 | `apply_mutations()` | Main mutation dispatcher — mutates Q-params, thresholds, RLE min run, **novel method pipeline** (rle_min_run ±1 @ 40%, discovery seed toggle @ 15%, pipeline swap @ 8%), and NN weights. Novel method mutations verified for invertibility before commit. |
| ~180 | `apply_hypermutation()` | Aggressive mutation for escaping stagnation |

**Key Features**:
- Preserves Q-values across threshold changes via interpolation.
- **Novel method mutation** (v0.9.7): Mutates pipeline parameters (rle_min_run, discovery_seed, pipeline name) with invertibility verification. Uses `NovelCompressionGenerator._build_pipeline()` to rebuild compress/decompress closures. Falls back to original method if verification fails.

---

### evolution_core/selection_methods.py
**Lines**: ~60  
**Purpose**: Parent selection strategies for breeding.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 4 | `tournament_selection()` | Tournament selection with configurable tournament size |
| 15 | `roulette_wheel_selection()` | Fitness-proportionate roulette selection |
| 27 | `rank_selection()` | Rank-based selection |
| 39 | `truncation_selection_for_breeding()` | Truncation selection  — top N% become parents |

---

## puffinzip_ai/gpu_core/ — GPU Acceleration

---

### gpu_core/\_\_init\_\_.py
**Lines**: 17  
**Purpose**: Package init — exports GPU classes and functions.

**Exports**: `PuffinZipAI_GPU`, `gpu_accelerated_rle_compress`, `gpu_accelerated_rle_decompress`, `array_to_gpu`, `array_to_cpu`, `get_gpu_memory_info`, `get_best_available_gpu_id`, `batch_update_q_table_gpu`, `get_batch_actions_gpu`

---

### gpu_core/gpu_ai_agent.py
**Lines**: ~510  
**Purpose**: GPU-accelerated AI agent extending PuffinZipAI — Q-table on GPU via CuPy, GPU RLE with CPU fallback.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| — | `PuffinZipAI_GPU` | Extends `PuffinZipAI` — GPU Q-table operations |

**PuffinZipAI_GPU Key Methods**:
| Method | Description |
|--------|-------------|
| `__init__(…, _defer_gpu_transfer=False)` | Parent init + cached GPU validation + optional deferred GPU setup |
| `_do_gpu_init()` | Uses `_validate_gpu_once()` cache for O(1) GPU setup instead of per-agent probing |
| `finalize_gpu_init()` | Finalizes GPU setup for agents created with `_defer_gpu_transfer=True` |
| `_initialize_gpu_device()` | Legacy per-agent GPU init (kept for compatibility, no longer called from `__init__`) |
| `_transfer_q_table_to_gpu()` | Transfers numpy Q-table to CuPy GPU array |
| `_reinitialize_state_dependent_vars()` | Parent reinit + GPU Q-table re-transfer |
| `_choose_action()` | GPU argmax on Q-table with CPU fallback |
| `_update_q_table()` | GPU Q-table update with CPU fallback |
| `_handle_item_processing_for_training()` | Full GPU-accelerated training pipeline with CPU fallback |
| `save_model()` | Syncs GPU Q-table to CPU before pickle save |
| `load_model()` | Loads CPU model then transfers Q-table to GPU |
| `clone_core_model()` | Deep-clones with GPU Q-table state preservation |

**Module-level Constants & Functions**:
| Name | Description |
|------|-------------|
| `CUPY_AVAILABLE` | Whether CuPy is importable |
| `NUMBA_AVAILABLE` | Whether Numba CUDA is available |
| `cp` | CuPy module reference (or None) |
| `_GPU_VALIDATION_CACHE` | Dict caching GPU device validation (validated, gpu_ok, gpu_id, device_name) |
| `_validate_gpu_once(target_device)` | Runs GPU health check once; all subsequent agents skip probing |

**Performance Notes**:
- GPU device init + health check runs **once** globally via `_validate_gpu_once()`, not per-agent.
- `_defer_gpu_transfer=True` allows bulk creation of agents on CPU, then batch GPU finalization.
- Population creation uses `ThreadPoolExecutor` for parallel CPU-side work (see `evolutionary_optimizer.py`).

**Key Dependencies**: `..ai_core` (PuffinZipAI), `.gpu_rle_interface`, `.gpu_model_utils`, `cupy`, `numba`

---

### gpu_core/gpu_model_utils.py
**Lines**: ~150 (including self-test)  
**Purpose**: GPU array transfer utilities.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 27 | `get_gpu_memory_info()` | Returns total/free/used VRAM for a GPU ID |
| 45 | `array_to_gpu()` | Transfers numpy array to CuPy GPU array |
| 62 | `array_to_cpu()` | Transfers CuPy GPU array back to numpy |
| 79 | `get_best_available_gpu_id()` | Returns best available GPU ID (currently returns 0) |

**Key Dependencies**: `cupy` (optional)

---

### gpu_core/gpu_rle_interface.py
**Lines**: 573  
**Purpose**: GPU-accelerated RLE compression/decompression with workspace management and CPU fallback.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~100 | `_get_total_mem_bytes()` | Extracts total GPU memory from device properties |
| ~107 | `_ensure_workspace_allocation()` | Allocates/reuses GPU workspace buffer with exponential retry |
| ~195 | `_acquire_workspace_slice()` | Returns a workspace slice of required length |
| ~205 | `_encode_string_to_codepoints()` | Converts string to UTF-32 codepoint array |
| ~213 | `_codepoints_to_string()` | Converts codepoint array back to string |
| ~219 | `_build_run_boundaries_gpu()` | GPU-based run-length boundary detection |
| ~230 | `_parse_compressed_segments()` | Parses marker-framed RLE compressed text into segments |
| ~330 | `gpu_accelerated_rle_compress()` | GPU RLE compression (simple method) with CPU fallback |
| ~420 | `gpu_accelerated_rle_decompress()` | GPU RLE decompression (simple method) with CPU fallback |

**Constants**:
| Name | Description |
|------|-------------|
| `_GPU_WORKSPACE_BUFFERS` | Dict of per-GPU workspace CuPy arrays |
| `_GPU_WORKSPACE_LOCK` | Threading lock for workspace access |

**Key Dependencies**: `cupy`, `numba`, `..rle_utils`, `..config` (GPU_RLE_* settings)

---

### gpu_core/gpu_training_utils.py
**Lines**: ~180 (including self-test)  
**Purpose**: Batch Q-table operations on GPU for evolutionary optimization.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 23 | `batch_update_q_table_gpu()` | Batch TD(0) Q-table update on GPU via CuPy |
| 72 | `get_batch_actions_gpu()` | Batch epsilon-greedy action selection from GPU Q-table |

**Key Dependencies**: `cupy`, `numpy`

---

## puffinzip_ai/nn_core/ — Neural Network (DQN) Agents

Provides Deep Q-Network (DQN) agents as an alternative to the tabular Q-table approach.
Requires **PyTorch** (`pip install torch`).  Enabled when `NN_ENABLED = True` in config.

**Architecture**: Each agent owns a Dueling DQN with NoisyNet exploration, Multi-Head Self-Attention,
Adaptive Feature Gates, and Residual Blocks (default: 20→256→256→4 ≈ 200K params).
Uses Prioritized Experience Replay (PER), Double DQN, N-step returns, CosineAnnealing LR,
and soft Polyak target updates.

**Integration**: `PuffinZipAI_NN` extends `PuffinZipAI` — passes `isinstance()` checks, so the
entire evolutionary pipeline (crossover, mutation, selection, GUI, WebUI) works transparently.

---

### nn_core/\_\_init\_\_.py
**Purpose**: Package init.  Sets `TORCH_AVAILABLE` flag; logs PyTorch version and CUDA status.

---

### nn_core/dqn_model.py
**Purpose**: Dueling DQN with NoisyNet, multi-head attention, adaptive feature gates, and residual blocks.

**Classes**:
| Name | Description |
|------|-------------|
| `NoisyLinear(nn.Module)` | Factorised Gaussian NoisyNet linear layer (Fortunato et al. 2018) |
| `FeatureAttentionBlock(nn.Module)` | Multi-head self-attention over feature dimensions |
| `AdaptiveFeatureGate(nn.Module)` | Sigmoid-gated feature importance weighting |
| `SharedEncoder(nn.Module)` | Attention → Gate → Residual blocks encoder (shared by value & advantage heads) |
| `DQNNetwork(nn.Module)` | Dueling DQN: SharedEncoder → Value stream + Advantage stream → combined Q-values |

**DQNNetwork Key Methods**:
| Method | Description |
|--------|-------------|
| `forward(state)` | Returns Q-values via dueling aggregation (V + A − mean(A)) |
| `clone()` | Deep-copy (new parameters, same architecture) |
| `hard_sync_from(source)` | Copy all parameters from source (target-net sync) |
| `soft_sync_from(source, tau)` | Polyak averaging: θ_target ← τ·θ_source + (1−τ)·θ_target |
| `reset_noise()` | Resample noise in all NoisyLinear layers |
| `get_noise_magnitude()` | Average absolute noise across NoisyLinear layers |
| `freeze_encoder() / unfreeze_encoder()` | Toggle encoder gradient computation |
| `get_architecture_summary()` | Dict summary: params, memory, layers, attention heads |
| `get_flat_params() / set_flat_params()` | Serialise/deserialise parameters as 1-D tensor |
| `parameter_count() / memory_bytes()` | Diagnostics |

---

### nn_core/replay_buffer.py
**Purpose**: Prioritized Experience Replay (PER) with SumTree for O(log N) proportional sampling.

**Classes**:
| Name | Description |
|------|-------------|
| `SumTree` | Binary tree for efficient priority management and proportional sampling |
| `Transition` | NamedTuple: `(state, action, reward, next_state, done)` |
| `ReplayBuffer` | PER buffer with importance-sampling weight correction |

**ReplayBuffer Key Methods**:
| Method | Description |
|--------|-------------|
| `push(state, action, reward, next_state, done)` | Add transition with max priority |
| `sample(batch_size)` | Return `(states, actions, rewards, next_states, dones, indices, is_weights)` — proportional sampling with IS weights |
| `update_priorities(indices, td_errors)` | Update priorities based on TD errors |
| `is_ready(min_size)` | True when buffer ≥ min_size |

---

### nn_core/nn_agent.py
**Purpose**: `PuffinZipAI_NN` — Advanced DQN-based compression agent. Drop-in replacement for
`PuffinZipAI` / `PuffinZipAI_GPU`.

**State Features** (20-dim continuous vector, `extract_features()`):
| # | Feature | Normalisation |
|---|---------|---------------|
| 0 | log-length | `log2(len+1) / 20` |
| 1 | unique-char ratio | `unique_chars / 256` |
| 2 | max-run ratio | `max_run / len` |
| 3 | byte entropy | Shannon entropy / 8 |
| 4 | avg-run ratio | `avg_run / len` |
| 5 | digit fraction | digits / len |
| 6 | alpha fraction | alphas / len |
| 7 | space fraction | spaces / len |
| 8 | punctuation fraction | punctuation / len |
| 9 | bigram entropy | bigram Shannon entropy / 16 |
| 10 | median run length | median / len |
| 11 | run length variance | sqrt(variance) / len |
| 12 | repeated block ratio | 4-byte repeated blocks / total |
| 13 | byte range spread | (max\_byte − min\_byte) / 255 |
| 14 | RLE compressibility estimate | simulated ratio |
| 15 | uppercase ratio | uppercase / alpha\_count |
| 16 | ASCII fraction | ASCII chars / len |
| 17 | char frequency skew | std / mean of char frequencies |
| 18 | length category | bucketed: tiny/small/medium/large/huge |
| 19 | padding | `0.0` |

**Training Enhancements**:
| Feature | Description |
|---------|-------------|
| Double DQN | Policy net selects action, target net evaluates Q-value |
| N-step Returns | 3-step bootstrapped returns via `_NStepBuffer` |
| PER + IS Weights | Importance-sampling weighted Huber loss |
| CosineAnnealingWarmRestarts | LR scheduler with T_0=5000, eta_min=1e-6 |
| Soft Target Updates | Polyak averaging (τ=0.005) instead of hard sync |
| NoisyNet Exploration | State-dependent exploration + ε-greedy fallback |
| Training Metrics | Tracks loss, avg_q, td_error, grad_norm, noise_magnitude, LR, beta (deque×500) |

**Overridden Methods**:
| Method | Change |
|--------|--------|
| `_get_state_representation()` | Extracts 20-feature vector + caches it |
| `_choose_action()` | NoisyNet + ε-greedy on Dueling DQN policy-net output |
| `_update_q_table()` | PER sampling → Double DQN loss → IS-weighted gradient → soft target sync |
| `clone_core_model()` | Deep-clones NN weights + config; fresh replay buffer |
| `save_model()` / `load_model()` | Saves `.npy` (base) + `_nn.pt` (PyTorch weights + scheduler) |
| `__getstate__()` / `__setstate__()` | Pickle support with backward compat for old checkpoints |

**Config Constants** (in `config.py`):
| Constant | Default | Description |
|----------|---------|-------------|
| `NN_ENABLED` | `True` | Master switch for DQN agents |
| `NN_STATE_FEATURE_DIM` | `20` | Continuous feature vector size |
| `NN_HIDDEN_SIZES` | `[256, 256]` | Dueling DQN hidden layer widths |
| `NN_REPLAY_BUFFER_CAPACITY` | `50000` | Per-agent PER buffer |
| `NN_REPLAY_MIN_SIZE` | `128` | Min entries before training |
| `NN_TRAIN_BATCH_SIZE` | `128` | Mini-batch for gradient updates |
| `NN_TARGET_NETWORK_UPDATE_FREQ` | `200` | Fallback hard-sync interval |
| `NN_LEARNING_RATE` | `3e-4` | AdamW LR |
| `NN_GRAD_CLIP_NORM` | `1.0` | Max gradient norm |
| `NN_PER_ALPHA` | `0.6` | PER prioritisation exponent |
| `NN_PER_BETA_START` | `0.4` | IS correction initial beta |
| `NN_PER_BETA_FRAMES` | `100000` | Beta annealing horizon |
| `NN_COSINE_LR_T_MAX` | `5000` | Cosine restart period |
| `NN_COSINE_LR_ETA_MIN` | `1e-6` | Min LR |
| `NN_SOFT_TARGET_TAU` | `0.005` | Polyak averaging coefficient |
| `NN_NSTEP_RETURNS` | `3` | N-step return horizon |
| `NN_DROPOUT` | `0.1` | Dropout in residual blocks |
| `NN_ATTENTION_HEADS` | `4` | Multi-head attention heads |
| `NN_NOISY_SIGMA` | `0.5` | NoisyNet initial sigma |
| `NN_MUTATION_WEIGHT_NOISE_STD` | `0.02` | Gaussian noise σ for weight mutation |
| `NN_MUTATION_WEIGHT_PROB` | `0.1` | Per-layer mutation probability |
| `NN_CROSSOVER_LAYER_SWAP_PROB` | `0.5` | Per-layer swap probability in crossover |

**Evolutionary Integration** (added to existing crossover/mutation modules):
- `crossover_methods._apply_nn_crossover()` — Layer-wise uniform weight swap
- `mutation_methods.mutate_nn_weights()` — Additive Gaussian noise on policy-net
- `mutation_methods.hypermutate_nn_weights()` — 3× noise for stagnation escape

---

## puffinzip_ai/utils/ — Utility Modules

---

### utils/\_\_init\_\_.py
**Lines**: 9  
**Purpose**: Package marker — empty (no exports).

---

### utils/benchmark_evaluator.py
**Lines**: ~1000  
**Purpose**: Population fitness evaluation with dynamic benchmark generation, generation-aware accelerated size scaling, baseline-subtracted scoring, method diversity adjustment, and centralised anti-corruption benchmark API.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~70 | `DataComplexity` | Enum: `VERY_SIMPLE`, `SIMPLE`, `MODERATE`, `COMPLEX`, `VERY_COMPLEX`, `USER_DEFINED_LARGE` |
| ~140 | `BenchmarkItemEvaluator` | Main evaluator class |

**BenchmarkItemEvaluator Key Methods**:
| Method | Description |
|--------|-------------|
| `__init__()` | Initializes with performance-tuned throttle params |
| `_generate_one_dynamic_item()` | Generates one benchmark item at target complexity/size using fast bulk chunk-based generation (no agent delegation) |
| `corrupt_compressed_data(compressed_text, corruption_level)` | **Static** — Injects bit flips (50%), byte insertions (25%), byte deletions (25%) at random positions in compressed data; all injected characters restricted to **printable ASCII (32-126)** — no null bytes or control characters |
| `_sanitize_to_printable(text)` | **Static** — Replaces any non-printable character (ord < 32 or ord > 126) with a deterministic printable substitute `chr(32 + (ord(ch) % 95))`; applied as post-processing after corruption to catch control chars from RLE-compressed input |
| `generate_corrupted_benchmark_items(clean_items, corruption_level, garbage_fraction)` | Compresses clean items with simple RLE, optionally injects garbage into clean data before compression when `garbage_fraction > 0` (wires `inject_garbage_into_clean_data`), corrupts the output, then applies `_sanitize_to_printable()` post-processing; returns list of `(corrupted_compressed, original_text)` tuples |
| `inject_garbage_into_clean_data(item_text, garbage_fraction)` | Inserts random **printable ASCII (32-126)** garbage characters into uncompressed data at random positions; now called from `generate_corrupted_benchmark_items()` when `garbage_fraction > 0` (phase 2+ corruption training) |
| `get_anti_corruption_benchmark_items(generation_num, clean_items, github_items, ...)` | **Central API** — Encapsulates all phased training logic: determines phase from generation number, scales corruption level, ramps `garbage_fraction` (0 in P1, up to 0.03 in P2+), blends corrupted + GitHub items at phase-appropriate ratios, handles GitHub fallback with `[DEGRADED]` tag; returns `(items, phase_label, corruption_level)` tuple |
| `determine_target_complexity(fitness, ratio, gold_standard_win_rate)` | **Continuous 1 %-at-a-time advancement** — `_complexity_pct` (0-100) advances by 1 % per qualifying refresh, gated by piecewise-linear interpolated ratio gates (`_RATIO_GATE_KNOTS`) and gold-standard gates (`_GS_GATE_KNOTS`). Minimum dwell = 1 refresh per step. `_current_complexity_tier` (DataComplexity enum) is derived from pct via `_tier_from_pct()` for backward compat. Can **drop multiple pct points** in one refresh when ratio falls below 75 % retention threshold. Only called from `generate_and_set_dynamic_benchmark_items()`. |
| `generate_and_set_dynamic_benchmark_items(..., best_compression_ratio, gold_standard_win_rate)` | Generates full benchmark set with generation-aware size scaling; passes `best_compression_ratio` and `gold_standard_win_rate` to `determine_target_complexity()` and `get_generation_size_limits()` / `compute_continuous_benchmark_size()`; stores ratio in `self._last_best_compression_ratio`; **per-item ceiling enforcement** (v0.9.7): each item’s ±30% variance target is clamped to `tier_ceiling` so no item exceeds the tier’s per-item ceiling; **budget enforcement fix** (v0.9.7): the budget-enforcement call to `get_generation_size_limits()` now passes `gold_standard_win_rate` to prevent the budget from using a higher tier than item sizing (previously defaulted to -1.0, bypassing the gold standard gate); **data generation fix**: VERY_SIMPLE `max_run_cap` = 80-150, SIMPLE `max_run_cap` = 40-80 (previously uncapped at full item length, producing trivially compressible data with 99.9%+ ratios) |
| `load_benchmark_data()` | Loads static benchmark JSON/text files |
| `get_total_benchmark_size_bytes()` | Returns total byte size of all items |
| `set_custom_benchmark_items()` | Sets user-provided benchmark items |
| `evaluate_agent_fitness()` | Evaluates one agent on all benchmark items; applies baseline subtraction, size scaling, diversity adjustment, and online Q-learning. Tracks `total_original_bytes` / `total_compressed_bytes` for overall compression ratio. **Per-method tracking** (v0.9.7): `method_bytes_saved`, `method_attempts`, and `method_successes` dicts track bytes saved, total attempts, and successful compressions per method (RLE, AdvancedRLE, NovelMethod, ReferenceMethod). Surfaced via METRICS_JSON → WebUI. |
| `evaluate_population_batch()` | Evaluates a population with throttling |
| `evaluate_population_pipelined(pop, cpu_pool)` | **GPU+CPU pipeline** — Phase 1 (GPU): batch forward pass per agent via `batch_choose_actions`. Phase 2 (CPU): distributes per-item compression across `ProcessPoolExecutor` workers for true multi-core parallelism (bypasses GIL). Phase 3 (GPU): batch Q-learning via `batch_push_experiences`. GPU work for agent N+1 overlaps with CPU work for agent N. **Per-method tracking** (v0.9.7 fix): `method_bytes_saved`, `method_attempts`, and `method_successes` dicts now tracked in the pipelined path's stats aggregation (was previously missing — only the sequential path had them, causing "No method data yet" in WebUI when workers > 1). |

**Pipeline Worker Functions** (module-level, picklable for `ProcessPoolExecutor`):
| Function | Description |
|----------|-------------|
| `_pipeline_worker_init(items)` | Initialises child process with benchmark items + compression functions. Called once per worker via `ProcessPoolExecutor(initializer=...)`. |
| `_compress_single_item(args)` | Compress → decompress → reward for one item in a child process. Receives `(item_idx, action_name, rle_min_run)`. Returns compact result dict (no full text — minimises IPC). |

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| ~110 | `get_generation_size_limits()` | Maps (generation, previous_tier_index, **best_compression_ratio**, **refreshes_at_tier**, **gold_standard_win_rate**) → (total_budget, per_item_ceiling, active_tier_index). Uses generation threshold + **ratio gating** + **single-step advancement with 2-refresh dwell** + **50 % hysteresis drops**. Tier 0 always retained. |
| ~150 | `compute_continuous_benchmark_size()` | Maps (**best_compression_ratio**, generation, previous_tier_index, **gold_standard_win_rate**) → (min_size, max_size, tier_index). Ratio (0-100%) maps to interpolation parameter `t` (0–1). **Gold standard dampening** (v0.9.7): when `gold_standard_win_rate ≥ 0`, caps `t` to `0.10 + 0.90 × win_rate` so sizes stay small when the AI can’t beat baselines (0% wins → t≤0.10, 50% → t≤0.55, 100% → no cap). **Warm-up cap**: when no gold standard data exists (`win_rate < 0`), caps `t` at 0.35 instead of no dampening. **Tight range** (v0.9.7): ±15% around target_center (was ±50%) to prevent random-walk drift. Exponential interpolation between floor and tier ceiling, with **bidirectional growth limiting** (2× max growth, 0.5× max shrink per refresh). **Ceiling hard-cap** (v0.9.7): `target_center` is clamped to `effective_ceiling` after growth limiter so items never exceed the tier ceiling. |

**Constants** (selected):
| Name | Value |
|------|-------|
| `DEFAULT_BENCHMARK_REPETITIONS` | `1` |
| `DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET` | `20` |
| `EVALUATION_FAIL_REWARD` | `-100.0` |
| `MAX_ITEM_PROCESS_TIME_SEC` | `30.0` |
| `CONTINUOUS_SIZE_FLOOR_BYTES` | `64 * 1024` (64 KB) — must exceed `SIZE_BONUS_BASE_THRESHOLD` (32 KB) |
| `CONTINUOUS_SIZE_CEILING_BYTES` | `10 * 1024 * 1024` (10 MB) — max per item at highest generation tier |
| `GENERATION_SIZE_TIERS` | List of (gen, budget, ceiling, _legacy_fitness): Gen 0→5MB, Gen 5→10MB, Gen 10→20MB, Gen 15→40MB, Gen 20→70MB, Gen 25→120MB. Fitness field is legacy/unused — ratio gates drive advancement. |
| `SIZE_TIER_RATIO_GATES` | List of minimum compression ratios (%) required to advance TO each size tier: [0%, 20%, 35%, 50%, 60%, 70%]. SOLE gate for size tier advancement (generation threshold must also be met). |
| `TOTAL_BENCHMARK_BUDGET_BYTES` | `20 * 1024 * 1024` (20 MB) — fallback when tiers don't apply |
| `MAX_SIZE_GROWTH_FACTOR` | `2.0` — max growth per refresh (conservative to prevent runaway size scaling) |
| `MIN_SIZE_SHRINK_FACTOR` | `0.5` — max shrinkage per refresh (prevents freefall after size spike) |
| `TIER_HYSTERESIS_MARGIN` | `2.0` — legacy, no longer used (ratio-based gating uses 0.5× ratio-drop threshold instead) |
| `COMPLEXITY_FITNESS_THRESHOLDS` | Legacy dict, no longer used for gating. All advancement/drops use `COMPLEXITY_RATIO_GATES` + `COMPLEXITY_GOLD_STANDARD_GATES` exclusively. |
| `COMPLEXITY_RATIO_GATES` | Dict: SIMPLE=25%, MODERATE=45%, COMPLEX=60%, VERY_COMPLEX=70% — legacy tier-based ratio gates (still defined for reference). Actual gating now uses piecewise-linear interpolation via `_RATIO_GATE_KNOTS` across 0-100 pct range. |
| `COMPLEXITY_GOLD_STANDARD_GATES` | Dict: SIMPLE=10%, MODERATE=30%, COMPLEX=50%, VERY_COMPLEX=70% — legacy tier-based gold standard gates (still defined for reference). Actual gating uses `_GS_GATE_KNOTS` interpolation. |
| `_RATIO_GATE_KNOTS` | Piecewise-linear knots for continuous ratio gating: (0,0%), (20,25%), (40,45%), (60,60%), (80,70%), (100,80%) |
| `_GS_GATE_KNOTS` | Piecewise-linear knots for continuous gold standard gating: (0,0.0), (20,0.10), (40,0.30), (60,0.50), (80,0.70), (100,0.80) |
| `SIZE_TIER_GOLD_STANDARD_GATES` | List aligned to `GENERATION_SIZE_TIERS`: [0%, 10%, 20%, 30%, 50%, 60%] — minimum gold standard win rate for size tier advancement; mirrors `COMPLEXITY_GOLD_STANDARD_GATES` for size tiers; tier 1 now requires ≥10% (was 0%); gate blocks advancement when no gold standard data exists (`win_rate < 0`) |
| `COMPLEXITY_LENGTH_RANGES_BYTES` | Dict mapping DataComplexity → (min, max) byte ranges |

**Key Dependencies**: `..ai_core`, `..rle_utils`, `..reward_system`, `..config`, `.performance_tuner`

---

### utils/github_file_fetcher.py
**Lines**: ~440  
**Purpose**: Downloads real-world text files from trusted GitHub repositories for use as benchmark training data. Enforces trust via a curated repo allowlist + star-count threshold, normalises files to a configurable byte range, and caches everything locally to avoid repeated API calls.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| ~130 | `GitHubFileFetcher` | Main fetcher class — downloads, caches, and serves GitHub files as benchmark items |

**GitHubFileFetcher Key Methods**:
| Method | Description |
|--------|-------------|
| `__init__()` | Resolves cache dir, size constraints, trusted repos, API token (env var → config → None), loads cache index |
| `fetch_from_repos(repos, max_per_repo, max_total)` | Downloads files from allowlisted repos into local cache; skips cached/fresh files; respects API rate limits |
| `get_benchmark_items(count, auto_fetch, target_min, target_max)` | Returns list of file-content strings from cache; auto-fetches if cache is insufficient; size-filters and truncates to range |
| `get_cached_count()` | Returns number of files in local cache |
| `clear_cache()` | Deletes all cached files and index |
| `cache_stats()` | Returns summary dict: total_files, total_bytes, per-repo counts |

**Internal Methods**:
| Method | Description |
|--------|-------------|
| `_check_rate_limit()` | Checks GitHub API remaining quota before fetching |
| `_verify_repo_trusted(repo)` | Returns True if repo is on allowlist or has ≥ `min_stars` stars |
| `_list_repo_files(repo)` | Uses Git Trees API (recursive, single call) to list all matching files in a repo; filters by extension, size range, and path (skips vendored/generated) |
| `_download_file(repo, file_path)` | Downloads raw file content via `raw.githubusercontent.com` (no API quota cost for public repos) |
| `_is_safe_text(content)` | Rejects binary-looking content (high ratio of control chars) |
| `_truncate_to_size(content, max_bytes)` | Truncates at line boundary to fit within size limit |

**Trust Model**:
- Curated allowlist in `DEFAULT_TRUSTED_REPOS` (17 high-star repos: cpython, flask, requests, django, fastapi, node, express, TypeScript, rust, deno, go, github/docs, mdn/content, home-assistant, ansible, linux, git)
- Auto-discovery via star-count threshold (`GITHUB_MIN_STARS`, default 500)
- File extension whitelist (`.py`, `.js`, `.ts`, `.md`, `.txt`, etc.)
- Content safety check (rejects high control-char ratio)
- Path blocklist (vendor/, node_modules/, dist/, build/, .min., generated, fixture, __pycache__, migrations/, locale/, .lock)

**Cache**:
- Directory: `data/github_cache/` (configurable via `GITHUB_CACHE_DIR`)
- Index file: `_index.json` — maps SHA256(repo:path)[:24] → {repo, path, size, fetched_at}
- Staleness: re-fetches after 7 days (`MAX_CACHE_AGE_SECONDS`)

**Constants**:
| Name | Value |
|------|-------|
| `DEFAULT_TRUSTED_REPOS` | 17 curated repos (see Trust Model above) |
| `DEFAULT_FILE_EXTENSIONS` | 26 text/code file extensions |
| `CACHE_INDEX_FILENAME` | `"_index.json"` |
| `MAX_CACHE_AGE_SECONDS` | `604800` (7 days) |

**Config Constants** (in `config.py`):
| Name | Default | Description |
|------|---------|-------------|
| `GITHUB_CACHE_DIR` | `data/github_cache/` | Local cache directory |
| `GITHUB_TARGET_FILE_SIZE_MIN` | `1024` (1 KB) | Minimum file size to fetch |
| `GITHUB_TARGET_FILE_SIZE_MAX` | `51200` (50 KB) | Maximum file size to fetch |
| `GITHUB_API_TOKEN` | `os.environ.get('GITHUB_TOKEN')` | Optional API token for higher rate limits |
| `GITHUB_MIN_STARS` | `500` | Star threshold for auto-trusted repos |
| `GITHUB_FETCH_TIMEOUT` | `15` | HTTP request timeout (seconds) |
| `GITHUB_CACHE_MAX_FILES` | `500` (env: `PUFFIN_CACHE_MAX_FILES`) | Max cached files before LRU eviction |
| `GITHUB_CACHE_MAX_MB` | `200` (env: `PUFFIN_CACHE_MAX_MB`) | Max cache size in MB before LRU eviction |
| `GITHUB_FILE_EXTENSIONS` | 26 extensions | File extension whitelist |
| `GITHUB_TRUSTED_REPOS` | 17 repos | Curated allowlist |

**Phased Training Config** (in `config.py`):
| Name | Default | Description |
|------|---------|-------------|
| `PHASED_TRAINING_ENABLED` | `True` | Enable 3-phase training progression |
| `PHASED_TRAINING_PHASE1_END` | `10` | Last gen of corruption-only phase |
| `PHASED_TRAINING_PHASE2_END` | `30` | Last gen of blended phase |
| `PHASED_TRAINING_PHASE3_GITHUB_RATIO` | `0.80` | Fraction of items from GitHub in phase 3 |
| `PHASED_TRAINING_GITHUB_ITEM_COUNT` | `20` | Number of GitHub files to blend in |

**Key Dependencies**: `..config`, `..logger`, `requests`, `hashlib`, `json`

---

### utils/benchmark_generator.py
**Lines**: ~250  
**Purpose**: Generates benchmark dataset files (numeric content) for training.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `generate_long_runs()` | Generates repeated digit runs |
| — | `generate_sequential_numbers()` | Generates sequential number strings |
| — | `generate_random_spaced_numbers()` | Generates random numbers with spacing |
| — | `generate_blocks_of_numbers()` | Generates blocks of repeated digits |
| — | `generate_mixed_numeric_content()` | Generates mixed patterns |
| — | `create_benchmark_files()` | Creates N benchmark files in a directory |
| — | `main_generate()` | Entry point — generates 75 files (70 standard + 5 large) |

**Constants**:
| Name | Value |
|------|-------|
| `NUM_FILES_TO_GENERATE` | `75` |
| `MIN_FILE_SIZE_KB` / `MAX_FILE_SIZE_KB` | `2` / `256` |
| `LARGE_FILE_SIZE_KB_MIN` / `LARGE_FILE_SIZE_KB_MAX` | `512` / `1024` |

**Key Dependencies**: `..logger`, `..config`

---

### utils/hardware_detector.py
**Lines**: ~280  
**Purpose**: Detects CPU, GPU hardware and provides processing device options for GUI.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `get_cpu_info()` | Detects CPU name via WMIC (Win), lscpu (Linux), sysctl (macOS) |
| — | `get_available_gpus_info()` | Detects GPUs via CuPy and/or Numba; returns list of dicts |
| — | `get_processing_device_options()` | Returns [(display_name, value)] for GUI combobox |

**Module-level Constants**:
| Name | Description |
|------|-------------|
| `PSUTIL_AVAILABLE` | Whether psutil is importable |
| `CUPY_AVAILABLE` | Whether CuPy is functional (runtime-tested) |
| `NUMBA_CUDA_AVAILABLE` | Whether Numba CUDA is available |
| `NVCC_AVAILABLE` | Whether nvcc is on PATH |
| `NVCC_VERSION_INFO` | nvcc version string if available |

**Key Dependencies**: `psutil`, `cupy` (optional), `numba` (optional), `platform`, `subprocess`

---

### utils/performance_tuner.py
**Lines**: ~175  
**Purpose**: Hardware-adaptive performance tuning with THREE tiers: LOW_END, BALANCED, HIGH_END.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `_simple_cpu_benchmark()` | Runs micro CPU benchmark (500K iterations) |
| — | `get_system_specs()` | Returns CPU cores and RAM via psutil |
| — | `suggest_performance_tier()` | Auto-detects tier based on benchmark + hardware specs |
| — | `get_tuned_parameters()` | Returns tuned parameter dict for given tier |

**Constants**:
| Name | Description |
|------|-------------|
| `DEFAULT_TUNABLE_PARAMS` | Default balanced throttle params |
| `PERFORMANCE_TIERS` | Dict of `LOW_END`, `BALANCED`, `HIGH_END` parameter sets |

**Key Dependencies**: `psutil`

---

### utils/settings_manager.py
**Lines**: ~340  
**Purpose**: Reads/writes config.py values and GUI state persistence.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| — | `get_config_values()` | Reads all EDITABLE_SETTINGS from config.py via regex parsing |
| — | `save_config_values()` | Writes updated settings back to config.py (in-place) |
| — | `load_gui_state()` | Loads GUI window state from gui_state.json |
| — | `save_gui_state()` | Saves GUI window state to gui_state.json |

**Constants**:
| Name | Description |
|------|-------------|
| `EDITABLE_SETTINGS` | Large dict defining all GUI-editable settings with types, defaults, labels, tooltips, min/max bounds |
| `GUI_STATE_FILENAME` | `"gui_state.json"` |
| `CONFIG_FILE_PATH` | Resolved path to config.py |
| `GUI_STATE_FILE_PATH` | Resolved path to gui_state.json |

---

## puffinzip_gui/ — Tkinter Desktop GUI

---

### puffinzip_gui/\_\_init\_\_.py
**Lines**: 22  
**Purpose**: Package init — imports all submodules and exports `PuffinZipApp`.

**Exports**: `PuffinZipApp`, `chart_utils`, `settings_gui`, `secondary_main_app`, `gui_utils`, `gui_style_setup`, `gui_layout_setup`, `generational_data_viewer`

---

### puffinzip_gui/primary_main_app.py
**Lines**: 1562  
**Purpose**: Main Tkinter application class — the central hub for all GUI functionality.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 149 | `PuffinZipApp(tk.Tk)` | Main application window |

**PuffinZipApp Key Methods** (56 total):
| Line | Name | Description |
|------|------|-------------|
| 151 | `_coerce_bool()` | Static — coerces value to bool |
| 166 | `__init__()` | Initializes theme, fonts, AI agent, optimizer, GUI layout, queue polling |
| 358 | `_setup_gui_layout_new()` | Delegates to gui_layout_setup + populates all tabs |
| 393 | `_cancel_analysis_refresh()` | Cancels periodic chart refresh |
| 403 | `_schedule_analysis_refresh_if_needed()` | Schedules chart auto-refresh |
| 412 | `_on_benchmark_strategy_change()` | Handles benchmark strategy combobox change |
| 422 | `_update_els_button_states()` | Enables/disables evolution buttons based on state |
| 468 | `_apply_els_adaptation()` | Applies bottleneck adaptation strategy |
| 482 | `apply_low_bottleneck()` | Applies low (conservative) bottleneck |
| 485 | `apply_medium_bottleneck()` | Applies medium bottleneck |
| 488 | `apply_high_bottleneck()` | Applies high (aggressive) bottleneck |
| 494 | `_update_els_chart_with_current_filters()` | Refreshes evolution chart with filter checkboxes |
| 497 | `_update_els_chart()` | Plots evolution fitness chart |
| 532 | `_populate_evolution_controls_tab()` | Populates Evolution Controls tab |
| 545 | `_populate_evolution_analytics_tab()` | Populates Evolution Analytics tab with charts |
| 583 | `_populate_gdv_tab()` | Populates Generational Deep Dive tab |
| 599 | `_populate_changelog_tab()` | Populates Change Log tab |
| 611 | `_populate_settings_tab_content()` | Populates Settings tab |
| 629 | `reload_and_apply_theme()` | Reloads config, rebuilds fonts and styles |
| 672 | `_handle_critical_setting_change()` | Handles device change (CPU ↔ GPU) |
| 696 | `_handle_critical_error()` | Shows error dialog; optionally exits |
| 730 | `on_frame_configure()` | Scrollable canvas frame resize handler |
| 746 | `_check_gui_queue()` | Polls GUI output queue for messages from AI threads |
| 806 | `log_message()` | Appends message to scrolled text log |
| 821 | `_log_to_els_console()` | Logs to evolution-specific console |
| 874 | `_ai_task_thread_wrapper()` | Thread wrapper for AI tasks with cleanup |
| 974 | `_sync_els_continuous_config()` | Syncs continuous mode config to optimizer |
| 991 | `_rehydrate_els_history_from_optimizer()` | Rebuilds fitness history from optimizer state |
| 1020 | `_start_ai_task()` | Starts AI task in background thread |
| 1049 | `request_task_stop()` | Signals stop event to running task |
| 1063 | `browse_folder()` | Opens folder browser dialog |
| 1097 | `display_q_table()` | Shows Q-table summary in log |
| 1101 | `test_ai()` | Tests AI on random items |
| 1116 | `save_model()` / `load_model()` | Model persistence via filedialog |
| 1152 | `on_closing()` | Window close handler — stops tasks, saves state |
| 1189 | `save_els_state_gui()` | Saves ELS state to file |
| 1213 | `load_els_state_gui()` | Loads ELS state from file |
| 1238 | `start_evolution_process_gui()` | Starts new evolution run |
| 1288 | `continue_evolution_process_gui()` | Continues existing evolution |
| 1308 | `pause_els_task()` | Pauses evolution |
| 1325 | `resume_els_task()` | Resumes evolution |
| 1342 | `save_champion_agent_gui()` | Saves best agent model |
| 1362 | `load_champion_to_seed_gui()` | Loads model to seed population |
| 1404 | `_run_benchmark_generator_script()` | Runs benchmark generator in subprocess |
| 1492 | `generate_numeric_benchmark_gui()` | GUI trigger for benchmark generation |

**Key Dependencies**: `puffinzip_ai` (all major classes), `puffinzip_gui.gui_utils`, `puffinzip_gui.gui_style_setup`, `puffinzip_gui.gui_layout_setup`, `puffinzip_gui.secondary_main_app`, `puffinzip_gui.settings_gui`, `puffinzip_gui.chart_utils`, `puffinzip_gui.generational_data_viewer`, `puffinzip_gui.checkpoint_manager_panel`

---

### puffinzip_gui/secondary_main_app.py
**Lines**: 313  
**Purpose**: Populates the Evolution Controls tab and Change Log tab content.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 60 | `_create_section_frame()` | Creates themed LabelFrame section |
| 66 | `populate_evolution_controls_tab_content()` | Builds evolution controls — benchmark config, Start/Stop/Pause buttons, continuous mode toggle, session management, adaptation bottleneck buttons, progress bar, logs |
| 296 | `populate_changelog_tab_content()` | Loads and displays changelog.md |

**Constants**: GUI symbols (SYMBOL_TRAIN, SYMBOL_PLAY, SYMBOL_STOP, etc.)

**Key Dependencies**: `puffinzip_ai.config`, `.gui_utils`

---

### puffinzip_gui/chart_utils.py
**Lines**: 706  
**Purpose**: Matplotlib charting utilities with full mock fallback when Matplotlib is unavailable.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 194 | `_get_theme_attr()` | Theme attribute accessor with fallback |
| 201 | `clear_frame_widgets()` | Destroys all child widgets in a frame |
| 209 | `display_placeholder_message()` | Shows "awaiting data" message in chart area |
| 229 | `plot_training_rewards()` | Plots reward history line chart |
| 281 | `plot_action_distribution()` | Plots action choice bar chart |
| 352 | `downsample_data()` | Downsamples data to max_points for performance |
| 358 | `plot_evolution_with_stats()` | Plots evolution fitness with compression score and dataset size |
| 550 | `plot_evolution_fitness()` | Plots generational fitness (best/avg/worst/median lines) |

**Mock Classes** (when Matplotlib unavailable): `MockSpine`, `MockBar`, `MockLegend`, `MockAx`, `Figure`, `FigureCanvasTkAgg`

**Constants**: `CHART_BACKGROUND_COLOR_DEFAULT`, `CHART_FIGURE_FACECOLOR_DEFAULT`, `CHART_TEXT_COLOR_DEFAULT`, `PLOT_LINE_COLOR_BEST_DEFAULT`, `PLOT_LINE_COLOR_AVG_DEFAULT`, `PLOT_BAR_COLOR_RLE_DEFAULT`, etc.

**Key Dependencies**: `matplotlib` (optional), `numpy`, `tkinter`

---

### puffinzip_gui/checkpoint_manager_panel.py
**Lines**: 436  
**Purpose**: GUI panel for managing and comparing evolution checkpoints.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 13 | `CheckpointManagerPanel(ttk.Frame)` | Checkpoint management panel with save/load/compare/delete UI |
| 312 | `ComparisonDialog(tk.Toplevel)` | Toplevel dialog for comparing two checkpoints side-by-side |

**CheckpointManagerPanel Methods**: `setup_ui()`, `_on_save_checkpoint()`, `_on_refresh_list()`, `_on_load_checkpoint()`, `_on_delete_checkpoint()`, `_on_compare_checkpoints()`

**ComparisonDialog Methods**: `setup_ui()`, `_on_compare()`, `_display_comparison()`

---

### puffinzip_gui/generational_data_viewer.py
**Lines**: ~740  
**Purpose**: Collapsible, lazily-loaded, paginated population viewer for ELS generational history with export capabilities.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 56 | `GenerationalDataViewerTab(ttk.Frame)` | "Gen. Deep Dive" tab — hierarchical tree: Generation → Batch → Agent. Collapsed by default; expands lazily to save RAM. 20 generations per page. Right-click context menu and toolbar button for CSV/JSON export. |

**Key Methods**: `on_generation_snapshot()` (queue-driven live update), `load_and_display_data()` (backward-compat refresh), `_populate_page()` (renders page of collapsed gen headers), `_on_expand()` / `_on_collapse()` (lazy load/unload children), `_load_batches()`, `_load_agents()`, `_show_agent_detail_popup()`, `_prev_page()` / `_next_page()`, `_on_right_click()` (context menu), `_export_selected()` (toolbar export), `_export_generation()` / `_export_batch()` (CSV/JSON export), `_write_csv()` / `_write_json()`

**Tree Columns**: Name (#0), Best Fit, Avg Fit, Agents, Gen Born, Learn Rate, Expl. Rate, RLE MinRun, Thresholds

**Export**: Right-click any generation or batch row → Export as CSV or JSON. CSV flattens agent + eval_stats into columns. JSON preserves full nested agent dicts. Toolbar "Export Selected" button also available.

**Data Source**: `EvolutionaryOptimizer.generation_snapshots` — lightweight per-generation snapshots captured after each evaluation (no Q-tables).

---

### puffinzip_gui/gui_layout_setup.py
**Lines**: ~120  
**Purpose**: Creates the main GUI layout — notebook tabs, paned windows, scrollable areas, log panel, chart areas.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 26 | `setup_main_layout()` | Creates 5 tabs (AI Controls, Evolution Lab, Gen. Deep Dive, Change Log, Settings), left controls pane with scrollable canvas, right log+charts pane |

**Created Widgets** (on app_instance): `main_notebook`, `main_controls_tab`, `evolution_lab_tab`, `gdv_tab`, `changelog_tab`, `settings_content_tab`, `output_scrolled_text`, `charts_area_on_main_tab`, `rewards_chart_area`, `actions_chart_area`

---

### puffinzip_gui/gui_style_setup.py
**Lines**: ~250  
**Purpose**: Configures all ttk styles for the dark-themed GUI.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 37 | `_get_theme_attr()` | Theme attribute accessor with fallback |
| 42 | `setup_styles()` | Configures TFrame, TLabel, TButton, TEntry, TCombobox, TCheckbutton, TNotebook, TScrollbar, TLabelframe, TPanedwindow, Treeview styles |

**Style Prefixes**: Default `T*` styles + `GenDataViewer.*` for treeview, `Small.TButton`, `Title.TLabel`, `Error.TLabel`, `Success.TLabel`

---

### puffinzip_gui/gui_utils.py
**Lines**: 519  
**Purpose**: Centralized GUI helper utilities — font resolution, theme lookups, scrollable canvas, widget utilities.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 94 | `_get_logger()` | Gets app logger or creates fallback |
| 118 | `get_theme_attr()` | Theme attribute accessor with module-level fallbacks |
| 138 | `_font_cache_for()` | Returns per-app font cache dict |
| 149 | `_normalise_font_tuple()` | Normalizes font tuple to (family, size, weight, slant) |
| 181 | `_attempt_font_resolution()` | Resolves font via Tk Font API |
| 209 | `_is_monospaced_request()` | Checks if font family is monospaced |
| 214 | `_build_generic_font_candidates()` | Builds fallback font candidate list |
| 234 | `_get_font_with_fallbacks()` | Resolves font with caching and multi-family fallback |
| 271 | `build_font_palette()` | Returns dict of all app font tuples |
| 293 | `build_font_fallbacks()` | Returns fallback font palette |
| 311 | `initialize_app_fonts()` | Initializes resolved fonts on app instance |
| 340 | `on_frame_configure()` | Scrollable frame configure handler |
| 371 | `on_canvas_configure()` | Canvas configure handler (width sync) |
| 396 | `_handle_canvas_scroll()` | Mouse wheel scroll handler |
| 421 | `_bind_events_recursively()` | Recursively binds scroll events to widget tree |
| 443 | `bind_scroll_events()` | Binds scroll events to a scrollable canvas |
| 461 | `create_scrollable_canvas()` | Creates a scrollable Canvas+Scrollbar pair |
| 494 | `clear_frame_widgets()` | Destroys all children of a frame |

**Constants**: `_THEME_FALLBACKS` (dict of 20 color fallbacks), `_MONO_FAMILY_CANDIDATES`, `_PRIMARY_FAMILY_CANDIDATES`, `_FONT_CACHE` (WeakKeyDictionary)

---

### puffinzip_gui/settings_gui.py
**Lines**: 766  
**Purpose**: Settings tab — reads/writes config.py values, provides theme presets, device selection, font picker.

**Classes**:
| Line | Name | Description |
|------|------|-------------|
| 39 | `SettingsTab(ttk.Frame)` | Complete settings editor panel |

**SettingsTab Key Methods**:
| Line | Name | Description |
|------|------|-------------|
| 93 | `_load_theme_presets()` | Loads gui_themes.json |
| 118 | `_filter_font_combobox()` | Live font search filter |
| 128 | `_setup_internal_styles()` | Configures SettingsGUI.* styles |
| 170 | `_create_setting_group()` | Creates a settings section LabelFrame |
| 177 | `_create_device_selection_combobox()` | Creates GPU/CPU device dropdown (auto-detected) |
| 209 | `_create_data_complexity_combobox()` | Creates benchmark complexity dropdown |
| 218 | `_add_setting_to_group()` | Adds one setting widget to a group |
| 300 | `_setup_ui()` | Builds full settings UI with groups: Theme, Font, AI, Evolution, Benchmark, GPU |
| 418 | `_apply_theme_preset()` | Applies a named theme preset |
| 445 | `_browse_path()` | File/folder browser for path settings |
| 471 | `load_settings()` | Loads values from config.py into GUI |
| 512 | `load_defaults()` | Resets all to defaults |
| 548 | `save_settings()` | Saves GUI values to config.py |
| 723 | `create_tooltip()` | Creates hover tooltips for widgets |

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 757 | `populate_settings_tab()` | Creates and packs SettingsTab into parent frame |

**Key Dependencies**: `puffinzip_ai.utils.settings_manager`, `puffinzip_ai.utils.hardware_detector`

---

## examples/ — Demo Scripts

---

### examples/compression_discovery_example.py
**Lines**: 185  
**Purpose**: Demo showcasing autonomous compression method discovery.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 20 | `analyze_data_characteristics()` | Analyzes text data properties |
| 60 | `recommend_compression_strategy()` | Recommends strategy based on data analysis |
| 80 | `autonomous_compression_discovery()` | Full autonomous discovery workflow |
| 126 | `llm_driven_method_search()` | Simulates LLM-guided method search |
| 165 | `continuous_learning_simulation()` | Simulates continuous learning loop |

**Key Dependencies**: `puffinzip_ai` (get_hybrid_engine, get_generator, get_registry, generate_novelty, evolve)

---

### examples/hybrid_compression_demo.py
**Lines**: 146  
**Purpose**: Interactive showcase of the hybrid compression engine.

**Functions**:
| Line | Name | Description |
|------|------|-------------|
| 14 | `showcase_methods()` | Lists all registered compression methods |
| 34 | `test_methods()` | Tests all methods on sample data |
| 62 | `generate_novelty()` | Generates a novel compression method |
| 76 | `compare_languages()` | Compares Python vs Rust implementations |
| 100 | `interactive_demo()` | Interactive CLI menu for demos |

**Key Dependencies**: `puffinzip_ai.hybrid_compression_engine`, `puffinzip_ai.compression_method_registry`

---

## scripts/ — Launcher Scripts & Build Helpers

---

| Script | Platform | Purpose |
|--------|----------|---------|
| `start.sh` (repo root) | Linux/macOS | **Universal launcher** — auto-detects hardware (CPU, RAM, GPU type & VRAM), clones repo if missing, creates venv, installs deps, generates credentials, reads `public_access` from credentials to determine bind address. Automatically loads `.env` (git-ignored) for local config overrides. Auto-detects RunPod (proxy URL) and Vast.ai; falls back to public IP detection for generic cloud/bare metal. Auto-starts Cloudflare Tunnel (`puffinzipai`) if `cloudflared` is installed. Exports hardware profile env vars, starts WebUI. Run presets (Test / Medium / Max) are available in the WebUI. Env vars: `PUFFIN_GPUS`, `PUFFIN_WORKERS`, `PUFFIN_CACHE_MAX_MB`, `PUFFIN_CACHE_MAX_FILES`, `PUFFIN_PORT`, `PUFFIN_HOST`, `PUFFIN_REPO_URL`, `PUFFIN_REPO_BRANCH`, `GITHUB_TOKEN`, `PUFFIN_ADMIN_USERNAME`, `PUFFIN_ADMIN_PASSWORD`, `PUFFIN_CUSTOM_URL` |
| `start.bat` (repo root) | Windows | **Universal launcher** — same as `start.sh` for Windows. Auto-detects hardware, creates venv, installs deps, generates credentials, detects RunPod/cloud platform, resolves connect URL, auto-starts Cloudflare Tunnel if available, starts WebUI |
| `run_webui_windows.bat` | Windows | Developer personal Windows launcher (port 5001) |
| `run_gui.spec` | — | PyInstaller build spec for GUI executable |
| `package_a100.bat` + `_package_a100_impl.ps1` | Windows | Packages all deployment files into a ZIP for pod deployment |
| `preflight_metrics_check.py` | — | Pre-flight cache and metrics validation |
