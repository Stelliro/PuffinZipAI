# PuffinZipAI_Project/puffinzip_ai/config.py
import os

LOGS_DIR_NAME = 'logs'
GENERATED_BENCHMARK_SUBDIR_NAME = 'generated_default_benchmark_subdir'
MODEL_FILENAME = 'puffin_ai_model_default.dat'

CONFIG_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(CONFIG_FILE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR_PATH = os.path.join(PROJECT_ROOT_DIR, LOGS_DIR_NAME)
BENCHMARK_DATA_DIR = os.path.join(DATA_DIR, "benchmark_sets")
GENERATED_BENCHMARK_DEFAULT_PATH = os.path.join(BENCHMARK_DATA_DIR, GENERATED_BENCHMARK_SUBDIR_NAME)
MODEL_FILE_DEFAULT = os.path.join(MODELS_DIR, MODEL_FILENAME)

COMPRESSED_FILE_SUFFIX = '.pfz'
DEFAULT_LEN_THRESHOLDS = [50, 150, 500]
DEFAULT_BATCH_COMPRESS_EXTENSIONS = ['.txt', '.log', '.md', '.csv']
DEFAULT_ALLOWED_LEARN_EXTENSIONS = ['.txt', '.md', '.py', '.js', '.html', '.css']
CORE_AI_LOG_FILENAME = 'puffin_ai_core.log'
APP_VERSION = '0.9.7'
DEFAULT_LOG_LEVEL = 'INFO'
LOG_MAX_BYTES = 5242880 
LOG_BACKUP_COUNT = 3

DEBUG_LOG_CONSOLE_OUTPUT_ENABLED = False 

DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.9
DEFAULT_EXPLORATION_RATE = 1.0
DEFAULT_EXPLORATION_DECAY_RATE = 0.9995
DEFAULT_MIN_EXPLORATION_RATE = 0.01
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_FOLDER_LEARN_BATCH_SIZE = 16
DEFAULT_TRAIN_LOG_INTERVAL_BATCHES = 10

CLI_MENU_LOG_FILENAME = 'puffin_cli_menu.log'
CLI_RUNNER_LOG_FILENAME = 'puffin_cli_runner.log'
GUI_RUNNER_LOG_FILENAME = 'puffin_gui_runner_status.log'

# Maximum target population size for extended searches
DEFAULT_POPULATION_SIZE = 50
# Smaller initial population to allow quick early feedback
INITIAL_POPULATION_SIZE = 12
# Additional agents introduced every few generations for broader coverage
POPULATION_GROWTH_STEP = 8
# Interval (in generations) between growth events
POPULATION_GROWTH_INTERVAL = 5
DEFAULT_NUM_GENERATIONS = 100 
DEFAULT_ADDITIONAL_ELS_GENERATIONS = 30 
DEFAULT_MUTATION_RATE = 0.15 
DEFAULT_ELITISM_COUNT = 2 
DEFAULT_SELECTION_STRATEGY = 'tournament'
BENCHMARK_DATASET_PATH = None 
EVOLUTIONARY_AI_LOG_FILENAME = 'evolutionary_optimizer.log'
BENCHMARK_GENERATOR_LOG_FILENAME = 'benchmark_generator.log'

MAX_THRESHOLDS_COUNT = 7
MIN_THRESHOLDS_COUNT = 1
MAX_THRESHOLDS_COUNT_MERGED = 8
ADVANCED_CROSSOVER_PROBABILITY = 0.6 
HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB = 0.3
HYPERMUTATION_PARAM_STRENGTH_FACTOR = 2.0
STAGNATION_GENERATIONS_THRESHOLD = 10 
MUTATION_RATE_BOOST_FACTOR = 1.8 
MUTATION_RATE_DECAY_FACTOR = 0.99
HYPERMUTATION_STAGNATION_THRESHOLD = 18 
HYPERMUTATION_FRACTION = 0.2 
RLE_MIN_RUN_INIT_MIN = 2 
RLE_MIN_RUN_INIT_MAX = 4 
RLE_MIN_RUN_MUTATION_PROB = 0.15 
RLE_MIN_RUN_BOUNDS_MIN = 2 
RLE_MIN_RUN_BOUNDS_MAX = 7 
RANDOM_IMMIGRANT_INTERVAL = 20 
RANDOM_IMMIGRANT_FRACTION = 0.1 

ELS_LOG_PREFIX = '[ELS]'
ELS_STATS_MSG_PREFIX = '[ELS_FITNESS_HISTORY]'
GEN_SNAPSHOT_MSG_PREFIX = 'GEN_SNAPSHOT:'

ACCELERATION_TARGET_DEVICE = 'GPU_ID:0'
GPU_RLE_TARGET_VRAM_USAGE_FRACTION = 0.3
GPU_RLE_WORKSPACE_MIN_MB = 64
GPU_RLE_WORKSPACE_MAX_MB = 512
GPU_RLE_WORKSPACE_TARGET_MB = 128
DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT = True 
DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS = 10 
INITIAL_BENCHMARK_COMPLEXITY_LEVEL = 'SIMPLE' 

# --- Compression Scaffolding Configuration ---
# Training-wheels system: agents can use known methods (gzip/zlib/bz2/lzma)
# with progressive penalties and temporary bans to encourage novel discovery.
SCAFFOLDING_ENABLED = True                 # Enable reference method scaffolding for agents
SCAFFOLDING_DEFAULT_REFERENCE = 'gzip'     # Default reference method for new agents
SCAFFOLDING_RELIANCE_BAN_THRESHOLD = 0.50  # Ban agent from reference if reliance ratio >= this
SCAFFOLDING_BAN_DURATION_ITEMS = 30        # Initial ban duration (items); doubles each repeat
SCAFFOLDING_BAN_MAX_DURATION_ITEMS = 200   # Maximum ban duration cap
SCAFFOLDING_GRACE_GENERATIONS = 10         # Generations before penalties start ramping
SCAFFOLDING_RAMP_GENERATIONS = 80          # Penalty ramp-up phase length (after grace)
SCAFFOLDING_MATURE_GENERATIONS = 150       # Generation at which reference auto-banned
SCAFFOLDING_BEAT_REFERENCE_BONUS = 4.0     # Reward bonus when own method beats reference
SCAFFOLDING_WEANING_BONUS = 2.0            # Bonus when agent stops using reference voluntarily

# --- Neural Network (DQN) Configuration ---
NN_ENABLED = True                          # Use DQN agents instead of tabular Q-tables
NN_STATE_FEATURE_DIM = 20                  # Continuous feature vector size (20-dim: see nn_core.nn_agent)
NN_HIDDEN_SIZES = [256, 256]               # Hidden layer widths for Dueling DQN
NN_REPLAY_BUFFER_CAPACITY = 50000          # Max transitions in per-agent PER buffer
NN_REPLAY_MIN_SIZE = 128                   # Min buffer entries before training starts
NN_TRAIN_BATCH_SIZE = 128                  # Mini-batch size for DQN gradient updates
NN_TARGET_NETWORK_UPDATE_FREQ = 200        # Steps between target network hard-syncs (fallback; soft update preferred)
NN_LEARNING_RATE = 3e-4                    # AdamW optimiser learning rate for DQN
NN_GRAD_CLIP_NORM = 1.0                    # Max gradient norm (0 = no clipping)
NN_WEIGHT_DECAY = 1e-5                     # L2 regularisation in AdamW
NN_SOFTMAX_ACTION_TEMPERATURE = 0.5        # Temperature for softmax action selection (lower = greedier)
NN_MUTATION_WEIGHT_NOISE_STD = 0.02        # Gaussian noise σ added to weights during mutation
NN_MUTATION_WEIGHT_PROB = 0.1              # Probability of mutating each parameter tensor
NN_CROSSOVER_LAYER_SWAP_PROB = 0.5         # Probability of swapping each layer during crossover

# --- Prioritized Experience Replay (PER) ---
NN_PER_ALPHA = 0.6                         # PER prioritisation exponent (0 = uniform, 1 = full priority)
NN_PER_BETA_START = 0.4                    # Initial importance-sampling correction exponent
NN_PER_BETA_FRAMES = 100000                # Frames over which beta anneals to 1.0

# --- Learning Rate Scheduler ---
NN_COSINE_LR_T_MAX = 5000                  # CosineAnnealingWarmRestarts T_0 (steps per restart)
NN_COSINE_LR_ETA_MIN = 1e-6                # Minimum learning rate at cosine trough

# --- Soft Target Updates & N-Step Returns ---
NN_SOFT_TARGET_TAU = 0.005                 # Polyak averaging coefficient for soft target sync
NN_NSTEP_RETURNS = 3                       # N-step return horizon length

# --- Architecture Hyperparameters ---
NN_DROPOUT = 0.1                           # Dropout probability in residual blocks
NN_ATTENTION_HEADS = 4                     # Number of attention heads in feature attention block
NN_NOISY_SIGMA = 0.5                       # Initial sigma for NoisyNet linear layers

# --- GitHub File Fetcher Configuration ---
# Downloads real-world text files from trusted GitHub repos for benchmark training.
# Files are cached locally to avoid repeated API calls.
GITHUB_CACHE_DIR = os.path.join(DATA_DIR, "github_cache")
GITHUB_TARGET_FILE_SIZE_MIN = 1024           # 1 KB — minimum file size to fetch
GITHUB_TARGET_FILE_SIZE_MAX = 51200          # 50 KB — maximum file size to fetch
GITHUB_API_TOKEN = os.environ.get('GITHUB_TOKEN', None)  # Optional: set GITHUB_TOKEN env var for higher rate limits
GITHUB_MIN_STARS = 500                       # Minimum star count for auto-trusted repos
GITHUB_FETCH_TIMEOUT = 15                    # HTTP request timeout in seconds

# --- GitHub Cache Eviction ---
# Prevents cache from filling pod storage during high-population runs.
# When the cache exceeds MAX_FILES, the oldest entries (by fetch time) are evicted.
# PUFFIN_CACHE_MAX_MB env var overrides the MB limit at runtime.
GITHUB_CACHE_MAX_FILES = int(os.environ.get('PUFFIN_CACHE_MAX_FILES', 500))     # Max cached files before LRU eviction
GITHUB_CACHE_MAX_MB = int(os.environ.get('PUFFIN_CACHE_MAX_MB', 200))           # Max cache size in MB before LRU eviction
GITHUB_FILE_EXTENSIONS = [                   # Only fetch files with these extensions
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.md', '.txt', '.rst', '.html', '.css',
    '.json', '.yaml', '.yml', '.toml', '.xml',
    '.rs', '.go', '.java', '.c', '.h', '.cpp',
    '.rb', '.sh', '.bat', '.ps1', '.cfg', '.ini',
]
GITHUB_TRUSTED_REPOS = [                     # Curated allowlist of high-star, well-known repos
    'python/cpython', 'pallets/flask', 'psf/requests',
    'django/django', 'fastapi/fastapi',
    'nodejs/node', 'expressjs/express', 'microsoft/TypeScript',
    'rust-lang/rust', 'denoland/deno', 'golang/go',
    'github/docs', 'mdn/content',
    'home-assistant/core', 'ansible/ansible',
    'torvalds/linux', 'git/git',
]

# --- Phased Training Configuration ---
# Controls how anti-corruption agents transition from synthetic corrupted data
# to real-world GitHub files over generations.  Three phases:
#   Phase 1 (corruption-only):  Gen 0 to PHASE1_END — 100% corrupted synthetic data
#   Phase 2 (blend):            Gen PHASE1_END+1 to PHASE2_END — mix of corrupted + GitHub files
#   Phase 3 (real-world):       Gen PHASE2_END+1 onward — mostly GitHub real-world files
PHASED_TRAINING_ENABLED = True
PHASED_TRAINING_PHASE1_END = 10              # Last gen of corruption-only phase
PHASED_TRAINING_PHASE2_END = 30              # Last gen of blended phase
PHASED_TRAINING_PHASE3_GITHUB_RATIO = 0.80   # Fraction of items from GitHub in phase 3
PHASED_TRAINING_GITHUB_ITEM_COUNT = 20       # Number of GitHub files to blend in

THEME_BG_COLOR = '#2E3440'
THEME_FG_COLOR = '#ECEFF4'
THEME_FRAME_BG = '#3B4252'
THEME_ACCENT_COLOR = '#88C0D0'
THEME_INPUT_BG = '#434C5E'
THEME_TEXT_AREA_BG = '#2E3440'
THEME_BUTTON_BG = '#4C566A'
THEME_BUTTON_FG = '#ECEFF4'
THEME_ERROR_FG = '#BF616A'

FONT_FAMILY_PRIMARY_CONFIG = 'Segoe UI' 
FONT_SIZE_BASE_CONFIG = 10 
FONT_FAMILY_MONO_CONFIG = 'Consolas' 

def ensure_dirs():
    dirs_to_create_keys = ['DATA_DIR', 'MODELS_DIR', 'LOGS_DIR_PATH', 'BENCHMARK_DATA_DIR', 'GENERATED_BENCHMARK_DEFAULT_PATH', 'GITHUB_CACHE_DIR']
    for d_key in dirs_to_create_keys:
        d_path = globals().get(d_key)
        if d_path and isinstance(d_path, str) and d_path.strip():
            if not os.path.exists(d_path):
                try: os.makedirs(d_path, exist_ok=True)
                except OSError as e_dir: pass 
            elif not os.path.isdir(d_path): pass
        elif d_key == 'GENERATED_BENCHMARK_DEFAULT_PATH' and d_path is None : pass
        elif d_path is None : pass 

ensure_dirs()