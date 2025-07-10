# PuffinZipAI_Project/puffinzip_ai/__init__.py
import logging
import os
import sys
import traceback
import importlib
import ast

_init_logger_pza = logging.getLogger("puffinzip_ai_package_init")
if not _init_logger_pza.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _f = logging.Formatter('%(asctime)s - PZ_AI_INIT - %(levelname)s - %(message)s')
    _h.setFormatter(_f)
    _init_logger_pza.addHandler(_h)
    _init_logger_pza.setLevel(logging.DEBUG)

_puffinzip_ai_dir_for_config = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH_FOR_INIT = os.path.join(_puffinzip_ai_dir_for_config, "config.py")
_PACKAGE_NAME_FOR_IMPORTS = "puffinzip_ai"

ALL_CONFIG_DEFAULTS_INIT_TIME = {
    "CONFIG_FILE_DIR": "os.path.dirname(os.path.abspath(__file__))",
    "PROJECT_ROOT_DIR": "os.path.dirname(CONFIG_FILE_DIR)",
    "DATA_DIR": "os.path.join(PROJECT_ROOT_DIR, \"data\")",
    "MODELS_DIR": "os.path.join(DATA_DIR, \"models\")",
    "LOGS_DIR_NAME": "logs",
    "LOGS_DIR_PATH": "os.path.join(PROJECT_ROOT_DIR, LOGS_DIR_NAME)",
    "BENCHMARK_DATA_DIR": "os.path.join(DATA_DIR, \"benchmark_sets\")",
    "GENERATED_BENCHMARK_SUBDIR_NAME": "generated_default_benchmark_subdir",
    "GENERATED_BENCHMARK_DEFAULT_PATH": "os.path.join(BENCHMARK_DATA_DIR, GENERATED_BENCHMARK_SUBDIR_NAME)",
    "MODEL_FILENAME": "puffin_ai_model_default.dat",
    "MODEL_FILE_DEFAULT": "os.path.join(MODELS_DIR, MODEL_FILENAME)",
    "COMPRESSED_FILE_SUFFIX": ".pfz", "DEFAULT_LEN_THRESHOLDS": [50, 150, 500],
    "DEFAULT_BATCH_COMPRESS_EXTENSIONS": [".txt", ".log", ".md", ".csv"],
    "DEFAULT_ALLOWED_LEARN_EXTENSIONS": [".txt", ".md", ".py", ".js", ".html", ".css"],
    "CORE_AI_LOG_FILENAME": "puffin_ai_core.log", "APP_VERSION": "0.9.6-dev",
    "DEFAULT_LOG_LEVEL": "INFO", "LOG_MAX_BYTES": 5 * 1024 * 1024, "LOG_BACKUP_COUNT": 3,
    "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED": False, # Added
    "DEFAULT_LEARNING_RATE": 0.1, "DEFAULT_DISCOUNT_FACTOR": 0.9, "DEFAULT_EXPLORATION_RATE": 1.0,
    "DEFAULT_EXPLORATION_DECAY_RATE": 0.9995, "DEFAULT_MIN_EXPLORATION_RATE": 0.01,
    "DEFAULT_TRAIN_BATCH_SIZE": 32, "DEFAULT_FOLDER_LEARN_BATCH_SIZE": 16,
    "DEFAULT_TRAIN_LOG_INTERVAL_BATCHES": 10, "CLI_MENU_LOG_FILENAME": "puffin_cli_menu.log",
    "CLI_RUNNER_LOG_FILENAME": "puffin_cli_runner.log",
    "GUI_RUNNER_LOG_FILENAME": "puffin_gui_runner_status.log",
    "DEFAULT_POPULATION_SIZE": 50, "DEFAULT_NUM_GENERATIONS": 100,
    "DEFAULT_ADDITIONAL_ELS_GENERATIONS": 30, "DEFAULT_MUTATION_RATE": 0.15,
    "DEFAULT_ELITISM_COUNT": 2, "DEFAULT_SELECTION_STRATEGY": "tournament",
    "BENCHMARK_DATASET_PATH": None, "EVOLUTIONARY_AI_LOG_FILENAME": "evolutionary_optimizer.log",
    "BENCHMARK_GENERATOR_LOG_FILENAME": "benchmark_generator.log",
    "MAX_THRESHOLDS_COUNT": 7, "MIN_THRESHOLDS_COUNT": 1, "MAX_THRESHOLDS_COUNT_MERGED": 8,
    "ADVANCED_CROSSOVER_PROBABILITY": 0.6, "HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB": 0.3,
    "HYPERMUTATION_PARAM_STRENGTH_FACTOR": 2.0, "STAGNATION_GENERATIONS_THRESHOLD": 10,
    "MUTATION_RATE_BOOST_FACTOR": 1.8, "MUTATION_RATE_DECAY_FACTOR": 0.99,
    "HYPERMUTATION_STAGNATION_THRESHOLD": 18, "HYPERMUTATION_FRACTION": 0.2,
    "RLE_MIN_RUN_INIT_MIN": 2, "RLE_MIN_RUN_INIT_MAX": 4, "RLE_MIN_RUN_MUTATION_PROB": 0.15,
    "RLE_MIN_RUN_BOUNDS_MIN": 2, "RLE_MIN_RUN_BOUNDS_MAX": 7,
    "RANDOM_IMMIGRANT_INTERVAL": 20, "RANDOM_IMMIGRANT_FRACTION": 0.1,
    "ELS_LOG_PREFIX": "[ELS]", "ELS_STATS_MSG_PREFIX": "[ELS_FITNESS_HISTORY]",
    "ACCELERATION_TARGET_DEVICE": "CPU", "DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT": True,
    "DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS": 10, "INITIAL_BENCHMARK_COMPLEXITY_LEVEL": "SIMPLE",
    "THEME_BG_COLOR": "#2E3440", "THEME_FG_COLOR": "#ECEFF4", "THEME_FRAME_BG": "#3B4252",
    "THEME_ACCENT_COLOR": "#88C0D0", "THEME_INPUT_BG": "#434C5E", "THEME_TEXT_AREA_BG": "#2E3440",
    "THEME_BUTTON_BG": "#4C566A", "THEME_BUTTON_FG": "#ECEFF4", "THEME_ERROR_FG": "#BF616A",
    "FONT_FAMILY_PRIMARY_CONFIG": "Segoe UI", "FONT_SIZE_BASE_CONFIG": 10,
    "FONT_FAMILY_MONO_CONFIG": "Consolas",
}

if not os.path.exists(CONFIG_FILE_PATH_FOR_INIT):
    try:
        editable_settings_defaults = {}
        _settings_manager_module = None
        try:
            _settings_manager_module = importlib.import_module(f"{_PACKAGE_NAME_FOR_IMPORTS}.utils.settings_manager")
        except ImportError:
            pass

        if _settings_manager_module and hasattr(_settings_manager_module, 'EDITABLE_SETTINGS'):
            editable_settings_defaults = {k: v.get("default") for k, v in
                                          _settings_manager_module.EDITABLE_SETTINGS.items()}
        else:
            pass

        config_content_lines = ["import os\n\n"]

        path_expression_keys = [
            "CONFIG_FILE_DIR", "PROJECT_ROOT_DIR", "DATA_DIR", "MODELS_DIR",
            "LOGS_DIR_PATH", "BENCHMARK_DATA_DIR",
            "GENERATED_BENCHMARK_DEFAULT_PATH", "MODEL_FILE_DEFAULT"
        ]
        literals_needed_by_paths = ["LOGS_DIR_NAME", "GENERATED_BENCHMARK_SUBDIR_NAME", "MODEL_FILENAME"]
        written_keys = set()
        current_globals_for_eval = {'os': os}

        for key in literals_needed_by_paths:
            master_default_value = ALL_CONFIG_DEFAULTS_INIT_TIME[key]
            value_to_write = editable_settings_defaults.get(key, master_default_value)
            config_content_lines.append(f"{key} = {repr(value_to_write)}\n")
            current_globals_for_eval[key] = value_to_write
            written_keys.add(key)
        config_content_lines.append("\n")

        for key in path_expression_keys:
            expression_string = ALL_CONFIG_DEFAULTS_INIT_TIME[key]
            config_content_lines.append(f"{key} = {expression_string}\n")
            try:
                current_globals_for_eval[key] = eval(expression_string, current_globals_for_eval.copy())
            except Exception as e_build_eval:
                pass
            written_keys.add(key)
        config_content_lines.append("\n")

        for key, master_default_value_expr in ALL_CONFIG_DEFAULTS_INIT_TIME.items():
            if key in written_keys:
                continue
            value_to_write = editable_settings_defaults.get(key, master_default_value_expr)
            config_content_lines.append(f"{key} = {repr(value_to_write)}\n") # Use repr for defaults
            written_keys.add(key)

        config_content_lines.append("\ndef ensure_dirs():\n")
        ensure_dirs_paths_list = ['DATA_DIR', 'MODELS_DIR', 'LOGS_DIR_PATH', 'BENCHMARK_DATA_DIR',
                                  'GENERATED_BENCHMARK_DEFAULT_PATH']
        config_content_lines.append(f"    dirs_to_create_keys = {ensure_dirs_paths_list}\n")
        config_content_lines.append("    for d_key in dirs_to_create_keys:\n")
        config_content_lines.append("        d_path = globals().get(d_key)\n")
        config_content_lines.append("        if d_path and isinstance(d_path, str) and d_path.strip():\n")
        config_content_lines.append("            if not os.path.exists(d_path):\n")
        config_content_lines.append("                try: os.makedirs(d_path, exist_ok=True)\n")
        config_content_lines.append(
            "                except OSError as e_dir: pass\n") # Removed print
        config_content_lines.append(
            "            elif not os.path.isdir(d_path): pass\n") # Removed print
        config_content_lines.append(
            "        elif d_key == 'GENERATED_BENCHMARK_DEFAULT_PATH' and d_path is None : pass\n")
        config_content_lines.append(
            "        elif d_path is None : pass\n") # Removed print
        config_content_lines.append("\nensure_dirs()\n")

        with open(CONFIG_FILE_PATH_FOR_INIT, 'w', encoding='utf-8') as f_cfg:
            f_cfg.writelines(config_content_lines)
    except Exception as e_create_cfg:
        pass

loaded_config_module = None
_PACKAGE_EXPORTS = []

try:
    loaded_config_module = importlib.import_module(f"{_PACKAGE_NAME_FOR_IMPORTS}.config")
    if callable(getattr(loaded_config_module, "ensure_dirs", None)):
        try:
            loaded_config_module.ensure_dirs()
        except Exception as e_ensure_dirs_call:
            pass
except ImportError as e_cfg_import:
    pass

_globals_context_for_eval = {'os': os}

if loaded_config_module:
    for key, default_val_expr_str in ALL_CONFIG_DEFAULTS_INIT_TIME.items():
        config_val_present_in_module = hasattr(loaded_config_module, key)
        current_value = None
        if config_val_present_in_module:
            current_value = getattr(loaded_config_module, key)
        else:
            try:
                if isinstance(default_val_expr_str, (list, dict, tuple, int, float, bool, type(None))):
                    current_value = default_val_expr_str
                elif isinstance(default_val_expr_str, str):
                    try:
                        current_value = ast.literal_eval(default_val_expr_str)
                    except (ValueError, SyntaxError):
                        current_value = eval(default_val_expr_str, _globals_context_for_eval.copy())
                else:
                    current_value = default_val_expr_str
            except Exception as e_eval_default:
                continue
        globals()[key] = current_value
        if key not in ["CONFIG_FILE_DIR"]: _globals_context_for_eval[key] = current_value
        _PACKAGE_EXPORTS.append(key)
else:  # Fallback if config module could not be loaded at all
    for key, default_val_expr_str in ALL_CONFIG_DEFAULTS_INIT_TIME.items():
        current_value = None
        try:
            if isinstance(default_val_expr_str, (list, dict, tuple, int, float, bool, type(None))):
                current_value = default_val_expr_str
            elif isinstance(default_val_expr_str, str):
                try:
                    current_value = ast.literal_eval(default_val_expr_str)
                except (ValueError, SyntaxError):
                    current_value = eval(default_val_expr_str, _globals_context_for_eval.copy())
            else:
                current_value = default_val_expr_str
        except Exception as e_eval_total_fallback:
            continue
        globals()[key] = current_value
        if key not in ["CONFIG_FILE_DIR"]: _globals_context_for_eval[key] = current_value
        _PACKAGE_EXPORTS.append(key)
    if callable(globals().get("ensure_dirs")):
        try:
            globals()["ensure_dirs"]();
        except Exception as e_ensure_global_fb:
            pass

CONFIG_TARGET_DEVICE_FROM_GLOBALS = globals().get('ACCELERATION_TARGET_DEVICE',
                                                  ALL_CONFIG_DEFAULTS_INIT_TIME['ACCELERATION_TARGET_DEVICE'])
APP_VERSION_FROM_GLOBALS = globals().get('APP_VERSION', ALL_CONFIG_DEFAULTS_INIT_TIME['APP_VERSION'])

PuffinZipAI_Selected_Core = None

if "GPU" in str(CONFIG_TARGET_DEVICE_FROM_GLOBALS).upper():
    _init_logger_pza.info(f"Config target device '{CONFIG_TARGET_DEVICE_FROM_GLOBALS}' indicates GPU usage. Attempting to import GPU AI core...")
    try:
        PuffinZipAI_Selected_Core = importlib.import_module(f".gpu_core",
                                                            package=_PACKAGE_NAME_FOR_IMPORTS).PuffinZipAI_GPU
        _init_logger_pza.info("Successfully imported PuffinZipAI_GPU from .gpu_core.")
    except ImportError as e_gpu_import:
        _init_logger_pza.warning(f"Failed to import GPU AI core due to ImportError: {e_gpu_import}. Will fall back to CPU core.")
        PuffinZipAI_Selected_Core = None
    except Exception as e_gpu_other:
        _init_logger_pza.error(f"An unexpected error occurred while importing GPU AI core: {e_gpu_other}", exc_info=True)
        PuffinZipAI_Selected_Core = None

if PuffinZipAI_Selected_Core is None:
    if "GPU" in str(CONFIG_TARGET_DEVICE_FROM_GLOBALS).upper():
        _init_logger_pza.warning("Falling back to CPU AI core because GPU core failed to load.")
    else:
        _init_logger_pza.info("Target device is not GPU. Loading standard CPU AI core.")
    try:
        PuffinZipAI_Selected_Core = importlib.import_module(f".ai_core", package=_PACKAGE_NAME_FOR_IMPORTS).PuffinZipAI
        _init_logger_pza.info("Successfully imported PuffinZipAI from .ai_core.")
    except ImportError as e_cpu_import:
        _init_logger_pza.critical(f"CRITICAL: Failed to import even the fallback CPU AI core: {e_cpu_import}", exc_info=True)
        PuffinZipAI_Selected_Core = None
    except Exception as e_cpu_other_init:
        _init_logger_pza.critical(f"CRITICAL: An unexpected error occurred while importing the fallback CPU AI core: {e_cpu_other_init}", exc_info=True)
        PuffinZipAI_Selected_Core = None

globals()['PuffinZipAI'] = PuffinZipAI_Selected_Core
if PuffinZipAI_Selected_Core:
    _PACKAGE_EXPORTS.append("PuffinZipAI")
else:
    pass

try:
    globals()["setup_logger"] = importlib.import_module(f".logger", package=_PACKAGE_NAME_FOR_IMPORTS).setup_logger
    _PACKAGE_EXPORTS.append("setup_logger")
except ImportError as e:
    pass

try:
    _rle_utils_mod = importlib.import_module(f".rle_utils", package=_PACKAGE_NAME_FOR_IMPORTS)
    globals()['rle_compress'] = _rle_utils_mod.rle_compress
    globals()['rle_decompress'] = _rle_utils_mod.rle_decompress
    _rle_constants_mod = importlib.import_module(f".rle_constants", package=_PACKAGE_NAME_FOR_IMPORTS)
    globals()['RLE_DECOMPRESSION_ERRORS'] = _rle_constants_mod.RLE_DECOMPRESSION_ERRORS
    globals()['RLE_ERROR_NO_CHAR'] = _rle_constants_mod.RLE_ERROR_NO_CHAR
    _PACKAGE_EXPORTS.extend(["rle_compress", "rle_decompress", "RLE_DECOMPRESSION_ERRORS", "RLE_ERROR_NO_CHAR"])
except ImportError as e:
    pass

try:
    globals()['calculate_reward'] = importlib.import_module(f".reward_system",
                                                            package=_PACKAGE_NAME_FOR_IMPORTS).calculate_reward
    _PACKAGE_EXPORTS.append("calculate_reward")
except ImportError as e:
    pass

try:
    _evo_core_mod = importlib.import_module(f".evolution_core", package=_PACKAGE_NAME_FOR_IMPORTS)
    globals()['EvolutionaryOptimizer'] = _evo_core_mod.EvolutionaryOptimizer
    globals()['EvolvingAgent'] = _evo_core_mod.EvolvingAgent
    if globals()['EvolutionaryOptimizer']: _PACKAGE_EXPORTS.append("EvolutionaryOptimizer")
    if globals()['EvolvingAgent']: _PACKAGE_EXPORTS.append("EvolvingAgent")
except ImportError as e_evo_init:
    pass
except Exception as e_evo_gen_init:
    pass

temp_all = []
for name_export_candidate in sorted(list(set(_PACKAGE_EXPORTS))):
    val_in_globals_final_check = globals().get(name_export_candidate)
    if val_in_globals_final_check is not None:
        temp_all.append(name_export_candidate)
    elif name_export_candidate in ALL_CONFIG_DEFAULTS_INIT_TIME and val_in_globals_final_check is None:
        pass
__all__ = temp_all

__version__ = str(globals().get('APP_VERSION', ALL_CONFIG_DEFAULTS_INIT_TIME['APP_VERSION']))