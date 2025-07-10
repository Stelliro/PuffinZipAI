# PuffinZipAI_Project/puffinzip_ai/utils/settings_manager.py
import os
import random
import re
import ast
import logging
import sys
import json  # Added for GUI state persistence

settings_manager_logger = logging.getLogger("puffinzip_ai.settings_manager")
if not settings_manager_logger.handlers:
    settings_manager_logger.setLevel(logging.INFO)
    settings_manager_logger.addHandler(logging.NullHandler())

GUI_STATE_FILENAME = "gui_state.json"  # New

try:
    _utils_dir = os.path.dirname(os.path.abspath(__file__))
    _puffinzip_ai_dir = os.path.dirname(_utils_dir)
    _project_root_dir = os.path.dirname(_puffinzip_ai_dir)
    CONFIG_FILE_PATH = os.path.join(_puffinzip_ai_dir, "config.py")
    GUI_STATE_FILE_PATH = os.path.join(_project_root_dir, GUI_STATE_FILENAME)  # New

    if not os.path.exists(CONFIG_FILE_PATH):
        for path_entry in sys.path:
            potential_path = os.path.join(path_entry, "puffinzip_ai", "config.py")
            if os.path.exists(potential_path):
                CONFIG_FILE_PATH = potential_path
                break
            potential_path_alt = os.path.join(path_entry, os.path.basename(_puffinzip_ai_dir), "config.py")
            if os.path.exists(potential_path_alt):
                CONFIG_FILE_PATH = potential_path_alt
                break
        else:
            pass
except Exception as e_path:
    CONFIG_FILE_PATH = os.path.join("puffinzip_ai", "config.py")
    GUI_STATE_FILE_PATH = os.path.join(os.getcwd(), GUI_STATE_FILENAME)  # Fallback for GUI state path

EDITABLE_SETTINGS = {
    "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED": {"type": bool, "default": False, "label": "Enable Debug Logs in Console",
                                         "tooltip": "Show detailed DEBUG level logs in the main application console output area. Otherwise, DEBUG logs primarily go to file."},
    "THEME_BG_COLOR": {"type": str, "default": "#2E3440", "label": "Main Background Color", "is_color": True,
                       "tooltip": "Main window and tab background color (e.g., #RRGGBB)."},
    "THEME_FG_COLOR": {"type": str, "default": "#ECEFF4", "label": "Main Foreground Color", "is_color": True,
                       "tooltip": "Default text color (e.g., #RRGGBB)."},
    "THEME_FRAME_BG": {"type": str, "default": "#3B4252", "label": "Frame Background Color", "is_color": True,
                       "tooltip": "Background for frames, LabelFrames, active tabs (e.g., #RRGGBB)."},
    "THEME_ACCENT_COLOR": {"type": str, "default": "#88C0D0", "label": "Accent Color", "is_color": True,
                           "tooltip": "Primary accent color for highlights, selections (e.g., #RRGGBB)."},
    "THEME_INPUT_BG": {"type": str, "default": "#434C5E", "label": "Input Field Background", "is_color": True,
                       "tooltip": "Background for Entry, Combobox, some chart areas (e.g., #RRGGBB)."},
    "THEME_TEXT_AREA_BG": {"type": str, "default": "#2E3440", "label": "Text Area Background", "is_color": True,
                           "tooltip": "Background for ScrolledText log areas (e.g., #RRGGBB)."},
    "THEME_BUTTON_BG": {"type": str, "default": "#4C566A", "label": "Button Background", "is_color": True,
                        "tooltip": "Background color for standard buttons (e.g., #RRGGBB)."},
    "THEME_BUTTON_FG": {"type": str, "default": "#ECEFF4", "label": "Button Foreground", "is_color": True,
                        "tooltip": "Text color for standard buttons (e.g., #RRGGBB)."},
    "THEME_ERROR_FG": {"type": str, "default": "#BF616A", "label": "Error Text Color", "is_color": True,
                       "tooltip": "Color for error messages and indicators (e.g., #RRGGBB)."},
    "FONT_FAMILY_PRIMARY_CONFIG": {"type": str, "default": "Segoe UI", "label": "Primary Font Family",
                                   "tooltip": "Main font family for the UI (e.g., Segoe UI, Arial, Calibri). System availability applies."},
    "FONT_SIZE_BASE_CONFIG": {"type": int, "min": 7, "max": 18, "default": 10, "label": "Base Font Size",
                              "tooltip": "Base font size for normal text. Other sizes (small, large) may scale relatively."},
    "FONT_FAMILY_MONO_CONFIG": {"type": str, "default": "Consolas", "label": "Monospace Font Family",
                                "tooltip": "Font for monospace text areas like logs (e.g., Consolas, Courier New)."},
    "ACCELERATION_TARGET_DEVICE": {"type": str, "default": "CPU", "label": "Processing Device",
                                   "tooltip": "Preferred processing device. Examples: CPU, GPU_AUTO, GPU_ID:0, GPU_ID:1. Auto-detected options will appear in GUI dropdown.",
                                   "options_logic": "detect_processing_devices"},
    "DEFAULT_POPULATION_SIZE": {"type": int, "min": 5, "max": 500, "default": 50, "label": "Population Size",
                                "tooltip": "Number of AI agents in each generation."},
    "DEFAULT_NUM_GENERATIONS": {"type": int, "min": 1, "max": 10000, "default": 100, "label": "Initial Generations",
                                "tooltip": "Number of generations for a new ELS run."},
    "DEFAULT_ADDITIONAL_ELS_GENERATIONS": {"type": int, "min": 1, "max": 1000, "default": 30,
                                           "label": "Additional Generations",
                                           "tooltip": "Generations to add when continuing ELS."},
    "DEFAULT_MUTATION_RATE": {"type": float, "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.15,
                              "label": "Base Mutation Rate",
                              "tooltip": "Base probability of an agent's parameters mutating (0.0-1.0)."},
    "DEFAULT_ELITISM_COUNT": {"type": int, "min": 0, "max": 20, "default": 2, "label": "Elitism Count",
                              "tooltip": "Number of best agents carried to next generation without changes."},
    "STAGNATION_GENERATIONS_THRESHOLD": {"type": int, "min": 3, "max": 100, "default": 10,
                                         "label": "Stagnation Threshold",
                                         "tooltip": "Generations without improvement before considering stagnation."},
    "MUTATION_RATE_BOOST_FACTOR": {"type": float, "min": 1.0, "max": 10.0, "step": 0.1, "default": 1.8,
                                   "label": "Mutation Boost Factor",
                                   "tooltip": "Multiplier for mutation rate during hypermutation phase."},
    "HYPERMUTATION_STAGNATION_THRESHOLD": {"type": int, "min": 5, "max": 100, "default": 18,
                                           "label": "Hypermutation Stagnation",
                                           "tooltip": "Stagnation generations to trigger hypermutation."},
    "HYPERMUTATION_FRACTION": {"type": float, "min": 0.0, "max": 0.75, "step": 0.05, "default": 0.20,
                               "label": "Hypermutation Fraction",
                               "tooltip": "Fraction of non-elite population to hypermutate."},
    "RANDOM_IMMIGRANT_INTERVAL": {"type": int, "min": 1, "max": 200, "default": 20,
                                  "label": "Immigrant Interval (Gens)",
                                  "tooltip": "Introduce random immigrants every N generations."},
    "RANDOM_IMMIGRANT_FRACTION": {"type": float, "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.10,
                                  "label": "Immigrant Fraction",
                                  "tooltip": "Fraction of population replaced by immigrants."},
    "ADVANCED_CROSSOVER_PROBABILITY": {"type": float, "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.6,
                                       "label": "Advanced Crossover Prob.",
                                       "tooltip": "Probability of using advanced breeding methods."},
    "RLE_MIN_RUN_INIT_MIN": {"type": int, "min": 1, "max": 5, "default": 2, "label": "RLE Min Run (Initial Min)",
                             "tooltip": "Min initial value for agent's RLE min_run_length."},
    "RLE_MIN_RUN_INIT_MAX": {"type": int, "min": 1, "max": 10, "default": 4, "label": "RLE Min Run (Initial Max)",
                             "tooltip": "Max initial value for agent's RLE min_run_length."},
    "RLE_MIN_RUN_MUTATION_PROB": {"type": float, "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.15,
                                  "label": "RLE Min Run Mut. Prob",
                                  "tooltip": "Probability RLE min_run_length mutates."},
    "RLE_MIN_RUN_BOUNDS_MIN": {"type": int, "min": 1, "max": 5, "default": 2, "label": "RLE Min Run (Bound Min)",
                               "tooltip": "Absolute minimum an agent's RLE min_run_length can be."},
    "RLE_MIN_RUN_BOUNDS_MAX": {"type": int, "min": 2, "max": 10, "default": 7, "label": "RLE Min Run (Bound Max)",
                               "tooltip": "Absolute maximum an agent's RLE min_run_length can be."},
    "DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT": {"type": bool, "default": True, "label": "Dynamic Benchmarking Active",
                                               "tooltip": "If ELS should dynamically generate benchmark items (True) or rely on BENCHMARK_DATASET_PATH (False)."},
    "DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS": {"type": int, "min": 0, "max": 100, "default": 10,
                                                "label": "Dynamic Bench. Refresh (Gens)",
                                                "tooltip": "Regenerate dynamic benchmarks every N ELS generations. 0=no refresh after initial set."},
    "INITIAL_BENCHMARK_COMPLEXITY_LEVEL": {"type": str, "default": "SIMPLE", "label": "Initial Dyn. Bench. Complexity",
                                           "tooltip": "Default complexity for first dynamic benchmark set if adaptive strategy is chosen. Options: VERY_SIMPLE, SIMPLE, MODERATE, COMPLEX, VERY_COMPLEX.",
                                           "options_logic": "get_data_complexity_levels"},
    "BENCHMARK_DATASET_PATH": {"type": str, "is_path": True, "default": None, "label": "Static Benchmark Path",
                               "tooltip": "Path to a folder of static benchmark files. If set, dynamic benchmarking (above) is usually off unless this path is invalid."},
}


def _is_valid_hex_color(s):
    if not isinstance(s, str): return False
    return bool(re.fullmatch(r"#([0-9a-fA-F]{3}){1,2}", s))


def get_config_values() -> dict:
    values = {}
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config_content = f.read()
        for name, details in EDITABLE_SETTINGS.items():
            pattern = re.compile(rf"^\s*{name}\s*=\s*(.+?)\s*(?:#.*)?$", re.MULTILINE)
            match = pattern.search(config_content)
            loaded_value_final = details.get("default")

            if match:
                value_str = match.group(1).strip()
                parsed_value_from_file = None
                try:
                    if value_str.lower() == "none":
                        parsed_value_from_file = None
                    elif details.get("is_path") or details.get("is_color") or details.get("options_logic"):
                        try:
                            attempt = ast.literal_eval(value_str)
                            if not isinstance(attempt, (str, type(None))):
                                parsed_value_from_file = value_str.strip("'\"")
                            else:
                                parsed_value_from_file = attempt
                        except (ValueError, SyntaxError):
                            parsed_value_from_file = value_str
                    else:
                        parsed_value_from_file = ast.literal_eval(value_str)

                    expected_type = details["type"]
                    if parsed_value_from_file is not None and not isinstance(parsed_value_from_file, expected_type):
                        if expected_type is float and isinstance(parsed_value_from_file, int):
                            parsed_value_from_file = float(parsed_value_from_file)
                        elif expected_type is int and isinstance(parsed_value_from_file,
                                                                 float) and parsed_value_from_file.is_integer():
                            parsed_value_from_file = int(parsed_value_from_file)

                    if details.get("is_color"):
                        if isinstance(parsed_value_from_file, str) and _is_valid_hex_color(parsed_value_from_file):
                            loaded_value_final = parsed_value_from_file
                        else:
                            loaded_value_final = details.get("default")
                    elif parsed_value_from_file is not None:
                        loaded_value_final = parsed_value_from_file

                except (ValueError, SyntaxError) as e_eval:
                    loaded_value_final = details.get("default")
                except Exception as e_parse_val:
                    loaded_value_final = details.get("default")
            else:
                loaded_value_final = details.get("default")

            values[name] = loaded_value_final

    except FileNotFoundError:
        for name, details in EDITABLE_SETTINGS.items():
            values[name] = details.get("default")
    except Exception as e_read_file:
        for name, details in EDITABLE_SETTINGS.items():
            values[name] = details.get("default")
    return values


def save_config_values(new_values: dict) -> bool:
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = []
        updated_keys_in_file = set()
        for line in lines:
            original_line = line
            for name, details in EDITABLE_SETTINGS.items():
                pattern = re.compile(r"^(\s*" + re.escape(name) + r"\s*=\s*)(.*?)\s*(#.*)?$")
                match = pattern.match(line.rstrip('\n'))
                if match and name in new_values:
                    leading_part = match.group(1)
                    comment_part = match.group(3) if match.group(3) else ""
                    value_to_write = new_values[name]

                    if details.get("is_color") and isinstance(value_to_write, str) and not _is_valid_hex_color(
                            value_to_write.strip()):
                        value_to_write = details.get("default", "")

                    if isinstance(value_to_write, str):
                        if details.get("is_path") and value_to_write and '\\' in value_to_write:
                            formatted_value = f"r'{value_to_write}'"
                        elif value_to_write is None and details.get("is_path"):
                            formatted_value = "None"
                        elif (details.get(
                                "options_logic") or name == "ACCELERATION_TARGET_DEVICE") or name == "INITIAL_BENCHMARK_COMPLEXITY_LEVEL":
                            formatted_value = f"'{value_to_write}'"
                        else:
                            formatted_value = f"'{value_to_write}'"
                    elif value_to_write is None:
                        formatted_value = "None"
                    elif isinstance(value_to_write, bool):
                        formatted_value = "True" if value_to_write else "False"
                    else:
                        formatted_value = str(value_to_write)
                    line = f"{leading_part}{formatted_value} {comment_part.strip()}\n"
                    updated_keys_in_file.add(name)
                    break
            new_lines.append(line)
        appended_settings_count = 0
        for name, value in new_values.items():
            if name in EDITABLE_SETTINGS and name not in updated_keys_in_file:
                details = EDITABLE_SETTINGS[name]
                value_to_write = value
                if details.get("is_color") and isinstance(value_to_write, str) and not _is_valid_hex_color(
                        value_to_write.strip()):
                    value_to_write = details.get("default", "")

                if isinstance(value_to_write, str):
                    if details.get("is_path") and value_to_write and '\\' in value_to_write:
                        formatted_value = f"r'{value_to_write}'"
                    elif value_to_write is None and details.get("is_path"):
                        formatted_value = "None"
                    elif (details.get(
                            "options_logic") or name == "ACCELERATION_TARGET_DEVICE") or name == "INITIAL_BENCHMARK_COMPLEXITY_LEVEL":
                        formatted_value = f"'{value_to_write}'"
                    else:
                        formatted_value = f"'{value_to_write}'"
                elif value_to_write is None:
                    formatted_value = "None"
                elif isinstance(value_to_write, bool):
                    formatted_value = "True" if value_to_write else "False"
                else:
                    formatted_value = str(value_to_write)
                new_lines.append(
                    f"\n{name} = {formatted_value}\n")  # Add a blank line before appending a new setting
                appended_settings_count += 1
        if appended_settings_count > 0 and not new_lines[-1].endswith("\n\n"):  # Ensure last line ends well
            new_lines.append("\n")  # Add a final newline if settings were appended
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        return False


# --- GUI State Persistence Functions ---
def load_gui_state() -> dict:
    if not os.path.exists(GUI_STATE_FILE_PATH):
        return {}
    try:
        with open(GUI_STATE_FILE_PATH, 'r', encoding='utf-8') as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return {}
        return state
    except (json.JSONDecodeError, IOError) as e:
        return {}  # Return empty on error


def save_gui_state(state: dict) -> bool:
    if not isinstance(state, dict):
        return False
    try:
        # Ensure the directory for gui_state.json exists
        gui_state_dir = os.path.dirname(GUI_STATE_FILE_PATH)
        if gui_state_dir and not os.path.exists(gui_state_dir):
            os.makedirs(gui_state_dir, exist_ok=True)

        with open(GUI_STATE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        return True
    except IOError as e:
        return False
    except Exception as e_generic:  # Catch other potential errors like permissions
        return False


if __name__ == '__main__':
    if not settings_manager_logger.handlers or isinstance(settings_manager_logger.handlers[0], logging.NullHandler):
        settings_manager_logger.handlers.clear()
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        settings_manager_logger.addHandler(ch)
        settings_manager_logger.setLevel(logging.DEBUG)

    if not os.path.exists(CONFIG_FILE_PATH):
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f_dummy:
                f_dummy.write("\n\n")
                f_dummy.write("DEFAULT_POPULATION_SIZE = 30 \n")
                f_dummy.write("THEME_ACCENT_COLOR = '#ABCDEF'\n")
                f_dummy.write("DEBUG_LOG_CONSOLE_OUTPUT_ENABLED = False\n")
        except Exception as e_create_dummy:
            pass

    current_vals = get_config_values()

    test_save_values = {}
    if "DEFAULT_POPULATION_SIZE" in EDITABLE_SETTINGS:
        test_save_values["DEFAULT_POPULATION_SIZE"] = current_vals.get("DEFAULT_POPULATION_SIZE", 50) + random.randint(
            1, 5)
    if "THEME_ACCENT_COLOR" in EDITABLE_SETTINGS:
        rand_hex = ''.join(random.choices('0123456789abcdef', k=6))
        test_save_values["THEME_ACCENT_COLOR"] = f"#{rand_hex}"
    if "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED" in EDITABLE_SETTINGS:
        test_save_values["DEBUG_LOG_CONSOLE_OUTPUT_ENABLED"] = not current_vals.get("DEBUG_LOG_CONSOLE_OUTPUT_ENABLED",
                                                                                    False)

    if test_save_values:
        save_success = save_config_values(test_save_values)
        if save_success:
            reloaded_vals = get_config_values()

    # --- Test GUI State Persistence ---
    initial_gui_state = load_gui_state()
    test_gui_state = {
        "main_window_geometry": "1200x800+100+50",
        "evolution_controls_pane_pos": [300, 700],  # Example
        "log_display_pane_pos_evo_tab": [500],  # Example for vertical pane
        "active_notebook_tab": "Evolution Controls"  # Example
    }
    if random.random() < 0.5:  # Randomly add more to test flexibility
        test_gui_state["last_model_loaded_path"] = f"/test/path/model_{random.randint(1, 100)}.dat"

    save_gui_success = save_gui_state(test_gui_state)

    if save_gui_success:
        loaded_back_gui_state = load_gui_state()

        if loaded_back_gui_state == test_gui_state:
            pass
        else:
            pass
            for k_gs, v_gs_orig in test_gui_state.items():
                if k_gs not in loaded_back_gui_state:
                    pass
                elif loaded_back_gui_state[k_gs] != v_gs_orig:
                    pass
    else:
        pass