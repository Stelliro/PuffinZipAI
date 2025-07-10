# PuffinZipAI_Project/puffinzip_gui/primary_main_app.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import tkinter.font
import threading
import queue
import os
import traceback
import time
import sys
import subprocess
import numpy as np
import logging
import re

try:
    from puffinzip_ai import (
        PuffinZipAI, MODEL_FILE_DEFAULT,
        EvolutionaryOptimizer
    )
    from puffinzip_ai.config import (
        GENERATED_BENCHMARK_DEFAULT_PATH, GENERATED_BENCHMARK_SUBDIR_NAME,
        APP_VERSION as CFG_APP_VERSION, LOGS_DIR_PATH, GUI_RUNNER_LOG_FILENAME,
        DEFAULT_TRAIN_BATCH_SIZE, DEFAULT_ALLOWED_LEARN_EXTENSIONS,
        DEFAULT_FOLDER_LEARN_BATCH_SIZE, DEFAULT_BATCH_COMPRESS_EXTENSIONS,
        ELS_LOG_PREFIX as CFG_ELS_LOG_PREFIX,
        ELS_STATS_MSG_PREFIX as CFG_ELS_STATS_MSG_PREFIX,
        INITIAL_BENCHMARK_COMPLEXITY_LEVEL as CONFIG_INITIAL_BENCH_COMPLEXITY_DEFAULT,
        DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT as CONFIG_DYN_BENCH_ACTIVE_DEFAULT,
        DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS as CONFIG_DYN_BENCH_REFRESH_DEFAULT,
        ACCELERATION_TARGET_DEVICE as CONFIG_TARGET_DEVICE_DEFAULT,

        THEME_BG_COLOR as DEFAULT_THEME_BG_COLOR,
        THEME_FG_COLOR as DEFAULT_THEME_FG_COLOR,
        THEME_FRAME_BG as DEFAULT_THEME_FRAME_BG,
        THEME_ACCENT_COLOR as DEFAULT_THEME_ACCENT_COLOR,
        THEME_INPUT_BG as DEFAULT_THEME_INPUT_BG,
        THEME_TEXT_AREA_BG as DEFAULT_THEME_TEXT_AREA_BG,
        THEME_BUTTON_BG as DEFAULT_THEME_BUTTON_BG,
        THEME_BUTTON_FG as DEFAULT_THEME_BUTTON_FG,
        THEME_ERROR_FG as DEFAULT_THEME_ERROR_FG,

        FONT_FAMILY_PRIMARY_CONFIG as DEFAULT_FONT_FAMILY_PRIMARY_CONFIG,
        FONT_SIZE_BASE_CONFIG as DEFAULT_FONT_SIZE_BASE_CONFIG,
        FONT_FAMILY_MONO_CONFIG as DEFAULT_FONT_FAMILY_MONO_CONFIG
    )
    from puffinzip_ai.logger import setup_logger
    from puffinzip_ai.utils.benchmark_evaluator import DataComplexity
    from puffinzip_ai.utils import settings_manager

    _DummyLoggerClass = None
    try:
        from puffinzip_ai.ai_core import DummyLogger as DL_Core

        _DummyLoggerClass = DL_Core
    except ImportError:
        class DummyLoggerFallback_PMA:
            def _log(self, level, msg, exc_info_flag=False): pass

            def info(self, msg): self._log("INFO", msg)

            def warning(self, msg): self._log("WARN", msg)

            def error(self, msg, exc_info=False): self._log("ERROR", msg, exc_info_flag=exc_info)

            def exception(self, msg): self._log("EXCEPTION", msg, exc_info_flag=True)

            def debug(self, msg): self._log("DEBUG", msg)

            def critical(self, msg, exc_info=False): self._log("CRITICAL", msg, exc_info_flag=exc_info)


        _DummyLoggerClass = DummyLoggerFallback_PMA
except ImportError as e_main_imports:
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S');
    fallback_log_file = "puffinzip_gui_CRITICAL_IMPORT_ERROR.log"
    try:
        _pma_dir_err = os.path.dirname(os.path.abspath(__file__));
        _pma_parent_dir_err = os.path.dirname(_pma_dir_err)
        _project_root_for_log_err = os.path.dirname(_pma_parent_dir_err)
        log_dir_for_critical_err = os.path.join(_project_root_for_log_err, "logs")
        os.makedirs(log_dir_for_critical_err, exist_ok=True);
        fallback_log_file = os.path.join(log_dir_for_critical_err, "puffinzip_gui_CRITICAL_IMPORT_ERROR.log")
    except Exception:
        fallback_log_file = os.path.join(os.getcwd(), "puffinzip_gui_CRITICAL_IMPORT_ERROR.log")
    error_details = f"{timestamp} - FATAL ERROR in primary_main_app.py: Could not import puffinzip_ai components (ImportError: {e_main_imports}).\nTraceback:\n{traceback.format_exc()}\nEnsure PuffinZipAI_Project is in PYTHONPATH or run from the project root."
    try:
        with open(fallback_log_file, "a", encoding='utf-8') as f_err:
            f_err.write(error_details)
    except Exception as e_log_io:
        pass
    sys.exit(1)

try:
    from puffinzip_gui import gui_utils, gui_style_setup, gui_layout_setup
except ImportError as e_gui_mods_abs:
    try:
        from . import gui_utils, gui_style_setup, gui_layout_setup
    except ImportError as e_gui_mods_rel:
        gui_utils = None;
        gui_style_setup = None;
        gui_layout_setup = None

BENCHMARK_GENERATOR_SCRIPT_PATH = None
try:
    from puffinzip_ai.utils import benchmark_generator

    if hasattr(benchmark_generator, '__file__') and benchmark_generator.__file__:
        BENCHMARK_GENERATOR_SCRIPT_PATH = os.path.abspath(benchmark_generator.__file__)
    elif not hasattr(benchmark_generator, '__file__') or not benchmark_generator.__file__:
        pass
except (ImportError, AttributeError) as e_bm_path:
    pass

SYMBOL_TRAIN = "🎓";
SYMBOL_FOLDER = "📁";
SYMBOL_COMPRESS = "📦";
SYMBOL_DECOMPRESS = "📂"
SYMBOL_SETTINGS = "⚙️";
SYMBOL_SAVE = "💾";
SYMBOL_LOAD = "📤";
SYMBOL_TEST = "🧪"
SYMBOL_VIEW = "👁️";
SYMBOL_PLAY = "▶";
SYMBOL_PAUSE = "⏸";
SYMBOL_STOP = "⏹"
SYMBOL_CONTINUE = "↪️";
SYMBOL_CHAMPION = "🏆";
SYMBOL_SEED = "🌱";
SYMBOL_REFRESH = "🔄"
SYMBOL_BOTTLENECK_LOW = "📉";
SYMBOL_BOTTLENECK_MED = "📊";
SYMBOL_BOTTLENECK_HIGH = "📈"
SYMBOL_RESET = "🔄"
SYMBOL_SAVE_SESSION = "💾"
SYMBOL_LOAD_SESSION = "📤"


MAIN_APP_DIR = os.path.dirname(os.path.abspath(__file__));
PROJECT_ROOT_FROM_GUI = os.path.dirname(MAIN_APP_DIR)

CHANGELOG_FILENAME = "changelog.md";
CHANGELOG_FILE_PATH = os.path.join(PROJECT_ROOT_FROM_GUI, CHANGELOG_FILENAME)

ELS_STATS_MSG_PREFIX = CFG_ELS_STATS_MSG_PREFIX;
ELS_LOG_PREFIX = CFG_ELS_LOG_PREFIX;


class PuffinZipApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tuned_params = kwargs.pop('tuned_params', None)
        super().__init__(*args, **kwargs)
        self._shutting_down = False

        self.tuned_params = tuned_params if tuned_params is not None else {}
        self.logger = None;
        self.APP_VERSION = "PMA_FallbackVer_Init"
        self.app_target_device = CONFIG_TARGET_DEVICE_DEFAULT  # Initialize target device attribute

        try:
            self.APP_VERSION = CFG_APP_VERSION;
            self.gui_runner_log_path = os.path.join(LOGS_DIR_PATH, GUI_RUNNER_LOG_FILENAME)
            self.logger = setup_logger("PuffinZip_App_Class", self.gui_runner_log_path, log_level=logging.DEBUG,
                                       log_to_console=True, console_level=logging.INFO)
        except Exception as e_log_cfg:
            self.logger = _DummyLoggerClass();

        from puffinzip_ai.config import THEME_BG_COLOR, THEME_FG_COLOR, THEME_FRAME_BG, THEME_ACCENT_COLOR, \
            THEME_INPUT_BG, THEME_TEXT_AREA_BG, THEME_BUTTON_BG, THEME_BUTTON_FG, \
            THEME_ERROR_FG, FONT_FAMILY_PRIMARY_CONFIG, FONT_SIZE_BASE_CONFIG, \
            FONT_FAMILY_MONO_CONFIG

        self.BG_COLOR = THEME_BG_COLOR if THEME_BG_COLOR else DEFAULT_THEME_BG_COLOR
        self.FG_COLOR = THEME_FG_COLOR if THEME_FG_COLOR else DEFAULT_THEME_FG_COLOR
        self.FRAME_BG = THEME_FRAME_BG if THEME_FRAME_BG else DEFAULT_THEME_FRAME_BG
        self.ACCENT_COLOR = THEME_ACCENT_COLOR if THEME_ACCENT_COLOR else DEFAULT_THEME_ACCENT_COLOR
        self.INPUT_BG = THEME_INPUT_BG if THEME_INPUT_BG else DEFAULT_THEME_INPUT_BG
        self.TEXT_AREA_BG = THEME_TEXT_AREA_BG if THEME_TEXT_AREA_BG else DEFAULT_THEME_TEXT_AREA_BG
        self.BUTTON_BG = THEME_BUTTON_BG if THEME_BUTTON_BG else DEFAULT_THEME_BUTTON_BG
        self.BUTTON_FG = THEME_BUTTON_FG if THEME_BUTTON_FG else DEFAULT_THEME_BUTTON_FG
        self.ERROR_FG_COLOR = THEME_ERROR_FG if THEME_ERROR_FG else DEFAULT_THEME_ERROR_FG

        self.INPUT_FG = self.FG_COLOR;
        self.LABEL_FG = self.FG_COLOR;
        self.TEXT_AREA_FG = self.FG_COLOR
        self.ACCENT_HOVER_COLOR = "#8FBCBB";
        self.ACCENT_PRESSED_COLOR = "#5E81AC";
        self.DISABLED_FG_COLOR = "#4C566A"
        self.TAB_BG = self.BG_COLOR;
        self.ACTIVE_TAB_BG = self.FRAME_BG;
        self.TAB_BORDER_COLOR = "#4C566A";
        self.TAB_HIGHLIGHT_COLOR = self.ACCENT_COLOR
        self.SCROLLBAR_TROUGH = self.FRAME_BG;
        self.SCROLLBAR_BG = "#4C566A";
        self.SCROLLBAR_ACTIVE_BG = self.ACCENT_COLOR
        self.CHART_BACKGROUND_COLOR_DEFAULT = self.INPUT_BG;
        self.CHART_FIGURE_FACECOLOR_DEFAULT = self.FRAME_BG
        self.CHART_TEXT_COLOR_DEFAULT = self.FG_COLOR;
        self.CHART_GRID_COLOR_DEFAULT = self.TAB_BORDER_COLOR
        self.PLOT_LINE_COLOR_BEST_DEFAULT = self.ACCENT_COLOR;
        self.PLOT_LINE_COLOR_AVG_DEFAULT = "#EBCB8B"
        self.PLOT_LINE_COLOR_WORST_DEFAULT = self.ERROR_FG_COLOR;
        self.PLOT_LINE_COLOR_MEDIAN_DEFAULT = "#A3BE8C"
        self.ADV_RLE_BAR_COLOR = "#B48EAD";
        self.PLOT_BAR_COLOR_NOCOMP_DEFAULT = "#D8DEE9"

        font_family_primary_from_cfg = FONT_FAMILY_PRIMARY_CONFIG if FONT_FAMILY_PRIMARY_CONFIG else DEFAULT_FONT_FAMILY_PRIMARY_CONFIG
        font_size_base_from_cfg = FONT_SIZE_BASE_CONFIG if isinstance(FONT_SIZE_BASE_CONFIG,
                                                                      int) and FONT_SIZE_BASE_CONFIG > 0 else DEFAULT_FONT_SIZE_BASE_CONFIG
        font_family_mono_from_cfg = FONT_FAMILY_MONO_CONFIG if FONT_FAMILY_MONO_CONFIG else DEFAULT_FONT_FAMILY_MONO_CONFIG

        font_size_small_derived = max(7, font_size_base_from_cfg - 1);
        font_size_large_derived = font_size_base_from_cfg + 2
        _pfn = (font_family_primary_from_cfg, font_size_base_from_cfg, "normal");
        _pfb = (font_family_primary_from_cfg, font_size_base_from_cfg, "bold");
        _pfst = (font_family_primary_from_cfg, font_size_large_derived, "bold");
        _pfbtn = (font_family_primary_from_cfg, font_size_base_from_cfg, "normal");
        _pfsml = (font_family_primary_from_cfg, font_size_small_derived, "normal")
        _pfsb = (font_family_primary_from_cfg, font_size_small_derived, "normal");
        _pfnte = (font_family_primary_from_cfg, font_size_small_derived, "italic");
        _pfmno = (font_family_mono_from_cfg, font_size_base_from_cfg, "normal")
        if gui_utils:
            self.FONT_NORMAL = gui_utils._get_font_with_fallbacks(self, _pfn,
                                                                  ("Arial", font_size_base_from_cfg, "normal"));
            self.FONT_BOLD = gui_utils._get_font_with_fallbacks(self, _pfb, ("Arial", font_size_base_from_cfg, "bold"));
            self.FONT_SECTION_TITLE = gui_utils._get_font_with_fallbacks(self, _pfst,
                                                                         ("Arial", font_size_large_derived, "bold"));
            self.FONT_BUTTON = gui_utils._get_font_with_fallbacks(self, _pfbtn,
                                                                  ("Arial", font_size_base_from_cfg, "normal"));
            self.FONT_SMALL = gui_utils._get_font_with_fallbacks(self, _pfsml, ("Arial", font_size_small_derived, "normal"))
            self.FONT_SMALL_BUTTON = gui_utils._get_font_with_fallbacks(self, _pfsb,
                                                                        ("Arial", font_size_small_derived, "normal"));
            self.FONT_NOTE = gui_utils._get_font_with_fallbacks(self, _pfnte,
                                                                ("Arial", font_size_small_derived, "italic"));
            self.FONT_MONO = gui_utils._get_font_with_fallbacks(self, _pfmno,
                                                                ("Courier New", font_size_base_from_cfg, "normal"))
        else:
            self.FONT_NORMAL = ("Arial", 10, "normal");
            self.FONT_BOLD = ("Arial", 10, "bold");
            self.FONT_SECTION_TITLE = ("Arial", 12, "bold");
            self.FONT_BUTTON = ("Arial", 10, "normal");
            self.FONT_SMALL = ("Arial", 9, "normal")
            self.FONT_SMALL_BUTTON = ("Arial", 9, "normal");
            self.FONT_NOTE = ("Arial", 9, "italic");
            self.FONT_MONO = ("Courier New", 10, "normal")
        self.tk_font = tkinter.font
        self.chart_utils = None;
        self.settings_gui_module = None;
        self.secondary_main_app_module = None;
        self.generational_data_viewer_module = None
        try:
            from puffinzip_gui import chart_utils as cu_local, settings_gui as sg_local, \
                secondary_main_app as sma_local, generational_data_viewer as gdv_local
            self.chart_utils = cu_local;
            self.settings_gui_module = sg_local;
            self.secondary_main_app_module = sma_local;
            self.generational_data_viewer_module = gdv_local
            if not all([self.chart_utils, self.settings_gui_module, self.secondary_main_app_module,
                        self.generational_data_viewer_module]): pass
        except ImportError as e_gui_subs_final:
            self._handle_critical_error("Essential GUI sub-components failed to load.", e_gui_subs_final);
            return
        self.BENCHMARK_GENERATOR_SCRIPT_PATH = BENCHMARK_GENERATOR_SCRIPT_PATH
        if not self.BENCHMARK_GENERATOR_SCRIPT_PATH: pass
        self.title(f"PuffinZipAI v{self.APP_VERSION} - Adaptive Compression Optimizer");
        
        self.logger.info("PMA: Attempting to load previous GUI state...")
        gui_state = settings_manager.load_gui_state()
        initial_geometry = gui_state.get("main_window_geometry", "1250x820")
        try:
            self.geometry(initial_geometry)
            self.logger.info(f"PMA: Restored window geometry to {initial_geometry}")
        except tk.TclError:
            self.logger.warning(f"PMA: Could not apply saved geometry '{initial_geometry}'. Using default.")
            self.geometry("1250x820")
            
        self.minsize(1000, 700);
        self.configure(bg=self.BG_COLOR)
        self.ai_agent = None;
        self.els_optimizer = None
        try:
            if PuffinZipAI is None: raise ImportError(
                "PMA Error: PuffinZipAI class (selected based on config) was None at AI instantiation.")
            self.ai_agent = PuffinZipAI(target_device=self.app_target_device)
        except Exception as e_ai_init_ex:
            self._handle_critical_error("Failed to initialize the core AI Agent.", e_ai_init_ex);
            return
        self.gui_output_queue = queue.Queue(maxsize=2500);
        self.gui_stop_event = threading.Event()
        if self.ai_agent: self.ai_agent.gui_output_queue = self.gui_output_queue; self.ai_agent.gui_stop_event = self.gui_stop_event
        self.task_running = False;
        self.current_task_thread = None;
        self.current_task_is_els = False;
        self.els_is_paused = False;
        self.els_run_completed = False;
        self.els_fitness_history_data = [];
        self.els_chart_canvas_agg = None;
        self.els_chart_figure = None;
        self.els_diag_log_scrolled_text = None
        self.els_initial_benchmark_strategy_var = tk.StringVar(value="Adaptive (Fitness-based)");
        self.els_fixed_complexity_var = tk.StringVar(value=CONFIG_INITIAL_BENCH_COMPLEXITY_DEFAULT);
        self.els_fixed_size_mb_var = tk.StringVar(value="0.1")
        self.els_plot_show_best_var = tk.BooleanVar(value=True);
        self.els_plot_show_avg_var = tk.BooleanVar(value=True);
        self.els_plot_show_worst_var = tk.BooleanVar(value=False);
        self.els_plot_show_median_var = tk.BooleanVar(value=False)
        try:
            if EvolutionaryOptimizer is not None:
                self.els_optimizer = EvolutionaryOptimizer(gui_output_queue=self.gui_output_queue,
                                                           gui_stop_event=self.gui_stop_event, benchmark_items=None,
                                                           tuned_params=self.tuned_params,
                                                           dynamic_benchmarking_active=CONFIG_DYN_BENCH_ACTIVE_DEFAULT,
                                                           benchmark_refresh_interval_gens=CONFIG_DYN_BENCH_REFRESH_DEFAULT,
                                                           target_device=self.app_target_device)
            else:
                pass
        except Exception as e_els_init_ex:
            self._handle_critical_error("ELS Optimizer init failed.", e_els_init_ex,
                                        recoverable=True);
            self.els_optimizer = None
        if self.ai_agent:
            try:
                self.ai_agent.load_model(MODEL_FILE_DEFAULT)
            except Exception as e_load_model_init:
                pass

        if gui_style_setup:
            gui_style_setup.setup_styles(self);
        else:
            ttk.Label(self, text="CRITICAL: UI Styling module error.", style="Error.TLabel").pack(expand=True)

        if gui_layout_setup:
            self._setup_gui_layout_new()
        else:
            ttk.Label(self, text="CRITICAL: Main layout setup error.", style="Error.TLabel").pack(expand=True);
            return

        self._populate_evolution_controls_tab();
        self._populate_evolution_analytics_tab();
        self._populate_gdv_tab();
        self._populate_changelog_tab();
        self._populate_settings_tab_content()
        if self.ai_agent: self._update_ui_fields_from_agent_state()
        self._check_gui_queue();
        self._update_els_button_states();
        self.protocol("WM_DELETE_WINDOW", self.on_closing);

    def _setup_gui_layout_new(self):
        self.main_notebook = ttk.Notebook(self, style="TNotebook")
        self.main_notebook.pack(expand=True, fill='both', padx=10, pady=(10, 0))

        self.evolution_controls_tab = ttk.Frame(self.main_notebook, style="TFrame", padding=(5, 5))
        self.evolution_analytics_tab = ttk.Frame(self.main_notebook, style="TFrame", padding=(5, 5))
        self.gdv_tab = ttk.Frame(self.main_notebook, style="TFrame", padding=(5, 5))
        self.changelog_tab = ttk.Frame(self.main_notebook, style="TFrame", padding=(5, 5))
        self.settings_content_tab = ttk.Frame(self.main_notebook, style="TFrame", padding=(5, 5))

        self.main_notebook.add(self.evolution_controls_tab, text='Evolution Controls')
        self.main_notebook.add(self.evolution_analytics_tab, text='Evolution Analytics')
        self.main_notebook.add(self.gdv_tab, text='Generational Deep Dive')
        self.main_notebook.add(self.changelog_tab, text='Change Log')
        self.main_notebook.add(self.settings_content_tab, text='Settings')

        log_output_frame_main_app = ttk.Frame(self, style="TFrame", height=150)
        log_output_frame_main_app.pack(expand=False, fill=tk.X, padx=10, pady=(5, 10))

        font_mono_l = getattr(self, 'FONT_MONO', ("Consolas", 10))
        text_area_bg_l = getattr(self, 'TEXT_AREA_BG', "#1E1E1E")
        text_area_fg_l = getattr(self, 'TEXT_AREA_FG', "#D0D0D0")
        fg_color_l = getattr(self, 'FG_COLOR', "#E0E0E0")
        accent_color_l = getattr(self, 'ACCENT_COLOR', "#0078D4")

        self.output_scrolled_text = scrolledtext.ScrolledText(
            log_output_frame_main_app, wrap=tk.WORD, font=font_mono_l,
            bg=text_area_bg_l, fg=text_area_fg_l, insertbackground=fg_color_l,
            selectbackground=accent_color_l, selectforeground=text_area_bg_l,
            borderwidth=1, relief="solid", padx=12, pady=10,
            spacing1=3, spacing3=3, undo=True, height=8
        )
        self.output_scrolled_text.pack(expand=True, fill=tk.BOTH)
        self.output_scrolled_text.configure(state='disabled')

    def _cancel_analysis_refresh(self):
        if hasattr(self, '_analysis_refresh_after_id') and self._analysis_refresh_after_id:
            try:
                self.after_cancel(self._analysis_refresh_after_id);
            except tk.TclError:
                pass
            except Exception as e_cancel:
                pass
            self._analysis_refresh_after_id = None

    def _schedule_analysis_refresh_if_needed(self):
        pass

    def _on_auto_refresh_toggle(self):
        pass

    def _manual_refresh_analysis_and_reschedule(self):
        pass

    def _on_benchmark_strategy_change(self, event=None):
        selected_strategy = self.els_initial_benchmark_strategy_var.get();
        complex_combo = getattr(self, 'els_fixed_complexity_combo', None);
        size_entry = getattr(self, 'els_fixed_size_mb_entry', None)
        if complex_combo and hasattr(complex_combo,
                                     'winfo_exists') and complex_combo.winfo_exists(): complex_combo.config(
            state='readonly' if selected_strategy == "Fixed Complexity Level" else 'disabled')
        if size_entry and hasattr(size_entry, 'winfo_exists') and size_entry.winfo_exists(): size_entry.config(
            state='normal' if selected_strategy == "Fixed Average Item Size (MB)" else 'disabled')

    def _update_els_button_states(self):
        benchmark_script_ready = self.BENCHMARK_GENERATOR_SCRIPT_PATH and os.path.exists(
            self.BENCHMARK_GENERATOR_SCRIPT_PATH)
        if hasattr(self, 'generate_benchmark_button') and hasattr(self.generate_benchmark_button,
                                                                  'winfo_exists') and self.generate_benchmark_button.winfo_exists(): self.generate_benchmark_button.config(
            state=tk.NORMAL if benchmark_script_ready and not self.task_running else tk.DISABLED)
        no_els_system = EvolutionaryOptimizer is None or self.els_optimizer is None

        can_apply_adaptation = not no_els_system and self.task_running and self.current_task_is_els and not self.els_is_paused
        
        can_load_or_save_state = not no_els_system and not self.task_running
        can_save_existing_state = can_load_or_save_state and self.els_optimizer and self.els_optimizer.population
        
        buttons_config = {
            'start_evolution_button': {'enabled': not no_els_system and not self.task_running},
            'continue_els_button': {
                'enabled': not no_els_system and not self.task_running and self.els_run_completed and hasattr(
                    self.els_optimizer, 'population') and self.els_optimizer.population},
            'pause_els_button': {
                'enabled': not no_els_system and self.task_running and self.current_task_is_els and not self.els_is_paused},
            'resume_els_button': {
                'enabled': not no_els_system and self.task_running and self.current_task_is_els and self.els_is_paused},
            'save_champion_button': {
                'enabled': not no_els_system and not self.task_running and self.els_optimizer and hasattr(
                    self.els_optimizer, 'best_agent_overall') and self.els_optimizer.best_agent_overall},
            'load_champion_to_els_button': {'enabled': not no_els_system and not self.task_running},
            'stop_els_button': {'enabled': not no_els_system and self.task_running and self.current_task_is_els},
            'bottleneck_low_button': {'enabled': can_apply_adaptation},
            'bottleneck_medium_button': {'enabled': can_apply_adaptation},
            'bottleneck_high_button': {'enabled': can_apply_adaptation},
            'clear_adaptation_button': {'enabled': can_apply_adaptation},
            'test_ai_button_evo_tab': {'enabled': not self.task_running and self.ai_agent is not None},
            'view_qtable_button_evo_tab': {'enabled': not self.task_running and self.ai_agent is not None},
            'load_model_button_evo_tab': {'enabled': not self.task_running and self.ai_agent is not None},
            'save_model_button_evo_tab': {'enabled': not self.task_running and self.ai_agent is not None},
            'save_els_state_button': {'enabled': can_save_existing_state},
            'load_els_state_button': {'enabled': can_load_or_save_state},
        }
        for btn_name, cfg in buttons_config.items():
            button_widget = getattr(self, btn_name, None)
            if button_widget and hasattr(button_widget, 'winfo_exists') and button_widget.winfo_exists():
                try:
                    button_widget.config(state=tk.NORMAL if cfg['enabled'] else tk.DISABLED)
                except tk.TclError:
                    pass

    def _apply_els_adaptation(self, strategy_name: str, is_clear_action: bool = False):
        if not self.els_optimizer: self.log_message("ELS Error: Evolutionary Optimizer not available.");return
        if not self.task_running or not self.current_task_is_els: self.log_message(
            "ELS Info: Adaptation strategies can only be applied during an active ELS run.");return
        if self.els_is_paused: self.log_message(
            "ELS Info: Cannot apply adaptation while ELS is paused. Resume first.");return

        if is_clear_action:
            self.els_optimizer.clear_bottleneck_strategy()
            self._log_to_els_console(f"{ELS_LOG_PREFIX} Evolution focus reset to default pace.")
        else:
            self.els_optimizer.apply_bottleneck_strategy(strategy_name);
        self._update_els_button_states()

    def apply_low_bottleneck(self):
        self._apply_els_adaptation("low")

    def apply_medium_bottleneck(self):
        self._apply_els_adaptation("medium")

    def apply_high_bottleneck(self):
        self._apply_els_adaptation("high")

    def clear_bottleneck_strategy(self):
        self._apply_els_adaptation("clear", is_clear_action=True)

    def _update_els_chart_with_current_filters(self):
        self._update_els_chart(self.els_fitness_history_data)

    def _update_els_chart(self, fitness_data=None):
        if fitness_data is None: fitness_data = self.els_fitness_history_data

        target_frame_for_chart = getattr(self, 'evolution_analytics_chart_area', None)
        if not target_frame_for_chart or not (
                hasattr(target_frame_for_chart, 'winfo_exists') and target_frame_for_chart.winfo_exists()):
            return

        if self.chart_utils and hasattr(self.chart_utils, 'plot_evolution_fitness'):
            try:
                self.els_chart_canvas_agg, self.els_chart_figure = self.chart_utils.plot_evolution_fitness(
                    parent_frame=target_frame_for_chart, fitness_history_data=fitness_data,
                    existing_canvas_agg=self.els_chart_canvas_agg, existing_figure=self.els_chart_figure,
                    app_instance=self,
                    show_best=self.els_plot_show_best_var.get(), show_avg=self.els_plot_show_avg_var.get(),
                    show_worst=self.els_plot_show_worst_var.get(), show_median=self.els_plot_show_median_var.get())
            except Exception as e_plot:
                pass

    def _populate_evolution_controls_tab(self):
        if self.secondary_main_app_module and hasattr(self.secondary_main_app_module,
                                                      'populate_evolution_controls_tab_content'):  # Changed from populate_evolution_lab_tab_content
            self.secondary_main_app_module.populate_evolution_controls_tab_content(self)
        else:
            if hasattr(self, 'evolution_controls_tab') and self.evolution_controls_tab and hasattr(
                    self.evolution_controls_tab,
                    'winfo_exists') and self.evolution_controls_tab.winfo_exists():
                ttk.Label(self.evolution_controls_tab, text="Error: Evolution Controls UI load fail.",
                          style="Error.TLabel").pack()
            else:
                pass

    def _populate_evolution_analytics_tab(self):
        if not hasattr(self, 'evolution_analytics_tab') or not (
                hasattr(self.evolution_analytics_tab, 'winfo_exists') and self.evolution_analytics_tab.winfo_exists()):
            return

        header_frame = ttk.Frame(self.evolution_analytics_tab, style="TFrame")
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(header_frame, text="Evolution Fitness Progression Analytics", style="Title.TLabel").pack(side=tk.LEFT)

        filters_frame = ttk.Frame(self.evolution_analytics_tab, style="TFrame", padding=(10, 5, 10, 10))
        filters_frame.pack(fill=tk.X)

        ttk.Label(filters_frame, text="Chart Series:", style="TLabel").pack(side=tk.LEFT, padx=(0, 5))
        self.els_plot_best_chk_analytics = ttk.Checkbutton(filters_frame, text="Best",
                                                           variable=self.els_plot_show_best_var,
                                                           command=self._update_els_chart_with_current_filters,
                                                           style="TCheckbutton")
        self.els_plot_best_chk_analytics.pack(side=tk.LEFT, padx=3)
        self.els_plot_avg_chk_analytics = ttk.Checkbutton(filters_frame, text="Average",
                                                          variable=self.els_plot_show_avg_var,
                                                          command=self._update_els_chart_with_current_filters,
                                                          style="TCheckbutton")
        self.els_plot_avg_chk_analytics.pack(side=tk.LEFT, padx=3)
        self.els_plot_worst_chk_analytics = ttk.Checkbutton(filters_frame, text="Worst",
                                                            variable=self.els_plot_show_worst_var,
                                                            command=self._update_els_chart_with_current_filters,
                                                            style="TCheckbutton")
        self.els_plot_worst_chk_analytics.pack(side=tk.LEFT, padx=3)
        self.els_plot_median_chk_analytics = ttk.Checkbutton(filters_frame, text="Median",
                                                             variable=self.els_plot_show_median_var,
                                                             command=self._update_els_chart_with_current_filters,
                                                             style="TCheckbutton")
        self.els_plot_median_chk_analytics.pack(side=tk.LEFT, padx=3)

        self.evolution_analytics_chart_area = ttk.Frame(self.evolution_analytics_tab, style="TFrame")
        self.evolution_analytics_chart_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        self._update_els_chart([])

    def _populate_gdv_tab(self):
        if self.generational_data_viewer_module and hasattr(self.generational_data_viewer_module,
                                                            'GenerationalDataViewerTab'):
            if not hasattr(self, 'gdv_tab') or not (
                    hasattr(self.gdv_tab, 'winfo_exists') and self.gdv_tab.winfo_exists()): return
            for child in self.gdv_tab.winfo_children(): child.destroy()
            self.gdv_tab_instance = self.generational_data_viewer_module.GenerationalDataViewerTab(self.gdv_tab, self,
                                                                                                   self.els_optimizer)
            self.gdv_tab_instance.pack(expand=True, fill="both");
        else:
            if hasattr(self, 'gdv_tab') and self.gdv_tab and hasattr(self.gdv_tab,
                                                                     'winfo_exists') and self.gdv_tab.winfo_exists():
                ttk.Label(self.gdv_tab, text="Error: Generational Deep Dive UI load fail.", style="Error.TLabel").pack()
            else:
                pass

    def _populate_changelog_tab(self):
        if self.secondary_main_app_module and hasattr(self.secondary_main_app_module, 'populate_changelog_tab_content'):
            if not hasattr(self, 'changelog_tab') or not (hasattr(self.changelog_tab,
                                                                  'winfo_exists') and self.changelog_tab.winfo_exists()): return
            self.secondary_main_app_module.populate_changelog_tab_content(self, CHANGELOG_FILE_PATH, CHANGELOG_FILENAME)
        else:
            if hasattr(self, 'changelog_tab') and self.changelog_tab and hasattr(self.changelog_tab,
                                                                                 'winfo_exists') and self.changelog_tab.winfo_exists():
                ttk.Label(self.changelog_tab, text="Error: Changelog UI load fail.", style="Error.TLabel").pack()
            else:
                pass

    def _populate_settings_tab_content(self):
        tab_frame_widget = getattr(self, 'settings_content_tab', None)
        if not tab_frame_widget or not (hasattr(tab_frame_widget, 'winfo_exists') and tab_frame_widget.winfo_exists()):
            return
        try:
            if self.settings_gui_module and hasattr(self.settings_gui_module, 'populate_settings_tab'):
                self.settings_gui_module.populate_settings_tab(self, tab_frame_widget)
            else:
                for child in tab_frame_widget.winfo_children(): child.destroy()
                ttk.Label(tab_frame_widget, text="Error: Settings UI module load fail.", style="Error.TLabel").pack(
                    expand=True, fill="both", padx=20, pady=20)
        except Exception as e_pop_settings:
            for child in tab_frame_widget.winfo_children(): child.destroy()
            error_text = f"Fatal Error Loading Settings UI:\n{type(e_pop_settings).__name__}: {str(e_pop_settings)[:100]}...\n(See logs for full details)"
            ttk.Label(tab_frame_widget, text=error_text, style="Error.TLabel", justify=tk.LEFT, wraplength=max(300,
                                                                                                               tab_frame_widget.winfo_width() - 40 if tab_frame_widget.winfo_width() > 50 else 300)).pack(
                expand=True, fill="both", padx=20, pady=20)

    def reload_and_apply_theme(self):
        try:
            import importlib;
            from puffinzip_ai import config as fresh_config_module
            try:
                importlib.reload(fresh_config_module);
            except Exception as e_mod_reload:
                pass

            from puffinzip_ai.config import THEME_BG_COLOR, THEME_FG_COLOR, THEME_FRAME_BG, THEME_ACCENT_COLOR, \
                THEME_INPUT_BG, THEME_TEXT_AREA_BG, THEME_BUTTON_BG, THEME_BUTTON_FG, \
                THEME_ERROR_FG, FONT_FAMILY_PRIMARY_CONFIG, FONT_SIZE_BASE_CONFIG, \
                FONT_FAMILY_MONO_CONFIG

            self.BG_COLOR = THEME_BG_COLOR if THEME_BG_COLOR else DEFAULT_THEME_BG_COLOR
            self.FG_COLOR = THEME_FG_COLOR if THEME_FG_COLOR else DEFAULT_THEME_FG_COLOR
            self.FRAME_BG = THEME_FRAME_BG if THEME_FRAME_BG else DEFAULT_THEME_FRAME_BG
            self.ACCENT_COLOR = THEME_ACCENT_COLOR if THEME_ACCENT_COLOR else DEFAULT_THEME_ACCENT_COLOR
            self.INPUT_BG = THEME_INPUT_BG if THEME_INPUT_BG else DEFAULT_THEME_INPUT_BG
            self.TEXT_AREA_BG = THEME_TEXT_AREA_BG if THEME_TEXT_AREA_BG else DEFAULT_THEME_TEXT_AREA_BG
            self.BUTTON_BG = THEME_BUTTON_BG if THEME_BUTTON_BG else DEFAULT_THEME_BUTTON_BG
            self.BUTTON_FG = THEME_BUTTON_FG if THEME_BUTTON_FG else DEFAULT_THEME_BUTTON_FG
            self.ERROR_FG_COLOR = THEME_ERROR_FG if THEME_ERROR_FG else DEFAULT_THEME_ERROR_FG

            font_fam_prim = FONT_FAMILY_PRIMARY_CONFIG if FONT_FAMILY_PRIMARY_CONFIG else DEFAULT_FONT_FAMILY_PRIMARY_CONFIG
            font_size_base = FONT_SIZE_BASE_CONFIG if isinstance(FONT_SIZE_BASE_CONFIG,
                                                                 int) and FONT_SIZE_BASE_CONFIG > 0 else DEFAULT_FONT_SIZE_BASE_CONFIG
            font_fam_mono = FONT_FAMILY_MONO_CONFIG if FONT_FAMILY_MONO_CONFIG else DEFAULT_FONT_FAMILY_MONO_CONFIG

            font_size_small = max(7, font_size_base - 1);
            font_size_large = font_size_base + 2
            _pfn_r = (font_fam_prim, font_size_base, "normal");
            _pfb_r = (font_fam_prim, font_size_base, "bold");
            _pfst_r = (font_fam_prim, font_size_large, "bold");
            _pfbtn_r = (font_fam_prim, font_size_base, "normal");
            _pfsb_r = (font_fam_prim, font_size_small, "normal");
            _pfsml_r = (font_fam_prim, font_size_small, "normal")
            _pfnte_r = (font_fam_prim, font_size_small, "italic");
            _pfmno_r = (font_fam_mono, font_size_base, "normal")
            if gui_utils:
                self.FONT_NORMAL = gui_utils._get_font_with_fallbacks(self, _pfn_r,
                                                                      ("Arial", font_size_base, "normal"));
                self.FONT_BOLD = gui_utils._get_font_with_fallbacks(self, _pfb_r, ("Arial", font_size_base, "bold"));
                self.FONT_SECTION_TITLE = gui_utils._get_font_with_fallbacks(self, _pfst_r,
                                                                             ("Arial", font_size_large, "bold"));
                self.FONT_BUTTON = gui_utils._get_font_with_fallbacks(self, _pfbtn_r,
                                                                      ("Arial", font_size_base, "normal"));
                self.FONT_SMALL = gui_utils._get_font_with_fallbacks(self, _pfsml_r, ("Arial", font_size_small, "normal"))
                self.FONT_SMALL_BUTTON = gui_utils._get_font_with_fallbacks(self, _pfsb_r,
                                                                            ("Arial", font_size_small, "normal"));
                self.FONT_NOTE = gui_utils._get_font_with_fallbacks(self, _pfnte_r,
                                                                    ("Arial", font_size_small, "italic"));
                self.FONT_MONO = gui_utils._get_font_with_fallbacks(self, _pfmno_r,
                                                                    ("Courier New", font_size_base, "normal"))
            self.configure(bg=self.BG_COLOR)
        except Exception as e_reload_cfg:
            pass
        if gui_style_setup:
            try:
                gui_style_setup.setup_styles(self);
            except Exception as e_style:
                pass
        else:
            pass
        self.log_message("Theme settings have been re-applied from configuration.")

    def _handle_critical_setting_change(self, new_target_device: str):
        self.log_message(f"Applying critical setting change: New Target Device = '{new_target_device}'", "info")
        self.app_target_device = new_target_device
        
        # Re-initialize the main AI agent
        if PuffinZipAI:
            self.log_message(f"Re-initializing main AI agent for target '{new_target_device}'...", "info")
            try:
                self.ai_agent = PuffinZipAI(target_device=new_target_device)
                if self.ai_agent:
                    self.ai_agent.gui_output_queue = self.gui_output_queue
                    self.ai_agent.gui_stop_event = self.gui_stop_event
                self.log_message("Main AI agent re-initialized.", "info")
            except Exception as e:
                self._handle_critical_error(f"Failed to re-initialize main AI agent for device '{new_target_device}'.", e, recoverable=True)
                return False
        
        # Update the ELS optimizer's target device for future runs
        if self.els_optimizer:
            self.els_optimizer.els_target_device = new_target_device
            self.log_message(f"ELS Optimizer target device updated to '{new_target_device}' for next run.", "info")
        
        return True

    def _handle_critical_error(self, user_message, exception_obj=None, recoverable=False):
        full_traceback = traceback.format_exc() if exception_obj else "No exception object provided.";
        log_file_path = "gui_critical_runtime_error.log";
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_error_message = (
            f"{timestamp}\nCRITICAL ERROR: {user_message}\nException: {type(exception_obj).__name__}: {exception_obj}\nTraceback:\n{full_traceback}\n")
        current_logger = getattr(self, 'logger', None);
        is_dummy_logger = not current_logger or isinstance(current_logger,
                                                           type(_DummyLoggerClass())) if _DummyLoggerClass else True
        if not is_dummy_logger:
            pass
        else:
            log_file_path = os.path.join(os.getcwd(), log_file_path)
        try:
            with open(log_file_path, "a", encoding="utf-8") as f_log:
                f_log.write(formatted_error_message)
        except Exception as e_log_write:
            pass
        can_use_messagebox = hasattr(self, 'winfo_exists') and self.winfo_exists() and messagebox
        if recoverable:
            warning_message_for_user = f"{user_message}\n\nSome application features may be disabled or unstable. Please check logs for details."
            if can_use_messagebox: messagebox.showwarning("Feature Error", warning_message_for_user, parent=self)
            log_func = getattr(self, 'log_message', print);
            log_func(f"RECOVERABLE CRITICAL ERROR: {user_message}")
        elif exception_obj is not None:
            error_display_message = f"{user_message}\nThe application may need to close. Details logged to:\n'{os.path.abspath(log_file_path)}'"
            if can_use_messagebox: messagebox.showerror("Critical Application Error", error_display_message,
                                                        parent=self)
            if hasattr(self, 'destroy') and callable(self.destroy):
                try:
                    self.destroy()
                except Exception:
                    pass

    def on_frame_configure(self, canvas, scrollable_frame, event=None):
        if gui_utils: gui_utils.on_frame_configure(canvas, scrollable_frame, event, app_instance=self)

    def on_canvas_configure(self, event, scrollable_frame_id, canvas):
        if gui_utils: gui_utils.on_canvas_configure(event, scrollable_frame_id, canvas, app_instance=self)

    def _handle_canvas_scroll(self, event, canvas):
        if gui_utils: return gui_utils._handle_canvas_scroll(event, canvas, app_instance=self)
        return None

    def _bind_events_recursively(self, widget, scroll_command, enter_command):
        if gui_utils: gui_utils._bind_events_recursively(widget, scroll_command, enter_command, app_instance=self)

    def _update_ui_fields_from_agent_state(self):
        pass

    def _check_gui_queue(self):
        if self._shutting_down:
            return
        try:
            while True:
                message = self.gui_output_queue.get_nowait()
                if message.startswith(ELS_STATS_MSG_PREFIX):
                    stats_data_str = message.replace(ELS_STATS_MSG_PREFIX, "").strip()
                    try:
                        eval_data = eval(stats_data_str, {"__builtins__": {}},
                                         {'np': np, 'inf': float('inf'), '-inf': float('-inf'), 'nan': float('nan')})
                        is_valid_format = isinstance(eval_data, list) and (not eval_data or all(
                            isinstance(item, tuple) and len(item) == 5 and all(
                                isinstance(num, (int, float, np.number)) for num in item) for item in eval_data))
                        if is_valid_format:
                            self.els_fitness_history_data = eval_data
                        else:
                            pass
                        self._update_els_chart(self.els_fitness_history_data)
                    except Exception as e_eval:
                        pass
                elif message.startswith(ELS_LOG_PREFIX) or "EvolutionaryOptimizer" in message or any(
                        kw in message for kw in
                        ["PuffinZipAI", " AI Agent ", " AI(id=", "Q-Table", "model loaded", "saving model", "Gen ",
                         "Fitness Eval", "Mutation", "Breeding", "Champion", "Benchmark", "Complexity", "Stagnation",
                         "Hypermutation", "Adaptation", "Bottleneck", "Immigrant"]):
                    self._log_to_els_console(message)
                else:
                    self.log_message(message)
        except queue.Empty:
            pass
        except Exception as e_queue:
            if not self._shutting_down:
                pass

        if hasattr(self, 'after') and callable(self.after) and not self._shutting_down:
            try:
                self.after(100, self._check_gui_queue)
            except tk.TclError:
                if not self._shutting_down:
                    pass
                self._shutting_down = True

    def log_message(self, message_text, source="app"):
        if hasattr(self, 'output_scrolled_text') and hasattr(self.output_scrolled_text,
                                                             'winfo_exists') and self.output_scrolled_text.winfo_exists():
            try:
                self.output_scrolled_text.config(state='normal');
                self.output_scrolled_text.insert(tk.END,
                                                 str(message_text) + "\n");
                self.output_scrolled_text.see(
                    tk.END);
                self.output_scrolled_text.config(state='disabled')
            except tk.TclError:
                pass
        else:
            pass

    def _log_to_els_console(self, message_text):
        cleaned_message = str(message_text).replace(ELS_LOG_PREFIX, "").strip()
        msg_lower = cleaned_message.lower()

        if hasattr(self, 'els_diag_log_scrolled_text') and hasattr(self.els_diag_log_scrolled_text,
                                                                   'winfo_exists') and self.els_diag_log_scrolled_text.winfo_exists():
            try:
                self.els_diag_log_scrolled_text.config(state='normal')
                self.els_diag_log_scrolled_text.insert(tk.END, cleaned_message + "\n")
                self.els_diag_log_scrolled_text.see(tk.END)
                self.els_diag_log_scrolled_text.config(state='disabled')
            except tk.TclError:
                pass
        elif not hasattr(self, 'els_diag_log_scrolled_text'):
            pass

        display_in_main_els_log = False
        main_log_keywords = ["gen summary", "eval done", "best agent", "fitness eval", "els segment ended",
                             "els: new run started", "els (cont.) run", "pause signal sent", "resume signal sent",
                             "evolution focus", "adaptation strategy", "bottleneck", "benchmark set",
                             "benchmarks refreshed", "complexity target", "hypermutation triggered",
                             "stagnation update", "mutation rate adjusted", "champion saved", "champion loaded",
                             "status:"]
        verbose_diagnostic_keywords = ["puffinzai", " ai(id=", "q-table", "rle_error"]

        if any(keyword in msg_lower for keyword in main_log_keywords):
            display_in_main_els_log = True

        if "critical" in msg_lower or "error" in msg_lower:
            if not any(verbose_kw in msg_lower for verbose_kw in verbose_diagnostic_keywords):
                display_in_main_els_log = True

        if not display_in_main_els_log and (
                ELS_LOG_PREFIX.lower() in msg_lower or "evolutionaryoptimizer" in msg_lower):
            if not any(verbose_kw in msg_lower for verbose_kw in verbose_diagnostic_keywords):
                if "agent" in msg_lower and "fitness eval complete" in msg_lower and not (
                        "best agent" in msg_lower or "gen summary" in msg_lower):
                    pass
                else:
                    display_in_main_els_log = True

        if display_in_main_els_log and hasattr(self, 'els_log_scrolled_text') and hasattr(self.els_log_scrolled_text,
                                                                                          'winfo_exists') and self.els_log_scrolled_text.winfo_exists():
            try:
                self.els_log_scrolled_text.config(state='normal')
                self.els_log_scrolled_text.insert(tk.END, cleaned_message + "\n")
                self.els_log_scrolled_text.see(tk.END)
                self.els_log_scrolled_text.config(state='disabled')
            except tk.TclError:
                pass
        elif not hasattr(self, 'els_log_scrolled_text') and display_in_main_els_log:
            pass

    def _ai_task_thread_wrapper(self, actual_ai_method, method_args, method_kwargs):
        task_name = getattr(actual_ai_method, '__name__', "Unnamed_AI_Task");
        start_log_msg = f"--- Task Started: {task_name} ---"
        if self.gui_output_queue:
            self.gui_output_queue.put(start_log_msg)
        else:
            pass
        is_els_task_for_status_update = task_name in ["start_evolution", "continue_evolution"]
        if is_els_task_for_status_update and hasattr(self, 'els_status_label') and hasattr(self.els_status_label,
                                                                                           'winfo_exists') and self.els_status_label.winfo_exists():
            current_gen_display = (self.els_optimizer.total_generations_elapsed + (
                0 if self.els_is_paused else 1)) if self.els_optimizer and hasattr(self.els_optimizer,
                                                                                   'total_generations_elapsed') else 1
            try:
                self.after(0,
                           lambda: self.els_status_label.config(text=f"Status: Running - Gen {current_gen_display}..."))
            except tk.TclError:
                pass
        try:
            actual_ai_method(*method_args, **method_kwargs)
            if is_els_task_for_status_update and not self.gui_stop_event.is_set():
                self.els_run_completed = True
                if hasattr(self, 'els_status_label') and hasattr(self.els_status_label,
                                                                 'winfo_exists') and self.els_status_label.winfo_exists():
                    total_gens_completed = self.els_optimizer.total_generations_elapsed if self.els_optimizer else 'N/A'
                    try:
                        self.after(0, lambda: self.els_status_label.config(
                            text=f"Status: Completed (Total Gens: {total_gens_completed})"))
                    except tk.TclError:
                        pass
            elif is_els_task_for_status_update and self.gui_stop_event.is_set():
                if hasattr(self, 'els_status_label') and hasattr(self.els_status_label,
                                                                 'winfo_exists') and self.els_status_label.winfo_exists():
                    total_gens_at_stop = self.els_optimizer.total_generations_elapsed if self.els_optimizer else 'N/A'
                    try:
                        self.after(0, lambda: self.els_status_label.config(
                            text=f"Status: Stopped (Total Gens: {total_gens_at_stop})"))
                    except tk.TclError:
                        pass
        except Exception as e:
            tb_full_info = traceback.format_exc();
            exception_type_name = type(e).__name__;
            exception_message = str(e)
            gui_error_summary = f"ERROR in task ({task_name}): Type: {exception_type_name}"
            if exception_message.strip():
                gui_error_summary += f", Msg: '{exception_message.strip()}'"
            else:
                gui_error_summary += ", Msg: [No specific message from exception]"
            max_traceback_lines_gui = 7;
            tb_lines_list = tb_full_info.splitlines();
            traceback_snippet_for_gui = "\n".join(tb_lines_list[:2]) + (
                "\n  ...\n" + "\n".join(tb_lines_list[-(max_traceback_lines_gui - 3):]) if len(
                    tb_lines_list) > max_traceback_lines_gui else "\n".join(tb_lines_list[2:]))
            full_gui_error_message = f"{gui_error_summary}\nTraceback Snippet (Full details in application log):\n{traceback_snippet_for_gui}"
            if self.gui_output_queue:
                self.gui_output_queue.put(full_gui_error_message)
            else:
                pass
            if is_els_task_for_status_update:
                self.els_run_completed = False
                if hasattr(self, 'els_status_label') and hasattr(self.els_status_label,
                                                                 'winfo_exists') and self.els_status_label.winfo_exists():
                    try:
                        self.after(0, lambda: self.els_status_label.config(text="Status: Error! Check Logs."))
                    except tk.TclError:
                        pass
        finally:
            self.task_running = False
            if is_els_task_for_status_update: self.els_is_paused = False
            self.current_task_is_els = (self.els_run_completed and self.els_optimizer and hasattr(self.els_optimizer,
                                                                                                  'population') and self.els_optimizer.population)
            status_label_text = "";
            if hasattr(self, 'els_status_label') and self.els_status_label.winfo_exists(): status_label_text = getattr(
                self.els_status_label, 'cget', lambda x="": "")('text')
            is_final_state_idle_appropriate = not (is_els_task_for_status_update and (
                    self.els_run_completed or (self.gui_stop_event and self.gui_stop_event.is_set()) or (
                    "Error!" in status_label_text)))
            if is_final_state_idle_appropriate and hasattr(self, 'els_status_label') and hasattr(self.els_status_label,
                                                                                                 'winfo_exists') and self.els_status_label.winfo_exists():
                try:
                    self.after(0, lambda: self.els_status_label.config(text="Status: Idle"))
                except tk.TclError:
                    pass

            def final_gui_updates_safe():
                self._update_els_button_states();
                finish_log_msg = f"--- Task Finished: {task_name} ---";
                if self.gui_output_queue:
                    self.gui_output_queue.put(finish_log_msg)
                else:
                    pass
                if is_els_task_for_status_update and hasattr(self,
                                                             'gdv_tab_instance') and self.gdv_tab_instance and hasattr(
                    self.gdv_tab_instance, 'load_and_display_data'):
                    try:
                        self.after(50, self.gdv_tab_instance.load_and_display_data)
                    except Exception as e_gdv_refresh:
                        pass

            if hasattr(self, 'after') and callable(self.after) and not self._shutting_down:
                try:
                    self.after(0, final_gui_updates_safe)
                except tk.TclError:
                    pass
                except Exception as e_after_final:
                    pass
            elif not self._shutting_down:
                try:
                    final_gui_updates_safe()
                except Exception as e_direct_call_final:
                    pass

    def _start_ai_task(self, actual_ai_method, *args, **kwargs):
        if self.task_running: self.log_message(
            "An AI task is already running. Please wait or stop the current task."); return
        self.gui_stop_event.clear();
        self.task_running = True
        self.current_task_is_els = (actual_ai_method == getattr(self.els_optimizer, 'start_evolution',
                                                                None) or actual_ai_method == getattr(self.els_optimizer,
                                                                                                     'continue_evolution',
                                                                                                     None))
        self.els_is_paused = False
        if self.current_task_is_els:
            self.els_run_completed = False
        self._update_els_button_states()
        if self.current_task_is_els:
            self.els_fitness_history_data.clear();
            self._update_els_chart([])
            if hasattr(self, 'els_status_label') and hasattr(self.els_status_label,
                                                             'winfo_exists') and self.els_status_label.winfo_exists():
                try:
                    self.els_status_label.config(text="Status: Starting Evolution...")
                except tk.TclError:
                    pass
        self.current_task_thread = threading.Thread(target=self._ai_task_thread_wrapper,
                                                    args=(actual_ai_method, args, kwargs), daemon=True);
        self.current_task_thread.start()

    def request_task_stop(self):
        if self.task_running and self.current_task_thread and self.current_task_thread.is_alive():
            self.gui_stop_event.set()
            if self.current_task_is_els and self.els_is_paused and self.els_optimizer and hasattr(self.els_optimizer,
                                                                                                  'pause_event'): self.els_optimizer.pause_event.clear(); self.els_is_paused = False; self.log_message(
                "ELS task was paused, unpausing to process stop signal.")
            self.log_message("Stop signal sent to current AI task. Please wait for it to finalize...")

            if hasattr(self,
                       'stop_els_button') and self.stop_els_button.winfo_exists() and self.current_task_is_els: self.stop_els_button.config(
                state=tk.DISABLED)
        else:
            self.log_message("No task is currently running or thread is not alive.")

    def browse_folder(self, entry_widget_or_var):
        entry_widget_to_update = None;
        string_var_to_update = None
        if isinstance(entry_widget_or_var, tk.StringVar):
            string_var_to_update = entry_widget_or_var;
        elif isinstance(entry_widget_or_var, ttk.Entry):
            entry_widget_to_update = entry_widget_or_var;
        else:
            self.log_message("Browse Error: Invalid argument type.");
            return
        current_path_for_dialog = "";
        if string_var_to_update:
            current_path_for_dialog = string_var_to_update.get()
        elif entry_widget_to_update:
            current_path_for_dialog = entry_widget_to_update.get()
        initial_dir_for_dialog = os.getcwd()
        if current_path_for_dialog:
            abs_curr = os.path.abspath(current_path_for_dialog)
            if os.path.isdir(abs_curr):
                initial_dir_for_dialog = abs_curr
            elif os.path.exists(os.path.dirname(abs_curr)):
                initial_dir_for_dialog = os.path.dirname(abs_curr)
        folder_selected_path = filedialog.askdirectory(parent=self, initialdir=initial_dir_for_dialog,
                                                       title="Select Folder")
        if folder_selected_path:
            normalized_path = os.path.normpath(folder_selected_path)
            if string_var_to_update: string_var_to_update.set(normalized_path)
            if entry_widget_to_update and hasattr(entry_widget_to_update,
                                                  'winfo_exists') and entry_widget_to_update.winfo_exists():
                if not string_var_to_update or string_var_to_update.get() != normalized_path: entry_widget_to_update.delete(
                    0, tk.END);entry_widget_to_update.insert(0, normalized_path)
            elif not string_var_to_update and not entry_widget_to_update:
                self.log_message(f"Selected folder (no UI var/widget to update): {normalized_path}")

    def display_q_table(self):
        if not self._is_ai_agent_ready(for_training=False): return
        self.ai_agent.display_q_table_summary()

    def test_ai(self):
        if not self._is_ai_agent_ready(for_training=False): return
        try:
            num_items_test_var = getattr(self, 'test_items_entry_evo_tab', None)
            if not num_items_test_var:
                return
            num_items_test_str = num_items_test_var.get().strip()
            num_items_test = int(num_items_test_str)
            if num_items_test <= 0: self.log_message("Test items must be positive.");return
            self.ai_agent.test_agent_on_random_items(num_items_test)
        except ValueError:
            self.log_message("Invalid number for test items.")
        except AttributeError:
            self.log_message("UI elements for testing AI agent not found on Evolution Controls tab.")

    def save_model(self):
        if self.task_running: self.log_message("Cannot save model while task is running.");return
        if not self.ai_agent: self.log_message("AI Agent not initialized.");return
        filepath_to_save = filedialog.asksaveasfilename(defaultextension=".dat",
                                                        filetypes=[("PuffinZipAI Model", "*.dat"),
                                                                   ("All files", "*.*")],
                                                        initialdir=os.path.dirname(MODEL_FILE_DEFAULT),
                                                        initialfile=os.path.basename(MODEL_FILE_DEFAULT),
                                                        title="Save Main AI Model As", parent=self)
        if filepath_to_save: self.ai_agent.save_model(filepath_to_save)

    def load_model(self):
        if self.task_running: self.log_message("Cannot load model while task is running.");return
        if not self.ai_agent: self.log_message("AI Agent not initialized.");return
        filepath_to_load = filedialog.askopenfilename(defaultextension=".dat",
                                                      filetypes=[("PuffinZipAI Model", "*.dat"), ("All files", "*.*")],
                                                      initialdir=os.path.dirname(MODEL_FILE_DEFAULT),
                                                      title="Load Main AI Model", parent=self)
        if filepath_to_load:
            if self.ai_agent.load_model(filepath_to_load):
                self._update_ui_fields_from_agent_state();
                self.log_message(f"Main AI Model '{os.path.basename(filepath_to_load)}' loaded successfully.")
                self.app_target_device = self.ai_agent.target_device  # Update app's notion
                if hasattr(self,
                           'settings_content_tab') and self.settings_content_tab.winfo_exists() and self.settings_gui_module:
                    for child in self.settings_content_tab.winfo_children():
                        if isinstance(child, self.settings_gui_module.SettingsTab): child.load_settings(); break
            else:
                self.log_message(
                    f"Failed to load Main AI Model from '{os.path.basename(filepath_to_load)}'. Check logs.")

    def _is_ai_agent_ready(self, for_training=False):
        if not self.ai_agent: self.log_message("AI Agent not initialized. Please restart.");return False
        if self.task_running: self.log_message("An AI task is running. Please wait or stop.");return False
        return True

    def on_closing(self):
        self.logger.info("PMA: Saving GUI state on closing...")
        current_geometry = self.geometry()
        gui_state_to_save = {"main_window_geometry": current_geometry}
        settings_manager.save_gui_state(gui_state_to_save)
        self.logger.info(f"PMA: Saved window geometry: {current_geometry}")
        
        self._shutting_down = True
        if self.task_running:
            if messagebox.askyesno("Task Running", "An AI task is running.\nStop task and exit?", parent=self):
                self.request_task_stop();
                self.after(750, self.destroy_if_safe)
            else:
                self._shutting_down = False
                return
        else:
            self.destroy_if_safe()

    def destroy_if_safe(self):
        self._shutting_down = True
        self._cancel_analysis_refresh()
        if self.task_running and self.current_task_thread and self.current_task_thread.is_alive():
            if messagebox.askretrycancel("Task Still Finalizing",
                                         "AI task is shutting down. Retry closing (recommended), or cancel to wait more?",
                                         parent=self): self.after(1000, self.destroy_if_safe); return
        if self.els_optimizer and hasattr(self.els_optimizer, 'gui_stop_event'): self.els_optimizer.gui_stop_event.set()
        if self.els_optimizer and hasattr(self.els_optimizer, 'pause_event'): self.els_optimizer.pause_event.clear()
        if self.ai_agent and hasattr(self.ai_agent, 'gui_stop_event'): self.ai_agent.gui_stop_event.set()
        if self.current_task_thread and self.current_task_thread.is_alive(): self.current_task_thread.join(timeout=1.5)
        if self.current_task_thread and self.current_task_thread.is_alive(): pass
        try:
            super().destroy()
        except Exception as e_destroy_tk:
            pass
        finally:
            logging.shutdown()

    def save_els_state_gui(self):
        """Handles the GUI action to save the entire ELS session."""
        if self.task_running:
            self.log_message("Cannot save ELS state while a task is running.")
            return
        if not self.els_optimizer or not self.els_optimizer.population:
            self.log_message("ELS Info: No active ELS session or population to save.")
            return

        initial_dir = os.path.join(os.path.dirname(MODEL_FILE_DEFAULT), "els_sessions")
        os.makedirs(initial_dir, exist_ok=True)
        initial_file = f"els_session_gen_{self.els_optimizer.total_generations_elapsed}.pkl"
        
        filepath = filedialog.asksaveasfilename(
            parent=self,
            title="Save ELS Session State",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".pkl",
            filetypes=[("ELS Session Files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            self.els_optimizer.save_state(filepath)

    def load_els_state_gui(self):
        """Handles the GUI action to load an entire ELS session state."""
        if self.task_running:
            self.log_message("Cannot load ELS state while a task is running.")
            return
        if not self.els_optimizer:
            self.log_message("ELS Optimizer not available. Cannot load state.")
            return

        initial_dir = os.path.join(os.path.dirname(MODEL_FILE_DEFAULT), "els_sessions")
        filepath = filedialog.askopenfilename(
            parent=self,
            title="Load ELS Session State",
            initialdir=initial_dir,
            defaultextension=".pkl",
            filetypes=[("ELS Session Files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            if self.els_optimizer.load_state(filepath):
                self.els_run_completed = True
                self._update_els_button_states()
                # Refresh data viewers
                if hasattr(self, 'gdv_tab_instance') and self.gdv_tab_instance:
                    self.gdv_tab_instance.load_and_display_data()

    def start_evolution_process_gui(self):
        if not EvolutionaryOptimizer: self.log_message(
            "ELS Error: Evolutionary Optimizer system not available (class not loaded)."); return
        if self.els_optimizer is None:
            try:
                self.els_optimizer = EvolutionaryOptimizer(gui_output_queue=self.gui_output_queue,
                                                           gui_stop_event=self.gui_stop_event,
                                                           tuned_params=self.tuned_params,
                                                           dynamic_benchmarking_active=CONFIG_DYN_BENCH_ACTIVE_DEFAULT,
                                                           benchmark_refresh_interval_gens=CONFIG_DYN_BENCH_REFRESH_DEFAULT,
                                                           target_device=self.app_target_device);
            except Exception as e_reinit_els:
                self.log_message(
                    f"ELS Error: Failed to initialize Evolutionary Optimizer: {e_reinit_els}");
                self.els_optimizer = None;
                return
        if self.task_running: self.log_message("ELS Error: Another AI task is already running."); return
        selected_bm_strategy = self.els_initial_benchmark_strategy_var.get();
        user_bm_size_mb_override = None;
        user_bm_fixed_complexity_override = None
        if selected_bm_strategy == "Fixed Complexity Level":
            user_bm_fixed_complexity_override = self.els_fixed_complexity_var.get()
            if not user_bm_fixed_complexity_override: messagebox.showerror("Input Error",
                                                                           "Please select a complexity level for fixed strategy.",
                                                                           parent=self);return
        elif selected_bm_strategy == "Fixed Average Item Size (MB)":
            try:
                size_str = self.els_fixed_size_mb_var.get()
                if not size_str.strip(): messagebox.showerror("Input Error",
                                                              "Avg item size (MB) cannot be empty for fixed size.",
                                                              parent=self);return
                user_bm_size_mb_override = float(size_str)
                if user_bm_size_mb_override <= 0: messagebox.showerror("Input Error",
                                                                       "Avg item size (MB) must be positive.",
                                                                       parent=self);return
            except ValueError:
                messagebox.showerror("Input Error", "Invalid number for avg item size (MB).", parent=self);
                return
        self.els_optimizer.dynamic_benchmarking_active = CONFIG_DYN_BENCH_ACTIVE_DEFAULT;
        self.els_optimizer.benchmark_refresh_interval_gens = CONFIG_DYN_BENCH_REFRESH_DEFAULT
        self.els_optimizer.initial_benchmark_target_size_mb = user_bm_size_mb_override;
        self.els_optimizer.initial_benchmark_fixed_complexity_name = user_bm_fixed_complexity_override
        self.els_optimizer.els_target_device = self.app_target_device
        self.els_run_completed = False;
        self.els_fitness_history_data.clear();
        self._update_els_chart([])
        self.log_message("ELS Info: Starting new evolution process...");
        self._start_ai_task(self.els_optimizer.start_evolution)

    def continue_evolution_process_gui(self):
        if not self.els_optimizer: self.log_message("ELS Error: Evolutionary Optimizer not available.");return
        if self.task_running: self.log_message("ELS Error: Another AI task is already running.");return
        if not self.els_run_completed or not (
                hasattr(self.els_optimizer, 'population') and self.els_optimizer.population): self.log_message(
            "ELS Info: No completed ELS run or existing population to continue. Start new evolution."); return
        self.log_message("ELS Info: Continuing evolution process...");
        self.els_run_completed = False
        self.els_optimizer.initial_benchmark_target_size_mb = None;
        self.els_optimizer.initial_benchmark_fixed_complexity_name = None
        self.els_optimizer.els_target_device = self.app_target_device
        self._start_ai_task(self.els_optimizer.continue_evolution, additional_generations=None)

    def pause_els_task(self):
        if self.task_running and self.current_task_is_els and self.els_optimizer and hasattr(self.els_optimizer,
                                                                                             'pause_event') and not self.els_is_paused:
            self.els_optimizer.pause_event.set();
            self.els_is_paused = True;
            self.log_message("ELS Info: Pause signal sent.")
            if hasattr(self, 'els_status_label') and self.els_status_label.winfo_exists():
                try:
                    gen_at_pause = self.els_optimizer.total_generations_elapsed + (
                        0 if self.gui_stop_event.is_set() else 1) if self.els_optimizer else 'N/A';
                    self.els_status_label.config(text=f"Status: Paused - Gen {gen_at_pause}")
                except tk.TclError:
                    pass
        else:
            self.log_message("ELS Info: No pausable ELS task running or task already paused.")
        self._update_els_button_states()

    def resume_els_task(self):
        if self.task_running and self.current_task_is_els and self.els_optimizer and hasattr(self.els_optimizer,
                                                                                             'pause_event') and self.els_is_paused:
            self.els_optimizer.pause_event.clear();
            self.els_is_paused = False;
            self.log_message("ELS Info: Resume signal sent.")
            if hasattr(self, 'els_status_label') and self.els_status_label.winfo_exists():
                try:
                    gen_at_resume = self.els_optimizer.total_generations_elapsed + (
                        0 if self.gui_stop_event.is_set() else 1) if self.els_optimizer else 'N/A';
                    self.els_status_label.config(text=f"Status: Resuming - Gen {gen_at_resume}...")
                except tk.TclError:
                    pass
        else:
            self.log_message("ELS Info: No paused ELS task to resume.")
        self._update_els_button_states()

    def save_champion_agent_gui(self):
        if not self.els_optimizer or not hasattr(self.els_optimizer,
                                                 'best_agent_overall') or not self.els_optimizer.best_agent_overall: self.log_message(
            "ELS Error: No champion agent to save."); return
        initial_dir_path = os.path.dirname(MODEL_FILE_DEFAULT);
        champions_subdir_name = "els_champions";
        suggested_initial_dir = os.path.join(initial_dir_path, champions_subdir_name);
        os.makedirs(suggested_initial_dir, exist_ok=True)
        fitness_value_str = f"{self.els_optimizer.best_fitness_overall:.3f}" if hasattr(self.els_optimizer,
                                                                                        'best_fitness_overall') and self.els_optimizer.best_fitness_overall is not None else "unknown_fit"
        initial_filename_suggestion = f"els_champion_fit{fitness_value_str}.npy"
        filepath_to_save_champ = filedialog.asksaveasfilename(defaultextension=".npy",
                                                              filetypes=[("ELS Champion Model", "*.npy"),
                                                                         ("All PuffinZip Models", "*.dat;*.npy"),
                                                                         ("All files", "*.*")],
                                                              initialdir=suggested_initial_dir,
                                                              initialfile=initial_filename_suggestion,
                                                              title="Save ELS Champion Agent Config As", parent=self)
        if filepath_to_save_champ: self.els_optimizer.save_best_agent(filepath_to_save_champ)

    def load_champion_to_seed_gui(self):
        if not self.els_optimizer: self.log_message(
            "ELS Error: Evolutionary Optimizer not available for seeding.");return
        if self.task_running: self.log_message("ELS Error: Cannot load champion while task is running.");return
        initial_dir_path = os.path.dirname(MODEL_FILE_DEFAULT);
        champions_subdir_name = "els_champions";
        suggested_initial_dir_seed = os.path.join(initial_dir_path, champions_subdir_name)
        if not os.path.isdir(suggested_initial_dir_seed): suggested_initial_dir_seed = initial_dir_path
        filepath_to_load_seed = filedialog.askopenfilename(defaultextension=".npy",
                                                           filetypes=[("PuffinZip/ELS Models", "*.npy;*.dat"),
                                                                      ("All files", "*.*")],
                                                           initialdir=suggested_initial_dir_seed,
                                                           title="Load Champion or Model Config to Seed Evolution",
                                                           parent=self)
        if filepath_to_load_seed:
            try:
                if not filepath_to_load_seed.lower().endswith((".npy", ".dat")): self.log_message(
                    f"Warning: File '{os.path.basename(filepath_to_load_seed)}' has uncommon extension. Attempting load as NumPy dict.")
                loaded_model_data = np.load(filepath_to_load_seed, allow_pickle=True).item()
                if not isinstance(loaded_model_data, dict): self.log_message(
                    "Error: Loaded file is not valid model data dict.");return
                required_els_keys = ['len_thresholds', 'learning_rate', 'discount_factor', 'exploration_rate',
                                     'exploration_decay_rate', 'min_exploration_rate', 'rle_min_encodable_run']
                config_dict_for_seeding = {key: loaded_model_data.get(key) for key in required_els_keys if
                                           key in loaded_model_data}
                if 'q_table' in loaded_model_data: config_dict_for_seeding['q_table'] = loaded_model_data['q_table']
                config_dict_for_seeding['target_device'] = loaded_model_data.get('target_device',
                                                                                 self.app_target_device)
                if not all(
                        key in config_dict_for_seeding for key in required_els_keys if
                        key != 'q_table'): self.log_message(
                    f"Error: File '{os.path.basename(filepath_to_load_seed)}' missing essential hyperparams for ELS seeding after trying to adapt."); return
                if self.els_optimizer.add_champion_from_config(config_dict_for_seeding):
                    self.log_message(
                        f"Config from '{os.path.basename(filepath_to_load_seed)}' prepared for ELS seeding.")
                else:
                    self.log_message(
                        f"Failed to prepare config from '{os.path.basename(filepath_to_load_seed)}' for seeding. Check ELS logs.")
            except Exception as e_load_seed:
                self.log_message(
                    f"Error loading file '{filepath_to_load_seed}' for ELS seeding: {e_load_seed}");

    def _run_benchmark_generator_script(self):
        self.log_message("ELS Info: Starting benchmark data generator script via subprocess...")
        if not self.BENCHMARK_GENERATOR_SCRIPT_PATH or not os.path.exists(
                self.BENCHMARK_GENERATOR_SCRIPT_PATH): self.log_message(
            "ERROR: Benchmark generator script path not found or not configured.");return
        process = None;
        stdout_data_str = None;
        stderr_data_str = None;
        process_returncode = -1;
        script_completed_without_popen_error = False
        try:
            process_creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            process = subprocess.Popen([sys.executable, self.BENCHMARK_GENERATOR_SCRIPT_PATH], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, creationflags=process_creation_flags,
                                       encoding='utf-8')
            stdout_data_str, stderr_data_str = process.communicate(timeout=300);
            process_returncode = process.returncode;
            script_completed_without_popen_error = True
        except subprocess.TimeoutExpired:
            self.log_message("Benchmark generator script timed out (5 minutes).");
            if process and process.poll() is None:
                try:
                    process.kill();
                    process.wait(timeout=5);
                except Exception as e_kill:
                    pass
        except FileNotFoundError as e_fnf:
            self.log_message(
                f"ERROR: Benchmark generator script or Python executable not found: {e_fnf}");
        except Exception as e_run_script:
            self.log_message(
                f"Error running benchmark generator script: {type(e_run_script).__name__} - {e_run_script}");
        if script_completed_without_popen_error:
            if stdout_data_str: self.log_message("Benchmark Generator Script Output:\n" + stdout_data_str.strip())
            if stderr_data_str:
                self.log_message("Benchmark Generator Script Errors/Warnings:\n" + stderr_data_str.strip())
                if process_returncode != 0: pass
            if process_returncode == 0:
                self.log_message("Benchmark generator script finished successfully.");
                last_line_stdout = stdout_data_str.strip().splitlines()[
                    -1] if stdout_data_str and stdout_data_str.strip() else "";
                generated_path_confirmed = None;
                gen_path_from_script = ""
                path_match = re.search(r"(?:files created in|files at:)\s*['\"]?([^'\"\n]+)['\"]?", last_line_stdout,
                                       re.IGNORECASE)
                if path_match: gen_path_from_script = path_match.group(1).strip()
                if gen_path_from_script and os.path.isdir(gen_path_from_script):
                    generated_path_confirmed = os.path.abspath(gen_path_from_script);
                    self.log_message(f"Path detected from script: {generated_path_confirmed}")
                    if messagebox.askyesno("Update Benchmark Path?",
                                           f"Benchmark data generated at:\n{generated_path_confirmed}\n\nUpdate BENCHMARK_DATASET_PATH in Settings?\n(Requires save & restart for ELS use).",
                                           parent=self):
                        try:
                            from puffinzip_ai.utils import settings_manager
                            if settings_manager:
                                current_cfg_values = settings_manager.get_config_values();
                                current_cfg_values["BENCHMARK_DATASET_PATH"] = generated_path_confirmed
                                if settings_manager.save_config_values(current_cfg_values):
                                    self.log_message(
                                        "BENCHMARK_DATASET_PATH updated. Go to Settings, verify, save, and restart.")
                                    if hasattr(self,
                                               'settings_content_tab') and self.settings_content_tab.winfo_exists() and self.settings_gui_module:
                                        for child in self.settings_content_tab.winfo_children():
                                            if isinstance(child,
                                                          self.settings_gui_module.SettingsTab): child.load_settings(); break
                                else:
                                    self.log_message("Failed to auto-update BENCHMARK_DATASET_PATH. Set manually.")
                            else:
                                self.log_message(
                                    "Settings manager not available for auto-update. Please set BENCHMARK_DATASET_PATH manually in settings.")
                        except Exception as e_setting_update:
                            self.log_message(
                                f"Error auto-updating BENCHMARK_DATASET_PATH: {e_setting_update}. Please set manually.")
                elif stdout_data_str:
                    self.log_message(
                        "Could not automatically determine generated benchmark path from script output. Please check the output above or application logs, then set the path in Settings if needed.")
                else:
                    default_gen_path_check = GENERATED_BENCHMARK_DEFAULT_PATH or os.path.join(PROJECT_ROOT_FROM_GUI,
                                                                                              "data", "benchmark_sets",
                                                                                              GENERATED_BENCHMARK_SUBDIR_NAME or "generated_bench_data")
                    self.log_message(
                        f"No output from benchmark generator. Check logs or default path '{default_gen_path_check}' and set in Settings if needed.")
            else:
                self.log_message(f"Benchmark generator script exited with error code {process_returncode}.")
        elif process_returncode != 0 and not script_completed_without_popen_error:
            self.log_message(
                f"Benchmark generator script could not be run or failed before completion (Popen error or similar). Check application logs for details.")

    def generate_numeric_benchmark_gui(self):
        if self.task_running: self.log_message(
            "Cannot generate benchmark data while another AI task is running."); return
        self._start_ai_task(self._run_benchmark_generator_script)


if __name__ == '__main__':
    main_runner_log_path = "primary_main_app_standalone_runner.log"
    try:
        main_runner_log_path = os.path.join(LOGS_DIR_PATH, "primary_main_app_standalone_runner.log");
    except Exception:
        pass
    if 'LOGS_DIR_PATH' in globals() and LOGS_DIR_PATH and not os.path.exists(LOGS_DIR_PATH):
        os.makedirs(LOGS_DIR_PATH, exist_ok=True)
    elif not ('LOGS_DIR_PATH' in globals() and LOGS_DIR_PATH) and not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True);
        main_runner_log_path = os.path.join("logs",
                                            "primary_main_app_standalone_runner.log")
    main_block_logger = logging.getLogger("PuffinZip_Standalone_Runner_Main");
    main_block_logger.setLevel(logging.DEBUG)
    if not main_block_logger.handlers:
        fh_main = logging.FileHandler(main_runner_log_path, mode='a', encoding='utf-8');
        fh_main.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'));
        main_block_logger.addHandler(fh_main)
        ch_main = logging.StreamHandler(sys.stdout);
        ch_main.setLevel(logging.INFO);
        ch_main.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - RUNNER_MAIN: %(message)s'));
        main_block_logger.addHandler(ch_main)
    app_ver_for_log = CFG_APP_VERSION if 'CFG_APP_VERSION' in globals() else "UnknownVer"
    main_block_logger.info(f"--- Running primary_main_app.py directly (App Version: {app_ver_for_log}) ---")
    app_instance = None
    try:
        tuned_params_for_app_standalone = {}
        try:
            from puffinzip_ai.utils import performance_tuner

            tuned_params_for_app_standalone = performance_tuner.get_tuned_parameters();
        except Exception as e_pt_main_standalone:
            tuned_params_for_app_standalone = {
                "AGENTS_PER_THROTTLE_CHECK": 5, "THROTTLE_SLEEP_DURATION_BENCH_EVAL": 0.001}
        app_instance = PuffinZipApp(tuned_params=tuned_params_for_app_standalone)
        app_instance.mainloop()
    except Exception as e_app_test_run:
        try:
            import tkinter as tk_err_final_standalone;
            from tkinter import messagebox as mb_err_final_standalone

            root_fb_final_sa = tk_err_final_standalone.Tk();
            root_fb_final_sa.withdraw();
            mb_err_final_standalone.showerror("PuffinZipAI Critical Error (Standalone Test)",
                                              f"Fatal error during app startup/runtime.\nCheck logs: '{os.path.abspath(main_runner_log_path)}'\nError: {e_app_test_run}",
                                              parent=None);
            root_fb_final_sa.destroy()
        except Exception:
            pass
    finally:
        if app_instance and hasattr(app_instance, 'logger') and app_instance.logger and hasattr(app_instance.logger,
                                                                                                'handlers'):
            for handler_to_close in list(app_instance.logger.handlers):
                try:
                    handler_to_close.close();
                    app_instance.logger.removeHandler(handler_to_close)
                except Exception:
                    pass
        logging.shutdown()