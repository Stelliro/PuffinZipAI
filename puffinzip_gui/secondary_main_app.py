# PuffinZipAI_Project/puffinzip_gui/secondary_main_app.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
import sys
import os

try:
    from . import gui_utils
except ImportError:
    try:
        from puffinzip_gui import gui_utils
    except ImportError:
        gui_utils = None

if gui_utils and hasattr(gui_utils, '_get_theme_attr'):
    _get_theme_attr_l = gui_utils._get_theme_attr
else:
    def _get_theme_attr_l(app_instance, attr_name, default_value):
        log_to_use = getattr(app_instance, 'logger', logging.getLogger("GuiLayoutSetup_FallbackLogger"))
        if isinstance(log_to_use, logging.Logger) and not log_to_use.handlers:
            _h = logging.StreamHandler(sys.stdout);
            _f = logging.Formatter('%(asctime)s - LAYOUT_FALLBACK - %(levelname)s - %(message)s');
            _h.setFormatter(_f);
            log_to_use.addHandler(_h);
            log_to_use.setLevel(logging.INFO)
        if app_instance and hasattr(app_instance, attr_name):
            val = getattr(app_instance, attr_name)
            if val is not None: return val
        elif app_instance and not hasattr(app_instance, attr_name) and log_to_use:
            pass
        return default_value

try:
    from puffinzip_ai.config import (
        DEFAULT_TRAIN_BATCH_SIZE, DEFAULT_ALLOWED_LEARN_EXTENSIONS,
        DEFAULT_FOLDER_LEARN_BATCH_SIZE, DEFAULT_BATCH_COMPRESS_EXTENSIONS
    )
    from puffinzip_ai.utils.benchmark_evaluator import DataComplexity
except ImportError:
    DEFAULT_TRAIN_BATCH_SIZE = 32
    DEFAULT_ALLOWED_LEARN_EXTENSIONS = [".txt", ".md"]
    DEFAULT_FOLDER_LEARN_BATCH_SIZE = 16
    DEFAULT_BATCH_COMPRESS_EXTENSIONS = [".txt", ".log"]


    class DataComplexity:
        @staticmethod
        def get_member_names(): return ["SIMPLE", "MODERATE", "COMPLEX"]

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
SYMBOL_DIAGNOSTICS = "🔬"
SYMBOL_SAVE_SESSION = "💾"
SYMBOL_LOAD_SESSION = "📤"


def _create_section_frame(parent_frame, title_text, app_instance):
    frame = ttk.LabelFrame(parent_frame, text=title_text, style="TLabelframe", padding=(15, 12, 15, 15))
    frame.pack(fill=tk.X, padx=5, pady=(10, 15))
    return frame


def populate_evolution_controls_tab_content(app):  # Renamed from populate_evolution_lab_tab_content
    active_chart_utils = app.chart_utils

    app.els_canvas = tk.Canvas(app.evolution_controls_tab, bg=_get_theme_attr_l(app, 'FRAME_BG', '#333333'),
                               highlightthickness=0, bd=0)
    app.els_scrollbar = ttk.Scrollbar(app.evolution_controls_tab, orient=tk.VERTICAL, command=app.els_canvas.yview,
                                      style="Vertical.TScrollbar")
    app.els_canvas.configure(yscrollcommand=app.els_scrollbar.set)
    app.els_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=(2, 0))
    app.els_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0), pady=(2, 0))

    app.els_scrollable_frame = ttk.Frame(app.els_canvas, style="Scrollable.TFrame", padding=(15, 15))
    app.els_scrollable_frame_id = app.els_canvas.create_window((0, 0), window=app.els_scrollable_frame, anchor="nw")

    app.els_scrollable_frame.bind("<Configure>",
                                  lambda e: app.on_frame_configure(app.els_canvas, app.els_scrollable_frame, e),
                                  add="+")
    app.els_canvas.bind("<Configure>",
                        lambda e, sfid=app.els_scrollable_frame_id, cv=app.els_canvas: app.on_canvas_configure(e, sfid,
                                                                                                               cv),
                        add="+")
    pf = app.els_scrollable_frame

    benchmark_config_frame = _create_section_frame(pf, "Benchmark Setup for New Evolution Run", app)
    benchmark_config_frame.columnconfigure(1, weight=1)
    ttk.Label(benchmark_config_frame, text="Strategy:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=5)
    app.els_initial_benchmark_strategy_var = tk.StringVar(value="Adaptive (Fitness-based)")
    benchmark_strategies = ["Adaptive (Fitness-based)", "Fixed Complexity Level", "Fixed Average Item Size (MB)"]
    app.els_benchmark_strategy_combo = ttk.Combobox(benchmark_config_frame,
                                                    textvariable=app.els_initial_benchmark_strategy_var,
                                                    values=benchmark_strategies, state="readonly", width=30,
                                                    font=_get_theme_attr_l(app, 'FONT_NORMAL', None), style="TCombobox")
    app.els_benchmark_strategy_combo.grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=5, padx=(0, 5))
    app.els_benchmark_strategy_combo.bind("<<ComboboxSelected>>", app._on_benchmark_strategy_change)
    ttk.Label(benchmark_config_frame, text="Fixed Complexity:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=5)
    complexity_levels = DataComplexity.get_member_names() if DataComplexity else ["SIMPLE", "MODERATE", "COMPLEX"]
    app.els_fixed_complexity_var = tk.StringVar(value=complexity_levels[0] if complexity_levels else "")
    app.els_fixed_complexity_combo = ttk.Combobox(benchmark_config_frame, textvariable=app.els_fixed_complexity_var,
                                                  values=complexity_levels, state='disabled', width=28,
                                                  font=_get_theme_attr_l(app, 'FONT_NORMAL', None), style="TCombobox")
    app.els_fixed_complexity_combo.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=(0, 5))
    ttk.Label(benchmark_config_frame, text="Avg Item Size (MB):").grid(row=2, column=0, sticky=tk.W, padx=(0, 5),
                                                                       pady=5)
    app.els_fixed_size_mb_var = tk.StringVar(value="0.1")
    app.els_fixed_size_mb_entry = ttk.Entry(benchmark_config_frame, textvariable=app.els_fixed_size_mb_var,
                                            state='disabled', width=10, style="TEntry")
    app.els_fixed_size_mb_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(0, 5))
    app._on_benchmark_strategy_change()

    config_info_frame = _create_section_frame(pf, "Note on Evolution Parameters", app)
    ttk.Label(config_info_frame,
              text="Core Evolution parameters (population size, generations, mutation rates, etc.) are managed via the 'Settings' tab. Changes there apply to new or continued evolution runs.",
              justify=tk.LEFT, wraplength=480, font=_get_theme_attr_l(app, 'FONT_NORMAL', None), style="TLabel").pack(
        padx=5, pady=10, fill=tk.X)

    action_buttons_frame_main = ttk.Frame(pf, style="TFrame", padding=(0, 10, 0, 5));
    action_buttons_frame_main.pack(fill=tk.X, pady=(10, 0))
    max_cols_main_els = 3;
    for i in range(max_cols_main_els): action_buttons_frame_main.columnconfigure(i, weight=1, minsize=120)
    app.start_evolution_button = ttk.Button(action_buttons_frame_main, text=SYMBOL_PLAY + " Start New Evolution",
                                            style="TButton", command=lambda: app.start_evolution_process_gui())
    app.start_evolution_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.EW, ipady=5)
    app.continue_els_button = ttk.Button(action_buttons_frame_main, text=SYMBOL_CONTINUE + " Continue Evolution",
                                         style="TButton", command=lambda: app.continue_evolution_process_gui(),
                                         state=tk.DISABLED)
    app.continue_els_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW, ipady=5)
    app.save_champion_button = ttk.Button(action_buttons_frame_main, text=SYMBOL_CHAMPION + " Save Champion Agent",
                                          style="TButton", command=lambda: app.save_champion_agent_gui(),
                                          state=tk.DISABLED)
    app.save_champion_button.grid(row=0, column=2, padx=(5, 0), pady=5, sticky=tk.EW, ipady=5)
    app.pause_els_button = ttk.Button(action_buttons_frame_main, text=SYMBOL_PAUSE + " Pause Evolution",
                                      style="TButton", command=lambda: app.pause_els_task(), state=tk.DISABLED)
    app.pause_els_button.grid(row=1, column=0, padx=(0, 5), pady=5, sticky=tk.EW, ipady=5)
    app.resume_els_button = ttk.Button(action_buttons_frame_main, text=SYMBOL_PLAY + " Resume Evolution",
                                       style="TButton", command=lambda: app.resume_els_task(), state=tk.DISABLED)
    app.resume_els_button.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW, ipady=5)
    app.stop_els_button = ttk.Button(action_buttons_frame_main, text=SYMBOL_STOP + " Stop Evolution Run",
                                     style="TButton", command=lambda: app.request_task_stop(), state=tk.DISABLED)
    app.stop_els_button.grid(row=1, column=2, padx=(5, 0), pady=5, sticky=tk.EW, ipady=5)

    utility_buttons_frame = ttk.Frame(pf, style="TFrame", padding=(0, 5, 0, 5));
    utility_buttons_frame.pack(fill=tk.X, pady=(5, 5))
    utility_buttons_frame.columnconfigure(0, weight=1);
    utility_buttons_frame.columnconfigure(1, weight=1)
    app.load_champion_to_els_button = ttk.Button(utility_buttons_frame,
                                                 text=SYMBOL_SEED + " Load Agent to Seed Evolution", style="TButton",
                                                 command=lambda: app.load_champion_to_seed_gui())
    app.load_champion_to_els_button.grid(row=0, column=0, padx=(0, 5), pady=(5, 5), sticky=tk.EW, ipady=5)
    app.generate_benchmark_button = ttk.Button(utility_buttons_frame,
                                               text=SYMBOL_SETTINGS + " Generate Static Benchmark Files",
                                               style="TButton", command=lambda: app.generate_numeric_benchmark_gui())
    app.generate_benchmark_button.grid(row=0, column=1, padx=(5, 0), pady=(5, 5), sticky=tk.EW, ipady=5)
    
    session_mgmt_frame = _create_section_frame(pf, "Save/Load Full ELS Session", app)
    session_mgmt_frame.columnconfigure(0, weight=1)
    session_mgmt_frame.columnconfigure(1, weight=1)
    app.save_els_state_button = ttk.Button(session_mgmt_frame, text=f"{SYMBOL_SAVE_SESSION} Save ELS State", command=app.save_els_state_gui)
    app.save_els_state_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.EW, ipady=5)
    app.load_els_state_button = ttk.Button(session_mgmt_frame, text=f"{SYMBOL_LOAD_SESSION} Load ELS State", command=app.load_els_state_gui)
    app.load_els_state_button.grid(row=0, column=1, padx=(5, 0), pady=5, sticky=tk.EW, ipady=5)


    adaptation_frame = _create_section_frame(pf, "Tune Evolution Focus (During Run)", app)
    adaptation_frame.columnconfigure(0, weight=1);
    adaptation_frame.columnconfigure(1, weight=1);
    adaptation_frame.columnconfigure(2, weight=1);
    adaptation_frame.columnconfigure(3, weight=1)
    app.bottleneck_low_button = ttk.Button(adaptation_frame, text=SYMBOL_BOTTLENECK_LOW + " Gentle Adaptation",
                                           style="TButton", command=app.apply_low_bottleneck, state=tk.DISABLED)
    app.bottleneck_low_button.grid(row=0, column=0, padx=(0, 5), pady=5, sticky=tk.EW, ipady=5)
    app.bottleneck_medium_button = ttk.Button(adaptation_frame, text=SYMBOL_BOTTLENECK_MED + " Balanced Adaptation",
                                              style="TButton", command=app.apply_medium_bottleneck, state=tk.DISABLED)
    app.bottleneck_medium_button.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW, ipady=5)
    app.bottleneck_high_button = ttk.Button(adaptation_frame, text=SYMBOL_BOTTLENECK_HIGH + " Strong Adaptation",
                                            style="TButton", command=app.apply_high_bottleneck, state=tk.DISABLED)
    app.bottleneck_high_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.EW, ipady=5)
    app.clear_adaptation_button = ttk.Button(adaptation_frame, text=SYMBOL_RESET + " Reset Adaptation", style="TButton",
                                             command=app.clear_bottleneck_strategy, state=tk.DISABLED)
    app.clear_adaptation_button.grid(row=0, column=3, padx=(5, 0), pady=5, sticky=tk.EW, ipady=5)

    app._update_els_button_states()

    log_display_paned_window = ttk.PanedWindow(pf, orient=tk.HORIZONTAL, style="Horizontal.TPanedwindow")
    log_display_paned_window.pack(expand=True, fill=tk.BOTH, pady=(15, 5))

    data_area_frame = ttk.LabelFrame(log_display_paned_window, text="Evolution Status & Log", style="TLabelframe",
                                     padding=(10, 10))
    log_display_paned_window.add(data_area_frame, weight=3)

    app.els_status_label = ttk.Label(data_area_frame, text="Status: Idle", anchor=tk.W, justify=tk.LEFT, style="TLabel",
                                     font=_get_theme_attr_l(app, 'FONT_NORMAL', None))
    app.els_status_label.pack(fill=tk.X, padx=5, pady=(0, 5))

    app.els_log_scrolled_text = scrolledtext.ScrolledText(data_area_frame, wrap=tk.WORD, height=10,
                                                          font=_get_theme_attr_l(app, 'FONT_MONO', ('Consolas', 9)),
                                                          bg=_get_theme_attr_l(app, 'TEXT_AREA_BG', '#1E1E1E'),
                                                          fg=_get_theme_attr_l(app, 'TEXT_AREA_FG', '#D0D0D0'),
                                                          insertbackground=_get_theme_attr_l(app, 'FG_COLOR',
                                                                                             '#FFFFFF'),
                                                          selectbackground=_get_theme_attr_l(app, 'ACCENT_COLOR',
                                                                                             '#007ACC'),
                                                          selectforeground=_get_theme_attr_l(app, 'TEXT_AREA_BG',
                                                                                             '#1E1E1E'), borderwidth=1,
                                                          relief="solid", padx=10, pady=10, spacing1=3, spacing3=3,
                                                          undo=True)
    app.els_log_scrolled_text.pack(expand=True, fill=tk.BOTH, padx=0, pady=0);
    app.els_log_scrolled_text.configure(state='disabled')

    diag_log_frame = ttk.LabelFrame(log_display_paned_window, text=f"{SYMBOL_DIAGNOSTICS} Diagnostic Log (All Data)",
                                    style="TLabelframe", padding=(10, 10))
    log_display_paned_window.add(diag_log_frame, weight=2)

    app.els_diag_log_scrolled_text = scrolledtext.ScrolledText(diag_log_frame, wrap=tk.WORD, height=10,
                                                               font=_get_theme_attr_l(app, 'FONT_MONO',
                                                                                      ('Consolas', 8)),
                                                               bg=_get_theme_attr_l(app, 'INPUT_BG', '#252526'),
                                                               fg=_get_theme_attr_l(app, 'LABEL_FG', '#C0C0C0'),
                                                               insertbackground=_get_theme_attr_l(app, 'FG_COLOR',
                                                                                                  '#FFFFFF'),
                                                               selectbackground=_get_theme_attr_l(app, 'ACCENT_COLOR',
                                                                                                  '#007ACC'),
                                                               selectforeground=_get_theme_attr_l(app, 'INPUT_BG',
                                                                                                  '#252526'),
                                                               borderwidth=1, relief="solid", padx=8, pady=8,
                                                               spacing1=2, spacing3=2, undo=True)
    app.els_diag_log_scrolled_text.pack(expand=True, fill=tk.BOTH, padx=0, pady=0);
    app.els_diag_log_scrolled_text.configure(state='disabled')

    # Remove the Main AI Management section from its old position
    # The new position is inside the main scrollable frame (`pf`)
    # ... code that previously created main_ai_manage_frame here is now deleted ...

    if hasattr(app, 'els_canvas') and app.els_canvas:
        sfc_els = lambda e: app.els_canvas.focus_set() if hasattr(app,
                                                                  'els_canvas') and app.els_canvas and app.els_canvas.winfo_exists() else None
        scroll_cmd_els = lambda e, cv=app.els_canvas: app._handle_canvas_scroll(e, cv)
        app.els_canvas.bind("<MouseWheel>", scroll_cmd_els, add="+");
        app.els_canvas.bind("<Button-4>", scroll_cmd_els, add="+")
        app.els_canvas.bind("<Button-5>", scroll_cmd_els, add="+");
        app.els_canvas.bind("<Enter>", sfc_els, add="+")
        if hasattr(app, '_bind_events_recursively'): app._bind_events_recursively(pf, scroll_cmd_els, sfc_els)


def populate_changelog_tab_content(app, changelog_file_path_from_primary, changelog_filename_from_primary):
    cta = scrolledtext.ScrolledText(app.changelog_tab, wrap=tk.WORD,
                                    font=_get_theme_attr_l(app, 'FONT_MONO', ('Segoe UI', 10)),
                                    bg=_get_theme_attr_l(app, 'TEXT_AREA_BG', '#1E1E1E'),
                                    fg=_get_theme_attr_l(app, 'TEXT_AREA_FG', '#D0D0D0'),
                                    insertbackground=_get_theme_attr_l(app, 'FG_COLOR', '#FFFFFF'),
                                    selectbackground=_get_theme_attr_l(app, 'ACCENT_COLOR', '#007ACC'),
                                    selectforeground=_get_theme_attr_l(app, 'TEXT_AREA_BG', '#1E1E1E'), borderwidth=1,
                                    relief="solid", padx=15, pady=15, spacing1=3, spacing3=3, undo=True)
    cta.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    actual_changelog_path = os.path.normpath(changelog_file_path_from_primary)
    try:
        if os.path.exists(actual_changelog_path):
            with open(actual_changelog_path, 'r', encoding='utf-8') as f:
                cta.insert(tk.END, f.read())
        else:
            cta.insert(tk.END,
                       f"Changelog File Not Found:\n{actual_changelog_path}\n\nPlease create '{changelog_filename_from_primary}' in project root.")
    except Exception as e_cl_load:
        cta.insert(tk.END, f"Error loading '{changelog_filename_from_primary}':\n{str(e_cl_load)}");
        traceback.print_exc()
    cta.configure(state='disabled')