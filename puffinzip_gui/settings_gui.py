# PuffinZipAI_Project/puffinzip_gui/settings_gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import logging
import sys
import json
import tkinter.font

try:
    from puffinzip_ai.utils import settings_manager
    from puffinzip_ai.utils import hardware_detector
    from puffinzip_ai.utils.benchmark_evaluator import DataComplexity
except ImportError as e_settings_imports:
    settings_manager = None;
    hardware_detector = None;
    DataComplexity = None

LABEL_FG_FALLBACK = "#D4D4D4";
INPUT_BG_FALLBACK = "#252525";
INPUT_FG_FALLBACK = "#D4D4D4"
FG_COLOR_FALLBACK = "#D4D4D4";
FRAME_BG_FALLBACK = "#333333";
ACCENT_COLOR_FALLBACK = "#0078D4"
SGUI_FONT_FAMILY_PRIMARY = "Segoe UI";
SGUI_FONT_SIZE_BASE = 10;
SGUI_FONT_SIZE_SMALL = 9;
SGUI_FONT_SIZE_LARGE = 12
SGUI_FONT_NORMAL_FALLBACK = (SGUI_FONT_FAMILY_PRIMARY, SGUI_FONT_SIZE_BASE)
SGUI_FONT_BOLD_FALLBACK = (SGUI_FONT_FAMILY_PRIMARY, SGUI_FONT_SIZE_BASE, "bold")
SGUI_FONT_SECTION_TITLE_FALLBACK = (SGUI_FONT_FAMILY_PRIMARY, SGUI_FONT_SIZE_LARGE, "bold")
SGUI_FONT_BUTTON_FALLBACK = (SGUI_FONT_FAMILY_PRIMARY, SGUI_FONT_SIZE_BASE, "normal")
SGUI_FONT_SMALL_FALLBACK = (SGUI_FONT_FAMILY_PRIMARY, SGUI_FONT_SIZE_SMALL)
STYLE_PREFIX = "SettingsGUI."

GUI_THEMES_FILE = "gui_themes.json"


class SettingsTab(ttk.Frame):
    def __init__(self, parent, app_instance, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.app = app_instance
        self.logger = getattr(app_instance, 'logger', logging.getLogger("SettingsGUIFallback"))
        if not self.logger.handlers:
            fb_handler = logging.StreamHandler(sys.stdout);
            fb_handler.setFormatter(logging.Formatter('%(asctime)s - SettingsGUI_FB - %(levelname)s - %(message)s'));
            self.logger.addHandler(fb_handler);
            self.logger.setLevel(logging.INFO)
        self.frame_bg = getattr(self.app, 'FRAME_BG', FRAME_BG_FALLBACK);
        self.label_fg = getattr(self.app, 'LABEL_FG', LABEL_FG_FALLBACK);
        self.input_bg = getattr(self.app, 'INPUT_BG', INPUT_BG_FALLBACK);
        self.input_fg = getattr(self.app, 'INPUT_FG', INPUT_FG_FALLBACK);
        self.fg_color_for_insert = getattr(self.app, 'FG_COLOR', FG_COLOR_FALLBACK);
        self.accent_color = getattr(self.app, 'ACCENT_COLOR', ACCENT_COLOR_FALLBACK)
        self.font_normal = getattr(self.app, 'FONT_NORMAL', SGUI_FONT_NORMAL_FALLBACK);
        self.font_bold = getattr(self.app, 'FONT_BOLD', SGUI_FONT_BOLD_FALLBACK);
        self.font_small = getattr(self.app, 'FONT_SMALL', SGUI_FONT_SMALL_FALLBACK);
        self.font_section_title = getattr(self.app, 'FONT_SECTION_TITLE', SGUI_FONT_SECTION_TITLE_FALLBACK);
        self.font_button = getattr(self.app, 'FONT_BUTTON', SGUI_FONT_BUTTON_FALLBACK)

        if settings_manager is None or hardware_detector is None or DataComplexity is None:
            error_message = "CRITICAL ERROR:\nOne or more required utility modules failed to import:\n";
            if settings_manager is None: error_message += "- Settings Manager\n"
            if hardware_detector is None: error_message += "- Hardware Detector\n"
            if DataComplexity is None: error_message += "- DataComplexity\n"
            error_message += "This settings tab will be non-functional."
            error_label = ttk.Label(self, text=error_message, foreground="red", justify=tk.LEFT, wraplength=380,
                                    font=self.font_bold);
            error_label.pack(padx=20, pady=50, expand=True);
            return

        self.theme_presets_loaded = self._load_theme_presets()
        self.editable_settings_meta = settings_manager.EDITABLE_SETTINGS
        self.current_settings_vars = {};
        self.device_options_map = {};
        self.complexity_options = []

        try:
            available_families = tkinter.font.families(self.app)
            self.system_font_families = sorted(
                list(set(f for f in available_families if not f.startswith('@'))))
            if not self.system_font_families:
                raise tk.TclError("No usable fonts found")
        except tk.TclError:
            self.system_font_families = ["Arial", "Calibri", "Consolas", "Courier New", "Georgia", "Helvetica",
                                         "Lucida Console", "MS Sans Serif", "Segoe UI", "System", "Tahoma",
                                         "Times New Roman", "Verdana"]

        self._setup_internal_styles();
        self._setup_ui();
        self.load_settings()

    def _load_theme_presets(self):
        script_dir_gui = os.path.dirname(os.path.abspath(__file__))
        themes_file_path_final = os.path.join(script_dir_gui, GUI_THEMES_FILE)

        try:
            if os.path.exists(themes_file_path_final):
                with open(themes_file_path_final, 'r', encoding='utf-8') as f:
                    loaded_themes = json.load(f)
                if isinstance(loaded_themes, dict) and loaded_themes:
                    return loaded_themes
                else:
                    pass
            else:
                pass
        except json.JSONDecodeError:
            pass
        except Exception as e:
            pass
        return {"Nordic Dark (Default)": {"THEME_BG_COLOR": "#2E3440", "THEME_FG_COLOR": "#ECEFF4",
                                          "THEME_FRAME_BG": "#3B4252", "THEME_ACCENT_COLOR": "#88C0D0",
                                          "THEME_INPUT_BG": "#434C5E", "THEME_TEXT_AREA_BG": "#2E3440",
                                          "THEME_BUTTON_BG": "#4C566A", "THEME_BUTTON_FG": "#ECEFF4",
                                          "THEME_ERROR_FG": "#BF616A", "FONT_FAMILY_PRIMARY_CONFIG": "Segoe UI",
                                          "FONT_FAMILY_MONO_CONFIG": "Consolas"}}.copy()

    def _filter_font_combobox(self, event, combobox_widget, all_font_list):
        current_text = combobox_widget.get().lower()
        if not current_text:
            combobox_widget['values'] = all_font_list
        else:
            filtered_list = [font for font in all_font_list if current_text in font.lower()];
            combobox_widget[
                'values'] = filtered_list if filtered_list else all_font_list
        if hasattr(combobox_widget, 'event_generate'): combobox_widget.event_generate("<Down>")

    def _setup_internal_styles(self):
        style = ttk.Style(self);
        style.configure(f"{STYLE_PREFIX}TFrame", background=self.frame_bg)
        labelframe_style_name = f"{STYLE_PREFIX}Section.TLabelFrame"
        style.configure(labelframe_style_name, background=self.frame_bg, borderwidth=1, relief="solid",
                        bordercolor=getattr(self.app, 'TAB_BORDER_COLOR', "#4A4A4A"))
        style.configure(f"{labelframe_style_name}.Label", background=self.frame_bg, foreground=self.accent_color,
                        font=self.font_section_title, padding=(0, 0, 0, 4))
        try:
            base_layout = style.layout("TLabelframe");
            if base_layout: style.layout(labelframe_style_name, base_layout)
        except tk.TclError:
            pass
        style.configure(f"{STYLE_PREFIX}TLabel", background=self.frame_bg, foreground=self.label_fg,
                        font=self.font_normal, anchor="w")
        style.configure(f"{STYLE_PREFIX}Tooltip.TLabel", background="#FFFFE0", foreground="black", relief="solid",
                        borderwidth=1, padding=4, font=self.font_small)
        style.configure(f"{STYLE_PREFIX}Error.TLabel", background=self.frame_bg,
                        foreground=getattr(self.app, 'ERROR_FG_COLOR', 'red'), font=self.font_normal)
        style.configure(f"{STYLE_PREFIX}TEntry", fieldbackground=self.input_bg, foreground=self.input_fg,
                        insertbackground=self.fg_color_for_insert, font=self.font_normal, padding=(6, 4),
                        relief="solid", bordercolor="#555555", borderwidth=1)
        style.map(f"{STYLE_PREFIX}TEntry", bordercolor=[('focus', self.accent_color)])
        style.configure(f"{STYLE_PREFIX}TCheckbutton", background=self.frame_bg, foreground=self.label_fg,
                        font=self.font_normal)
        style.map(f"{STYLE_PREFIX}TCheckbutton",
                  indicatorbackground=[('!selected', self.input_bg), ('selected', self.accent_color)],
                  indicatorforeground=[('!selected', self.input_fg), ('selected', self.input_bg)])
        style.configure(f"{STYLE_PREFIX}Path.TFrame", background=self.frame_bg)
        style.configure(f"{STYLE_PREFIX}TCombobox", fieldbackground=self.input_bg, background=self.input_bg,
                        foreground=self.input_fg, arrowcolor=self.label_fg, insertbackground=self.fg_color_for_insert,
                        font=self.font_normal, padding=(5, 5), relief="solid", bordercolor="#555555", borderwidth=1)
        style.map(f"{STYLE_PREFIX}TCombobox", bordercolor=[('focus', self.accent_color)],
                  selectbackground=[('readonly', self.accent_color)], selectforeground=[
                ('readonly', self.input_bg if self.accent_color != self.input_bg else self.label_fg)])
        self.app.option_add(f"*TCombobox*Listbox*background", self.input_bg);
        self.app.option_add(f"*TCombobox*Listbox*foreground", self.label_fg);
        self.app.option_add(f"*TCombobox*Listbox*selectBackground", self.accent_color);
        self.app.option_add(f"*TCombobox*Listbox*selectForeground",
                            self.input_bg if self.accent_color != self.input_bg else self.label_fg);
        self.app.option_add(f"*TCombobox*Listbox*font", self.font_normal)

    def _create_setting_group(self, parent_frame_for_group, title_text):
        group_frame = ttk.LabelFrame(parent_frame_for_group, text=title_text,
                                     style=f"{STYLE_PREFIX}Section.TLabelFrame", padding=(15, 10, 15, 15));
        group_frame.pack(fill=tk.X, expand=True, pady=(0, 15), padx=5);
        group_frame.grid_columnconfigure(1, weight=1);
        return group_frame

    def _create_device_selection_combobox(self, parent_frame, tk_var, key, metadata):
        if not hardware_detector: return None
        # --- Diagnostic Logging Start ---
        hd_logger = logging.getLogger("puffinzip_ai.hardware_detector")
        original_level = hd_logger.level
        original_handlers = list(hd_logger.handlers)
        handler_was_added = False
        temp_handler = None
        if not original_handlers or isinstance(original_handlers[0], logging.NullHandler):
            temp_handler = logging.StreamHandler(sys.stdout)
            temp_handler.setFormatter(logging.Formatter('%(asctime)s - HD_DIAGNOSTIC - %(levelname)s - %(name)s - %(message)s'))
            hd_logger.addHandler(temp_handler)
            hd_logger.setLevel(logging.DEBUG)
            handler_was_added = True
        # --- Diagnostic Logging End ---
        self.logger.info("Getting processing device options for settings UI...")
        detected_options = hardware_detector.get_processing_device_options()
        # --- Restore Logger State ---
        if handler_was_added and temp_handler:
            hd_logger.removeHandler(temp_handler)
        hd_logger.setLevel(original_level)
        # --- End Restore ---
        self.device_options_map = {display: value for display, value in detected_options}
        display_names = [opt[0] for opt in detected_options]
        if not display_names:
            display_names = ["CPU (Detection Failed)"]
            self.device_options_map = {"CPU (Detection Failed)": "CPU"}
        combobox = ttk.Combobox(parent_frame, textvariable=tk_var, values=display_names,
                                style=f"{STYLE_PREFIX}TCombobox", state="readonly", width=35)
        combobox.set(display_names[0])
        return combobox

    def _create_data_complexity_combobox(self, parent_frame, tk_var, key, metadata):
        if not DataComplexity: return None
        self.complexity_options = DataComplexity.get_member_names();
        if not self.complexity_options: self.complexity_options = ["SIMPLE", "MODERATE", "COMPLEX"]
        combobox = ttk.Combobox(parent_frame, textvariable=tk_var, values=self.complexity_options,
                                style=f"{STYLE_PREFIX}TCombobox", state="readonly", width=28);
        combobox.set(self.complexity_options[0] if self.complexity_options else "");
        return combobox

    def _add_setting_to_group(self, group_widget, setting_key, setting_metadata, current_row_index):
        label_text = setting_metadata.get("label", setting_key.replace('_', ' ').title());
        tooltip_text_for_label = setting_metadata.get("tooltip", "")
        lbl_widget = ttk.Label(group_widget, text=f"{label_text}:", style=f"{STYLE_PREFIX}TLabel");
        lbl_widget.grid(row=current_row_index, column=0, sticky=tk.W, padx=(0, 10), pady=6)
        if tooltip_text_for_label: self.create_tooltip(lbl_widget, tooltip_text_for_label)
        var_type_expected = setting_metadata["type"];
        tk_variable_for_setting = None;
        input_widget_for_setting = None
        widget_padding_x = (0, 0);
        widget_padding_y = (3, 6);
        options_logic = setting_metadata.get("options_logic")
        if options_logic:
            tk_variable_for_setting = tk.StringVar()
            if options_logic == "detect_processing_devices":
                input_widget_for_setting = self._create_device_selection_combobox(group_widget, tk_variable_for_setting,
                                                                                  setting_key, setting_metadata)
            elif options_logic == "get_data_complexity_levels":
                input_widget_for_setting = self._create_data_complexity_combobox(group_widget, tk_variable_for_setting,
                                                                                 setting_key, setting_metadata)
            else:
                pass
        if input_widget_for_setting is None:
            if setting_key in ["FONT_FAMILY_PRIMARY_CONFIG", "FONT_FAMILY_MONO_CONFIG"]:
                tk_variable_for_setting = tk.StringVar();
                input_widget_for_setting = ttk.Combobox(group_widget, textvariable=tk_variable_for_setting,
                                                        values=self.system_font_families,
                                                        style=f"{STYLE_PREFIX}TCombobox", width=28)
                input_widget_for_setting.bind('<KeyRelease>', lambda event, cb=input_widget_for_setting,
                                                                     fl=self.system_font_families: self._filter_font_combobox(
                    event, cb, fl))
                input_widget_for_setting.bind('<FocusIn>', lambda event, cb=input_widget_for_setting,
                                                                  fl=self.system_font_families: setattr(cb, 'values',
                                                                                                        fl))
            elif var_type_expected == int:
                tk_variable_for_setting = tk.IntVar();
                input_widget_for_setting = ttk.Entry(group_widget,
                                                     textvariable=tk_variable_for_setting,
                                                     width=12,
                                                     style=f"{STYLE_PREFIX}TEntry")
            elif var_type_expected == float:
                tk_variable_for_setting = tk.DoubleVar();
                input_widget_for_setting = ttk.Entry(group_widget,
                                                     textvariable=tk_variable_for_setting,
                                                     width=12,
                                                     style=f"{STYLE_PREFIX}TEntry")
            elif var_type_expected == bool:
                tk_variable_for_setting = tk.BooleanVar();
                input_widget_for_setting = ttk.Checkbutton(group_widget,
                                                           variable=tk_variable_for_setting,
                                                           style=f"{STYLE_PREFIX}TCheckbutton")
            elif var_type_expected == str:
                tk_variable_for_setting = tk.StringVar()
                if setting_metadata.get("is_path"):
                    path_entry_frame = ttk.Frame(group_widget, style=f"{STYLE_PREFIX}Path.TFrame")
                    path_text_entry = ttk.Entry(path_entry_frame, textvariable=tk_variable_for_setting, width=35,
                                                style=f"{STYLE_PREFIX}TEntry");
                    path_text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=1)
                    browse_button = ttk.Button(path_entry_frame, text="Browse",
                                               command=lambda v=tk_variable_for_setting,
                                                              k_name=setting_key: self._browse_path(v, k_name),
                                               style="Small.TButton");
                    browse_button.pack(side=tk.LEFT, padx=(8, 0))
                    input_widget_for_setting = path_entry_frame
                else:
                    input_widget_for_setting = ttk.Entry(group_widget, textvariable=tk_variable_for_setting, width=30,
                                                         style=f"{STYLE_PREFIX}TEntry")
        if input_widget_for_setting and tk_variable_for_setting:
            if var_type_expected == bool:
                input_widget_for_setting.grid(row=current_row_index, column=1, sticky=tk.W, padx=widget_padding_x,
                                              pady=widget_padding_y)
            else:
                input_widget_for_setting.grid(row=current_row_index, column=1, sticky=tk.EW, padx=widget_padding_x,
                                              pady=widget_padding_y)
            self.current_settings_vars[setting_key] = tk_variable_for_setting
        else:
            unsupported_widget = ttk.Label(group_widget, text="UI Error",
                                           style=f"{STYLE_PREFIX}Error.TLabel");
            unsupported_widget.grid(
                row=current_row_index, column=1, sticky=tk.EW, padx=widget_padding_x,
                pady=widget_padding_y);

    def _setup_ui(self):
        main_content_frame = ttk.Frame(self, style=f"{STYLE_PREFIX}TFrame");
        main_content_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        canvas = tk.Canvas(main_content_frame, bg=self.frame_bg, highlightthickness=0);
        scrollbar = ttk.Scrollbar(main_content_frame, orient="vertical", command=canvas.yview,
                                  style="Vertical.TScrollbar")
        scrollable_frame_content = ttk.Frame(canvas, style=f"{STYLE_PREFIX}TFrame", padding=(20, 20, 25, 20));
        scrollable_frame_window_id = canvas.create_window((0, 0), window=scrollable_frame_content, anchor="nw")

        def _configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _configure_canvas_width(event):
            canvas.itemconfig(scrollable_frame_window_id, width=event.width)

        scrollable_frame_content.bind("<Configure>", _configure_scroll_region);
        canvas.bind("<Configure>", _configure_canvas_width);
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True);
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        try:
            quick_themes_group = self._create_setting_group(scrollable_frame_content, "Quick Themes")
            theme_button_frame = ttk.Frame(quick_themes_group, style=f"{STYLE_PREFIX}TFrame");
            theme_button_frame.pack(fill=tk.X, pady=(5, 0))
            max_cols = 4
            for i, (theme_name, theme_values) in enumerate(self.theme_presets_loaded.items()):
                btn_bg = theme_values.get("THEME_BUTTON_BG", self.app.BUTTON_BG);
                btn_fg = theme_values.get("THEME_BUTTON_FG", self.app.BUTTON_FG)
                unique_btn_style_name = f"{STYLE_PREFIX}{theme_name.replace(' ', '').replace('(', '').replace(')', '').replace('.', '')}.TButton";
                s = ttk.Style(self)
                s.configure(unique_btn_style_name, font=self.font_button, background=btn_bg, foreground=btn_fg,
                            padding=(8, 5), relief=tk.RAISED, borderwidth=1)
                s.map(unique_btn_style_name,
                      background=[('pressed', theme_values.get("THEME_ACCENT_COLOR", self.app.ACCENT_COLOR)),
                                  ('active', btn_bg)], relief=[('pressed', tk.SUNKEN)])
                theme_btn = ttk.Button(theme_button_frame, text=theme_name,
                                       command=lambda tn=theme_name: self._apply_theme_preset(tn),
                                       style=unique_btn_style_name)
                theme_btn.grid(row=i // max_cols, column=i % max_cols, padx=4, pady=4, sticky=tk.EW)
                theme_button_frame.grid_columnconfigure(i % max_cols, weight=1)
            setting_groups_layout = {
                "Processing & Logging": ["ACCELERATION_TARGET_DEVICE", "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED"],
                # Moved Debug toggle here
                "User Interface & Theme": ["FONT_FAMILY_PRIMARY_CONFIG", "FONT_FAMILY_MONO_CONFIG",
                                           "FONT_SIZE_BASE_CONFIG", "THEME_BG_COLOR",
                                           "THEME_FG_COLOR", "THEME_FRAME_BG",
                                           "THEME_ACCENT_COLOR", "THEME_INPUT_BG",
                                           "THEME_TEXT_AREA_BG", "THEME_BUTTON_BG",
                                           "THEME_BUTTON_FG", "THEME_ERROR_FG", ],
                "Evolution Algorithm - General": ["DEFAULT_POPULATION_SIZE",
                                                  "DEFAULT_NUM_GENERATIONS",
                                                  "DEFAULT_ADDITIONAL_ELS_GENERATIONS",
                                                  "DEFAULT_ELITISM_COUNT", ],
                "Mutation & Adaptation Strategies": ["DEFAULT_MUTATION_RATE",
                                                     "STAGNATION_GENERATIONS_THRESHOLD",
                                                     "MUTATION_RATE_BOOST_FACTOR",
                                                     "HYPERMUTATION_STAGNATION_THRESHOLD",
                                                     "HYPERMUTATION_FRACTION", ],
                "Breeding, Diversity & Agent Structure": ["ADVANCED_CROSSOVER_PROBABILITY",
                                                          "RANDOM_IMMIGRANT_INTERVAL",
                                                          "RANDOM_IMMIGRANT_FRACTION", ],
                "RLE Parameter Evolution Limits": ["RLE_MIN_RUN_INIT_MIN", "RLE_MIN_RUN_INIT_MAX",
                                                   "RLE_MIN_RUN_MUTATION_PROB",
                                                   "RLE_MIN_RUN_BOUNDS_MIN",
                                                   "RLE_MIN_RUN_BOUNDS_MAX"],
                "Evolutionary Learning - Benchmarking": ["DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT",
                                                         "DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS",
                                                         "INITIAL_BENCHMARK_COMPLEXITY_LEVEL",
                                                         "BENCHMARK_DATASET_PATH", ], }
            all_available_setting_keys = set(self.editable_settings_meta.keys());
            keys_already_grouped = set()
            for group_title, keys_in_group in setting_groups_layout.items():
                group_frame_widget = self._create_setting_group(scrollable_frame_content, group_title);
                row_idx_in_group = 0
                for setting_key_to_add in keys_in_group:
                    if setting_key_to_add in self.editable_settings_meta:
                        self._add_setting_to_group(group_frame_widget, setting_key_to_add,
                                                   self.editable_settings_meta[setting_key_to_add],
                                                   row_idx_in_group);
                        row_idx_in_group += 1;
                        keys_already_grouped.add(
                            setting_key_to_add)
                    else:
                        pass
            remaining_ungrouped_keys = all_available_setting_keys - keys_already_grouped
            if remaining_ungrouped_keys:
                misc_settings_group = self._create_setting_group(scrollable_frame_content, "Other Available Settings");
                row_idx_misc = 0
                for key_misc in sorted(list(remaining_ungrouped_keys)):
                    if key_misc in self.editable_settings_meta: self._add_setting_to_group(misc_settings_group,
                                                                                           key_misc,
                                                                                           self.editable_settings_meta[
                                                                                               key_misc],
                                                                                           row_idx_misc); row_idx_misc += 1
            button_frame_container = ttk.Frame(scrollable_frame_content, style=f"{STYLE_PREFIX}TFrame",
                                               padding=(0, 15, 0, 0));
            button_frame_container.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))
            button_frame_actual = ttk.Frame(button_frame_container, style=f"{STYLE_PREFIX}TFrame");
            button_frame_actual.pack(anchor=tk.E)
            defaults_button = ttk.Button(button_frame_actual, text="Load Defaults", command=self.load_defaults,
                                         style="TButton");
            defaults_button.pack(side=tk.LEFT, padx=(0, 8))
            load_current_button = ttk.Button(button_frame_actual, text="Reload Current", command=self.load_settings,
                                             style="TButton");
            load_current_button.pack(side=tk.LEFT, padx=8)
            save_button = ttk.Button(button_frame_actual, text="Save Settings", command=self.save_settings,
                                     style="TButton");
            save_button.pack(side=tk.LEFT, padx=(8, 0))
        except Exception as e_setup_ui_detailed:
            for child in scrollable_frame_content.winfo_children(): child.destroy()
            error_text = f"Error populating settings UI:\n{type(e_setup_ui_detailed).__name__}: {str(e_setup_ui_detailed)[:150]}...\n(See application logs for full details)"
            error_label_specific = ttk.Label(scrollable_frame_content, text=error_text,
                                             style=f"{STYLE_PREFIX}Error.TLabel", justify=tk.LEFT, wraplength=max(300,
                                                                                                                  scrollable_frame_content.winfo_width() - 40 if scrollable_frame_content.winfo_width() > 50 else 300));
            error_label_specific.pack(padx=10, pady=20, expand=True, fill=tk.BOTH)
            if hasattr(self.app, 'log_message'): self.app.log_message(
                f"ERROR in Settings Tab UI setup: {e_setup_ui_detailed}", "error_no_prefix")

    def _apply_theme_preset(self, theme_name):
        if theme_name not in self.theme_presets_loaded: return
        preset_values = self.theme_presets_loaded[theme_name];

        base_font_size_from_theme = preset_values.get("FONT_SIZE_BASE_CONFIG")
        if base_font_size_from_theme is not None and isinstance(base_font_size_from_theme, int):
            if "FONT_SIZE_BASE_CONFIG" in self.current_settings_vars:
                self.current_settings_vars["FONT_SIZE_BASE_CONFIG"].set(base_font_size_from_theme)

        for key, value in preset_values.items():
            if key == "FONT_SIZE_BASE_CONFIG" and base_font_size_from_theme is not None: continue
            if key == "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED" and isinstance(value,
                                                                        bool) and "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED" in self.current_settings_vars:
                self.current_settings_vars["DEBUG_LOG_CONSOLE_OUTPUT_ENABLED"].set(value)
                continue

            if key in self.current_settings_vars:
                tk_var = self.current_settings_vars[key]
                try:
                    tk_var.set(value)
                except tk.TclError as e:
                    pass
            else:
                pass
        if hasattr(self.app, 'log_message'): self.app.log_message(
            f"Theme preset '{theme_name}' applied to UI fields. Click 'Save Settings' to persist.", "info_no_prefix")

    def _browse_path(self, tk_string_var_for_path: tk.StringVar, setting_key_name: str):
        setting_meta = self.editable_settings_meta.get(setting_key_name, {});
        is_folder_path_type = setting_meta.get("path_type", "file") == "folder" or any(
            keyword in setting_key_name.upper() for keyword in ["DIR", "FOLDER"]) or (
                                      "PATH" in setting_key_name.upper() and not any(
                                  setting_key_name.lower().endswith(ext) for ext in
                                  [".dat", ".json", ".txt", ".log", ".csv"]))
        current_path_value = tk_string_var_for_path.get();
        initial_directory_for_dialog = os.getcwd()
        if current_path_value:
            abs_current_path = os.path.abspath(current_path_value)
            if os.path.isdir(abs_current_path):
                initial_directory_for_dialog = abs_current_path
            elif os.path.exists(os.path.dirname(abs_current_path)):
                initial_directory_for_dialog = os.path.dirname(abs_current_path)
        selected_path = None
        if is_folder_path_type:
            selected_path = filedialog.askdirectory(parent=self.app,
                                                    title=f"Select Folder for {setting_meta.get('label', setting_key_name)}",
                                                    initialdir=initial_directory_for_dialog)
        else:
            selected_path = filedialog.askopenfilename(parent=self.app,
                                                       title=f"Select File for {setting_meta.get('label', setting_key_name)}",
                                                       initialdir=initial_directory_for_dialog)
        if selected_path: tk_string_var_for_path.set(os.path.normpath(selected_path))

    def load_settings(self):
        if settings_manager is None: return
        current_config_values = settings_manager.get_config_values()
        for setting_key, tk_var_instance in self.current_settings_vars.items():
            meta_for_key = self.editable_settings_meta.get(setting_key, {});
            value_from_config = current_config_values.get(setting_key, meta_for_key.get("default"))
            if meta_for_key.get("options_logic") == "detect_processing_devices":
                display_to_set = value_from_config
                for display_name, config_val in self.device_options_map.items():
                    if config_val == value_from_config: display_to_set = display_name; break
                if display_to_set not in self.device_options_map.keys() and value_from_config in self.device_options_map.keys():
                    display_to_set = value_from_config
                elif display_to_set not in self.device_options_map.keys() and self.device_options_map:
                    display_to_set = list(self.device_options_map.keys())[0]
                tk_var_instance.set(display_to_set)
            elif meta_for_key.get("options_logic") == "get_data_complexity_levels":
                tk_var_instance.set(value_from_config if value_from_config in self.complexity_options else (
                    self.complexity_options[0] if self.complexity_options else ""))
            elif setting_key in ["FONT_FAMILY_PRIMARY_CONFIG", "FONT_FAMILY_MONO_CONFIG"]:
                tk_var_instance.set(value_from_config if value_from_config else "")
            else:
                if value_from_config is None:
                    if isinstance(tk_var_instance, tk.StringVar):
                        tk_var_instance.set("")
                    elif isinstance(tk_var_instance, (tk.IntVar, tk.DoubleVar)):
                        tk_var_instance.set(0)
                    elif isinstance(tk_var_instance, tk.BooleanVar):
                        tk_var_instance.set(False)
                else:
                    try:
                        tk_var_instance.set(value_from_config)
                    except tk.TclError as e_set_var_tcl:
                        if isinstance(tk_var_instance, tk.StringVar):
                            tk_var_instance.set(str(value_from_config))
                        elif isinstance(tk_var_instance, tk.BooleanVar):
                            tk_var_instance.set(bool(value_from_config))
                        else:
                            tk_var_instance.set(0)
        if hasattr(self.app, 'log_message'): self.app.log_message(
            "Settings Tab: Loaded current settings from config file into UI fields.")

    def load_defaults(self):
        if settings_manager is None: return
        for setting_key, tk_var_instance in self.current_settings_vars.items():
            meta_for_key = self.editable_settings_meta[setting_key];
            default_value_from_meta = meta_for_key.get("default")
            if meta_for_key.get("options_logic") == "detect_processing_devices":
                display_to_set = default_value_from_meta
                for display_name, config_val in self.device_options_map.items():
                    if config_val == default_value_from_meta: display_to_set = display_name; break
                tk_var_instance.set(display_to_set)
            elif meta_for_key.get("options_logic") == "get_data_complexity_levels":
                tk_var_instance.set(default_value_from_meta if default_value_from_meta in self.complexity_options else (
                    self.complexity_options[0] if self.complexity_options else ""))
            elif setting_key in ["FONT_FAMILY_PRIMARY_CONFIG", "FONT_FAMILY_MONO_CONFIG"]:
                tk_var_instance.set(default_value_from_meta if default_value_from_meta else "")
            else:
                if default_value_from_meta is None:
                    if isinstance(tk_var_instance, tk.StringVar):
                        tk_var_instance.set("")
                    elif isinstance(tk_var_instance, (tk.IntVar, tk.DoubleVar)):
                        tk_var_instance.set(0)
                    elif isinstance(tk_var_instance, tk.BooleanVar):
                        tk_var_instance.set(False)
                else:
                    try:
                        tk_var_instance.set(default_value_from_meta)
                    except tk.TclError:
                        if isinstance(tk_var_instance, (tk.IntVar, tk.DoubleVar)):
                            tk_var_instance.set(0)
                        elif isinstance(tk_var_instance, tk.BooleanVar):
                            tk_var_instance.set(False)
                        else:
                            tk_var_instance.set("")
        if hasattr(self.app, 'log_message'): self.app.log_message(
            "Settings Tab: Default settings loaded into UI fields (not saved yet).")

    def save_settings(self):
        if settings_manager is None: return
        new_values_dict_to_save = {};
        validation_successful = True
        settings_requiring_restart = set();
        critical_els_params_changed = False

        if self.app.task_running and self.app.current_task_is_els:
            temp_new_values_dict = {}  # Check potential changes first
            for setting_key, tk_var_instance in self.current_settings_vars.items():
                setting_meta_info = self.editable_settings_meta[setting_key]
                ui_value_raw = tk_var_instance.get()
                converted_val_check = None
                # Simplified conversion for check only
                try:
                    if setting_meta_info["type"] == bool:
                        converted_val_check = bool(ui_value_raw)
                    elif setting_meta_info["type"] == int:
                        converted_val_check = int(float(ui_value_raw))
                    elif setting_meta_info["type"] == float:
                        converted_val_check = float(ui_value_raw)
                    else:
                        converted_val_check = str(ui_value_raw) if not setting_meta_info.get(
                            "options_logic") else ui_value_raw
                except:
                    converted_val_check = ui_value_raw
                temp_new_values_dict[setting_key] = converted_val_check

            # Now check if any *critical ELS* params changed
            original_cfg_vals_check = settings_manager.get_config_values()
            for key, new_val_check in temp_new_values_dict.items():
                is_critical_els_param = key.startswith("DEFAULT_") or key in [
                    "ADVANCED_CROSSOVER_PROBABILITY", "RLE_MIN_RUN_INIT_MIN",
                    "RLE_MIN_RUN_INIT_MAX", "STAGNATION_GENERATIONS_THRESHOLD",
                    "MIN_THRESHOLDS_COUNT", "MAX_THRESHOLDS_COUNT"
                ]  # More comprehensive list of critical ELS params
                original_val_check = original_cfg_vals_check.get(key)

                if setting_manager.EDITABLE_SETTINGS[key].get("options_logic") == "detect_processing_devices":
                    original_val_check_display = original_val_check
                    for d_name, c_val in self.device_options_map.items():
                        if c_val == original_val_check: original_val_check_display = d_name; break
                    if new_val_check != original_val_check_display and self.device_options_map.get(
                            new_val_check) != original_val_check:
                        if is_critical_els_param: critical_els_params_changed = True; break

                elif new_val_check != original_val_check:
                    if is_critical_els_param:
                        critical_els_params_changed = True;
                        break

            if critical_els_params_changed:
                messagebox.showerror("Save Error - ELS Active",
                                     "Cannot save changes to core ELS parameters "
                                     "while an ELS run is active.\n\n"
                                     "Theme/font/logging changes can be saved, but critical ELS settings are locked.\n"
                                     "Stop the current ELS run to modify its core parameters.", parent=self.app)
                return

        for setting_key, tk_var_instance in self.current_settings_vars.items():
            setting_meta_info = self.editable_settings_meta[setting_key];
            ui_value_raw = tk_var_instance.get()
            converted_value_for_saving = None;
            options_logic_type = setting_meta_info.get("options_logic")
            original_config_value = settings_manager.get_config_values().get(setting_key)

            if options_logic_type == "detect_processing_devices":
                converted_value_for_saving = self.device_options_map.get(ui_value_raw, "CPU");
            elif options_logic_type == "get_data_complexity_levels":
                converted_value_for_saving = ui_value_raw;
            else:
                try:
                    expected_python_type = setting_meta_info["type"]
                    if expected_python_type == int:
                        converted_value_for_saving = int(float(ui_value_raw))
                    elif expected_python_type == float:
                        converted_value_for_saving = float(ui_value_raw)
                    elif expected_python_type == bool:
                        converted_value_for_saving = bool(ui_value_raw)
                    elif expected_python_type == str:
                        if setting_meta_info.get("is_path") and isinstance(ui_value_raw,
                                                                           str) and not ui_value_raw.strip():
                            converted_value_for_saving = None
                        elif setting_meta_info.get("is_path") and isinstance(ui_value_raw, str):
                            converted_value_for_saving = os.path.normpath(ui_value_raw.strip())
                        elif isinstance(ui_value_raw, str):
                            converted_value_for_saving = ui_value_raw
                        else:
                            converted_value_for_saving = str(ui_value_raw)
                    else:
                        converted_value_for_saving = ui_value_raw
                    if converted_value_for_saving is not None and not isinstance(converted_value_for_saving,
                                                                                 bool) and not isinstance(
                        converted_value_for_saving, str):
                        if "min" in setting_meta_info and converted_value_for_saving < setting_meta_info[
                            "min"]: messagebox.showerror("Validation Error",
                                                         f"Value for '{setting_meta_info.get('label', setting_key)}' ({converted_value_for_saving}) is below minimum ({setting_meta_info['min']}).",
                                                         parent=self.app); validation_successful = False; break
                        if "max" in setting_meta_info and converted_value_for_saving > setting_meta_info[
                            "max"]: messagebox.showerror("Validation Error",
                                                         f"Value for '{setting_meta_info.get('label', setting_key)}' ({converted_value_for_saving}) is above maximum ({setting_meta_info['max']}).",
                                                         parent=self.app); validation_successful = False; break
                except ValueError as e_val_convert:
                    messagebox.showerror("Validation Error",
                                         f"Invalid value for '{setting_meta_info.get('label', setting_key)}'. Expected type: {setting_meta_info['type'].__name__}.\nDetails: {e_val_convert}",
                                         parent=self.app);
                    validation_successful = False;
                    break
                except Exception as e_process_setting:
                    messagebox.showerror("Error",
                                         f"Unexpected error processing setting '{setting_meta_info.get('label', setting_key)}': {e_process_setting}",
                                         parent=self.app);
                    validation_successful = False;
                    break
            new_values_dict_to_save[setting_key] = converted_value_for_saving

            is_non_els_critical_setting = setting_key in ["ACCELERATION_TARGET_DEVICE",
                                                          "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED"]
            is_theme_setting = setting_key.startswith("THEME_") or setting_key.startswith("FONT_")

            original_value_comp = original_config_value
            new_value_comp = converted_value_for_saving

            if options_logic_type == "detect_processing_devices":  # Ensure comparison uses same format (map display name to config value)
                pass  # new_value_comp is already the config value

            value_changed = new_value_comp != original_value_comp

            if value_changed and (
                    is_non_els_critical_setting or (critical_els_params_changed and not is_theme_setting)):
                settings_requiring_restart.add(setting_key)

            if not validation_successful: break

        if validation_successful:
            save_status = settings_manager.save_config_values(new_values_dict_to_save)
            if save_status:
                if hasattr(self.app, 'log_message'): self.app.log_message("Settings Tab: Settings saved to config.py.")

                # Check specifically for theme/font/logging related changes that might need immediate UI update or specific logger restart advice
                theme_related_keys_changed_for_immediate_reload = any(
                    (k.startswith("THEME_") or k.startswith("FONT_")) and
                    new_values_dict_to_save.get(k) != settings_manager.get_config_values().get(k)
                    for k in new_values_dict_to_save
                )
                logging_console_toggle_changed = "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED" in new_values_dict_to_save and \
                                                 new_values_dict_to_save[
                                                     "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED"] != settings_manager.get_config_values().get(
                    "DEBUG_LOG_CONSOLE_OUTPUT_ENABLED")

                info_msg = "Settings saved successfully to config.py."

                if theme_related_keys_changed_for_immediate_reload and hasattr(self.app, 'reload_and_apply_theme'):
                    info_msg += "\n\nTheme/font changes applied."
                    self.app.reload_and_apply_theme()

                final_restart_recs = set()
                if logging_console_toggle_changed: final_restart_recs.add("Console Debug Logging")
                if "ACCELERATION_TARGET_DEVICE" in settings_requiring_restart: final_restart_recs.add(
                    "Processing Device")

                if final_restart_recs:
                    info_msg += "\n\nThe following settings require an application restart to take full effect:\n - "
                    info_msg += "\n - ".join(sorted(list(final_restart_recs)))
                    messagebox.showwarning("Restart Recommended", info_msg, parent=self.app)
                elif not theme_related_keys_changed_for_immediate_reload:  # Only non-theme, non-restart settings changed
                    messagebox.showinfo("Settings Saved", info_msg, parent=self.app)
                # If only theme changes, the "Theme/font changes applied" is part of info_msg.

            else:
                if hasattr(self.app, 'log_message'): self.app.log_message(
                    "Settings Tab: ERROR - Failed to save settings. Check logs.")
                messagebox.showerror("Save Error", "Could not save settings to config.py. Check logs for details.",
                                     parent=self.app)

    def create_tooltip(self, widget_to_hover, tooltip_text):
        tooltip_window = None

        def show_tooltip(event):
            nonlocal tooltip_window
            if tooltip_window or not tooltip_text: return
            x_root, y_root = widget_to_hover.winfo_rootx(), widget_to_hover.winfo_rooty();
            x_final = x_root + widget_to_hover.winfo_width() // 2;
            y_final = y_root + widget_to_hover.winfo_height() + 6
            tooltip_window = tk.Toplevel(widget_to_hover);
            tooltip_window.wm_overrideredirect(True)
            try:
                tooltip_window.wm_attributes("-topmost", 1)
            except tk.TclError:
                pass
            screen_width_available = widget_to_hover.winfo_screenwidth();
            label_wraplength = min(350,
                                   screen_width_available - x_final - 20 if x_final + 350 > screen_width_available else 350)
            tooltip_label = ttk.Label(tooltip_window, text=tooltip_text, style=f"{STYLE_PREFIX}Tooltip.TLabel",
                                      wraplength=label_wraplength, justify=tk.LEFT)
            tooltip_label.pack(ipadx=2, ipady=2);
            tooltip_window.update_idletasks();
            tooltip_width_actual = tooltip_window.winfo_width()
            if x_final + tooltip_width_actual > screen_width_available - 10: x_final = screen_width_available - tooltip_width_actual - 10
            tooltip_window.wm_geometry(f"+{int(x_final)}+{int(y_final)}")

        def hide_tooltip(event):
            nonlocal tooltip_window
            if tooltip_window: tooltip_window.destroy(); tooltip_window = None

        widget_to_hover.bind("<Enter>", show_tooltip, add="+");
        widget_to_hover.bind("<Leave>", hide_tooltip, add="+")


def populate_settings_tab(app_instance, parent_notebook_tab_frame_widget):
    for child_widget_item in parent_notebook_tab_frame_widget.winfo_children(): child_widget_item.destroy()
    settings_tab_ui_instance = SettingsTab(parent_notebook_tab_frame_widget, app_instance)
    settings_tab_ui_instance.pack(expand=True, fill="both")
    if hasattr(app_instance, 'logger'): app_instance.logger.info(
        "Settings tab UI populated using settings_gui.populate_settings_tab.")


if __name__ == '__main__':
    pass