# PuffinZipAI_Project/puffinzip_gui/gui_layout_setup.py
import tkinter as tk
from tkinter import ttk, scrolledtext
import logging
import sys

try:
    from . import gui_utils
except ImportError:
    print("WARNING (gui_layout_setup.py): Could not import .gui_utils via relative path. Trying package-level import.")
    try:
        from puffinzip_gui import gui_utils
    except ImportError:
        print("CRITICAL (gui_layout_setup.py): gui_utils.py not found via relative or package-level import. Layout helper functions will be missing.")
        gui_utils = None

if gui_utils and hasattr(gui_utils, '_get_theme_attr'):
    _get_theme_attr_l = gui_utils._get_theme_attr
else:
    def _get_theme_attr_l(app_instance, attr_name, default_value):
        log_to_use = getattr(app_instance, 'logger', logging.getLogger("GuiLayoutSetup_FallbackLogger"))
        if isinstance(log_to_use, logging.Logger) and not log_to_use.handlers:
            _h = logging.StreamHandler(sys.stdout); _f = logging.Formatter('%(asctime)s - LAYOUT_FALLBACK - %(levelname)s - %(message)s'); _h.setFormatter(_f); log_to_use.addHandler(_h); log_to_use.setLevel(logging.INFO)
        if app_instance and hasattr(app_instance, attr_name):
            val = getattr(app_instance, attr_name)
            if val is not None: return val
            if log_to_use: log_to_use.debug(f"Layout Fallback: Theme attribute '{attr_name}' is None, using default.")
        elif app_instance and not hasattr(app_instance, attr_name) and log_to_use:
            log_to_use.debug(f"Layout Fallback: Theme attribute '{attr_name}' not found, using default.")
        return default_value
    print("WARNING (gui_layout_setup.py): Using local fallback for _get_theme_attr_l as gui_utils or its _get_theme_attr method was not found.")

def setup_main_layout(app_instance):
    logger = getattr(app_instance, 'logger', logging.getLogger("GuiLayoutSetupFallbackLogger"))
    if isinstance(logger, logging.Logger) and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - LAYOUT_FALLBACK - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.info("Setting up main GUI layout...")
    app_instance.main_notebook = ttk.Notebook(app_instance, style="TNotebook")
    app_instance.main_notebook.pack(expand=True, fill='both', padx=10, pady=(10, 5))
    app_instance.main_controls_tab = ttk.Frame(app_instance.main_notebook, style="TFrame", padding=0)
    app_instance.evolution_lab_tab = ttk.Frame(app_instance.main_notebook, style="TFrame", padding=(5, 5))
    app_instance.gdv_tab = ttk.Frame(app_instance.main_notebook, style="TFrame", padding=(5, 5))
    app_instance.changelog_tab = ttk.Frame(app_instance.main_notebook, style="TFrame", padding=(5, 5))
    app_instance.settings_content_tab = ttk.Frame(app_instance.main_notebook, style="TFrame", padding=(5, 5))
    app_instance.main_notebook.add(app_instance.main_controls_tab, text='AI Controls')
    app_instance.main_notebook.add(app_instance.evolution_lab_tab, text='Evolution Lab')
    app_instance.main_notebook.add(app_instance.gdv_tab, text='Gen. Deep Dive')
    app_instance.main_notebook.add(app_instance.changelog_tab, text='Change Log')
    app_instance.main_notebook.add(app_instance.settings_content_tab, text='Settings')
    main_controls_paned_window = ttk.PanedWindow(app_instance.main_controls_tab, orient=tk.HORIZONTAL, style="Horizontal.TPanedwindow")
    main_controls_paned_window.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
    left_controls_outer_frame = ttk.Frame(main_controls_paned_window, style="TFrame", width=430)
    left_controls_outer_frame.pack_propagate(False)
    main_controls_paned_window.add(left_controls_outer_frame, weight=0)
    frame_bg_for_canvas = _get_theme_attr_l(app_instance, 'FRAME_BG', "#3C3C3C")
    app_instance.ctrl_canvas = tk.Canvas(left_controls_outer_frame, bg=frame_bg_for_canvas, highlightthickness=0, bd=0)
    app_instance.ctrl_scrollbar = ttk.Scrollbar(left_controls_outer_frame, orient=tk.VERTICAL, command=app_instance.ctrl_canvas.yview, style="Vertical.TScrollbar")
    app_instance.ctrl_canvas.configure(yscrollcommand=app_instance.ctrl_scrollbar.set)
    app_instance.ctrl_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app_instance.ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    app_instance.main_controls_tab_scrollable_frame = ttk.Frame(app_instance.ctrl_canvas, style="Scrollable.TFrame", padding=(15, 15, 20, 15))
    app_instance.scrollable_frame_window_id = app_instance.ctrl_canvas.create_window((0, 0), window=app_instance.main_controls_tab_scrollable_frame, anchor="nw")
    if hasattr(app_instance, 'on_frame_configure') and callable(app_instance.on_frame_configure):
        app_instance.main_controls_tab_scrollable_frame.bind("<Configure>", lambda e: app_instance.on_frame_configure(app_instance.ctrl_canvas, app_instance.main_controls_tab_scrollable_frame, e))
    else:
        logger.error("PMA missing on_frame_configure method! Scrollable area might not resize.")
    if hasattr(app_instance, 'on_canvas_configure') and callable(app_instance.on_canvas_configure):
        app_instance.ctrl_canvas.bind("<Configure>", lambda e: app_instance.on_canvas_configure(e, app_instance.scrollable_frame_window_id, app_instance.ctrl_canvas))
    else:
        logger.error("PMA missing on_canvas_configure method! Scrollable area might not resize correctly.")
    right_log_charts_paned_window = ttk.PanedWindow(main_controls_paned_window, orient=tk.VERTICAL, style="Vertical.TPanedwindow")
    main_controls_paned_window.add(right_log_charts_paned_window, weight=1)
    log_output_frame = ttk.Frame(right_log_charts_paned_window, style="TFrame", height=300)
    right_log_charts_paned_window.add(log_output_frame, weight=1)
    font_mono_l = _get_theme_attr_l(app_instance, 'FONT_MONO', ("Consolas", 10))
    text_area_bg_l = _get_theme_attr_l(app_instance, 'TEXT_AREA_BG', "#1E1E1E")
    text_area_fg_l = _get_theme_attr_l(app_instance, 'TEXT_AREA_FG', "#D0D0D0")
    fg_color_l = _get_theme_attr_l(app_instance, 'FG_COLOR', "#E0E0E0")
    accent_color_l = _get_theme_attr_l(app_instance, 'ACCENT_COLOR', "#0078D4")
    app_instance.output_scrolled_text = scrolledtext.ScrolledText(log_output_frame, wrap=tk.WORD, font=font_mono_l, bg=text_area_bg_l, fg=text_area_fg_l, insertbackground=fg_color_l, selectbackground=accent_color_l, selectforeground=text_area_bg_l, borderwidth=1, relief="solid", padx=12, pady=10, spacing1=3, spacing3=3, undo=True)
    app_instance.output_scrolled_text.pack(expand=True, fill=tk.BOTH)
    app_instance.output_scrolled_text.configure(state='disabled')
    app_instance.charts_area_on_main_tab = ttk.LabelFrame(right_log_charts_paned_window, text="Training Analysis", style="TLabelframe", padding=(10, 5))
    right_log_charts_paned_window.add(app_instance.charts_area_on_main_tab, weight=1)
    analysis_controls_frame = ttk.Frame(app_instance.charts_area_on_main_tab, style="TFrame")
    analysis_controls_frame.pack(pady=(5, 10), padx=5, side="top", fill=tk.X)
    app_instance.refresh_analysis_button = ttk.Button(analysis_controls_frame, text="🔄 Refresh Charts", style="Small.TButton", command=app_instance._manual_refresh_analysis_and_reschedule)
    app_instance.refresh_analysis_button.pack(side="left", anchor="nw", padx=(0, 10))
    if not hasattr(app_instance, 'auto_refresh_analysis_var'):
        app_instance.auto_refresh_analysis_var = tk.BooleanVar(value=False)
    app_instance.auto_refresh_analysis_checkbox = ttk.Checkbutton(analysis_controls_frame, text="Auto-Refresh (5s)", style="TCheckbutton", variable=app_instance.auto_refresh_analysis_var, command=app_instance._on_auto_refresh_toggle)
    app_instance.auto_refresh_analysis_checkbox.pack(side="left", anchor="nw")
    analysis_charts_paned_window = ttk.PanedWindow(app_instance.charts_area_on_main_tab, orient=tk.HORIZONTAL, style="Horizontal.TPanedwindow")
    analysis_charts_paned_window.pack(expand=True, fill="both")
    app_instance.rewards_chart_area = ttk.LabelFrame(analysis_charts_paned_window, text="Training Rewards", style="TLabelframe")
    analysis_charts_paned_window.add(app_instance.rewards_chart_area, weight=1)
    app_instance.actions_chart_area = ttk.LabelFrame(analysis_charts_paned_window, text="Action Distribution", style="TLabelframe")
    analysis_charts_paned_window.add(app_instance.actions_chart_area, weight=1)
    chart_utils_module = getattr(app_instance, 'chart_utils', None)
    if chart_utils_module and hasattr(chart_utils_module, 'display_placeholder_message'):
        chart_utils_module.display_placeholder_message(app_instance.rewards_chart_area, "Rewards Chart - Awaiting Data", app_instance=app_instance)
        chart_utils_module.display_placeholder_message(app_instance.actions_chart_area, "Action Distribution - Awaiting Data", app_instance=app_instance)
    else:
        logger.warning("chart_utils or its display_placeholder_message not available during layout setup.")
        if hasattr(app_instance.rewards_chart_area, 'winfo_exists') and app_instance.rewards_chart_area.winfo_exists():
            ttk.Label(app_instance.rewards_chart_area, text="Charting utilities error.", style="Error.TLabel", anchor="center").pack(expand=True, fill='both')
        if hasattr(app_instance.actions_chart_area, 'winfo_exists') and app_instance.actions_chart_area.winfo_exists():
            ttk.Label(app_instance.actions_chart_area, text="Charting utilities error.", style="Error.TLabel", anchor="center").pack(expand=True, fill='both')
    logger.info("Main GUI layout setup complete.")

if __name__ == '__main__':
    print("gui_layout_setup.py contains the setup_main_layout function.")
    print("Run the main PuffinZipApp to test its application.")