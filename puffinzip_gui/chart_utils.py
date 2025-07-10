# PuffinZipAI_Project/puffinzip_gui/chart_utils.py
import tkinter as tk
from tkinter import ttk
import traceback
import numpy as np
import random
import logging

chart_utils_logger = logging.getLogger("puffinzip_gui.chart_utils")
if not chart_utils_logger.handlers:
    chart_utils_logger.addHandler(logging.NullHandler())

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    chart_utils_logger.info("Matplotlib not found. Charting will use dummy fallback classes.")
    class MockSpine:
        def set_color(self, color): pass
        def set_linewidth(self, width): pass
        def set_visible(self, b): pass
    class MockBar:
        def __init__(self, x, height, width=0.8): pass
        def get_height(self): return 0
        def get_x(self): return 0
        def get_width(self): return 0.8
        def set_height(self, h): pass
        def set_x(self, x_pos): pass
        def set_width(self, w): pass
        def set_color(self, c): pass
        def get_facecolor(self): return "gray"
    class MockLegend:
        def get_texts(self): return []
        def remove(self): pass
        def set_visible(self, v): pass
        def get_frame(self): return MockSpine()
    class MockAx:
        def __init__(self, figure): self.figure = figure; self.spines = {'top': MockSpine(), 'right': MockSpine(),
                                                                         'bottom': MockSpine(), 'left': MockSpine()}
        def plot(self, *args, **kwargs): return []
        def bar(self, *args, **kwargs): return [MockBar(0, 0)]
        def text(self, x, y, s, **kwargs): pass
        def set_xlabel(self, label, **kwargs): pass
        def set_ylabel(self, label, **kwargs): pass
        def set_title(self, label, **kwargs): pass
        def set_facecolor(self, color, **kwargs): pass
        def tick_params(self, **kwargs): pass
        def grid(self, b=None, **kwargs): pass
        def set_xticks(self, ticks, labels=None, **kwargs): pass
        def set_yticks(self, ticks, labels=None, **kwargs): pass
        def set_xticklabels(self, labels, **kwargs): pass
        def set_yticklabels(self, labels, **kwargs): pass
        def set_ylim(self, bottom=None, top=None, **kwargs): pass
        def set_xlim(self, left=None, right=None, **kwargs): pass
        def legend(self, *args, **kwargs): return MockLegend()
        def clear(self): pass
        def xaxis_date(self, tz=None): pass
        def yaxis_date(self, tz=None): pass
        def get_legend(self): return None
        def twinx(self): return self
        def get_figure(self): return self.figure
        def add_artist(self, artist): pass
    class Figure:
        def __init__(self, figsize=None, dpi=None, facecolor=None, **kwargs): self.axes = []
        def clear(self): self.axes.clear()
        def add_subplot(self, *args, **kwargs): ax = MockAx(self); self.axes.append(ax); return ax
        def tight_layout(self, pad=None, **kwargs): pass
        def suptitle(self, t, **kwargs): pass
        def subplots(self, nrows=1, ncols=1, **kwargs): return self, MockAx(self)
    class FigureCanvasTkAgg:
        def __init__(self, figure, master=None):
            self.figure = figure;
            self.master = master;
            self._widget = None
        def get_tk_widget(self):
            if self.master and hasattr(self.master, 'winfo_exists') and self.master.winfo_exists():
                if not (self._widget and hasattr(self._widget, 'winfo_exists') and self._widget.winfo_exists()):
                    self._widget = ttk.Label(self.master,
                                             text="Matplotlib not installed (ImportError).\nCharts disabled.",
                                             anchor="center", justify=tk.CENTER)
                return self._widget
            return None
        def draw_idle(self): pass
        def draw(self): pass
        def flush_events(self): pass
        def mpl_connect(self, s, func): return 0
except Exception as e_general_mpl:
    chart_utils_logger.critical(f"Unexpected error during Matplotlib setup or primary fallback: {e_general_mpl}. MATPLOTLIB_AVAILABLE set to False.", exc_info=True)
    MATPLOTLIB_AVAILABLE = False
    if 'MockSpine' not in globals():
        class MockSpine:
            def set_color(self, color): pass
            def set_linewidth(self, width): pass
            def set_visible(self, b): pass
    if 'MockBar' not in globals():
        class MockBar:
            def __init__(self, x, height, width=0.8): pass
            def get_height(self): return 0
            def get_x(self): return 0
            def get_width(self): return 0.8
            def set_height(self, h): pass
            def set_x(self, x_pos): pass
            def set_width(self, w): pass
            def set_color(self, c): pass
            def get_facecolor(self): return "gray"
    if 'MockLegend' not in globals():
        class MockLegend:
            def get_texts(self): return []
            def remove(self): pass
            def set_visible(self, v): pass
            def get_frame(self): return MockSpine()
    if 'MockAx' not in globals():
        class MockAx:
            def __init__(self, figure): self.figure = figure; self.spines = {'top': MockSpine(), 'right': MockSpine(),
                                                                             'bottom': MockSpine(), 'left': MockSpine()}
            def plot(self, *args, **kwargs): return []
            def bar(self, *args, **kwargs): return [MockBar(0, 0)]
            def text(self, *args, **kwargs): pass
            def set_xlabel(self, *args, **kwargs): pass
            def set_ylabel(self, *args, **kwargs): pass
            def set_title(self, *args, **kwargs): pass
            def set_facecolor(self, *args, **kwargs): pass
            def tick_params(self, *args, **kwargs): pass
            def grid(self, *args, **kwargs): pass
            def set_xticks(self, *args, **kwargs): pass
            def set_yticks(self, *args, **kwargs): pass
            def set_xticklabels(self, *args, **kwargs): pass
            def set_yticklabels(self, *args, **kwargs): pass
            def set_ylim(self, *args, **kwargs): pass
            def set_xlim(self, *args, **kwargs): pass
            def legend(self, *args, **kwargs): return MockLegend()
            def clear(self): pass
            def twinx(self): return self
            def get_figure(self): return self.figure
            def add_artist(self, artist): pass
    if 'Figure' not in globals():
        class Figure:
            def __init__(self, figsize=None, dpi=None, facecolor=None, **kwargs): self.axes = []
            def clear(self): self.axes.clear()
            def add_subplot(self, *args, **kwargs):
                ax = MockAx(self); self.axes.append(ax); return ax
            def tight_layout(self, pad=None, **kwargs): pass
            def suptitle(self, t, **kwargs): pass
            def subplots(self, nrows=1, ncols=1, **kwargs):
                return self, MockAx(self)
    if 'FigureCanvasTkAgg' not in globals():
        class FigureCanvasTkAgg:
            def __init__(self, figure, master=None):
                self.figure = figure; self.master = master; self._widget = None
            def get_tk_widget(self):
                if self.master and hasattr(self.master, 'winfo_exists') and self.master.winfo_exists():
                    if not (self._widget and hasattr(self._widget, 'winfo_exists') and self._widget.winfo_exists()):
                        self._widget = ttk.Label(self.master,
                                                 text="Matplotlib General Error (Fallback).\nCharts disabled.",
                                                 anchor="center", justify=tk.CENTER)
                    return self._widget
                return None
            def draw(self): pass
            def draw_idle(self): pass
            def flush_events(self): pass
            def mpl_connect(self, s, f): return 0

CHART_BACKGROUND_COLOR_DEFAULT = "#252526"
CHART_FIGURE_FACECOLOR_DEFAULT = "#3C3C3C"
CHART_TEXT_COLOR_DEFAULT = "#E0E0E0"
CHART_TICK_COLOR_DEFAULT = "#C0C0C0"
CHART_SPINE_COLOR_DEFAULT = "#6A6A6A"
CHART_GRID_COLOR_DEFAULT = "#4A4A4A"
PLOT_LINE_COLOR_BEST_DEFAULT = "#00AEEF"
PLOT_LINE_COLOR_AVG_DEFAULT = "#FF8C00"
PLOT_LINE_COLOR_WORST_DEFAULT = "#E53935"
PLOT_LINE_COLOR_MEDIAN_DEFAULT = "#7CB342"
PLOT_BAR_COLOR_RLE_DEFAULT = "#007ACC"
PLOT_BAR_COLOR_NOCOMP_DEFAULT = "#5A5A5A"
PLOT_BAR_COLOR_ADVRLE_DEFAULT = "#4CAF50"
PLOT_BAR_TEXT_COLOR_DEFAULT = CHART_TEXT_COLOR_DEFAULT
FRAME_BG_FALLBACK_DEFAULT = "#333333"
ERROR_FG_COLOR_DEFAULT = "#FF6B6B"

FONT_FAMILY_DEFAULT = "Segoe UI"
FONT_SIZE_TITLE_DEFAULT = 12
FONT_SIZE_LABEL_DEFAULT = 10
FONT_SIZE_TICKS_DEFAULT = 9
FONT_SIZE_LEGEND_DEFAULT = 9

FONT_NOTE_DEFAULT = (FONT_FAMILY_DEFAULT, FONT_SIZE_TICKS_DEFAULT, "italic")
FONT_TITLE_DEFAULT = (FONT_FAMILY_DEFAULT, FONT_SIZE_TITLE_DEFAULT, "bold")
FONT_LABEL_DEFAULT = (FONT_FAMILY_DEFAULT, FONT_SIZE_LABEL_DEFAULT)
FONT_TICKS_DEFAULT = (FONT_FAMILY_DEFAULT, FONT_SIZE_TICKS_DEFAULT)
FONT_LEGEND_DEFAULT = (FONT_FAMILY_DEFAULT, FONT_SIZE_LEGEND_DEFAULT)

def _get_theme_attr(app_instance, attr_name, default_value):
    if app_instance and hasattr(app_instance, attr_name):
        val = getattr(app_instance, attr_name)
        if val is not None:
            return val
    return default_value

def clear_frame_widgets(frame: tk.Frame):
    if frame and hasattr(frame, 'winfo_exists') and frame.winfo_exists():
        for widget in frame.winfo_children():
            try:
                widget.destroy()
            except tk.TclError:
                pass

def display_placeholder_message(parent_frame: tk.Frame, message: str = "Chart Area - Awaiting Data", app_instance=None):
    clear_frame_widgets(parent_frame)
    font_to_use = _get_theme_attr(app_instance, 'FONT_NOTE', FONT_NOTE_DEFAULT)
    fg_color = _get_theme_attr(app_instance, 'DISABLED_FG_COLOR', CHART_TEXT_COLOR_DEFAULT)
    bg_color = _get_theme_attr(app_instance, 'FRAME_BG', FRAME_BG_FALLBACK_DEFAULT)
    label_style_name = "Placeholder.TLabel"
    style_name_to_use = "TLabel"
    try:
        s = ttk.Style(parent_frame)
        s.configure(label_style_name, background=bg_color, foreground=fg_color, font=font_to_use, anchor="center")
        style_name_to_use = label_style_name
    except tk.TclError:
        logger_to_use = getattr(app_instance, 'logger', None)
        if logger_to_use:
             logger_to_use.warning("display_placeholder_message: Could not configure Placeholder.TLabel. Using default TLabel.")
        else:
            print("WARNING: display_placeholder_message: Could not configure Placeholder.TLabel. Using default TLabel.")
    placeholder_label = ttk.Label(parent_frame, text=message, style=style_name_to_use, justify=tk.CENTER)
    placeholder_label.pack(expand=True, fill="both", padx=10, pady=10)

def plot_training_rewards(parent_frame: tk.Frame, rewards_history: list, title: str = "Average Reward per Batch",
                          existing_canvas_agg=None, existing_figure=None, app_instance=None):
    fig_face_color = _get_theme_attr(app_instance, 'FRAME_BG', CHART_FIGURE_FACECOLOR_DEFAULT)
    ax_face_color = _get_theme_attr(app_instance, 'INPUT_BG', CHART_BACKGROUND_COLOR_DEFAULT)
    text_color = _get_theme_attr(app_instance, 'FG_COLOR', CHART_TEXT_COLOR_DEFAULT)
    grid_color = _get_theme_attr(app_instance, 'TAB_BORDER_COLOR', CHART_GRID_COLOR_DEFAULT)
    spine_color = _get_theme_attr(app_instance, 'DISABLED_FG_COLOR', CHART_SPINE_COLOR_DEFAULT)
    tick_color = _get_theme_attr(app_instance, 'LABEL_FG', CHART_TICK_COLOR_DEFAULT)
    line_color = _get_theme_attr(app_instance, 'ACCENT_COLOR', PLOT_LINE_COLOR_BEST_DEFAULT)
    font_title_tuple = _get_theme_attr(app_instance, 'FONT_SECTION_TITLE', FONT_TITLE_DEFAULT)
    font_label_tuple = _get_theme_attr(app_instance, 'FONT_NORMAL', FONT_LABEL_DEFAULT)
    font_ticks_tuple = _get_theme_attr(app_instance, 'FONT_SMALL', FONT_TICKS_DEFAULT)
    font_title_mpl = {'family': font_title_tuple[0], 'size': font_title_tuple[1],
                      'weight': font_title_tuple[2] if len(font_title_tuple) > 2 else 'bold'}
    font_label_mpl = {'family': font_label_tuple[0], 'size': font_label_tuple[1]}
    font_ticks_mpl_size = font_ticks_tuple[1]
    fig = existing_figure; canvas = existing_canvas_agg; ax = None
    if not MATPLOTLIB_AVAILABLE:
        if canvas and canvas.get_tk_widget() and canvas.get_tk_widget().winfo_exists(): canvas.get_tk_widget().pack_forget()
        display_placeholder_message(parent_frame, "Matplotlib library not found.\nRewards chart cannot be plotted.", app_instance)
        return None, None
    if not (parent_frame and parent_frame.winfo_exists()): return None, None
    if canvas and fig:
        fig.clear(); ax = fig.add_subplot(111)
    else:
        clear_frame_widgets(parent_frame)
        fig = Figure(figsize=(6, 4), dpi=100, facecolor=fig_face_color)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        if canvas.get_tk_widget(): canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=1, pady=1)
    if not ax:
        logger_to_use = getattr(app_instance, 'logger', None)
        if logger_to_use: logger_to_use.error("plot_training_rewards: ax is None after setup attempt.")
        return canvas, fig
    if not rewards_history:
        ax.text(0.5, 0.5, "No reward data available.", ha='center', va='center', color=text_color, fontdict=font_label_mpl)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.plot(rewards_history, marker='.', linestyle='-', color=line_color, markersize=6, linewidth=1.5, alpha=0.9)
        ax.set_xlabel("Batch Number", color=text_color, fontdict=font_label_mpl)
        ax.set_ylabel("Average Reward", color=text_color, fontdict=font_label_mpl)
        ax.tick_params(axis='x', colors=tick_color, labelsize=font_ticks_mpl_size); ax.tick_params(axis='y', colors=tick_color, labelsize=font_ticks_mpl_size)
        ax.grid(True, linestyle=':', alpha=0.5, color=grid_color)
    ax.set_title(title, color=text_color, fontdict=font_title_mpl)
    ax.set_facecolor(ax_face_color)
    for spine_obj in ax.spines.values(): spine_obj.set_color(spine_color); spine_obj.set_linewidth(0.8)
    try:
        fig.tight_layout(pad=1.5)
    except Exception: pass
    if canvas: canvas.draw_idle()
    return canvas, fig

def plot_action_distribution(parent_frame: tk.Frame, action_counts: dict, title: str = "Action Distribution",
                             existing_canvas_agg=None, existing_figure=None, app_instance=None):
    fig_face_color = _get_theme_attr(app_instance, 'FRAME_BG', CHART_FIGURE_FACECOLOR_DEFAULT)
    ax_face_color = _get_theme_attr(app_instance, 'INPUT_BG', CHART_BACKGROUND_COLOR_DEFAULT)
    text_color = _get_theme_attr(app_instance, 'FG_COLOR', CHART_TEXT_COLOR_DEFAULT)
    grid_color = _get_theme_attr(app_instance, 'TAB_BORDER_COLOR', CHART_GRID_COLOR_DEFAULT)
    spine_color = _get_theme_attr(app_instance, 'DISABLED_FG_COLOR', CHART_SPINE_COLOR_DEFAULT)
    tick_color = _get_theme_attr(app_instance, 'LABEL_FG', CHART_TICK_COLOR_DEFAULT)
    bar_color_rle = _get_theme_attr(app_instance, 'ACCENT_COLOR', PLOT_BAR_COLOR_RLE_DEFAULT)
    bar_color_nocomp = _get_theme_attr(app_instance, 'BUTTON_BG', PLOT_BAR_COLOR_NOCOMP_DEFAULT)
    bar_color_advrle = _get_theme_attr(app_instance, 'ADV_RLE_BAR_COLOR', PLOT_BAR_COLOR_ADVRLE_DEFAULT)
    bar_text_color_on_bar = text_color
    font_title_tuple = _get_theme_attr(app_instance, 'FONT_SECTION_TITLE', FONT_TITLE_DEFAULT)
    font_label_tuple = _get_theme_attr(app_instance, 'FONT_NORMAL', FONT_LABEL_DEFAULT)
    font_ticks_tuple = _get_theme_attr(app_instance, 'FONT_SMALL', FONT_TICKS_DEFAULT)
    font_title_mpl = {'family': font_title_tuple[0], 'size': font_title_tuple[1],
                      'weight': font_title_tuple[2] if len(font_title_tuple) > 2 else 'bold'}
    font_label_mpl = {'family': font_label_tuple[0], 'size': font_label_tuple[1]}
    font_ticks_mpl_size = font_ticks_tuple[1]
    fig = existing_figure; canvas = existing_canvas_agg; ax = None
    if not MATPLOTLIB_AVAILABLE:
        if canvas and canvas.get_tk_widget() and canvas.get_tk_widget().winfo_exists(): canvas.get_tk_widget().pack_forget()
        display_placeholder_message(parent_frame, "Matplotlib library not found.\nAction distribution chart cannot be plotted.", app_instance)
        return None, None
    if not (parent_frame and parent_frame.winfo_exists()): return None, None
    if canvas and fig:
        fig.clear(); ax = fig.add_subplot(111)
    else:
        clear_frame_widgets(parent_frame)
        fig = Figure(figsize=(5.5, 4.5), dpi=100, facecolor=fig_face_color)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        if canvas.get_tk_widget(): canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=1, pady=1)
    if not ax:
        logger_to_use = getattr(app_instance, 'logger', None)
        if logger_to_use: logger_to_use.error("plot_action_distribution: ax is None after setup.")
        return canvas, fig
    if not action_counts or not any(v for v in action_counts.values() if isinstance(v, (int, float)) and v > 0):
        ax.text(0.5, 0.5, "No action data for distribution.", ha='center', va='center', color=text_color, fontdict=font_label_mpl)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        labels = list(action_counts.keys()); counts = list(action_counts.values())
        color_map = []; default_other_action_bar_color = _get_theme_attr(app_instance, 'INPUT_BG', CHART_BACKGROUND_COLOR_DEFAULT)
        for lbl_action in labels:
            lbl_action_upper = lbl_action.upper()
            if lbl_action_upper == "RLE": color_map.append(bar_color_rle)
            elif lbl_action_upper == "NOCOMP": color_map.append(bar_color_nocomp)
            elif lbl_action_upper == "ADVRLE": color_map.append(bar_color_advrle)
            else: color_map.append(default_other_action_bar_color)
        bars = ax.bar(labels, counts, color=color_map, width=0.5)
        for bar_widget in bars:
            yval = bar_widget.get_height()
            if yval > 0:
                ax.text(bar_widget.get_x() + bar_widget.get_width() / 2.0,
                        yval + (max(counts) * 0.018 if counts and max(counts) > 0 else 0.1),
                        f'{int(yval)}', ha='center', va='bottom', color=bar_text_color_on_bar,
                        fontfamily=font_ticks_tuple[0], fontsize=font_ticks_mpl_size - 1)
        ax.set_ylabel("Times Chosen", color=text_color, fontdict=font_label_mpl)
        ax.tick_params(axis='x', colors=tick_color, labelsize=font_ticks_mpl_size, rotation=0); ax.tick_params(axis='y', colors=tick_color, labelsize=font_ticks_mpl_size)
        ax.grid(True, axis='y', linestyle=':', alpha=0.4, color=grid_color)
        max_count_val = max(counts) if counts and any(c > 0 for c in counts) else 10
        ax.set_ylim(top=max_count_val * 1.22 if max_count_val > 0 else 10)
    ax.set_title(title, color=text_color, fontdict=font_title_mpl)
    ax.set_facecolor(ax_face_color)
    for spine_obj in ax.spines.values(): spine_obj.set_color(spine_color); spine_obj.set_linewidth(0.8)
    try:
        fig.tight_layout(pad=1.8)
    except Exception: pass
    if canvas: canvas.draw_idle()
    return canvas, fig

def plot_evolution_fitness(parent_frame: tk.Frame,
                           fitness_history_data: list,
                           title: str = "Evolution Fitness",
                           existing_canvas_agg=None, existing_figure=None, app_instance=None,
                           show_best: bool = True, show_avg: bool = True,
                           show_worst: bool = False, show_median: bool = False):
    fig_face_color = _get_theme_attr(app_instance, 'FRAME_BG', CHART_FIGURE_FACECOLOR_DEFAULT)
    ax_face_color = _get_theme_attr(app_instance, 'INPUT_BG', CHART_BACKGROUND_COLOR_DEFAULT)
    text_color = _get_theme_attr(app_instance, 'FG_COLOR', CHART_TEXT_COLOR_DEFAULT)
    grid_color = _get_theme_attr(app_instance, 'TAB_BORDER_COLOR', CHART_GRID_COLOR_DEFAULT)
    spine_color = _get_theme_attr(app_instance, 'DISABLED_FG_COLOR', CHART_SPINE_COLOR_DEFAULT)
    tick_color = _get_theme_attr(app_instance, 'LABEL_FG', CHART_TICK_COLOR_DEFAULT)
    best_line_color_theme = _get_theme_attr(app_instance, 'ACCENT_COLOR', PLOT_LINE_COLOR_BEST_DEFAULT)
    avg_line_color_theme = _get_theme_attr(app_instance, 'SCROLLBAR_BG', PLOT_LINE_COLOR_AVG_DEFAULT)
    worst_line_color_theme = _get_theme_attr(app_instance, 'ERROR_FG_COLOR', PLOT_LINE_COLOR_WORST_DEFAULT)
    median_line_color_theme = _get_theme_attr(app_instance, 'ADV_RLE_BAR_COLOR', PLOT_LINE_COLOR_MEDIAN_DEFAULT)
    font_title_tuple = _get_theme_attr(app_instance, 'FONT_SECTION_TITLE', FONT_TITLE_DEFAULT)
    font_label_tuple = _get_theme_attr(app_instance, 'FONT_NORMAL', FONT_LABEL_DEFAULT)
    font_ticks_tuple = _get_theme_attr(app_instance, 'FONT_SMALL', FONT_TICKS_DEFAULT)
    font_legend_tuple = _get_theme_attr(app_instance, 'FONT_SMALL', FONT_LEGEND_DEFAULT)
    font_title_mpl = {'family': font_title_tuple[0], 'size': font_title_tuple[1],
                      'weight': font_title_tuple[2] if len(font_title_tuple) > 2 else 'bold'}
    font_label_mpl = {'family': font_label_tuple[0], 'size': font_label_tuple[1]}
    font_ticks_mpl_size = font_ticks_tuple[1]
    font_legend_mpl_dict = {'family': font_legend_tuple[0], 'size': font_legend_tuple[1]}
    fig = existing_figure; canvas = existing_canvas_agg; ax = None
    if not MATPLOTLIB_AVAILABLE:
        if canvas and canvas.get_tk_widget() and canvas.get_tk_widget().winfo_exists(): canvas.get_tk_widget().pack_forget()
        display_placeholder_message(parent_frame, "Matplotlib library not found.\nEvolutionary fitness chart cannot be plotted.", app_instance)
        return None, None
    if not (parent_frame and parent_frame.winfo_exists()):
        logger_to_use = getattr(app_instance, 'logger', None)
        if logger_to_use: logger_to_use.error("plot_evolution_fitness: Parent frame is invalid or destroyed.")
        else: print("ERROR (chart_utils): ELS Chart parent frame invalid.")
        return None, None
    if canvas and fig:
        try:
            fig.clear(); ax = fig.add_subplot(111)
        except Exception as e_clear_fig:
            logger_to_use = getattr(app_instance, 'logger', None)
            if logger_to_use: logger_to_use.error(f"Error clearing/re-adding subplot to existing ELS figure: {e_clear_fig}")
            if hasattr(canvas,'get_tk_widget') and canvas.get_tk_widget() and canvas.get_tk_widget().winfo_exists():
                canvas.get_tk_widget().destroy()
            canvas, fig = None, None; clear_frame_widgets(parent_frame)
    if not (canvas and fig):
        clear_frame_widgets(parent_frame)
        fig = Figure(figsize=(7.5, 5.5), dpi=100, facecolor=fig_face_color)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        if hasattr(canvas, 'get_tk_widget') and canvas.get_tk_widget(): canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=1, pady=1)
    if not ax:
        logger_to_use = getattr(app_instance, 'logger', None)
        if logger_to_use: logger_to_use.error("plot_evolution_fitness: Axes object (ax) is None. Cannot plot ELS fitness.")
        return canvas, fig
    if not fitness_history_data:
        ax.text(0.5, 0.5, "No ELS fitness data available to plot.", ha='center', va='center', color=text_color, fontdict=font_label_mpl)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        try:
            generations = [item[0] + 1 for item in fitness_history_data]
            series_to_plot = []
            if show_best:
                best_fitness_per_gen = [item[1] for item in fitness_history_data]
                ax.plot(generations, best_fitness_per_gen, marker='o', linestyle='-', color=best_line_color_theme, markersize=5, linewidth=2, label="Best Fitness", alpha=0.9)
                series_to_plot.extend(best_fitness_per_gen)
            if show_avg:
                avg_fitness_per_gen = [item[2] for item in fitness_history_data]
                ax.plot(generations, avg_fitness_per_gen, marker='x', linestyle='--', color=avg_line_color_theme, markersize=5, linewidth=1.5, alpha=0.7, label="Average Fitness")
                series_to_plot.extend(avg_fitness_per_gen)
            if show_worst:
                worst_fitness_per_gen = [item[3] for item in fitness_history_data]
                ax.plot(generations, worst_fitness_per_gen, marker='^', linestyle=':', color=worst_line_color_theme, markersize=4, linewidth=1.2, label="Worst Fitness", alpha=0.6)
                series_to_plot.extend(worst_fitness_per_gen)
            if show_median:
                median_fitness_per_gen = [item[4] for item in fitness_history_data]
                ax.plot(generations, median_fitness_per_gen, marker='s', linestyle='-.', color=median_line_color_theme, markersize=4, linewidth=1.5, label="Median Fitness", alpha=0.65)
                series_to_plot.extend(median_fitness_per_gen)
            ax.set_xlabel("Generation", color=text_color, fontdict=font_label_mpl)
            ax.set_ylabel("Fitness Score", color=text_color, fontdict=font_label_mpl)
            ax.tick_params(axis='x', colors=tick_color, labelsize=font_ticks_mpl_size); ax.tick_params(axis='y', colors=tick_color, labelsize=font_ticks_mpl_size)
            all_finite_fitness_values = [f for f in series_to_plot if isinstance(f, (int, float)) and np.isfinite(f)]
            if all_finite_fitness_values:
                min_fit, max_fit = min(all_finite_fitness_values), max(all_finite_fitness_values)
                padding_ratio = 0.10; fit_range = max_fit - min_fit
                if abs(fit_range) < 1e-6: fit_range = abs(max_fit) * 0.2 if abs(max_fit) > 1e-6 else 0.2
                padding = fit_range * padding_ratio; y_bottom, y_top = min_fit - padding, max_fit + padding
                if abs(y_top - y_bottom) < 1e-6: y_top += 0.5; y_bottom -= 0.5
                ax.set_ylim(bottom=y_bottom, top=y_top)
            else:
                ax.set_ylim(-1, 1)
            if generations: ax.set_xlim(left=min(generations) - 0.5 if len(generations) > 1 else 0.5, right=max(generations) + 0.5)
            ax.grid(True, linestyle=':', alpha=0.4, color=grid_color)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(handles, labels, facecolor=_get_theme_attr(app_instance, 'INPUT_BG', CHART_BACKGROUND_COLOR_DEFAULT), edgecolor=spine_color, prop=font_legend_mpl_dict)
                if legend:
                    for text_obj in legend.get_texts(): text_obj.set_color(text_color)
            elif ax.get_legend() is not None:
                 ax.get_legend().remove()
        except Exception as e_plotting_data_els:
            err_msg_plot = f"Error plotting ELS data:\n{str(e_plotting_data_els)[:100]}..."
            logger_to_use = getattr(app_instance, 'logger', None)
            if logger_to_use: logger_to_use.error(f"Error rendering ELS fitness data on chart: {e_plotting_data_els}", exc_info=True)
            ax.text(0.5, 0.5, err_msg_plot, ha='center', va='center', color=_get_theme_attr(app_instance, 'ERROR_FG_COLOR', ERROR_FG_COLOR_DEFAULT), fontfamily=font_ticks_tuple[0], fontsize=font_ticks_mpl_size - 1, wrap=True)
    ax.set_title(title, color=text_color, fontdict=font_title_mpl)
    ax.set_facecolor(ax_face_color)
    for spine_obj in ax.spines.values(): spine_obj.set_color(spine_color); spine_obj.set_linewidth(0.8)
    try:
        fig.tight_layout(pad=2.0)
    except Exception: pass
    if canvas: canvas.draw_idle()
    return canvas, fig

if __name__ == '__main__':
    print(f"\n--- CHART_UTILS.PY STANDALONE TEST --- Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    root = tk.Tk(); root.title("Chart Utils Standalone Test"); root.geometry("1200x800")
    root.FRAME_BG = FRAME_BG_FALLBACK_DEFAULT; root.FG_COLOR = CHART_TEXT_COLOR_DEFAULT; root.ACCENT_COLOR = PLOT_LINE_COLOR_BEST_DEFAULT
    root.FONT_SECTION_TITLE = FONT_TITLE_DEFAULT; root.FONT_NORMAL = FONT_LABEL_DEFAULT; root.FONT_SMALL = FONT_TICKS_DEFAULT
    root.FONT_NOTE = ('Segoe UI', 9, 'italic'); root.INPUT_BG = CHART_BACKGROUND_COLOR_DEFAULT; root.TAB_BORDER_COLOR = CHART_GRID_COLOR_DEFAULT
    root.DISABLED_FG_COLOR = CHART_SPINE_COLOR_DEFAULT; root.LABEL_FG = CHART_TICK_COLOR_DEFAULT; root.BUTTON_BG = PLOT_BAR_COLOR_NOCOMP_DEFAULT
    root.SCROLLBAR_BG = PLOT_LINE_COLOR_AVG_DEFAULT; root.ERROR_FG_COLOR = ERROR_FG_COLOR_DEFAULT; root.ADV_RLE_BAR_COLOR = PLOT_BAR_COLOR_ADVRLE_DEFAULT
    try:
        root.style = ttk.Style(root)
    except tk.TclError: root.style = None
    root.configure(bg=root.FRAME_BG)
    main_frame = ttk.Frame(root, padding=10); main_frame.pack(expand=True, fill="both")
    if hasattr(root, 'style') and root.style: root.style.configure("TFrame", background=root.FRAME_BG)
    ttk.Label(main_frame, text="Chart Utilities Test Viewer", font=root.FONT_SECTION_TITLE, background=root.FRAME_BG, foreground=root.ACCENT_COLOR).pack(pady=(0, 15))
    charts_container_top = ttk.Frame(main_frame); charts_container_top.pack(expand=True, fill="both")
    charts_container_bottom = ttk.Frame(main_frame); charts_container_bottom.pack(expand=True, fill="both", pady=(10, 0))
    rewards_outer_frame = ttk.LabelFrame(charts_container_top, text="Rewards Plot Test", padding=10); rewards_outer_frame.pack(side=tk.LEFT, expand=True, fill="both", padx=5, pady=5)
    rewards_plot_area = ttk.Frame(rewards_outer_frame); rewards_plot_area.pack(expand=True, fill="both")
    dummy_rewards_data = [random.uniform(0.2, 0.8) + (i * 0.005) + (random.random() - 0.5) * 0.05 for i in range(150)]
    plot_training_rewards(rewards_plot_area, dummy_rewards_data, title="Sample Rewards Plot", app_instance=root)
    actions_outer_frame = ttk.LabelFrame(charts_container_top, text="Action Distribution Test", padding=10); actions_outer_frame.pack(side=tk.LEFT, expand=True, fill="both", padx=5, pady=5)
    actions_plot_area = ttk.Frame(actions_outer_frame); actions_plot_area.pack(expand=True, fill="both")
    dummy_actions_data = {'RLE': random.randint(200, 800), 'NoComp': random.randint(200, 800), 'AdvRLE': random.randint(50, 300)}
    plot_action_distribution(actions_plot_area, dummy_actions_data, title="Sample Action Distribution", app_instance=root)
    els_outer_frame = ttk.LabelFrame(charts_container_bottom, text="ELS Fitness Plot Test (Live Update)", padding=10); els_outer_frame.pack(expand=True, fill="both", padx=5, pady=5)
    els_plot_area = ttk.Frame(els_outer_frame); els_plot_area.pack(expand=True, fill="both")
    main_els_chart_canvas_test = None; main_els_chart_figure_test = None; main_global_dummy_els_data_test = []
    root.els_plot_show_best_var = tk.BooleanVar(value=True); root.els_plot_show_avg_var = tk.BooleanVar(value=True); root.els_plot_show_worst_var = tk.BooleanVar(value=True); root.els_plot_show_median_var = tk.BooleanVar(value=False)
    def update_els_test_chart_standalone_scoped():
        global main_els_chart_canvas_test, main_els_chart_figure_test
        data_src = main_global_dummy_els_data_test; new_gen_idx = len(data_src)
        current_best = (data_src[-1][1] if data_src else 0.5) + (random.random() - 0.4) * 0.1
        current_avg = current_best * random.uniform(0.7, 0.95); current_median = current_avg * random.uniform(0.9, 1.05); current_worst = current_avg * random.uniform(0.6, 0.8)
        data_src.append((new_gen_idx, current_best, current_avg, current_worst, current_median))
        if len(data_src) > 20: data_src.pop(0)
        temp_canvas, temp_figure = plot_evolution_fitness(els_plot_area, data_src, title="Sample ELS Fitness (Live Update Test)", existing_canvas_agg=main_els_chart_canvas_test, existing_figure=main_els_chart_figure_test, app_instance=root, show_best=root.els_plot_show_best_var.get(), show_avg=root.els_plot_show_avg_var.get(), show_worst=root.els_plot_show_worst_var.get(), show_median=root.els_plot_show_median_var.get())
        main_els_chart_canvas_test, main_els_chart_figure_test = temp_canvas, temp_figure
        if root.winfo_exists(): root.after(2000, update_els_test_chart_standalone_scoped)
    initial_canvas, initial_figure = plot_evolution_fitness(els_plot_area, [], title="Sample ELS Fitness (Live Update Test)", app_instance=root, show_best=root.els_plot_show_best_var.get(), show_avg=root.els_plot_show_avg_var.get(), show_worst=root.els_plot_show_worst_var.get(), show_median=root.els_plot_show_median_var.get())
    main_els_chart_canvas_test, main_els_chart_figure_test = initial_canvas, initial_figure
    if main_els_chart_canvas_test: root.after(1000, update_els_test_chart_standalone_scoped)
    root.mainloop()
    print(f"\nchart_utils.py standalone test execution finished. Matplotlib available: {MATPLOTLIB_AVAILABLE}")