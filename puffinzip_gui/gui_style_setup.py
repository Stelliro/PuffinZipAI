# PuffinZipAI_Project/puffinzip_gui/gui_style_setup.py
import tkinter as tk
from tkinter import ttk
import logging
import sys

BG_COLOR_FALLBACK = "#2E3440"
FG_COLOR_FALLBACK = "#ECEFF4"
FRAME_BG_FALLBACK = "#3B4252"
ACCENT_COLOR_FALLBACK = "#88C0D0"
INPUT_BG_FALLBACK = "#434C5E"
TEXT_AREA_FG_FALLBACK = FG_COLOR_FALLBACK
BUTTON_BG_FALLBACK = "#4C566A"
BUTTON_FG_FALLBACK = FG_COLOR_FALLBACK
TAB_BG_FALLBACK = BG_COLOR_FALLBACK
ACTIVE_TAB_BG_FALLBACK = FRAME_BG_FALLBACK
TAB_BORDER_COLOR_FALLBACK = "#4C566A"
FONT_FAMILY_FALLBACK = "Segoe UI"
FONT_SIZE_BASE_FALLBACK = 10
FONT_NORMAL_FALLBACK = (FONT_FAMILY_FALLBACK, FONT_SIZE_BASE_FALLBACK)
FONT_BOLD_FALLBACK = (FONT_FAMILY_FALLBACK, FONT_SIZE_BASE_FALLBACK, "bold")
FONT_SECTION_TITLE_FALLBACK = (FONT_FAMILY_FALLBACK, FONT_SIZE_BASE_FALLBACK + 2, "bold")
FONT_SMALL_BUTTON_FALLBACK = (FONT_FAMILY_FALLBACK, FONT_SIZE_BASE_FALLBACK -1)
SCROLLBAR_TROUGH_FALLBACK = FRAME_BG_FALLBACK
SCROLLBAR_BG_FALLBACK = "#4C566A"
SCROLLBAR_ACTIVE_BG_FALLBACK = ACCENT_COLOR_FALLBACK
DISABLED_FG_FALLBACK = "#4C566A"
ERROR_FG_FALLBACK = "#BF616A"
FONT_MONO_FALLBACK = ("Consolas", FONT_SIZE_BASE_FALLBACK)


def _get_theme_attr(app_instance, attr_name, default_value):
    if app_instance and hasattr(app_instance, attr_name):
        val = getattr(app_instance, attr_name)
        if val is not None:
            return val
    return default_value

def setup_styles(app_instance):
    logger = getattr(app_instance, 'logger', None)
    if not logger or (hasattr(logger,'handlers') and not logger.handlers): # Ensure logger has handlers
        logger = logging.getLogger("GuiStyleSetupFallback_PMA")
        if not logger.handlers: # Double check, could be first init
            _fb_handler_style = logging.StreamHandler(sys.stdout)
            _fb_formatter_style = logging.Formatter('%(asctime)s - STYLE_FB_PMA - %(levelname)s - %(message)s')
            _fb_handler_style.setFormatter(_fb_formatter_style)
            logger.addHandler(_fb_handler_style)
            logger.setLevel(logging.INFO)

    style = ttk.Style(app_instance)

    bg_color = _get_theme_attr(app_instance, 'BG_COLOR', BG_COLOR_FALLBACK)
    fg_color = _get_theme_attr(app_instance, 'FG_COLOR', FG_COLOR_FALLBACK)
    frame_bg = _get_theme_attr(app_instance, 'FRAME_BG', FRAME_BG_FALLBACK)
    input_bg = _get_theme_attr(app_instance, 'INPUT_BG', INPUT_BG_FALLBACK)
    button_bg = _get_theme_attr(app_instance, 'BUTTON_BG', BUTTON_BG_FALLBACK)
    button_fg = _get_theme_attr(app_instance, 'BUTTON_FG', BUTTON_FG_FALLBACK)
    accent_color = _get_theme_attr(app_instance, 'ACCENT_COLOR', ACCENT_COLOR_FALLBACK)
    tab_bg = _get_theme_attr(app_instance, 'TAB_BG', TAB_BG_FALLBACK)
    active_tab_bg = _get_theme_attr(app_instance, 'ACTIVE_TAB_BG', ACTIVE_TAB_BG_FALLBACK)
    tab_border_color = _get_theme_attr(app_instance, 'TAB_BORDER_COLOR', TAB_BORDER_COLOR_FALLBACK)
    scrollbar_trough_color = _get_theme_attr(app_instance, 'SCROLLBAR_TROUGH', SCROLLBAR_TROUGH_FALLBACK)
    scrollbar_handle_bg = _get_theme_attr(app_instance, 'SCROLLBAR_BG', SCROLLBAR_BG_FALLBACK)
    scrollbar_handle_active_bg = _get_theme_attr(app_instance, 'SCROLLBAR_ACTIVE_BG', SCROLLBAR_ACTIVE_BG_FALLBACK)
    disabled_fg = _get_theme_attr(app_instance, 'DISABLED_FG_COLOR', DISABLED_FG_FALLBACK)
    error_fg = _get_theme_attr(app_instance, 'ERROR_FG_COLOR', ERROR_FG_FALLBACK)
    text_area_bg = _get_theme_attr(app_instance, 'TEXT_AREA_BG', bg_color) # Fallback to main bg if not specific

    font_normal = _get_theme_attr(app_instance, 'FONT_NORMAL', FONT_NORMAL_FALLBACK)
    font_base_family = font_normal[0]
    font_base_size = font_normal[1]

    font_bold = _get_theme_attr(app_instance, 'FONT_BOLD', (font_base_family, font_base_size, "bold"))
    font_section_title = _get_theme_attr(app_instance, 'FONT_SECTION_TITLE', (font_base_family, font_base_size + 2, "bold"))
    font_small_button = _get_theme_attr(app_instance, 'FONT_SMALL_BUTTON', (font_base_family, font_base_size -1))
    font_button = _get_theme_attr(app_instance, 'FONT_BUTTON', font_normal)
    font_mono = _get_theme_attr(app_instance, 'FONT_MONO', (FONT_MONO_FALLBACK[0], font_base_size))

    accent_hover_color = _get_theme_attr(app_instance, 'ACCENT_HOVER_COLOR', "#8FBCBB") # Fallback needed if not in config
    accent_pressed_color = _get_theme_attr(app_instance, 'ACCENT_PRESSED_COLOR', "#5E81AC") # Fallback needed

    try:
        available_themes = style.theme_names()
        preferred_themes = ['clam', 'alt', 'default']
        for theme_name in preferred_themes:
            if theme_name in available_themes:
                style.theme_use(theme_name)
                break
        else:
            pass
    except tk.TclError as e_theme:
        pass

    style.configure("TFrame", background=frame_bg)
    style.configure("Scrollable.TFrame", background=frame_bg)

    style.configure("TLabel", background=frame_bg, foreground=fg_color, font=font_normal, padding=(3, 3))
    style.configure("Title.TLabel", font=font_section_title, foreground=accent_color, background=frame_bg,
                    padding=(0, 5, 0, 8))
    style.configure("Error.TLabel", foreground=error_fg, background=frame_bg, font=font_normal)
    style.configure("Success.TLabel", foreground=_get_theme_attr(app_instance, 'PLOT_LINE_COLOR_MEDIAN_DEFAULT', "#A3BE8C"),
                    background=frame_bg, font=font_normal)

    style.configure("TButton", font=font_button, background=button_bg, foreground=button_fg, relief=tk.FLAT,
                    padding=(10, 6), borderwidth=1, focusthickness=1, focuscolor=accent_color)
    style.map("TButton",
              background=[('pressed', accent_pressed_color), ('active', accent_hover_color), ('disabled', input_bg)],
              foreground=[('disabled', disabled_fg)],
              relief=[('pressed', tk.SUNKEN), ('!pressed', tk.FLAT)],
              focuscolor=[('focus', accent_color)])

    style.configure("Small.TButton", font=font_small_button, padding=(6, 3), relief=tk.FLAT, background=button_bg,
                    foreground=button_fg)
    style.map("Small.TButton",
              background=[('pressed', accent_pressed_color), ('active', accent_hover_color)],
              relief=[('pressed', tk.SUNKEN), ('!pressed', tk.FLAT)])

    style.configure("TEntry", font=font_normal, fieldbackground=input_bg, foreground=fg_color,
                    insertbackground=fg_color, selectbackground=accent_color, selectforeground=input_bg, relief=tk.FLAT,
                    borderwidth=1, bordercolor="#555555")
    style.map("TEntry", bordercolor=[('focus', accent_color)], relief=[('focus', tk.SOLID)])

    style.configure("TCombobox", font=font_normal, fieldbackground=input_bg, foreground=fg_color,
                    selectbackground=input_bg, selectforeground=accent_color, arrowcolor=fg_color, relief=tk.FLAT,
                    padding=(4, 4), insertwidth=1, insertbackground=fg_color)
    style.map("TCombobox",
              fieldbackground=[('readonly', input_bg), ('disabled', frame_bg)],
              foreground=[('disabled', disabled_fg)],
              arrowcolor=[('disabled', disabled_fg)],
              background=[('active', input_bg)],
              relief=[('focus', tk.SOLID), ('hover', tk.FLAT)],
              bordercolor=[('focus', accent_color)])

    app_instance.option_add("*TCombobox*Listbox.background", input_bg)
    app_instance.option_add("*TCombobox*Listbox.foreground", fg_color)
    app_instance.option_add("*TCombobox*Listbox.selectBackground", accent_color)
    app_instance.option_add("*TCombobox*Listbox.selectForeground", input_bg)
    app_instance.option_add("*TCombobox*Listbox.font", font_normal)

    style.configure("TCheckbutton", background=frame_bg, foreground=fg_color, font=font_normal, indicatorrelief=tk.FLAT,
                    indicatormargin=3, indicatorsize=14, padding=(5, 3))
    style.map("TCheckbutton",
              indicatorbackground=[
                  ('selected', 'disabled', accent_color),
                  ('disabled', frame_bg),
                  ('selected', accent_color),
                  ('!selected', input_bg)
              ],
              indicatorforeground=[
                  ('selected', 'disabled', input_bg),
                  ('disabled', disabled_fg),
                  ('selected', input_bg),
                  ('!selected', fg_color)
              ],
              background=[('active', frame_bg)],
              foreground=[('disabled', disabled_fg)]
              )

    style.configure("TNotebook", background=bg_color, tabmargins=[2, 5, 2, 0], borderwidth=0)
    style.configure("TNotebook.Tab", background=tab_bg, foreground=fg_color, font=font_normal, padding=[12, 5],
                    focuscolor=input_bg, bordercolor=tab_border_color, lightcolor=frame_bg, darkcolor=bg_color)
    style.map("TNotebook.Tab",
              background=[("selected", active_tab_bg), ("active", frame_bg)],
              foreground=[("selected", accent_color), ("active", fg_color)],
              font=[("selected", font_bold)],
              expand=[("selected", [1, 1, 1, 0])])
    try:
        style.configure("TNotebook.Tabpane", background=frame_bg, borderwidth=1, relief="solid",
                        bordercolor=tab_border_color)
    except tk.TclError:
        pass


    style.configure("Vertical.TScrollbar", gripcount=0, background=scrollbar_handle_bg,
                    troughcolor=scrollbar_trough_color, bordercolor=frame_bg, arrowcolor=fg_color, relief=tk.FLAT,
                    width=14)
    style.map("Vertical.TScrollbar",
              background=[('active', scrollbar_handle_active_bg), ('disabled', frame_bg)],
              arrowcolor=[('disabled', disabled_fg)]
              )
    style.configure("Horizontal.TScrollbar", gripcount=0, background=scrollbar_handle_bg,
                    troughcolor=scrollbar_trough_color, bordercolor=frame_bg, arrowcolor=fg_color, relief=tk.FLAT,
                    height=14)
    style.map("Horizontal.TScrollbar",
              background=[('active', scrollbar_handle_active_bg), ('disabled', frame_bg)],
              arrowcolor=[('disabled', disabled_fg)]
              )

    style.configure("TLabelframe", background=frame_bg, labelmargins=(0, 0, 0, 4), padding=(10, 5, 10, 10),
                    relief="solid", bordercolor=tab_border_color, borderwidth=1)
    style.configure("TLabelframe.Label", background=frame_bg, foreground=accent_color, font=font_section_title,
                    padding=(0, 0, 0, 5))

    style.configure("TPanedwindow", background=bg_color)
    style.configure("PanedWindow.Sash", background=frame_bg, sashthickness=6, gripcount=10, relief=tk.RAISED,
                    borderwidth=1)
    style.map("PanedWindow.Sash", background=[('active', accent_color)])

    gdv_style_prefix = "GenDataViewer." # Assuming this is how GDV accesses its styles from PuffinZipApp
    gdv_tree_bg = _get_theme_attr(app_instance, 'TEXT_AREA_BG', bg_color)
    gdv_tree_fg = _get_theme_attr(app_instance, 'FG_COLOR', fg_color)
    gdv_heading_bg = _get_theme_attr(app_instance, 'BUTTON_BG', button_bg)
    gdv_heading_fg = _get_theme_attr(app_instance, 'BUTTON_FG', button_fg)
    gdv_selected_bg = _get_theme_attr(app_instance, 'ACCENT_COLOR', accent_color)
    gdv_selected_fg = _get_theme_attr(app_instance, 'TEXT_AREA_BG', gdv_tree_bg) # Contrast with selection

    style.configure(f"{gdv_style_prefix}Treeview", background=gdv_tree_bg, fieldbackground=gdv_tree_bg,
                    foreground=gdv_tree_fg, font=font_mono,
                    rowheight=int(font_mono[1] * 2.2 if font_mono and len(font_mono) > 1 else 25))
    style.configure(f"{gdv_style_prefix}Treeview.Heading", background=gdv_heading_bg, foreground=gdv_heading_fg, font=font_bold,
                    relief=tk.FLAT, padding=(8, 6), borderwidth=0)
    style.map(f"{gdv_style_prefix}Treeview.Heading", background=[('active', accent_color)],
              relief=[('active', 'groove')])
    style.map(f"{gdv_style_prefix}Treeview", background=[('selected', gdv_selected_bg)],
              foreground=[('selected', gdv_selected_fg)])


if __name__ == '__main__':
    pass