"""Utility helpers for PuffinZip's Tkinter-based interface.

This module centralises a number of common GUI helpers so that widgets share
the same look-and-feel and so that expensive theme lookups are cached.  The
functions are intentionally defensive – the production application is often
executed in environments where Tk fonts or colours may not be present.  The
helpers therefore provide sensible fallbacks while trying to keep the
interface consistent across the different windows.
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk
import tkinter.font
import weakref
from tkinter import ttk
from typing import Any, Dict, Iterable, Optional, Tuple

# ---------------------------------------------------------------------------
# Theme fallbacks & constants
# ---------------------------------------------------------------------------

# Colour palette fallbacks.  These values mirror the defaults defined inside
# ``gui_style_setup`` so that the helpers can be used in isolation (for
# instance in unit tests) without importing the styling module and creating a
# circular dependency.
_THEME_FALLBACKS: Dict[str, Any] = {
    "BG_COLOR": "#2E3440",
    "FG_COLOR": "#ECEFF4",
    "FRAME_BG": "#3B4252",
    "ACCENT_COLOR": "#88C0D0",
    "ACCENT_HOVER_COLOR": "#8FBCBB",
    "ACCENT_PRESSED_COLOR": "#5E81AC",
    "INPUT_BG": "#434C5E",
    "TEXT_AREA_BG": "#1E1E1E",
    "TEXT_AREA_FG": "#D0D0D0",
    "BUTTON_BG": "#4C566A",
    "BUTTON_FG": "#ECEFF4",
    "TAB_BG": "#2E3440",
    "ACTIVE_TAB_BG": "#3B4252",
    "TAB_BORDER_COLOR": "#4C566A",
    "SCROLLBAR_TROUGH": "#3B4252",
    "SCROLLBAR_BG": "#4C566A",
    "SCROLLBAR_ACTIVE_BG": "#88C0D0",
    "DISABLED_FG_COLOR": "#4C566A",
    "ERROR_FG_COLOR": "#BF616A",
    "PLOT_LINE_COLOR_MEDIAN_DEFAULT": "#A3BE8C",
}

FONT_FAMILY_PRIMARY_DEFAULT = "Segoe UI"
FONT_FAMILY_FALLBACK_GENERIC_DEFAULT = "Arial"
FONT_FAMILY_ITALIC_COMMON_FALLBACK_DEFAULT = "Verdana"
FONT_SIZE_BASE_DEFAULT = 10
FONT_MONO_FAMILY_DEFAULT = "Consolas"

_MONO_FAMILY_CANDIDATES: Tuple[str, ...] = (
    "Consolas",
    "Courier New",
    "Courier",
    "Source Code Pro",
    "Menlo",
    "Liberation Mono",
    "DejaVu Sans Mono",
)

_PRIMARY_FAMILY_CANDIDATES: Tuple[str, ...] = (
    FONT_FAMILY_PRIMARY_DEFAULT,
    FONT_FAMILY_FALLBACK_GENERIC_DEFAULT,
    "Helvetica",
    "Arial",
    "Sans",
)

_ITALIC_FAMILY_CANDIDATES: Tuple[str, ...] = (
    FONT_FAMILY_ITALIC_COMMON_FALLBACK_DEFAULT,
    FONT_FAMILY_FALLBACK_GENERIC_DEFAULT,
    FONT_FAMILY_PRIMARY_DEFAULT,
)

_FONT_CACHE: "weakref.WeakKeyDictionary[tk.Misc, Dict[Any, Tuple[str, int, str]]]" = (
    weakref.WeakKeyDictionary()
)
_GLOBAL_FONT_CACHE: Dict[Any, Tuple[str, int, str]] = {}

_SCROLL_BIND_MARKER = "_pz_scroll_binding_active"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_logger(app_instance: Optional[tk.Misc], fallback_name: str) -> logging.Logger:
    """Return a logger associated with the application or a fallback logger."""

    logger = getattr(app_instance, "logger", None)
    if logger:
        return logger

    logger = logging.getLogger(fallback_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - GUI_UTILS - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _get_theme_attr(app_instance: Optional[tk.Misc], attr_name: str, default_value: Any) -> Any:
    """Backward-compatible private accessor used throughout the code base."""

    return get_theme_attr(app_instance, attr_name, default_value)


def get_theme_attr(app_instance: Optional[tk.Misc], attr_name: str, default_value: Any = None) -> Any:
    """Return a themed attribute from the app with fallbacks.

    The helper first checks the ``app_instance`` for the attribute, returning it
    when it is defined and not ``None``.  When the attribute is absent the
    function falls back to the module level defaults and finally to the explicit
    ``default_value`` provided by the caller.
    """

    if app_instance and hasattr(app_instance, attr_name):
        value = getattr(app_instance, attr_name)
        if value is not None:
            return value

    if attr_name in _THEME_FALLBACKS:
        return _THEME_FALLBACKS[attr_name]

    return default_value


def _font_cache_for(app_instance: Optional[tk.Misc]) -> Dict[Any, Tuple[str, int, str]]:
    if app_instance is None:
        return _GLOBAL_FONT_CACHE

    cached = _FONT_CACHE.get(app_instance)
    if cached is None:
        cached = {}
        _FONT_CACHE[app_instance] = cached
    return cached


def _normalise_font_tuple(font_tuple: Tuple[Any, ...]) -> Tuple[str, int, str, str]:
    """Normalise a font tuple into (family, size, weight, slant)."""

    if not font_tuple or len(font_tuple) < 2:
        return FONT_FAMILY_PRIMARY_DEFAULT, FONT_SIZE_BASE_DEFAULT, "normal", "roman"

    family = str(font_tuple[0])
    try:
        size = int(font_tuple[1])
    except (TypeError, ValueError):
        size = FONT_SIZE_BASE_DEFAULT

    style = str(font_tuple[2]).lower() if len(font_tuple) > 2 else "normal"
    weight = "bold" if "bold" in style else "normal"
    if "italic" in style or "oblique" in style:
        slant = "italic"
    else:
        slant = "roman"

    return family, size, weight, slant


def _style_from_actual(actual: Dict[str, Any]) -> str:
    parts: Iterable[str] = []
    if actual.get("weight") == "bold":
        parts = list(parts) + ["bold"]
    if actual.get("slant") in {"italic", "oblique"}:
        parts = list(parts) + ["italic"]
    joined = " ".join(parts)
    return joined if joined else "normal"


def _attempt_font_resolution(
    app_instance: Optional[tk.Misc],
    candidate: Tuple[str, int, str, str],
) -> Tuple[str, int, str]:
    """Try to resolve a font, returning Tk's actual family/size/style tuple."""

    font_module = getattr(app_instance, "tk_font", tkinter.font) if app_instance else tkinter.font
    try:
        font_obj = font_module.Font(
            family=candidate[0], size=candidate[1], weight=candidate[2], slant=candidate[3]
        )
        actual = font_obj.actual()
    finally:
        try:
            font_obj.destroy()
        except Exception:  # pragma: no cover - Tk may not support destroy during shutdown
            pass

    resolved_family = actual.get("family", candidate[0])
    try:
        resolved_size = int(actual.get("size", candidate[1]))
    except (TypeError, ValueError):
        resolved_size = candidate[1]
    resolved_style = _style_from_actual(actual)

    return resolved_family, resolved_size, resolved_style


def _is_monospaced_request(family: str) -> bool:
    family_lower = family.lower()
    return any(monospaced.lower() == family_lower for monospaced in _MONO_FAMILY_CANDIDATES)


def _build_generic_font_candidates(primary: Tuple[str, int, str, str]) -> Iterable[Tuple[str, int, str, str]]:
    family, size, weight, slant = primary
    candidates = []

    if _is_monospaced_request(family):
        for mono_family in _MONO_FAMILY_CANDIDATES:
            if mono_family.lower() != family.lower():
                candidates.append((mono_family, size, weight, slant))
    elif slant == "italic":
        for italic_family in _ITALIC_FAMILY_CANDIDATES:
            if italic_family.lower() != family.lower():
                candidates.append((italic_family, size, weight, slant))
    else:
        for primary_family in _PRIMARY_FAMILY_CANDIDATES:
            if primary_family.lower() != family.lower():
                candidates.append((primary_family, size, weight, slant))

    return candidates


def _get_font_with_fallbacks(
    app_instance: Optional[tk.Misc],
    primary_font_tuple: Tuple[Any, ...],
    secondary_font_tuple: Optional[Tuple[Any, ...]] = None,
) -> Tuple[str, int, str]:
    """Resolve a Tk font tuple with caching and sensible fallbacks."""

    cache = _font_cache_for(app_instance)
    cache_key = (primary_font_tuple, secondary_font_tuple)
    if cache_key in cache:
        return cache[cache_key]

    logger = _get_logger(app_instance, "GuiUtilsFallbackLogger_PMA_Font")

    candidate_tuples = [_normalise_font_tuple(primary_font_tuple)]
    if secondary_font_tuple:
        candidate_tuples.append(_normalise_font_tuple(secondary_font_tuple))

    candidate_tuples.extend(_build_generic_font_candidates(candidate_tuples[0]))

    for candidate in candidate_tuples:
        try:
            resolved = _attempt_font_resolution(app_instance, candidate)
            cache[cache_key] = resolved
            return resolved
        except tk.TclError:
            continue
        except Exception:  # pragma: no cover - defensive logging for Tk edge cases
            logger.debug("Unexpected font resolution issue", exc_info=True)

    # Final fallback – let Tk decide using its default font.
    fallback_style = "italic" if candidate_tuples[0][3] == "italic" else "normal"
    resolved = ("TkDefaultFont", candidate_tuples[0][1], fallback_style)
    cache[cache_key] = resolved
    return resolved


def build_font_palette(
    primary_family: str,
    base_size: int,
    mono_family: str,
) -> Dict[str, Tuple[Any, ...]]:
    """Return the desired font tuples for the application."""

    small_size = max(7, base_size - 1)
    large_size = base_size + 2

    return {
        "FONT_NORMAL": (primary_family, base_size, "normal"),
        "FONT_BOLD": (primary_family, base_size, "bold"),
        "FONT_SECTION_TITLE": (primary_family, large_size, "bold"),
        "FONT_BUTTON": (primary_family, base_size, "normal"),
        "FONT_SMALL": (primary_family, small_size, "normal"),
        "FONT_SMALL_BUTTON": (primary_family, small_size, "normal"),
        "FONT_NOTE": (primary_family, small_size, "italic"),
        "FONT_MONO": (mono_family, base_size, "normal"),
    }


def build_font_fallbacks(base_size: int, mono_family: str) -> Dict[str, Tuple[Any, ...]]:
    """Create fallback tuples for the font palette using common fonts."""

    small_size = max(7, base_size - 1)
    large_size = base_size + 2

    return {
        "FONT_NORMAL": (FONT_FAMILY_FALLBACK_GENERIC_DEFAULT, base_size, "normal"),
        "FONT_BOLD": (FONT_FAMILY_FALLBACK_GENERIC_DEFAULT, base_size, "bold"),
        "FONT_SECTION_TITLE": (FONT_FAMILY_FALLBACK_GENERIC_DEFAULT, large_size, "bold"),
        "FONT_BUTTON": (FONT_FAMILY_FALLBACK_GENERIC_DEFAULT, base_size, "normal"),
        "FONT_SMALL": (FONT_FAMILY_FALLBACK_GENERIC_DEFAULT, small_size, "normal"),
        "FONT_SMALL_BUTTON": (FONT_FAMILY_FALLBACK_GENERIC_DEFAULT, small_size, "normal"),
        "FONT_NOTE": (FONT_FAMILY_ITALIC_COMMON_FALLBACK_DEFAULT, small_size, "italic"),
        "FONT_MONO": (mono_family or FONT_MONO_FAMILY_DEFAULT, base_size, "normal"),
    }


def initialize_app_fonts(
    app_instance: tk.Misc,
    primary_family: str,
    base_size: int,
    mono_family: str,
) -> Dict[str, Tuple[str, int, str]]:
    """Resolve and assign the standard font palette on ``app_instance``.

    Returns a dictionary containing the resolved font tuples.  All font
    attributes (``FONT_NORMAL``, ``FONT_BOLD`` …) are created or updated on the
    provided ``app_instance``.
    """

    palette = build_font_palette(primary_family, base_size, mono_family)
    fallbacks = build_font_fallbacks(base_size, mono_family)

    resolved_fonts: Dict[str, Tuple[str, int, str]] = {}
    for attr_name, spec in palette.items():
        resolved = _get_font_with_fallbacks(app_instance, spec, fallbacks.get(attr_name))
        setattr(app_instance, attr_name, resolved)
        resolved_fonts[attr_name] = resolved

    return resolved_fonts


# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------

def on_frame_configure(
    canvas: tk.Canvas,
    scrollable_frame: tk.Widget,
    event: Optional[tk.Event] = None,
    app_instance: Optional[tk.Misc] = None,
) -> None:
    """Update the canvas scrollregion when the frame size changes."""

    if not canvas or not canvas.winfo_exists() or not scrollable_frame or not scrollable_frame.winfo_exists():
        return

    canvas.update_idletasks()
    scroll_region = canvas.bbox("all")
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    if scroll_region:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = scroll_region
        content_width = bbox_x2 - bbox_x1
        content_height = bbox_y2 - bbox_y1
        final_width = max(content_width, canvas_width)
        final_height = max(content_height, canvas_height)
        new_region = (bbox_x1, bbox_y1, bbox_x1 + final_width, bbox_y1 + final_height)
    else:
        new_region = (0, 0, canvas_width, canvas_height)

    if getattr(canvas, "_pz_last_scrollregion", None) != new_region:
        canvas.config(scrollregion=new_region)
        canvas._pz_last_scrollregion = new_region


def on_canvas_configure(
    event: tk.Event,
    scrollable_frame_id: int,
    canvas: tk.Canvas,
    app_instance: Optional[tk.Misc] = None,
) -> None:
    """Ensure the scrollable frame matches the canvas width when resized."""

    if not canvas or not canvas.winfo_exists():
        return

    all_items = canvas.find_all()
    if scrollable_frame_id in all_items:
        canvas.itemconfig(scrollable_frame_id, width=event.width)
        canvas.update_idletasks()
        try:
            frame_widget = canvas.nametowidget(canvas.itemcget(scrollable_frame_id, "-window"))
        except Exception:
            frame_widget = None
        if frame_widget and frame_widget.winfo_exists():
            on_frame_configure(canvas, frame_widget, app_instance=app_instance)
    elif len(all_items) == 1 and canvas.type(all_items[0]) == "window":
        canvas.itemconfig(all_items[0], width=event.width)


def _handle_canvas_scroll(event: tk.Event, canvas: tk.Canvas, app_instance: Optional[tk.Misc] = None) -> Optional[str]:
    delta = 0
    platform = sys.platform

    try:
        if platform == "darwin":
            delta = -1 * event.delta
        elif platform.startswith("win"):
            delta = -1 * (event.delta // 120)
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif hasattr(event, "delta") and event.delta:
            delta = -1 * (event.delta // 120) if abs(event.delta) >= 120 else -1 * event.delta

        if delta != 0 and canvas and canvas.winfo_exists():
            canvas.yview_scroll(delta, "units")
            return "break"
    except tk.TclError:
        return None

    return None


def _bind_events_recursively(
    widget: tk.Widget,
    scroll_command,
    enter_command,
    app_instance: Optional[tk.Misc] = None,
) -> None:
    if not widget or not widget.winfo_exists():
        return

    if getattr(widget, _SCROLL_BIND_MARKER, False):
        return

    widget.bind("<MouseWheel>", scroll_command, add="+")
    widget.bind("<Button-4>", scroll_command, add="+")
    widget.bind("<Button-5>", scroll_command, add="+")
    widget.bind("<Enter>", enter_command, add="+")
    setattr(widget, _SCROLL_BIND_MARKER, True)

    for child in widget.winfo_children():
        _bind_events_recursively(child, scroll_command, enter_command, app_instance)


def bind_scroll_events(
    root_widget: tk.Widget,
    canvas: tk.Canvas,
    app_instance: Optional[tk.Misc] = None,
) -> None:
    """Bind mouse-wheel events on ``root_widget`` to scroll ``canvas``."""

    if not root_widget or not canvas:
        return

    def _focus_canvas(_event: tk.Event) -> None:
        if canvas.winfo_exists():
            canvas.focus_set()

    scroll_cmd = lambda event: _handle_canvas_scroll(event, canvas, app_instance)
    _bind_events_recursively(root_widget, scroll_cmd, _focus_canvas, app_instance)


def create_scrollable_canvas(
    parent: tk.Widget,
    app_instance: Optional[tk.Misc] = None,
    orient: int = tk.VERTICAL,
    **canvas_kwargs: Any,
) -> Tuple[tk.Canvas, ttk.Scrollbar]:
    """Create a themed canvas+scrollbar pair ready for scrollable content."""

    defaults = {
        "bg": get_theme_attr(app_instance, "FRAME_BG", _THEME_FALLBACKS["FRAME_BG"]),
        "highlightthickness": 0,
        "bd": 0,
    }
    defaults.update(canvas_kwargs)

    canvas = tk.Canvas(parent, **defaults)

    if orient == tk.VERTICAL:
        command = canvas.yview
        style = "Vertical.TScrollbar"
    else:
        command = canvas.xview
        style = "Horizontal.TScrollbar"

    scrollbar = ttk.Scrollbar(parent, orient=orient, command=command, style=style)
    if orient == tk.VERTICAL:
        canvas.configure(yscrollcommand=scrollbar.set)
    else:
        canvas.configure(xscrollcommand=scrollbar.set)

    return canvas, scrollbar


def clear_frame_widgets(frame: tk.Frame, app_instance: Optional[tk.Misc] = None) -> None:
    """Destroy all child widgets from ``frame`` safely."""

    if not frame or not hasattr(frame, "winfo_exists") or not frame.winfo_exists():
        return

    for child in frame.winfo_children():
        try:
            child.destroy()
        except tk.TclError:
            continue


__all__ = [
    "get_theme_attr",
    "initialize_app_fonts",
    "build_font_palette",
    "build_font_fallbacks",
    "on_frame_configure",
    "on_canvas_configure",
    "bind_scroll_events",
    "create_scrollable_canvas",
    "clear_frame_widgets",
]

