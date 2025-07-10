# PuffinZipAI_Project/puffinzip_gui/gui_utils.py
import tkinter as tk
from tkinter import ttk
import sys
import logging
import tkinter.font  # Explicit import for clarity

FRAME_BG_FALLBACK = "#3C3C3C"
FONT_FAMILY_PRIMARY_DEFAULT = "Segoe UI"  # Changed from FONT_FAMILY_PRIMARY
FONT_FAMILY_FALLBACK_GENERIC_DEFAULT = "Arial"  # Changed
FONT_FAMILY_ITALIC_COMMON_FALLBACK_DEFAULT = "Verdana"  # Changed
FONT_SIZE_BASE_DEFAULT = 10  # Changed
FONT_NORMAL_FALLBACK_DEFAULT = (FONT_FAMILY_PRIMARY_DEFAULT, FONT_SIZE_BASE_DEFAULT)  # Changed
FONT_MONO_FAMILY_DEFAULT = "Courier New"  # Changed
FONT_MONO_FALLBACK_DEFAULT = (FONT_MONO_FAMILY_DEFAULT, FONT_SIZE_BASE_DEFAULT)  # Changed


def _get_theme_attr(app_instance, attr_name, default_value):
    if app_instance and hasattr(app_instance, attr_name):
        val = getattr(app_instance, attr_name)
        if val is not None:
            return val
    return default_value


def _get_font_with_fallbacks(app_instance, primary_font_tuple, secondary_font_tuple=None):
    logger = _get_theme_attr(app_instance, 'logger', logging.getLogger("GuiUtilsFallbackLogger_PMA"))
    if hasattr(logger, 'handlers') and not logger.handlers:
        _fb_handler_gu = logging.StreamHandler(sys.stdout)
        _fb_formatter_gu = logging.Formatter('%(asctime)s - GUI_UTILS_FB_PMA - %(levelname)s - %(message)s')
        _fb_handler_gu.setFormatter(_fb_formatter_gu)
        logger.addHandler(_fb_handler_gu)
        logger.setLevel(logging.DEBUG)

    font_to_try_configs = [primary_font_tuple]
    if secondary_font_tuple:
        font_to_try_configs.append(secondary_font_tuple)

    for idx, font_config_tuple in enumerate(font_to_try_configs):
        if not (font_config_tuple and isinstance(font_config_tuple, tuple) and len(font_config_tuple) >= 2):
            continue

        family_name = font_config_tuple[0]
        size = font_config_tuple[1]
        style_parts = []
        slant_part = "roman"
        weight_part = "normal"

        if len(font_config_tuple) > 2:
            style_str_lower = str(font_config_tuple[2]).lower()
            if "bold" in style_str_lower:
                weight_part = "bold"
            if "italic" in style_str_lower:
                slant_part = "italic"
            elif "oblique" in style_str_lower:  # Numba also uses oblique for italic sometimes
                slant_part = "oblique"

        current_full_tuple_for_tk = (family_name, size, weight_part, slant_part)

        try:
            font_module = getattr(app_instance, 'tk_font', tkinter.font) if app_instance else tkinter.font
            font_obj = font_module.Font(family=family_name, size=size, weight=weight_part, slant=slant_part)
            actual_font_props = font_obj.actual()  # Check if font can be realized

            # Tk might silently fallback, try to check if what we got is what we asked for
            actual_family_returned = actual_font_props.get('family', '').lower()
            requested_family_lower = family_name.lower()

            # Basic check: if Tk returns a vastly different family, it's a significant fallback
            if actual_family_returned != requested_family_lower and not (
                    requested_family_lower in actual_family_returned or actual_family_returned in requested_family_lower):
                if idx == 0:  # Only log primary attempt failure of this type if it happens
                    pass  # Allow Tk's own fallback for primary if it's subtle
                else:  # For secondary / explicit fallbacks, be stricter
                    raise tk.TclError(f"Tk returned '{actual_font_props.get('family')}' for requested '{family_name}'")

            final_style_str = []
            if actual_font_props.get('weight') == 'bold': final_style_str.append('bold')
            if actual_font_props.get('slant') == 'italic': final_style_str.append('italic')

            return (actual_font_props.get('family'), actual_font_props.get('size'),
                    " ".join(final_style_str) if final_style_str else "normal")

        except tk.TclError:
            pass  # Will try next in list or generic fallbacks
        except Exception as e_font_other:  # Catch other potential font errors
            pass

    # Generic fallbacks if all specific attempts fail
    primary_family_req = primary_font_tuple[0]
    primary_size_req = primary_font_tuple[1]
    primary_style_req_str = str(primary_font_tuple[2]).lower() if len(primary_font_tuple) > 2 else "normal"

    weight_fb = "bold" if "bold" in primary_style_req_str else "normal"
    slant_fb = "italic" if "italic" in primary_style_req_str or "oblique" in primary_style_req_str else "roman"

    is_mono_request_fb = primary_family_req.lower() in ["consolas", "courier new", "courier", "fixedsys", "terminal",
                                                        "monaco", "menlo", "source code pro", "liberation mono",
                                                        "dejavu sans mono"]

    generic_families_to_try = []
    if is_mono_request_fb:
        generic_families_to_try.extend([FONT_MONO_FAMILY_DEFAULT, FONT_FAMILY_FALLBACK_GENERIC_DEFAULT])
    elif "italic" in primary_style_req_str or "oblique" in primary_style_req_str:
        generic_families_to_try.extend(
            [FONT_FAMILY_ITALIC_COMMON_FALLBACK_DEFAULT, FONT_FAMILY_FALLBACK_GENERIC_DEFAULT,
             FONT_FAMILY_PRIMARY_DEFAULT])
    else:
        generic_families_to_try.extend([FONT_FAMILY_PRIMARY_DEFAULT, FONT_FAMILY_FALLBACK_GENERIC_DEFAULT])

    for fam_fb in generic_families_to_try:
        try:
            font_module = getattr(app_instance, 'tk_font', tkinter.font) if app_instance else tkinter.font
            font_obj_fb = font_module.Font(family=fam_fb, size=primary_size_req, weight=weight_fb, slant=slant_fb)
            actual_fb = font_obj_fb.actual()
            final_style_str_fb = []
            if actual_fb.get('weight') == 'bold': final_style_str_fb.append('bold')
            if actual_fb.get('slant') == 'italic': final_style_str_fb.append('italic')
            return (actual_fb.get('family'), actual_fb.get('size'),
                    " ".join(final_style_str_fb) if final_style_str_fb else "normal")
        except tk.TclError:
            continue
        except Exception:
            continue

    # Absolute last resort
    final_style_for_tk_default = []
    if weight_fb == 'bold': final_style_for_tk_default.append('bold')
    if slant_fb == 'italic': final_style_for_tk_default.append('italic')
    return ("TkDefaultFont", primary_size_req,
            " ".join(final_style_for_tk_default) if final_style_for_tk_default else "normal")


def on_frame_configure(canvas, scrollable_frame, event=None, app_instance=None):
    logger = _get_theme_attr(app_instance, 'logger', logging.getLogger("GuiUtilsFallbackLogger_PMA_FrameCfg"))
    try:
        if canvas and canvas.winfo_exists() and scrollable_frame and scrollable_frame.winfo_exists():
            canvas.update_idletasks()
            scroll_region = canvas.bbox("all")

            current_canvas_width = canvas.winfo_width()
            current_canvas_height = canvas.winfo_height()

            if scroll_region:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = scroll_region
                content_width = bbox_x2 - bbox_x1
                content_height = bbox_y2 - bbox_y1

                # Ensure scroll region width isn't less than canvas width (can happen if frame hasn't expanded yet)
                # And ensure scroll region height isn't less than canvas height (for vertical scrollbar visibility)
                final_scroll_width = max(content_width, current_canvas_width)
                final_scroll_height = max(content_height, current_canvas_height)

                canvas.config(
                    scrollregion=(bbox_x1, bbox_y1, bbox_x1 + final_scroll_width, bbox_y1 + final_scroll_height))
            else:  # No content, or content is zero-size
                canvas.config(scrollregion=(0, 0, current_canvas_width, current_canvas_height))

    except tk.TclError as e_tcl:
        if "application has been destroyed" not in str(e_tcl).lower():  # Avoid spamming on clean exit
            pass
    except Exception as e:
        pass


def on_canvas_configure(event, scrollable_frame_id, canvas, app_instance=None):
    logger = _get_theme_attr(app_instance, 'logger', logging.getLogger("GuiUtilsFallbackLogger_PMA_CvsCfg"))
    try:
        if canvas and canvas.winfo_exists():
            all_items_on_canvas = canvas.find_all()
            if scrollable_frame_id in all_items_on_canvas:
                new_canvas_width = event.width
                canvas.itemconfig(scrollable_frame_id, width=new_canvas_width)
                canvas.update_idletasks()
                # Call on_frame_configure to re-calculate bbox based on potentially new frame width requirements
                frame_widget = canvas.nametowidget(canvas.itemcget(scrollable_frame_id, "-window"))
                if frame_widget and frame_widget.winfo_exists():
                    on_frame_configure(canvas, frame_widget, app_instance=app_instance)
            elif scrollable_frame_id not in all_items_on_canvas and all_items_on_canvas:
                # This might happen if the frame was removed/recreated. If only one frame, assume it's the one.
                if len(all_items_on_canvas) == 1 and canvas.type(all_items_on_canvas[0]) == 'window':
                    canvas.itemconfig(all_items_on_canvas[0], width=event.width)


    except tk.TclError as e_tcl:
        if "application has been destroyed" not in str(e_tcl).lower():
            pass
    except Exception as e:
        pass


def _handle_canvas_scroll(event, canvas, app_instance=None):
    logger = _get_theme_attr(app_instance, 'logger', logging.getLogger("GuiUtilsFallbackLogger_PMA_Scroll"))
    delta = 0;
    current_platform = sys.platform
    try:
        if current_platform == "darwin":  # macOS
            delta = -1 * event.delta
        elif current_platform.startswith("win"):  # Windows
            delta = -1 * (event.delta // 120)  # Mouse wheel units
        elif event.num == 4:  # Linux scroll up
            delta = -1
        elif event.num == 5:  # Linux scroll down
            delta = 1
        else:  # General fallback, might capture touchpad scroll events if delta attribute exists
            if hasattr(event, 'delta') and event.delta != 0:
                delta = -1 * (event.delta // 120) if abs(event.delta) >= 120 else -1 * event.delta
            else:
                delta = 0

        if delta != 0 and hasattr(canvas, 'yview_scroll') and canvas.winfo_exists():
            canvas.yview_scroll(delta, "units");
            return "break"  # Prevents event from propagating further
    except tk.TclError:
        pass  # Widget might have been destroyed
    except Exception as e_scroll:
        pass
    return None


def _bind_events_recursively(widget, scroll_command, enter_command, app_instance=None):
    logger = _get_theme_attr(app_instance, 'logger', logging.getLogger("GuiUtilsFallbackLogger_PMA_BindRec"))
    try:
        if widget and widget.winfo_exists():
            # Bind mouse wheel for vertical scrolling
            widget.bind("<MouseWheel>", scroll_command, add="+")  # Windows, some Linux
            widget.bind("<Button-4>", scroll_command, add="+")  # Linux scroll up
            widget.bind("<Button-5>", scroll_command, add="+")  # Linux scroll down

            # Set focus on enter for keyboard scrolling if canvas supports it
            widget.bind("<Enter>", enter_command, add="+")

            for child in widget.winfo_children():
                _bind_events_recursively(child, scroll_command, enter_command, app_instance)
    except tk.TclError:
        pass  # Widget might be destroyed
    except Exception as e_bind_rec:
        pass


def clear_frame_widgets(frame: tk.Frame, app_instance=None):
    logger = _get_theme_attr(app_instance, 'logger', logging.getLogger("GuiUtilsFallbackLogger_PMA_Clear"))
    if frame and hasattr(frame, 'winfo_exists') and frame.winfo_exists():
        for widget_item in frame.winfo_children():
            try:
                widget_item.destroy()
            except tk.TclError:
                pass  # Widget already destroyed
            except Exception as e_clear:
                pass
    elif not frame and logger:
        pass