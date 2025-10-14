"""Convenient entry points for the PuffinZip GUI package.

The previous initialisation routine performed a large number of eager imports
purely for logging purposes.  This lightweight variant keeps the imports
explicit and predictable while dramatically reducing start-up cost.
"""

from __future__ import annotations

from . import chart_utils, gui_layout_setup, gui_style_setup, gui_utils, settings_gui
from . import generational_data_viewer, secondary_main_app
from .primary_main_app import PuffinZipApp

__all__ = [
    "PuffinZipApp",
    "chart_utils",
    "settings_gui",
    "secondary_main_app",
    "gui_utils",
    "gui_style_setup",
    "gui_layout_setup",
    "generational_data_viewer",
]

