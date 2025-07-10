# PuffinZipAI_Project/puffinzip_gui/__init__.py
import traceback
import logging
import sys

init_logger = logging.getLogger("puffinzip_gui_init_szczegolowy")
init_logger.handlers.clear()
init_handler = logging.StreamHandler(sys.stdout)
init_handler.setFormatter(logging.Formatter('%(asctime)s - GUIMOD_INIT - %(levelname)s - %(message)s'))
init_logger.addHandler(init_handler)
init_logger.setLevel(logging.DEBUG)

init_logger.debug(f"--- puffinzip_gui/__init__.py starting execution. Python version: {sys.version_info}")
init_logger.debug(f"Current sys.path: {sys.path}")
init_logger.debug(f"Current __file__: {__file__}")
init_logger.debug(f"Current __name__: {__name__}")
init_logger.debug(f"Current __package__: {__package__}")

chart_utils_mod = None
settings_gui_mod = None
secondary_main_app_mod = None
gui_utils_mod = None
gui_style_setup_mod = None
gui_layout_setup_mod = None
generational_data_viewer_mod = None
PuffinZipApp_class = None

init_logger.debug("Attempting to import modules using 'from puffinzip_gui import ...'")
try:
    from puffinzip_gui import gui_utils as gu_temp
    gui_utils_mod = gu_temp
    init_logger.info(f"Successfully imported puffinzip_gui.gui_utils: {gui_utils_mod}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.gui_utils: {e}", exc_info=True)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.gui_utils: {e_gen}", exc_info=True)

try:
    from puffinzip_gui import gui_style_setup as gss_temp
    gui_style_setup_mod = gss_temp
    init_logger.info(f"Successfully imported puffinzip_gui.gui_style_setup: {gui_style_setup_mod}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.gui_style_setup: {e}", exc_info=True)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.gui_style_setup: {e_gen}", exc_info=True)

try:
    from puffinzip_gui import gui_layout_setup as gls_temp
    gui_layout_setup_mod = gls_temp
    init_logger.info(f"Successfully imported puffinzip_gui.gui_layout_setup: {gui_layout_setup_mod}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.gui_layout_setup: {e}", exc_info=True)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.gui_layout_setup: {e_gen}", exc_info=True)

try:
    from puffinzip_gui import chart_utils as cu_temp
    chart_utils_mod = cu_temp
    init_logger.info(f"Successfully imported puffinzip_gui.chart_utils: {chart_utils_mod}")
    if chart_utils_mod: init_logger.debug(f"    chart_utils_mod type: {type(chart_utils_mod).__name__}, MPL_Available: {getattr(chart_utils_mod, 'MATPLOTLIB_AVAILABLE', 'N/A')}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.chart_utils: {e}", exc_info=True)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.chart_utils: {e_gen}", exc_info=True)

try:
    from puffinzip_gui import settings_gui as sg_temp
    settings_gui_mod = sg_temp
    init_logger.info(f"Successfully imported puffinzip_gui.settings_gui: {settings_gui_mod}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.settings_gui: {e}", exc_info=True)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.settings_gui: {e_gen}", exc_info=True)

try:
    from puffinzip_gui import secondary_main_app as sma_temp
    secondary_main_app_mod = sma_temp
    init_logger.info(f"Successfully imported puffinzip_gui.secondary_main_app: {secondary_main_app_mod}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.secondary_main_app: {e}", exc_info=False)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.secondary_main_app: {e_gen}", exc_info=True)

try:
    from puffinzip_gui import generational_data_viewer as gdv_temp
    generational_data_viewer_mod = gdv_temp
    init_logger.info(f"Successfully imported puffinzip_gui.generational_data_viewer: {generational_data_viewer_mod}")
except ImportError as e:
    init_logger.error(f"Failed to import puffinzip_gui.generational_data_viewer: {e}", exc_info=False)
except Exception as e_gen:
    init_logger.error(f"Generic error importing puffinzip_gui.generational_data_viewer: {e_gen}", exc_info=True)

init_logger.debug("Attempting to import PuffinZipApp class from puffinzip_gui.primary_main_app...")
try:
    from puffinzip_gui.primary_main_app import PuffinZipApp as pma_temp_class
    PuffinZipApp_class = pma_temp_class
    init_logger.info(f"Successfully imported PuffinZipApp class from puffinzip_gui.primary_main_app: {PuffinZipApp_class}")
except ImportError as e:
    init_logger.error(f"Failed to import PuffinZipApp class from puffinzip_gui.primary_main_app: {e}", exc_info=True)
except Exception as e_gen:
     init_logger.error(f"Generic error importing PuffinZipApp class from puffinzip_gui.primary_main_app: {e_gen}", exc_info=True)

gui_utils = gui_utils_mod
gui_style_setup = gui_style_setup_mod
gui_layout_setup = gui_layout_setup_mod
chart_utils = chart_utils_mod
settings_gui = settings_gui_mod
secondary_main_app = secondary_main_app_mod
generational_data_viewer = generational_data_viewer_mod
PuffinZipApp = PuffinZipApp_class


__all__ = [
    name for name, obj in {
        "PuffinZipApp": PuffinZipApp,
        "chart_utils": chart_utils,
        "settings_gui": settings_gui,
        "secondary_main_app": secondary_main_app,
        "gui_utils": gui_utils,
        "gui_style_setup": gui_style_setup,
        "gui_layout_setup": gui_layout_setup,
        "generational_data_viewer": generational_data_viewer
    }.items() if obj is not None
]

init_logger.debug(f"puffinzip_gui/__init__.py execution finished. __all__: {__all__}")