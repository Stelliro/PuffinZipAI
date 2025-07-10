# PuffinZipAI_Project/run_gui.py
import sys
import os
import logging
import traceback
import time
import importlib

# --- Dependency Check ---
REQUIRED_PACKAGES = {
    "numpy": "NumPy (for data operations)",
    "matplotlib": "Matplotlib (for charting)",
    "psutil": "psutil (for system detection)"
}

missing_packages = []
for package, description in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(f"- {package} ({description})")

if missing_packages:
    error_msg = (
        "Required Python packages are missing:\n\n"
        + "\n".join(missing_packages)
        + "\n\nPlease install them by running the following command in your terminal:\n"
        "pip install -r requirements.txt"
    )
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Dependency Error", error_msg)
        root.destroy()
    except Exception: # Catch broader errors, e.g. display not found on headless system
        print("=" * 70)
        print("ERROR: MISSING DEPENDENCIES")
        print(error_msg)
        print("=" * 70)
    sys.exit(1)
# --- End Dependency Check ---


TUNED_THROTTLE_PARAMS = {}
_tuner_successful = False
_performance_tuner_logger_configured = False

try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from puffinzip_ai.utils import performance_tuner

    tuner_log = logging.getLogger("puffinzip_ai.performance_tuner")
    if not tuner_log.handlers or isinstance(tuner_log.handlers[0], logging.NullHandler):
        tuner_log.handlers.clear()
        tuner_log.setLevel(logging.INFO)
        _console_handler_tuner = logging.StreamHandler(sys.stdout)
        _formatter_tuner = logging.Formatter('%(asctime)s - TUNER - %(levelname)s - %(message)s')
        _console_handler_tuner.setFormatter(_formatter_tuner)
        tuner_log.addHandler(_console_handler_tuner)
        _performance_tuner_logger_configured = True

    TUNED_THROTTLE_PARAMS = performance_tuner.get_tuned_parameters()
    _tuner_successful = True

except ImportError as e_tuner_imp:
    pass
except Exception as e_tuner_exec:
    pass
finally:
    _default_throttles_fallback = {
        "AGENTS_PER_THROTTLE_CHECK": 5,
        "ITEMS_PER_THROTTLE_CHECK": 10,
        "THROTTLE_SLEEP_DURATION_BENCH_EVAL": 0.001,
        "RLE_THROTTLE_RUN_LENGTH_THRESHOLD": 2 * 1024 * 1024,
        "RLE_THROTTLE_CHUNK_SIZE": 512 * 1024,
        "RLE_THROTTLE_SLEEP_DURATION": 0.001,
    }
    if not _tuner_successful:
        TUNED_THROTTLE_PARAMS = _default_throttles_fallback.copy()
    else:
        for key, default_value in _default_throttles_fallback.items():
            if key not in TUNED_THROTTLE_PARAMS:
                TUNED_THROTTLE_PARAMS[key] = default_value
    if _performance_tuner_logger_configured and '_console_handler_tuner' in locals():
        if 'tuner_log' in locals() and tuner_log:
            tuner_log.removeHandler(_console_handler_tuner)
            if not tuner_log.handlers:
                tuner_log.addHandler(logging.NullHandler())

try:
    from puffinzip_gui.primary_main_app import PuffinZipApp
    from puffinzip_ai.logger import setup_logger
    from puffinzip_ai.config import LOGS_DIR_PATH, GUI_RUNNER_LOG_FILENAME

except ImportError as e:
    fallback_error_log = "run_gui_critical_import_error.log"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    error_message = (
        f"{timestamp} - FATAL ERROR in run_gui.py: Could not import essential modules.\n"
        f"Error: {e}\nTraceback:\n{traceback.format_exc()}\n"
        f"Ensure PuffinZipAI_Project is correctly structured and in PYTHONPATH, or run from root.\n")
    try:
        with open(fallback_error_log, "a", encoding="utf-8") as f_err:
            f_err.write(error_message)
    except Exception as log_ex:
        pass
    sys.exit(1)

TKINTER_AVAILABLE_FOR_FALLBACK = False
try:
    import tkinter as tk
    from tkinter import messagebox
    TKINTER_AVAILABLE_FOR_FALLBACK = True
except ImportError:
    pass

if __name__ == "__main__":
    if 'LOGS_DIR_PATH' not in locals() or LOGS_DIR_PATH is None:
        LOGS_DIR_PATH = os.path.join(os.getcwd(), "logs")

    if not os.path.exists(LOGS_DIR_PATH):
        try:
            os.makedirs(LOGS_DIR_PATH, exist_ok=True)
        except OSError as e_mkdir:
            LOGS_DIR_PATH = "."

    gui_runner_log_path = os.path.join(LOGS_DIR_PATH, GUI_RUNNER_LOG_FILENAME)

    gui_runner_logger = None
    try:
        gui_runner_logger = setup_logger(
            logger_name="PuffinZip_App_GUI_Runner",
            log_filename=gui_runner_log_path,
            log_level=logging.INFO,
            log_to_console=True,
            console_level=logging.DEBUG
        )
        gui_runner_logger.info("--- PuffinZipAI GUI Application Script Starting (Logger Active) ---")
    except Exception as e_log_setup:
        class FallbackLogger:
            def _log(self, lvl, msg, exc=False): print(
                f"FB_LOG-{lvl}: {msg}" + (f"\n{traceback.format_exc()}" if exc else ""))
            def info(self, msg): self._log("INFO", msg)
            def warning(self, msg): self._log("WARN", msg)
            def error(self, msg, exc_info=False): self._log("ERROR", msg, exc=exc_info)
            def critical(self, msg, exc_info=False): self._log("CRITICAL", msg, exc=exc_info)
            def debug(self, msg): self._log("DEBUG", msg)
        gui_runner_logger = FallbackLogger()
        gui_runner_logger.info("--- PuffinZipAI GUI Application (Fallback Print Logger due to setup error) ---")

    app = None
    try:
        gui_runner_logger.info("Initializing PuffinZipApp instance...")
        app = PuffinZipApp(tuned_params=TUNED_THROTTLE_PARAMS)
        gui_runner_logger.info("PuffinZipApp initialized. Starting mainloop...")
        app.mainloop()
    except Exception as e_app_run:
        if gui_runner_logger:
            gui_runner_logger.critical("Unhandled exception during GUI execution:", exc_info=True)
        else:
            pass
        if TKINTER_AVAILABLE_FOR_FALLBACK and (
                app is None or not (hasattr(app, 'winfo_exists') and app.winfo_exists())):
            try:
                root_err_fb = tk.Tk();
                root_err_fb.withdraw()
                messagebox.showerror("Fatal GUI Error",
                                     "A critical error occurred during application startup/runtime.\n"
                                     f"Please check logs, especially: '{os.path.abspath(gui_runner_log_path)}'.")
                root_err_fb.destroy()
            except Exception as tk_err_ex:
                if gui_runner_logger: gui_runner_logger.error(f"Failed to display Tkinter fallback: {tk_err_ex}")
    finally:
        if gui_runner_logger:
            gui_runner_logger.info("--- PuffinZipAI GUI Application Script Closed ---")
        else:
            pass