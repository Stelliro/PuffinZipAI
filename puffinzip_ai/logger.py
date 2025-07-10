# PuffinZipAI_Project/puffinzip_ai/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
import sys
import importlib

try:
    from .config import LOGS_DIR_PATH, CORE_AI_LOG_FILENAME, DEFAULT_LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT

    DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG = False
    try:

        config_module_for_debug_toggle = importlib.import_module('puffinzip_ai.config')
        DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG = getattr(config_module_for_debug_toggle,
                                                               'DEBUG_LOG_CONSOLE_OUTPUT_ENABLED', False)
    except (ImportError, AttributeError):
        print(
            "Warning (logger.py): Could not access DEBUG_LOG_CONSOLE_OUTPUT_ENABLED from config. Console debug toggle might not work as expected initially.")
        DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG = False


except ImportError:
    print("Warning (logger.py): Could not import from .config. Using hardcoded defaults.")
    LOGS_DIR_PATH = "logs"
    CORE_AI_LOG_FILENAME = "puffin_ai_core_fallback.log"
    DEFAULT_LOG_LEVEL = "INFO"
    LOG_MAX_BYTES = 5 * 1024 * 1024
    LOG_BACKUP_COUNT = 3
    DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG = False

LOG_LEVEL_NAME_FROM_CONFIG = DEFAULT_LOG_LEVEL.upper()
LOG_LEVEL_ACTUAL_DEFAULT = getattr(logging, LOG_LEVEL_NAME_FROM_CONFIG, logging.INFO)


def setup_logger(logger_name='PuffinZipAI_Default',
                 log_filename=None,
                 log_level=None,
                 log_to_console=False,
                 console_level=None,
                 max_bytes=None,
                 backup_count=None,
                 log_formatter_str='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
                 date_format_str='%Y-%m-%d %H:%M:%S'):
    logger = logging.getLogger(logger_name)

    effective_log_level_for_file = log_level if log_level is not None else LOG_LEVEL_ACTUAL_DEFAULT

    user_specified_console_level = console_level

    effective_console_level_target = effective_log_level_for_file
    if user_specified_console_level is not None:
        effective_console_level_target = user_specified_console_level

    if DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG:
        if user_specified_console_level is None or user_specified_console_level > logging.DEBUG:
            effective_console_level_target = logging.DEBUG

    effective_max_bytes = max_bytes if max_bytes is not None else LOG_MAX_BYTES
    effective_backup_count = backup_count if backup_count is not None else LOG_BACKUP_COUNT

    overall_min_level_for_logger = effective_log_level_for_file
    if log_to_console:
        overall_min_level_for_logger = min(effective_log_level_for_file, effective_console_level_target)

    if not logger.handlers:
        logger.setLevel(overall_min_level_for_logger)
        logger.propagate = False

        log_formatter = logging.Formatter(log_formatter_str, datefmt=date_format_str)

        default_log_name_for_this_logger = CORE_AI_LOG_FILENAME if logger_name == 'PuffinZipAI_Core' else f"{logger_name.lower().replace(' ', '_')}.log"
        chosen_log_filename = log_filename if log_filename is not None else default_log_name_for_this_logger

        if os.path.isabs(chosen_log_filename):
            final_log_path = chosen_log_filename
        else:
            final_log_path = os.path.join(LOGS_DIR_PATH, chosen_log_filename)

        log_file_dir = os.path.dirname(final_log_path)
        if log_file_dir and not os.path.exists(log_file_dir):
            try:
                os.makedirs(log_file_dir, exist_ok=True)
            except OSError as e_dir:
                final_log_path = os.path.basename(final_log_path)

        try:
            file_handler = RotatingFileHandler(final_log_path, maxBytes=effective_max_bytes,
                                               backupCount=effective_backup_count, encoding='utf-8')
            file_handler.setLevel(effective_log_level_for_file)
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
        except Exception as e_fh:
            log_to_console = True
            effective_console_level_target = min(effective_console_level_target, effective_log_level_for_file)

        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(effective_console_level_target)
            console_handler.setFormatter(log_formatter)
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                logger.addHandler(console_handler)

        if not logger.handlers:
            fallback_handler = logging.StreamHandler(sys.stderr)
            fallback_handler.setFormatter(log_formatter)
            logger.addHandler(fallback_handler)
            logger.setLevel(logging.WARNING)


    elif log_to_console:
        console_handler_exists = False
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream in [sys.stdout, sys.stderr]:
                console_handler_exists = True
                if h.level != effective_console_level_target:
                    h.setLevel(effective_console_level_target)
                break
        if not console_handler_exists:
            ch_new = logging.StreamHandler(sys.stdout)
            ch_new.setLevel(effective_console_level_target)
            ch_new.setFormatter(
                logger.handlers[0].formatter if logger.handlers and logger.handlers[0].formatter else logging.Formatter(
                    log_formatter_str, datefmt=date_format_str))
            logger.addHandler(ch_new)

    current_min_level_on_logger_obj = logger.getEffectiveLevel()
    new_overall_min_level_for_logger = effective_log_level_for_file
    if log_to_console and any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        new_overall_min_level_for_logger = min(effective_log_level_for_file, effective_console_level_target)

    if current_min_level_on_logger_obj != new_overall_min_level_for_logger:
        logger.setLevel(new_overall_min_level_for_logger)

    return logger


if __name__ == '__main__':
    if not os.path.exists(LOGS_DIR_PATH):
        try:
            os.makedirs(LOGS_DIR_PATH, exist_ok=True)
        except OSError as e:
            LOGS_DIR_PATH = ".";

    logger1_nodbg = setup_logger("TestLogger1_NoConsoleDbg", log_filename="test1_noconsdebug.log",
                                 log_level=logging.DEBUG, log_to_console=True, console_level=logging.INFO)
    logger1_nodbg.debug("TestLogger1_NoConsoleDbg: This is a debug message (should be file only).")
    logger1_nodbg.info("TestLogger1_NoConsoleDbg: This is an info message (should be file and console).")

    print(f"\n--- Simulating DEBUG_LOG_CONSOLE_OUTPUT_ENABLED = True (by overriding for this test) ---")
    DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG_original = DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG
    DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG = True

    logger1_withdbg = setup_logger("TestLogger1_WithConsoleDbg", log_filename="test1_withconsdebug.log",
                                   log_level=logging.DEBUG, log_to_console=True,
                                   console_level=logging.INFO)  # console_level INFO initially
    logger1_withdbg.debug(
        "TestLogger1_WithConsoleDbg: This is a debug message (should be file AND CONSOLE due to override).")
    logger1_withdbg.info("TestLogger1_WithConsoleDbg: This is an info message (file and console).")

    DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG = DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG_original  # Restore
    print(
        f"--- Restored DEBUG_LOG_CONSOLE_OUTPUT_ENABLED to original ({DEBUG_LOG_CONSOLE_OUTPUT_ENABLED_FROM_CONFIG}) ---")

    core_logger = setup_logger("PuffinZipAI_Core", log_to_console=True, console_level=logging.DEBUG)

    another_logger = setup_logger("AnotherTest", log_to_console=True, log_level=logging.WARNING,
                                  console_level=logging.ERROR)  # Here console_level=ERROR explicitly


