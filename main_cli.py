import logging
import os
import sys
import threading
import traceback

try:
    from puffinzip_ai import PuffinZipAI, setup_logger
    from puffinzip_ai.config import (
        MODEL_FILE_DEFAULT,
        COMPRESSED_FILE_SUFFIX,
        DEFAULT_LEN_THRESHOLDS,
        DEFAULT_BATCH_COMPRESS_EXTENSIONS,
        DEFAULT_ALLOWED_LEARN_EXTENSIONS,
        DEFAULT_TRAIN_BATCH_SIZE,
        DEFAULT_FOLDER_LEARN_BATCH_SIZE,
        APP_VERSION,
        LOGS_DIR_PATH,
        CLI_MENU_LOG_FILENAME,
        CORE_AI_LOG_FILENAME
    )
except ImportError as e:
    print(f"FATAL ERROR (main_cli.py): Could not import core components. Exception: {e}")
    print("Ensure 'puffinzip_ai' package is correctly installed and in PYTHONPATH, or run from project root.")
    sys.exit(1)

def print_header(title="PuffinZipAI CLI"):
    print("\n" + "=" * 50)
    print(f"{title:^50}")
    print("=" * 50)

def get_user_input(prompt, default=None, type_cast=str, allow_empty_for_default=True):
    while True:
        full_prompt = f"{prompt} "
        if default is not None:
            full_prompt += f"[{default}]: "
        else:
            full_prompt += ": "
        user_str = input(full_prompt).strip()
        if not user_str and default is not None and allow_empty_for_default:
            return default
        if not user_str and not allow_empty_for_default and default is None:
            print("Input cannot be empty.")
            continue
        try:
            if type_cast == bool:
                if user_str.lower() in ['y', 'yes', 'true', '1']: return True
                if user_str.lower() in ['n', 'no', 'false', '0']: return False
                raise ValueError("Invalid boolean input.")
            elif type_cast == list_str_parser:
                return list_str_parser(user_str)
            elif type_cast == list_int_parser:
                return list_int_parser(user_str)
            val = type_cast(user_str)
            if type_cast == int and val <= 0 and prompt.lower().count(
                    'batch size') == 0 and prompt.lower().count("items") == 0 and prompt.lower().count("episodes") == 0:
                if not ("delay" in prompt.lower() or "threshold" in prompt.lower() and val == 0):
                    print("Integer value must be positive for this input.")
                    continue
            return val
        except ValueError:
            print(f"Invalid input. Please enter a value of type '{type_cast.__name__}'.")
        except Exception as e:
            print(f"An unexpected error occurred with your input: {e}")

def list_str_parser(s, delimiter=','):
    if not s: return []
    return [item.strip() for item in s.split(delimiter) if item.strip()]

def list_int_parser(s, delimiter=','):
    if not s: return []
    str_list = list_str_parser(s, delimiter)
    int_list = []
    for item in str_list:
        try:
            int_list.append(int(item))
        except ValueError:
            raise ValueError(f"Invalid integer '{item}' found in list.")
    return int_list

def display_status(message):
    print(f"\n>>> {message} <<<\n")

def train_random_cli(ai_agent, cli_logger):
    print_header("Train AI with Random Data")
    cli_logger.info("CLI: Train with random data initiated.")
    continuous_str = get_user_input("Run continuously? (y/n or c for continuous)", default='n', type_cast=str).lower()
    run_continuously = continuous_str == 'y' or continuous_str == 'c'
    num_items_str = "N/A"
    if not run_continuously:
        num_items_str = get_user_input(f"Number of random items to train on (e.g., {DEFAULT_TRAIN_BATCH_SIZE * 10})", default=str(DEFAULT_TRAIN_BATCH_SIZE * 10), type_cast=str)
    batch_size_str = get_user_input("Batch size for random training", default=str(DEFAULT_TRAIN_BATCH_SIZE), type_cast=str)
    batch_size = None
    if batch_size_str.isdigit() and int(batch_size_str) > 0:
        batch_size = int(batch_size_str)
    else:
        cli_logger.warning(f"Invalid batch size '{batch_size_str}', using AI default.")
    display_status("Starting random training... (Check core AI log for detailed progress)")
    if run_continuously:
        ai_agent.train(run_continuously=True, batch_size=batch_size)
    else:
        try:
            num_items = int(num_items_str)
            if num_items <= 0:
                display_status("Number of items must be positive.")
                return
            ai_agent.train(num_episodes=num_items, batch_size=batch_size)
        except ValueError:
            display_status(f"Invalid number of items: {num_items_str}")
            cli_logger.error(f"Invalid number of items entered for random training: {num_items_str}")

def train_folder_cli(ai_agent, cli_logger):
    print_header("Train AI with Data from Folder")
    cli_logger.info("CLI: Train from folder initiated.")
    folder_path = get_user_input("Enter path to folder with training files", type_cast=str, allow_empty_for_default=False)
    if not os.path.isdir(folder_path):
        display_status(f"Error: Folder '{folder_path}' not found.")
        cli_logger.error(f"Training folder not found: {folder_path}")
        return
    allowed_exts_str = get_user_input("Allowed file extensions (comma-separated, e.g., .txt,.md)", default=",".join(DEFAULT_ALLOWED_LEARN_EXTENSIONS))
    allowed_extensions = list_str_parser(allowed_exts_str)
    run_continuously = get_user_input("Run continuously on folder (monitor for new files)? (y/n)", default='n', type_cast=bool)
    batch_size_str = get_user_input("Batch size for folder training", default=str(DEFAULT_FOLDER_LEARN_BATCH_SIZE), type_cast=str)
    batch_size = None
    if batch_size_str.isdigit() and int(batch_size_str) > 0:
        batch_size = int(batch_size_str)
    else:
        cli_logger.warning(f"Invalid batch size '{batch_size_str}' for folder training, using AI default.")
    display_status(f"Starting folder training on '{folder_path}'... (Check core AI log)")
    ai_agent.learn_from_folder(folder_path, allowed_extensions=allowed_extensions, run_continuously=run_continuously, batch_size=batch_size)

def batch_compress_cli(ai_agent, cli_logger):
    print_header("Batch Compress Folder")
    cli_logger.info("CLI: Batch compress folder initiated.")
    input_folder = get_user_input("Enter path to input folder to compress", allow_empty_for_default=False)
    if not os.path.isdir(input_folder):
        display_status(f"Error: Input folder '{input_folder}' not found."); cli_logger.error(f"Batch compress input folder not found: {input_folder}"); return
    output_folder = get_user_input("Enter path to output folder for compressed files", allow_empty_for_default=False)
    if not output_folder: display_status("Output folder cannot be empty."); cli_logger.error("Batch compress output folder empty."); return
    if input_folder == output_folder: display_status("Input and output folders must be different."); cli_logger.error("Batch compress I/O folders same."); return
    allowed_exts_str = get_user_input("Allowed file extensions (comma-separated, e.g., .txt,.log)", default=",".join(DEFAULT_BATCH_COMPRESS_EXTENSIONS))
    allowed_extensions = list_str_parser(allowed_exts_str)
    display_status(f"Starting batch compression from '{input_folder}' to '{output_folder}'...")
    ai_agent.batch_compress_folder(input_folder, output_folder, allowed_extensions=allowed_extensions)

def batch_decompress_cli(ai_agent, cli_logger):
    print_header("Batch Decompress Folder")
    cli_logger.info("CLI: Batch decompress folder initiated.")
    input_folder = get_user_input(f"Enter path to input folder with compressed files (e.g., *{COMPRESSED_FILE_SUFFIX})", allow_empty_for_default=False)
    if not os.path.isdir(input_folder):
        display_status(f"Error: Input folder '{input_folder}' not found."); cli_logger.error(f"Batch decompress input folder not found: {input_folder}"); return
    output_folder = get_user_input("Enter path to output folder for decompressed files", allow_empty_for_default=False)
    if not output_folder: display_status("Output folder cannot be empty."); cli_logger.error("Batch decompress output folder empty."); return
    if input_folder == output_folder: display_status("Input and output folders must be different."); cli_logger.error("Batch decompress I/O folders same."); return
    display_status(f"Starting batch decompression from '{input_folder}' to '{output_folder}'...")
    ai_agent.batch_decompress_folder(input_folder, output_folder)

def single_item_cli(ai_agent, cli_logger):
    print_header("Process Single Item")
    cli_logger.info("CLI: Single item processing initiated.")
    item_text = get_user_input("Enter text to process (or 'file:path/to/file.txt' to load from file)", allow_empty_for_default=False)
    if item_text.lower().startswith("file:"):
        filepath = item_text[5:].strip()
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    item_text = f.read()
                display_status(f"Loaded text from '{filepath}'. Length: {len(item_text)}")
            except Exception as e:
                display_status(f"Error reading file '{filepath}': {e}"); cli_logger.error(f"Error reading file {filepath}: {e}"); return
        else:
            display_status(f"Error: File '{filepath}' not found."); cli_logger.error(f"File not found for single item: {filepath}"); return
    operation = get_user_input("Choose operation: (c)ompress with AI, (d)ecompress RLE", default='c').lower()
    if operation in ['c', 'compress']:
        ai_agent.compress_user_item(item_text)
    elif operation in ['d', 'decompress']:
        ai_agent.decompress_user_item_rle(item_text)
    else:
        display_status("Invalid operation selected.")
        cli_logger.warning(f"Invalid single item operation: {operation}")

def configure_ai_cli(ai_agent, cli_logger):
    print_header("Configure AI Parameters")
    cli_logger.info("CLI: Configure AI initiated.")
    print(f"Current length thresholds: {ai_agent.len_thresholds}")
    thresh_str = get_user_input(f"New length thresholds (comma-sep positive integers, blank to keep current)", default=",".join(map(str, ai_agent.len_thresholds)))
    try:
        new_thresholds = list_int_parser(thresh_str)
        if any(t <= 0 for t in new_thresholds) and new_thresholds:
            display_status("Thresholds must be positive integers."); return
        ai_agent.configure_data_categories(new_thresholds if new_thresholds else ai_agent.len_thresholds)
    except ValueError as e:
        display_status(f"Invalid threshold format: {e}"); return
    print(f"Current inter-batch delay: {ai_agent.inter_batch_delay_seconds}s")
    delay_str = get_user_input("New inter-batch delay in seconds (e.g., 0.5, blank to keep)", default=str(ai_agent.inter_batch_delay_seconds))
    ai_agent.configure_inter_batch_delay(delay_str)
    display_status("AI parameters updated if valid inputs were provided.")

def manage_model_cli(ai_agent, cli_logger):
    print_header("Manage AI Model")
    cli_logger.info("CLI: Manage model initiated.")
    action = get_user_input("Action: (s)ave model, (l)oad model, (q) Q-table summary, (t) test agent", default='q').lower()
    if action in ['s', 'save']:
        filepath = get_user_input("Enter filepath to save model", default=MODEL_FILE_DEFAULT)
        ai_agent.save_model(filepath)
    elif action in ['l', 'load']:
        filepath = get_user_input("Enter filepath to load model from", default=MODEL_FILE_DEFAULT)
        if os.path.exists(filepath):
            ai_agent.load_model(filepath)
        else:
            display_status(f"Error: Model file '{filepath}' not found.")
            cli_logger.error(f"Model file for loading not found: {filepath}")
    elif action in ['q', 'summary']:
        ai_agent.display_q_table_summary()
    elif action in ['t', 'test']:
        num_items = get_user_input("Number of random items to test agent on", default="10", type_cast=int)
        if num_items > 0:
            ai_agent.test_agent_on_random_items(num_items)
        else:
            display_status("Number of test items must be positive.")
    else:
        display_status("Invalid model management action.")
        cli_logger.warning(f"Invalid model management action: {action}")

def main_menu(ai_agent, cli_logger):
    print(f"\nPuffinZipAI CLI v{APP_VERSION} - Using model: '{os.path.basename(MODEL_FILE_DEFAULT)}'")
    while True:
        print_header("PuffinZipAI Main Menu")
        print("1. Train AI - Random Data")
        print("2. Train AI - From Folder")
        print("3. Batch Compress Folder")
        print("4. Batch Decompress Folder")
        print("5. Process Single Item (Compress/Decompress)")
        print("6. Configure AI Parameters")
        print("7. Manage Model (Save/Load/Summary/Test)")
        print("8. Exit")
        choice = get_user_input("Enter your choice (1-8)", type_cast=str)
        if choice == '1': train_random_cli(ai_agent, cli_logger)
        elif choice == '2': train_folder_cli(ai_agent, cli_logger)
        elif choice == '3': batch_compress_cli(ai_agent, cli_logger)
        elif choice == '4': batch_decompress_cli(ai_agent, cli_logger)
        elif choice == '5': single_item_cli(ai_agent, cli_logger)
        elif choice == '6': configure_ai_cli(ai_agent, cli_logger)
        elif choice == '7': manage_model_cli(ai_agent, cli_logger)
        elif choice == '8': display_status("Exiting PuffinZipAI CLI."); cli_logger.info("CLI: Exiting."); break
        else: display_status("Invalid choice. Please try again.")
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    if not os.path.exists(LOGS_DIR_PATH):
        try:
            os.makedirs(LOGS_DIR_PATH, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create logs directory '{LOGS_DIR_PATH}': {e}. CLI Log will be in current directory.")
            LOGS_DIR_PATH = "."
    cli_log_file_path = os.path.join(LOGS_DIR_PATH, CLI_MENU_LOG_FILENAME)
    cli_logger = setup_logger(logger_name='PuffinZipAI_CLI_Menu', log_filename=cli_log_file_path, log_level=logging.INFO, log_to_console=False)
    cli_logger.info("--- PuffinZipAI CLI Application Starting ---")
    display_status("Initializing PuffinZipAI agent...")
    try:
        puffin_ai_agent = PuffinZipAI()
        if not puffin_ai_agent.load_model():
            display_status("Default model not found or failed to load. Initializing a new model.")
            cli_logger.warning("Default model could not be loaded. New model created.")
        else:
            display_status(f"Successfully loaded default model: {MODEL_FILE_DEFAULT}")
            cli_logger.info(f"Default model loaded: {MODEL_FILE_DEFAULT}")
        puffin_ai_agent.gui_output_queue = None
        puffin_ai_agent.gui_stop_event = threading.Event()
        main_menu(puffin_ai_agent, cli_logger)
    except Exception as e:
        cli_logger.critical("Unhandled exception during CLI execution:", exc_info=True)
        print("\n--- FATAL CLI ERROR ---")
        traceback.print_exc()
        print("-----------------------")
        print("An unexpected error occurred. The CLI application will close.")
        print(f"Please check logs in '{os.path.abspath(LOGS_DIR_PATH)}' for details,")
        print(f"specifically '{CLI_MENU_LOG_FILENAME}' and '{CORE_AI_LOG_FILENAME}'.")
    finally:
        cli_logger.info("--- PuffinZipAI CLI Application Closed ---")