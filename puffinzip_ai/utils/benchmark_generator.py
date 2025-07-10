# PuffinZipAI_Project/puffinzip_ai/utils/benchmark_generator.py
import os
import random
import string
import time
import logging

DEFAULT_GENERATED_BENCHMARK_SUBDIR = "generated_numeric_benchmark_v1"
NUM_FILES_TO_GENERATE = 75
MIN_FILE_SIZE_KB = 2
MAX_FILE_SIZE_KB = 256
NUM_LARGE_FILES = 5
LARGE_FILE_SIZE_KB_MIN = 512
LARGE_FILE_SIZE_KB_MAX = 1024

TARGET_CHARS = string.digits + " .,\n\t"

try:
    from ..logger import setup_logger
    from ..config import LOGS_DIR_PATH, BENCHMARK_GENERATOR_LOG_FILENAME
except ImportError:
    print(
        "Warning (benchmark_generator.py): Could not import from ..logger or ..config. Using fallback paths and logger setup.")
    script_path_abs = os.path.abspath(__file__)
    utils_dir = os.path.dirname(script_path_abs)
    puffinzip_ai_dir_fallback = os.path.dirname(utils_dir)
    project_root_fallback = os.path.dirname(puffinzip_ai_dir_fallback)
    LOGS_DIR_PATH = os.path.join(project_root_fallback, "logs")
    BENCHMARK_GENERATOR_LOG_FILENAME = "benchmark_generator_standalone.log"
    def setup_logger(logger_name, log_filename, log_level=logging.INFO, log_to_console=False,
                     console_level=logging.INFO, **kwargs):
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            logger.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file_dir_fallback = os.path.dirname(log_filename)
            if log_file_dir_fallback and not os.path.exists(log_file_dir_fallback):
                os.makedirs(log_file_dir_fallback, exist_ok=True)
            fh = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            if log_to_console:
                ch = logging.StreamHandler()
                ch.setLevel(console_level)
                ch.setFormatter(formatter)
                logger.addHandler(ch)
        return logger

logger = None

def generate_long_runs(length):
    char = random.choice(string.digits)
    return char * length

def generate_sequential_numbers(length, step=1, sep=' '):
    parts = []
    current_num = random.randint(0, 1000)
    current_content_len = 0
    while current_content_len < length:
        num_str = str(current_num)
        if not parts:
            parts.append(num_str)
            current_content_len += len(num_str)
        else:
            if current_content_len + len(sep) + len(num_str) > length:
                break
            parts.append(sep)
            parts.append(num_str)
            current_content_len += len(sep) + len(num_str)
        current_num += step
        if len(parts) > length * 2: break
    return "".join(parts)[:length]

def generate_random_spaced_numbers(length, avg_spacing=5, num_len_min=1, num_len_max=5):
    content = []
    current_len = 0
    while current_len < length:
        num_len = random.randint(num_len_min, num_len_max)
        num_str = "".join(random.choice(string.digits) for _ in range(num_len))
        if current_len + len(num_str) > length:
            content.append(num_str[:length - current_len])
            current_len = length
            break
        content.append(num_str)
        current_len += len(num_str)
        if current_len >= length: break
        spacing = random.randint(1, max(1, avg_spacing * 2 - 1))
        space_str = " " * spacing
        if current_len + len(space_str) > length:
            content.append(space_str[:length - current_len])
            current_len = length
            break
        content.append(space_str)
        current_len += len(space_str)
        if len(content) > length * 1.5: break
    return "".join(content)[:length]

def generate_blocks_of_numbers(length, min_block=50, max_block=500):
    content = []
    current_len = 0
    while current_len < length:
        block_char = random.choice(string.digits)
        block_len = random.randint(min_block, max_block)
        actual_len = min(block_len, length - current_len)
        if actual_len <= 0: break
        content.append(block_char * actual_len)
        current_len += actual_len
        if current_len >= length: break
        if random.random() < 0.3:
            sep = random.choice([" ", "\n", ",", " "])
            if current_len + len(sep) > length:
                break
            content.append(sep)
            current_len += len(sep)
    return "".join(content)[:length]

def generate_mixed_numeric_content(length):
    content = []
    current_len = 0
    while current_len < length:
        rand_val = random.random()
        chunk_len = random.randint(max(1, length // 20), max(1, length // 5))
        actual_chunk_len = min(chunk_len, length - current_len)
        if actual_chunk_len <= 0: break
        chunk = ""
        if rand_val < 0.3:
            chunk = generate_long_runs(actual_chunk_len);
        elif rand_val < 0.6:
            chunk = generate_sequential_numbers(actual_chunk_len,
                                                step=random.choice([-1, 1, 2, 5]),
                                                sep=random.choice([" ", ",", "\n"]))
        elif rand_val < 0.8:
            chunk = generate_random_spaced_numbers(actual_chunk_len, avg_spacing=random.randint(1, 10))
        else:
            chunk = "".join(random.choice(TARGET_CHARS) for _ in range(actual_chunk_len))
        content.append(chunk)
        current_len = sum(len(c) for c in content)
    return "".join(content)[:length]

def create_benchmark_files(base_path, num_files, min_kb, max_kb, file_prefix="numeric_bm_"):
    global logger
    if not os.path.exists(base_path):
        try:
            os.makedirs(base_path, exist_ok=True)
            logger.info(f"Created directory: {base_path}")
        except OSError as e:
            logger.error(f"Could not create directory {base_path}: {e}")
            print(f"ERROR: Could not create directory {base_path}: {e}")
            return []
    generated_files = []
    for i in range(num_files):
        target_size_bytes = random.randint(min_kb * 1024, max_kb * 1024)
        strategy_choice = random.random()
        content = ""
        gen_type = "unknown"
        if strategy_choice < 0.20:
            content = generate_long_runs(target_size_bytes);
            gen_type = "longrun"
        elif strategy_choice < 0.40:
            content = generate_sequential_numbers(target_size_bytes,
                                                  step=random.choice([-2, -1, 1, 2, 3]),
                                                  sep=random.choice([" ", ",", "\n", ""]));
            gen_type = "seqnums"
        elif strategy_choice < 0.60:
            content = generate_random_spaced_numbers(target_size_bytes,
                                                     avg_spacing=random.randint(2, 8),
                                                     num_len_max=random.randint(3, 7));
            gen_type = "randspaced"
        elif strategy_choice < 0.80:
            content = generate_blocks_of_numbers(target_size_bytes,
                                                 min_block=max(10, target_size_bytes // 20),
                                                 max_block=max(50, target_size_bytes // 5));
            gen_type = "blocks"
        else:
            content = generate_mixed_numeric_content(target_size_bytes);
            gen_type = "mixed"
        filename = f"{file_prefix}{gen_type}_{i:03d}_{target_size_bytes // 1024}kb.txt"
        filepath = os.path.join(base_path, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            generated_files.append(filepath)
            log_msg = f"Generated: {filename} ({len(content) / 1024:.2f} KB, type: {gen_type})"
            logger.info(log_msg)
            print(log_msg)
        except Exception as e:
            err_msg = f"Error generating file {filename}: {e}"
            logger.error(err_msg)
            print(err_msg)
        time.sleep(0.005)
    return generated_files

def main_generate(output_dir_name=DEFAULT_GENERATED_BENCHMARK_SUBDIR):
    global logger
    if not os.path.exists(LOGS_DIR_PATH):
        os.makedirs(LOGS_DIR_PATH, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR_PATH, BENCHMARK_GENERATOR_LOG_FILENAME)
    logger = setup_logger(
        logger_name="PuffinBenchmarkGenerator",
        log_filename=log_file_path,
        log_level=logging.INFO,
        log_to_console=False
    )
    try:
        from ..config import BENCHMARK_DATA_DIR as cfg_benchmark_data_dir
        output_path_base = cfg_benchmark_data_dir
    except ImportError:
        script_dir_abs = os.path.dirname(os.path.abspath(__file__))
        puffinzip_ai_dir_abs = os.path.dirname(script_dir_abs)
        project_root_abs = os.path.dirname(puffinzip_ai_dir_abs)
        data_dir_abs = os.path.join(project_root_abs, "data")
        output_path_base = os.path.join(data_dir_abs, "benchmark_sets")
        logger.warning(f"Using fallback benchmark data directory: {output_path_base}")
    output_path = os.path.join(output_path_base, output_dir_name)
    start_msg = f"Starting benchmark file generation in: {os.path.abspath(output_path)}"
    logger.info(start_msg)
    print(start_msg)
    files_std = create_benchmark_files(output_path,
                                       NUM_FILES_TO_GENERATE - NUM_LARGE_FILES,
                                       MIN_FILE_SIZE_KB, MAX_FILE_SIZE_KB,
                                       file_prefix="numeric_std_")
    files_large = create_benchmark_files(output_path,
                                         NUM_LARGE_FILES,
                                         LARGE_FILE_SIZE_KB_MIN, LARGE_FILE_SIZE_KB_MAX,
                                         file_prefix="numeric_large_")
    total_generated = len(files_std) + len(files_large)
    end_msg1 = f"\nBenchmark generation complete. {total_generated} files created in {os.path.abspath(output_path)}"
    end_msg2 = f"To use these files for ELS, you might need to update BENCHMARK_DATASET_PATH in your config.py."
    end_msg3 = f"Example: BENCHMARK_DATASET_PATH = r\"{os.path.abspath(output_path)}\""
    logger.info(end_msg1)
    logger.info(end_msg2)
    logger.info(end_msg3)
    print(end_msg1)
    print(end_msg2)
    print(end_msg3)
    return os.path.abspath(output_path)

if __name__ == "__main__":
    print("--- Running Benchmark Generator Standalone ---")
    generated_path = main_generate()
    print(f"\nStandalone generation finished. Files at: {generated_path}")