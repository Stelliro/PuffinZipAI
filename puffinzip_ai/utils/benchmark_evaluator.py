# PuffinZipAI_Project/puffinzip_ai/utils/benchmark_evaluator.py
import os
import time
import random
import numpy as np
import json
import logging
import traceback
from enum import Enum
import string

_PuffinZipAI_cls = None
_rle_compress_func, _rle_decompress_func = None, None
_RLE_DECOMPRESSION_ERRORS_set = set()
_calculate_reward_func = None
_GENERATED_BENCHMARK_DEFAULT_PATH_val = "./fallback_benchmark_data_dir"
_BENCHMARK_DATA_DIR_val, _DEFAULT_LEARNING_RATE_val, _DEFAULT_EXPLORATION_RATE_val = None, 0.1, 1.0
_DEFAULT_DISCOUNT_FACTOR_val, _DEFAULT_EXPLORATION_DECAY_RATE_val, _DEFAULT_MIN_EXPLORATION_RATE_val = 0.9, 0.999, 0.01
_RLE_MIN_RUN_INIT_MIN_val, _RLE_MIN_RUN_INIT_MAX_val = 2, 4
_setup_logger_func_val = lambda *args, **kwargs: logging.getLogger("BenchmarkEvaluator_Fallback_Setup")

try:
    from ..ai_core import PuffinZipAI

    _PuffinZipAI_cls = PuffinZipAI
    from ..rle_utils import rle_compress, rle_decompress

    _rle_compress_func, _rle_decompress_func = rle_compress, rle_decompress
    from ..rle_constants import RLE_DECOMPRESSION_ERRORS

    _RLE_DECOMPRESSION_ERRORS_set = RLE_DECOMPRESSION_ERRORS
    from ..reward_system import calculate_reward

    _calculate_reward_func = calculate_reward
    from ..config import (
        GENERATED_BENCHMARK_DEFAULT_PATH,
        BENCHMARK_DATA_DIR, DEFAULT_LEARNING_RATE, DEFAULT_EXPLORATION_RATE,
        DEFAULT_DISCOUNT_FACTOR, DEFAULT_EXPLORATION_DECAY_RATE, DEFAULT_MIN_EXPLORATION_RATE,
        RLE_MIN_RUN_INIT_MIN, RLE_MIN_RUN_INIT_MAX
    )

    _GENERATED_BENCHMARK_DEFAULT_PATH_val = GENERATED_BENCHMARK_DEFAULT_PATH
    _BENCHMARK_DATA_DIR_val = BENCHMARK_DATA_DIR
    _DEFAULT_LEARNING_RATE_val = DEFAULT_LEARNING_RATE
    _DEFAULT_EXPLORATION_RATE_val = DEFAULT_EXPLORATION_RATE
    _DEFAULT_DISCOUNT_FACTOR_val = DEFAULT_DISCOUNT_FACTOR
    _DEFAULT_EXPLORATION_DECAY_RATE_val = DEFAULT_EXPLORATION_DECAY_RATE
    _DEFAULT_MIN_EXPLORATION_RATE_val = DEFAULT_MIN_EXPLORATION_RATE
    _RLE_MIN_RUN_INIT_MIN_val = RLE_MIN_RUN_INIT_MIN
    _RLE_MIN_RUN_INIT_MAX_val = RLE_MIN_RUN_INIT_MAX
    from ..logger import setup_logger

    _setup_logger_func_val = setup_logger
except ImportError as e_be_imp:
    _fallback_logger_be = logging.getLogger("BenchmarkEvaluator_ImportError")
    if not _fallback_logger_be.handlers:
        _h = logging.StreamHandler();
        _f = logging.Formatter('%(asctime)s - BE_ImportERR - %(levelname)s - %(message)s');
        _h.setFormatter(_f)
        _fallback_logger_be.addHandler(_h);
        _fallback_logger_be.setLevel(logging.WARNING)
    _fallback_logger_be.critical(
        f"CRITICAL ERROR (benchmark_evaluator.py): Failed to import core components. Error: {e_be_imp}", exc_info=True)

DEFAULT_BENCHMARK_REPETITIONS = 1
DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET = 30
EVALUATION_FAIL_REWARD = -100.0
EVALUATION_TIMEOUT_REWARD_PENALTY = -50.0
MAX_ITEM_PROCESS_TIME_SEC = 20.0

# Optimized default throttle settings
AGENTS_PER_THROTTLE_CHECK = 5
ITEMS_PER_THROTTLE_CHECK = 50  # Increased from 10 to 50 to reduce sleep overhead
THROTTLE_SLEEP_DURATION_BENCH_EVAL = 0.001


class DataComplexity(Enum):
    VERY_SIMPLE = 0
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4
    USER_DEFINED_LARGE = 5

    @classmethod
    def get_member_names(cls):
        return [member.name for member in cls]


COMPLEXITY_FITNESS_THRESHOLDS = {
    DataComplexity.SIMPLE: -5.0,
    DataComplexity.MODERATE: -2.0,
    DataComplexity.COMPLEX: 0.2,
    DataComplexity.VERY_COMPLEX: 0.75
}

COMPLEXITY_LENGTH_RANGES_BYTES = {
    DataComplexity.VERY_SIMPLE: (200 * 1024, 500 * 1024),
    DataComplexity.SIMPLE: (500 * 1024, 5 * 1024 * 1024),
    DataComplexity.MODERATE: (5 * 1024 * 1024, 50 * 1024 * 1024),
    DataComplexity.COMPLEX: (50 * 1024 * 1024, 150 * 1024 * 1024),
    DataComplexity.VERY_COMPLEX: (150 * 1024 * 1024, 300 * 1024 * 1024)
}


class BenchmarkItemEvaluator:
    def __init__(self, benchmark_dataset_path=None, logger_instance=None, tuned_params=None, dynamic_benchmarking=True):
        self.benchmark_dataset_path = benchmark_dataset_path
        self.benchmark_items = []
        self.logger = logger_instance if logger_instance else _setup_logger_func_val("BenchmarkEvaluator",
                                                                                     log_level=logging.INFO)
        self.tuned_params = tuned_params if tuned_params is not None else {}
        self.agents_per_throttle_check = self.tuned_params.get("AGENTS_PER_THROTTLE_CHECK", AGENTS_PER_THROTTLE_CHECK)
        self.items_per_throttle_check = self.tuned_params.get("ITEMS_PER_THROTTLE_CHECK", ITEMS_PER_THROTTLE_CHECK)
        self.throttle_sleep_duration = self.tuned_params.get("THROTTLE_SLEEP_DURATION_BENCH_EVAL",
                                                             THROTTLE_SLEEP_DURATION_BENCH_EVAL)
        self.dynamic_benchmarking_enabled = dynamic_benchmarking
        self._temp_agent_for_generation = None
        self.logger.info(
            f"BenchmarkItemEvaluator initialized. Dynamic Benchmarking: {self.dynamic_benchmarking_enabled}.")
        if not self.dynamic_benchmarking_enabled and self.benchmark_dataset_path:
            self.load_benchmark_data(self.benchmark_dataset_path)
        elif self.dynamic_benchmarking_enabled:
            self.logger.info("Dynamic benchmarking enabled. Initial items will be generated on demand.")

    def _get_temp_generation_agent(self):
        if self._temp_agent_for_generation is None and _PuffinZipAI_cls is not None:
            try:
                self._temp_agent_for_generation = _PuffinZipAI_cls(
                    len_thresholds=None, learning_rate=_DEFAULT_LEARNING_RATE_val,
                    exploration_rate=_DEFAULT_EXPLORATION_RATE_val,
                    discount_factor=_DEFAULT_DISCOUNT_FACTOR_val,
                    exploration_decay_rate=_DEFAULT_EXPLORATION_DECAY_RATE_val,
                    min_exploration_rate=_DEFAULT_MIN_EXPLORATION_RATE_val,
                    rle_min_encodable_run=random.randint(_RLE_MIN_RUN_INIT_MIN_val, _RLE_MIN_RUN_INIT_MAX_val)
                )
            except Exception as e_temp_agent:
                self.logger.error(f"Failed to create temp PuffinZipAI for item generation: {e_temp_agent}")
        return self._temp_agent_for_generation

    def _generate_one_dynamic_item(self, complexity_level: DataComplexity,
                                   target_size_bytes_override: int = None) -> str:
        agent = self._get_temp_generation_agent()
        min_l, max_l = COMPLEXITY_LENGTH_RANGES_BYTES.get(complexity_level,
                                                          COMPLEXITY_LENGTH_RANGES_BYTES[DataComplexity.SIMPLE])
        run_likelihood = 0.3
        unique_focus = 0.5

        if target_size_bytes_override is not None and target_size_bytes_override > 0:
            variance_factor = 0.15
            min_l = int(target_size_bytes_override * (1 - variance_factor))
            max_l = int(target_size_bytes_override * (1 + variance_factor))
            min_l = max(10 * 1024, min_l)
            max_l = max(min_l + (10 * 1024), max_l)
            if target_size_bytes_override > 100 * 1024 * 1024:
                run_likelihood = random.uniform(0.15, 0.45)
                unique_focus = random.uniform(0.5, 0.8)
            elif target_size_bytes_override > 10 * 1024 * 1024:
                run_likelihood = random.uniform(0.2, 0.5)
                unique_focus = random.uniform(0.4, 0.7)
            self.logger.debug(
                f"Generating item with target size override: approx {target_size_bytes_override}B. Actual range: {min_l}-{max_l}B")
        else:
            if DataComplexity and complexity_level == DataComplexity.VERY_SIMPLE:
                run_likelihood = random.uniform(0.5, 0.7);
                unique_focus = random.uniform(0.2, 0.4)
            elif DataComplexity and complexity_level == DataComplexity.SIMPLE:
                run_likelihood = random.uniform(0.4, 0.6);
                unique_focus = random.uniform(0.3, 0.5)
            elif DataComplexity and complexity_level == DataComplexity.MODERATE:
                run_likelihood = random.uniform(0.3, 0.5);
                unique_focus = random.uniform(0.4, 0.6)
            elif DataComplexity and complexity_level == DataComplexity.COMPLEX:
                run_likelihood = random.uniform(0.2, 0.4);
                unique_focus = random.uniform(0.5, 0.7)
            elif DataComplexity and complexity_level == DataComplexity.VERY_COMPLEX:
                run_likelihood = random.uniform(0.15, 0.35);
                unique_focus = random.uniform(0.6, 0.8)

        length = max(1, random.randint(min_l, max_l))
        if length > 10 * 1024 * 1024:
            self.logger.info(
                f"Starting generation of a large item: complexity={getattr(complexity_level, 'name', 'UNKNOWN')}, approx_size={length / (1024 * 1024):.2f}MB")

        item_content = ""
        try:
            # Use agent generation if available, otherwise fast fallback
            if agent and hasattr(agent, '_generate_random_item'):
                item_content = agent._generate_random_item(min_len=length, max_len=length,
                                                           run_likelihood_factor=run_likelihood,
                                                           unique_char_focus_factor=unique_focus)
            else:
                raise AttributeError("Agent generation unavailable")
        except Exception as e_gen_item:
            # OPTIMIZED FALLBACK using random.choices and join
            # This is significantly faster than repeated string multiplication/concatenation logic
            pool = string.ascii_letters + string.digits + " ._-\n"
            item_content = ''.join(random.choices(pool, k=length))
            # self.logger.debug(f"Used fast fallback generation due to: {e_gen_item}")

        if length > 10 * 1024 * 1024:
            self.logger.info(
                f"Finished generation of large item. Actual size: {len(item_content) / (1024 * 1024):.2f}MB")
        return item_content

    def determine_target_complexity(self, population_average_fitness: float) -> DataComplexity:
        if not DataComplexity: return type('MockDataComplexity', (),
                                           {'VERY_SIMPLE': 0, 'SIMPLE': 1, 'MODERATE': 2, 'COMPLEX': 3,
                                            'VERY_COMPLEX': 4})()
        if population_average_fitness >= COMPLEXITY_FITNESS_THRESHOLDS.get(DataComplexity.VERY_COMPLEX, 0.75):
            return DataComplexity.VERY_COMPLEX
        elif population_average_fitness >= COMPLEXITY_FITNESS_THRESHOLDS.get(DataComplexity.COMPLEX, 0.2):
            return DataComplexity.COMPLEX
        elif population_average_fitness >= COMPLEXITY_FITNESS_THRESHOLDS.get(DataComplexity.MODERATE, -2.0):
            return DataComplexity.MODERATE
        elif population_average_fitness >= COMPLEXITY_FITNESS_THRESHOLDS.get(DataComplexity.SIMPLE, -5.0):
            return DataComplexity.SIMPLE
        else:
            return DataComplexity.VERY_SIMPLE

    def generate_and_set_dynamic_benchmark_items(self, num_items_to_generate: int = DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET,
                                                 population_average_fitness: float = -100.0,
                                                 current_generation: int = 0,
                                                 target_item_size_mb_override: float = None,
                                                 fixed_complexity_override_name: str = None):
        if not self.dynamic_benchmarking_enabled:
            self.logger.info("Dynamic benchmarking is disabled. No new items generated.")
            if not self.benchmark_items and self.benchmark_dataset_path: self.load_benchmark_data()
            if not self.benchmark_items: self.benchmark_items = ["Fallback AAA", "Fallback BBBCCC",
                                                                 "Fallback DDDDEEEEFFFF"]
            return bool(self.benchmark_items)

        target_size_bytes_final = None
        target_complexity_for_generation = DataComplexity.SIMPLE if DataComplexity else "SIMPLE"

        if target_item_size_mb_override is not None and target_item_size_mb_override > 0:
            target_size_bytes_final = int(target_item_size_mb_override * 1024 * 1024)
            target_complexity_for_generation = DataComplexity.USER_DEFINED_LARGE if DataComplexity else "USER_DEFINED_LARGE"
            self.logger.info(
                f"Generating {num_items_to_generate} new dynamic benchmark items. User override: Target Avg Size: {target_item_size_mb_override:.2f} MB (~{target_size_bytes_final} bytes per item).")
        elif fixed_complexity_override_name and DataComplexity:
            try:
                target_complexity_for_generation = DataComplexity[fixed_complexity_override_name.upper()]
                self.logger.info(
                    f"Generating {num_items_to_generate} new dynamic benchmark items. User override: Fixed Complexity: {target_complexity_for_generation.name}")
            except KeyError:
                self.logger.warning(
                    f"Invalid fixed_complexity_override_name '{fixed_complexity_override_name}'. Falling back to fitness-adaptive complexity.")
                target_complexity_for_generation = self.determine_target_complexity(population_average_fitness)
        else:
            target_complexity_for_generation = self.determine_target_complexity(population_average_fitness)
            if DataComplexity and current_generation > 0 and current_generation % 20 == 0:
                current_complexity_val = getattr(target_complexity_for_generation, 'value', DataComplexity.SIMPLE.value)
                if current_complexity_val < getattr(DataComplexity.VERY_COMPLEX, 'value',
                                                    DataComplexity.SIMPLE.value + 3):
                    try:
                        nudged_complexity = DataComplexity(current_complexity_val + 1)
                    except ValueError:
                        nudged_complexity = target_complexity_for_generation

                    if getattr(nudged_complexity, 'value', current_complexity_val) > current_complexity_val:
                        target_complexity_for_generation = nudged_complexity
                        self.logger.info(
                            f"Gen {current_generation}: Nudging benchmark complexity up to {getattr(target_complexity_for_generation, 'name', 'UNKNOWN')} due to generation progress.")
            self.logger.info(
                f"Generating {num_items_to_generate} new dynamic benchmark items. Target Complexity: {getattr(target_complexity_for_generation, 'name', 'UNKNOWN')} (based on AvgFit: {population_average_fitness:.3f}, Gen: {current_generation}).")

        new_items = [self._generate_one_dynamic_item(target_complexity_for_generation, target_size_bytes_final) for _ in
                     range(num_items_to_generate)]
        self.benchmark_items = new_items
        total_size_mb = sum(len(item) for item in self.benchmark_items) / (1024 * 1024)
        avg_size_mb = total_size_mb / len(self.benchmark_items) if self.benchmark_items else 0
        self.logger.info(
            f"Dynamically generated and set {len(self.benchmark_items)} benchmark items. Total size: {total_size_mb:.2f} MB, Avg size: {avg_size_mb:.2f} MB.")
        return bool(self.benchmark_items)

    def load_benchmark_data(self, dataset_path=None):
        path_to_load = dataset_path if dataset_path else self.benchmark_dataset_path
        self.benchmark_items = []
        self.logger.info(f"Attempting to load benchmark data from: {path_to_load}")
        if not (path_to_load and os.path.exists(path_to_load) and os.path.isdir(path_to_load)):
            self.logger.warning(f"Static benchmark dataset path '{path_to_load}' not valid. No data loaded.")
            return False
        max_items_to_load = DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET if self.dynamic_benchmarking_enabled else 100
        loaded_count = 0
        for filename in os.listdir(path_to_load):
            if filename.endswith(".json"):
                filepath = os.path.join(path_to_load, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list) and all(isinstance(item, dict) and "content" in item for item in data):
                        self.benchmark_items.extend(
                            [item['content'] for item in data if isinstance(item.get('content'), str)])
                        loaded_count += len(data)
                    elif isinstance(data, dict) and "content" in data and isinstance(data['content'], str):
                        self.benchmark_items.append(data['content'])
                        loaded_count += 1
                except Exception as e:
                    self.logger.error(f"Error loading static benchmark file '{filename}': {e}", exc_info=True)
            elif filename.lower().endswith((".txt", ".log", ".md", ".csv")):
                filepath = os.path.join(path_to_load, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        item_content = f.read()
                    if item_content.strip(): self.benchmark_items.append(item_content); loaded_count += 1
                except Exception as e_txt:
                    self.logger.error(f"Error loading static text benchmark file '{filename}': {e_txt}", exc_info=True)
            if loaded_count >= max_items_to_load: break

        self.benchmark_items = self.benchmark_items[:max_items_to_load]
        if not self.benchmark_items:
            self.logger.warning(f"No valid static benchmark items loaded from '{path_to_load}'.")
        else:
            total_size_mb = sum(len(item) for item in self.benchmark_items) / (1024 * 1024)
            avg_size_mb = total_size_mb / len(self.benchmark_items) if self.benchmark_items else 0
            self.logger.info(
                f"Successfully loaded {len(self.benchmark_items)} static benchmark items from '{path_to_load}'. Total size: {total_size_mb:.2f} MB, Avg size: {avg_size_mb:.2f} MB.")
        return bool(self.benchmark_items)

    def get_total_benchmark_size_bytes(self) -> int:
        """Calculate total size of all benchmark items in bytes."""
        return sum(len(item) if isinstance(item, (str, bytes)) else 0 for item in self.benchmark_items)

    def set_custom_benchmark_items(self, items_list: list):
        if not isinstance(items_list, list) or not all(isinstance(item, str) for item in items_list):
            self.logger.error("Failed to set custom benchmark items: input must be a list of strings.")
            return False
        self.dynamic_benchmarking_enabled = False
        self.benchmark_items = items_list[:DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET]
        total_size_mb = sum(len(item) for item in self.benchmark_items) / (1024 * 1024)
        avg_size_mb = total_size_mb / len(self.benchmark_items) if self.benchmark_items else 0
        self.logger.info(
            f"Custom benchmark dataset set with {len(self.benchmark_items)} items. Dynamic generation paused. Total size: {total_size_mb:.2f} MB, Avg size: {avg_size_mb:.2f} MB.")
        return True

    def evaluate_agent_fitness(self, agent_ai, repetitions: int = DEFAULT_BENCHMARK_REPETITIONS, gui_stop_event=None):
        if not self.benchmark_items:
            self.logger.warning(
                f"Agent {getattr(agent_ai, 'agent_id', 'Unknown')}: No benchmark items for evaluation. Returning poor fitness.")
            if self.dynamic_benchmarking_enabled: self.generate_and_set_dynamic_benchmark_items(num_items_to_generate=5,
                                                                                                population_average_fitness=-1000)
            if not self.benchmark_items: return EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkDataAvailable"}

        if _PuffinZipAI_cls is None or not isinstance(agent_ai, _PuffinZipAI_cls):
            self.logger.error(
                f"Agent eval skipped: agent_ai not valid (type: {type(agent_ai)}). PuffinZipAI class not properly loaded.")
            return EVALUATION_FAIL_REWARD, {}
        if _rle_compress_func is None or _rle_decompress_func is None or _calculate_reward_func is None:
            self.logger.error(
                f"Agent {getattr(agent_ai, 'agent_id', 'Unknown')}: Core RLE/reward functions missing in evaluator.")
            return EVALUATION_FAIL_REWARD, {}

        agent_id_str = f"AI(id={getattr(agent_ai, 'id_short', 'N/A')},min_run={getattr(agent_ai, 'rle_min_encodable_run_length', 'N/A')})"
        self.logger.debug(f"Starting fitness evaluation for Agent {agent_id_str} ({repetitions} reps per item).")
        total_reward_for_agent = 0.0
        items_evaluated_count = 0
        eval_stats = {"total_reward": 0.0, "items_evaluated": 0, "successful_rle": 0, "rle_expansion": 0,
                      "rle_no_change": 0,
                      "chose_nocompression": 0, "chose_adv_rle": 0, "sum_compression_ratios_rle_success": 0.0,
                      "sum_expansion_ratios_rle_fail": 0.0, "decomp_failures_mismatch": 0, "rle_errors_returned": 0,
                      "total_processing_time_ms": 0.0}

        for item_idx, item_text in enumerate(self.benchmark_items):
            if gui_stop_event and gui_stop_event.is_set():
                self.logger.info(f"Agent {agent_id_str}: Eval stopped by GUI item {item_idx}.")
                break

            # Throttling Logic
            if items_evaluated_count > 0 and items_evaluated_count % self.items_per_throttle_check == 0:
                if self.throttle_sleep_duration > 0:
                    time.sleep(self.throttle_sleep_duration)

            sum_reward_for_item = 0.0
            item_processed_successfully_all_reps = True
            for rep_num in range(repetitions):
                if gui_stop_event and gui_stop_event.is_set(): break
                start_time_ns_item_rep = time.perf_counter_ns()
                try:
                    state_idx = agent_ai._get_state_representation(item_text)
                    action_idx = agent_ai._choose_action(state_idx, use_exploration=False)
                    action_name = agent_ai.action_names.get(action_idx, f"UnknownAction({action_idx})")

                    compressed_text_item_rep, decompressed_text_item_rep = "", ""
                    rle_error_code_item_rep = None
                    original_size = len(item_text)
                    rle_chosen_and_successful = False

                    rle_min_run = getattr(agent_ai, 'rle_min_encodable_run_length', 2)

                    if action_name == "RLE":
                        compressed_text_item_rep = _rle_compress_func(item_text, method="simple",
                                                                      min_run_len_override=rle_min_run)
                        decompressed_text_item_rep = _rle_decompress_func(compressed_text_item_rep, method="simple",
                                                                          min_run_len_override=rle_min_run)
                    elif action_name == "NoCompression":
                        compressed_text_item_rep = item_text
                        decompressed_text_item_rep = item_text
                        eval_stats['chose_nocompression'] += 1
                    elif action_name == "AdvancedRLE":
                        compressed_text_item_rep = _rle_compress_func(item_text, method="advanced")
                        decompressed_text_item_rep = _rle_decompress_func(compressed_text_item_rep, method="advanced")
                        eval_stats['chose_adv_rle'] += 1
                    else:
                        self.logger.error(
                            f"Agent {agent_id_str} chose unknown action: {action_name} for item {item_idx}.")
                        decompressed_text_item_rep = "ERROR_UNKNOWN_ACTION_IN_EVAL"

                    if action_name in ["RLE", "AdvancedRLE"]:
                        if original_size > 0:
                            if len(compressed_text_item_rep) < original_size and decompressed_text_item_rep == item_text:
                                rle_chosen_and_successful = True
                                eval_stats['sum_compression_ratios_rle_success'] += original_size / (
                                    len(compressed_text_item_rep) if len(compressed_text_item_rep) > 0 else 1)
                            elif len(compressed_text_item_rep) > original_size:
                                eval_stats['rle_expansion'] += 1
                                eval_stats['sum_expansion_ratios_rle_fail'] += original_size / (
                                    len(compressed_text_item_rep) if len(compressed_text_item_rep) > 0 else 1)
                            elif len(compressed_text_item_rep) == original_size:
                                eval_stats['rle_no_change'] += 1
                        if rle_chosen_and_successful: eval_stats['successful_rle'] += 1

                        if decompressed_text_item_rep in _RLE_DECOMPRESSION_ERRORS_set:
                            rle_error_code_item_rep = decompressed_text_item_rep
                            eval_stats["rle_errors_returned"] += 1
                            # Reduced log spam: only log every 10th error per agent
                            if eval_stats["rle_errors_returned"] % 10 == 0:
                                self.logger.warning(
                                    f"Agent {agent_id_str} Item {item_idx + 1}: RLE_Error='{rle_error_code_item_rep}', Action='{action_name}', Input(S={len(item_text)}):'{item_text[:60]}'")
                            if rle_chosen_and_successful: eval_stats[
                                'successful_rle'] -= 1; rle_chosen_and_successful = False
                        elif decompressed_text_item_rep != item_text:
                            eval_stats["decomp_failures_mismatch"] += 1
                            if eval_stats["decomp_failures_mismatch"] % 5 == 0:
                                self.logger.warning(
                                    f"Agent {agent_id_str} Item {item_idx + 1}: Mismatch! Action='{action_name}', MinRun={rle_min_run}.")
                            if rle_chosen_and_successful: eval_stats[
                                'successful_rle'] -= 1; rle_chosen_and_successful = False

                    processing_time_ms_rep = (time.perf_counter_ns() - start_time_ns_item_rep) / 1_000_000
                    eval_stats["total_processing_time_ms"] += processing_time_ms_rep
                    reward_rep = _calculate_reward_func(item_text, compressed_text_item_rep, decompressed_text_item_rep,
                                                        action_name, processing_time_ms_rep, rle_error_code_item_rep)
                    sum_reward_for_item += reward_rep
                    if processing_time_ms_rep > (MAX_ITEM_PROCESS_TIME_SEC * 1000):
                        self.logger.warning(
                            f"Agent {agent_id_str}: Item {item_idx} rep {rep_num} EXCEEDED MAX_ITEM_PROCESS_TIME_SEC ({MAX_ITEM_PROCESS_TIME_SEC}s). Actual: {processing_time_ms_rep:.1f}ms. Penalizing.")
                        sum_reward_for_item += EVALUATION_TIMEOUT_REWARD_PENALTY
                        item_processed_successfully_all_reps = False
                except Exception as e_item_eval:
                    self.logger.error(
                        f"Agent {agent_id_str}: EXCEPTION during item {item_idx} rep {rep_num + 1} processing: {e_item_eval}",
                        exc_info=True)
                    sum_reward_for_item += EVALUATION_FAIL_REWARD
                    item_processed_successfully_all_reps = False
                    eval_stats["rle_errors_returned"] += 1

            items_evaluated_count += 1
            total_reward_for_agent += (sum_reward_for_item / repetitions if repetitions > 0 else sum_reward_for_item)

        eval_stats["items_evaluated"] = items_evaluated_count
        eval_stats["total_reward"] = total_reward_for_agent
        final_fitness_score = total_reward_for_agent if items_evaluated_count == 0 else total_reward_for_agent / items_evaluated_count

        log_msg = (
            f"Agent {agent_id_str} - Fitness Eval Complete. Items: {items_evaluated_count}, FinalAvgFit: {final_fitness_score:.4f}. "
            f"Stats: SRLE:{eval_stats['successful_rle']}, Exp:{eval_stats['rle_expansion']}, NC:{eval_stats['chose_nocompression']}, Adv:{eval_stats['chose_adv_rle']}, "
            f"MM:{eval_stats['decomp_failures_mismatch']}, RLErr:{eval_stats['rle_errors_returned']}")
        self.logger.info(log_msg)
        return final_fitness_score, eval_stats

    def evaluate_population_batch(self, population: list, repetitions_per_item: int = DEFAULT_BENCHMARK_REPETITIONS,
                                  gui_stop_event=None):
        if not self.benchmark_items:
            self.logger.warning("No benchmark items loaded/generated. Cannot evaluate population.")
            if self.dynamic_benchmarking_enabled: self.generate_and_set_dynamic_benchmark_items(
                num_items_to_generate=10, population_average_fitness=-1000)
            if not self.benchmark_items: return [(EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkDataAvailable"}) for _
                                                 in population]

        results = []
        num_agents_processed_since_throttle_check = 0
        self.logger.info(f"Starting batch evaluation for population of {len(population)} agents.")
        for agent_idx, evolving_agent_instance in enumerate(population):
            if gui_stop_event and gui_stop_event.is_set():
                self.logger.info(f"Population evaluation stopped by GUI at agent {agent_idx}/{len(population)}.")
                results.extend([(EVALUATION_FAIL_REWARD, {"notes": "EvaluationInterrupted"}) for _ in
                                range(len(population) - agent_idx)])
                break

            # Agent throttling - separate from item throttling to keep GUI responsive during heavy loads
            if num_agents_processed_since_throttle_check >= self.agents_per_throttle_check:
                if self.throttle_sleep_duration > 0:
                    time.sleep(self.throttle_sleep_duration)
                num_agents_processed_since_throttle_check = 0

            if not (hasattr(evolving_agent_instance, 'puffin_ai') and evolving_agent_instance.puffin_ai is not None):
                self.logger.error(
                    f"Agent {getattr(evolving_agent_instance, 'agent_id', f'Idx_{agent_idx}')} missing PuffinZipAI core. Assigning fail reward.")
                results.append((EVALUATION_FAIL_REWARD, {"notes": "MissingCoreAI"}))
                continue

            agent_fitness, agent_stats_dict = self.evaluate_agent_fitness(evolving_agent_instance.puffin_ai,
                                                                          repetitions=repetitions_per_item,
                                                                          gui_stop_event=gui_stop_event)
            results.append((agent_fitness, agent_stats_dict))
            num_agents_processed_since_throttle_check += 1

        self.logger.info(f"Batch evaluation finished. Processed {len(results)} agent results out of {len(population)}.")
        return results