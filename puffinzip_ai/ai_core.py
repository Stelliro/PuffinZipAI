# PuffinZipAI_Project/puffinzip_ai/ai_core.py
from __future__ import annotations

import logging
import os
import queue
import random
import shutil
import time
import traceback
import numpy as np

from .config import (
    MODEL_FILE_DEFAULT, DEFAULT_LEN_THRESHOLDS, COMPRESSED_FILE_SUFFIX,
    DEFAULT_LEARNING_RATE, DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_EXPLORATION_RATE, DEFAULT_EXPLORATION_DECAY_RATE,
    DEFAULT_MIN_EXPLORATION_RATE, DEFAULT_BATCH_COMPRESS_EXTENSIONS,
    DEFAULT_ALLOWED_LEARN_EXTENSIONS, DEFAULT_TRAIN_LOG_INTERVAL_BATCHES,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_FOLDER_LEARN_BATCH_SIZE,
    LOGS_DIR_PATH,
    CORE_AI_LOG_FILENAME,
    ACCELERATION_TARGET_DEVICE as CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT
)
from .logger import setup_logger
from .reward_system import calculate_reward
from .rle_utils import rle_compress, rle_decompress
from .rle_constants import RLE_DECOMPRESSION_ERRORS

MAX_REWARD_HISTORY_LEN = 500

class DummyLogger:
    def _log(self, level, msg, exc_info_flag=False): print(
        f"DummyLog-{level}: {msg}" + (f"\n{traceback.format_exc()}" if exc_info_flag else ""))
    def info(self, msg): self._log("INFO", msg)
    def warning(self, msg): self._log("WARN", msg)
    def error(self, msg, exc_info=False): self._log("ERROR", msg, exc_info_flag=exc_info)
    def critical(self, msg, exc_info=False): self._log("CRITICAL", msg, exc_info_flag=exc_info)
    def debug(self, msg): self._log("DEBUG", msg)
    def exception(self, msg): self._log("EXCEPTION", msg, exc_info_flag=True)

class PuffinZipAI:
    NUM_UNIQUE_RATIO_CATS = 3
    NUM_RUN_CATS = 3

    def __init__(self,
                 len_thresholds=None,
                 learning_rate=None,
                 discount_factor=None,
                 exploration_rate=None,
                 exploration_decay_rate=None,
                 min_exploration_rate=None,
                 rle_min_encodable_run: int | None = None,
                 rle_min_encodable_run_length: int | None = None,
                 target_device: str | None = None,
                 **kwargs):

        log_file_full_path = os.path.join(LOGS_DIR_PATH, CORE_AI_LOG_FILENAME)
        
        # --- FIX: WINERROR 32 (LOG ROTATION) ---
        # Use a constant logger name so all agents share ONE logger instance.
        # This prevents opening 50+ file handles to the same file.
        shared_logger_name = 'PuffinZipAI_Core_Shared'
        
        try:
            # Check if this logger already exists and has handlers (initialized by another agent)
            existing_logger = logging.getLogger(shared_logger_name)
            if existing_logger.handlers:
                self.logger = existing_logger
            else:
                # First time initialization
                self.logger = setup_logger(logger_name=shared_logger_name, 
                                         log_filename=log_file_full_path,
                                         log_level=logging.INFO)
        except Exception as e_log:
            print(f"CRITICAL: Failed to setup logger in PuffinZipAI: {e_log}. Using print/DummyLogger.")
            self.logger = DummyLogger()

        self.len_thresholds = list(len_thresholds) if len_thresholds is not None else list(DEFAULT_LEN_THRESHOLDS)
        self.action_names = {0: "RLE", 1: "NoCompression", 2: "AdvancedRLE", 3: "NovelMethod", 4: "ReferenceMethod"}
        self.action_space_size = len(self.action_names)
        self.gui_stop_event = None
        self.gui_output_queue = None

        # Novel compression method — unique per agent, evolved via genetic operators
        self.novel_method = None  # Will be a CompressionMethod from NovelCompressionGenerator
        self._novel_compress_fn = None
        self._novel_decompress_fn = None

        # Scaffolding — reference method access (training wheels)
        self._scaffolding_enabled = True
        self._preferred_reference = "gzip"  # default reference method
        self._reference_compress_fn = None
        self._reference_decompress_fn = None
        self._scaffold_agent_id = None  # set by EvolvingAgent wrapper
        try:
            from .compression_scaffolding import get_reference_method
            ref = get_reference_method(self._preferred_reference)
            if ref:
                self._reference_compress_fn = ref.compress_fn
                self._reference_decompress_fn = ref.decompress_fn
        except ImportError:
            self._scaffolding_enabled = False

        self.learning_rate = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
        self.discount_factor = discount_factor if discount_factor is not None else DEFAULT_DISCOUNT_FACTOR
        self.exploration_rate = exploration_rate if exploration_rate is not None else DEFAULT_EXPLORATION_RATE
        self.exploration_decay_rate = exploration_decay_rate if exploration_decay_rate is not None else DEFAULT_EXPLORATION_DECAY_RATE
        self.min_exploration_rate = min_exploration_rate if min_exploration_rate is not None else DEFAULT_MIN_EXPLORATION_RATE
        self.inter_batch_delay_seconds = 0.0

        self.target_device = target_device if target_device is not None else CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT
        self.use_gpu_acceleration = "GPU" in self.target_device.upper()

        self.logger.info(
            f"PuffinZipAI (Base Core) Initializing. "
            f"Target Device: '{self.target_device}', Effective GPU Use (for this instance): {self.use_gpu_acceleration}. "
            f"(Actual GPU ops depend on PuffinZipAI_GPU class and libraries)."
        )

        effective_rle_min = rle_min_encodable_run
        if effective_rle_min is None:
            effective_rle_min = rle_min_encodable_run_length

        if effective_rle_min is not None and isinstance(effective_rle_min, int) and effective_rle_min >= 1:
            self.rle_min_encodable_run_length = effective_rle_min
        else:
            try:
                from .rle_utils import MIN_ENCODABLE_RUN_LENGTH as RLE_GLOBAL_DEFAULT_MIN_RUN
                self.rle_min_encodable_run_length = max(6, RLE_GLOBAL_DEFAULT_MIN_RUN)
            except ImportError:
                self.logger.warning("Could not import RLE_GLOBAL_DEFAULT_MIN_RUN. Simple RLE min_run defaulting to 6.")
                self.rle_min_encodable_run_length = 6

        self.logger.info(
            f"PuffinZipAI params: SimpleRLE_MinRun: {self.rle_min_encodable_run_length}, LR: {self.learning_rate:.4f}, "
            f"DF: {self.discount_factor:.2f}, ER: {self.exploration_rate:.4f}, "
            f"ER_decay: {self.exploration_decay_rate:.5f}, Min_ER: {self.min_exploration_rate:.5f}"
        )
        self.training_stats = self._get_default_training_stats()
        self._reinitialize_state_dependent_vars()

    def _interruptible_sleep(self, duration_seconds):
        if not self.gui_stop_event or duration_seconds <= 0:
            if duration_seconds > 0: time.sleep(duration_seconds)
            return
        end_time = time.monotonic() + duration_seconds
        while time.monotonic() < end_time:
            if self.gui_stop_event.is_set(): return
            sleep_interval = min(0.1, end_time - time.monotonic())
            if sleep_interval > 0: time.sleep(sleep_interval)

    def configure_inter_batch_delay(self, delay_seconds_str: str):
        user_msg = ""
        try:
            delay_val = float(delay_seconds_str)
            if delay_val < 0:
                self.inter_batch_delay_seconds = 0.0
                user_msg = "Delay was negative. Set to 0.0s."
            else:
                self.inter_batch_delay_seconds = delay_val
                user_msg = f"Inter-batch delay set to {self.inter_batch_delay_seconds:.3f}s."
            self._send_to_gui(user_msg)
            return True
        except ValueError:
            user_msg = f"Invalid delay format: '{delay_seconds_str}'. Please enter a number."
            self._send_to_gui(user_msg)
            return False
        except Exception as e:
            user_msg = f"Error configuring inter-batch delay: {e}"
            self.logger.exception(user_msg)
            self._send_to_gui(f"ERROR: {user_msg}")
            return False

    def _send_to_gui(self, message):
        if self.gui_output_queue:
            try:
                self.gui_output_queue.put_nowait(str(message))
            except queue.Full:
                self.logger.warning(f"GUI output queue full. Dropping message: {str(message)[:100]}...")
        else:
            print(message)

    def _get_default_training_stats(self):
        return {
            'total_items_processed': 0, 'cumulative_reward': 0.0, 'decomp_errors': 0,
            'rle_chosen_count': 0, 'nocomp_chosen_count': 0, 'advanced_rle_chosen_count': 0,
            'novel_method_chosen_count': 0, 'reference_method_chosen_count': 0,
            'reward_history': []
        }

    def _reinitialize_state_dependent_vars(self):
        self.NUM_LEN_CATS = len(self.len_thresholds) + 1
        self.state_space_size = self.NUM_LEN_CATS * self.NUM_UNIQUE_RATIO_CATS * self.NUM_RUN_CATS
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.training_stats = self._get_default_training_stats()

    def configure_data_categories(self, new_thresh_str_list):
        user_msg = ""
        parsed_thresholds = []
        use_defaults_flag = False
        try:
            if new_thresh_str_list and any(s.strip() for s in new_thresh_str_list):
                parsed_thresholds = sorted(list(
                    set(int(t.strip()) for t in new_thresh_str_list if t.strip().isdigit() and int(t.strip()) > 0)))

            if not parsed_thresholds and any(s.strip() for s in new_thresh_str_list):
                user_msg = "Invalid threshold values provided. Using current/default thresholds."
                use_defaults_flag = True
            elif not parsed_thresholds:
                user_msg = "No thresholds provided. Using current/default thresholds."
                use_defaults_flag = True

            if use_defaults_flag:
                self.len_thresholds = list(self.len_thresholds)
                if not self.len_thresholds: self.len_thresholds = list(DEFAULT_LEN_THRESHOLDS)
                user_msg += f" Current effective thresholds: {self.len_thresholds}."
            else:
                self.len_thresholds = parsed_thresholds
                user_msg = f"Length thresholds set to: {self.len_thresholds}."

            self._reinitialize_state_dependent_vars()
            self.exploration_rate = DEFAULT_EXPLORATION_RATE
            final_message = (
                f"{user_msg} Q-Table, training stats, and exploration rate have been reset. New state space size: {self.state_space_size}.")
            self._send_to_gui(final_message)
            return True
        except ValueError:
            user_msg = "Invalid threshold format. Ensure thresholds are comma-separated positive integers."
            self._send_to_gui(user_msg)
            return False
        except Exception as e:
            user_msg = f"Error configuring data categories: {e}"
            self.logger.exception(user_msg)
            self._send_to_gui(f"ERROR: {user_msg}")
            return False

    def _get_state_representation(self, item_text):
        length = len(item_text);
        unique_chars = len(set(item_text)) if length > 0 else 0
        len_cat = self.NUM_LEN_CATS - 1
        for i, threshold in enumerate(self.len_thresholds):
            if length < threshold: len_cat = i; break
        unique_ratio_cat = 0
        if length > 0:
            ratio = unique_chars / length
            if ratio < 0.3:
                unique_ratio_cat = 0
            elif ratio < 0.7:
                unique_ratio_cat = 1
            else:
                unique_ratio_cat = 2
        max_run_val = 0
        if length > 0:
            current_run_val = 1
            for i_run in range(1, length):
                if item_text[i_run] == item_text[i_run - 1]:
                    current_run_val += 1
                else:
                    max_run_val = max(max_run_val, current_run_val);
                    current_run_val = 1
            max_run_val = max(max_run_val, current_run_val)
        run_cat = 0
        if max_run_val < 3:
            run_cat = 0
        elif max_run_val < 7:
            run_cat = 1
        else:
            run_cat = 2
        state_index = (len_cat * (
                self.NUM_UNIQUE_RATIO_CATS * self.NUM_RUN_CATS) + unique_ratio_cat * self.NUM_RUN_CATS + run_cat)
        if not (0 <= state_index < self.state_space_size):
            self.logger.warning(
                f"Calculated state_index {state_index} out of bounds [0, {self.state_space_size - 1}]. Clamping. Len: {length}, Unique: {unique_chars}, MaxRun: {max_run_val}")
            state_index = max(0, min(state_index, self.state_space_size - 1))
        return state_index

    def _choose_action(self, state_idx, use_exploration=True):
        action_idx = 0
        # Determine effective action space:
        # - Actions 0-2 always available (RLE, NoCompression, AdvancedRLE)
        # - Action 3 (NovelMethod) only if agent has a novel compress fn
        # - Action 4 (ReferenceMethod) only if scaffolding enabled + not banned
        effective_action_count = 3  # base: RLE, NoCompression, AdvancedRLE
        if self._novel_compress_fn:
            effective_action_count = 4  # include NovelMethod
        if self._scaffolding_enabled and self._reference_compress_fn:
            # Check ban status via scaffolding manager
            ref_allowed = True
            try:
                from .compression_scaffolding import get_scaffolding_manager
                agent_id = self._scaffold_agent_id or id(self)
                ref_allowed = get_scaffolding_manager().is_reference_allowed(
                    str(agent_id), getattr(self, '_current_generation', 0)
                )
            except ImportError:
                pass
            if ref_allowed:
                # Only include ReferenceMethod (idx 4) if NovelMethod (idx 3) is also
                # available, otherwise action 3 would be an unusable gap in the space.
                if effective_action_count >= 4:  # novel method available
                    effective_action_count = 5  # include ReferenceMethod at idx 4
                # If novel method not available (effective=3), don't expand to 5
                # because action 3 (NovelMethod) has no handler and would be wasted.
        if use_exploration and random.random() < self.exploration_rate:
            action_idx = random.randint(0, effective_action_count - 1)
        else:
            q_row = self.q_table[state_idx][:effective_action_count]
            action_idx = int(np.argmax(q_row))
        if use_exploration:
            if action_idx == 0:
                self.training_stats['rle_chosen_count'] += 1
            elif action_idx == 1:
                self.training_stats['nocomp_chosen_count'] += 1
            elif action_idx == 2:
                self.training_stats['advanced_rle_chosen_count'] += 1
            elif action_idx == 3:
                self.training_stats['novel_method_chosen_count'] += 1
            elif action_idx == 4:
                self.training_stats['reference_method_chosen_count'] = self.training_stats.get('reference_method_chosen_count', 0) + 1
        return action_idx

    def _update_q_table(self, state_idx, action_idx, reward_val, next_state_idx=None):
        current_q = self.q_table[state_idx, action_idx]
        # Proper TD(0) Q-learning: Q(s,a) += lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
        if next_state_idx is not None and 0 <= next_state_idx < self.state_space_size:
            max_next_q = np.max(self.q_table[next_state_idx])
        else:
            max_next_q = 0.0
        td_target = reward_val + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (td_target - current_q)
        self.q_table[state_idx, action_idx] = new_q

    def _generate_random_item(self, min_len=5, max_len=40,
                              run_likelihood_factor: float = 0.33,
                              unique_char_focus_factor: float = 0.33
                              ):
        """Generate a random benchmark string with controllable run structure.

        Optimised for large items (MB-scale): builds data in bulk chunks
        using ``random.choices`` and string multiplication instead of
        character-by-character appending.
        """
        actual_min_len = max(1, min_len)
        actual_max_len = max(actual_min_len + 1, max_len)
        length = random.randint(actual_min_len, actual_max_len)

        # --- Build character pool (same logic as before) ---
        alpha_num_sym = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{};':\",./<>? "
        char_pool_size_ratio = 0.2 + (0.8 * unique_char_focus_factor)
        effective_char_pool_size = max(1, int(len(alpha_num_sym) * char_pool_size_ratio))
        if effective_char_pool_size < len(alpha_num_sym) / 2 and effective_char_pool_size > 0:
            common_subset = list("abcdefghijklmnopqrstuvwxyz0123456789")
            random.shuffle(common_subset)
            if effective_char_pool_size <= len(common_subset):
                current_char_pool = common_subset[:effective_char_pool_size]
            else:
                current_char_pool = list(set(common_subset))
                remaining_needed = effective_char_pool_size - len(current_char_pool)
                other_symbols = [s for s in alpha_num_sym if s not in current_char_pool]
                if remaining_needed > 0 and other_symbols:
                    current_char_pool.extend(
                        random.sample(other_symbols, min(remaining_needed, len(other_symbols))))
        else:
            current_char_pool = list(alpha_num_sym)
        if not current_char_pool:
            current_char_pool = ['a']

        # --- Pre-compute run-length parameters once ---
        base_max_run = max(2, int(length * 0.05 + (length * 0.2 * run_likelihood_factor)))
        min_run_len_for_gen = 1
        if run_likelihood_factor > 0.6: min_run_len_for_gen = 2
        if run_likelihood_factor > 0.8: min_run_len_for_gen = 3
        if unique_char_focus_factor > 0.7:
            base_max_run = max(min_run_len_for_gen, int(base_max_run * 0.3))
        elif unique_char_focus_factor > 0.5:
            base_max_run = max(min_run_len_for_gen, int(base_max_run * 0.6))
        max_random_seg = max(1, int(3 * (1.0 - run_likelihood_factor)))

        # --- Bulk chunk generation (the fast path) ---
        # Instead of appending one char at a time we build string chunks
        # of runs / random segments and join once at the end.
        chunks = []
        remaining = length
        # Check stop event only every ~256KB worth of output
        stop_check_interval = max(1, 256 * 1024)
        next_stop_check = stop_check_interval

        while remaining > 0:
            # Periodic stop-event check (cheap — just a flag test)
            if remaining < next_stop_check:
                if self.gui_stop_event and self.gui_stop_event.is_set():
                    break
                next_stop_check = remaining - stop_check_interval

            if random.random() < run_likelihood_factor:
                # --- Run of identical characters ---
                run_len = random.randint(min_run_len_for_gen,
                                         max(min_run_len_for_gen + 1, base_max_run))
                run_len = min(run_len, remaining)
                c = random.choice(current_char_pool)
                chunks.append(c * run_len)       # string multiply — very fast
                remaining -= run_len
            else:
                # --- Random segment (bulk via random.choices) ---
                seg_len = min(random.randint(1, max_random_seg), remaining)
                chunks.append(''.join(random.choices(current_char_pool, k=seg_len)))
                remaining -= seg_len

        result = ''.join(chunks)
        # Pad if we fell short (e.g. due to stop event not set, just rounding)
        if len(result) < length:
            result += ''.join(random.choices(current_char_pool, k=length - len(result)))
        return result[:length]

    def _handle_item_processing_for_training(self, item_text, counter_info=""):
        state_idx = self._get_state_representation(item_text);
        action_idx = self._choose_action(state_idx, use_exploration=True);
        action_name = self.action_names[action_idx]
        original_size = len(item_text);
        compressed_text_final = "";
        decompressed_text_final = "";
        rle_error_code_final = None
        start_time_ns = time.perf_counter_ns()
        if action_idx == 0:
            compressed_text_final = rle_compress(item_text, method="simple",
                                                 min_run_len_override=self.rle_min_encodable_run_length);
            decompressed_text_final = rle_decompress(compressed_text_final, method="simple",
                                                     min_run_len_override=self.rle_min_encodable_run_length)
        elif action_idx == 1:
            compressed_text_final = item_text;
            decompressed_text_final = item_text
        elif action_idx == 2:
            compressed_text_final = rle_compress(item_text, method="advanced");
            decompressed_text_final = rle_decompress(compressed_text_final, method="advanced")
        elif action_idx == 3:
            if self._novel_compress_fn and self._novel_decompress_fn:
                try:
                    compressed_text_final = self._novel_compress_fn(item_text)
                    decompressed_text_final = self._novel_decompress_fn(compressed_text_final)
                except Exception:
                    compressed_text_final = item_text
                    decompressed_text_final = item_text
            else:
                # NovelMethod not available — fall back to NoCompression
                compressed_text_final = item_text
                decompressed_text_final = item_text
        elif action_idx == 4:
            # ReferenceMethod — use scaffolded known compression method
            if self._reference_compress_fn and self._reference_decompress_fn:
                try:
                    compressed_bytes = self._reference_compress_fn(item_text)
                    decompressed_text_final = self._reference_decompress_fn(compressed_bytes)
                    # Store byte length as string so ratio calculations work
                    compressed_text_final = "X" * len(compressed_bytes)  # placeholder of correct length
                except Exception:
                    compressed_text_final = item_text
                    decompressed_text_final = item_text
            else:
                compressed_text_final = item_text
                decompressed_text_final = item_text
        if decompressed_text_final in RLE_DECOMPRESSION_ERRORS: rle_error_code_final = decompressed_text_final
        end_time_ns = time.perf_counter_ns();
        processing_time_ms = (end_time_ns - start_time_ns) / 1e6
        current_reward = calculate_reward(item_text, compressed_text_final, decompressed_text_final, action_name,
                                          processing_time_ms, rle_error_code_final)
        # Apply scaffolding reward adjustment for reference method usage
        if self._scaffolding_enabled and action_idx == 4:
            try:
                from .reward_system import calculate_scaffolded_reward
                agent_id = str(self._scaffold_agent_id or id(self))
                current_reward = calculate_scaffolded_reward(
                    current_reward, action_name, agent_id,
                    item_text, len(compressed_text_final), original_size,
                    generation=getattr(self, '_current_generation', 0),
                )
            except ImportError:
                pass
        elif self._scaffolding_enabled and action_idx in (0, 2, 3):
            # Own method — check for bonus vs reference methods
            try:
                from .reward_system import calculate_scaffolded_reward
                agent_id = str(self._scaffold_agent_id or id(self))
                current_reward = calculate_scaffolded_reward(
                    current_reward, action_name, agent_id,
                    item_text, len(compressed_text_final), original_size,
                    generation=getattr(self, '_current_generation', 0),
                )
            except ImportError:
                pass
        decompression_mismatch = False
        if (action_idx in (0, 2, 3, 4)) and (
                rle_error_code_final or (decompressed_text_final != item_text)):
            decompression_mismatch = True;
            self.training_stats['decomp_errors'] += 1
        self.training_stats['total_items_processed'] += 1;
        self.training_stats['cumulative_reward'] += current_reward
        # Compute next_state from compressed output for TD(0) Q-learning
        next_state_idx = self._get_state_representation(compressed_text_final) if compressed_text_final else state_idx
        return (state_idx, action_idx, current_reward, item_text[:50], action_name, compressed_text_final[:50],
                decompressed_text_final[:50], decompression_mismatch, next_state_idx)

    def _process_batch(self, batch_experiences, session_info=""):
        batch_len = len(batch_experiences);
        if batch_len == 0: return 0.0
        sum_rewards_batch = 0.0;
        items_processed_in_batch = 0
        for i, experience_tuple in enumerate(batch_experiences):
            if self.gui_stop_event and self.gui_stop_event.is_set(): break
            # Support both old 8-tuple and new 9-tuple (with next_state_idx)
            if len(experience_tuple) >= 9:
                state_idx, action_idx, reward_val, _, _, _, _, _, next_state_idx = experience_tuple
            else:
                state_idx, action_idx, reward_val, _, _, _, _, _ = experience_tuple
                next_state_idx = None
            self._update_q_table(state_idx, action_idx, reward_val, next_state_idx=next_state_idx)
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
            sum_rewards_batch += reward_val;
            items_processed_in_batch += 1
            if i > 0 and i % 200 == 0: time.sleep(0)
        avg_reward_this_batch = sum_rewards_batch / items_processed_in_batch if items_processed_in_batch > 0 else 0.0
        if items_processed_in_batch > 0: self.training_stats['reward_history'].append(avg_reward_this_batch)
        if len(self.training_stats['reward_history']) > MAX_REWARD_HISTORY_LEN: self.training_stats[
            'reward_history'].pop(0)
        return avg_reward_this_batch

    def train(self, num_eps=None, run_continuously=False, b_size=None):
        effective_batch_size = b_size if b_size and b_size > 0 else DEFAULT_TRAIN_BATCH_SIZE
        batch_experience = [];
        items_in_current_batch = 0;
        session_id = "RndTrain";
        processed_items_session = 0
        LOG_INTERVAL_ITEMS = max(1, DEFAULT_TRAIN_LOG_INTERVAL_BATCHES * effective_batch_size)
        mode_str = "ContinuousRnd" if run_continuously else f"FixedItems({num_eps or 'N/A'})Rnd"
        start_message = (
            f"Starting {mode_str} training (BatchSize: {effective_batch_size}). Inter-batch delay: {self.inter_batch_delay_seconds}s. Target Device: '{self.target_device}'")
        self._send_to_gui(start_message);
        is_active_loop = True
        try:
            while is_active_loop:
                if self.gui_stop_event and self.gui_stop_event.is_set(): self._send_to_gui(
                    f"{mode_str} training stopped. Processed: {processed_items_session}."); break
                if not run_continuously and num_eps and processed_items_session >= num_eps: is_active_loop = False; continue
                run_like_factor = 0.2 + (0.6 * (1.0 - self.exploration_rate));
                unique_focus_factor = 0.8 - (0.6 * (1.0 - self.exploration_rate))
                min_len_train, max_len_train = 10, 200
                if self.exploration_rate < 0.3:
                    min_len_train, max_len_train = 50, 800
                elif self.exploration_rate < 0.7:
                    min_len_train, max_len_train = 20, 400
                item_length_train = random.randint(max(1, min_len_train), max(1, max_len_train))
                item_text = self._generate_random_item(min_len=item_length_train, max_len=item_length_train,
                                                       run_likelihood_factor=run_like_factor,
                                                       unique_char_focus_factor=unique_focus_factor)
                experience_tuple = self._handle_item_processing_for_training(item_text,
                                                                             f"Item {self.training_stats['total_items_processed'] + 1}")
                batch_experience.append(experience_tuple);
                items_in_current_batch += 1;
                processed_items_session += 1
                force_process_batch_now = (self.gui_stop_event and self.gui_stop_event.is_set() and batch_experience)
                if items_in_current_batch >= effective_batch_size or (
                        not run_continuously and num_eps and processed_items_session >= num_eps and batch_experience) or force_process_batch_now:
                    if batch_experience: self._process_batch(batch_experience,
                                                             f"{session_id}-Batch-{processed_items_session // effective_batch_size if effective_batch_size > 0 else 0}")
                    batch_experience.clear();
                    items_in_current_batch = 0
                    if self.inter_batch_delay_seconds > 0 and is_active_loop and not (
                            self.gui_stop_event and self.gui_stop_event.is_set()):
                        self._interruptible_sleep(self.inter_batch_delay_seconds)
                        if self.gui_stop_event and self.gui_stop_event.is_set(): break
                    if force_process_batch_now: break
                if self.training_stats['total_items_processed'] % LOG_INTERVAL_ITEMS == 0 and self.training_stats[
                    'total_items_processed'] > 0:
                    avg_reward_recent_str = "N/A"
                    if self.training_stats['reward_history']: num_recent_batches_for_avg = min(
                        len(self.training_stats['reward_history']),
                        DEFAULT_TRAIN_LOG_INTERVAL_BATCHES); recent_rewards_for_avg = self.training_stats[
                                                                                          'reward_history'][
                                                                                      -num_recent_batches_for_avg:]; avg_reward_recent_str = f"{sum(recent_rewards_for_avg) / len(recent_rewards_for_avg):.3f}" if recent_rewards_for_avg else "N/A"
                    total_choices_made = self.training_stats['rle_chosen_count'] + self.training_stats[
                        'nocomp_chosen_count'] + self.training_stats['advanced_rle_chosen_count']
                    rle_perc = (self.training_stats[
                                    'rle_chosen_count'] / total_choices_made * 100) if total_choices_made > 0 else 0.0;
                    adv_rle_perc = (self.training_stats[
                                        'advanced_rle_chosen_count'] / total_choices_made * 100) if total_choices_made > 0 else 0.0
                    stats_msg_parts = [
                        f"{mode_str}: SessItems {processed_items_session}. TotalItems {self.training_stats['total_items_processed']}.",
                        f"AvgRew(last ~{DEFAULT_TRAIN_LOG_INTERVAL_BATCHES}b): {avg_reward_recent_str}.",
                        f"ExpRate: {self.exploration_rate:.4f}.",
                        f"DecompErrs: {self.training_stats['decomp_errors']}.",
                        f"RLE%: {rle_perc:.1f}, AdvRLE%: {adv_rle_perc:.1f}"]
                    self._send_to_gui(" ".join(stats_msg_parts))
                if run_continuously and items_in_current_batch == 0: time.sleep(0)
        except Exception as e:
            self.logger.exception(f"Error during {mode_str} training: {e}"); self._send_to_gui(
                f"ERROR during training: {e}")
        finally:
            if batch_experience: self._process_batch(batch_experience, f"{session_id}-Finally")
            if self.gui_stop_event: self.gui_stop_event.clear()
            overall_avg_reward_val = (
                        self.training_stats['cumulative_reward'] / self.training_stats['total_items_processed']) if \
            self.training_stats['total_items_processed'] > 0 else 0.0
            final_message_parts = [f"{mode_str} training ended. Processed session items: {processed_items_session}.",
                                   f"Total global items processed: {self.training_stats['total_items_processed']}.",
                                   f"Final ExplRate: {self.exploration_rate:.4f}. Overall AvgReward: {overall_avg_reward_val:.3f}."];
            final_msg_str = " ".join(final_message_parts)
            self.logger.info(final_msg_str);
            self._send_to_gui(final_msg_str)

    def learn_from_folder(self, fp, aexts=None, run_cont=False, bsize=None):
        session_id = "FoldLearn";
        effective_extensions = aexts or DEFAULT_ALLOWED_LEARN_EXTENSIONS
        effective_batch_size = bsize if bsize and bsize > 0 else DEFAULT_FOLDER_LEARN_BATCH_SIZE
        LOG_INTERVAL = max(1, DEFAULT_TRAIN_LOG_INTERVAL_BATCHES * effective_batch_size)
        batch_experience, items_in_batch, files_this_session = [], 0, 0
        continuity_str = "continuously" if run_cont else "once"
        self._send_to_gui(
            f"Starting folder learning for '{fp}' ({continuity_str}). BatchSize: {effective_batch_size}. Inter-batch delay: {self.inter_batch_delay_seconds}s. Target Device: '{self.target_device}'.")
        pass_num = 0;
        outer_loop_active = True
        try:
            while outer_loop_active:
                if self.gui_stop_event and self.gui_stop_event.is_set(): break
                pass_num += 1;
                files_processed_this_pass = 0
                if run_cont: self._send_to_gui(f"{session_id}(Continuous): Starting Pass {pass_num} on '{fp}'")
                try:
                    dir_items = [f for f in os.listdir(fp) if os.path.isfile(os.path.join(fp, f))]
                except FileNotFoundError:
                    self._send_to_gui(
                        f"ERROR: Folder '{fp}' not found (Pass {pass_num}). Aborting folder learn."); break
                except Exception as e_lsdir:
                    self._send_to_gui(f"ERROR listing files in '{fp}' (Pass {pass_num}): {e_lsdir}"); break
                processable_files = [f_name for f_name in dir_items if
                                     any(f_name.lower().endswith(ext) for ext in effective_extensions)]
                num_processable = len(processable_files);
                self._send_to_gui(f"Pass {pass_num}: Found {num_processable} files matching extensions.")
                if num_processable == 0:
                    if run_cont:
                        self._send_to_gui(
                            f"{session_id} Pass {pass_num}: No processable files. Waiting..."); self._interruptible_sleep(
                            5.0)
                    else:
                        self._send_to_gui(f"{session_id}: No processable files for single pass."); break
                    if self.gui_stop_event and self.gui_stop_event.is_set(): outer_loop_active = False; break
                    continue
                for file_idx, filename in enumerate(processable_files):
                    if self.gui_stop_event and self.gui_stop_event.is_set():
                        if batch_experience: self._process_batch(batch_experience,
                                                                 f"{session_id}-StopSignal-Pass{pass_num}")
                        self._send_to_gui(
                            f"{session_id} learning stopped by user. Global total items: {self.training_stats['total_items_processed']}.");
                        return
                    filepath = os.path.join(fp, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f_in:
                            item_text = f_in.read()
                        if not item_text.strip(): self._send_to_gui(
                            f"Skipping empty file: {filename} (Pass {pass_num})"); continue
                        experience_data = self._handle_item_processing_for_training(item_text,
                                                                                    f"Pass{pass_num}-F:{filename[:20]}")
                        batch_experience.append(experience_data);
                        items_in_batch += 1;
                        files_processed_this_pass += 1;
                        files_this_session += 1
                        force_batch_processing = (
                                    self.gui_stop_event and self.gui_stop_event.is_set() and batch_experience)
                        if items_in_batch >= effective_batch_size or (
                                file_idx == num_processable - 1 and batch_experience) or force_batch_processing:
                            if batch_experience: self._process_batch(batch_experience,
                                                                     f"{session_id}-Pass{pass_num}-FileBatch"); batch_experience.clear(); items_in_batch = 0
                            if self.inter_batch_delay_seconds > 0 and not (
                                    self.gui_stop_event and self.gui_stop_event.is_set()) and not force_batch_processing and (
                                    file_idx < num_processable - 1 or run_cont):
                                self._interruptible_sleep(self.inter_batch_delay_seconds)
                                if self.gui_stop_event and self.gui_stop_event.is_set(): break
                            if force_batch_processing: break
                        if files_processed_this_pass % 5 == 0 or files_processed_this_pass == 1 or file_idx == num_processable - 1: self._send_to_gui(
                            f"FolderLearn(Pass {pass_num}): {files_processed_this_pass}/{num_processable}. File:'{filename}'. ExpRate:{self.exploration_rate:.4f}")
                        if self.training_stats['total_items_processed'] > 0 and self.training_stats[
                            'total_items_processed'] % LOG_INTERVAL == 0:
                            avg_rew_str = "N/A"
                            if self.training_stats['reward_history']: hist_len = min(
                                len(self.training_stats['reward_history']),
                                DEFAULT_TRAIN_LOG_INTERVAL_BATCHES);recent_slice = self.training_stats[
                                                                                       'reward_history'][
                                                                                   -hist_len:];avg_rew_str = f"{sum(recent_slice) / len(recent_slice):.3f}" if recent_slice else "N/A"
                            tot_choices = self.training_stats['rle_chosen_count'] + self.training_stats[
                                'nocomp_chosen_count'] + self.training_stats['advanced_rle_chosen_count']
                            rle_p = (self.training_stats[
                                         'rle_chosen_count'] / tot_choices * 100) if tot_choices > 0 else 0;
                            adv_rle_p = (self.training_stats[
                                             'advanced_rle_chosen_count'] / tot_choices * 100) if tot_choices > 0 else 0
                            s_msg = (
                                f"{session_id}(GlobalTotal {self.training_stats['total_items_processed']}): AvgRew(~{DEFAULT_TRAIN_LOG_INTERVAL_BATCHES}b): {avg_rew_str}. ExplRate: {self.exploration_rate:.4f}. DecompErrs: {self.training_stats['decomp_errors']}. RLE%: {rle_p:.1f}, AdvRLE%: {adv_rle_p:.1f}");
                            self.logger.info(s_msg);
                            self._send_to_gui(s_msg)
                        if file_idx > 0 and file_idx % 10 == 0: time.sleep(0.001)
                    except Exception as e_file_proc:
                        self.logger.exception(
                            f"Error processing file '{filename}' (Pass {pass_num}): {e_file_proc}"); self._send_to_gui(
                            f"ERROR processing file '{filename}': {e_file_proc}")
                if not outer_loop_active or (self.gui_stop_event and self.gui_stop_event.is_set()): break
                if batch_experience: self._process_batch(batch_experience,
                                                         f"{session_id}-EndOfPass{pass_num}"); batch_experience.clear(); items_in_batch = 0
                self._send_to_gui(
                    f"{session_id}: Pass {pass_num} complete. Files this pass: {files_processed_this_pass}. Total files this session: {files_this_session}.")
                if not run_cont: break
                if run_cont: self._interruptible_sleep(1.0)
        except Exception as e_outer:
            self.logger.exception(
                f"Outer loop error during folder learning ({session_id}): {e_outer}"); self._send_to_gui(
                f"Major ERROR during {session_id} processing: {e_outer}")
        finally:
            if batch_experience: self._process_batch(batch_experience, f"{session_id}-Finally-Pass{pass_num}")
            if self.gui_stop_event: self.gui_stop_event.clear()
            overall_avg_rwd = (
                        self.training_stats['cumulative_reward'] / self.training_stats['total_items_processed']) if \
            self.training_stats['total_items_processed'] > 0 else 0.0
            final_log_msg = (
                f"{session_id} ({continuity_str}) finished. Total files this session: {files_this_session}. Global items processed: {self.training_stats['total_items_processed']}. Final ExplRate: {self.exploration_rate:.4f}. Overall AvgReward: {overall_avg_rwd:.3f}.");
            self.logger.info(final_log_msg);
            self._send_to_gui(final_log_msg)

    def _perform_batch_file_op(self, op_key, input_filepath, output_folder_path, original_filename,
                               content_for_rle=None):
        output_filename = original_filename
        if op_key == "rle_compress" or op_key == "advanced_rle_compress":
            name, _ = os.path.splitext(original_filename); output_filename = name + COMPRESSED_FILE_SUFFIX
        elif op_key == "rle_decompress":
            if original_filename.lower().endswith(COMPRESSED_FILE_SUFFIX):
                output_filename = original_filename[:-len(COMPRESSED_FILE_SUFFIX)]
            else:
                op_key = "copy_asis"
        output_filepath = os.path.join(output_folder_path, output_filename)
        try:
            if op_key == "rle_compress":
                if content_for_rle is None: return "error_op_no_content"
                compressed_content = rle_compress(content_for_rle, method="simple",
                                                  min_run_len_override=self.rle_min_encodable_run_length)
                decompressed_check = rle_decompress(compressed_content, method="simple",
                                                    min_run_len_override=self.rle_min_encodable_run_length)
                if decompressed_check in RLE_DECOMPRESSION_ERRORS or decompressed_check != content_for_rle: shutil.copy2(
                    input_filepath, os.path.join(output_folder_path,
                                                 original_filename + ".RLE_FAILED_COPY")); return "fail_check_copied_orig_simple"
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    f_out.write(compressed_content)
                return "rle_simple"
            elif op_key == "advanced_rle_compress":
                if content_for_rle is None: return "error_op_no_content_adv"
                compressed_content = rle_compress(content_for_rle, method="advanced")
                decompressed_check = rle_decompress(compressed_content, method="advanced")
                if decompressed_check in RLE_DECOMPRESSION_ERRORS or decompressed_check != content_for_rle: shutil.copy2(
                    input_filepath, os.path.join(output_folder_path,
                                                 original_filename + ".ADV_RLE_FAILED_COPY")); return "fail_check_copied_orig_advanced"
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    f_out.write(compressed_content)
                return "rle_advanced"
            elif op_key == "rle_decompress":
                with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as f_in:
                    compressed_content_from_file = f_in.read()
                decompressed_content = rle_decompress(compressed_content_from_file, method="simple",
                                                      min_run_len_override=self.rle_min_encodable_run_length)
                if decompressed_content in RLE_DECOMPRESSION_ERRORS:
                    shutil.copy2(input_filepath,
                                 output_filepath + ".DECOMP_FAIL_COPY"); return "fail_decomp_copied_orig"
                else:
                    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                        f_out.write(decompressed_content)
                    return "ok_decomp"
            elif op_key.startswith("copy_"):
                shutil.copy2(input_filepath, output_filepath); return op_key
        except Exception as e:
            self.logger.exception(f"Error during batch file op '{op_key}' on '{original_filename}': {e}")
            try:
                shutil.copy2(input_filepath, os.path.join(output_folder_path, original_filename + ".OP_ERR_COPY"))
            except Exception:
                pass
            return "error_op_exception"
        return "error_unknown_op_path"

    def batch_compress_folder(self, in_f_path, out_f_path, a_exts=None):
        session_id = "BatchCompress";
        effective_extensions = a_exts or DEFAULT_BATCH_COMPRESS_EXTENSIONS
        self._send_to_gui(
            f"Starting {session_id} from: '{in_f_path}' to '{out_f_path}'. Extensions: {effective_extensions}. Target Device: '{self.target_device}'")
        if not os.path.isdir(in_f_path): self._send_to_gui(f"ERROR: Input folder '{in_f_path}' not found."); return
        try:
            os.makedirs(out_f_path, exist_ok=True)
        except OSError as e:
            self._send_to_gui(f"ERROR creating output folder '{out_f_path}': {e}"); return
        counts = {'total_scanned': 0, 'rle_simple': 0, 'rle_advanced': 0, 'copy_no_comp_decision': 0, 'copy_empty': 0,
                  'op_errors': 0, 'rle_simple_verify_fails': 0, 'rle_advanced_verify_fails': 0}
        try:
            files_to_process = [f for f in os.listdir(in_f_path) if os.path.isfile(os.path.join(in_f_path, f)) and any(
                f.lower().endswith(ext) for ext in effective_extensions)]
            num_files = len(files_to_process);
            self._send_to_gui(f"Found {num_files} files matching extensions for compression.")
            if num_files == 0: self._send_to_gui("No matching files to compress."); return
            for i, filename in enumerate(files_to_process):
                if self.gui_stop_event and self.gui_stop_event.is_set(): self._send_to_gui(
                    f"{session_id} operation stopped by user."); break
                if i > 0 and i % 5 == 0: time.sleep(0.001)
                input_filepath = os.path.join(in_f_path, filename);
                counts['total_scanned'] += 1
                try:
                    with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as fin:
                        content = fin.read()
                    operation_key_to_perform = "copy_empty";
                    action_idx = -1
                    if content.strip(): state_idx = self._get_state_representation(
                        content); action_idx = self._choose_action(state_idx, use_exploration=False)
                    if action_idx == 0:
                        operation_key_to_perform = "rle_compress"
                    elif action_idx == 1:
                        operation_key_to_perform = "copy_no_comp_decision"
                    elif action_idx == 2:
                        operation_key_to_perform = "advanced_rle_compress"
                    result_key = self._perform_batch_file_op(operation_key_to_perform, input_filepath, out_f_path,
                                                             filename, content if operation_key_to_perform.endswith(
                            "_compress") else None)
                    if result_key == "rle_simple":
                        counts['rle_simple'] += 1
                    elif result_key == "rle_advanced":
                        counts['rle_advanced'] += 1
                    elif result_key == "copy_no_comp_decision":
                        counts['copy_no_comp_decision'] += 1
                    elif result_key == "copy_empty":
                        counts['copy_empty'] += 1
                    elif result_key == "fail_check_copied_orig_simple":
                        counts['rle_simple_verify_fails'] += 1
                    elif result_key == "fail_check_copied_orig_advanced":
                        counts['rle_advanced_verify_fails'] += 1
                    else:
                        counts['op_errors'] = counts.get('op_errors', 0) + 1; self.logger.warning(
                            f"Batch compress encountered op status: {result_key} for file {filename}")
                    if counts['total_scanned'] % 10 == 0 or counts['total_scanned'] == num_files: self._send_to_gui(
                        f"{session_id} progress: {counts['total_scanned']}/{num_files} files processed...")
                except Exception as e_file:
                    self.logger.exception(f"Error processing file '{filename}' for {session_id}: {e_file}");
                    self._send_to_gui(f"ERROR processing file '{filename}': {e_file}");
                    counts['op_errors'] = counts.get('op_errors', 0) + 1
                    try:
                        shutil.copy2(input_filepath, os.path.join(out_f_path, filename + ".PROC_ERR_COPY_MAINLOOP"))
                    except Exception as e_copy_fail:
                        self.logger.error(f"Failed to fallback copy errored file {filename}: {e_copy_fail}")
            summary_message = (
                f"{session_id} finished. Scanned: {counts['total_scanned']}. SimpleRLE: {counts['rle_simple']}. AdvRLE: {counts['rle_advanced']}. NoComp(AI): {counts['copy_no_comp_decision']}. EmptyCopied: {counts['copy_empty']}. SimpleRLEFails: {counts['rle_simple_verify_fails']}. AdvRLEFails: {counts['rle_advanced_verify_fails']}. OtherOpErrors: {counts['op_errors']}.")
            self.logger.info(summary_message);
            self._send_to_gui(summary_message)
        except Exception as e_outer_compress:
            self.logger.exception(f"Major error during {session_id}: {e_outer_compress}"); self._send_to_gui(
                f"ERROR during batch compression: {e_outer_compress}")
        finally:
            if self.gui_stop_event: self.gui_stop_event.clear()

    def batch_decompress_folder(self, in_f_path, out_f_path):
        session_id = "BatchDecompress";
        self._send_to_gui(f"Starting {session_id} from: '{in_f_path}' to '{out_f_path}'.")
        if not os.path.isdir(in_f_path): self._send_to_gui(f"Error: Input folder '{in_f_path}' not found."); return
        try:
            os.makedirs(out_f_path, exist_ok=True)
        except OSError as e:
            self._send_to_gui(f"Error creating output folder '{out_f_path}': {e}"); return
        counts = {'total_scanned': 0, 'ok_decomp': 0, 'fail_decomp_copied_orig': 0, 'copy_asis': 0, 'op_errors': 0}
        try:
            all_files_in_dir = [f for f in os.listdir(in_f_path) if os.path.isfile(os.path.join(in_f_path, f))]
            num_files = len(all_files_in_dir);
            self._send_to_gui(f"Found {num_files} files in input folder.")
            if num_files == 0: self._send_to_gui("Input folder is empty. Nothing to decompress."); return
            for i, filename in enumerate(all_files_in_dir):
                if self.gui_stop_event and self.gui_stop_event.is_set(): self._send_to_gui(
                    f"{session_id} operation stopped by user."); break
                if i > 0 and i % 5 == 0: time.sleep(0.001)
                input_filepath = os.path.join(in_f_path, filename);
                counts['total_scanned'] += 1
                try:
                    operation_key = "rle_decompress" if filename.lower().endswith(
                        COMPRESSED_FILE_SUFFIX) else "copy_asis"
                    result_key = self._perform_batch_file_op(operation_key, input_filepath, out_f_path, filename)
                    if result_key == "ok_decomp":
                        counts['ok_decomp'] += 1
                    elif result_key == "fail_decomp_copied_orig":
                        counts['fail_decomp_copied_orig'] += 1
                    elif result_key == "copy_asis":
                        counts['copy_asis'] += 1
                    else:
                        counts['op_errors'] = counts.get('op_errors', 0) + 1; self.logger.warning(
                            f"Batch decompress got op status: {result_key} for {filename}")
                    if counts['total_scanned'] % 10 == 0 or counts['total_scanned'] == num_files: self._send_to_gui(
                        f"{session_id} progress: {counts['total_scanned']}/{num_files} files processed...")
                except Exception as e_file_de:
                    self.logger.exception(f"Error processing file '{filename}' for {session_id}: {e_file_de}");
                    self._send_to_gui(f"ERROR processing file '{filename}': {e_file_de}");
                    counts['op_errors'] = counts.get('op_errors', 0) + 1
                    try:
                        shutil.copy2(input_filepath,
                                     os.path.join(out_f_path, filename + ".DECOMP_PROC_ERR_COPY_MAINLOOP"))
                    except Exception as e_copy_fail_de:
                        self.logger.error(
                            f"Failed to fallback copy errored (decompress) file {filename}: {e_copy_fail_de}")
            summary_decompress_msg = (
                f"{session_id} finished. Scanned: {counts['total_scanned']}. DecompressedOK: {counts['ok_decomp']}. DecompFail(Copied): {counts['fail_decomp_copied_orig']}. CopiedAsIs: {counts['copy_asis']}. OtherOpErrors: {counts['op_errors']}.")
            self.logger.info(summary_decompress_msg);
            self._send_to_gui(summary_decompress_msg)
        except Exception as e_outer_decompress:
            self.logger.exception(f"Major error during {session_id}: {e_outer_decompress}"); self._send_to_gui(
                f"ERROR during batch decompression: {e_outer_decompress}")
        finally:
            if self.gui_stop_event: self.gui_stop_event.clear()

    def _format_q_table_summary_line(self, state_idx):
        if not (0 <= state_idx < self.state_space_size): return f"State {state_idx}: Invalid"
        len_cat_divisor = (self.NUM_UNIQUE_RATIO_CATS * self.NUM_RUN_CATS)
        if len_cat_divisor == 0: return f"State {state_idx}: Error - category divisor is zero."
        len_idx = state_idx // len_cat_divisor;
        remainder_after_len = state_idx % len_cat_divisor
        if self.NUM_RUN_CATS == 0: return f"State {state_idx}: Error - NUM_RUN_CATS is zero."
        unique_ratio_idx = remainder_after_len // self.NUM_RUN_CATS;
        run_cat_idx = remainder_after_len % self.NUM_RUN_CATS
        state_description = f"L{len_idx}U{unique_ratio_idx}R{run_cat_idx}"
        q_values_for_state = self.q_table[state_idx];
        preferred_action_idx = np.argmax(q_values_for_state);
        preferred_action_name = self.action_names[int(preferred_action_idx)]
        q_info_parts = [f"{self.action_names[act_i]}:{q_values_for_state[act_i]:.3f}" for act_i in
                        range(self.action_space_size)];
        q_info_str = "|".join(q_info_parts)
        if np.all(q_values_for_state == 0):
            q_info_str += " (Untrained)"
        elif self.action_space_size > 1 and abs(q_values_for_state[0] - q_values_for_state[1]) < 0.01:
            q_info_str += " (RLE/NoComp Close)"
            if self.action_space_size > 2:
                is_all_close = True
                for act_i in range(self.action_space_size):
                    for act_j in range(act_i + 1, self.action_space_size):
                        if abs(
                            q_values_for_state[act_i] - q_values_for_state[act_j]) >= 0.01: is_all_close = False; break
                    if not is_all_close: break
                if is_all_close: q_info_str += " (All Actions Close)"
        return f"State {state_description}(Idx {state_idx:03d}): -> {preferred_action_name} ({q_info_str})"

    def display_q_table_summary(self):
        q_table_shape_str = str(self.q_table.shape) if hasattr(self, 'q_table') and self.q_table is not None else 'N/A'
        output_lines = [f"\n--- Q-Table Summary (Thresholds: {self.len_thresholds}) ---",
                        f"Q-Table Shape: {q_table_shape_str}", f"State Space Size: {self.state_space_size}",
                        f"Action Space Size: {self.action_space_size} ({', '.join(self.action_names.values())})",
                        f"Current SimpleRLE Min Encodable Run: {self.rle_min_encodable_run_length}",
                        f"Effective Hardware Target: '{self.target_device}', GPU Ops Active (if GPU Core): {self.use_gpu_acceleration}"
                        ]
        stats = self.training_stats;
        output_lines.append(f"\n--- Training Stats ---");
        output_lines.append(f"Total Items Trained Globally: {stats.get('total_items_processed', 0)}")
        avg_rew = (stats.get('cumulative_reward', 0.0) / stats.get('total_items_processed',
                                                                   1 if stats.get('total_items_processed',
                                                                                  0) == 0 else stats.get(
                                                                       'total_items_processed', 0)));
        output_lines.append(f"Overall Average Reward: {avg_rew:.4f}");
        output_lines.append(f"Decompression Errors (during training): {stats.get('decomp_errors', 0)}")
        rle_c = stats.get('rle_chosen_count', 0);
        nocomp_c = stats.get('nocomp_chosen_count', 0);
        adv_rle_c = stats.get('advanced_rle_chosen_count', 0)
        total_act_c = rle_c + nocomp_c + adv_rle_c
        rle_p = (rle_c / total_act_c * 100) if total_act_c > 0 else 0.0;
        adv_rle_p = (adv_rle_c / total_act_c * 100) if total_act_c > 0 else 0.0;
        nocomp_p = (nocomp_c / total_act_c * 100) if total_act_c > 0 else 0.0
        output_lines.append(
            f"Training Actions Chosen: RLE {rle_c}({rle_p:.1f}%), NoComp {nocomp_c}({nocomp_p:.1f}%), AdvancedRLE {adv_rle_c}({adv_rle_p:.1f}%)")
        output_lines.append(f"Current Exploration Rate: {self.exploration_rate:.4f}")
        if hasattr(self, 'q_table') and self.q_table is not None and self.q_table.size > 0:
            output_lines.append("\n--- Q-Values by State (L=LenCat, U=UniqRatioCat, R=RunLenCat) ---");
            max_states_to_display = 100
            for i in range(min(self.q_table.shape[0], max_states_to_display)):
                if i >= self.state_space_size: break
                output_lines.append(self._format_q_table_summary_line(i))
            if self.q_table.shape[0] > max_states_to_display: output_lines.append(
                f"... (displaying first {max_states_to_display} of {self.q_table.shape[0]} states) ...")
        else:
            output_lines.append("Q-Table not initialized or is empty.")
        output_lines.append("--- End Q-Table Summary ---");
        self._send_to_gui("\n".join(output_lines))

    def test_agent_on_random_items(self, num_items=5):
        session_id = "TestRandomItems";
        self._send_to_gui(
            f"\n--- Testing Agent on {num_items} Random Items (Thresholds: {self.len_thresholds}, SimpleRLE Min_Run: {self.rle_min_encodable_run_length}) ---")
        cumulative_reward_test, items_tested_count = 0.0, 0
        for i in range(num_items):
            if self.gui_stop_event and self.gui_stop_event.is_set(): break
            min_len_test, max_len_test = 3, 45
            if self.len_thresholds and len(self.len_thresholds) > 0:
                roll = random.random()
                if roll < 0.1 and self.len_thresholds[0] > 1:
                    min_len_test, max_len_test = 1, self.len_thresholds[0] - 1
                elif roll < 0.3:
                    min_len_test, max_len_test = self.len_thresholds[0], (
                        self.len_thresholds[-1] if len(self.len_thresholds) > 1 else self.len_thresholds[0] + 50)
                elif roll < 0.5:
                    min_len_test, max_len_test = self.len_thresholds[-1] + 1, self.len_thresholds[-1] + random.randint(
                        30, 150)
            item_text = self._generate_random_item(min_len=max(1, min_len_test),
                                                   max_len=max(min_len_test + 1, max_len_test),
                                                   run_likelihood_factor=random.random(),
                                                   unique_char_focus_factor=random.random())
            if not item_text and max_len_test > 0: item_text = random.choice("abc")
            items_tested_count += 1;
            state_idx = self._get_state_representation(item_text);
            action_idx = self._choose_action(state_idx, use_exploration=False);
            action_name_chosen = self.action_names[action_idx]
            original_size = len(item_text);
            info_messages_test = [];
            test_reward_this_item = 0.0;
            rle_error_code_test = None
            state_desc_test = self._format_q_table_summary_line(state_idx).split('->')[
                0] if 0 <= state_idx < self.state_space_size else f"State_Invalid({state_idx})"
            self._send_to_gui(
                f"\nItem {i + 1}: \"{item_text[:60]}{'...' if len(item_text) > 60 else ''}\" (Size: {original_size}, State: {state_desc_test})")
            test_op_start_time_ns = time.perf_counter_ns();
            compressed_output, decompressed_output_check = "", ""
            if action_idx == 0:
                compressed_output = rle_compress(item_text, method="simple",
                                                 min_run_len_override=self.rle_min_encodable_run_length);
                decompressed_output_check = rle_decompress(compressed_output, method="simple",
                                                           min_run_len_override=self.rle_min_encodable_run_length)
                if decompressed_output_check in RLE_DECOMPRESSION_ERRORS: rle_error_code_test = decompressed_output_check
                test_reward_this_item = calculate_reward(item_text, compressed_output, decompressed_output_check, "RLE",
                                                         0, rle_error_code_test)
                info_messages_test.append(
                    f"SimpleRLE Chosen. Compressed:'{compressed_output[:60]}...', NewSize:{len(compressed_output)}, EffRewRatio:{test_reward_this_item:.2f}" + (
                        f" (DECOMP FAIL: {decompressed_output_check})" if rle_error_code_test or decompressed_output_check != item_text else ""))
            elif action_idx == 1:
                compressed_output, decompressed_output_check = item_text, item_text;
                test_reward_this_item = calculate_reward(item_text, item_text, item_text, "NoCompression", 0);
                info_messages_test.append(
                    f"NoCompression Chosen. Size: {original_size}, EffRewRatio:{test_reward_this_item:.2f}")
            elif action_idx == 2:
                compressed_output = rle_compress(item_text, method="advanced");
                decompressed_output_check = rle_decompress(compressed_output, method="advanced");
                if decompressed_output_check in RLE_DECOMPRESSION_ERRORS: rle_error_code_test = decompressed_output_check
                test_reward_this_item = calculate_reward(item_text, compressed_output, decompressed_output_check,
                                                         "AdvancedRLE", 0, rle_error_code_test)
                info_messages_test.append(
                    f"AdvancedRLE Chosen. Compressed:'{compressed_output[:60]}...', NewSize:{len(compressed_output)}, EffRewRatio:{test_reward_this_item:.2f}" + (
                        f" (DECOMP FAIL: {decompressed_output_check})" if rle_error_code_test or decompressed_output_check != item_text else ""))
            processing_time_test_op_ms = (time.perf_counter_ns() - test_op_start_time_ns) / 1e6
            self._send_to_gui(
                f"  AgentChose: {action_name_chosen} -> {' '.join(info_messages_test)}. TestOpTime: {processing_time_test_op_ms:.3f}ms");
            cumulative_reward_test += test_reward_this_item
            if i > 0 and i % 10 == 0: time.sleep(0.001)
        average_test_reward = cumulative_reward_test / items_tested_count if items_tested_count > 0 else 0.0
        self._send_to_gui(
            f"\nAverage effective reward over {items_tested_count} test items: {average_test_reward:.3f}\n--- End Agent Test ---")
        if self.gui_stop_event: self.gui_stop_event.clear()

    def compress_user_item(self, item_text):
        log_snippet = item_text[:50] + ('...' if len(item_text) > 50 else '');
        self.logger.info(
            f"Compressing user item (len {len(item_text)}): '{log_snippet}', Agent SimpleRLE Min_Run: {self.rle_min_encodable_run_length}")
        output_parts = [
            f"\n--- Compress User Item (Thresholds: {self.len_thresholds}, SimpleRLE Min_Run: {self.rle_min_encodable_run_length}) ---",
            f"Original Text: '{item_text[:100]}{'...' if len(item_text) > 100 else ''}'"]
        if not item_text:
            output_parts.append("Input text is empty. Nothing to compress.")
        else:
            state_idx = self._get_state_representation(item_text);
            action_idx = self._choose_action(state_idx, use_exploration=False);
            action_name = self.action_names[action_idx];
            original_size = len(item_text)
            state_desc = self._format_q_table_summary_line(state_idx).split('->')[
                0] if 0 <= state_idx < self.state_space_size else f"State_Invalid({state_idx})";
            output_parts.append(f"Original Size: {original_size}, State: {state_desc}, AI Chose: {action_name}")
            if action_idx == 0:
                compressed_text = rle_compress(item_text, method="simple",
                                               min_run_len_override=self.rle_min_encodable_run_length);
                decompressed_check = rle_decompress(compressed_text, method="simple",
                                                    min_run_len_override=self.rle_min_encodable_run_length);
                compressed_size = len(compressed_text)
                if decompressed_check in RLE_DECOMPRESSION_ERRORS or decompressed_check != item_text:
                    output_parts.append(
                        f"  CRITICAL ERROR: Simple RLE verification FAILED! Decompressed:'{decompressed_check[:80]}...'")
                else:
                    ratio = (original_size / compressed_size) if compressed_size > 0 else (
                        float('inf') if original_size > 0 else 1.0); output_parts.append(
                        f"  SimpleRLE Applied. Compressed Text: '{compressed_text[:100]}{'...' if len(compressed_text) > 100 else ''}', New Size: {compressed_size}, Ratio: {ratio:.2f}")
            elif action_idx == 1:
                output_parts.append(f"  NoCompression Applied. Size remains: {original_size}")
            elif action_idx == 2:
                compressed_text = rle_compress(item_text, method="advanced");
                decompressed_check = rle_decompress(compressed_text, method="advanced");
                compressed_size = len(compressed_text)
                if decompressed_check in RLE_DECOMPRESSION_ERRORS or decompressed_check != item_text:
                    output_parts.append(
                        f"  CRITICAL ERROR: Advanced RLE verification FAILED! Decompressed:'{decompressed_check[:80]}...'")
                else:
                    ratio = (original_size / compressed_size) if compressed_size > 0 else (
                        float('inf') if original_size > 0 else 1.0); output_parts.append(
                        f"  AdvancedRLE Applied. Compressed Text: '{compressed_text[:100]}{'...' if len(compressed_text) > 100 else ''}', New Size: {compressed_size}, Ratio: {ratio:.2f}")
        output_parts.append("--- End User Item Compression ---");
        self._send_to_gui("\n".join(output_parts))

    def decompress_user_item_rle(self, compressed_item_text):
        log_snippet = compressed_item_text[:50] + ('...' if len(compressed_item_text) > 50 else '');
        self.logger.info(
            f"Decompressing user-provided RLE (len {len(compressed_item_text)}): '{log_snippet}'. Agent's SimpleRLE_Min_Run (for context): {self.rle_min_encodable_run_length}")
        output_parts = [
            f"\n--- Decompress User-Provided RLE (Attempting with Simple RLE params, Min_Run: {self.rle_min_encodable_run_length}) ---",
            f"Input RLE Text: '{compressed_item_text[:100]}{'...' if len(compressed_item_text) > 100 else ''}'"]
        if not compressed_item_text:
            output_parts.append("Input RLE text is empty.")
        else:
            decompressed_text = rle_decompress(compressed_item_text, method="simple",
                                               min_run_len_override=self.rle_min_encodable_run_length)
            if decompressed_text in RLE_DECOMPRESSION_ERRORS:
                output_parts.append(f"  Error during RLE decompression: {decompressed_text}"); output_parts.append(
                    f"  NOTE: If this data was compressed with 'AdvancedRLE', try specific AdvancedRLE decompress if available.")
            else:
                output_parts.append(
                    f"  Decompressed Text: '{decompressed_text[:100]}{'...' if len(decompressed_text) > 100 else ''}' (Size: {len(decompressed_text)})")
        output_parts.append("--- End User RLE Decompression ---");
        self._send_to_gui("\n".join(output_parts))

    def get_config_dict(self):
        return {'len_thresholds': list(self.len_thresholds),
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay_rate': self.exploration_decay_rate,
                'min_exploration_rate': self.min_exploration_rate,
                'rle_min_encodable_run': self.rle_min_encodable_run_length,
                'target_device': self.target_device,
                }

    # ------------------------------------------------------------------
    # Pickle support — exclude unpicklable closures, loggers, GUI refs
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable closures
        state['_novel_compress_fn'] = None
        state['_novel_decompress_fn'] = None
        state['_reference_compress_fn'] = None
        state['_reference_decompress_fn'] = None
        # Preserve novel method reconstruction info, drop the object itself.
        # The full pipeline definition (pipeline name, discovery_seed,
        # rle_min_run, steps) is stored so novel methods survive
        # save/load/restart.  This is the key to persistence: we store the
        # *recipe*, not the closure, then rebuild the closures on load.
        # v0.9.9: The metadata may also contain a 'recipe' key with the full
        # NovelMethodRecipe dict — that takes priority during restore.
        novel_meta = None
        if self.novel_method is not None:
            md = getattr(self.novel_method, 'metadata', {})
            novel_meta = {
                'pipeline': md.get('pipeline', 'rle_only'),
                'discovery_seed': md.get('discovery_seed'),
                'rle_min_run': md.get('rle_min_run', 3),
                'steps': md.get('steps', []),
                'name': getattr(self.novel_method, 'name', 'restored_novel'),
                'description': getattr(self.novel_method, 'description', ''),
                'author': getattr(self.novel_method, 'author', 'NovelCompressionGenerator_v2'),
                'recipe': md.get('recipe'),       # v0.9.9: full recipe dict
                'is_mature': md.get('is_mature'),  # v0.9.9: maturity flag
            }
        state['novel_method'] = None
        state['_novel_method_meta'] = novel_meta
        # Remove unpicklable logger / GUI refs
        state.pop('logger', None)
        state.pop('gui_stop_event', None)
        state.pop('gui_output_queue', None)
        return state

    def __setstate__(self, state):
        novel_meta = state.pop('_novel_method_meta', None)
        self.__dict__.update(state)
        # Restore logger
        shared_logger_name = 'PuffinZipAI_Core_Shared'
        self.logger = logging.getLogger(shared_logger_name)
        if not self.logger.handlers:
            try:
                log_file = os.path.join(LOGS_DIR_PATH, CORE_AI_LOG_FILENAME)
                self.logger = setup_logger(logger_name=shared_logger_name,
                                          log_filename=log_file,
                                          log_level=logging.INFO)
            except Exception:
                self.logger = DummyLogger()
        # Restore GUI refs
        self.gui_stop_event = None
        self.gui_output_queue = None
        # Rebuild novel compression closures from the stored pipeline recipe.
        # The recipe (pipeline name, discovery_seed, rle_min_run, steps) was
        # saved by __getstate__ so that novel methods survive save/load/restart.
        # v0.9.9: If a full recipe dict is present, use build_method_from_recipe
        # for more accurate reconstruction (including maturity state).
        if novel_meta:
            try:
                from .novel_compression_generator import NovelCompressionGenerator
                gen = NovelCompressionGenerator()

                # v0.9.9: Try recipe-based restore first
                recipe_dict = novel_meta.get('recipe')
                if recipe_dict is not None:
                    from .novel_compression_generator import NovelMethodRecipe
                    recipe = NovelMethodRecipe.from_dict(recipe_dict)
                    method = gen.build_method_from_recipe(
                        recipe,
                        method_name=novel_meta.get('name', 'restored_novel'),
                    )
                    self.novel_method = method
                    self._novel_compress_fn = method.compress_fn
                    self._novel_decompress_fn = method.decompress_fn
                else:
                    # Legacy fallback: rebuild from pipeline name + seed
                    cfn, dfn = gen._build_pipeline(
                        novel_meta['pipeline'],
                        discovery_seed=novel_meta.get('discovery_seed'),
                        rle_min_run=novel_meta.get('rle_min_run'),
                    )
                    from .compression_method_registry import CompressionMethod, CompressionLanguage
                    restored_metadata = {
                        'pipeline': novel_meta.get('pipeline', 'rle_only'),
                        'discovery_seed': novel_meta.get('discovery_seed'),
                        'rle_min_run': novel_meta.get('rle_min_run', 3),
                        'steps': novel_meta.get('steps', []),
                    }
                    self.novel_method = CompressionMethod(
                        name=novel_meta.get('name', 'restored_novel'),
                        language=CompressionLanguage.HYBRID,
                        compress_fn=cfn,
                        decompress_fn=dfn,
                        description=novel_meta.get('description', ''),
                        author=novel_meta.get('author', 'NovelCompressionGenerator_v2'),
                        is_novelty=True,
                        metadata=restored_metadata,
                    )
                    self._novel_compress_fn = cfn
                    self._novel_decompress_fn = dfn
            except Exception:
                self.novel_method = None
                self._novel_compress_fn = None
                self._novel_decompress_fn = None
        # Restore reference method closures for scaffolding
        if getattr(self, '_scaffolding_enabled', False):
            try:
                from .compression_scaffolding import get_reference_method
                pref = getattr(self, '_preferred_reference', 'gzip')
                ref = get_reference_method(pref)
                if ref:
                    self._reference_compress_fn = ref.compress_fn
                    self._reference_decompress_fn = ref.decompress_fn
            except ImportError:
                self._reference_compress_fn = None
                self._reference_decompress_fn = None
        else:
            self._reference_compress_fn = None
            self._reference_decompress_fn = None

    def clone_core_model(self):
        config_params_for_clone = self.get_config_dict();
        cloned_ai_agent = PuffinZipAI(**config_params_for_clone)
        if self.q_table is not None:
            if cloned_ai_agent.q_table is None or cloned_ai_agent.q_table.shape == self.q_table.shape:
                cloned_ai_agent.q_table = np.copy(self.q_table)
            else:
                self.logger.warning(
                    f"Q-table shape mismatch during clone. Original: {self.q_table.shape}, Cloned (after init): {cloned_ai_agent.q_table.shape if cloned_ai_agent.q_table is not None else 'None'}. Cloned Q-table is re-initialized one.")
        # Propagate novel compression method to clone
        cloned_ai_agent.novel_method = self.novel_method
        cloned_ai_agent._novel_compress_fn = self._novel_compress_fn
        cloned_ai_agent._novel_decompress_fn = self._novel_decompress_fn
        # Propagate scaffolding settings to clone
        cloned_ai_agent._scaffolding_enabled = self._scaffolding_enabled
        cloned_ai_agent._preferred_reference = self._preferred_reference
        cloned_ai_agent._reference_compress_fn = self._reference_compress_fn
        cloned_ai_agent._reference_decompress_fn = self._reference_decompress_fn
        self.logger.debug(
            f"Cloned AI core. Original SimpleRLE_MinRun: {self.rle_min_encodable_run_length}, Cloned: {cloned_ai_agent.rle_min_encodable_run_length}. "
            f"Cloned LR: {cloned_ai_agent.learning_rate:.4f}. Cloned TargetDevice: '{cloned_ai_agent.target_device}'."
            f" NovelMethod: {'yes' if cloned_ai_agent._novel_compress_fn else 'no'}")
        return cloned_ai_agent

    def save_model(self, fp=None):
        target_filepath = fp if fp is not None else MODEL_FILE_DEFAULT;
        abs_filepath = os.path.abspath(target_filepath);
        self.logger.info(f"Attempting to save model to: '{abs_filepath}'")
        try:
            dir_name = os.path.dirname(abs_filepath)
            if dir_name and not os.path.exists(dir_name): os.makedirs(dir_name, exist_ok=True)
            model_state = self.get_config_dict()
            model_state['q_table'] = self.q_table
            model_state['training_stats'] = self.training_stats
            model_state['version_aicore_save'] = "1.3.6_aicore_target_device"

            np.save(abs_filepath, model_state, allow_pickle=True)  # type: ignore[arg-type]
            items_trained_count = self.training_stats.get('total_items_processed', 0);
            success_msg = f"Model saved to: '{abs_filepath}'. Items Trained (Global): {items_trained_count}";
            self.logger.info(success_msg);
            self._send_to_gui(success_msg);
            return True
        except Exception as e:
            error_msg = f"Error saving model to '{abs_filepath}': {e}";
            self.logger.exception(error_msg);
            self._send_to_gui(f"ERROR: {error_msg}");
            return False

    def load_model(self, fp=None):
        target_filepath = fp if fp is not None else MODEL_FILE_DEFAULT;
        abs_filepath = os.path.abspath(target_filepath);
        self.logger.info(f"Attempting to load model from: '{abs_filepath}'")
        try:
            if not os.path.exists(abs_filepath):
                not_found_msg = f"Model file '{abs_filepath}' not found. Initializing a new model with default parameters.";
                self.logger.warning(not_found_msg);
                self._send_to_gui(not_found_msg);
                self._reinitialize_state_dependent_vars();
                self.target_device = CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT
                self.use_gpu_acceleration = "GPU" in self.target_device.upper()
                return False
            data = np.load(abs_filepath, allow_pickle=True).item()

            self.learning_rate = data.get('learning_rate', DEFAULT_LEARNING_RATE);
            self.discount_factor = data.get('discount_factor', DEFAULT_DISCOUNT_FACTOR);
            self.exploration_decay_rate = data.get('exploration_decay_rate', DEFAULT_EXPLORATION_DECAY_RATE);
            self.min_exploration_rate = data.get('min_exploration_rate', DEFAULT_MIN_EXPLORATION_RATE)

            self.target_device = data.get('target_device', CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT)
            self.use_gpu_acceleration = "GPU" in self.target_device.upper()

            loaded_rle_min_run = data.get('rle_min_encodable_run', data.get('rle_min_encodable_run_length'))
            if loaded_rle_min_run is not None and isinstance(loaded_rle_min_run, int) and loaded_rle_min_run >= 1:
                self.rle_min_encodable_run_length = loaded_rle_min_run
            else:
                default_rle_min_run_fb = 6;
                try:
                    from .rle_utils import \
                        MIN_ENCODABLE_RUN_LENGTH as RLE_DEFAULT_FB; default_rle_min_run_fb = max(6, RLE_DEFAULT_FB)
                except:
                    pass;
                self.rle_min_encodable_run_length = getattr(self, 'rle_min_encodable_run_length',
                                                            default_rle_min_run_fb)
            loaded_thresholds = data.get('len_thresholds', list(DEFAULT_LEN_THRESHOLDS))
            expected_q_state_size = (len(loaded_thresholds) + 1) * self.NUM_UNIQUE_RATIO_CATS * self.NUM_RUN_CATS;
            structure_changed = False
            if list(self.len_thresholds) != list(loaded_thresholds) or (
                    not hasattr(self, 'q_table') or self.q_table is None or self.q_table.shape[
                0] != expected_q_state_size):
                self.len_thresholds = list(loaded_thresholds);
                self._reinitialize_state_dependent_vars();
                structure_changed = True
                self.logger.info(
                    "Model structure (thresholds/Q-size) changed upon load. Reinitialized related state vars & training stats.")

            loaded_q_table = data.get('q_table')
            if loaded_q_table is not None and self.q_table is not None and loaded_q_table.shape == self.q_table.shape:
                self.q_table = loaded_q_table;
                self.logger.info("Q-table successfully loaded from model file.")
            elif loaded_q_table is not None:
                loaded_shape_str = str(loaded_q_table.shape) if hasattr(loaded_q_table, 'shape') else "UnknownShape";
                current_q_shape_str = str(self.q_table.shape) if hasattr(self,
                                                                         'q_table') and self.q_table is not None else "N/A";
                self.logger.warning(
                    f"Loaded Q-Table from '{abs_filepath}' has incompatible shape ({loaded_shape_str}) with current model (expected {current_q_shape_str}). Q-Table remains re-initialized.");
                if not structure_changed: self._reinitialize_state_dependent_vars()
            else:
                self.logger.warning(
                    f"No Q-Table found in model file '{abs_filepath}'. Q-Table remains initialized/re-initialized.");
                if not structure_changed and (self.q_table is None or self.q_table.shape[
                    0] != expected_q_state_size): self._reinitialize_state_dependent_vars()

            self.exploration_rate = data.get('exploration_rate', DEFAULT_EXPLORATION_RATE)
            loaded_training_stats = data.get('training_stats')
            if loaded_training_stats and not structure_changed:
                self.training_stats = loaded_training_stats;
                self.training_stats.setdefault('rle_chosen_count', 0);
                self.training_stats.setdefault('nocomp_chosen_count', 0);
                self.training_stats.setdefault('advanced_rle_chosen_count', 0);
                self.training_stats.setdefault('reward_history', [])
                if len(self.training_stats['reward_history']) > MAX_REWARD_HISTORY_LEN: self.training_stats[
                    'reward_history'] = self.training_stats['reward_history'][-MAX_REWARD_HISTORY_LEN:]
                self.logger.info("Training statistics loaded from model file.")
            elif structure_changed:
                self.logger.info("Training stats were reset due to model structure change.")
            else:
                self.logger.info("No training stats found in model file. Using default/current training stats.");
            if not hasattr(self,
                           'training_stats') or not self.training_stats: self.training_stats = self._get_default_training_stats()

            items_trained_prev = self.training_stats.get('total_items_processed', "N/A")
            success_load_msg = (
                f"Model loaded from: '{abs_filepath}'. Items Previously Trained: {items_trained_prev}. Thresholds: {self.len_thresholds}. ExplRate: {self.exploration_rate:.4f}. SimpleRLE Min_Run: {self.rle_min_encodable_run_length}. TargetDevice: '{self.target_device}', EffectiveGPUUse: {self.use_gpu_acceleration}");
            self.logger.info(success_load_msg);
            self._send_to_gui(success_load_msg);
            return True
        except Exception as el_load:
            error_load_msg = f"Error loading model from '{abs_filepath}': {el_load}. Using new/default model parameters.";
            self.logger.exception(error_load_msg);
            self._send_to_gui(f"ERROR: {error_load_msg}");
            self._reinitialize_state_dependent_vars();
            self.target_device = CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT
            self.use_gpu_acceleration = "GPU" in self.target_device.upper()
            return False