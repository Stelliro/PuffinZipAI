import logging
import os
import time
import traceback
import random
import numpy as np

_PuffinZipAI_base = None
_DummyLogger_base = None
_cpu_rle_compress_base_func = None
_cpu_rle_decompress_base_func = None
_gpu_accelerated_rle_compress_func = None
_gpu_accelerated_rle_decompress_func = None
_calculate_reward_func = None
_RLE_DECOMPRESSION_ERRORS_set = set()

_DEFAULT_LEARNING_RATE = 0.1
_DEFAULT_DISCOUNT_FACTOR = 0.9
_DEFAULT_EXPLORATION_RATE = 1.0
_DEFAULT_EXPLORATION_DECAY_RATE = 0.999
_DEFAULT_MIN_EXPLORATION_RATE = 0.01
_CORE_AI_LOG_FILENAME_fallback = "gpu_agent_fallback.log"
_LOGS_DIR_PATH_fallback = "."
_CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT_fallback = "CPU"
_setup_logger_func = lambda name, filename, **kwargs: logging.getLogger(name)

try:
    from ..ai_core import PuffinZipAI as PZAIBase, DummyLogger as DLBase
    _PuffinZipAI_base = PZAIBase
    _DummyLogger_base = DLBase
    from ..config import (
        DEFAULT_LEARNING_RATE, DEFAULT_DISCOUNT_FACTOR,
        DEFAULT_EXPLORATION_RATE, DEFAULT_EXPLORATION_DECAY_RATE,
        DEFAULT_MIN_EXPLORATION_RATE, CORE_AI_LOG_FILENAME, LOGS_DIR_PATH,
        ACCELERATION_TARGET_DEVICE as CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT
    )
    _DEFAULT_LEARNING_RATE = DEFAULT_LEARNING_RATE
    _DEFAULT_DISCOUNT_FACTOR = DEFAULT_DISCOUNT_FACTOR
    _DEFAULT_EXPLORATION_RATE = DEFAULT_EXPLORATION_RATE
    _DEFAULT_EXPLORATION_DECAY_RATE = DEFAULT_EXPLORATION_DECAY_RATE
    _DEFAULT_MIN_EXPLORATION_RATE = DEFAULT_MIN_EXPLORATION_RATE
    _CORE_AI_LOG_FILENAME_fallback = CORE_AI_LOG_FILENAME
    _LOGS_DIR_PATH_fallback = LOGS_DIR_PATH
    _CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT_fallback = CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT
    from ..logger import setup_logger
    _setup_logger_func = setup_logger
    from ..reward_system import calculate_reward
    _calculate_reward_func = calculate_reward
    from ..rle_constants import RLE_DECOMPRESSION_ERRORS
    _RLE_DECOMPRESSION_ERRORS_set = RLE_DECOMPRESSION_ERRORS
    from ..rle_utils import rle_compress as cpu_rle_compress_imp
    from ..rle_utils import rle_decompress as cpu_rle_decompress_imp
    _cpu_rle_compress_base_func = cpu_rle_compress_imp
    _cpu_rle_decompress_base_func = cpu_rle_decompress_imp
    from .gpu_rle_interface import gpu_accelerated_rle_compress as gpu_rle_c_imp, \
        gpu_accelerated_rle_decompress as gpu_rle_d_imp
    _gpu_accelerated_rle_compress_func = gpu_rle_c_imp
    _gpu_accelerated_rle_decompress_func = gpu_rle_d_imp
    from .gpu_model_utils import get_best_available_gpu_id
except ImportError as e_base_import_gpu_agent:
    print(f"CRITICAL ERROR (gpu_ai_agent.py): Failed to import base AI or RLE components: {e_base_import_gpu_agent}")
    traceback.print_exc()
    if _PuffinZipAI_base is None:
        class PuffinZipAI_Placeholder:
            def __init__(self, len_thresholds=None, learning_rate=None, discount_factor=None, exploration_rate=None,
                         exploration_decay_rate=None, min_exploration_rate=None, rle_min_encodable_run=None,
                         target_device=None, *args, **kwargs):
                self.training_stats = {}
                self.len_thresholds = len_thresholds if len_thresholds is not None else []
                self.q_table = None
                self.logger = _DummyLogger_base() if _DummyLogger_base else logging.getLogger("PZAIPH_Logger")
                self.rle_min_encodable_run_length = rle_min_encodable_run if rle_min_encodable_run is not None else 3
                self.action_names = {0: "RLE_PH", 1: "NoComp_PH", 2: "AdvRLE_PH"}
                self.action_space_size = len(self.action_names)
                self.NUM_LEN_CATS = 1;
                self.NUM_UNIQUE_RATIO_CATS = 1;
                self.NUM_RUN_CATS = 1
                self.learning_rate = learning_rate if learning_rate is not None else _DEFAULT_LEARNING_RATE
                self.discount_factor = discount_factor if discount_factor is not None else _DEFAULT_DISCOUNT_FACTOR
                self.exploration_rate = exploration_rate if exploration_rate is not None else _DEFAULT_EXPLORATION_RATE
                self.exploration_decay_rate = exploration_decay_rate if exploration_decay_rate is not None else _DEFAULT_EXPLORATION_DECAY_RATE
                self.min_exploration_rate = min_exploration_rate if min_exploration_rate is not None else _DEFAULT_MIN_EXPLORATION_RATE
                self.target_device = target_device if target_device is not None else "CPU_PH"
                self.use_gpu_acceleration = False
                self._reinitialize_state_dependent_vars()

            def _reinitialize_state_dependent_vars(self):
                self.action_space_size = len(self.action_names) if hasattr(self, 'action_names') else 3
                self.state_space_size = (
                                                    len(self.len_thresholds) + 1) * self.NUM_UNIQUE_RATIO_CATS * self.NUM_RUN_CATS if hasattr(
                    self, 'len_thresholds') else (1 + 0) * 1 * 1
                if self.state_space_size > 0 and self.action_space_size > 0:
                    self.q_table = np.zeros((self.state_space_size, self.action_space_size))
                else:
                    self.q_table = np.zeros((1, 3))
                self.training_stats = {'rle_chosen_count': 0, 'nocomp_chosen_count': 0, 'advanced_rle_chosen_count': 0,
                                       'decomp_errors': 0, 'total_items_processed': 0, 'cumulative_reward': 0.0,
                                       'reward_history': []}
            def get_config_dict(self):
                return {'target_device': self.target_device, 'len_thresholds': self.len_thresholds,
                        'rle_min_encodable_run': self.rle_min_encodable_run_length, 'learning_rate': self.learning_rate,
                        'discount_factor': self.discount_factor, 'exploration_rate': self.exploration_rate,
                        'exploration_decay_rate': self.exploration_decay_rate,
                        'min_exploration_rate': self.min_exploration_rate}
            def load_model(self, fp=None): return False
            def save_model(self, fp=None): return False
            def _get_state_representation(self, item_text): return 0
            def _choose_action(self, state_idx, use_exploration=True): return 0
            def _update_q_table(self, state_idx, action_idx, reward_val): pass
        _PuffinZipAI_base = PuffinZipAI_Placeholder
    if _DummyLogger_base is None:
        class DummyLogger_Placeholder:
            def _log(self, level, msg, exc_info_flag=False): print(
                f"DummyGPUAgentLog-{level}: {msg}" + (f"\n{traceback.format_exc()}" if exc_info_flag else ""))
            def info(self, msg): self._log("INFO", msg)
            def warning(self, msg): self._log("WARN", msg)
            def error(self, msg, exc_info=False): self._log("ERROR", msg, exc_info_flag=exc_info)
            def critical(self, msg, exc_info=False): self._log("CRITICAL", msg, exc_info_flag=exc_info)
            def debug(self, msg): self._log("DEBUG", msg)
            def exception(self, msg, exc_info_flag=True): self._log("EXCEPTION", msg, exc_info_flag=exc_info_flag)
        _DummyLogger_base = DummyLogger_Placeholder
    if _cpu_rle_compress_base_func is None: _cpu_rle_compress_base_func = lambda d, **k: "ERROR_CPU_RLE_UNAVAILABLE"
    if _cpu_rle_decompress_base_func is None: _cpu_rle_decompress_base_func = lambda d, **k: "ERROR_CPU_RLE_UNAVAILABLE"
    if _gpu_accelerated_rle_compress_func is None: _gpu_accelerated_rle_compress_func = lambda d, **kwargs: "ERROR_GPU_RLE_IMPORT_FAILED_IN_AGENT"
    if _gpu_accelerated_rle_decompress_func is None: _gpu_accelerated_rle_decompress_func = lambda d, **kwargs: "ERROR_GPU_RLE_IMPORT_FAILED_IN_AGENT"
    if _calculate_reward_func is None: _calculate_reward_func = lambda *args: 0.0
    get_best_available_gpu_id = lambda: 0

CUPY_AVAILABLE = False; cp = None
NUMBA_AVAILABLE = False; cuda = None
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError: pass
try:
    from numba import cuda as nb_cuda
    if nb_cuda.is_available(): NUMBA_AVAILABLE = True; cuda = nb_cuda
except ImportError: pass


class PuffinZipAI_GPU(_PuffinZipAI_base):
    def __init__(self,
                 len_thresholds=None, learning_rate=None, discount_factor=None,
                 exploration_rate=None, exploration_decay_rate=None, min_exploration_rate=None,
                 rle_min_encodable_run: int = None, target_device: str = None):
        self.gpu_id = -1
        self.q_table_gpu = None
        lr_to_use = learning_rate if learning_rate is not None else _DEFAULT_LEARNING_RATE
        df_to_use = discount_factor if discount_factor is not None else _DEFAULT_DISCOUNT_FACTOR
        er_to_use = exploration_rate if exploration_rate is not None else _DEFAULT_EXPLORATION_RATE
        erd_to_use = exploration_decay_rate if exploration_decay_rate is not None else _DEFAULT_EXPLORATION_DECAY_RATE
        mer_to_use = min_exploration_rate if min_exploration_rate is not None else _DEFAULT_MIN_EXPLORATION_RATE
        td_to_use = target_device if target_device is not None else _CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT_fallback
        super().__init__(
            len_thresholds=len_thresholds, learning_rate=lr_to_use, discount_factor=df_to_use,
            exploration_rate=er_to_use, exploration_decay_rate=erd_to_use,
            min_exploration_rate=mer_to_use, rle_min_encodable_run=rle_min_encodable_run,
            target_device=td_to_use
        )
        current_logger_assigned_by_super = getattr(self, 'logger', None)
        is_super_logger_a_placeholder = False
        if current_logger_assigned_by_super is None: is_super_logger_a_placeholder = True
        elif isinstance(current_logger_assigned_by_super, type(_DummyLogger_base())): is_super_logger_a_placeholder = True
        elif isinstance(current_logger_assigned_by_super, logging.Logger) and hasattr(current_logger_assigned_by_super, 'name') and current_logger_assigned_by_super.name == 'PZAIPH_Logger': is_super_logger_a_placeholder = True
        if is_super_logger_a_placeholder:
            log_file_full_path_gpu = os.path.join(_LOGS_DIR_PATH_fallback, f"gpu_{_CORE_AI_LOG_FILENAME_fallback}")
            try:
                self.logger = _setup_logger_func(logger_name=f'PuffinZipAI_GPU_SelfLog_{id(self)}', log_filename=log_file_full_path_gpu, log_level=logging.INFO)
                if isinstance(self.logger, logging.Logger) and self.logger.name.endswith(str(id(self))): self.logger.info("PuffinZipAI_GPU re-initialized its own logger using _setup_logger_func.")
                elif not isinstance(self.logger, logging.Logger): self.logger = _DummyLogger_base(); self.logger.error(f"PuffinZipAI_GPU: _setup_logger_func did not return a standard logger. Using _DummyLogger_base for PuffinZipAI_GPU SelfLog.")
            except Exception as e_gpu_log_reinit: print(f"ERROR (PuffinZipAI_GPU): Failed to re-setup its specific logger: {e_gpu_log_reinit}."); self.logger = _DummyLogger_base(); self.logger.error("Fell back to _DummyLogger_base after logger re-init attempt failed.")
        
        if self.use_gpu_acceleration:
            self.logger.info(f"PZAI_GPU: GPU acceleration IS requested by target_device '{self.target_device}'.")
            if CUPY_AVAILABLE:
                self._initialize_gpu_device(self.target_device)
                if self.gpu_id != -1:
                    # --- ADDED: More robust health check ---
                    try:
                        with cp.cuda.Device(self.gpu_id):
                            # Attempt a simple operation to catch runtime/driver issues
                            _ = cp.array([1]) 
                        self.logger.info("PZAI_GPU: CuPy basic functionality check PASSED.")
                    except Exception as e_cupy_runtime:
                        self.logger.critical(f"PZAI_GPU: CuPy runtime test FAILED for GPU {self.gpu_id}. This can be due to driver/toolkit issues (like missing nvrtc64.dll). Disabling GPU ops. Error: {e_cupy_runtime}", exc_info=True)
                        self.use_gpu_acceleration = False
                        self.gpu_id = -1
                    # --- END OF HEALTH CHECK ---

                    if self.use_gpu_acceleration: # Re-check after health test
                        self.logger.info(f"PZAI_GPU: GPU ID {self.gpu_id} is functional. Attempting Q-table transfer.")
                        self._transfer_q_table_to_gpu()
                        if self.q_table_gpu is None:
                            self.logger.warning("PZAI_GPU: Q-table transfer to GPU FAILED. Disabling GPU Q-table ops.")
                            self.use_gpu_acceleration = False
                        else:
                            self.logger.info(f"PZAI_GPU: Q-table successfully transferred to GPU {self.gpu_id}.")
                else:
                    self.logger.warning(f"PZAI_GPU: _initialize_gpu_device did not set a valid GPU ID. Disabling GPU ops.")
                    self.use_gpu_acceleration = False
            else:
                self.logger.warning("PZAI_GPU: Target device indicates GPU, but no GPU libs (CuPy) found. Disabling GPU ops.")
                self.use_gpu_acceleration = False
        else:
            self.logger.info(f"PZAI_GPU: GPU acceleration is NOT enabled by target_device '{self.target_device}'. Using CPU ops.")
        
        self.logger.info(f"PZAI_GPU Final Init State: TargetDevice='{getattr(self, 'target_device', 'N/A')}', GPU Ops Active={self.use_gpu_acceleration}, GPU_ID={self.gpu_id}")

    def _initialize_gpu_device(self, target_device_str: str):
        if CUPY_AVAILABLE:
            try:
                num_gpus = cp.cuda.runtime.getDeviceCount()
                if num_gpus == 0: self.gpu_id = -1; self.logger.info("CuPy: No CUDA devices found."); return
                
                parsed_id = -1 
                if target_device_str.upper() == "GPU_AUTO":
                    parsed_id = get_best_available_gpu_id()
                    if parsed_id == -1: 
                         self.logger.warning(f"GPU_AUTO selected, but get_best_available_gpu_id found no suitable device. Disabling GPU ops.")
                         self.gpu_id = -1; return
                    self.logger.info(f"GPU_AUTO selected. Best available device ID determined to be: {parsed_id}")

                elif target_device_str.upper().startswith("GPU_ID:"):
                    try:
                        selected_id_user = int(target_device_str.split(":")[1])
                        if 0 <= selected_id_user < num_gpus: parsed_id = selected_id_user
                        else: self.logger.warning(f"Requested GPU_ID:{selected_id_user} OOR (0-{num_gpus - 1}). Defaulting to 0."); parsed_id = 0
                    except (ValueError, IndexError): self.logger.warning(f"Invalid GPU_ID format: '{target_device_str}'. Defaulting to 0."); parsed_id = 0
                else: 
                    self.logger.warning(f"Unclear GPU target '{target_device_str}' with CuPy. Defaulting to GPU_ID:0."); parsed_id = 0
                
                if parsed_id == -1 and num_gpus > 0:
                    self.logger.debug("Device ID parsing resulted in -1, but GPUs exist. Fallback to ID 0.")
                    parsed_id = 0
                elif parsed_id == -1:
                    self.gpu_id = -1; return
                
                self.gpu_id = parsed_id; cp.cuda.runtime.setDevice(self.gpu_id); props = cp.cuda.runtime.getDeviceProperties(self.gpu_id)
                device_name = str(props.get('name', 'Unknown CuPy Device')) if isinstance(props, dict) else (props.name.decode() if isinstance(props.name, bytes) else str(props.name))
                total_mem_mb = props.get('totalGlobalMem', 0) / (1024**2) if isinstance(props, dict) else props.totalGlobalMem / (1024**2)
                self.logger.info(f"CuPy active on GPU {self.gpu_id}: {device_name} (TotalMem: {total_mem_mb:.0f}MB)")
            except Exception as e: self.logger.error(f"CuPy error initializing GPU device '{target_device_str}': {e}", exc_info=True); self.gpu_id = -1
        elif NUMBA_AVAILABLE:
            try:
                if cuda and hasattr(cuda, 'gpus') and len(cuda.gpus) > 0: self.gpu_id = 0
                else: self.gpu_id = -1; self.logger.info("Numba: No CUDA devices found.")
            except Exception as e: self.logger.error(f"Numba error detecting GPU: {e}", exc_info=True); self.gpu_id = -1
        else: self.gpu_id = -1
        if self.gpu_id == -1: self.logger.warning(f"Failed to initialize any GPU device for target '{target_device_str}'.")

    def _transfer_q_table_to_gpu(self):
        q_table_cpu = getattr(self, 'q_table', None)
        if self.use_gpu_acceleration and self.gpu_id != -1 and CUPY_AVAILABLE and q_table_cpu is not None:
            try:
                with cp.cuda.Device(self.gpu_id): self.q_table_gpu = cp.asarray(q_table_cpu)
                self.logger.info(f"Q-table (shape: {q_table_cpu.shape}, type: {self.q_table_gpu.dtype}) transferred to GPU ID: {self.gpu_id} via CuPy.")
            except Exception as e_transfer: self.logger.error(f"Failed to transfer Q-table to GPU {self.gpu_id} with CuPy: {e_transfer}", exc_info=True); self.q_table_gpu = None; self.use_gpu_acceleration = False
        elif q_table_cpu is None: self.logger.warning("Cannot transfer Q-table to GPU: Base Q-table (self.q_table) is None.")
        elif not self.use_gpu_acceleration or self.gpu_id == -1 or not CUPY_AVAILABLE: self.logger.debug("Skipping Q-table to GPU transfer: Conditions not met.")

    def _reinitialize_state_dependent_vars(self):
        super()._reinitialize_state_dependent_vars()
        if hasattr(self, 'use_gpu_acceleration') and self.use_gpu_acceleration and hasattr(self, 'gpu_id') and self.gpu_id != -1:
            self.logger.debug(f"PZAI_GPU: Post-reinit, attempting Q-table transfer to GPU {self.gpu_id}.")
            self._transfer_q_table_to_gpu()
        else: self.logger.debug("PZAI_GPU: Post-reinit, GPU Q-table transfer skipped (conditions not met).")

    def _choose_action(self, state_idx, use_exploration=True):
        action_space_size = getattr(self, 'action_space_size', 3)
        exploration_rate_val = getattr(self, 'exploration_rate', _DEFAULT_EXPLORATION_RATE)
        action_names_map = getattr(self, 'action_names', {})
        if self.use_gpu_acceleration and self.q_table_gpu is not None and CUPY_AVAILABLE and self.gpu_id != -1:
            if use_exploration and random.random() < exploration_rate_val: action_idx = random.randint(0, action_space_size - 1)
            else:
                try:
                    with cp.cuda.Device(self.gpu_id): q_row_gpu = self.q_table_gpu[state_idx]; action_idx_gpu = cp.argmax(q_row_gpu)
                    action_idx = int(action_idx_gpu.get())
                except Exception as e_gpu_argmax:
                    self.logger.warning(f"Error during GPU Q-table argmax: {e_gpu_argmax}. Falling back to CPU.");
                    q_table_cpu = getattr(self, 'q_table', None)
                    if q_table_cpu is not None and isinstance(q_table_cpu, np.ndarray) and q_table_cpu.size > 0 and state_idx < q_table_cpu.shape[0]: action_idx = np.argmax(q_table_cpu[state_idx])
                    else: self.logger.error("CPU Q-table unavailable/invalid for fallback in _choose_action. Choosing random action."); action_idx = random.randint(0, action_space_size - 1)
        else: action_idx = super()._choose_action(state_idx, use_exploration=use_exploration)
        if use_exploration:
            if not hasattr(self, 'training_stats') or not isinstance(self.training_stats, dict): self.training_stats = self._get_default_training_stats()
            for key in ['rle_chosen_count', 'nocomp_chosen_count', 'advanced_rle_chosen_count']: self.training_stats.setdefault(key, 0)
            chosen_action_name = action_names_map.get(action_idx)
            if chosen_action_name == "RLE": self.training_stats['rle_chosen_count'] += 1
            elif chosen_action_name == "NoCompression": self.training_stats['nocomp_chosen_count'] += 1
            elif chosen_action_name == "AdvancedRLE": self.training_stats['advanced_rle_chosen_count'] += 1
        return action_idx

    def _update_q_table(self, state_idx, action_idx, reward_val):
        learning_rate_val = getattr(self, 'learning_rate', _DEFAULT_LEARNING_RATE)
        q_table_cpu = getattr(self, 'q_table', None)
        if self.use_gpu_acceleration and self.q_table_gpu is not None and CUPY_AVAILABLE and self.gpu_id != -1:
            try:
                with cp.cuda.Device(self.gpu_id): current_q_gpu = self.q_table_gpu[state_idx, action_idx]; new_q_gpu = current_q_gpu + learning_rate_val * (reward_val - current_q_gpu); self.q_table_gpu[state_idx, action_idx] = new_q_gpu
                return
            except Exception as e_gpu_update: self.logger.warning(f"Error during GPU Q-table update: {e_gpu_update}. Falling back to CPU.")
        if q_table_cpu is not None and isinstance(q_table_cpu, np.ndarray) and q_table_cpu.size > 0 and state_idx < q_table_cpu.shape[0] and action_idx < q_table_cpu.shape[1]: super()._update_q_table(state_idx, action_idx, reward_val)
        else: self.logger.error(f"CPU Q-table not available/invalid for fallback update for S={state_idx}, A={action_idx}. Update skipped. q_table type: {type(q_table_cpu)}")

    def _handle_item_processing_for_training(self, item_text, counter_info=""):
        action_names_map = getattr(self, 'action_names', {0: "RLE_PH", 1: "NoComp_PH", 2: "AdvRLE_PH"})
        rle_min_encodable_run_length_val = getattr(self, 'rle_min_encodable_run_length', 3)
        if not hasattr(self, 'training_stats') or not isinstance(self.training_stats, dict): self.training_stats = self._get_default_training_stats() if hasattr(super(), '_get_default_training_stats') else {'decomp_errors': 0, 'total_items_processed': 0, 'cumulative_reward': 0.0, 'rle_chosen_count': 0, 'nocomp_chosen_count': 0, 'advanced_rle_chosen_count': 0, 'reward_history': []}
        if not action_names_map: self.logger.error("action_names not initialized in PuffinZipAI_GPU. Cannot process item."); self.training_stats.setdefault('decomp_errors', 0); self.training_stats['decomp_errors'] += 1; return (0, 0, 0.0, "ERROR", "ERROR_NO_ACTIONS", "ERROR_NO_ACTIONS", True)
        
        state_idx = self._get_state_representation(item_text)
        action_idx = self._choose_action(state_idx, use_exploration=True)
        action_name = action_names_map.get(action_idx, f"UnknownAction({action_idx})")
        
        original_size = len(item_text)
        compressed_text_final, decompressed_text_final = "", ""
        rle_error_code_final = None
        start_time_ns = time.perf_counter_ns()
        
        rle_min_override_val = rle_min_encodable_run_length_val if action_idx == 0 and action_name == action_names_map.get(0, "RLE") else None
        
        if action_idx == 0 or action_idx == 2: # RLE or AdvancedRLE action
            rle_method_to_use = "simple" if action_idx == 0 else "advanced"
            attempt_gpu_rle = self.use_gpu_acceleration and self.gpu_id != -1
            
            # --- Compression Step with Fallback ---
            compressed_via_gpu = False
            if attempt_gpu_rle and _gpu_accelerated_rle_compress_func:
                compressed_text_final = _gpu_accelerated_rle_compress_func(
                    item_text, method=rle_method_to_use, min_run_len_override=rle_min_override_val, 
                    use_gpu=True, gpu_id=self.gpu_id)
                if not isinstance(compressed_text_final, str) or not compressed_text_final.startswith("ERROR_GPU_"):
                    compressed_via_gpu = True
            
            if not compressed_via_gpu: # Fallback to CPU compression
                if _cpu_rle_compress_base_func:
                    compressed_text_final = _cpu_rle_compress_base_func(
                        item_text, method=rle_method_to_use, min_run_len_override=rle_min_override_val)
                else: # Should not happen if imports work
                    self.logger.error("CPU RLE compress function unavailable! Defaulting to no compression.")
                    compressed_text_final = item_text

            # --- Decompression Step with Fallback ---
            decompressed_via_gpu = False
            if attempt_gpu_rle and _gpu_accelerated_rle_decompress_func:
                decompressed_text_final = _gpu_accelerated_rle_decompress_func(
                    compressed_text_final, method=rle_method_to_use, min_run_len_override=rle_min_override_val,
                    expected_output_size_hint=original_size, use_gpu=True, gpu_id=self.gpu_id)
                if not isinstance(decompressed_text_final, str) or not decompressed_text_final.startswith("ERROR_GPU_"):
                    decompressed_via_gpu = True

            if not decompressed_via_gpu: # Fallback to CPU decompression
                if _cpu_rle_decompress_base_func:
                    decompressed_text_final = _cpu_rle_decompress_base_func(
                        compressed_text_final, method=rle_method_to_use, min_run_len_override=rle_min_override_val)
                else: # Should not happen
                    self.logger.error("CPU RLE decompress function unavailable!")
                    decompressed_text_final = "ERROR_CPU_RLE_UNAVAILABLE"

        elif action_idx == 1: # NoCompression action
            compressed_text_final, decompressed_text_final = item_text, item_text
        else: # Unhandled action
            self.logger.warning(f"PZAI_GPU: Unhandled action '{action_name}' (idx {action_idx}). Defaulting to NoCompression.")
            compressed_text_final, decompressed_text_final = item_text, item_text
            action_name = action_names_map.get(1, "NoCompression_Fallback")

        if decompressed_text_final in _RLE_DECOMPRESSION_ERRORS_set: rle_error_code_final = decompressed_text_final
        
        end_time_ns = time.perf_counter_ns(); processing_time_ms = (end_time_ns - start_time_ns) / 1e6
        current_reward = _calculate_reward_func(item_text, compressed_text_final, decompressed_text_final, action_name, processing_time_ms, rle_error_code_final) if _calculate_reward_func else 0.0
        
        decompression_mismatch = False
        self.training_stats.setdefault('decomp_errors', 0); self.training_stats.setdefault('total_items_processed', 0); self.training_stats.setdefault('cumulative_reward', 0.0)
        if (action_name == "RLE" or action_name == "AdvancedRLE") and (rle_error_code_final or (decompressed_text_final != item_text)): decompression_mismatch = True; self.training_stats['decomp_errors'] += 1
        self.training_stats['total_items_processed'] += 1; self.training_stats['cumulative_reward'] += current_reward
        
        return (state_idx, action_idx, current_reward, item_text[:50], action_name, compressed_text_final[:50], decompressed_text_final[:50], decompression_mismatch)

    def save_model(self, fp=None):
        if self.use_gpu_acceleration and self.q_table_gpu is not None and CUPY_AVAILABLE and self.gpu_id != -1:
            self.logger.info(f"PZAI_GPU: Syncing Q-table from GPU {self.gpu_id} to CPU before saving.")
            try:
                with cp.cuda.Device(self.gpu_id): self.q_table = cp.asnumpy(self.q_table_gpu)
            except Exception as e_sync: self.logger.error(f"PZAI_GPU: Failed to sync GPU Q-table to CPU for saving: {e_sync}. Model might save older CPU Q-table.", exc_info=True)
        return super().save_model(fp)

    def load_model(self, fp=None):
        load_successful_cpu_part = super().load_model(fp)
        if load_successful_cpu_part and self.use_gpu_acceleration and self.gpu_id != -1:
            self.logger.info(f"PZAI_GPU: Model CPU part loaded. Attempting to transfer new Q-table to GPU {self.gpu_id}...")
            self._transfer_q_table_to_gpu()
            if self.q_table_gpu is None and self.use_gpu_acceleration: self.logger.warning("PZAI_GPU: Post-load Q-table transfer to GPU FAILED. GPU Q-table ops now OFF for this instance."); self.use_gpu_acceleration = False
        elif load_successful_cpu_part and not self.use_gpu_acceleration: self.logger.info("PZAI_GPU: Model loaded. GPU use not active. Q-table remains on CPU.")
        return load_successful_cpu_part

    def clone_core_model(self):
        config_params_for_clone = {}; is_placeholder_base = not callable(getattr(super(), "display_q_table_summary", None))
        if is_placeholder_base or not hasattr(super(), 'get_config_dict'):
            config_params_for_clone = {
                'len_thresholds': getattr(self, 'len_thresholds', []), 'learning_rate': getattr(self, 'learning_rate', _DEFAULT_LEARNING_RATE),
                'discount_factor': getattr(self, 'discount_factor', _DEFAULT_DISCOUNT_FACTOR), 'exploration_rate': getattr(self, 'exploration_rate', _DEFAULT_EXPLORATION_RATE),
                'exploration_decay_rate': getattr(self, 'exploration_decay_rate', _DEFAULT_EXPLORATION_DECAY_RATE), 'min_exploration_rate': getattr(self, 'min_exploration_rate', _DEFAULT_MIN_EXPLORATION_RATE),
                'rle_min_encodable_run': getattr(self, 'rle_min_encodable_run_length', 3), 'target_device': getattr(self, 'target_device', _CONFIG_ACCELERATION_TARGET_DEVICE_DEFAULT_fallback)
            }
            if hasattr(self, 'logger') and self.logger: self.logger.warning("clone_core_model: Using placeholder-derived or manually constructed config for clone.")
        else: config_params_for_clone = super().get_config_dict()
        cloned_ai_agent = PuffinZipAI_GPU(**config_params_for_clone)
        source_q_for_clone = None; q_table_cpu = getattr(self, 'q_table', None)
        if self.use_gpu_acceleration and self.q_table_gpu is not None and CUPY_AVAILABLE:
            try:
                with cp.cuda.Device(self.gpu_id): source_q_for_clone = cp.asnumpy(self.q_table_gpu)
                if hasattr(self, 'logger') and self.logger: self.logger.debug("Clone: Source is GPU agent, using its q_table_gpu state as numpy for clone.")
            except Exception as e_asnumpy:
                if hasattr(self, 'logger') and self.logger: self.logger.error(f"Error converting GPU Q-table to numpy for clone: {e_asnumpy}. Using CPU Q-table if available.", exc_info=True)
                source_q_for_clone = q_table_cpu
        else:
            source_q_for_clone = q_table_cpu
            if hasattr(self, 'logger') and self.logger: self.logger.debug("Clone: Source is CPU-based or its GPU Q-table inactive/unavailable, using its CPU q_table state for clone.")
        if source_q_for_clone is not None:
            cloned_q_table_cpu = getattr(cloned_ai_agent, 'q_table', None)
            if cloned_q_table_cpu is not None and cloned_q_table_cpu.shape == source_q_for_clone.shape:
                cloned_ai_agent.q_table = np.copy(source_q_for_clone)
                if cloned_ai_agent.use_gpu_acceleration and cloned_ai_agent.gpu_id != -1:
                    if hasattr(self, 'logger') and self.logger: self.logger.debug(f"Clone: Attempting to transfer its (copied) Q-table to its GPU {cloned_ai_agent.gpu_id}.")
                    cloned_ai_agent._transfer_q_table_to_gpu()
                    if cloned_ai_agent.q_table_gpu is None:
                        if hasattr(self, 'logger') and self.logger: self.logger.warning("Clone: Transfer of Q-table to clone's GPU FAILED.")
                    else:
                        if hasattr(self, 'logger') and self.logger: self.logger.debug("Clone: Q-table successfully transferred to clone's GPU.")
            else:
                if hasattr(self, 'logger') and self.logger: self.logger.warning(f"Clone Q-table shape mismatch or None (Source: {source_q_for_clone.shape if source_q_for_clone is not None else 'None'}, Clone Initial: {cloned_q_table_cpu.shape if cloned_q_table_cpu is not None else 'None'}). Clone reinitialized with fresh Q-table.")
                cloned_ai_agent._reinitialize_state_dependent_vars()
        else:
            if hasattr(self, 'logger') and self.logger: self.logger.warning("Clone: Source Q-table was None. Clone will have a new zeroed Q-table (via its _reinitialize_state_dependent_vars).")
            if not hasattr(cloned_ai_agent, 'q_table') or cloned_ai_agent.q_table is None: cloned_ai_agent._reinitialize_state_dependent_vars()
        if hasattr(self, 'logger') and self.logger: self.logger.info(f"Cloned PuffinZipAI_GPU. Clone Target: '{cloned_ai_agent.target_device}', GPUOps: {cloned_ai_agent.use_gpu_acceleration}, GPU_ID: {cloned_ai_agent.gpu_id}")
        return cloned_ai_agent