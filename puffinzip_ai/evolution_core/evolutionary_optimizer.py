# PuffinZipAI_Project/puffinzip_ai/evolution_core/evolutionary_optimizer.py
import logging
import os
import queue
import random
import threading
import time
import traceback
import numpy as np
import importlib
import pickle

try:
    from ..gpu_core.gpu_ai_agent import cp
except ImportError:
    cp = None

try:
    from .. import PuffinZipAI
except ImportError as e_core_init:

    PuffinZipAI = None

from ..config import (
    DEFAULT_POPULATION_SIZE, DEFAULT_NUM_GENERATIONS, DEFAULT_MUTATION_RATE, DEFAULT_ELITISM_COUNT,
    DEFAULT_SELECTION_STRATEGY, STAGNATION_GENERATIONS_THRESHOLD,
    DEFAULT_LEN_THRESHOLDS,
    MUTATION_RATE_BOOST_FACTOR, MUTATION_RATE_DECAY_FACTOR, EVOLUTIONARY_AI_LOG_FILENAME,
    DEFAULT_LEARNING_RATE, DEFAULT_DISCOUNT_FACTOR, DEFAULT_EXPLORATION_RATE,
    DEFAULT_EXPLORATION_DECAY_RATE, DEFAULT_MIN_EXPLORATION_RATE,
    BENCHMARK_DATASET_PATH, GENERATED_BENCHMARK_DEFAULT_PATH, LOGS_DIR_PATH,
    HYPERMUTATION_STAGNATION_THRESHOLD, HYPERMUTATION_FRACTION,
    HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB, HYPERMUTATION_PARAM_STRENGTH_FACTOR,
    RLE_MIN_RUN_INIT_MIN, RLE_MIN_RUN_INIT_MAX,
    RANDOM_IMMIGRANT_INTERVAL, RANDOM_IMMIGRANT_FRACTION,
    ELS_LOG_PREFIX as CFG_ELS_LOG_PREFIX,
    ELS_STATS_MSG_PREFIX as CFG_ELS_STATS_MSG_PREFIX,
    MIN_THRESHOLDS_COUNT, MAX_THRESHOLDS_COUNT, ADVANCED_CROSSOVER_PROBABILITY,
    DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT,
    DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS as CONFIG_DYNAMIC_BENCH_REFRESH,
    INITIAL_BENCHMARK_COMPLEXITY_LEVEL as CONFIG_INITIAL_BENCH_COMPLEXITY,
    ACCELERATION_TARGET_DEVICE as CONFIG_ACCELERATION_TARGET_DEVICE,
    DEFAULT_ADDITIONAL_ELS_GENERATIONS
)
from ..logger import setup_logger

BenchmarkItemEvaluator = None;
DEFAULT_BENCHMARK_REPETITIONS = 1;
DataComplexity = None
try:
    from ..utils.benchmark_evaluator import BenchmarkItemEvaluator as BIE_temp, \
        DEFAULT_BENCHMARK_REPETITIONS as DBR_temp, DataComplexity as DC_temp

    BenchmarkItemEvaluator = BIE_temp;
    DEFAULT_BENCHMARK_REPETITIONS = DBR_temp;
    DataComplexity = DC_temp
except ModuleNotFoundError:
    pass
except ImportError as e_be:

    pass
except Exception as e_other_be:

    pass

EvolvingAgent = None
try:
    from .individual_agent import EvolvingAgent as EA_temp

    EvolvingAgent = EA_temp
except ImportError as e_ia:

    raise


def _placeholder_selection_method(error_message_detail, *args, **kwargs):
    if args and isinstance(args[0], list) and args[0]:
        num_to_select_fallback = 1
        if len(args) > 1 and isinstance(args[1], int) and args[1] > 0: num_to_select_fallback = args[1]
        return args[0][:min(len(args[0]), num_to_select_fallback)]
    raise NotImplementedError(f"Selection method not loaded due to: {error_message_detail}")


try:
    from .selection_methods import tournament_selection, roulette_wheel_selection, rank_selection
except ImportError as e_sm:

    def tournament_selection(*args, **kwargs):
        return _placeholder_selection_method(f"tournament_selection not loaded: {e_sm}", *args, **kwargs)


    def roulette_wheel_selection(*args, **kwargs):
        return _placeholder_selection_method(f"roulette_wheel_selection not loaded: {e_sm}", *args, **kwargs)


    def rank_selection(*args, **kwargs):
        return _placeholder_selection_method(f"rank_selection not loaded: {e_sm}", *args, **kwargs)

try:
    from .crossover_methods import apply_crossover as advanced_crossover_pipeline
except ImportError as e_co_real:

    def advanced_crossover_pipeline(*args, **kwargs):
        parent1_ai = kwargs.get('parent1_ai', args[0] if args else None);
        parent2_ai = kwargs.get('parent2_ai', args[1] if len(args) > 1 else None)

        if not (parent1_ai and hasattr(parent1_ai, 'clone_core_model') and parent2_ai and hasattr(parent2_ai,
                                                                                                  'clone_core_model')): raise AttributeError(
            "Parent AI(s) missing or 'clone_core_model' method not found in placeholder.")
        return parent1_ai.clone_core_model(), parent2_ai.clone_core_model()

apply_mutations_func = None
apply_hypermutation_func = None

try:
    from .mutation_methods import apply_mutations, apply_hypermutation

    apply_mutations_func = apply_mutations
    apply_hypermutation_func = apply_hypermutation

except ImportError as e_mu_composite_direct:

    try:
        mutation_methods_module = importlib.import_module("puffinzip_ai.evolution_core.mutation_methods")
        apply_mutations_func = getattr(mutation_methods_module, 'apply_mutations', None)
        apply_hypermutation_func = getattr(mutation_methods_module, 'apply_hypermutation', None)
        if apply_mutations_func and apply_hypermutation_func:
            pass
        else:
            raise ImportError("Functions not found in dynamically imported mutation_methods module.")
    except Exception as e_mu_composite_dynamic:

        def apply_mutations_placeholder(agent, rate_config_dict):

            return False


        def apply_hypermutation_placeholder(agent, hyper_config_dict=None):
            pass


        apply_mutations_func = apply_mutations_placeholder
        apply_hypermutation_func = apply_hypermutation_placeholder


class EvolutionaryOptimizer:
    def __init__(self, population_size=None, num_generations=None, mutation_rate=None, elitism_count=None,
                 gui_output_queue=None, gui_stop_event=None, benchmark_items=None, benchmark_path=None,
                 tuned_params=None, dynamic_benchmarking_active: bool = None,
                 benchmark_refresh_interval_gens: int = None, initial_benchmark_target_size_mb: float = None,
                 initial_benchmark_fixed_complexity_name: str = None, use_gpu_acceleration: bool = None,
                 target_device: str = None):
        log_file_path = os.path.join(LOGS_DIR_PATH, EVOLUTIONARY_AI_LOG_FILENAME)
        self.logger = setup_logger(logger_name='EvolutionaryOptimizer', log_filename=log_file_path,
                                   log_level=logging.DEBUG)
        self.tuned_params = tuned_params if tuned_params is not None else {}
        self.config = {
            'DEFAULT_POPULATION_SIZE': DEFAULT_POPULATION_SIZE, 'DEFAULT_NUM_GENERATIONS': DEFAULT_NUM_GENERATIONS,
            'DEFAULT_ADDITIONAL_ELS_GENERATIONS': DEFAULT_ADDITIONAL_ELS_GENERATIONS,
            'DEFAULT_MUTATION_RATE': DEFAULT_MUTATION_RATE,
            'DEFAULT_ELITISM_COUNT': DEFAULT_ELITISM_COUNT, 'DEFAULT_SELECTION_STRATEGY': DEFAULT_SELECTION_STRATEGY,
            'STAGNATION_GENERATIONS_THRESHOLD': STAGNATION_GENERATIONS_THRESHOLD,
            'MUTATION_RATE_BOOST_FACTOR': MUTATION_RATE_BOOST_FACTOR,
            'MUTATION_RATE_DECAY_FACTOR': MUTATION_RATE_DECAY_FACTOR,
            'HYPERMUTATION_STAGNATION_THRESHOLD': HYPERMUTATION_STAGNATION_THRESHOLD,
            'HYPERMUTATION_FRACTION': HYPERMUTATION_FRACTION,
            'HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB': HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB,
            'HYPERMUTATION_PARAM_STRENGTH_FACTOR': HYPERMUTATION_PARAM_STRENGTH_FACTOR,
            'RLE_MIN_RUN_INIT_MIN': RLE_MIN_RUN_INIT_MIN,
            'RLE_MIN_RUN_INIT_MAX': RLE_MIN_RUN_INIT_MAX, 'RANDOM_IMMIGRANT_INTERVAL': RANDOM_IMMIGRANT_INTERVAL,
            'RANDOM_IMMIGRANT_FRACTION': RANDOM_IMMIGRANT_FRACTION, 'MIN_THRESHOLDS_COUNT': MIN_THRESHOLDS_COUNT,
            'MAX_THRESHOLDS_COUNT': MAX_THRESHOLDS_COUNT,
            'ADVANCED_CROSSOVER_PROBABILITY': ADVANCED_CROSSOVER_PROBABILITY,
            'DEFAULT_LEARNING_RATE': DEFAULT_LEARNING_RATE, 'DEFAULT_DISCOUNT_FACTOR': DEFAULT_DISCOUNT_FACTOR,
            'DEFAULT_EXPLORATION_RATE': DEFAULT_EXPLORATION_RATE,
            'DEFAULT_EXPLORATION_DECAY_RATE': DEFAULT_EXPLORATION_DECAY_RATE,
            'DEFAULT_MIN_EXPLORATION_RATE': DEFAULT_MIN_EXPLORATION_RATE, 'ELS_LOG_PREFIX': CFG_ELS_LOG_PREFIX,
            'ELS_STATS_MSG_PREFIX': CFG_ELS_STATS_MSG_PREFIX,
            'DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT': DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT,
            'DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS': CONFIG_DYNAMIC_BENCH_REFRESH,
            'INITIAL_BENCHMARK_COMPLEXITY_LEVEL': CONFIG_INITIAL_BENCH_COMPLEXITY,
            'ACCELERATION_TARGET_DEVICE': CONFIG_ACCELERATION_TARGET_DEVICE
        }
        self.els_target_device = target_device if target_device is not None else self.config_get(
            'ACCELERATION_TARGET_DEVICE')
        if use_gpu_acceleration is not None:
            self.logger.warning("EvolutionaryOptimizer init: 'use_gpu_acceleration' passed. Prefer 'target_device'.")
            if use_gpu_acceleration and self.els_target_device == "CPU":
                self.els_target_device = "GPU_AUTO"
            elif not use_gpu_acceleration and "GPU" in self.els_target_device.upper():
                self.els_target_device = "CPU"

        self.dynamic_benchmarking_active = dynamic_benchmarking_active if dynamic_benchmarking_active is not None else self.config_get(
            'DYNAMIC_BENCHMARKING_ACTIVE_BY_DEFAULT')
        self.benchmark_refresh_interval_gens = benchmark_refresh_interval_gens if benchmark_refresh_interval_gens is not None else self.config_get(
            'DYNAMIC_BENCHMARK_REFRESH_INTERVAL_GENS')
        self.initial_benchmark_target_size_mb = initial_benchmark_target_size_mb
        self.initial_benchmark_fixed_complexity_name = initial_benchmark_fixed_complexity_name
        self.initial_benchmark_complexity_config_default = self.config_get('INITIAL_BENCHMARK_COMPLEXITY_LEVEL')

        self.logger.info(
            f"EvolutionaryOptimizer initializing. DynamicBM: {self.dynamic_benchmarking_active} (Refresh: {self.benchmark_refresh_interval_gens} gens). InitBenchSizeMB: {self.initial_benchmark_target_size_mb}, InitBenchComplex: '{self.initial_benchmark_fixed_complexity_name}'. ELS AI Agents TargetDevice: '{self.els_target_device}'")

        self.default_elitism_count_from_config = elitism_count if elitism_count is not None else self.config_get(
            'DEFAULT_ELITISM_COUNT')
        self.population_size = population_size if population_size is not None else self.config_get(
            'DEFAULT_POPULATION_SIZE')
        self.initial_num_generations = num_generations if num_generations is not None else self.config_get(
            'DEFAULT_NUM_GENERATIONS')
        self.base_mutation_rate = mutation_rate if mutation_rate is not None else self.config_get(
            'DEFAULT_MUTATION_RATE')
        self.current_mutation_rate = self.base_mutation_rate;
        self.elitism_count = self.default_elitism_count_from_config
        self.selection_strategy = self.config_get('DEFAULT_SELECTION_STRATEGY')
        self.gui_output_queue = gui_output_queue;
        self.gui_stop_event = gui_stop_event if gui_stop_event is not None else threading.Event();
        self.pause_event = threading.Event()
        self.stagnation_counter = 0;
        self.last_best_fitness_improvement_gen = 0;
        self.hypermutation_active_until_gen = 0
        self.fitness_history_per_generation = [];
        self.best_agent_overall = None;
        self.best_fitness_overall = float('-inf')
        self.average_fitness_overall_history = [];
        self.total_generations_elapsed = 0;
        self.seeded_champion_configs = [];
        self.population = []

        actual_static_benchmark_path = benchmark_path or BENCHMARK_DATASET_PATH or GENERATED_BENCHMARK_DEFAULT_PATH
        if BenchmarkItemEvaluator is not None:
            self.benchmark_evaluator = BenchmarkItemEvaluator(
                benchmark_dataset_path=actual_static_benchmark_path if not self.dynamic_benchmarking_active else None,
                logger_instance=self.logger, tuned_params=self.tuned_params,
                dynamic_benchmarking=self.dynamic_benchmarking_active
            )
            if benchmark_items:
                self.benchmark_evaluator.set_custom_benchmark_items(benchmark_items)
            elif not self.dynamic_benchmarking_active and not self.benchmark_evaluator.benchmark_items:
                self.logger.warning(
                    "Static benchmark mode selected, but no static items loaded. Attempting fallback to generate an initial dynamic set (once).")
                if not self.benchmark_evaluator.load_benchmark_data():
                    self.benchmark_evaluator.dynamic_benchmarking_enabled = True
                    self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(population_average_fitness=-1000,
                                                                                      num_items_to_generate=20,
                                                                                      target_item_size_mb_override=self.initial_benchmark_target_size_mb,
                                                                                      fixed_complexity_override_name=self.initial_benchmark_fixed_complexity_name or self.initial_benchmark_complexity_config_default)
                    self.benchmark_evaluator.dynamic_benchmarking_enabled = False
                    if not self.benchmark_evaluator.benchmark_items: self.logger.error(
                        "Fallback dynamic benchmark generation also failed to produce items!")
            elif self.dynamic_benchmarking_active and not self.benchmark_evaluator.benchmark_items:
                self.logger.info(
                    "Dynamic benchmarking enabled. Items will be generated by _maybe_refresh_dynamic_benchmarks using ELS preferences.")
        else:
            self.logger.error("BenchmarkItemEvaluator class not available. ELS evaluation cannot proceed.");
            self.benchmark_evaluator = None

    def config_get(self, key, default_val=None):
        return self.config.get(key, default_val)

    def clear_pause_resume_events(self):
        self.pause_event.clear()
        if self.gui_stop_event:
            self.gui_stop_event.clear()
        self.logger.debug("Pause and GUI Stop events cleared for ELS.")

    def add_champion_from_config(self, champion_config_dict: dict):
        if not isinstance(champion_config_dict, dict): self.logger.warning(
            "Failed to add champion: Config must be dict."); self._send_to_gui(
            f"{self.config_get('ELS_LOG_PREFIX')} Error: Champion config invalid.", log_level="error");return False
        required_keys = ['len_thresholds', 'learning_rate', 'discount_factor', 'exploration_rate',
                         'exploration_decay_rate', 'min_exploration_rate', 'rle_min_encodable_run']
        if not all(key in champion_config_dict for key in required_keys): missing = [key for key in required_keys if
                                                                                     key not in champion_config_dict]; self.logger.warning(
            f"Failed to add champion: Seed config missing: {missing}."); self._send_to_gui(
            f"{self.config_get('ELS_LOG_PREFIX')} Error: Champion seed config incomplete (missing: {missing}).",
            log_level="error"); return False
        self.seeded_champion_configs.append(dict(champion_config_dict));
        self.logger.info(f"Champion config added. Thresh: {champion_config_dict.get('len_thresholds')}");
        self._send_to_gui(f"{self.config_get('ELS_LOG_PREFIX')} Champion model prepared for seeding.",
                          log_level="info");
        return True

    def _send_to_gui(self, message, log_level="info"):
        log_prefix = self.config_get('ELS_LOG_PREFIX', "[ELS]");
        formatted_message = f"{log_prefix} {str(message)}"
        if self.gui_output_queue:
            try:
                self.gui_output_queue.put_nowait(formatted_message)
            except queue.Full:
                self.logger.warning(f"{log_prefix}: GUI queue full: {str(message)[:100]}")
        else:
            print(formatted_message)
        if hasattr(self.logger, log_level):
            getattr(self.logger, log_level)(str(message))
        else:
            self.logger.info(f"({log_level.upper()}) {str(message)}")

    def _send_fitness_history_to_gui(self):
        if self.gui_output_queue:
            serializable_history = []
            for entry in self.fitness_history_per_generation:
                try:
                    if isinstance(entry, (tuple, list)) and len(entry) == 5 and all(
                            isinstance(x, (int, float, np.number)) for x in entry):
                        serializable_history.append(
                            (int(entry[0]), float(entry[1]), float(entry[2]), float(entry[3]), float(entry[4])))
                    else:
                        self.logger.warning(f"Invalid entry in fitness_history (expected 5-tuple): {entry}. Skipping.")
                except (TypeError, ValueError) as e_ser:
                    self.logger.warning(f"Error serializing fitness entry {entry}: {e_ser}. Skipping.")
            history_str = repr(serializable_history);
            stats_prefix = self.config_get('ELS_STATS_MSG_PREFIX', "[ELS_FITNESS_HISTORY]")
            try:
                self.gui_output_queue.put_nowait(f"{stats_prefix} {history_str}")
            except queue.Full:
                self.logger.warning(f"{stats_prefix}: GUI stats queue full (fitness history).")

    def _pre_run_hardware_check(self) -> bool:
        """Checks if GPU is selected and if agents are correctly initialized for it."""
        is_gpu_target = "GPU" in self.els_target_device.upper()
        if not is_gpu_target:
            self.logger.info("Pre-run check: CPU target selected. Skipping GPU checks.")
            return True

        self.logger.info(f"Pre-run check: GPU target '{self.els_target_device}' selected. Verifying agent states.")
        if not self.population:
            self.logger.warning("Pre-run check: No population exists yet.")
            return True

        for agent in self.population:
            if not hasattr(agent, 'puffin_ai') or not agent.puffin_ai:
                self._send_to_gui(f"FATAL ELS PRE-CHECK: Agent {agent.agent_id} has no AI core.", "critical")
                return False
            
            ai = agent.puffin_ai
            # `use_gpu_acceleration` is the definitive flag set by the agent's __init__
            if not getattr(ai, 'use_gpu_acceleration', False):
                # The agent itself determined it cannot run on GPU, despite the ELS target
                msg = f"FATAL ELS PRE-CHECK: ELS target is '{self.els_target_device}', but agent '{agent.agent_id}' has disabled GPU acceleration due to an internal error (e.g., failed CuPy health check). See agent logs for details."
                self.logger.critical(msg)
                self._send_to_gui(msg, "critical")
                return False

        self.logger.info("Pre-run check: All agents appear correctly configured for the selected GPU target.")
        return True

    def _create_initial_population(self):
        if PuffinZipAI is None or EvolvingAgent is None: self.logger.critical(
            "PuffinZipAI/EvolvingAgent not available for population creation.");self._send_to_gui(
            "CRITICAL ERROR: Core AI components missing for ELS pop.", "critical"); return []
        self.logger.info(
            f"Creating initial population. Target: {self.population_size}. Seeded: {len(self.seeded_champion_configs)}");
        self._send_to_gui(f"Creating initial ELS population (target {self.population_size})...")
        current_population = [];
        num_seeded = 0
        for champ_idx, champ_config in enumerate(self.seeded_champion_configs):
            if self.gui_stop_event.is_set(): self.logger.info("Population creation (seeding) interrupted."); break
            if len(current_population) >= self.population_size: break
            try:
                ai_init_params = {'len_thresholds': champ_config.get('len_thresholds', list(
                    self.config_get('DEFAULT_LEN_THRESHOLDS', DEFAULT_LEN_THRESHOLDS))),
                                  'learning_rate': champ_config.get('learning_rate',
                                                                    self.config_get('DEFAULT_LEARNING_RATE')),
                                  'discount_factor': champ_config.get('discount_factor',
                                                                      self.config_get('DEFAULT_DISCOUNT_FACTOR')),
                                  'exploration_rate': champ_config.get('exploration_rate',
                                                                       self.config_get('DEFAULT_EXPLORATION_RATE')),
                                  'exploration_decay_rate': champ_config.get('exploration_decay_rate', self.config_get(
                                      'DEFAULT_EXPLORATION_DECAY_RATE')),
                                  'min_exploration_rate': champ_config.get('min_exploration_rate', self.config_get(
                                      'DEFAULT_MIN_EXPLORATION_RATE')),
                                  'rle_min_encodable_run': champ_config.get('rle_min_encodable_run',
                                                                            self.config_get('RLE_MIN_RUN_INIT_MIN')),
                                  'target_device': self.els_target_device}
                core_ai = PuffinZipAI(**ai_init_params)
                if 'q_table' in champ_config and champ_config['q_table'] is not None:
                    if core_ai.use_gpu_acceleration and hasattr(core_ai,
                                                                'q_table_gpu') and core_ai.q_table_gpu is not None and cp is not None:
                        if core_ai.q_table_gpu.shape == champ_config['q_table'].shape:
                            core_ai.q_table_gpu = cp.asarray(champ_config['q_table']); core_ai.q_table = np.copy(
                                champ_config['q_table'])
                        else:
                            self.logger.warning(
                                f"Seed {champ_idx + 1}: GPU Q-table shape mismatch. Using re-init Q-table.")
                    elif core_ai.q_table is not None and core_ai.q_table.shape == champ_config['q_table'].shape:
                        core_ai.q_table = np.copy(champ_config['q_table'])
                    else:
                        self.logger.warning(
                            f"Seed {champ_idx + 1}: Q-table shape mismatch/core None. Using re-init Q-table.")
                current_population.append(EvolvingAgent(puffin_ai_instance=core_ai, generation_born=0,
                                                        agent_id=f"seeded_champ_{num_seeded}"));
                num_seeded += 1
            except Exception as e_seed:
                self.logger.error(f"Error creating agent from seed {champ_idx + 1}: {e_seed}", exc_info=True)
        if self.gui_stop_event.is_set(): return []

        if num_seeded > 0: self._send_to_gui(
            f"Added {num_seeded} champion(s) to initial population.");self.seeded_champion_configs.clear()
        num_random_to_create = self.population_size - len(current_population)
        if num_random_to_create > 0: self._send_to_gui(
            f"Generating {num_random_to_create} random agents (with increased variance)...")
        rle_min_run_init_min_wide = max(1, self.config_get('RLE_MIN_RUN_INIT_MIN') - 1);
        rle_min_run_init_max_wide = self.config_get('RLE_MIN_RUN_INIT_MAX') + 1
        for i in range(num_random_to_create):
            if self.gui_stop_event.is_set(): self.logger.info("Population creation (random gen) interrupted."); break
            try:
                num_thresholds_to_gen = random.randint(self.config_get('MIN_THRESHOLDS_COUNT'),
                                                       self.config_get('MAX_THRESHOLDS_COUNT') + 2)
                threshold_values = sorted(random.sample(range(3, 3000), num_thresholds_to_gen))[
                                   :self.config_get('MAX_THRESHOLDS_COUNT')]
                if not threshold_values: threshold_values = [random.randint(10, 200)]
                ai_init_params_rand = {'len_thresholds': threshold_values, 'learning_rate': random.uniform(0.0005, 0.4),
                                       'discount_factor': random.uniform(0.70, 0.9999),
                                       'exploration_rate': random.uniform(0.8, 1.0),
                                       'exploration_decay_rate': random.uniform(0.985, 0.99995),
                                       'min_exploration_rate': random.uniform(0.0001, 0.05),
                                       'rle_min_encodable_run': random.randint(rle_min_run_init_min_wide,
                                                                               rle_min_run_init_max_wide),
                                       'target_device': self.els_target_device}
                core_ai_rand = PuffinZipAI(**ai_init_params_rand)
                current_population.append(
                    EvolvingAgent(puffin_ai_instance=core_ai_rand, generation_born=0, agent_id=f"gen0_rand_{i}"))
            except Exception as e_rand_agent:
                self.logger.error(f"Error creating random agent {i}: {e_rand_agent}", exc_info=True)
        self.population = current_population;
        self.logger.info(
            f"Initial population of {len(self.population)} agents created. Agent hardware target: '{self.els_target_device}'");
        return self.population

    def _evaluate_population(self, population_to_evaluate, generation_num):
        if self.gui_stop_event.is_set(): self.logger.info(
            f"Gen {generation_num + 1} Evaluation skipped (Stop Event)."); return float('-inf')
        self.logger.info(f"Starting evaluation for Gen {generation_num}. Pop size: {len(population_to_evaluate)}");
        self._send_to_gui(f"Gen {generation_num + 1}: Evaluating {len(population_to_evaluate)} agents...")
        best_fitness_this_gen = float('-inf');
        avg_fitness_this_gen_val = float('-inf');
        worst_fitness_this_gen = float('inf');
        median_fitness_this_gen = float('-inf')
        if not self.benchmark_evaluator:
            self.logger.error("Cannot eval pop: BenchmarkItemEvaluator N/A.");
            self._send_to_gui(f"Gen {generation_num + 1}: CRITICAL ERROR - Benchmark system N/A.", "error");
            return float('-inf')
        else:
            evaluation_results = self.benchmark_evaluator.evaluate_population_batch(population_to_evaluate,
                                                                                    repetitions_per_item=DEFAULT_BENCHMARK_REPETITIONS,
                                                                                    gui_stop_event=self.gui_stop_event)
            if self.gui_stop_event.is_set(): self.logger.info(
                f"Gen {generation_num + 1} Evaluation INTERRUPTED by GUI after batch call."); return float('-inf')
            if len(evaluation_results) != len(population_to_evaluate): self.logger.error(
                f"Mismatch: Got {len(evaluation_results)} eval results for {len(population_to_evaluate)} agents.")
            valid_fitness_scores = []
            for i, agent in enumerate(population_to_evaluate):
                if self.gui_stop_event.is_set(): break
                fitness = float('-inf')
                if i < len(evaluation_results): fitness, stats_dict = evaluation_results[i]; agent.set_fitness(
                    fitness); agent.evaluation_stats = stats_dict
                if isinstance(fitness, (int, float)) and fitness != float('-inf') and np.isfinite(fitness):
                    valid_fitness_scores.append(fitness)
                else:
                    agent.set_fitness(float('-inf')); agent.evaluation_stats = {}

            if self.gui_stop_event.is_set(): self.logger.info(
                f"Gen {generation_num + 1} Evaluation INTERRUPTED during fitness assignment."); return float('-inf')

            population_to_evaluate.sort(key=lambda ag: ag.get_fitness(), reverse=True)
            if population_to_evaluate:
                best_agent_this_gen = population_to_evaluate[0];
                best_fitness_this_gen = best_agent_this_gen.get_fitness() if np.isfinite(
                    best_agent_this_gen.get_fitness()) else float('-inf')
                worst_fitness_this_gen = population_to_evaluate[-1].get_fitness() if np.isfinite(
                    population_to_evaluate[-1].get_fitness()) else float('inf')
                if valid_fitness_scores:
                    avg_fitness_this_gen_val = sum(valid_fitness_scores) / len(valid_fitness_scores);
                    median_fitness_this_gen = np.median(valid_fitness_scores)
                else:
                    avg_fitness_this_gen_val = float('-inf'); median_fitness_this_gen = float('-inf')
                if not np.isfinite(best_fitness_this_gen): best_fitness_this_gen = float('-inf')
                if not np.isfinite(worst_fitness_this_gen): worst_fitness_this_gen = float('-inf')
                if hasattr(best_agent_this_gen,
                           'evaluation_stats') and best_agent_this_gen.evaluation_stats: stats_str = self._format_eval_stats(
                    best_agent_this_gen.evaluation_stats);self.logger.info(
                    f"Gen {generation_num + 1} Best Agent ({best_agent_this_gen.agent_id}) Stats: {stats_str}")
            else:
                worst_fitness_this_gen = float('-inf'); median_fitness_this_gen = float('-inf')
            if not np.isfinite(median_fitness_this_gen): median_fitness_this_gen = float('-inf')
        self.logger.info(
            f"Gen {generation_num + 1} eval complete. Best: {best_fitness_this_gen:.4f}, Avg: {avg_fitness_this_gen_val:.4f}, Median: {median_fitness_this_gen:.4f}, Worst: {worst_fitness_this_gen:.4f}");
        self._send_to_gui(
            f"Gen {generation_num + 1}: Eval Done. Best: {best_fitness_this_gen:.4f}, Avg: {avg_fitness_this_gen_val:.4f}, Median: {median_fitness_this_gen:.4f}, Worst: {worst_fitness_this_gen:.4f}")
        self.fitness_history_per_generation.append(
            (generation_num, best_fitness_this_gen, avg_fitness_this_gen_val, worst_fitness_this_gen,
             median_fitness_this_gen));
        self.average_fitness_overall_history.append(avg_fitness_this_gen_val)
        if not self.gui_stop_event.is_set():
            self._send_fitness_history_to_gui();
        return avg_fitness_this_gen_val

    def _format_eval_stats(self, stats_dict: dict) -> str:
        if not stats_dict: return "No evaluation stats."
        items_eval = float(stats_dict.get("items_evaluated", 0))
        if items_eval == 0: return "Items evaluated: 0."
        s_rle, e_rle, nc_chose, d_mismatch, rle_err = stats_dict.get("successful_rle", 0), stats_dict.get(
            "rle_expansion", 0), stats_dict.get("chose_nocompression", 0), stats_dict.get("decomp_failures_mismatch",
                                                                                          0), stats_dict.get(
            "rle_errors_returned", 0)
        s_rle_pct = (s_rle / items_eval) * 100 if items_eval else 0;
        avg_good_r = (stats_dict.get("sum_compression_ratios_rle_success", 0) / s_rle) if s_rle > 0 else 0
        e_rle_pct = (e_rle / items_eval) * 100 if items_eval else 0;
        avg_bad_r = (stats_dict.get("sum_expansion_ratios_rle_fail", 0) / e_rle) if e_rle > 0 else 0
        nc_pct = (nc_chose / items_eval) * 100 if items_eval else 0;
        err_m_pct = (d_mismatch / items_eval) * 100 if items_eval else 0;
        err_rle_pct = (rle_err / items_eval) * 100 if items_eval else 0
        avg_time = stats_dict.get("total_processing_time_ms", 0) / items_eval if items_eval else 0
        return (
            f"Items:{items_eval:.0f}; RLE Ok:{s_rle_pct:.1f}%(R:{avg_good_r:.2f}); RLE Bad:{e_rle_pct:.1f}%(R:{avg_bad_r:.2f}); NoComp:{nc_pct:.1f}%; Errs(Mismatch:{err_m_pct:.1f}%,RLEFail:{err_rle_pct:.1f}%); AvgTime:{avg_time:.1f}ms")

    def _select_parents(self, sorted_population):
        if self.gui_stop_event.is_set(): return []
        num_offspring_needed = self.population_size - self.elitism_count;
        num_to_select = num_offspring_needed * 2;
        parents = []
        if num_to_select <= 0 or not sorted_population: return []
        k_for_selection = max(2, num_to_select)
        if len(sorted_population) < k_for_selection and len(sorted_population) >= 2:
            k_for_selection = len(sorted_population)
        elif len(sorted_population) < 2:
            return list(sorted_population)
        sel_method_name = self.config_get('DEFAULT_SELECTION_STRATEGY', "tournament")
        try:
            if sel_method_name == "tournament":
                parents = tournament_selection(sorted_population, k_for_selection,
                                               tournament_size=max(2, min(3, len(sorted_population))))
            elif sel_method_name == "roulette":
                parents = roulette_wheel_selection(sorted_population, k_for_selection)
            elif sel_method_name == "rank":
                parents = rank_selection(sorted_population, k_for_selection)
            else:
                self.logger.warning(
                    f"Unknown selection '{sel_method_name}', using tournament."); parents = tournament_selection(
                    sorted_population, k_for_selection, tournament_size=max(2, min(3, len(sorted_population))))
        except Exception as e_sel:
            self.logger.error(f"Error during parent selection '{sel_method_name}': {e_sel}",
                              exc_info=True); parents = sorted_population[:k_for_selection]
        if not parents and sorted_population: self.logger.warning(
            "Parent selection empty, fallback random.choices."); parents = random.choices(sorted_population,
                                                                                          k=k_for_selection)
        if parents and len(parents) % 2 != 0 and sorted_population: parents.append(random.choice(parents))
        return parents

    def _breed_population(self, parent_pool, current_generation_num):
        next_gen_offspring = [];
        num_offspring_needed = self.population_size - self.elitism_count;
        offspring_created_count = 0
        if not parent_pool or num_offspring_needed <= 0: self.logger.info(
            f"Gen {current_generation_num + 1}: Breeding skipped. Parents: {len(parent_pool)}, Offspring needed: {num_offspring_needed}"); return next_gen_offspring
        if self.gui_stop_event.is_set(): self.logger.info(
            "Breeding preparation interrupted."); return next_gen_offspring
        self.logger.info(f"Gen {current_generation_num + 1}: Breeding {num_offspring_needed} offspring.");
        random.shuffle(parent_pool);
        parent_idx = 0
        while offspring_created_count < num_offspring_needed:
            if self.gui_stop_event.is_set(): self.logger.info("Breeding loop interrupted mid-way.");break
            if len(parent_pool) < 2:
                source_parent = parent_pool[0] if parent_pool else (self.population[0] if self.population else None)
                if not source_parent: self.logger.error("Critical: No source parent for cloning in breeding."); break
                for i in range(num_offspring_needed - offspring_created_count):
                    if self.gui_stop_event.is_set(): break
                    next_gen_offspring.append(source_parent.clone(
                        new_agent_id=f"gen{current_generation_num + 1}_clonefill_{offspring_created_count + i}",
                        new_generation_born=current_generation_num + 1));
                    offspring_created_count += 1
                break
            if self.gui_stop_event.is_set(): break
            parent1 = parent_pool[parent_idx % len(parent_pool)];
            parent2 = parent_pool[(parent_idx + 1) % len(parent_pool)];
            parent_idx += 2
            child1_core, child2_core = None, None
            try:
                child1_core, child2_core = advanced_crossover_pipeline(parent1.puffin_ai, parent2.puffin_ai,
                                                                       parent1.get_fitness(), parent2.get_fitness(),
                                                                       self.logger, self.config)
            except Exception as e_co:
                self.logger.error(f"Error during crossover: {e_co}",
                                  exc_info=True); child1_core = parent1.puffin_ai.clone_core_model(); child2_core = parent2.puffin_ai.clone_core_model()
            if child1_core: next_gen_offspring.append(
                EvolvingAgent(child1_core, agent_id=f"gen{current_generation_num + 1}_c{offspring_created_count}",
                              generation_born=current_generation_num + 1,
                              parent_ids=[parent1.agent_id, parent2.agent_id])); offspring_created_count += 1
            if offspring_created_count >= num_offspring_needed: break
            if self.gui_stop_event.is_set(): break
            if child2_core: next_gen_offspring.append(
                EvolvingAgent(child2_core, agent_id=f"gen{current_generation_num + 1}_c{offspring_created_count}",
                              generation_born=current_generation_num + 1,
                              parent_ids=[parent1.agent_id, parent2.agent_id])); offspring_created_count += 1
        self.logger.info(f"Gen {current_generation_num + 1}: Bred {len(next_gen_offspring)} new agents.");
        return next_gen_offspring

    def _mutate_population(self, population_to_mutate, current_generation_num):
        if self.gui_stop_event.is_set(): self.logger.info("Mutation phase skipped (Stop Event)."); return
        mut_count = 0;
        is_hyper = current_generation_num < self.hypermutation_active_until_gen
        mutation_config_dict = {'base_rate': self.current_mutation_rate, 'param_factor': 1.0, 'threshold_factor': 1.0,
                                'rle_min_run_prob': self.config_get('RLE_MIN_RUN_MUTATION_PROB')}
        hypermutation_config_dict = {
            'HYPERMUTATION_PARAM_STRENGTH_FACTOR': self.config_get('HYPERMUTATION_PARAM_STRENGTH_FACTOR'),
            'HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB': self.config_get('HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB')}
        for agent_idx, agent in enumerate(population_to_mutate):
            if self.gui_stop_event.is_set(): self.logger.info(f"Mutation loop interrupted at agent {agent_idx}.");break
            mutated_this_pass = False
            if is_hyper and agent_idx >= self.elitism_count and random.random() < self.config_get(
                    'HYPERMUTATION_FRACTION'):
                if apply_hypermutation_func: apply_hypermutation_func(agent, hypermutation_config_dict)
                mutated_this_pass = True
            elif apply_mutations_func:
                if apply_mutations_func(agent, mutation_config_dict):
                    mutated_this_pass = True
            if mutated_this_pass: mut_count += 1
        self.logger.info(
            f"Mutation Gen {current_generation_num + 1}. Affected ~{mut_count} agents (rate: {self.current_mutation_rate:.3f}, hyper: {is_hyper}).")

    def _introduce_random_immigrants(self, target_population_segment, num_to_introduce, current_generation_num):
        if self.gui_stop_event.is_set() or num_to_introduce <= 0 or not target_population_segment: return
        self.logger.info(
            f"Introducing {num_to_introduce} immigrants into segment size {len(target_population_segment)} for gen {current_generation_num}.")
        temp_sorted_segment = sorted(target_population_segment,
                                     key=lambda ag: ag.get_fitness() if hasattr(ag, 'fitness') else float('-inf'));
        replaced_count = 0
        for i in range(min(num_to_introduce, len(temp_sorted_segment))):
            if self.gui_stop_event.is_set(): self.logger.info(
                f"Immigrant introduction interrupted at immigrant {i}.");break
            agent_to_replace = temp_sorted_segment[i]
            try:
                idx_in_orig = target_population_segment.index(agent_to_replace)
                num_thresholds_to_gen_imm = random.randint(self.config_get('MIN_THRESHOLDS_COUNT'),
                                                           self.config_get('MAX_THRESHOLDS_COUNT') + 2);
                threshold_values_imm = sorted(random.sample(range(3, 3000), num_thresholds_to_gen_imm))[
                                       :self.config_get('MAX_THRESHOLDS_COUNT')]
                if not threshold_values_imm: threshold_values_imm = [random.randint(10, 200)]
                rle_min_run_init_min_wide_imm = max(1, self.config_get('RLE_MIN_RUN_INIT_MIN') - 1);
                rle_min_run_init_max_wide_imm = self.config_get('RLE_MIN_RUN_INIT_MAX') + 1
                ai_params_imm = {'len_thresholds': threshold_values_imm, 'learning_rate': random.uniform(0.0005, 0.4),
                                 'discount_factor': random.uniform(0.70, 0.9999),
                                 'exploration_rate': random.uniform(0.8, 1.0),
                                 'exploration_decay_rate': random.uniform(0.985, 0.99995),
                                 'min_exploration_rate': random.uniform(0.0001, 0.05),
                                 'rle_min_encodable_run': random.randint(rle_min_run_init_min_wide_imm,
                                                                         rle_min_run_init_max_wide_imm),
                                 'target_device': self.els_target_device}
                core_ai_imm = PuffinZipAI(**ai_params_imm)
                target_population_segment[idx_in_orig] = EvolvingAgent(core_ai_imm,
                                                                       agent_id=f"gen{current_generation_num}_immig_{replaced_count}",
                                                                       generation_born=current_generation_num);
                replaced_count += 1
            except ValueError:
                self.logger.warning(
                    f"Agent {agent_to_replace.agent_id} not found in segment for immigrant replacement.")
            except Exception as e_imm:
                self.logger.error(f"Error creating/replacing immigrant: {e_imm}", exc_info=True)
        if replaced_count > 0: self.logger.info(f"Introduced {replaced_count} immigrants by replacement.")

    def _update_stagnation_and_mutation_rate(self, current_gen_num, best_fitness_this_gen):
        if self.gui_stop_event.is_set(): return
        if best_fitness_this_gen > self.best_fitness_overall:
            self.best_fitness_overall = best_fitness_this_gen;
            if self.population: self.best_agent_overall = self.population[0]
            self.stagnation_counter = 0;
            self.last_best_fitness_improvement_gen = current_gen_num;
            decay = self.config_get('MUTATION_RATE_DECAY_FACTOR')
            if not (
                    current_gen_num < self.hypermutation_active_until_gen and self.current_mutation_rate > self.base_mutation_rate): self.logger.info(
                f"Fitness improved to {self.best_fitness_overall:.4f}. Stagnation reset.")
            self.current_mutation_rate = max(self.base_mutation_rate, self.current_mutation_rate * decay);
            self.logger.info(f"Fitness improved (mut rate decaying if was boosted): {self.current_mutation_rate:.4f}")
        else:
            self.stagnation_counter += 1;
            self.logger.info(f"No fitness improvement for {self.stagnation_counter} generation(s).")
        if current_gen_num >= self.hypermutation_active_until_gen:
            if self.current_mutation_rate > self.base_mutation_rate:
                self.current_mutation_rate = max(self.base_mutation_rate, self.current_mutation_rate * self.config_get(
                    'MUTATION_RATE_DECAY_FACTOR'));
                log_msg_decay = f"Post-hyper/bottleneck: Mut rate decaying: {self.current_mutation_rate:.4f}" if self.current_mutation_rate > self.base_mutation_rate else f"Mut rate returned to base: {self.base_mutation_rate:.4f}";
                self.logger.info(log_msg_decay)
            else:
                self.current_mutation_rate = self.base_mutation_rate
            if self.stagnation_counter >= self.config_get('HYPERMUTATION_STAGNATION_THRESHOLD'):
                self.logger.warning(
                    f"Hypermutation triggered (stagnation: {self.stagnation_counter} gens) at Gen {current_gen_num + 1}.");
                self._send_to_gui(f"Gen {current_gen_num + 1}: Hypermutation triggered! Boosting mutation.", "warning")
                self.current_mutation_rate = min(0.85, self.base_mutation_rate * self.config_get(
                    'MUTATION_RATE_BOOST_FACTOR') * 2.0);
                self.hypermutation_active_until_gen = current_gen_num + random.randint(3, 7);
                self.stagnation_counter = 0
                self.logger.info(
                    f"Stagnation-Hypermutation active until Gen ~{self.hypermutation_active_until_gen}. MutRate: {self.current_mutation_rate:.4f}")
        else:
            self.logger.debug(
                f"Gen {current_gen_num + 1}: Within manual Hypermutation/Bottleneck period (until ~Gen {self.hypermutation_active_until_gen}). Current MutRate: {self.current_mutation_rate:.4f}")

    def _maybe_refresh_dynamic_benchmarks(self, current_gen_num_0_indexed, user_size_override_mb=None,
                                          user_complexity_override_name=None):
        if not self.dynamic_benchmarking_active or not self.benchmark_evaluator or self.gui_stop_event.is_set(): self.logger.debug(
            "Dynamic benchmarks not active or evaluator missing, or stop event set. Refresh skipped."); return
        if current_gen_num_0_indexed == 0:
            self.logger.info(
                f"Gen 0: Setting initial benchmark set. User size override (from ELS init): {user_size_override_mb} MB, User complexity override (from ELS init): {user_complexity_override_name}")
            avg_fitness_for_complexity = -1000.0
            if self.average_fitness_overall_history: avg_fitness_for_complexity = self.average_fitness_overall_history[
                -1]
            self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(
                population_average_fitness=avg_fitness_for_complexity, current_generation=current_gen_num_0_indexed,
                target_item_size_mb_override=user_size_override_mb,
                fixed_complexity_override_name=user_complexity_override_name or self.initial_benchmark_complexity_config_default)
            if not self.benchmark_evaluator.benchmark_items:
                self.logger.error("Initial dynamic benchmark generation resulted in NO items! ELS might fail.");
                self._send_to_gui("CRITICAL ERROR: Initial dynamic benchmark generation failed!", "error")
            else:
                self._send_to_gui(
                    f"Initial dynamic benchmarks set. Items: {len(self.benchmark_evaluator.benchmark_items)}")
            return

        if (
                self.benchmark_refresh_interval_gens is not None and self.benchmark_refresh_interval_gens > 0 and current_gen_num_0_indexed > 0 and current_gen_num_0_indexed % self.benchmark_refresh_interval_gens == 0):
            avg_fitness_for_complexity = -1000.0
            if self.average_fitness_overall_history: avg_fitness_for_complexity = self.average_fitness_overall_history[
                -1]
            self._send_to_gui(
                f"Gen {current_gen_num_0_indexed + 1}: Refreshing dynamic benchmarks based on AvgFit {avg_fitness_for_complexity:.3f} (Adaptive strategy always used for refresh)...")
            self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(
                population_average_fitness=avg_fitness_for_complexity, current_generation=current_gen_num_0_indexed)
            if not self.benchmark_evaluator.benchmark_items:
                self.logger.error("Dynamic benchmark generation resulted in NO items! ELS might fail.");
                self._send_to_gui(f"CRITICAL ERROR: Dynamic benchmark generation failed!", "error")
            else:
                self._send_to_gui(
                    f"Dynamic benchmarks refreshed (Adaptive). Items: {len(self.benchmark_evaluator.benchmark_items)}")

    def apply_bottleneck_strategy(self, strategy_name: str):
        if not self.population: self._send_to_gui("Cannot apply adaptation: ELS not running or population is empty.",
                                                  "warning"); return
        if self.gui_stop_event.is_set(): self.logger.info("Adaptation application skipped due to stop event."); return
        original_mutation_rate = self.current_mutation_rate;
        original_elitism_count = self.elitism_count
        message = f"Applying {strategy_name.title()} Adaptation Strategy (Gen {self.total_generations_elapsed + 1}):"
        is_gpu_target = "GPU" in self.els_target_device.upper()
        if is_gpu_target:
            message += f"\n  (ELS Target Device: '{self.els_target_device}' - GPU parameters placeholder)"; self.logger.info(
                f"Adaptation strategy '{strategy_name}' applying with GPU target '{self.els_target_device}'. Current logic uses same params as CPU.")
        else:
            message += f"\n  (ELS Target Device: '{self.els_target_device}' - CPU parameters apply)"
        if strategy_name == "low":
            self.current_mutation_rate = min(0.5, self.base_mutation_rate * 1.25);
            self.elitism_count = max(0, self.default_elitism_count_from_config - 1);
            message += f"\n  - Mutation rate adjusted: {original_mutation_rate:.4f} -> {self.current_mutation_rate:.4f}";
            message += f"\n  - Elitism count adjusted: {original_elitism_count} -> {self.elitism_count}"
        elif strategy_name == "medium":
            self.current_mutation_rate = min(0.7, self.base_mutation_rate * 1.75);
            self.elitism_count = max(0, self.default_elitism_count_from_config // 2);
            message += f"\n  - Mutation rate adjusted: {original_mutation_rate:.4f} -> {self.current_mutation_rate:.4f}";
            message += f"\n  - Elitism count adjusted: {original_elitism_count} -> {self.elitism_count}"
        elif strategy_name == "high":
            self.current_mutation_rate = min(0.85, self.base_mutation_rate * self.config_get(
                'MUTATION_RATE_BOOST_FACTOR') * 2.5);
            self.hypermutation_active_until_gen = self.total_generations_elapsed + random.randint(2, 4);
            self.stagnation_counter = 0
            self.elitism_count = max(0, self.default_elitism_count_from_config // 3)
            message += f"\n  - IMMEDIATE HYPERMUTATION TRIGGERED.";
            message += f"\n  - Mutation rate set to: {self.current_mutation_rate:.4f} until Gen ~{self.hypermutation_active_until_gen + 1}";
            message += f"\n  - Elitism count adjusted: {original_elitism_count} -> {self.elitism_count}"
        else:
            self._send_to_gui(f"Unknown adaptation strategy: {strategy_name}", "warning");return
        self.logger.info(message.replace("\n  - ", " | "));
        self._send_to_gui(message)

    def clear_bottleneck_strategy(self):
        if not self.population: self._send_to_gui("Cannot clear adaptation: ELS not running or population empty.",
                                                  "warning"); return
        if self.gui_stop_event.is_set(): self.logger.info("Clear adaptation skipped due to stop event."); return
        message = f"Clearing Adaptation Strategy (Gen {self.total_generations_elapsed + 1}):"
        self.current_mutation_rate = self.base_mutation_rate
        self.elitism_count = self.default_elitism_count_from_config
        self.hypermutation_active_until_gen = 0
        message += f"\n  - Mutation rate reset to base: {self.current_mutation_rate:.4f}"
        message += f"\n  - Elitism count reset to default: {self.elitism_count}"
        message += f"\n  - Hypermutation period deactivated."
        self.logger.info(message.replace("\n  - ", " | "));
        self._send_to_gui(message)

    def start_evolution(self):
        self.clear_pause_resume_events();
        is_continuation = self.total_generations_elapsed > 0 and self.population
        start_gen_0_idx = self.total_generations_elapsed
        if is_continuation:
            target_num_generations_in_run = self.config_get('DEFAULT_ADDITIONAL_ELS_GENERATIONS');
            run_title = f"--- ELS (Cont.) Run (+{target_num_generations_in_run} gens from Gen {start_gen_0_idx + 1}) ---"
            if self.gui_stop_event.is_set(): self.gui_stop_event.clear(); self.logger.info(
                "Continuing evolution. Explicitly cleared stop event before this run segment.")
        else:
            target_num_generations_in_run = self.initial_num_generations;
            run_title = f"--- ELS: New Run Started --- Pop: {self.population_size}, TargetGens: {target_num_generations_in_run}"
            self.population = self._create_initial_population()
            if self.gui_stop_event.is_set() or not self.population: self.logger.critical(
                "Initial population creation failed or was stopped. Halting ELS."); self._send_to_gui(
                "CRITICAL ERROR: ELS failed to create initial population or was stopped.", "critical"); return

            # --- PRE-RUN HARDWARE CHECK ---
            if not self._pre_run_hardware_check():
                self.logger.critical("Pre-run hardware check FAILED. Aborting ELS start.")
                self._send_to_gui("CRITICAL: Hardware check failed. ELS aborted. See logs for details.", "critical")
                return
            # --- END OF CHECK ---

            self.fitness_history_per_generation = [];
            self.best_agent_overall = None;
            self.best_fitness_overall = float('-inf')
            self.stagnation_counter = 0;
            self.last_best_fitness_improvement_gen = 0;
            self.current_mutation_rate = self.base_mutation_rate;
            self.hypermutation_active_until_gen = 0;
            self.average_fitness_overall_history.clear();
            self.elitism_count = self.default_elitism_count_from_config
            self._maybe_refresh_dynamic_benchmarks(0, self.initial_benchmark_target_size_mb,
                                                   self.initial_benchmark_fixed_complexity_name)
            if self.gui_stop_event.is_set(): self.logger.info(
                "ELS setup for new run stopped (post-benchmark refresh).");return
        
        # Continuation also needs a check if something changed
        if is_continuation and not self._pre_run_hardware_check():
            self.logger.critical("Pre-run hardware check FAILED for continuation. Aborting ELS continue.")
            self._send_to_gui("CRITICAL: Hardware check failed for continuing run. ELS aborted.", "critical")
            return

        self.logger.info(run_title);
        self._send_to_gui(run_title)
        if not is_continuation: self._send_fitness_history_to_gui()
        completed_generations_this_segment = 0;
        target_total_gens_to_reach = start_gen_0_idx + target_num_generations_in_run
        try:
            for current_absolute_gen_0_idx in range(start_gen_0_idx, target_total_gens_to_reach):
                if self.gui_stop_event.is_set(): self.logger.info(
                    f"ELS run stop signal DETECTED at start of Gen {current_absolute_gen_0_idx + 1} loop.");break
                self.total_generations_elapsed = current_absolute_gen_0_idx;
                current_gen_display = current_absolute_gen_0_idx + 1
                if current_absolute_gen_0_idx > 0: self._maybe_refresh_dynamic_benchmarks(current_absolute_gen_0_idx)
                if self.gui_stop_event.is_set(): self.logger.info(
                    f"ELS run stopping before Gen {current_gen_display} full processing (post-benchmark).");break

                self.logger.info(
                    f"--- ELS Gen {current_gen_display}/{target_total_gens_to_reach} --- MutRate: {self.current_mutation_rate:.4f}, Elitism: {self.elitism_count} ---");
                self._send_to_gui(
                    f"--- ELS Gen {current_gen_display}/{target_total_gens_to_reach} --- (Rate: {self.current_mutation_rate:.3f}, Elitism: {self.elitism_count})")

                if self.pause_event.is_set():
                    self.logger.info(f"ELS PAUSED Gen {current_gen_display}.");
                    self._send_to_gui(f"ELS PAUSED Gen {current_gen_display}. Waiting...", "info")
                    while self.pause_event.is_set() and not self.gui_stop_event.is_set(): time.sleep(0.2)
                    if self.gui_stop_event.is_set(): self.logger.info(
                        f"ELS run STOPPED during pause Gen {current_gen_display}."); break
                    if not self.pause_event.is_set() and not self.gui_stop_event.is_set(): self.logger.info(
                        f"ELS RESUMED Gen {current_gen_display}.")

                if self.gui_stop_event.is_set(): self.logger.info(
                    f"ELS run stopping after pause/resume check Gen {current_gen_display}.");break

                current_avg_fitness = self._evaluate_population(self.population, current_absolute_gen_0_idx)
                if self.gui_stop_event.is_set(): self.logger.info(
                    f"ELS run stopping after _evaluate_population for Gen {current_gen_display}.");break

                if current_avg_fitness == float('-inf') and not self.population:
                    self.logger.warning(
                        f"Evaluation returned unusable fitness or population became empty. Stop might have occurred during eval for Gen {current_gen_display}.");
                    if not self.gui_stop_event.is_set(): self.logger.error(
                        "ELS hard failure during evaluation or eval cleared population without stop_event."); self._send_to_gui(
                        "CRITICAL: ELS failure during population evaluation.", "error")
                    break

                if self.population and self.population[0]:
                    current_gen_best_agent = self.population[0];
                    cg_best_params_msg = (
                        f"Gen {current_gen_display} Best Agent ({current_gen_best_agent.agent_id}): Fit={current_gen_best_agent.get_fitness():.4f}, Thresh={current_gen_best_agent.puffin_ai.len_thresholds}, LR={current_gen_best_agent.puffin_ai.learning_rate:.4f}, ER={current_gen_best_agent.puffin_ai.exploration_rate:.3f}, RLE_MinRun={current_gen_best_agent.puffin_ai.rle_min_encodable_run_length}")
                    self.logger.info(cg_best_params_msg);
                    if not self.gui_stop_event.is_set(): self._send_to_gui(
                        f"{self.config_get('ELS_LOG_PREFIX')} Gen Summary: {cg_best_params_msg}")
                if not self.population: self.logger.error(
                    f"Population empty after Gen {current_gen_display} eval. Halting."); self._send_to_gui(
                    f"CRITICAL: ELS pop empty after Gen {current_gen_display} eval.", "error");break

                self._update_stagnation_and_mutation_rate(current_absolute_gen_0_idx,
                                                          self.population[0].get_fitness() if self.population and
                                                                                              self.population[
                                                                                                  0] else float('-inf'))
                if self.gui_stop_event.is_set(): break
                next_gen_population = []
                if self.elitism_count > 0 and self.population: elites = [
                    agent.clone(new_agent_id=f"gen{current_gen_display}_elite{i}",
                                new_generation_born=current_gen_display) for i, agent in
                    enumerate(self.population[:self.elitism_count])]; next_gen_population.extend(elites)
                num_offspring_needed = self.population_size - len(next_gen_population);
                num_immigrants = 0
                if num_offspring_needed > 0:
                    parents = self._select_parents(self.population)
                    if self.gui_stop_event.is_set(): break
                    if parents:
                        offspring = self._breed_population(parents, current_absolute_gen_0_idx)
                        if self.gui_stop_event.is_set(): break
                        if offspring:
                            self._mutate_population(offspring, current_gen_display + 1)
                            if self.gui_stop_event.is_set(): break
                            if (current_gen_display % self.config_get('RANDOM_IMMIGRANT_INTERVAL',
                                                                      20)) == 0 and current_gen_display > 0: num_immigrants = int(
                                len(offspring) * self.config_get('RANDOM_IMMIGRANT_FRACTION', 0.1));
                            if num_immigrants > 0: self._introduce_random_immigrants(offspring, num_immigrants,
                                                                                     current_gen_display + 1)
                            if self.gui_stop_event.is_set(): break
                            next_gen_population.extend(offspring)
                        else:
                            self.logger.warning(f"Gen {current_gen_display}: Breeding produced no offspring.")
                    else:
                        self.logger.warning(f"Gen {current_gen_display}: No parents selected.")

                if self.gui_stop_event.is_set(): break

                if len(next_gen_population) < self.population_size:
                    fill_needed = self.population_size - len(next_gen_population)
                    self.logger.info(f"Gen {current_gen_display}: Filling {fill_needed} population slots.")
                    fill_source = sorted(next_gen_population, key=lambda ag: ag.get_fitness(),
                                         reverse=True) if next_gen_population else (
                        self.population if self.population else [])
                    if fill_source:
                        for i in range(fill_needed):
                            if self.gui_stop_event.is_set(): break
                            next_gen_population.append(
                                random.choice(fill_source).clone(new_agent_id=f"gen{current_gen_display}_fill{i}",
                                                                 new_generation_born=current_gen_display))
                    elif PuffinZipAI:
                        self.logger.warning(
                            f"Gen {current_gen_display}: No source for cloning fill. Adding new random agents for {fill_needed} slots.");
                        for i in range(fill_needed):
                            if self.gui_stop_event.is_set(): break
                            num_thresholds_to_gen_fill = random.randint(self.config_get('MIN_THRESHOLDS_COUNT'),
                                                                        self.config_get('MAX_THRESHOLDS_COUNT') + 2);
                            threshold_values_fill = sorted(random.sample(range(3, 3000), num_thresholds_to_gen_fill))[
                                                    :self.config_get('MAX_THRESHOLDS_COUNT')]
                            if not threshold_values_fill: threshold_values_fill = [random.randint(10, 200)]
                            rle_min_run_init_min_wide_fill = max(1, self.config_get('RLE_MIN_RUN_INIT_MIN') - 1);
                            rle_min_run_init_max_wide_fill = self.config_get('RLE_MIN_RUN_INIT_MAX') + 1
                            rand_ai_params = {'len_thresholds': threshold_values_fill,
                                              'learning_rate': random.uniform(0.0005, 0.4),
                                              'discount_factor': random.uniform(0.70, 0.9999),
                                              'exploration_rate': random.uniform(0.7, 1.0),
                                              'exploration_decay_rate': random.uniform(0.985, 0.99995),
                                              'min_exploration_rate': random.uniform(0.0001, 0.05),
                                              'rle_min_encodable_run': random.randint(rle_min_run_init_min_wide_fill,
                                                                                      rle_min_run_init_max_wide_fill),
                                              'target_device': self.els_target_device}
                            next_gen_population.append(
                                EvolvingAgent(PuffinZipAI(**rand_ai_params), generation_born=current_gen_display,
                                              agent_id=f"gen{current_gen_display}_randfill{i}"))
                if self.gui_stop_event.is_set(): break
                self.population = next_gen_population[:self.population_size]
                if not self.population: self.logger.critical(
                    f"Gen {current_gen_display}: Population empty after breeding/fill! Halting."); self._send_to_gui(
                    f"CRITICAL: ELS pop empty Gen {current_gen_display}.", "error");break
                self.logger.debug(f"Gen {current_gen_display}: New pop size: {len(self.population)}")
                completed_generations_this_segment = current_absolute_gen_0_idx - start_gen_0_idx + 1

        except Exception as e_evo_loop:
            self.logger.error(f"Exception in ELS main loop Gen {self.total_generations_elapsed + 1}: {e_evo_loop}",
                              exc_info=True); self._send_to_gui(
                f"ERROR in ELS main loop Gen {self.total_generations_elapsed + 1}: {e_evo_loop}", "error")
        finally:
            self.total_generations_elapsed = start_gen_0_idx + completed_generations_this_segment
            final_message = f"--- ELS Segment Ended. Gens in segment: {completed_generations_this_segment}, Total ELS gens overall: {self.total_generations_elapsed} ---"
            self.logger.info(final_message)
            if self.best_agent_overall:
                best_ai = self.best_agent_overall.puffin_ai;
                params_log = (
                    f"Best Agent Params: Thresh={best_ai.len_thresholds}, LR={best_ai.learning_rate:.4f}, DF={best_ai.discount_factor:.3f}, ER={best_ai.exploration_rate:.3f}, ERdecay={best_ai.exploration_decay_rate:.5f}, MinER={best_ai.min_exploration_rate:.5f}, RLE_MinRun={best_ai.rle_min_encodable_run_length}, TargetDevice='{best_ai.target_device}'")
                self._send_to_gui(
                    f"{final_message.split('---')[1].strip()} Best Fit: {self.best_fitness_overall:.4f} ---");
                self._send_to_gui(params_log);
                self.logger.info(params_log)
                if hasattr(self.best_agent_overall,
                           'evaluation_stats') and self.best_agent_overall.evaluation_stats: stats_summary = self._format_eval_stats(
                    self.best_agent_overall.evaluation_stats);self.logger.info(
                    f"Best Agent Eval Summary: {stats_summary}"); self._send_to_gui(
                    f"Best Agent Eval Summary: {stats_summary}")
            else:
                self.logger.warning("No best agent recorded for this segment."); self._send_to_gui(
                    f"{final_message.split('---')[1].strip()} No best agent recorded. ---", "warning")

            if self.gui_stop_event and self.gui_stop_event.is_set(): self.logger.info(
                "ELS stop event was set during this run. It will be cleared by the GUI thread or before the next ELS start if triggered from PMA.")

    def continue_evolution(self, additional_generations: int = None):
        if not self.population: self.logger.warning("Cannot continue: No existing population."); self._send_to_gui(
            "ELS Error: No population to continue.", "error"); return
        if self.gui_stop_event and self.gui_stop_event.is_set():
            self.logger.info("Continue evolution called, but stop event is set. Clearing stop event before starting.")
            self.gui_stop_event.clear()

        num_add_gens = additional_generations if additional_generations is not None else self.config_get(
            'DEFAULT_ADDITIONAL_ELS_GENERATIONS')
        original_setting_for_initial_gens = self.initial_num_generations;
        self.initial_num_generations = num_add_gens
        original_user_size_override = self.initial_benchmark_target_size_mb;
        original_user_complexity_override = self.initial_benchmark_fixed_complexity_name
        self.initial_benchmark_target_size_mb = None;
        self.initial_benchmark_fixed_complexity_name = None
        self.start_evolution()
        self.initial_num_generations = original_setting_for_initial_gens;
        self.initial_benchmark_target_size_mb = original_user_size_override;
        self.initial_benchmark_fixed_complexity_name = original_user_complexity_override

    def save_state(self, filepath: str):
        """Saves the entire state of the evolutionary optimizer to a file."""
        if not self.population:
            self.logger.warning("Attempted to save ELS state, but population is empty. Nothing saved.")
            self._send_to_gui("Warning: ELS state not saved, population is empty.", "warning")
            return False
        
        try:
            state = {
                'population': self.population,
                'total_generations_elapsed': self.total_generations_elapsed,
                'best_agent_overall': self.best_agent_overall,
                'best_fitness_overall': self.best_fitness_overall,
                'fitness_history_per_generation': self.fitness_history_per_generation,
                'average_fitness_overall_history': self.average_fitness_overall_history,
                'stagnation_counter': self.stagnation_counter,
                'last_best_fitness_improvement_gen': self.last_best_fitness_improvement_gen,
                'hypermutation_active_until_gen': self.hypermutation_active_until_gen,
                'current_mutation_rate': self.current_mutation_rate,
                'version_info': 'PuffinZip_ELS_State_v1.0'
            }

            dir_name = os.path.dirname(filepath)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
                
            msg = f"ELS state saved to '{os.path.basename(filepath)}'. (Gen: {self.total_generations_elapsed}, Pop: {len(self.population)})"
            self.logger.info(msg)
            self._send_to_gui(msg, "info")
            return True
        except Exception as e:
            msg = f"Failed to save ELS state to '{filepath}': {e}"
            self.logger.error(msg, exc_info=True)
            self._send_to_gui(msg, "error")
            return False

    def load_state(self, filepath: str):
        """Loads the optimizer state from a file, replacing the current state."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            if not isinstance(state, dict) or state.get('version_info') != 'PuffinZip_ELS_State_v1.0':
                 self.logger.error(f"File '{filepath}' is not a valid ELS state file or has an incompatible version.")
                 self._send_to_gui("Error: Invalid ELS state file.", "error")
                 return False

            self.population = state.get('population', [])
            self.total_generations_elapsed = state.get('total_generations_elapsed', 0)
            self.best_agent_overall = state.get('best_agent_overall', None)
            self.best_fitness_overall = state.get('best_fitness_overall', float('-inf'))
            self.fitness_history_per_generation = state.get('fitness_history_per_generation', [])
            self.average_fitness_overall_history = state.get('average_fitness_overall_history', [])
            self.stagnation_counter = state.get('stagnation_counter', 0)
            self.last_best_fitness_improvement_gen = state.get('last_best_fitness_improvement_gen', 0)
            self.hypermutation_active_until_gen = state.get('hypermutation_active_until_gen', 0)
            self.current_mutation_rate = state.get('current_mutation_rate', self.base_mutation_rate)

            # ELS run can be continued now
            self.clear_pause_resume_events()

            # Ensure all loaded agents are configured for the current GUI session
            for agent in self.population:
                if agent and hasattr(agent, 'puffin_ai') and agent.puffin_ai:
                    agent.puffin_ai.gui_output_queue = self.gui_output_queue
                    agent.puffin_ai.gui_stop_event = self.gui_stop_event

            msg = f"ELS state loaded from '{os.path.basename(filepath)}'. Ready to continue from Gen {self.total_generations_elapsed + 1}."
            self.logger.info(msg)
            self._send_to_gui(msg, "info")

            # Update the GUI with loaded historical data
            self._send_fitness_history_to_gui()

            return True

        except FileNotFoundError:
            msg = f"ELS state file not found: {filepath}"
            self.logger.error(msg)
            self._send_to_gui(msg, "error")
            return False
        except Exception as e:
            msg = f"Failed to load ELS state from '{filepath}': {e}"
            self.logger.error(msg, exc_info=True)
            self._send_to_gui(msg, "error")
            return False

    def save_best_agent(self, filepath):
        if self.best_agent_overall and hasattr(self.best_agent_overall.puffin_ai, 'get_config_dict'):
            puffin_ai_to_save = self.best_agent_overall.puffin_ai
            model_data_to_save = puffin_ai_to_save.get_config_dict()
            model_data_to_save['q_table'] = puffin_ai_to_save.q_table;
            model_data_to_save['training_stats'] = puffin_ai_to_save.training_stats
            model_data_to_save['champion_version_info'] = f"PuffinZipAI_ELS_Champion_v1.5_target_device_aware";
            model_data_to_save['fitness_achieved'] = self.best_fitness_overall
            model_data_to_save['generation_born_in_els'] = self.best_agent_overall.generation_born
            model_data_to_save['champion_evaluation_stats'] = self.best_agent_overall.evaluation_stats if hasattr(
                self.best_agent_overall, 'evaluation_stats') else {}
            try:
                dir_name = os.path.dirname(filepath);
                if dir_name and not os.path.exists(dir_name): os.makedirs(dir_name, exist_ok=True)
                np.save(filepath, model_data_to_save, allow_pickle=True)
                self.logger.info(
                    f"Best agent ({self.best_agent_overall.agent_id}, Fit: {self.best_fitness_overall:.4f}) config saved to '{filepath}'.");
                self._send_to_gui(
                    f"Champion agent model data saved to '{os.path.basename(filepath)}'. Fit: {self.best_fitness_overall:.4f}",
                    "info")
            except Exception as e_save:
                self.logger.error(f"Failed to save best agent to '{filepath}': {e_save}",
                                  exc_info=True); self._send_to_gui(f"Error saving champion: {e_save}", "error")
        else:
            self.logger.warning("No best agent or PuffinZipAI malformed."); self._send_to_gui(
                "No best agent to save or agent's AI core not configured.", "warning")