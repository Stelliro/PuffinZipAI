from __future__ import annotations

import itertools
import logging
import os
import queue
import random
import threading
import time
import numpy as np
import pickle
import json
import math
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Any, List, Dict

try:
    from ..gpu_core.gpu_ai_agent import cp
except ImportError:
    cp = None

PuffinZipAI: Any = None
try:
    from ..ai_core import PuffinZipAI
except ImportError:
    pass

# Novelty/Diversity scoring
_calculate_population_novelty_scores = None
_calculate_generation_repetition_penalty = None
try:
    from ..reward_system import calculate_population_novelty_scores as _calculate_population_novelty_scores
    from ..reward_system import calculate_generation_repetition_penalty as _calculate_generation_repetition_penalty
except ImportError:
    pass

# Config Imports
_config_module: Any = None
DEFAULT_POPULATION_SIZE: Any = 50
DEFAULT_NUM_GENERATIONS: Any = 100
DEFAULT_MUTATION_RATE: Any = 0.15
EVOLUTIONARY_AI_LOG_FILENAME: Any = 'evolutionary_optimizer.log'
LOGS_DIR_PATH: Any = 'logs'
DEBUG_LOG_CONSOLE_OUTPUT_ENABLED: Any = False
CheckpointManager: Any = None
CompressionBenchmark: Any = None
try:
    from ..config import (
        DEFAULT_POPULATION_SIZE, DEFAULT_NUM_GENERATIONS, DEFAULT_MUTATION_RATE, 
        EVOLUTIONARY_AI_LOG_FILENAME, LOGS_DIR_PATH,
        DEBUG_LOG_CONSOLE_OUTPUT_ENABLED
    )
    from .. import config as _config_module
    from ..logger import setup_logger
    from ..checkpoint_manager import CheckpointManager
    from ..compression_benchmark import CompressionBenchmark
except ImportError:
    pass


def _dprint(*args, **kwargs):
    """Debug print — only outputs when DEBUG_LOG_CONSOLE_OUTPUT_ENABLED is True."""
    if _config_module and getattr(_config_module, 'DEBUG_LOG_CONSOLE_OUTPUT_ENABLED', False):
        print(*args, **kwargs)

# Dynamic Imports
BenchmarkItemEvaluator: Any = None
try:
    from ..utils.benchmark_evaluator import BenchmarkItemEvaluator
except ImportError:
    pass

EvolvingAgent: Any = None
try:
    from .individual_agent import EvolvingAgent
except ImportError:
    pass

def tournament_selection(*args: Any, **kwargs: Any) -> Any: return args[0][:2]
try: from .selection_methods import tournament_selection
except ImportError: pass

def apply_mutations(*args: Any, **kwargs: Any) -> Any: return False
try: from .mutation_methods import apply_mutations
except ImportError: pass

def apply_hypermutation(*args: Any, **kwargs: Any) -> Any: pass
try: from .mutation_methods import apply_hypermutation
except ImportError: pass

def apply_crossover(*args: Any, **kwargs: Any) -> Any: return args[0].clone_core_model(), args[1].clone_core_model()
try: from .crossover_methods import apply_crossover
except ImportError: pass

# Novel compression method generator
NovelCompressionGenerator: Any = None
get_novel_generator: Any = None
try:
    from ..novel_compression_generator import NovelCompressionGenerator, get_generator as get_novel_generator
except ImportError:
    pass

# Gold standard head-to-head benchmarking
GoldStandardBenchmark: Any = None
try:
    from ..gold_standard_benchmark import GoldStandardBenchmark
except ImportError:
    pass

# GitHub file fetcher for real-world benchmark training data
GitHubFileFetcher: Any = None
try:
    from ..utils.github_file_fetcher import GitHubFileFetcher
except ImportError:
    pass

# Phased training config
_PHASED_TRAINING_ENABLED = True
_PHASED_TRAINING_PHASE1_END = 10
_PHASED_TRAINING_PHASE2_END = 30
_PHASED_TRAINING_PHASE3_GITHUB_RATIO = 0.80
_PHASED_TRAINING_GITHUB_ITEM_COUNT = 20
try:
    from ..config import (
        PHASED_TRAINING_ENABLED as _PHASED_TRAINING_ENABLED,
        PHASED_TRAINING_PHASE1_END as _PHASED_TRAINING_PHASE1_END,
        PHASED_TRAINING_PHASE2_END as _PHASED_TRAINING_PHASE2_END,
        PHASED_TRAINING_PHASE3_GITHUB_RATIO as _PHASED_TRAINING_PHASE3_GITHUB_RATIO,
        PHASED_TRAINING_GITHUB_ITEM_COUNT as _PHASED_TRAINING_GITHUB_ITEM_COUNT,
    )
except ImportError:
    pass

class EvolutionaryOptimizer:
    def __init__(self, population_size=None, num_generations=None, mutation_rate=None, 
                 gui_output_queue=None, gui_stop_event=None, target_device="GPU_AUTO", 
                 dynamic_benchmarking_active=True, infinite_mode=False, **kwargs):
        
        # --- INIT TIMING ---
        _t_init_start = time.perf_counter()
        _init_timings = {}  # step_name -> elapsed_ms
        SLOW_STEP_MS = 2000  # Warn if any step takes > 2 seconds

        def _time_step(name):
            """Record elapsed time since last call and warn if slow."""
            nonlocal _t_init_start
            elapsed_ms = (time.perf_counter() - _t_init_start) * 1000
            _init_timings[name] = elapsed_ms
            if elapsed_ms > SLOW_STEP_MS:
                _dprint(f"DEBUG-TIMING: *** SLOW *** [{name}] took {elapsed_ms:.0f}ms (> {SLOW_STEP_MS}ms threshold)")
            else:
                _dprint(f"DEBUG-TIMING: [{name}] {elapsed_ms:.0f}ms")
            _t_init_start = time.perf_counter()  # reset for next step

        # --- RESOURCE DETECTION ---
        self._system_ram_gb, self._system_cpu_cores = self._detect_system_resources()
        self._gpu_mem_gb = self._detect_gpu_memory()
        max_safe_pop = self._calculate_safe_population_size()
        max_safe_gens = self._calculate_safe_generations()
        _time_step('resource_detection')
        
        # --- CONFIGURATION ---
        requested_pop = int(population_size) if population_size else DEFAULT_POPULATION_SIZE
        self.population_size = min(requested_pop, max_safe_pop)
        requested_gens = int(num_generations) if num_generations else DEFAULT_NUM_GENERATIONS
        self.initial_num_generations = min(requested_gens, max_safe_gens)
        self.infinite_mode = bool(infinite_mode)
        self.base_mutation_rate = mutation_rate if mutation_rate else DEFAULT_MUTATION_RATE
        
        # --- POPULATION BATCH SIZE ---
        # User-configurable batch size: how many agents evaluate concurrently per batch.
        # Next batch is prefetched/prepared while the current batch evaluates.
        user_batch_size = kwargs.get('population_batch_size', None)
        
        # CPU evaluation workers — split evaluation across multiple processes
        self._cpu_eval_workers = max(1, min(int(kwargs.get('cpu_eval_workers', 1)), 32))
        
        # --- RESOURCE UTILIZATION TARGET: 70% ---
        # Process all agents every generation, but batch them to stay within 70% resource usage
        self._resource_target_fraction = 0.70
        self._auto_batch_size = self._calculate_agent_batch_size()
        
        # User override takes precedence over auto-calculated batch size
        if user_batch_size is not None:
            self._agent_batch_size = max(1, min(int(user_batch_size), self.population_size))
        else:
            self._agent_batch_size = self._auto_batch_size
        
        # --- GPU MODE ---
        self.target_device = target_device
        self.dynamic_benchmarking_active = bool(dynamic_benchmarking_active)
        
        self.gui_output_queue = gui_output_queue
        self.gui_stop_event = gui_stop_event or threading.Event()
        
        # --- STATE ---
        self.population = []
        self.fitness_history_per_generation = []
        self.generation_snapshots = []  # Lightweight per-generation population snapshots for GDV
        self._max_generation_snapshots = 500  # Cap snapshot memory for long/infinite runs
        self._last_eval_batch_agent_ids = []  # Populated by _evaluate_population per batch
        self.total_generations_elapsed = 0
        self.best_fitness_overall = 0.0
        self.best_agent_overall = None
        self._stagnation_counter = 0
        self._last_best_fitness = 0.0
        # Best robustness score from the most recent generation's
        # anti-corruption agents.  Reset every gen (robustness is measured
        # on different data each time, so an all-time best is meaningless).
        self._last_gen_best_robustness = 0.0
        # Best fitness measured on the CURRENT benchmark set (resets on refresh).
        # Used for benchmark sizing instead of best_fitness_overall, which is
        # inflated by scores on earlier, smaller benchmarks.
        self._current_benchmark_best_fitness = 0.0
        # Exponential moving average of sizing fitness across benchmark
        # refreshes.  Instead of hard-resetting to 0.0 (which causes the
        # oscillation: high score → size spike → AI crashes → size freefall),
        # we decay the old fitness so the sizing function has memory of the
        # AI's recent capability level.
        self._sizing_fitness_ema = 0.0
        # EMA decay factor: 0.0 = hard reset (old behaviour), 1.0 = no decay.
        # 0.4 means 40% of previous fitness is carried forward.
        self._sizing_fitness_ema_decay = 0.4

        # Best compression ratio (%) from the most recent generation's best agent.
        # Used to gate complexity advancement so the AI must prove it can
        # compress at ~70%+ before facing harder data patterns.
        self._last_best_compression_ratio = 0.0

        # Gold standard win rate from the most recent generation's head-to-head
        # benchmark.  Fraction (0.0-1.0) of items where the AI beat ALL
        # baseline compressors (gzip/bz2/lzma/zlib/zstd).
        # -1.0 means no gold standard data has been collected yet — gates
        # are bypassed until the first benchmark runs.
        self._last_gold_standard_win_rate = -1.0

        # Anti-corruption training phase & corruption level from the most
        # recent generation.  Surfaced in the WebUI dashboard.
        self._last_training_phase = ''
        self._last_corruption_level = 0.0
        
        # --- CHECKPOINT AUTO-SAVE ---
        # Auto-save every N generations and at the end of training.
        # The lock prevents race conditions when the WebUI also triggers a save.
        self._checkpoint_lock = threading.Lock()
        self._auto_checkpoint_interval = 10  # generations between auto-saves
        self._max_auto_checkpoints = 10      # rotate: keep only the N most recent auto-checkpoints
        
        # --- NOVELTY TRACKING ---
        # History of population-average method profiles per generation
        # Each entry: dict mapping method_name -> average usage fraction
        self._generation_method_history = []
        self._max_history_length = 50  # only track last 50 generations
        
        # --- LOGGING ---
        self.logger = logging.getLogger('EvolutionaryOptimizer')
        
        # --- SUBSYSTEMS ---
        _time_step('config_and_state')

        self.checkpoint_manager = None
        try:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=os.path.join(os.path.dirname(LOGS_DIR_PATH), "checkpoints"),
                logger=self.logger
            )
        except Exception: pass
        _time_step('checkpoint_manager')

        # --- GOLD STANDARD BENCHMARK ---
        self.gold_standard_benchmark = None
        if GoldStandardBenchmark:
            try:
                self.gold_standard_benchmark = GoldStandardBenchmark(
                    project_root=os.path.dirname(LOGS_DIR_PATH),
                    logger=self.logger,
                )
            except Exception as e_gsb:
                self.logger.warning(f"Gold standard benchmark init failed (non-fatal): {e_gsb}")
        _time_step('gold_standard_benchmark')

        self.benchmark_evaluator = None
        if BenchmarkItemEvaluator:
            self.benchmark_evaluator = BenchmarkItemEvaluator(
                logger_instance=self.logger,
                dynamic_benchmarking=self.dynamic_benchmarking_active
            )
            
            # Init Benchmarks
            if self.dynamic_benchmarking_active:
                try:
                    self._send_to_gui("Generating initial dynamic benchmark dataset...")
                    self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(
                        population_average_fitness=-1.0, 
                        current_generation=0,
                        best_compression_ratio=0.0
                    )
                    # --- DEBUG & FIX: Check Sizes ---
                    self._enforce_gpu_safe_benchmark_size()
                except Exception as e:
                    self._send_to_gui(f"Warning: Failed to init benchmarks: {e}", "warning")
        _time_step('benchmark_evaluator_init')
        
        if requested_pop != self.population_size or requested_gens != self.initial_num_generations:
            self._send_to_gui(
                f"Resource limits applied: Pop {requested_pop}->{self.population_size}, "
                f"Gens {requested_gens}->{self.initial_num_generations} "
                f"(RAM: {self._system_ram_gb:.1f}GB, CPU: {self._system_cpu_cores} cores, GPU: {self._gpu_mem_gb:.1f}GB)")
        mode_str = "INFINITE" if self.infinite_mode else str(self.initial_num_generations)
        workers_str = f", Workers: {self._cpu_eval_workers}" if self._cpu_eval_workers > 1 else ""
        self._send_to_gui(f"Optimizer Initialized. Device: {self.target_device}, Pop: {self.population_size}, Gens: {mode_str}, Batch: {self._agent_batch_size}{workers_str}")

        # --- BENCHMARK PRE-GENERATION STATE ---
        # Background thread produces the next benchmark set while the current
        # generation is evaluating / breeding, so refreshes are zero-cost.
        self._prefetch_benchmark_future = None   # Future from ThreadPoolExecutor
        self._prefetch_benchmark_items = None     # The pre-generated item list (set by worker)
        self._prefetch_benchmark_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bench_prefetch")

        # --- GITHUB FILE FETCHER (v0.9.7: phased training) ---
        # Lazy-initialised: the first time phased training needs GitHub files,
        # it creates the fetcher and triggers a background download.
        self._github_fetcher = None
        self._github_items_cache: list[str] = []  # Cached GitHub benchmark items

        # --- INIT TIMING SUMMARY ---
        total_init_ms = sum(_init_timings.values())
        timing_summary = ' | '.join(f"{k}: {v:.0f}ms" for k, v in _init_timings.items())
        _dprint(f"DEBUG-TIMING: Optimizer __init__ total: {total_init_ms:.0f}ms  [{timing_summary}]")
        self._send_to_gui(f"Init timing: {total_init_ms:.0f}ms — {timing_summary}")

    @staticmethod
    def _detect_system_resources():
        """Detect available RAM and CPU cores for safe resource limits."""
        ram_gb = 4.0
        cpu_cores = 2
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_gb = mem.available / (1024 ** 3)  # Available RAM, not total
            cpu_cores = psutil.cpu_count(logical=False) or 2
        except ImportError:
            try:
                import os as _os
                cpu_cores = _os.cpu_count() or 2
                # Rough fallback: assume 8GB available if we can't check
                ram_gb = 8.0
            except Exception:
                pass
        except Exception:
            pass
        return ram_gb, cpu_cores

    def _calculate_safe_population_size(self):
        """Calculate max safe population based on available RAM.
        Each agent uses ~2-5MB (Q-table + parameters + benchmark overhead).
        """
        # Reserve 2GB for OS + Python + overhead, rest for agents
        usable_ram_gb = max(0.5, self._system_ram_gb - 2.0)
        # ~5MB per agent as a conservative estimate
        agent_memory_mb = 5.0
        max_agents = int((usable_ram_gb * 1024) / agent_memory_mb)
        # Clamp between 10 and 500
        return max(10, min(500, max_agents))

    def _calculate_safe_generations(self):
        """Calculate max safe generations. Mainly bounded by time and memory growth."""
        # On low-RAM systems, cap generations to prevent memory fragmentation
        if self._system_ram_gb < 4.0:
            return 500
        elif self._system_ram_gb < 8.0:
            return 2000
        else:
            return 100000  # Effectively unlimited for infinite mode

    def set_continuous_run_enabled(self, enabled: bool):
        """Toggle infinite mode at runtime (e.g. from the GUI checkbox).

        When *enabled* is True the evolution loop ignores the generation
        limit and runs until the stop event is set.  When False the loop
        will stop at ``initial_num_generations``.
        """
        self.infinite_mode = bool(enabled)

    @staticmethod
    def _detect_gpu_memory():
        """Detect available GPU memory in GB."""
        gpu_mem_gb = 0.0
        try:
            import cupy as _cp
            free, total = _cp.cuda.Device(0).mem_info
            gpu_mem_gb = total / (1024 ** 3)
        except Exception:
            try:
                import subprocess
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    text=True, timeout=5
                )
                gpu_mem_gb = float(result.strip().split('\n')[0]) / 1024
            except Exception:
                pass
        return gpu_mem_gb

    def _calculate_agent_batch_size(self):
        """Calculate how many agents to evaluate in parallel batches.
        
        Targets 70% of available resources. All agents still run every generation,
        but they are processed in batches of this size to control memory/GPU pressure.
        """
        # Base batch size on available RAM (70% utilization target)
        usable_ram_gb = max(0.5, self._system_ram_gb * self._resource_target_fraction)
        # Each agent evaluation uses ~10-50MB (benchmark items + compress buffers)
        agent_eval_memory_mb = 30.0  # conservative estimate
        
        ram_batch = int((usable_ram_gb * 1024) / agent_eval_memory_mb)
        
        # If GPU is available, also consider VRAM
        gpu_batch = 999  # default no GPU limit
        if self._gpu_mem_gb > 0:
            usable_vram_gb = self._gpu_mem_gb * self._resource_target_fraction
            # GPU RLE workspace + data buffers per agent
            gpu_agent_mem_mb = 50.0
            gpu_batch = max(5, int((usable_vram_gb * 1024) / gpu_agent_mem_mb))
        
        # Also consider CPU cores — no point having more parallel agents than cores
        cpu_batch = max(5, self._system_cpu_cores * 4)  # agents are I/O-light, so 4x cores is fine
        
        # Take the minimum of all resource constraints
        batch_size = min(ram_batch, gpu_batch, cpu_batch)
        
        # Clamp to reasonable range: at least 5, at most population_size (all at once)
        batch_size = max(5, min(batch_size, self.population_size))
        
        return batch_size

    def _send_to_gui(self, message, log_level="info"):
        print(f"[Optimizer] {message}")
        if self.gui_output_queue:
            try: self.gui_output_queue.put_nowait(f"[ELS] {message}")
            except queue.Full: pass

    def _send_metrics_json(self, generation, fitness, ratio, benchmark_size):
        if self.gui_output_queue:
            try:
                # Gather complexity info from the benchmark evaluator
                complexity_tier = 'UNKNOWN'
                complexity_value = 0
                tier_budget_mb = 0.0
                tier_ceiling_kb = 0.0
                try:
                    if self.benchmark_evaluator:
                        from ..utils.benchmark_evaluator import (
                            get_generation_size_limits, DataComplexity, COMPLEXITY_FITNESS_THRESHOLDS
                        )
                        # READ the current complexity tier — do NOT call
                        # determine_target_complexity() here because that
                        # mutates _current_complexity_tier and would cause
                        # double-advancement (once here, once in
                        # generate_and_set_dynamic_benchmark_items).
                        be = self.benchmark_evaluator
                        tier_enum = getattr(be, '_current_complexity_tier', None)
                        if tier_enum is not None:
                            complexity_tier = getattr(tier_enum, 'name', 'UNKNOWN')
                            # Use continuous _complexity_pct (0-100) directly
                            complexity_value = getattr(be, '_complexity_pct', 0)
                        # Determine active generation size tier (with ratio gating)
                        last_ratio = getattr(self, '_last_best_compression_ratio', 0.0)
                        size_dwell = getattr(be, '_previous_size_tier_refreshes', 0)
                        prev_tier_idx = getattr(be, '_previous_tier_index', -1)
                        budget, ceiling, _tier_idx = get_generation_size_limits(
                            generation, fitness,
                            previous_tier_index=prev_tier_idx,
                            best_compression_ratio=last_ratio,
                            refreshes_at_tier=size_dwell,
                            gold_standard_win_rate=getattr(self, '_last_gold_standard_win_rate', -1.0))
                        tier_budget_mb = budget / (1024 * 1024)
                        tier_ceiling_kb = ceiling / 1024
                except Exception:
                    pass

                # Gather top agent's decompression stats
                decomp_mismatches = 0
                items_evaluated = 0
                successful_compressions = 0
                try:
                    if self.population:
                        top = self.population[0]
                        es = getattr(top, 'evaluation_stats', None)
                        if es and isinstance(es, dict):
                            decomp_mismatches = es.get('decomp_failures_mismatch', 0)
                            items_evaluated = es.get('items_evaluated', 0)
                            successful_compressions = es.get('successful_rle', 0)
                except Exception:
                    pass

                # --- Per-method compression stats from top agent ---
                method_stats = {}
                novel_pipeline_name = 'none'
                try:
                    if self.population:
                        top = self.population[0]
                        es = getattr(top, 'evaluation_stats', None)
                        if es and isinstance(es, dict):
                            method_stats = {
                                'bytes_saved': es.get('method_bytes_saved', {}),
                                'attempts': es.get('method_attempts', {}),
                                'successes': es.get('method_successes', {}),
                                'profile': es.get('method_profile', {}),
                            }
                        ai = getattr(top, 'puffin_ai', None) or (
                            top.get_puffin_ai() if hasattr(top, 'get_puffin_ai') else None)
                        if ai:
                            nm = getattr(ai, 'novel_method', None)
                            if nm and hasattr(nm, 'metadata'):
                                novel_pipeline_name = nm.metadata.get('pipeline', 'none')
                except Exception:
                    pass

                payload = json.dumps({
                    'generation': generation,
                    'fitness': float(fitness),
                    'ratio': float(ratio),
                    'benchmark_size': float(benchmark_size),
                    'complexity_tier': complexity_tier,
                    'complexity_value': complexity_value,
                    'tier_budget_mb': round(tier_budget_mb, 1),
                    'tier_ceiling_kb': round(tier_ceiling_kb, 0),
                    'best_robustness': round(float(self._last_gen_best_robustness), 4),
                    'training_phase': self._last_training_phase,
                    'corruption_level': round(float(self._last_corruption_level), 4),
                    'decomp_mismatches': decomp_mismatches,
                    'items_evaluated': items_evaluated,
                    'successful_compressions': successful_compressions,
                    'gold_standard_win_rate': round(float(self._last_gold_standard_win_rate), 3),
                    'method_stats': method_stats,
                    'novel_pipeline': novel_pipeline_name,
                })
                self.gui_output_queue.put_nowait(f"METRICS_JSON:{payload}")
            except queue.Full: pass

    def _snapshot_generation(self, generation_num, reported_best_fitness=None):
        """Capture lightweight snapshot of the current population for the GDV history.

        Stores per-batch agent summaries (IDs, fitness, key params) without Q-tables.
        Sends GEN_SNAPSHOT:<gen> via the queue so the GUI can update lazily.

        Args:
            generation_num: The generation number (1-based).
            reported_best_fitness: If provided, use this as the snapshot's best_fitness
                to stay consistent with the dashboard metrics (which apply a carry-over
                guard against 0.0 drops after benchmark refreshes).
        """
        if not self.population:
            return

        import time as _time_mod
        # Build a quick agent-id → agent lookup from the (already sorted) population
        agent_lookup = {a.agent_id: a for a in self.population}

        batches_data = []
        for batch_idx, id_list in enumerate(self._last_eval_batch_agent_ids):
            batch_agents = []
            batch_fitnesses = []
            for aid in id_list:
                agent = agent_lookup.get(aid)
                if agent is None:
                    continue
                fit = agent.get_fitness()
                ai_core = agent.get_puffin_ai() if hasattr(agent, 'get_puffin_ai') else None
                summary = {
                    "agent_id": agent.agent_id,
                    "fitness": fit if fit is not None else float('nan'),
                    "generation_born": getattr(agent, 'generation_born', 0),
                    "parent_ids": list(getattr(agent, 'parent_ids', [])),
                    "learning_rate": getattr(ai_core, 'learning_rate', 0.0) if ai_core else 0.0,
                    "exploration_rate": getattr(ai_core, 'exploration_rate', 0.0) if ai_core else 0.0,
                    "rle_min_run": getattr(ai_core, 'rle_min_encodable_run_length', 'N/A') if ai_core else 'N/A',
                    "thresholds_str": ", ".join(
                        map(str, ai_core.len_thresholds)) if ai_core and hasattr(ai_core, 'len_thresholds') and ai_core.len_thresholds else "N/A",
                    "evaluation_stats": dict(getattr(agent, 'evaluation_stats', {}) or {}),
                    # Novel method pipeline info for method-level visibility
                    "novel_pipeline": (
                        ai_core.novel_method.metadata.get('pipeline', 'none')
                        if ai_core and hasattr(ai_core, 'novel_method') and ai_core.novel_method
                           and hasattr(ai_core.novel_method, 'metadata')
                        else 'none'
                    ),
                    "has_novel_method": bool(
                        ai_core and hasattr(ai_core, '_novel_compress_fn')
                        and ai_core._novel_compress_fn is not None
                    ),
                }
                batch_agents.append(summary)
                if fit is not None and fit > -999:
                    batch_fitnesses.append(fit)

            batches_data.append({
                "batch_idx": batch_idx,
                "agents": batch_agents,
                "best_fitness": max(batch_fitnesses) if batch_fitnesses else 0.0,
                "avg_fitness": (sum(batch_fitnesses) / len(batch_fitnesses)) if batch_fitnesses else 0.0,
            })

        all_fitnesses = [a.get_fitness() for a in self.population
                         if a.get_fitness() is not None and a.get_fitness() > -999]

        # Use the reported best_fitness (from metrics, which applies carry-over guard)
        # so the deep-dive chart stays consistent with the dashboard display.
        snapshot_best = reported_best_fitness if reported_best_fitness is not None else (
            max(all_fitnesses) if all_fitnesses else 0.0)

        # Compute avg_fitness excluding extreme outliers from failed/timed-out evaluations.
        # Agents with fitness <= -50 are dominated by EVALUATION_FAIL_REWARD (-100) or
        # EVALUATION_TIMEOUT_REWARD_PENALTY (-50) and would distort the population average.
        _OUTLIER_THRESHOLD = -50.0
        meaningful_fitnesses = [f for f in all_fitnesses if f > _OUTLIER_THRESHOLD]
        avg_fit = (sum(meaningful_fitnesses) / len(meaningful_fitnesses)) if meaningful_fitnesses else (
            (sum(all_fitnesses) / len(all_fitnesses)) if all_fitnesses else 0.0)

        snapshot = {
            "generation": generation_num,
            "timestamp": _time_mod.time(),
            "best_fitness": snapshot_best,
            "avg_fitness": avg_fit,
            "min_fitness": min(all_fitnesses) if all_fitnesses else 0.0,
            "agent_count": len(self.population),
            "batch_count": len(batches_data),
            "batches": batches_data,
        }
        self.generation_snapshots.append(snapshot)

        # Cap memory usage for long / infinite runs.
        if len(self.generation_snapshots) > self._max_generation_snapshots:
            # Keep the most recent snapshots; drop the oldest.
            self.generation_snapshots = self.generation_snapshots[-self._max_generation_snapshots:]

        # Notify GUI
        if self.gui_output_queue:
            try:
                self.gui_output_queue.put_nowait(f"GEN_SNAPSHOT:{generation_num}")
            except queue.Full:
                pass

    def _enforce_gpu_safe_benchmark_size(self):
        """
        DEBUG METHOD: Checks the size of generated benchmark items.
        If they are too small for the GPU kernel (e.g. < 4KB), it pads them.
        """
        if not self.benchmark_evaluator or not self.benchmark_evaluator.benchmark_items:
            return

        # GPU Kernels often fail on items smaller than block size (e.g. 1024 bytes)
        # We enforce a 4KB minimum to be safe and debug the issue.
        GPU_SAFE_MIN_SIZE = 4096
        
        items = self.benchmark_evaluator.benchmark_items
        fixed_items = []
        sizes = []
        was_fixed = False

        for i, item in enumerate(items):
            # item is usually a tuple (text, something) or just text
            text = item
            if isinstance(item, tuple): text = item[0]
            
            current_len = len(text)
            sizes.append(current_len)
            
            # Print specific debug info for Item 8 (where you saw the error)
            if i == 8:
                self._send_to_gui(f"DEBUG: Item 8 Original Size: {current_len} bytes.")

            if current_len < GPU_SAFE_MIN_SIZE and current_len > 0:
                # Calculate how many times to repeat
                repeats = math.ceil(GPU_SAFE_MIN_SIZE / current_len)
                new_text = text * repeats
                fixed_items.append(new_text)
                was_fixed = True
            else:
                fixed_items.append(text)
        
        if was_fixed:
            self.benchmark_evaluator.benchmark_items = fixed_items
            min_s = min(sizes)
            self._send_to_gui(f"DEBUG: Enforced GPU Safe Sizes. Smallest was {min_s}b -> Now all > {GPU_SAFE_MIN_SIZE}b.")
        else:
             self._send_to_gui(f"DEBUG: All items safe (Min: {min(sizes)}b).")

    def _sanitize_agent(self, agent):
        if not hasattr(agent, 'puffin_ai'): return
        ai = agent.puffin_ai
        # Use config-aligned bounds (not hardcoded 32) so mutations aren't nullified
        SAFE_THRESH_MIN = 2  # Matches config thresholds
        RLE_MIN = getattr(ai, 'RLE_MIN_RUN_BOUNDS_MIN', 2)
        RLE_MAX = getattr(ai, 'RLE_MIN_RUN_BOUNDS_MAX', 7)
        if hasattr(ai, 'len_thresholds'):
            safe_thresholds = [max(SAFE_THRESH_MIN, int(x)) for x in ai.len_thresholds]
            ai.len_thresholds = sorted(set(safe_thresholds))  # also deduplicate
        if hasattr(ai, 'rle_min_encodable_run_length'):
            ai.rle_min_encodable_run_length = max(RLE_MIN, min(RLE_MAX, ai.rle_min_encodable_run_length))

    def _create_initial_population(self):
        _pop_t0 = time.perf_counter()
        self._send_to_gui(f"Creating {self.population_size} agents...")
        if not PuffinZipAI or not EvolvingAgent:
            self._send_to_gui("CRITICAL: AI Core not loaded.", "error")
            return []

        # Detect if we're using NN agents
        _is_nn_mode = getattr(PuffinZipAI, 'MODEL_TYPE', None) == 'dqn'
        if _is_nn_mode:
            self._send_to_gui("Neural-network (DQN) agents enabled — each agent gets a trainable MLP policy.")

        # Get the novel compression generator
        novel_gen = None
        if get_novel_generator:
            try:
                novel_gen = get_novel_generator()
                self._send_to_gui("Novel Compression Generator loaded — each agent will get a unique method.")
            except Exception as e:
                self._send_to_gui(f"Warning: NovelCompressionGenerator not available: {e}", "warning")
        _pop_t1 = time.perf_counter()
        _dprint(f"DEBUG-TIMING: [pop:novel_gen_load] {(_pop_t1 - _pop_t0)*1000:.0f}ms")

        # --- WARM the GPU validation cache ONCE before creating any agent ---
        # This avoids 50x redundant GPU device probes.
        # (Only needed for CuPy-based GPU agents, not NN agents which use PyTorch.)
        if not _is_nn_mode:
            try:
                from ..gpu_core.gpu_ai_agent import _validate_gpu_once
                cache = _validate_gpu_once(self.target_device)
                if cache.get('gpu_ok'):
                    self._send_to_gui(f"GPU pre-validated: GPU {cache['gpu_id']} ({cache['device_name']}) — skipping per-agent device probes.")
                else:
                    self._send_to_gui("GPU pre-validation: GPU not available, agents will use CPU.", "warning")
            except ImportError:
                pass  # Non-GPU build, that's fine
        _pop_t2 = time.perf_counter()
        _dprint(f"DEBUG-TIMING: [pop:gpu_cache_warmup] {(_pop_t2 - _pop_t1)*1000:.0f}ms")

        new_pop = []
        start_time = time.perf_counter()

        # --- Phase 1: Create agents with DEFERRED GPU init (CPU-only work) ---
        # This is the parallelizable part: Q-table numpy creation, novel method gen, param randomization
        agents_pending_gpu = []

        def _build_single_agent(index):
            """Create a single agent on CPU with deferred GPU transfer."""
            thresholds = sorted(random.sample(range(2, 500), random.randint(2, 8)))
            rle_min_run = random.randint(2, 7)
            ai_params = {
                'len_thresholds': thresholds,
                'learning_rate': random.uniform(0.001, 0.5),
                'target_device': self.target_device,
                'rle_min_encodable_run': rle_min_run,          # matches GPU constructor param name
                '_defer_gpu_transfer': True,                    # skip per-agent GPU init
            }

            def randomize_q(core):
                if hasattr(core, 'q_table'):
                    core.q_table = np.random.uniform(-0.1, 0.1, core.q_table.shape)

            try:
                ai_core = PuffinZipAI(**ai_params)
                randomize_q(ai_core)
            except TypeError:
                # Fallback: try without _defer_gpu_transfer (CPU-only build)
                # Still pass rle_min via the alternate name the base class accepts
                fallback_params = {
                    'len_thresholds': thresholds,
                    'learning_rate': ai_params['learning_rate'],
                    'target_device': self.target_device,
                    'rle_min_encodable_run_length': rle_min_run,
                }
                try:
                    ai_core = PuffinZipAI(**fallback_params)
                    randomize_q(ai_core)
                except Exception as e:
                    print(f"Failed to create agent {index}: {e}")
                    return None

            # Assign novel compression method (CPU-only work)
            if novel_gen:
                try:
                    num_patterns = random.randint(1, 3)
                    method = novel_gen.generate_novelty_method(
                        method_name=f"agent_{index}_novel_v1",
                        pattern_combo_size=num_patterns
                    )
                    ai_core.novel_method = method
                    ai_core._novel_compress_fn = method.compress_fn
                    ai_core._novel_decompress_fn = method.decompress_fn
                except Exception as e:
                    pass  # Novel methods are optional

            # --- v0.9.7: 50/50 agent type split ---
            # First half = compression agents, second half = anti-corruption agents
            half = self.population_size // 2
            agent_type = "compression" if index < half else "anti_corruption"

            try:
                agent = EvolvingAgent(
                    ai_core, generation_born=0,
                    agent_id=f"gen0_agent_{index}",
                    agent_type=agent_type,
                )
                self._sanitize_agent(agent)
                return agent
            except Exception as e:
                print(f"Error initializing agent {index}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel CPU-side agent creation
        # Novel method generation + numpy Q-table creation can overlap
        MAX_WORKERS = min(8, os.cpu_count() or 4)
        _agent_times = []  # Track per-agent creation time for slow-agent detection
        _deferred_count = 0
        _fallback_count = 0

        def _build_timed(index):
            """Wrapper that times each agent build and detects deferred vs fallback."""
            t0 = time.perf_counter()
            agent = _build_single_agent(index)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _agent_times.append((index, elapsed_ms))
            if elapsed_ms > 2000:
                _dprint(f"DEBUG-TIMING: *** SLOW AGENT *** agent {index} took {elapsed_ms:.0f}ms to create")
            return agent

        with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="agent_init") as pool:
            futures = {}
            for i in range(self.population_size):
                if self.gui_stop_event.is_set():
                    break
                futures[pool.submit(_build_timed, i)] = i

            for future in futures:
                if self.gui_stop_event.is_set():
                    break
                try:
                    agent = future.result(timeout=60)
                    if agent is not None:
                        new_pop.append(agent)
                except Exception as e:
                    print(f"Agent creation future failed: {e}")
                    traceback.print_exc()

        cpu_elapsed = time.perf_counter() - start_time

        # Check how many agents actually used deferred mode vs fallback
        for agent in new_pop:
            ai = getattr(agent, 'puffin_ai', None)
            if ai and getattr(ai, '_deferred_gpu', False):
                _deferred_count += 1
            else:
                _fallback_count += 1

        # Per-agent timing diagnostics
        if _agent_times:
            times_ms = [t for _, t in _agent_times]
            avg_ms = sum(times_ms) / len(times_ms)
            max_ms = max(times_ms)
            slow_agents = [(idx, ms) for idx, ms in _agent_times if ms > 2000]
            _dprint(f"DEBUG-TIMING: [pop:cpu_phase] {cpu_elapsed*1000:.0f}ms total | "
                  f"avg {avg_ms:.0f}ms/agent | max {max_ms:.0f}ms | "
                  f"deferred={_deferred_count}, fallback(no defer)={_fallback_count}, "
                  f"slow(>2s)={len(slow_agents)}")
            if _fallback_count > 0:
                _dprint(f"DEBUG-TIMING: *** WARNING *** {_fallback_count} agents fell back to non-deferred init "
                      f"(full GPU probe per agent). This is the primary cause of slow population init.")
                self._send_to_gui(f"WARNING: {_fallback_count}/{len(new_pop)} agents used slow fallback init (no deferred GPU). Check console.", "warning")

        self._send_to_gui(f"CPU-side agent creation done in {cpu_elapsed:.2f}s for {len(new_pop)} agents. "
                          f"(deferred={_deferred_count}, fallback={_fallback_count})")

        # --- Phase 2: Bulk GPU finalization (one batch transfer instead of 50 individual ones) ---
        gpu_start = time.perf_counter()
        gpu_finalized = 0
        for agent in new_pop:
            ai = getattr(agent, 'puffin_ai', None)
            if ai and hasattr(ai, 'finalize_gpu_init') and getattr(ai, '_deferred_gpu', False):
                ai.finalize_gpu_init()
                gpu_finalized += 1

        gpu_elapsed = time.perf_counter() - gpu_start
        total_elapsed = time.perf_counter() - _pop_t0  # from very start of method
        _dprint(f"DEBUG-TIMING: [pop:gpu_finalize] {gpu_elapsed*1000:.0f}ms ({gpu_finalized} agents)")
        _dprint(f"DEBUG-TIMING: [pop:TOTAL] {total_elapsed*1000:.0f}ms")
        if total_elapsed > 10:
            _dprint(f"DEBUG-TIMING: *** SLOW POPULATION *** Total init > 10s ({total_elapsed:.1f}s) — "
                  f"check per-agent breakdown above")
        self._send_to_gui(f"Population created. Count: {len(new_pop)} | CPU: {cpu_elapsed:.2f}s | "
                          f"GPU finalize ({gpu_finalized} agents): {gpu_elapsed:.2f}s | Total: {total_elapsed:.2f}s")
        return new_pop

    def _prepare_batch(self, agents):
        """Prepare a batch of agents for evaluation (sanitize, reset fitness).
        Called from a prefetch thread so the next batch is ready while the current one evaluates.
        """
        for agent in agents:
            self._sanitize_agent(agent)
        return agents

    def _evaluate_population(self, population_to_evaluate, generation_num):
        if not population_to_evaluate or not self.benchmark_evaluator: return 0.0
        
        # 1. Force Reset Fitness & propagate generation number
        for agent in population_to_evaluate:
            if hasattr(agent, 'fitness'): agent.fitness = None
            if hasattr(agent, 'set_fitness'): agent.set_fitness(None)
            # Set _current_generation on the AI core so scaffolding knows
            # what generation this agent is being evaluated at
            ai_core = getattr(agent, 'puffin_ai', None)
            if ai_core is not None:
                ai_core._current_generation = generation_num

        # 2. Benchmark Check & SIZING FIX
        if not self.benchmark_evaluator.benchmark_items:
            self._send_to_gui("No benchmark items found! Generating fallback dataset...")
            try:
                self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(
                    population_average_fitness=-1.0, 
                    current_generation=generation_num,
                    best_compression_ratio=self._last_best_compression_ratio
                )
                self._enforce_gpu_safe_benchmark_size()
            except Exception as e:
                self._send_to_gui(f"Failed to regenerate benchmarks: {e}", "error")
                return 0.0

        BATCH_SIZE = self._agent_batch_size
        total_agents = len(population_to_evaluate)
        valid_scores = []
        self._last_eval_batch_agent_ids = []  # Reset for this generation
        
        # Split population into batches
        batches = []
        for i in range(0, total_agents, BATCH_SIZE):
            batches.append(population_to_evaluate[i : i + BATCH_SIZE])
        
        num_batches = len(batches)
        workers = self._cpu_eval_workers
        workers_str = f" ({workers} workers)" if workers > 1 else ""
        self._send_to_gui(f"Gen {generation_num}: Evaluating {total_agents} agents in {num_batches} batches of {BATCH_SIZE}{workers_str}")
        
        # Prefetched batch evaluation:
        # While batch N evaluates, batch N+1 is being prepared (sanitized) in a thread
        prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        
        # GPU+CPU pipeline: create process pool for parallel compression
        cpu_pool = None
        if workers > 1:
            try:
                from puffinzip_ai.utils.benchmark_evaluator import _pipeline_worker_init
                cpu_pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_pipeline_worker_init,
                    initargs=(list(self.benchmark_evaluator.benchmark_items),)
                )
                self._send_to_gui(f"Gen {generation_num}: ProcessPool started ({workers} CPU workers + GPU batch inference)")
            except Exception as e_pool:
                self._send_to_gui(f"ProcessPool init failed ({e_pool}), falling back to sequential", "warning")
                cpu_pool = None
        
        try:
            # Prepare the first batch immediately
            self._prepare_batch(batches[0])
            prefetch_future = None
            
            for batch_idx, chunk in enumerate(batches):
                if self.gui_stop_event.is_set(): 
                    self._send_to_gui("Evaluation interrupted by user.")
                    return 0.0
                
                # If there's a next batch, start prefetching it now
                if batch_idx + 1 < num_batches:
                    prefetch_future = prefetch_executor.submit(self._prepare_batch, batches[batch_idx + 1])
                else:
                    prefetch_future = None

                # Evaluate current batch — GPU+CPU pipeline or sequential
                _batch_t0 = time.perf_counter()
                try:
                    if workers > 1 and cpu_pool is not None and len(chunk) > 1:
                        # GPU+CPU pipeline: batch inference → parallel compression
                        pipeline_results = self.benchmark_evaluator.evaluate_population_pipelined(
                            chunk, cpu_pool, self.gui_stop_event
                        )
                        if pipeline_results:
                            for j, agent in enumerate(chunk):
                                if j < len(pipeline_results):
                                    fit, stats = pipeline_results[j]
                                    agent.set_fitness(fit)
                                    agent.evaluation_stats = stats
                                    if fit > -999: valid_scores.append(fit)
                    else:
                        # Single-worker path (original)
                        results = self.benchmark_evaluator.evaluate_population_batch(
                            chunk,
                            repetitions_per_item=1,
                            gui_stop_event=self.gui_stop_event
                        )
                        
                        if results:
                            for j, agent in enumerate(chunk):
                                if j < len(results):
                                    fit, stats = results[j]
                                    agent.set_fitness(fit)
                                    agent.evaluation_stats = stats
                                    if fit > -999: valid_scores.append(fit)
                except Exception as e:
                    self._send_to_gui(f"Batch Error (G{generation_num} B{batch_idx}): {e}", "error")
                
                _batch_elapsed = time.perf_counter() - _batch_t0
                batch_best = max((s for s in valid_scores), default=0.0)
                self._send_to_gui(f"Gen {generation_num}: Batch {batch_idx + 1}/{num_batches} done ({_batch_elapsed:.1f}s, best so far: {batch_best:.4f})")
                # Record batch agent IDs for generation snapshot
                self._last_eval_batch_agent_ids.append([a.agent_id for a in chunk])
                _dprint(f"DEBUG-TIMING: [eval_batch_{batch_idx}] {_batch_elapsed*1000:.0f}ms ({len(chunk)} agents)")
                
                # Wait for prefetch to complete before proceeding to next batch
                if prefetch_future is not None:
                    try:
                        prefetch_future.result(timeout=60)
                    except Exception as e_prefetch:
                        self._send_to_gui(f"Prefetch warning: {e_prefetch}", "warning")
        except KeyboardInterrupt:
            self._send_to_gui("Evaluation interrupted (KeyboardInterrupt) — shutting down worker pool.", "warning")
            return 0.0
        finally:
            prefetch_executor.shutdown(wait=False)
            if cpu_pool is not None:
                cpu_pool.shutdown(wait=False, cancel_futures=True)

        # --- POPULATION-LEVEL NOVELTY SCORING ---
        # After all agents are evaluated, apply novelty bonuses/penalties
        # based on how unique each agent's strategy is vs. the population
        try:
            method_profiles = []
            for agent in population_to_evaluate:
                profile = {}
                if hasattr(agent, 'evaluation_stats') and isinstance(agent.evaluation_stats, dict):
                    profile = agent.evaluation_stats.get('method_profile', {})
                method_profiles.append(profile)
            
            # 1. Population novelty: reward unique strategies, penalize conformists
            if _calculate_population_novelty_scores and any(method_profiles):
                novelty_scores = _calculate_population_novelty_scores(method_profiles)
                for idx, agent in enumerate(population_to_evaluate):
                    if idx < len(novelty_scores) and agent.get_fitness() is not None:
                        old_fit = agent.get_fitness()
                        agent.set_fitness(old_fit + novelty_scores[idx])
                        if hasattr(agent, 'evaluation_stats') and isinstance(agent.evaluation_stats, dict):
                            agent.evaluation_stats['novelty_adjustment'] = novelty_scores[idx]
            
            # 2. Generation history penalty: penalize methods overused across generations
            if _calculate_generation_repetition_penalty and self._generation_method_history:
                for idx, agent in enumerate(population_to_evaluate):
                    if idx < len(method_profiles) and method_profiles[idx] and agent.get_fitness() is not None:
                        gen_penalty = _calculate_generation_repetition_penalty(
                            method_profiles[idx],
                            self._generation_method_history
                        )
                        old_fit = agent.get_fitness()
                        agent.set_fitness(old_fit + gen_penalty)
                        if hasattr(agent, 'evaluation_stats') and isinstance(agent.evaluation_stats, dict):
                            agent.evaluation_stats['gen_repetition_penalty'] = gen_penalty
            
            # 3. Record this generation's average method profile for future penalty calculation
            if method_profiles:
                avg_profile = {}
                valid_profiles = [p for p in method_profiles if p]
                if valid_profiles:
                    all_methods = set()
                    for p in valid_profiles:
                        all_methods.update(p.keys())
                    for method in all_methods:
                        avg_profile[method] = sum(p.get(method, 0.0) for p in valid_profiles) / len(valid_profiles)
                    self._generation_method_history.append(avg_profile)
                    # Trim history to max length
                    if len(self._generation_method_history) > self._max_history_length:
                        self._generation_method_history = self._generation_method_history[-self._max_history_length:]
        except Exception as e_novelty:
            self._send_to_gui(f"Novelty scoring error (non-fatal): {e_novelty}", "warning")

        # --- v0.9.7: TYPE-AWARE EVALUATION + PHASED TRAINING ---
        # After clean evaluation, run a SPECIALISED second pass for
        # anti_corruption agents.  The data they see is built by the
        # evaluator's central ``get_anti_corruption_benchmark_items()`` API
        # which encapsulates all phase/ratio/corruption/garbage logic.
        #
        # The second pass uses the GPU+CPU pipeline when available
        # (``workers > 1``) so anti-corruption agents get the same
        # high-performance evaluation path as compression agents.
        #
        # Compression agents keep their clean fitness untouched.
        try:
            anti_corruption_agents = [
                a for a in population_to_evaluate
                if getattr(a, 'agent_type', 'compression') == 'anti_corruption'
            ]
            if anti_corruption_agents and self.benchmark_evaluator:
                # Store clean fitness as compression_fitness for ALL agents first
                for agent in population_to_evaluate:
                    fit = agent.get_fitness()
                    if fit is not None and hasattr(agent, 'compression_fitness'):
                        agent.compression_fitness = fit if fit > -999 else 0.0

                # --- Fetch GitHub items if phased training asks for them ---
                phased_enabled = _PHASED_TRAINING_ENABLED
                github_items: list[str] = []
                if phased_enabled and generation_num > _PHASED_TRAINING_PHASE1_END:
                    if GitHubFileFetcher is not None:
                        github_items = self._get_github_benchmark_items(
                            count=_PHASED_TRAINING_GITHUB_ITEM_COUNT
                        )
                    else:
                        self._send_to_gui(
                            f"Gen {generation_num}: GitHubFileFetcher unavailable "
                            f"(missing 'requests' package?); using corrupted data only.",
                            "warning",
                        )

                # --- Build anti-corruption benchmark set via central API ---
                clean_snapshot = list(self.benchmark_evaluator.benchmark_items)
                anti_corr_items, phase_label, corruption_level = (
                    self.benchmark_evaluator.get_anti_corruption_benchmark_items(
                        generation_num=generation_num,
                        clean_items=clean_snapshot,
                        github_items=github_items if github_items else None,
                        phased_enabled=phased_enabled,
                        phase1_end=_PHASED_TRAINING_PHASE1_END,
                        phase2_end=_PHASED_TRAINING_PHASE2_END,
                        phase3_github_ratio=_PHASED_TRAINING_PHASE3_GITHUB_RATIO,
                    )
                )
                github_items_used = getattr(
                    self.benchmark_evaluator, '_last_anti_corr_github_used', 0
                )

                if anti_corr_items:
                    # Save originals BEFORE the swap; the swap + eval is
                    # entirely inside try/finally so the evaluator is always
                    # restored — even on stop-event, KeyboardInterrupt, or
                    # any other exception path.
                    original_items = list(self.benchmark_evaluator.benchmark_items)
                    old_baseline_gen = getattr(
                        self.benchmark_evaluator, '_cached_baseline_generation', None
                    )
                    try:
                        self.benchmark_evaluator.benchmark_items = anti_corr_items
                        self.benchmark_evaluator._cached_baseline_generation = None

                        # Evaluate anti-corruption agents on the phased data.
                        # Use the GPU+CPU pipeline when workers > 1 so the
                        # anti-corruption pass is NOT slower than the clean
                        # compression pass.
                        corrupt_results = None
                        if self._cpu_eval_workers > 1 and len(anti_corruption_agents) > 1:
                            acorr_pool = None
                            try:
                                from puffinzip_ai.utils.benchmark_evaluator import _pipeline_worker_init
                                acorr_pool = ProcessPoolExecutor(
                                    max_workers=self._cpu_eval_workers,
                                    initializer=_pipeline_worker_init,
                                    initargs=(list(anti_corr_items),),
                                )
                                corrupt_results = self.benchmark_evaluator.evaluate_population_pipelined(
                                    anti_corruption_agents, acorr_pool, self.gui_stop_event
                                )
                            except Exception as e_pipe:
                                self._send_to_gui(
                                    f"Gen {generation_num}: Anti-corr pipeline failed ({e_pipe}), "
                                    f"falling back to sequential.",
                                    "warning",
                                )
                                corrupt_results = None
                            finally:
                                if acorr_pool is not None:
                                    acorr_pool.shutdown(wait=False, cancel_futures=True)

                        if corrupt_results is None:
                            # Sequential fallback (single worker or pipeline error)
                            corrupt_results = self.benchmark_evaluator.evaluate_population_batch(
                                anti_corruption_agents,
                                repetitions_per_item=1,
                                gui_stop_event=self.gui_stop_event,
                            )

                        if corrupt_results:
                            for j, agent in enumerate(anti_corruption_agents):
                                if j < len(corrupt_results):
                                    rfit, rstats = corrupt_results[j]
                                    agent.set_robustness_fitness(rfit if rfit > -999 else 0.0)
                                    if hasattr(agent, 'evaluation_stats') and isinstance(agent.evaluation_stats, dict):
                                        agent.evaluation_stats['robustness_fitness'] = rfit
                                        agent.evaluation_stats['corruption_eval'] = rstats
                                        agent.evaluation_stats['training_phase'] = phase_label
                    except Exception as e_corrupt:
                        self._send_to_gui(f"Gen {generation_num}: Corruption eval error (non-fatal): {e_corrupt}", "warning")
                    finally:
                        # ALWAYS restore clean benchmark items + cached baseline
                        self.benchmark_evaluator.benchmark_items = original_items
                        self.benchmark_evaluator._cached_baseline_generation = old_baseline_gen

                    self._send_to_gui(
                        f"Gen {generation_num}: {phase_label} — "
                        f"{len(anti_corruption_agents)} anti-corruption agents, "
                        f"{len(anti_corr_items)} items "
                        f"({len(anti_corr_items) - github_items_used} corrupted"
                        f"{f' + {github_items_used} GitHub' if github_items_used > 0 else ''}"
                        f"), corruption={corruption_level:.3f}")
                    # Store phase info for METRICS_JSON (surfaced in WebUI)
                    self._last_training_phase = phase_label
                    self._last_corruption_level = corruption_level
                else:
                    for agent in anti_corruption_agents:
                        if hasattr(agent, 'robustness_fitness'):
                            agent.robustness_fitness = agent.get_fitness() or 0.0
            else:
                # No anti-corruption agents or no evaluator — set compression_fitness = fitness
                for agent in population_to_evaluate:
                    fit = agent.get_fitness()
                    if fit is not None and hasattr(agent, 'compression_fitness'):
                        agent.compression_fitness = fit if fit > -999 else 0.0
        except Exception as e_type_eval:
            self._send_to_gui(f"Gen {generation_num}: Type-aware eval error (non-fatal): {e_type_eval}", "warning")

        # --- v0.9.7: Type-aware heritage trick recording ---
        # After evaluation, record the agent's novel method as a heritage
        # trick if it performed well.  This ensures children and grandchildren
        # can inherit proven pipelines via the "grandpapi" lineage memory.
        #
        # KEY: anti_corruption agents only record tricks based on their
        # ROBUSTNESS fitness (i.e. pipelines that survived corruption), not
        # their clean compression fitness.  This ensures grandpapi memory
        # for anti-corruption lineages contains only corruption-resistant
        # tricks.
        _HERITAGE_FITNESS_THRESHOLD = 2.0  # Minimum fitness to record a trick
        for agent in population_to_evaluate:
            agent_type = getattr(agent, 'agent_type', 'compression')
            # Use the correct fitness for this agent's specialization
            if agent_type == 'anti_corruption':
                trick_fitness = getattr(agent, 'robustness_fitness', 0.0) or 0.0
            else:
                trick_fitness = agent.get_fitness() or 0.0
            if trick_fitness <= _HERITAGE_FITNESS_THRESHOLD:
                continue
            ai_core = getattr(agent, 'puffin_ai', None)
            if ai_core is None:
                continue
            # Record the agent's current novel method into heritage if it exists
            novel = getattr(ai_core, 'novel_method', None)
            if novel is not None and hasattr(novel, 'metadata') and novel.metadata:
                md = novel.metadata
                trick_label = md.get('pipeline', 'unknown')
                if md.get('discovery_seed') is not None:
                    trick_label += f"+seed_{md['discovery_seed']}"
                # Only record if this trick isn't already in heritage
                existing_tricks = {
                    (e.get("ancestor_id", ""), e.get("trick", ""))
                    for e in getattr(agent, 'heritage', [])
                }
                if (agent.agent_id, trick_label) not in existing_tricks:
                    agent.record_trick(
                        trick_label=trick_label,
                        pipeline=md.get('pipeline', 'rle_only'),
                        discovery_seed=md.get('discovery_seed'),
                        rle_min_run=md.get('rle_min_run', 3),
                        fitness=trick_fitness,
                        generation=generation_num,
                    )

        # Update valid_scores after novelty adjustments
        # --- v0.9.7: TYPE-AWARE SCORING & SORTING ---
        # best_fit uses COMPRESSION fitness only (the real optimization
        # objective); robustness is a training metric, not the target.
        # Each sub-population is sorted by its own specialised fitness so
        # per-type elitism in the breeding loop picks the right champions.
        compression_scores = []
        robustness_scores = []
        for agent in population_to_evaluate:
            atype = getattr(agent, 'agent_type', 'compression')
            if atype == 'anti_corruption':
                rfit = getattr(agent, 'robustness_fitness', None)
                if rfit is not None and rfit > -999:
                    robustness_scores.append(rfit)
            else:
                cfit = agent.get_fitness()
                if cfit is not None and cfit > -999:
                    compression_scores.append(cfit)

        # Sort: within each type, rank by the type-appropriate fitness.
        # The combined sort puts compression agents first (group 1, desc)
        # then anti-corruption agents (group 0, desc).  With reverse=True,
        # group 1 > group 0, so compression agents occupy the top of the
        # list — population[0] is always the best compression agent.
        def _type_aware_sort_key(agent):
            atype = getattr(agent, 'agent_type', 'compression')
            if atype == 'anti_corruption':
                rfit = getattr(agent, 'robustness_fitness', None)
                return (0, rfit if rfit is not None else -9999)
            else:
                cfit = agent.get_fitness()
                return (1, cfit if cfit is not None else -9999)

        population_to_evaluate.sort(key=_type_aware_sort_key, reverse=True)

        # Store best robustness for metrics / logging (surfaced in start_evolution)
        self._last_gen_best_robustness = max(robustness_scores) if robustness_scores else 0.0

        # Return the best COMPRESSION fitness — this drives stagnation
        # detection, benchmark sizing, and the dashboard metric.
        return max(compression_scores) if compression_scores else 0.0

    # ------------------------------------------------------------------
    # Benchmark pre-generation helpers
    # ------------------------------------------------------------------

    def _get_github_benchmark_items(self, count: int = 20) -> list[str]:
        """Lazily fetch GitHub benchmark items for phased training.

        On the first call this creates the GitHubFileFetcher instance and
        triggers a download.  On subsequent calls it returns cached items,
        refreshing when the cache runs low.

        Returns:
            List of file-content strings from trusted GitHub repos.
            Empty list if fetching is unavailable (no internet, no token,
            package missing, etc.).
        """
        # Lazy init
        if self._github_fetcher is None:
            if GitHubFileFetcher is None:
                return []
            try:
                self._github_fetcher = GitHubFileFetcher(logger_instance=self.logger)
                self._send_to_gui("GitHub file fetcher initialised for phased training.")
            except Exception as e:
                self._send_to_gui(f"GitHub fetcher init failed (non-fatal): {e}", "warning")
                return []

        # Refresh cache if needed
        if len(self._github_items_cache) < count:
            try:
                self._github_items_cache = self._github_fetcher.get_benchmark_items(
                    count=max(count * 2, 40),  # fetch extra for future gens
                    auto_fetch=True,
                )
                self._send_to_gui(
                    f"GitHub cache refreshed: {len(self._github_items_cache)} items available "
                    f"({self._github_fetcher.get_cached_count()} total in disk cache)"
                )
            except Exception as e:
                self._send_to_gui(f"GitHub fetch error (non-fatal): {e}", "warning")

        # Return a random sample (or all if fewer than count)
        if not self._github_items_cache:
            self._send_to_gui(
                "GitHub item cache is empty after refresh attempt — "
                "phased training will use corrupted data only this generation.",
                "warning",
            )
            return []
        if len(self._github_items_cache) <= count:
            return list(self._github_items_cache)
        return random.sample(self._github_items_cache, count)

    def _should_refresh_at_gen(self, gen_display: int) -> bool:
        """Return True if benchmark data should be refreshed at *gen_display*.

        Centralises the adaptive-refresh schedule so the same logic drives
        both the synchronous fallback and the background pre-generation.
        """
        if not self.dynamic_benchmarking_active or not self.benchmark_evaluator:
            return False
        if gen_display >= 3 and gen_display % 3 == 0:
            return True
        if gen_display in (5, 10, 15, 20, 25):
            return True
        return False

    def _start_prefetch_benchmarks(self, sizing_fitness: float, target_generation: int,
                                    best_compression_ratio: float = 0.0,
                                    gold_standard_win_rate: float = -1.0):
        """Kick off benchmark data generation in a background thread.

        The worker produces a list of benchmark-item strings and stores
        them in ``self._prefetch_benchmark_items``.  The next iteration of
        the evolution loop picks them up via ``_collect_prefetched_benchmarks``.

        Returns a dict with ``items``, ``complexity_tier``, ``tier_index`` so
        the caller can synchronize the live evaluator's state after swap-in.
        Also inherits the live evaluator's previous avg item size so the
        bidirectional growth rate limiter works correctly.
        """
        # If a previous prefetch is still running, don't stack another one
        if self._prefetch_benchmark_future is not None:
            if not self._prefetch_benchmark_future.done():
                return  # still running — let it finish

        self._prefetch_benchmark_items = None  # clear stale data

        # Capture the live evaluator's current avg item size so the growth
        # rate limiter in compute_continuous_benchmark_size works correctly.
        # Without this the tmp evaluator has no items and prev_avg_size = 0,
        # which bypasses the 2x growth cap entirely.
        live_items = []
        live_complexity = None
        live_tier_idx = -1
        live_complexity_dwell = 0
        live_size_dwell = 0
        if self.benchmark_evaluator:
            live_items = list(self.benchmark_evaluator.benchmark_items or [])
            live_complexity = self.benchmark_evaluator._current_complexity_tier
            live_complexity_pct = getattr(self.benchmark_evaluator, '_complexity_pct', 0)
            live_tier_idx = self.benchmark_evaluator._previous_tier_index
            live_complexity_dwell = getattr(self.benchmark_evaluator, '_refreshes_at_current_tier', 0)
            live_size_dwell = getattr(self.benchmark_evaluator, '_previous_size_tier_refreshes', 0)

        def _worker():
            """Generate benchmark items in background thread."""
            try:
                # Create a *temporary* evaluator so we don't touch the
                # live evaluator's state from another thread.
                tmp_evaluator = BenchmarkItemEvaluator(
                    logger_instance=self.logger,
                    dynamic_benchmarking=True,
                )
                # Inherit the current complexity tier and size tier from the
                # live evaluator so the prefetch generates data at the
                # correct difficulty and applies the growth rate limiter.
                tmp_evaluator._current_complexity_tier = live_complexity
                tmp_evaluator._complexity_pct = live_complexity_pct
                tmp_evaluator._previous_tier_index = live_tier_idx
                # Inherit dwell counters so advancement rate-limiting works.
                tmp_evaluator._refreshes_at_current_tier = live_complexity_dwell
                tmp_evaluator._previous_size_tier_refreshes = live_size_dwell
                # Give the tmp evaluator the live items so prev_avg_size > 0
                # and the bidirectional growth limiter is not bypassed.
                tmp_evaluator.benchmark_items = live_items
                tmp_evaluator.generate_and_set_dynamic_benchmark_items(
                    population_average_fitness=sizing_fitness,
                    current_generation=target_generation,
                    best_compression_ratio=best_compression_ratio,
                    gold_standard_win_rate=gold_standard_win_rate,
                )
                return {
                    'items': list(tmp_evaluator.benchmark_items),
                    'complexity_tier': tmp_evaluator._current_complexity_tier,
                    'complexity_pct': tmp_evaluator._complexity_pct,
                    'tier_index': tmp_evaluator._previous_tier_index,
                    'complexity_dwell': tmp_evaluator._refreshes_at_current_tier,
                    'size_dwell': tmp_evaluator._previous_size_tier_refreshes,
                }
            except Exception as e:
                self.logger.warning(f"Background benchmark prefetch failed: {e}")
                return None

        self._prefetch_benchmark_future = self._prefetch_benchmark_executor.submit(_worker)
        self._send_to_gui(
            f"Gen {target_generation}: Benchmark pre-generation started in background")

    def _collect_prefetched_benchmarks(self):
        """Return pre-generated benchmark result if ready, else ``None``.

        Non-blocking: if the worker hasn't finished yet we return None and
        the caller will fall back to synchronous generation.

        Returns a dict ``{items, complexity_tier, tier_index}`` or None.
        """
        if self._prefetch_benchmark_future is None:
            return None
        if not self._prefetch_benchmark_future.done():
            # Not ready yet — let the synchronous path handle it
            return None
        try:
            result = self._prefetch_benchmark_future.result(timeout=0)
            self._prefetch_benchmark_future = None
            if result is None:
                return None
            # Support both legacy (bare list) and new (dict) return formats
            if isinstance(result, dict):
                items = result.get('items', [])
                if items and len(items) > 0:
                    self._prefetch_benchmark_items = None  # consumed
                    return result
            elif isinstance(result, list) and len(result) > 0:
                # Legacy path — wrap in dict
                self._prefetch_benchmark_items = None
                return {'items': result}
        except Exception as e:
            self.logger.warning(f"Prefetched benchmark collection failed: {e}")
        self._prefetch_benchmark_future = None
        return None

    def continue_evolution(self, additional_gens: int = 100, switch_infinite: bool = False):
        """Resume evolution from the last generation without recreating the population.

        Parameters
        ----------
        additional_gens : int
            Number of extra generations to run on top of the current elapsed count.
        switch_infinite : bool
            If *True*, enable infinite mode (ignores *additional_gens* limit).
        """
        if not self.population:
            self._send_to_gui("No population to continue — use start_evolution() first.")
            return
        if switch_infinite:
            self.infinite_mode = True
        else:
            self.initial_num_generations = self.total_generations_elapsed + additional_gens
            self.infinite_mode = False
        if self.gui_stop_event:
            self.gui_stop_event.clear()
        self._send_to_gui(
            f"Continuing evolution from gen {self.total_generations_elapsed} "
            f"({'INFINITE' if self.infinite_mode else f'target {self.initial_num_generations}'})..."
        )
        self.start_evolution(_continue=True)

    def start_evolution(self, _continue: bool = False):
        if not _continue:
            self._send_to_gui(f"Starting Evolution Engine (Target: {self.target_device})...")
            
            self.population = self._create_initial_population()
            if not self.population: 
                self._send_to_gui("Population generation failed. Aborting.")
                return
        else:
            self._send_to_gui(f"Resuming Evolution Engine from gen {self.total_generations_elapsed}...")

        # Track previous fitness to detect improvement for adaptive refresh
        _prev_best_fitness = 0.0 if not _continue else (self.best_fitness_overall or 0.0)
        _fitness_improved_recently = False
        
        gen = self.total_generations_elapsed if _continue else 0
        while True:
            if self.gui_stop_event.is_set(): break
            
            # Check generation limit (unless infinite mode)
            if not self.infinite_mode and gen >= self.initial_num_generations:
                break
            
            current_gen_display = gen + 1
            
            # --- ADAPTIVE BENCHMARK REFRESH (with background pre-generation) ---
            # If the previous iteration pre-generated data for THIS generation,
            # swap it in instantly.  Otherwise fall back to synchronous generation.
            should_refresh = self._should_refresh_at_gen(current_gen_display)
                    
            if should_refresh and self.benchmark_evaluator:
                prefetch_result = self._collect_prefetched_benchmarks()
                if prefetch_result is not None:
                    # Pre-generated data is ready — zero-wait swap
                    prefetched_items = prefetch_result['items']
                    self.benchmark_evaluator.benchmark_items = prefetched_items
                    # Synchronize complexity tier and size tier from the
                    # tmp evaluator so the live evaluator's state stays
                    # consistent.  Without this the displayed complexity
                    # lags behind and the next sync refresh double-advances.
                    if 'complexity_tier' in prefetch_result and prefetch_result['complexity_tier'] is not None:
                        self.benchmark_evaluator._current_complexity_tier = prefetch_result['complexity_tier']
                    if 'complexity_pct' in prefetch_result:
                        self.benchmark_evaluator._complexity_pct = prefetch_result['complexity_pct']
                    if 'tier_index' in prefetch_result:
                        self.benchmark_evaluator._previous_tier_index = prefetch_result['tier_index']
                    # Sync dwell counters so rate-limiting survives the swap.
                    if 'complexity_dwell' in prefetch_result:
                        self.benchmark_evaluator._refreshes_at_current_tier = prefetch_result['complexity_dwell']
                    if 'size_dwell' in prefetch_result:
                        self.benchmark_evaluator._previous_size_tier_refreshes = prefetch_result['size_dwell']
                    # --- EMA DECAY instead of hard reset ---
                    # Carry forward a fraction of the previous fitness so the
                    # sizing function doesn't think the AI went from great to
                    # zero overnight.  This dampens the oscillation cycle.
                    self._sizing_fitness_ema = (
                        self._current_benchmark_best_fitness * self._sizing_fitness_ema_decay)
                    self._current_benchmark_best_fitness = self._sizing_fitness_ema
                    # Reset period-sensitive trackers — old scores were measured on
                    # different benchmark data and are not comparable.
                    self.best_fitness_overall = 0.0
                    self._last_best_fitness = 0.0
                    self._stagnation_counter = 0
                    self._enforce_gpu_safe_benchmark_size()
                    try:
                        bsize_new = self.benchmark_evaluator.get_total_benchmark_size_bytes()
                        num_items = len(prefetched_items)
                        avg_kb = (bsize_new / num_items / 1024) if num_items > 0 else 0
                        self._send_to_gui(
                            f"Gen {current_gen_display}: Pre-generated benchmarks ready — "
                            f"{num_items} items, avg {avg_kb:.0f}KB each, "
                            f"total {bsize_new / (1024*1024):.2f}MB")
                    except Exception:
                        pass
                else:
                    # No pre-generated data — synchronous fallback
                    try:
                        sizing_fitness = self._current_benchmark_best_fitness
                        self._send_to_gui(f"Gen {current_gen_display}: Refreshing benchmark data (sizing_fitness={sizing_fitness:.4f}, all-time={self.best_fitness_overall:.4f})...")
                        self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(
                            population_average_fitness=sizing_fitness,
                            current_generation=current_gen_display,
                            best_compression_ratio=self._last_best_compression_ratio,
                            gold_standard_win_rate=self._last_gold_standard_win_rate,
                        )
                        # --- EMA DECAY instead of hard reset ---
                        self._sizing_fitness_ema = (
                            self._current_benchmark_best_fitness * self._sizing_fitness_ema_decay)
                        self._current_benchmark_best_fitness = self._sizing_fitness_ema
                        # Reset period-sensitive trackers — old scores were measured on
                        # different benchmark data and are not comparable.
                        self.best_fitness_overall = 0.0
                        self._last_best_fitness = 0.0
                        self._stagnation_counter = 0
                        self._enforce_gpu_safe_benchmark_size()
                        try:
                            bsize_new = self.benchmark_evaluator.get_total_benchmark_size_bytes()
                            num_items = len(self.benchmark_evaluator.benchmark_items) if self.benchmark_evaluator.benchmark_items else 0
                            avg_kb = (bsize_new / num_items / 1024) if num_items > 0 else 0
                            self._send_to_gui(f"Gen {current_gen_display}: New benchmarks — {num_items} items, avg {avg_kb:.0f}KB each, total {bsize_new / (1024*1024):.2f}MB")
                        except Exception:
                            pass
                    except Exception as e:
                        self._send_to_gui(f"Failed to refresh benchmarks: {e}", "warning")

            # 1. Evaluate
            best_fit = self._evaluate_population(self.population, current_gen_display)
            
            # 2. Stagnation detection — MUST be BEFORE _prev_best_fitness update.
            #    Compare against previous gen's fitness (gen-to-gen), not the all-time
            #    best which is inflated by early gens with tiny/easy benchmarks.
            _STAGNATION_IMPROVEMENT_THRESHOLD = 0.05
            if (best_fit > 0
                    and _prev_best_fitness > 0
                    and abs(best_fit - _prev_best_fitness) < _STAGNATION_IMPROVEMENT_THRESHOLD):
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
            
            # Detect fitness improvement for next gen's adaptive refresh
            _fitness_improved_recently = (best_fit > _prev_best_fitness + 0.1) and best_fit > 0
            _prev_best_fitness = best_fit
            
            self.best_fitness_overall = max(self.best_fitness_overall, best_fit)
            self._current_benchmark_best_fitness = max(self._current_benchmark_best_fitness, best_fit)
            self.total_generations_elapsed = current_gen_display
            
            # Track the best agent
            if self.population:
                top_agent = self.population[0]
                if top_agent.get_fitness() is not None and top_agent.get_fitness() >= self.best_fitness_overall:
                    self.best_agent_overall = top_agent
            
            # Guard against 0.0 drops: carry over previous best if evaluation failed
            if best_fit <= 0.0 and self._last_best_fitness > 0.0:
                best_fit = self._last_best_fitness
                self._send_to_gui(f"Gen {current_gen_display}: Eval returned 0 — carried over previous best {best_fit:.4f}")
            self._last_best_fitness = best_fit

            # 2. Metrics
            bsize = 0.0
            if self.benchmark_evaluator:
                try: bsize = self.benchmark_evaluator.get_total_benchmark_size_bytes()
                except: pass
            
            # Compute real compression ratio from best agent's eval stats
            ratio = 0.0
            if self.population:
                top = self.population[0]
                es = getattr(top, 'evaluation_stats', None)
                if es and isinstance(es, dict):
                    # Primary: total bytes ratio across ALL compression attempts
                    total_orig = es.get('total_original_bytes', 0)
                    total_comp = es.get('total_compressed_bytes', 0)
                    if total_orig > 0 and total_comp > 0:
                        ratio = max(0.0, (1.0 - total_comp / total_orig) * 100)
                    else:
                        # Fallback: per-success ratio (legacy)
                        srle = es.get('successful_rle', 0)
                        sum_ratios = es.get('sum_compression_ratios_rle_success', 0.0)
                        if srle > 0 and sum_ratios > 0:
                            avg_ratio = sum_ratios / srle
                            ratio = max(0.0, (1.0 - 1.0 / avg_ratio) * 100)
            self._send_metrics_json(current_gen_display, best_fit, ratio, bsize)
            self._last_best_compression_ratio = ratio
            gens_display = "∞" if self.infinite_mode else str(self.initial_num_generations)
            robustness_str = f" | Robustness: {self._last_gen_best_robustness:.4f}" if self._last_gen_best_robustness > 0 else ""
            self._send_to_gui(f"Gen {current_gen_display}/{gens_display} Complete. Best Fitness: {best_fit:.4f}{robustness_str} | Benchmark: {bsize / (1024*1024):.2f}MB")

            # 2b. Snapshot population for GDV history (before breeding replaces it)
            try:
                self._snapshot_generation(current_gen_display, reported_best_fitness=best_fit)
            except Exception as e_snap:
                self.logger.warning(f"Generation snapshot failed (non-fatal): {e_snap}")

            # 2b-ii. GOLD STANDARD HEAD-TO-HEAD BENCHMARK
            #        Pit the best agent against gzip/bz2/lzma/zlib/zstd.
            #        Saves a gold-standard checkpoint if the agent beats ALL of them.
            #        On failure, writes compressed + decompressed artefacts for diagnosis.
            _gs_advanced_this_gen = False  # Track if gold standard triggered advancement
            if self.gold_standard_benchmark and self.population:
                try:
                    top_agent_for_h2h = self.population[0]
                    h2h_items = []
                    if self.benchmark_evaluator and self.benchmark_evaluator.benchmark_items:
                        h2h_items = list(self.benchmark_evaluator.benchmark_items)
                    if top_agent_for_h2h and h2h_items:
                        gs_report = self.gold_standard_benchmark.benchmark_generation(
                            generation=current_gen_display,
                            best_agent=top_agent_for_h2h,
                            test_items=h2h_items,
                            checkpoint_save_fn=self.save_checkpoint,
                            gui_msg_fn=self._send_to_gui,
                        )
                        # Store gold standard win rate for advancement gating.
                        # win_rate = fraction of items where AI beat ALL baselines.
                        if gs_report and gs_report.num_items > 0:
                            wins = sum(1 for it in gs_report.items if it.ai_beats_all)
                            self._last_gold_standard_win_rate = wins / gs_report.num_items
                        elif gs_report:
                            self._last_gold_standard_win_rate = 0.0

                        # --- IMMEDIATE ADVANCEMENT ON GOLD STANDARD SUCCESS ---
                        # If the AI beat ALL baselines on ALL items, immediately
                        # raise difficulty by triggering a synchronous benchmark
                        # refresh.  This ensures the AI keeps learning even at
                        # gen 500+ with low-complexity data — it never stalls
                        # waiting for the 3-gen refresh schedule.
                        if (gs_report and getattr(gs_report, 'gold_standard', False)
                                and self.benchmark_evaluator
                                and self.dynamic_benchmarking_active):
                            self._send_to_gui(
                                f"Gen {current_gen_display}: ★ GOLD STANDARD ACHIEVED — "
                                f"immediate difficulty advancement triggered!",
                                "info",
                            )
                            try:
                                self.benchmark_evaluator.generate_and_set_dynamic_benchmark_items(
                                    population_average_fitness=self._current_benchmark_best_fitness,
                                    current_generation=current_gen_display,
                                    best_compression_ratio=self._last_best_compression_ratio,
                                    gold_standard_win_rate=self._last_gold_standard_win_rate,
                                )
                                # EMA decay — carry forward fitness fraction
                                self._sizing_fitness_ema = (
                                    self._current_benchmark_best_fitness * self._sizing_fitness_ema_decay)
                                self._current_benchmark_best_fitness = self._sizing_fitness_ema
                                # Reset period-sensitive trackers
                                self.best_fitness_overall = 0.0
                                self._last_best_fitness = 0.0
                                self._stagnation_counter = 0
                                self._enforce_gpu_safe_benchmark_size()
                                try:
                                    bsize_new = self.benchmark_evaluator.get_total_benchmark_size_bytes()
                                    num_items = len(self.benchmark_evaluator.benchmark_items) if self.benchmark_evaluator.benchmark_items else 0
                                    avg_kb = (bsize_new / num_items / 1024) if num_items > 0 else 0
                                    self._send_to_gui(
                                        f"Gen {current_gen_display}: Gold-standard advancement — "
                                        f"{num_items} items, avg {avg_kb:.0f}KB, "
                                        f"total {bsize_new / (1024*1024):.2f}MB")
                                except Exception:
                                    pass
                                _gs_advanced_this_gen = True
                            except Exception as e_gs_adv:
                                self.logger.warning(
                                    f"Gen {current_gen_display}: Gold-standard advancement refresh failed: {e_gs_adv}")
                except Exception as e_h2h:
                    self.logger.warning(f"Gen {current_gen_display}: Gold-standard benchmark failed (non-fatal): {e_h2h}")

            # 2c. Auto-checkpoint at regular intervals (thread-safe, from evolution thread)
            if (self._auto_checkpoint_interval > 0
                    and current_gen_display % self._auto_checkpoint_interval == 0
                    and self.checkpoint_manager):
                try:
                    cp_name = f"auto_gen{current_gen_display}"
                    ok = self.save_checkpoint(cp_name)
                    if ok:
                        self._send_to_gui(f"Gen {current_gen_display}: Auto-checkpoint saved.")
                        # Rotate: delete oldest auto-checkpoints beyond the limit.
                        self._rotate_auto_checkpoints()
                    else:
                        self._send_to_gui(f"Gen {current_gen_display}: Auto-checkpoint failed (see logs).", "warning")
                except Exception as e_cp:
                    self.logger.warning(f"Auto-checkpoint failed (non-fatal): {e_cp}")

            # 3. Breeding (with stagnation-aware hypermutation)
            self._run_breeding_cycle(current_gen_display)
            
            # 4. Periodic GC to prevent memory bloat on long runs
            if current_gen_display % 10 == 0:
                gc.collect()
            
            # 5. BACKGROUND PRE-GENERATION for next generation
            #    If the next gen will need a benchmark refresh, kick off data
            #    generation now in a background thread so it's ready by the time
            #    the next iteration's top-of-loop swap-in runs.
            #    Skip if gold standard already triggered an immediate advancement
            #    this gen — avoid double-advancing in the same cycle.
            next_gen_display = current_gen_display + 1
            if self._should_refresh_at_gen(next_gen_display) and not _gs_advanced_this_gen:
                self._start_prefetch_benchmarks(
                    sizing_fitness=self._current_benchmark_best_fitness,
                    target_generation=next_gen_display,
                    best_compression_ratio=self._last_best_compression_ratio,
                    gold_standard_win_rate=self._last_gold_standard_win_rate,
                )

            # Increment generation counter
            gen += 1
        
        # --- End of evolution loop: save final checkpoint ---
        if self.checkpoint_manager and self.total_generations_elapsed > 0:
            try:
                cp_name = f"final_gen{self.total_generations_elapsed}"
                ok = self.save_checkpoint(cp_name)
                if ok:
                    self._send_to_gui(f"Final checkpoint saved (gen {self.total_generations_elapsed}).")
            except Exception as e_final:
                self.logger.warning(f"Final checkpoint save failed: {e_final}")

    def _run_breeding_cycle(self, gen):
        """Breed the next generation with lineage-aware novel method inheritance,
        gradual evolution phases, and agent type specialization.

        Key design decisions (v0.9.7):

        1. **"Grandpapi" lineage memory**
           Each child merges heritage dicts from both parents.  Because the
           parent's heritage already contains *their* ancestors' entries, the
           child transitively inherits tricks from grandparents, great-
           grandparents, etc. without an external lineage DB.

        2. **Lineage-aware novel method selection**
           When deciding which novel pipeline to give a child, we check the
           merged heritage for the highest-fitness trick and rebuild that
           pipeline with high probability (``HERITAGE_REUSE_PROB``).  This
           ensures proven family methods are passed down, not discarded.

        3. **Gradual evolution phases**
           Gen 0-20:  Heavily favour known/scaffolded methods (low novel %).
           Gen 21-50: Transition — increasing novel method creation.
           Gen 51+:   Full novelty — novel methods dominate breeding.

        4. **Agent type split + cross-type breeding**
           Children inherit the type of their dominant parent (higher fitness),
           but cross-type breeds (compression × anti_corruption) are allowed
           and encouraged (they inherit traits from both specializations).

        5. **Both children from apply_crossover are used**
           ``apply_crossover`` returns two children (child1_ai, child2_ai).
           Both are wrapped as EvolvingAgents and added to next_pop (space
           permitting).  The first child gets the dominant parent's type
           preference, the second child gets the recessive parent's type.

        6. **Strict 50/50 type balance via budget**
           Instead of a weak nudge after the fact, the breeding loop
           pre-computes comp_budget and anti_budget (accounting for elites)
           and assigns types based on remaining budget.  This guarantees
           the population stays at exactly 50/50 ± 1 every generation.

        7. **Per-type elitism**
           Top-1 agent from EACH type is kept as elite (2 total) so
           neither sub-population is wiped in a single generation.
        """
        STAGNATION_THRESHOLD = 5  # gens without improvement → trigger hypermutation
        HYPERMUTATION_FRACTION = 0.2  # fraction of non-elite children that get hypermutation
        use_hypermutation = self._stagnation_counter >= STAGNATION_THRESHOLD

        if use_hypermutation:
            self._send_to_gui(f"Gen {gen}: Stagnation detected ({self._stagnation_counter} gens) — applying hypermutation to {HYPERMUTATION_FRACTION*100:.0f}% of offspring.")

        # --- Gradual evolution: compute novel method probability for this gen ---
        # Gen 0-20:  5-15% novel chance  (scaffold/known methods dominate)
        # Gen 21-50: 15-60% novel chance (transition phase)
        # Gen 51+:   60-80% novel chance (full novelty)
        if gen <= 20:
            novel_method_prob = 0.05 + (gen / 20.0) * 0.10   # 5% → 15%
        elif gen <= 50:
            novel_method_prob = 0.15 + ((gen - 20) / 30.0) * 0.45  # 15% → 60%
        else:
            novel_method_prob = min(0.80, 0.60 + (gen - 50) * 0.002)  # 60% → 80% cap

        # Probability of inheriting a proven trick from heritage vs generating fresh
        HERITAGE_REUSE_PROB = 0.65

        next_pop = []
        # --- Elitism: keep top 2 from EACH type so neither type is wiped ---
        if self.population:
            comp_elites = [a for a in self.population if getattr(a, 'agent_type', 'compression') == 'compression']
            anti_elites = [a for a in self.population if getattr(a, 'agent_type', 'compression') == 'anti_corruption']
            for pool_label, pool in [("comp", comp_elites), ("anti", anti_elites)]:
                for i in range(min(len(pool), 1)):   # 1 elite per type = 2 total
                    if pool[i].get_fitness() is not None and pool[i].get_fitness() > -999:
                        elite = pool[i].clone(
                            new_agent_id=f"gen{gen}_elite_{pool_label}_{i}",
                            new_generation_born=gen,
                        )
                        self._sanitize_agent(elite)
                        next_pop.append(elite)

        # --- Pre-compute target counts for strict 50/50 balance ---
        half = self.population_size // 2
        # Count types already placed by elitism
        _comp_placed  = sum(1 for a in next_pop if getattr(a, 'agent_type', 'compression') == 'compression')
        _anti_placed  = sum(1 for a in next_pop if getattr(a, 'agent_type', 'compression') == 'anti_corruption')
        comp_budget   = half - _comp_placed
        anti_budget   = (self.population_size - half) - _anti_placed

        child_index = 0
        while len(next_pop) < self.population_size:
            if self.gui_stop_event.is_set(): break
            
            parents = tournament_selection(self.population, 2)
            if not parents or len(parents) < 2:
                break

            parent1, parent2 = parents[0], parents[1]

            # --- Merge heritage from BOTH parents ("grandpapi" inheritance) ---
            merged_heritage = EvolvingAgent.merge_heritage(
                getattr(parent1, 'heritage', []),
                getattr(parent2, 'heritage', []),
            )

            p1_fit = parent1.get_fitness() or 0.0
            p2_fit = parent2.get_fitness() or 0.0
            dominant_parent = parent1 if p1_fit >= p2_fit else parent2
            recessive_parent = parent2 if dominant_parent is parent1 else parent1

            # --- CROSSOVER: produce TWO children from each parent pair ---
            try:
                child1_ai, child2_ai = apply_crossover(
                    parent1.puffin_ai, parent2.puffin_ai,
                    p1_fit, p2_fit,
                    self.logger, {}
                )
            except Exception:
                child1_ai = None
                child2_ai = None

            # --- Build up to 2 children from this crossover ---
            child_cores = []
            if child1_ai is not None:
                child_cores.append(child1_ai)
            if child2_ai is not None:
                child_cores.append(child2_ai)
            if not child_cores:
                # Fallback: clone the dominant parent if crossover failed completely
                child_cores.append(dominant_parent.puffin_ai.clone_core_model())

            for core_idx, child_ai in enumerate(child_cores):
                if len(next_pop) >= self.population_size:
                    break

                # --- Agent type: strict budget-based assignment ---
                # The dominant parent's type is preferred, but the budget
                # forces the population to stay at 50/50.
                preferred_type = getattr(dominant_parent, 'agent_type', 'compression')
                if core_idx == 1:
                    # Second child gets the recessive parent's type preference
                    preferred_type = getattr(recessive_parent, 'agent_type', 'compression')

                if preferred_type == 'compression' and comp_budget > 0:
                    child_type = 'compression'
                    comp_budget -= 1
                elif preferred_type == 'anti_corruption' and anti_budget > 0:
                    child_type = 'anti_corruption'
                    anti_budget -= 1
                elif comp_budget > 0:
                    child_type = 'compression'
                    comp_budget -= 1
                elif anti_budget > 0:
                    child_type = 'anti_corruption'
                    anti_budget -= 1
                else:
                    child_type = preferred_type  # safety fallback

                child = EvolvingAgent(
                    child_ai,
                    agent_id=f"gen{gen}_child_{child_index}",
                    generation_born=gen,
                    parent_ids=[parent1.agent_id, parent2.agent_id],
                    agent_type=child_type,
                    heritage=list(merged_heritage),
                )

                # --- MUTATION ---
                if use_hypermutation and random.random() < HYPERMUTATION_FRACTION:
                    apply_hypermutation(child)
                else:
                    apply_mutations(child, {'base_rate': self.base_mutation_rate})

                # Q-Table noise
                if hasattr(child, 'puffin_ai') and hasattr(child.puffin_ai, 'q_table'):
                    noise_rate = 0.2 if use_hypermutation else 0.1
                    noise_std = 0.1 if use_hypermutation else 0.05
                    mutation_mask = np.random.random(child.puffin_ai.q_table.shape) < noise_rate
                    noise = np.random.normal(0, noise_std, child.puffin_ai.q_table.shape)
                    child.puffin_ai.q_table[mutation_mask] += noise[mutation_mask]

                # --- Novel method: lineage-aware inheritance + gradual evolution ---
                if random.random() < novel_method_prob:
                    novel_assigned = False

                    # Try to inherit a proven trick from heritage ("grandpapi" reuse)
                    if merged_heritage and random.random() < HERITAGE_REUSE_PROB:
                        best_trick = max(
                            merged_heritage,
                            key=lambda e: e.get("fitness_when_learned", 0.0),
                        )
                        try:
                            from ..novel_compression_generator import NovelCompressionGenerator
                            gen_instance = NovelCompressionGenerator()
                            pipeline_name = best_trick.get("pipeline", "rle_only")
                            cfn, dfn = gen_instance._build_pipeline(
                                pipeline_name,
                                discovery_seed=best_trick.get("discovery_seed"),
                                rle_min_run=best_trick.get("rle_min_run", 3),
                            )
                            from ..compression_method_registry import CompressionMethod, CompressionLanguage
                            method = CompressionMethod(
                                name=f"gen{gen}_heritage_{child_index}",
                                language=CompressionLanguage.HYBRID,
                                compress_fn=cfn,
                                decompress_fn=dfn,
                                description=f"Inherited from ancestor {best_trick.get('ancestor_id', '?')} (gen {best_trick.get('generation', '?')})",
                                author="heritage_inheritance",
                                is_novelty=True,
                                metadata={
                                    "pipeline": pipeline_name,
                                    "discovery_seed": best_trick.get("discovery_seed"),
                                    "rle_min_run": best_trick.get("rle_min_run", 3),
                                    "steps": [],
                                    "inherited_from": best_trick.get("ancestor_id", ""),
                                },
                            )
                            child.puffin_ai.novel_method = method
                            child.puffin_ai._novel_compress_fn = cfn
                            child.puffin_ai._novel_decompress_fn = dfn
                            novel_assigned = True
                        except Exception:
                            pass  # Fall through to fresh generation

                    # If heritage reuse didn't happen, generate a fresh novel method
                    if not novel_assigned and get_novel_generator:
                        try:
                            novel_gen = get_novel_generator()
                            method = novel_gen.generate_novelty_method(
                                method_name=f"gen{gen}_novel_{child_index}",
                                pattern_combo_size=random.randint(1, 3)
                            )
                            child.puffin_ai.novel_method = method
                            child.puffin_ai._novel_compress_fn = method.compress_fn
                            child.puffin_ai._novel_decompress_fn = method.decompress_fn
                        except Exception:
                            pass
                else:
                    # Gradual evolution: early gens keep parent's novel method
                    if hasattr(dominant_parent.puffin_ai, '_novel_compress_fn') and dominant_parent.puffin_ai._novel_compress_fn:
                        child.puffin_ai.novel_method = dominant_parent.puffin_ai.novel_method
                        child.puffin_ai._novel_compress_fn = dominant_parent.puffin_ai._novel_compress_fn
                        child.puffin_ai._novel_decompress_fn = dominant_parent.puffin_ai._novel_decompress_fn

                self._sanitize_agent(child)
                next_pop.append(child)
                child_index += 1
        
        self.population = next_pop

    def save_checkpoint(self, name):
        """Save a checkpoint with current evolution state and metrics.
        
        Thread-safe: uses _checkpoint_lock so the WebUI and evolution thread
        can both call this without corrupting the pickle.
        Returns True on success.
        """
        if not self.checkpoint_manager:
            return False
        with self._checkpoint_lock:
            # Gather current metrics for metadata
            best_fit = self.best_fitness_overall if hasattr(self, 'best_fitness_overall') else 0.0
            avg_fit = 0.0
            if hasattr(self, 'population') and self.population:
                fitnesses = [getattr(ind, 'fitness', 0.0) or 0.0 for ind in self.population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
            gen = self.total_generations_elapsed if hasattr(self, 'total_generations_elapsed') else 0
            pop_size = len(self.population) if hasattr(self, 'population') and self.population else 0
            return self.checkpoint_manager.save_checkpoint(
                checkpoint_name=name,
                optimizer_state={
                    'population': self.population,
                    'total_generations_elapsed': self.total_generations_elapsed,
                    'best_fitness_overall': self.best_fitness_overall
                },
                best_fitness=best_fit,
                avg_fitness=avg_fit,
                generation=gen,
                population_size=pop_size,
            )

    def _rotate_auto_checkpoints(self):
        """Delete oldest auto-checkpoints beyond ``_max_auto_checkpoints``.

        Only touches checkpoints whose name starts with ``auto_gen`` —
        user-named and gold-standard checkpoints are never deleted.

        Thread-safe: acquires ``_checkpoint_lock`` so rotation cannot race
        with a concurrent save triggered by the WebUI or another thread.
        """
        if not self.checkpoint_manager:
            return
        with self._checkpoint_lock:
            try:
                all_cps = self.checkpoint_manager.list_checkpoints()
                auto_cps = [cp for cp in all_cps
                            if isinstance(cp, dict) and str(cp.get('name', '')).startswith('auto_gen')]
                if len(auto_cps) <= self._max_auto_checkpoints:
                    return
                # Sort by generation number (ascending) so oldest are first.
                def _gen_num(cp):
                    name = str(cp.get('name', ''))
                    import re as _re
                    m = _re.search(r'auto_gen(\d+)', name)
                    return int(m.group(1)) if m else 0
                auto_cps.sort(key=_gen_num)
                to_delete = auto_cps[:len(auto_cps) - self._max_auto_checkpoints]
                for cp in to_delete:
                    key = cp.get('key') or cp.get('filename') or cp.get('name', '')
                    if key:
                        try:
                            self.checkpoint_manager.delete_checkpoint(str(key))
                        except Exception:
                            pass
            except Exception as e:
                self.logger.debug(f"Auto-checkpoint rotation failed (non-fatal): {e}")