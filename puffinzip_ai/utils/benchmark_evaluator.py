from __future__ import annotations

import os
import time
import random
import numpy as np
import json
import logging
import traceback
from enum import Enum
import string
from typing import Any

_PuffinZipAI_cls: Any = None
_rle_compress_func: Any = None
_rle_decompress_func: Any = None
_RLE_DECOMPRESSION_ERRORS_set = set()
_calculate_reward_func: Any = None
_GENERATED_BENCHMARK_DEFAULT_PATH_val = "./fallback_benchmark_data_dir"
_BENCHMARK_DATA_DIR_val, _DEFAULT_LEARNING_RATE_val, _DEFAULT_EXPLORATION_RATE_val = None, 0.1, 1.0
_DEFAULT_DISCOUNT_FACTOR_val = _DEFAULT_EXPLORATION_DECAY_RATE_val = _DEFAULT_MIN_EXPLORATION_RATE_val = 0.01
_RLE_MIN_RUN_INIT_MIN_val, _RLE_MIN_RUN_INIT_MAX_val = 6, 64
_setup_logger_func_val = lambda *args, **kwargs: logging.getLogger("BenchmarkEvaluator_Fallback_Setup")
_config_module = None

try:
    from ..ai_core import PuffinZipAI
    _PuffinZipAI_cls = PuffinZipAI
    from ..rle_utils import rle_compress, rle_decompress
    _rle_compress_func, _rle_decompress_func = rle_compress, rle_decompress
    from ..rle_constants import RLE_DECOMPRESSION_ERRORS
    _RLE_DECOMPRESSION_ERRORS_set = RLE_DECOMPRESSION_ERRORS
    from ..reward_system import calculate_reward, calculate_method_diversity_adjustment, calculate_size_scaled_reward
    _calculate_reward_func = calculate_reward
    _calculate_method_diversity_adjustment_func = calculate_method_diversity_adjustment
    _calculate_size_scaled_reward_func = calculate_size_scaled_reward
    
    # IMPORT THE NEW BATCHED GPU INTERFACES SAFELY
    try:
        from ..gpu_core.gpu_rle_interface import gpu_accelerated_rle_compress_batch, gpu_accelerated_rle_decompress_batch, CUPY_AVAILABLE
    except ImportError:
        try:
            from puffinzip_ai.gpu_core.gpu_rle_interface import gpu_accelerated_rle_compress_batch, gpu_accelerated_rle_decompress_batch, CUPY_AVAILABLE
        except ImportError:
            CUPY_AVAILABLE = False
            
    from ..config import (
        GENERATED_BENCHMARK_DEFAULT_PATH,
        BENCHMARK_DATA_DIR, DEFAULT_LEARNING_RATE, DEFAULT_EXPLORATION_RATE,
        DEFAULT_DISCOUNT_FACTOR, DEFAULT_EXPLORATION_DECAY_RATE, DEFAULT_MIN_EXPLORATION_RATE,
        RLE_MIN_RUN_INIT_MIN, RLE_MIN_RUN_INIT_MAX,
        DEBUG_LOG_CONSOLE_OUTPUT_ENABLED
    )

    _GENERATED_BENCHMARK_DEFAULT_PATH_val = GENERATED_BENCHMARK_DEFAULT_PATH
    _BENCHMARK_DATA_DIR_val = BENCHMARK_DATA_DIR
    _DEFAULT_LEARNING_RATE_val = DEFAULT_LEARNING_RATE
    _DEFAULT_EXPLORATION_RATE_val = DEFAULT_EXPLORATION_RATE
    _DEFAULT_DISCOUNT_FACTOR_val = DEFAULT_DISCOUNT_FACTOR
    _DEFAULT_EXPLORATION_DECAY_RATE_val = DEFAULT_EXPLORATION_DECAY_RATE
    from .. import config as _config_module
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
    _calculate_method_diversity_adjustment_func: Any = None
    _calculate_size_scaled_reward_func: Any = None
    CUPY_AVAILABLE = False
    _fallback_logger_be.critical(
        f"CRITICAL ERROR (benchmark_evaluator.py): Failed to import core components. Error: {e_be_imp}", exc_info=True)

DEFAULT_BENCHMARK_REPETITIONS = 1
DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET = 20

def _is_debug():
    return _config_module and getattr(_config_module, 'DEBUG_LOG_CONSOLE_OUTPUT_ENABLED', False)

EVALUATION_FAIL_REWARD = -100.0
EVALUATION_TIMEOUT_REWARD_PENALTY = -50.0
MAX_ITEM_PROCESS_TIME_SEC = 30.0

AGENTS_PER_THROTTLE_CHECK = 50
ITEMS_PER_THROTTLE_CHECK = 500
THROTTLE_SLEEP_DURATION_BENCH_EVAL = 0.0  


_pipeline_ctx: dict = {}  

def _pipeline_worker_init(benchmark_items_list):
    import signal, sys
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    _pipeline_ctx["items"] = benchmark_items_list
    try:
        from ..rle_utils import rle_compress, rle_decompress
        from ..reward_system import calculate_reward, calculate_size_scaled_reward
        from ..rle_constants import RLE_DECOMPRESSION_ERRORS
    except ImportError:
        from puffinzip_ai.rle_utils import rle_compress, rle_decompress
        from puffinzip_ai.reward_system import calculate_reward, calculate_size_scaled_reward
        from puffinzip_ai.rle_constants import RLE_DECOMPRESSION_ERRORS

    _pipeline_ctx["compress"] = rle_compress
    _pipeline_ctx["decompress"] = rle_decompress
    _pipeline_ctx["reward_fn"] = calculate_reward
    _pipeline_ctx["size_reward_fn"] = calculate_size_scaled_reward
    _pipeline_ctx["rle_errors"] = RLE_DECOMPRESSION_ERRORS


def _compress_single_item(args):
    item_idx, action_name, rle_min_run = args
    ctx = _pipeline_ctx
    item_text = ctx["items"][item_idx]
    compress_fn = ctx["compress"]
    decompress_fn = ctx["decompress"]
    reward_fn = ctx["reward_fn"]
    size_reward_fn = ctx["size_reward_fn"]
    rle_errors = ctx["rle_errors"]

    compressed = ""
    decompressed = ""
    try:
        _t0 = time.perf_counter_ns()
        if action_name == "RLE":
            compressed = compress_fn(
                item_text, method="simple", min_run_len_override=rle_min_run
            )
            decompressed = decompress_fn(
                compressed, method="simple", min_run_len_override=rle_min_run
            )
        elif action_name == "AdvancedRLE":
            compressed = compress_fn(item_text, method="advanced")
            decompressed = decompress_fn(compressed, method="advanced")
        elif action_name == "NoCompression":
            compressed = item_text
            decompressed = item_text
        else:
            return {"fallback": True, "item_idx": item_idx}
        t_ns = time.perf_counter_ns() - _t0

        original_size = len(item_text)
        compressed_size = len(compressed) if compressed else 0
        proc_ms = t_ns / 1_000_000
        decompression_ok = decompressed == item_text
        rle_err = (
            decompressed
            if (rle_errors and decompressed in rle_errors)
            else None
        )
        was_success = (
            compressed_size < original_size
            and decompression_ok
            and action_name in ("RLE", "AdvancedRLE")
            and rle_err is None
        )

        reward = reward_fn(
            item_text, compressed, decompressed, action_name, proc_ms, rle_err
        )
        if size_reward_fn and original_size > 0:
            reward = size_reward_fn(
                reward, original_size, compressed_size, was_success
            )

        return {
            "fallback": False,
            "item_idx": item_idx,
            "reward": reward,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "was_success": was_success,
            "decompression_ok": decompression_ok,
            "rle_error": rle_err is not None,
            "action_name": action_name,
            "proc_ms": proc_ms,
        }
    except Exception:
        return {
            "fallback": False,
            "item_idx": item_idx,
            "reward": EVALUATION_FAIL_REWARD,
            "original_size": len(item_text) if item_text else 0,
            "compressed_size": len(item_text) if item_text else 0,
            "was_success": False,
            "decompression_ok": False,
            "rle_error": True,
            "action_name": action_name,
            "proc_ms": 0.0,
        }


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
    DataComplexity.SIMPLE: -3.0,
    DataComplexity.MODERATE: 2.0,
    DataComplexity.COMPLEX: 6.0,
    DataComplexity.VERY_COMPLEX: 12.0
}
COMPLEXITY_RATIO_GATES = {
    DataComplexity.SIMPLE: 25.0,
    DataComplexity.MODERATE: 45.0,
    DataComplexity.COMPLEX: 60.0,
    DataComplexity.VERY_COMPLEX: 70.0
}
COMPLEXITY_GOLD_STANDARD_GATES = {
    DataComplexity.SIMPLE: 0.10,
    DataComplexity.MODERATE: 0.3,
    DataComplexity.COMPLEX: 0.5,
    DataComplexity.VERY_COMPLEX: 0.7,
}
COMPLEXITY_LENGTH_RANGES_BYTES = {
    DataComplexity.VERY_SIMPLE: (4 * 1024, 32 * 1024),
    DataComplexity.SIMPLE: (32 * 1024, 256 * 1024),
    DataComplexity.MODERATE: (256 * 1024, 1 * 1024 * 1024),
    DataComplexity.COMPLEX: (1 * 1024 * 1024, 4 * 1024 * 1024),
    DataComplexity.VERY_COMPLEX: (4 * 1024 * 1024, 10 * 1024 * 1024)
}

CONTINUOUS_SIZE_FLOOR_BYTES = 64 * 1024
CONTINUOUS_SIZE_CEILING_BYTES = 10 * 1024 * 1024
GENERATION_SIZE_TIERS = [
    (0,   5 * 1024 * 1024,     256 * 1024,           -10.0),
    (5,   10 * 1024 * 1024,    512 * 1024,             0.0),
    (10,  20 * 1024 * 1024,    1 * 1024 * 1024,        2.0),
    (15,  40 * 1024 * 1024,    2 * 1024 * 1024,        4.0),
    (20,  70 * 1024 * 1024,    4 * 1024 * 1024,        7.0),
    (25,  120 * 1024 * 1024,   7 * 1024 * 1024,       10.0),
]
SIZE_TIER_RATIO_GATES = [0.0, 20.0, 35.0, 50.0, 60.0, 70.0]
SIZE_TIER_GOLD_STANDARD_GATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6]
TOTAL_BENCHMARK_BUDGET_BYTES = 20 * 1024 * 1024
MAX_SIZE_GROWTH_FACTOR = 2.0
MIN_SIZE_SHRINK_FACTOR = 0.5
TIER_HYSTERESIS_MARGIN = 2.0

_RATIO_GATE_KNOTS = [(0, 0.0), (20, 25.0), (40, 45.0), (60, 60.0), (80, 70.0), (100, 80.0)]
_GS_GATE_KNOTS = [(0, 0.0), (20, 0.10), (40, 0.30), (60, 0.50), (80, 0.70), (100, 0.80)]

# --- ANTI-CORRUPTION PROGRESSION (independent of the compression track) ---
# The compression track (``_complexity_pct``) advances on best_compression_ratio.
# The anti-corruption track (``_corruption_pct``) advances on a *robustness
# recovery rate* (0.0-1.0): the fraction of corrupted/noisy items the best
# anti-corruption agent compresses AND decompresses without error.  The two
# tracks are completely separate, so anti-corruption can keep advancing while
# compression stagnates and vice-versa.
#
# _CORRUPTION_GATE_KNOTS maps a candidate corruption pct -> the robustness
# recovery rate required to reach it (absolute path).  Because higher corruption
# naturally lowers the achievable recovery rate, this forms a self-limiting
# equilibrium: the corruption level only climbs when the AI actually gets more
# robust.  In addition to this absolute gate, the track also advances on
# *relative improvement* (see determine_target_corruption) so it never gets
# permanently stuck just because the absolute gate was mis-calibrated for a
# given data regime — as long as robustness keeps improving, corruption climbs.
_CORRUPTION_GATE_KNOTS = [(0, 0.0), (20, 0.15), (40, 0.25), (60, 0.35), (80, 0.45), (100, 0.55)]
# Minimum recovery rate before ANY advancement (agents must show some skill).
_CORRUPTION_MIN_RATE = 0.02
# Relative-improvement margin: beating the tier-entry recovery rate by this much
# also advances the track, independent of the absolute gate.
_CORRUPTION_REL_MARGIN = 0.05

# --- ROBUSTNESS GOLD-STANDARD GATE (anti-corruption track) ---
# Mirror of _GS_GATE_KNOTS for the compression track.  The compression track
# gates complexity advancement on a head-to-head win rate vs gzip/bz2/lzma/
# zlib/zstd on CLEAN data.  The anti-corruption track gates corruption
# advancement on the equivalent head-to-head win rate run on CORRUPTED data:
# the fraction of corrupted items where the best anti-corruption agent both
# survived the noise (verified round-trip) AND beat every baseline compressor.
# Baselines are brittle on corrupted streams, so this is a meaningful bar that
# proves the anti-corruption lineage is genuinely more resilient, not just
# tuning against its own recovery metric.  Knots are gentler than the
# compression GS gates because head-to-head on noisy data is intrinsically
# harder.  A win rate of -1.0 means "no gold-standard data yet" and is treated
# leniently (does not block) so the track is never stalled before the first
# head-to-head runs.  The first band (0-20%) is intentionally ungated so young
# anti-corruption agents can climb into mild corruption purely on their recovery
# rate; the gold-standard superiority requirement only bites for the harder
# corruption tiers (>20%), where beating brittle baselines is the real proof.
_ROBUSTNESS_GS_GATE_KNOTS = [(0, 0.0), (20, 0.0), (40, 0.15), (60, 0.25), (80, 0.35), (100, 0.45)]

# corruption pct 0-100 -> per-character corruption probability of the input.
CORRUPTION_LEVEL_MIN = 0.02
CORRUPTION_LEVEL_MAX = 0.30
# corruption pct 0-100 -> fraction of injected garbage characters.
CORRUPTION_GARBAGE_MAX = 0.08

def _piecewise_lerp(knots: list, x: float) -> float:
    if x <= knots[0][0]: return knots[0][1]
    for i in range(len(knots) - 1):
        x0, y0 = knots[i]; x1, y1 = knots[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + t * (y1 - y0)
    return knots[-1][1]

def _ratio_gate_for_pct(pct: int) -> float: return _piecewise_lerp(_RATIO_GATE_KNOTS, pct)
def _gs_gate_for_pct(pct: int) -> float: return _piecewise_lerp(_GS_GATE_KNOTS, pct)
def _corruption_gate_for_pct(pct: int) -> float: return _piecewise_lerp(_CORRUPTION_GATE_KNOTS, pct)
def _robustness_gs_gate_for_pct(pct: int) -> float: return _piecewise_lerp(_ROBUSTNESS_GS_GATE_KNOTS, pct)
def _corruption_level_for_pct(pct: int) -> float:
    t = max(0.0, min(100.0, pct)) / 100.0
    return CORRUPTION_LEVEL_MIN + t * (CORRUPTION_LEVEL_MAX - CORRUPTION_LEVEL_MIN)
def _corruption_garbage_for_pct(pct: int) -> float:
    t = max(0.0, min(100.0, pct)) / 100.0
    return CORRUPTION_GARBAGE_MAX * t
def _tier_from_pct(pct: int) -> 'DataComplexity':
    if pct < 20: return DataComplexity.VERY_SIMPLE
    elif pct < 40: return DataComplexity.SIMPLE
    elif pct < 60: return DataComplexity.MODERATE
    elif pct < 80: return DataComplexity.COMPLEX
    else: return DataComplexity.VERY_COMPLEX

def _interpolate_generation_params(pct: int) -> tuple:
    t = max(0.0, min(1.0, pct / 100.0))
    run_likelihood = 0.60 - 0.50 * t
    run_likelihood = max(0.05, min(0.70, run_likelihood + random.uniform(-0.05, 0.05)))
    unique_focus = 0.30 + 0.60 * t
    unique_focus = max(0.10, min(0.95, unique_focus + random.uniform(-0.05, 0.05)))
    max_run_cap = int(150 * (4.0 / 150.0) ** t)
    max_run_cap = max(2, max_run_cap + random.randint(-2, 2))
    return run_likelihood, unique_focus, max_run_cap

def get_generation_size_limits(generation: int, best_fitness: float = 0.0, previous_tier_index: int = -1, best_compression_ratio: float = 0.0, refreshes_at_tier: int = 0, gold_standard_win_rate: float = -1.0) -> tuple:
    _SIZE_DROP_HYSTERESIS = 0.50
    _MIN_SIZE_TIER_REFRESHES = 2
    active_budget = GENERATION_SIZE_TIERS[0][1]; active_ceiling = GENERATION_SIZE_TIERS[0][2]; active_index = 0
    for idx, (gen_threshold, budget, ceiling, _) in enumerate(GENERATION_SIZE_TIERS):
        if generation < gen_threshold: break
        ratio_gate = SIZE_TIER_RATIO_GATES[idx] if idx < len(SIZE_TIER_RATIO_GATES) else 0.0
        gs_gate = SIZE_TIER_GOLD_STANDARD_GATES[idx] if idx < len(SIZE_TIER_GOLD_STANDARD_GATES) else 0.0
        if idx > previous_tier_index:
            if best_compression_ratio < ratio_gate: continue
            if gs_gate > 0 and (gold_standard_win_rate < 0 or gold_standard_win_rate < gs_gate): continue
            if idx > previous_tier_index + 1: continue
            if refreshes_at_tier < _MIN_SIZE_TIER_REFRESHES: continue
        elif idx > 0 and best_compression_ratio > 0:
            drop_threshold = ratio_gate * _SIZE_DROP_HYSTERESIS
            if best_compression_ratio < drop_threshold: continue
        active_budget = budget; active_ceiling = ceiling; active_index = idx
    return (active_budget, active_ceiling, active_index)

def compute_continuous_benchmark_size(best_fitness: float = 0.0, previous_avg_size: int = 0, current_generation: int = 0, previous_tier_index: int = -1, best_compression_ratio: float = 0.0, refreshes_at_tier: int = 0, gold_standard_win_rate: float = -1.0) -> tuple:
    import math
    _, tier_ceiling, active_tier_index = get_generation_size_limits(current_generation, best_fitness, previous_tier_index=previous_tier_index, best_compression_ratio=best_compression_ratio, refreshes_at_tier=refreshes_at_tier, gold_standard_win_rate=gold_standard_win_rate)
    effective_ceiling = min(tier_ceiling, CONTINUOUS_SIZE_CEILING_BYTES)
    ratio_clamped = max(0.0, min(100.0, best_compression_ratio))
    t = ratio_clamped / 100.0
    if gold_standard_win_rate >= 0:
        gs_dampen = 0.10 + 0.90 * max(0.0, min(1.0, gold_standard_win_rate))
        t = min(t, gs_dampen)
    else:
        t = min(t, 0.35)
    log_floor = math.log(max(1, CONTINUOUS_SIZE_FLOOR_BYTES))
    log_ceiling = math.log(max(2, effective_ceiling))
    log_target = log_floor + t * (log_ceiling - log_floor)
    target_center = int(math.exp(log_target))
    if previous_avg_size > 0:
        max_allowed = int(previous_avg_size * MAX_SIZE_GROWTH_FACTOR)
        min_allowed = int(previous_avg_size * MIN_SIZE_SHRINK_FACTOR)
        min_allowed = max(min_allowed, CONTINUOUS_SIZE_FLOOR_BYTES)
        target_center = min(target_center, max_allowed)
        target_center = max(target_center, min_allowed)
    target_center = min(target_center, effective_ceiling)
    min_size = max(CONTINUOUS_SIZE_FLOOR_BYTES, int(target_center * 0.85))
    max_size = min(effective_ceiling, int(target_center * 1.15))
    if min_size >= max_size: max_size = min_size + 1024
    return (min_size, max_size, active_tier_index)


class BenchmarkItemEvaluator:
    def __init__(self, benchmark_dataset_path=None, logger_instance=None, tuned_params=None, dynamic_benchmarking=True):
        self.benchmark_dataset_path = benchmark_dataset_path
        self.benchmark_items = []
        self.logger = logger_instance if logger_instance else _setup_logger_func_val("BenchmarkEvaluator", log_level=logging.INFO)
        self.tuned_params = tuned_params if tuned_params is not None else {}
        from .performance_tuner import get_tuned_parameters
        tier_params = get_tuned_parameters("BALANCED")
        self.tuned_params.update(tier_params)
        self.items_per_throttle_check = self.tuned_params.get("ITEMS_PER_THROTTLE_CHECK", 50)
        self.agents_per_throttle_check = self.tuned_params.get("AGENTS_PER_THROTTLE_CHECK", AGENTS_PER_THROTTLE_CHECK)
        self.items_per_throttle_check = self.tuned_params.get("ITEMS_PER_THROTTLE_CHECK", ITEMS_PER_THROTTLE_CHECK)
        self.throttle_sleep_duration = self.tuned_params.get("THROTTLE_SLEEP_DURATION_BENCH_EVAL", THROTTLE_SLEEP_DURATION_BENCH_EVAL)
        self.dynamic_benchmarking_enabled = dynamic_benchmarking
        self._temp_agent_for_generation = None
        self._previous_tier_index = -1
        self._previous_size_tier_refreshes = 0
        self._last_best_compression_ratio = 0.0
        self._current_complexity_tier = DataComplexity.VERY_SIMPLE
        self._complexity_pct: int = 0
        self._refreshes_at_current_tier = 0
        self._manual_benchmark_size_bytes: int | None = None
        self._manual_complexity_pct: int | None = None
        # --- Anti-corruption progression (independent second scoring track) ---
        # Advances on robustness recovery rate, NOT on compression ratio, so the
        # two tracks can progress independently.  See _corruption_gate_for_pct.
        self._corruption_pct: int = 0
        self._refreshes_at_current_corruption_tier = 0
        self._best_robustness_rate_at_tier: float = 0.0
        self._corruption_tier_entry_rate: float = 0.0
        self._last_robustness_rate: float = -1.0
        # Robustness gold-standard head-to-head win rate from the most recent
        # generation (fraction 0.0-1.0 of corrupted items where the best
        # anti-corruption agent survived AND beat every baseline compressor).
        # -1.0 = no data yet.  Gates corruption advancement, mirroring the way
        # the compression gold-standard win rate gates complexity advancement.
        self._last_robustness_gs_win_rate: float = -1.0
        self._manual_corruption_pct: int | None = None
        self.logger.info(f"BenchmarkItemEvaluator initialized. Dynamic Benchmarking: {self.dynamic_benchmarking_enabled}.")
        if not self.dynamic_benchmarking_enabled and self.benchmark_dataset_path:
            self.load_benchmark_data(self.benchmark_dataset_path)
        elif self.dynamic_benchmarking_enabled:
            self.logger.info("Dynamic benchmarking enabled. Initial items will be generated on demand.")

    def set_manual_overrides(self, benchmark_size_kb: int | None = None, complexity_pct: int | None = None):
        if benchmark_size_kb is not None:
            self._manual_benchmark_size_bytes = max(1024, int(benchmark_size_kb) * 1024)
            self.logger.info(f"Manual floor set: benchmark_size >= {benchmark_size_kb} KB")
        else:
            self._manual_benchmark_size_bytes = None
            self.logger.info("Manual floor cleared: benchmark_size = auto")
        if complexity_pct is not None:
            self._manual_complexity_pct = max(0, min(100, int(complexity_pct)))
            self.logger.info(f"Manual floor set: complexity_pct >= {self._manual_complexity_pct}%")
        else:
            self._manual_complexity_pct = None
            self.logger.info("Manual floor cleared: complexity_pct = auto")

    def determine_target_corruption(self, best_robustness_rate: float,
                                     robustness_gs_win_rate: float = -1.0) -> int:
        """Advance / retreat the anti-corruption difficulty (``_corruption_pct``).

        This is the SECOND, independent scoring track.  It is driven solely by
        *robustness* — the recovery rate (0.0-1.0) achieved by the best
        anti-corruption agent on the previous generation's corrupted data — and
        never looks at compression ratio.  As a result the corruption level can
        keep climbing while compression stagnates (and vice-versa), which is the
        whole point of separating the two systems.

        Args:
            best_robustness_rate: Fraction (0.0-1.0) of corrupted/noisy items the
                best anti-corruption agent handled successfully last generation.
                ``< 0`` means "no data yet" (first generation) — hold position.
            robustness_gs_win_rate: Fraction (0.0-1.0) of corrupted items where
                the best anti-corruption agent beat EVERY baseline compressor in
                the robustness gold-standard head-to-head.  This is an ADDITIONAL
                gate on advancement (mirrors the compression track's gold-standard
                gate): corruption only climbs when the lineage is provably more
                resilient than off-the-shelf compressors, not merely improving on
                its own internal recovery metric.  ``< 0`` = no data yet (lenient:
                does not block).

        Returns:
            The (possibly updated) corruption pct, 0-100.
        """
        self._refreshes_at_current_corruption_tier += 1
        self._last_robustness_rate = best_robustness_rate
        self._last_robustness_gs_win_rate = robustness_gs_win_rate

        # No robustness signal yet — stay where we are (respect manual floor below).
        if best_robustness_rate is not None and best_robustness_rate >= 0.0:
            if best_robustness_rate > self._best_robustness_rate_at_tier:
                self._best_robustness_rate_at_tier = best_robustness_rate

            # --- Advance one point on either path (whichever fires first) ---
            #   ABSOLUTE: recovery rate clears the next gate.
            #   RELATIVE: recovery rate beat the tier-entry rate by the margin
            #             (i.e. robustness genuinely improved) — this keeps the
            #             track responsive even when the absolute gate is out of
            #             reach for the current data regime.
            if self._corruption_pct < 100:
                next_pct = self._corruption_pct + 1
                gate = _corruption_gate_for_pct(next_pct)
                dwell_ok = (self._refreshes_at_current_corruption_tier
                            >= self._MIN_REFRESHES_BEFORE_ADVANCE)
                abs_ok = best_robustness_rate >= gate
                rel_ok = best_robustness_rate >= (self._corruption_tier_entry_rate + _CORRUPTION_REL_MARGIN)
                # Gold-standard gate: the best anti-corruption agent must also be
                # beating the baseline compressors on corrupted data.  Lenient
                # when no head-to-head data exists yet (win_rate < 0) so the
                # track is never stalled before the first robustness benchmark.
                gs_gate = _robustness_gs_gate_for_pct(next_pct)
                gs_ok = (gs_gate <= 0.0
                         or robustness_gs_win_rate < 0.0
                         or robustness_gs_win_rate >= gs_gate)
                if dwell_ok and best_robustness_rate > _CORRUPTION_MIN_RATE and (abs_ok or rel_ok) and gs_ok:
                    old_pct = self._corruption_pct
                    self._corruption_pct = next_pct
                    self._refreshes_at_current_corruption_tier = 0
                    self._corruption_tier_entry_rate = best_robustness_rate
                    self._best_robustness_rate_at_tier = best_robustness_rate
                    _why = "gate" if abs_ok else "improvement"
                    _gs_str = (f", GS {robustness_gs_win_rate:.0%}≥{gs_gate:.0%}"
                               if gs_gate > 0 and robustness_gs_win_rate >= 0 else "")
                    self.logger.info(
                        f"Corruption +1%: {old_pct}% → {self._corruption_pct}% "
                        f"(robustness {best_robustness_rate:.0%}, via {_why}{_gs_str})")
                    return self._corruption_pct

            # --- Drop with hysteresis when robustness collapses ---
            _DROP_HYSTERESIS = 0.75
            if self._corruption_pct > 0:
                target_pct = self._corruption_pct
                while target_pct > 0:
                    gate = _corruption_gate_for_pct(target_pct)
                    if best_robustness_rate < gate * _DROP_HYSTERESIS:
                        target_pct -= 1
                    else:
                        break
                if target_pct < self._corruption_pct:
                    dropped_from = self._corruption_pct
                    self._corruption_pct = target_pct
                    self._refreshes_at_current_corruption_tier = 0
                    self._corruption_tier_entry_rate = best_robustness_rate
                    self._best_robustness_rate_at_tier = best_robustness_rate
                    self.logger.info(
                        f"Corruption dropped: {dropped_from}% → {self._corruption_pct}% "
                        f"(robustness {best_robustness_rate:.0%}, 75% hysteresis)")

        # Manual floor: never let the corruption track sit below the user floor.
        if self._manual_corruption_pct is not None and self._corruption_pct < self._manual_corruption_pct:
            self._corruption_pct = self._manual_corruption_pct
        return self._corruption_pct

    def _get_temp_generation_agent(self):
        if self._temp_agent_for_generation is None and _PuffinZipAI_cls is not None:
            try:
                self._temp_agent_for_generation = _PuffinZipAI_cls(
                    len_thresholds=None, learning_rate=_DEFAULT_LEARNING_RATE_val,
                    exploration_rate=_DEFAULT_EXPLORATION_RATE_val,
                    discount_factor=_DEFAULT_DISCOUNT_FACTOR_val,
                    exploration_decay_rate=_DEFAULT_EXPLORATION_DECAY_RATE_val,
                    min_exploration_rate=_DEFAULT_MIN_EXPLORATION_RATE_val,
                rle_min_encodable_run=random.randint(8, 64),
                )
            except Exception as e_temp_agent:
                self.logger.error(f"Failed to create temp PuffinZipAI for item generation: {e_temp_agent}")
        return self._temp_agent_for_generation

    def _generate_one_dynamic_item(self, complexity_level: DataComplexity, target_size_bytes_override: int | None = None) -> str:
        min_l, max_l = COMPLEXITY_LENGTH_RANGES_BYTES.get(complexity_level, COMPLEXITY_LENGTH_RANGES_BYTES[DataComplexity.SIMPLE])
        if target_size_bytes_override is not None and target_size_bytes_override > 0:
            variance_factor = random.uniform(0.8, 1.2)
            length = max(1, int(target_size_bytes_override * variance_factor))
        else:
            length = max(1, random.randint(min_l, max_l))
        if complexity_level == DataComplexity.USER_DEFINED_LARGE:
            run_likelihood = random.uniform(0.2, 0.4); unique_focus = random.uniform(0.5, 0.7); max_run_cap = 50
        else:
            run_likelihood, unique_focus, max_run_cap = _interpolate_generation_params(self._complexity_pct)
        alpha_num_sym = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{};':\",./<>? "
        pool_size = max(2, int(len(alpha_num_sym) * (0.2 + 0.8 * unique_focus)))
        char_pool = list(alpha_num_sym[:pool_size])
        base_max_run = max(2, int(length * 0.05 + (length * 0.2 * run_likelihood)))
        effective_max_run = min(base_max_run, max_run_cap)
        min_run = 2
        max_random_seg = max(1, int(5 * (1.0 - run_likelihood)))
        chunks = []
        remaining = length
        try:
            while remaining > 0:
                if random.random() < run_likelihood:
                    rlen = random.randint(min_run, max(min_run, min(effective_max_run, remaining)))
                    c = random.choice(char_pool)
                    chunks.append(c * rlen)
                    remaining -= rlen
                else:
                    seg_len = min(random.randint(1, max_random_seg), remaining)
                    chunks.append(''.join(random.choices(char_pool, k=seg_len)))
                    remaining -= seg_len
        except Exception:
            if remaining > 0:
                pool = string.ascii_letters + string.digits + " ._-\n"
                chunks.append(''.join(random.choices(pool, k=remaining)))
        item_content = ''.join(chunks)
        if len(item_content) < length: item_content += ''.join(random.choices(char_pool, k=length - len(item_content)))
        item_content = item_content[:length]
        return item_content

    @staticmethod
    def corrupt_compressed_data(compressed_text: str, corruption_level: float = 0.05) -> str:
        if not compressed_text or corruption_level <= 0.0: return compressed_text
        chars = list(compressed_text)
        n = len(chars)
        num_corruptions = max(1, int(n * corruption_level))
        for _ in range(num_corruptions):
            pos = random.randint(0, n - 1)
            op = random.random()
            if op < 0.5: chars[pos] = chr(random.randint(32, 126))
            elif op < 0.75: chars.insert(pos, chr(random.randint(32, 126))); n += 1
            else:
                if n > 1: chars.pop(pos); n -= 1
        return ''.join(chars)

    @staticmethod
    def _sanitize_to_printable(text: str) -> str:
        if not text: return text
        return ''.join(ch if 32 <= ord(ch) <= 126 else chr(32 + (ord(ch) % 95)) for ch in text)

    def generate_corrupted_benchmark_items(self, clean_items: list | None = None, corruption_level: float = 0.05, garbage_fraction: float = 0.0) -> list:
        source_items = clean_items if clean_items is not None else self.benchmark_items
        if not source_items: return []
        corrupted_pairs = []
        for item_text in source_items:
            if not item_text: continue
            try:
                if garbage_fraction > 0.0: item_text = self.inject_garbage_into_clean_data(item_text, garbage_fraction=garbage_fraction)
                if _rle_compress_func: compressed = _rle_compress_func(item_text, method="simple", min_run_len_override=3)
                else: compressed = item_text
                corrupted = self.corrupt_compressed_data(compressed, corruption_level)
                corrupted = self._sanitize_to_printable(corrupted)
                corrupted_pairs.append((corrupted, item_text))
            except Exception:
                corrupted = self.corrupt_compressed_data(item_text, corruption_level)
                corrupted = self._sanitize_to_printable(corrupted)
                corrupted_pairs.append((corrupted, item_text))
        return corrupted_pairs

    def inject_garbage_into_clean_data(self, item_text: str, garbage_fraction: float = 0.05) -> str:
        if not item_text or garbage_fraction <= 0.0: return item_text
        n = len(item_text)
        num_garbage = max(1, int(n * garbage_fraction))
        chars = list(item_text)
        for _ in range(num_garbage):
            pos = random.randint(0, len(chars))
            garbage_char = chr(random.randint(32, 126))
            chars.insert(pos, garbage_char)
        return ''.join(chars)

    def get_anti_corruption_benchmark_items(self, generation_num: int, clean_items: list[str] | None = None, github_items: list[str] | None = None, *, phased_enabled: bool = True, phase1_end: int = 10, phase2_end: int = 30, phase3_github_ratio: float = 0.80, best_robustness_rate: float = -1.0, robustness_gs_win_rate: float = -1.0) -> tuple[list[str], str, float]:
        clean_items = clean_items if clean_items is not None else list(self.benchmark_items)
        github_ratio = 0.0
        if phased_enabled:
            if generation_num <= phase1_end:
                github_ratio = 0.0
                phase_label = "Phase 1 (corruption-only)"
            elif generation_num <= phase2_end:
                progress = (generation_num - phase1_end) / max(1, phase2_end - phase1_end)
                github_ratio = progress * phase3_github_ratio
                phase_label = f"Phase 2 (blend {github_ratio:.0%} GitHub)"
            else:
                github_ratio = phase3_github_ratio
                phase_label = f"Phase 3 (GitHub {github_ratio:.0%})"
        else:
            phase_label = "corruption-only (phased training disabled)"
        # --- Robustness-driven corruption difficulty (independent track) ---
        # The corruption intensity is derived from _corruption_pct, which advances
        # on robustness (not generation number and not compression ratio).  This
        # is what lets the anti-corruption track flourish while compression
        # stagnates — the two scoring systems are fully decoupled.
        self.determine_target_corruption(best_robustness_rate, robustness_gs_win_rate=robustness_gs_win_rate)
        corruption_level = _corruption_level_for_pct(self._corruption_pct)
        garbage_fraction = _corruption_garbage_for_pct(self._corruption_pct)
        # Garbage injection only kicks in once phased training has started blending
        # real-world data, so early corruption-only phases stay pure corruption.
        if not (phased_enabled and generation_num > phase1_end):
            garbage_fraction = 0.0
        corrupted_pairs = self.generate_corrupted_benchmark_items(clean_items=clean_items, corruption_level=corruption_level, garbage_fraction=garbage_fraction)
        corrupted_items = [pair[0] for pair in corrupted_pairs] if corrupted_pairs else []
        if github_items is None: github_items = []
        if not github_items and github_ratio > 0.0:
            github_ratio = 0.0
            phase_label += " [DEGRADED: no GitHub data]"
        total_items = max(len(corrupted_items), 1)
        github_items_used = 0
        if github_items and github_ratio > 0:
            n_github = max(1, int(total_items * github_ratio))
            n_corrupt = total_items - n_github
            if len(github_items) < n_github:
                n_github = len(github_items)
                n_corrupt = total_items - n_github
            anti_corr_items = corrupted_items[:n_corrupt] + github_items[:n_github]
            github_items_used = min(n_github, len(github_items))
            random.shuffle(anti_corr_items)
        else:
            anti_corr_items = corrupted_items
        self._last_anti_corr_github_used = github_items_used
        return anti_corr_items, phase_label, corruption_level

    _MIN_REFRESHES_BEFORE_ADVANCE = 1

    def determine_target_complexity(self, population_average_fitness: float, best_compression_ratio: float = 0.0, gold_standard_win_rate: float = -1.0) -> DataComplexity:
        if not DataComplexity: return type('MockDataComplexity', (), {'VERY_SIMPLE': 0, 'SIMPLE': 1, 'MODERATE': 2, 'COMPLEX': 3, 'VERY_COMPLEX': 4})()
        self._refreshes_at_current_tier += 1
        old_pct = self._complexity_pct
        old_tier = self._current_complexity_tier
        if self._complexity_pct < 100:
            next_pct = self._complexity_pct + 1
            ratio_gate = _ratio_gate_for_pct(next_pct)
            gs_gate = _gs_gate_for_pct(next_pct)
            dwell_ok = (self._refreshes_at_current_tier >= self._MIN_REFRESHES_BEFORE_ADVANCE)
            gs_ok = (gs_gate <= 0 or (gold_standard_win_rate >= 0 and gold_standard_win_rate >= gs_gate))
            ratio_ok = best_compression_ratio >= ratio_gate
            if ratio_ok and dwell_ok and gs_ok:
                self._complexity_pct = next_pct
                self._current_complexity_tier = _tier_from_pct(next_pct)
                self._refreshes_at_current_tier = 0
                new_tier = self._current_complexity_tier
                tier_changed = (new_tier != old_tier)
                gs_str = (f", gs_wr {gold_standard_win_rate:.0%} ≥ {gs_gate:.0%}" if gs_gate > 0 and gold_standard_win_rate >= 0 else "")
                self.logger.info(f"Complexity +1%: {old_pct}% → {self._complexity_pct}% (ratio {best_compression_ratio:.1f}% ≥ {ratio_gate:.1f}%{gs_str})" + (f" [tier: {old_tier.name} → {new_tier.name}]" if tier_changed else ""))
                return self._current_complexity_tier
        _DROP_HYSTERESIS = 0.75
        if best_compression_ratio > 0 and self._complexity_pct > 0:
            target_pct = self._complexity_pct
            while target_pct > 0:
                gate = _ratio_gate_for_pct(target_pct)
                drop_threshold = gate * _DROP_HYSTERESIS
                if best_compression_ratio < drop_threshold: target_pct -= 1
                else: break
            if target_pct < self._complexity_pct:
                dropped_from = self._complexity_pct
                self._complexity_pct = target_pct
                self._current_complexity_tier = _tier_from_pct(target_pct)
                self._refreshes_at_current_tier = 0
                self.logger.info(f"Complexity dropped: {dropped_from}% → {self._complexity_pct}% (ratio {best_compression_ratio:.1f}%, 75% hysteresis)")
                return self._current_complexity_tier
        return self._current_complexity_tier

    def generate_and_set_dynamic_benchmark_items(self, num_items_to_generate: int = DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET, population_average_fitness: float = -100.0, current_generation: int = 0, target_item_size_mb_override: float | None = None, fixed_complexity_override_name: str | None = None, best_compression_ratio: float = 0.0, gold_standard_win_rate: float = -1.0):
        if not self.dynamic_benchmarking_enabled:
            self.logger.info("Dynamic benchmarking is disabled. No new items generated.")
            if not self.benchmark_items and self.benchmark_dataset_path: self.load_benchmark_data()
            if not self.benchmark_items: self.benchmark_items = ["Fallback AAA", "Fallback BBBCCC", "Fallback DDDDEEEEFFFF"]
            return bool(self.benchmark_items)
        if best_compression_ratio > 0: self._last_best_compression_ratio = best_compression_ratio
        target_size_bytes_final = None
        target_complexity_for_generation: DataComplexity = DataComplexity.SIMPLE

        if target_item_size_mb_override is not None and target_item_size_mb_override > 0:
            target_size_bytes_final = int(target_item_size_mb_override * 1024 * 1024)
            target_complexity_for_generation = DataComplexity.USER_DEFINED_LARGE
        elif fixed_complexity_override_name and DataComplexity:
            try: target_complexity_for_generation = DataComplexity[fixed_complexity_override_name.upper()]
            except KeyError: target_complexity_for_generation = self.determine_target_complexity(population_average_fitness, best_compression_ratio=best_compression_ratio, gold_standard_win_rate=gold_standard_win_rate)
        else:
            target_complexity_for_generation = self.determine_target_complexity(population_average_fitness, best_compression_ratio=best_compression_ratio, gold_standard_win_rate=gold_standard_win_rate)
            _active_item_ceiling = None
            try:
                prev_avg_size = 0
                if self.benchmark_items: prev_avg_size = sum(len(it) for it in self.benchmark_items if it) // max(1, len(self.benchmark_items))
                tier_budget, tier_ceiling, new_tier_index = get_generation_size_limits(current_generation, population_average_fitness, previous_tier_index=self._previous_tier_index, best_compression_ratio=best_compression_ratio, refreshes_at_tier=self._previous_size_tier_refreshes, gold_standard_win_rate=gold_standard_win_rate)
                _active_item_ceiling = tier_ceiling
                continuous_min, continuous_max, _ = compute_continuous_benchmark_size(population_average_fitness, previous_avg_size=prev_avg_size, current_generation=current_generation, previous_tier_index=self._previous_tier_index, best_compression_ratio=best_compression_ratio, refreshes_at_tier=self._previous_size_tier_refreshes, gold_standard_win_rate=gold_standard_win_rate)
                if new_tier_index != self._previous_tier_index: self._previous_size_tier_refreshes = 0
                else: self._previous_size_tier_refreshes += 1
                self._previous_tier_index = new_tier_index
                target_size_bytes_final = random.randint(continuous_min, continuous_max)
            except Exception as e_cont:
                target_size_bytes_final = None

        _floor_size = self._manual_benchmark_size_bytes
        _floor_cpx = self._manual_complexity_pct
        if _floor_cpx is not None and self._complexity_pct < _floor_cpx:
            self._complexity_pct = _floor_cpx
            self._current_complexity_tier = _tier_from_pct(_floor_cpx)
            target_complexity_for_generation = self._current_complexity_tier
        if _floor_size is not None:
            if target_size_bytes_final is None or target_size_bytes_final < _floor_size:
                target_size_bytes_final = _floor_size

        new_items = []
        for _ in range(num_items_to_generate):
            if target_size_bytes_final is not None:
                item_target = int(target_size_bytes_final * random.uniform(0.7, 1.3))
                item_target = max(CONTINUOUS_SIZE_FLOOR_BYTES, item_target)
                if _active_item_ceiling is not None: item_target = min(item_target, _active_item_ceiling)
            else:
                item_target = None
            new_items.append(self._generate_one_dynamic_item(target_complexity_for_generation, item_target))
        self.benchmark_items = new_items
        
        try:
            active_budget, _, _ = get_generation_size_limits(current_generation, population_average_fitness, previous_tier_index=self._previous_tier_index, best_compression_ratio=best_compression_ratio, refreshes_at_tier=self._previous_size_tier_refreshes, gold_standard_win_rate=gold_standard_win_rate)
        except Exception:
            active_budget = TOTAL_BENCHMARK_BUDGET_BYTES
        
        total_bytes = sum(len(item) for item in self.benchmark_items)
        if total_bytes > active_budget and self.benchmark_items:
            scale_factor = active_budget / total_bytes
            trimmed_items = []
            for item in self.benchmark_items:
                trimmed_len = max(CONTINUOUS_SIZE_FLOOR_BYTES, int(len(item) * scale_factor))
                trimmed_items.append(item[:trimmed_len])
            self.benchmark_items = trimmed_items
        
        return bool(self.benchmark_items)

    def load_benchmark_data(self, dataset_path=None):
        path_to_load = dataset_path if dataset_path else self.benchmark_dataset_path
        self.benchmark_items = []
        if not (path_to_load and os.path.exists(path_to_load) and os.path.isdir(path_to_load)): return False
        max_items_to_load = DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET if self.dynamic_benchmarking_enabled else 100
        loaded_count = 0
        for filename in os.listdir(path_to_load):
            if filename.endswith(".json"):
                filepath = os.path.join(path_to_load, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
                    if isinstance(data, list) and all(isinstance(item, dict) and "content" in item for item in data):
                        self.benchmark_items.extend([item['content'] for item in data if isinstance(item.get('content'), str)])
                        loaded_count += len(data)
                    elif isinstance(data, dict) and "content" in data and isinstance(data['content'], str):
                        self.benchmark_items.append(data['content'])
                        loaded_count += 1
                except Exception: pass
            elif filename.lower().endswith((".txt", ".log", ".md", ".csv")):
                filepath = os.path.join(path_to_load, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: item_content = f.read()
                    if item_content.strip(): self.benchmark_items.append(item_content); loaded_count += 1
                except Exception: pass
            if loaded_count >= max_items_to_load: break
        self.benchmark_items = self.benchmark_items[:max_items_to_load]
        return bool(self.benchmark_items)

    def get_total_benchmark_size_bytes(self) -> int:
        return sum(len(item) if isinstance(item, (str, bytes)) else 0 for item in self.benchmark_items)

    def set_custom_benchmark_items(self, items_list: list):
        if not isinstance(items_list, list) or not all(isinstance(item, str) for item in items_list): return False
        self.dynamic_benchmarking_enabled = False
        self.benchmark_items = items_list[:DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET]
        return True

    def evaluate_agent_fitness(self, agent_ai, repetitions: int = DEFAULT_BENCHMARK_REPETITIONS, gui_stop_event=None):
        if not self.benchmark_items:
            if self.dynamic_benchmarking_enabled: self.generate_and_set_dynamic_benchmark_items(num_items_to_generate=5, population_average_fitness=-1000)
            if not self.benchmark_items: return EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkDataAvailable"}
        if _PuffinZipAI_cls is None or not isinstance(agent_ai, _PuffinZipAI_cls): return EVALUATION_FAIL_REWARD, {}
        if _rle_compress_func is None or _rle_decompress_func is None or _calculate_reward_func is None: return EVALUATION_FAIL_REWARD, {}

        agent_id_str = f"AI(id={getattr(agent_ai, 'id_short', 'N/A')},min_run={getattr(agent_ai, 'rle_min_encodable_run_length', 'N/A')})"
        total_reward_for_agent = 0.0
        items_evaluated_count = 0
        eval_stats = {"total_reward": 0.0, "items_evaluated": 0, "successful_rle": 0, "rle_expansion": 0,
                      "rle_no_change": 0, "chose_nocompression": 0, "chose_adv_rle": 0, "chose_novel_method": 0, "chose_reference_method": 0,
                      "sum_compression_ratios_rle_success": 0.0, "sum_expansion_ratios_rle_fail": 0.0, "decomp_failures_mismatch": 0, "rle_errors_returned": 0,
                      "total_processing_time_ms": 0.0, "total_original_bytes": 0, "total_compressed_bytes": 0,
                      "method_bytes_saved": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                      "method_attempts": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                      "method_successes": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0}, }

        if not hasattr(self, '_cached_baseline_reward') or self._cached_baseline_generation != id(self.benchmark_items):
            baseline_total = 0.0; baseline_count = 0
            for item_text_bl in self.benchmark_items:
                if not item_text_bl: continue
                try:
                    bl_compressed = _rle_compress_func(item_text_bl, method="simple", min_run_len_override=6)
                    bl_decompressed = _rle_decompress_func(bl_compressed, method="simple", min_run_len_override=6)
                    bl_reward = _calculate_reward_func(item_text_bl, bl_compressed, bl_decompressed, "RLE", 0.5, None)
                    baseline_total += bl_reward; baseline_count += 1
                except Exception: pass
            self._cached_baseline_reward = baseline_total / max(baseline_count, 1)
            self._cached_baseline_generation = id(self.benchmark_items)
        baseline_avg_reward = self._cached_baseline_reward

        for item_idx, item_text in enumerate(self.benchmark_items):
            if gui_stop_event and gui_stop_event.is_set(): break
            if items_evaluated_count > 0 and items_evaluated_count % self.items_per_throttle_check == 0:
                if self.throttle_sleep_duration > 0: time.sleep(self.throttle_sleep_duration)

            sum_reward_for_item = 0.0
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
                        compressed_text_item_rep = _rle_compress_func(item_text, method="simple", min_run_len_override=rle_min_run)
                        decompressed_text_item_rep = _rle_decompress_func(compressed_text_item_rep, method="simple", min_run_len_override=rle_min_run)
                    elif action_name == "NoCompression":
                        compressed_text_item_rep = item_text; decompressed_text_item_rep = item_text; eval_stats['chose_nocompression'] += 1
                    elif action_name == "AdvancedRLE":
                        compressed_text_item_rep = _rle_compress_func(item_text, method="advanced")
                        decompressed_text_item_rep = _rle_decompress_func(compressed_text_item_rep, method="advanced"); eval_stats['chose_adv_rle'] += 1
                    elif action_name == "NovelMethod":
                        eval_stats['chose_novel_method'] += 1
                        novel_compress = getattr(agent_ai, '_novel_compress_fn', None); novel_decompress = getattr(agent_ai, '_novel_decompress_fn', None)
                        if novel_compress and novel_decompress:
                            try: compressed_text_item_rep = novel_compress(item_text); decompressed_text_item_rep = novel_decompress(compressed_text_item_rep)
                            except Exception: compressed_text_item_rep = item_text; decompressed_text_item_rep = "ERROR_NOVEL_METHOD_FAILED"
                        else: compressed_text_item_rep = item_text; decompressed_text_item_rep = item_text
                    elif action_name == "ReferenceMethod":
                        eval_stats['chose_reference_method'] += 1
                        ref_compress = getattr(agent_ai, '_reference_compress_fn', None); ref_decompress = getattr(agent_ai, '_reference_decompress_fn', None)
                        if ref_compress and ref_decompress:
                            try:
                                item_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
                                compressed_bytes = ref_compress(item_bytes)
                                decompressed_bytes = ref_decompress(compressed_bytes)
                                compressed_text_item_rep = "X" * len(compressed_bytes)
                                decompressed_text_item_rep = decompressed_bytes.decode('utf-8') if isinstance(decompressed_bytes, bytes) else decompressed_bytes
                            except Exception: compressed_text_item_rep = item_text; decompressed_text_item_rep = "ERROR_REFERENCE_METHOD_FAILED"
                        else: compressed_text_item_rep = item_text; decompressed_text_item_rep = item_text
                    else: decompressed_text_item_rep = "ERROR_UNKNOWN_ACTION_IN_EVAL"

                    if action_name in ["RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"]:
                        if original_size > 0:
                            compressed_size = len(compressed_text_item_rep) if compressed_text_item_rep else 0
                            eval_stats['total_original_bytes'] += original_size
                            eval_stats['total_compressed_bytes'] += compressed_size
                            _method_key = action_name
                            eval_stats['method_attempts'][_method_key] = eval_stats['method_attempts'].get(_method_key, 0) + 1
                            eval_stats['method_bytes_saved'][_method_key] = eval_stats['method_bytes_saved'].get(_method_key, 0) + (original_size - compressed_size)
                            if compressed_size < original_size and decompressed_text_item_rep == item_text:
                                rle_chosen_and_successful = True
                                eval_stats['sum_compression_ratios_rle_success'] += original_size / (compressed_size if compressed_size > 0 else 1)
                                eval_stats['method_successes'][_method_key] = eval_stats['method_successes'].get(_method_key, 0) + 1
                            elif compressed_size > original_size:
                                eval_stats['rle_expansion'] += 1
                                eval_stats['sum_expansion_ratios_rle_fail'] += original_size / (compressed_size if compressed_size > 0 else 1)
                            elif compressed_size == original_size: eval_stats['rle_no_change'] += 1
                        if rle_chosen_and_successful: eval_stats['successful_rle'] += 1

                        if decompressed_text_item_rep in _RLE_DECOMPRESSION_ERRORS_set:
                            rle_error_code_item_rep = decompressed_text_item_rep; eval_stats["rle_errors_returned"] += 1
                            if rle_chosen_and_successful: eval_stats['successful_rle'] -= 1; rle_chosen_and_successful = False
                        elif decompressed_text_item_rep != item_text:
                            eval_stats["decomp_failures_mismatch"] += 1
                            if rle_chosen_and_successful: eval_stats['successful_rle'] -= 1; rle_chosen_and_successful = False

                    processing_time_ms_rep = (time.perf_counter_ns() - start_time_ns_item_rep) / 1_000_000
                    eval_stats["total_processing_time_ms"] += processing_time_ms_rep
                    reward_rep = _calculate_reward_func(item_text, compressed_text_item_rep, decompressed_text_item_rep, action_name, processing_time_ms_rep, rle_error_code_item_rep)
                    
                    if _calculate_size_scaled_reward_func and original_size > 0:
                        was_success = (rle_chosen_and_successful or (action_name in ["RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"] and len(compressed_text_item_rep) < original_size and decompressed_text_item_rep == item_text))
                        reward_rep = _calculate_size_scaled_reward_func(reward_rep, original_size, len(compressed_text_item_rep), was_success)
                    
                    sum_reward_for_item += reward_rep
                    
                    if hasattr(agent_ai, '_update_q_table'):
                        try:
                            next_state = agent_ai._get_state_representation(compressed_text_item_rep) if compressed_text_item_rep else state_idx
                            agent_ai._update_q_table(state_idx, action_idx, reward_rep, next_state_idx=next_state)
                        except Exception: pass
                    
                    if processing_time_ms_rep > (MAX_ITEM_PROCESS_TIME_SEC * 1000): sum_reward_for_item += EVALUATION_TIMEOUT_REWARD_PENALTY
                except Exception: sum_reward_for_item += EVALUATION_FAIL_REWARD; eval_stats["rle_errors_returned"] += 1

            items_evaluated_count += 1
            total_reward_for_agent += (sum_reward_for_item / repetitions if repetitions > 0 else sum_reward_for_item)

        eval_stats["items_evaluated"] = items_evaluated_count; eval_stats["total_reward"] = total_reward_for_agent
        raw_fitness = total_reward_for_agent if items_evaluated_count == 0 else total_reward_for_agent / items_evaluated_count
        fitness_after_baseline = raw_fitness - baseline_avg_reward

        method_counts = {
            'RLE': max(0, items_evaluated_count - eval_stats['chose_nocompression'] - eval_stats['chose_adv_rle'] - eval_stats['chose_novel_method'] - eval_stats['chose_reference_method']),
            'NoCompression': eval_stats['chose_nocompression'], 'AdvancedRLE': eval_stats['chose_adv_rle'],
            'NovelMethod': eval_stats['chose_novel_method'], 'ReferenceMethod': eval_stats['chose_reference_method'],
        }
        diversity_adjustment = 0.0
        if _calculate_method_diversity_adjustment_func and items_evaluated_count > 0: diversity_adjustment = _calculate_method_diversity_adjustment_func(method_counts, items_evaluated_count)
        
        final_fitness_score = fitness_after_baseline + diversity_adjustment
        eval_stats['method_profile'] = {k: v / items_evaluated_count for k, v in method_counts.items()} if items_evaluated_count > 0 else {}
        eval_stats['diversity_adjustment'] = diversity_adjustment
        return final_fitness_score, eval_stats

    def evaluate_population_batch(self, population: list, repetitions_per_item: int = DEFAULT_BENCHMARK_REPETITIONS, gui_stop_event=None):
        if not self.benchmark_items:
            if self.dynamic_benchmarking_enabled: self.generate_and_set_dynamic_benchmark_items(num_items_to_generate=10, population_average_fitness=-1000)
            if not self.benchmark_items: return [(EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkDataAvailable"}) for _ in population]
        results = []
        for agent_idx, evolving_agent_instance in enumerate(population):
            if gui_stop_event and gui_stop_event.is_set(): results.extend([(EVALUATION_FAIL_REWARD, {"notes": "EvaluationInterrupted"}) for _ in range(len(population) - agent_idx)]); break
            if not (hasattr(evolving_agent_instance, 'puffin_ai') and evolving_agent_instance.puffin_ai is not None): results.append((EVALUATION_FAIL_REWARD, {"notes": "MissingCoreAI"})); continue
            agent_fitness, agent_stats_dict = self.evaluate_agent_fitness(evolving_agent_instance.puffin_ai, repetitions=repetitions_per_item, gui_stop_event=gui_stop_event)
            results.append((agent_fitness, agent_stats_dict))
        return results


    # --- Heterogeneous Computing Pipeline ---
    def evaluate_population_pipelined(self, population, cpu_pool, gui_stop_event=None):
        """GPU-batched inference -> CPU-parallel & GPU-batched compression.

        Phase 1 (GPU): Batched NN Action Inference for ALL agents.
        Phase 2 (Heterogeneous Split):
            - Collect all RLE tasks across all agents.
            - Send 50% to CPU multiprocessing pool.
            - Send 50% to a massive batched CuPy CUDA execution.
        Phase 3 (GPU & CPU): Collect results and feed to Q-learning batch updates.
        """
        if not self.benchmark_items:
            return [(EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkData"}) for _ in population]

        if not hasattr(self, "_cached_baseline_reward") or self._cached_baseline_generation != id(self.benchmark_items):
            bl_total, bl_count = 0.0, 0
            for it in self.benchmark_items:
                if not it: continue
                try:
                    bc = _rle_compress_func(it, method="simple", min_run_len_override=6)
                    bd = _rle_decompress_func(bc, method="simple", min_run_len_override=6)
                    bl_total += _calculate_reward_func(it, bc, bd, "RLE", 0.5, None)
                    bl_count += 1
                except Exception: pass
            self._cached_baseline_reward = bl_total / max(bl_count, 1)
            self._cached_baseline_generation = id(self.benchmark_items)
            
        baseline = self._cached_baseline_reward
        num_items = len(self.benchmark_items)

        # ── Phase 1 & 2 Setup ──
        pending = []  
        global_gpu_texts = []
        global_gpu_minruns = []
        global_gpu_agent_item_refs = []

        for aidx, agent in enumerate(population):
            if gui_stop_event and gui_stop_event.is_set():
                remaining = len(population) - len(pending)
                pending.extend([(None, None, None, [], [], []) for _ in range(remaining)])
                break

            if not (hasattr(agent, "puffin_ai") and agent.puffin_ai):
                pending.append((agent, None, None, [], [], []))
                continue

            ai = agent.puffin_ai
            rle_min_run = getattr(ai, "rle_min_encodable_run_length", 2)

            if hasattr(ai, "batch_choose_actions"): action_results = ai.batch_choose_actions(self.benchmark_items, use_exploration=False)
            else: action_results = [(ai._choose_action(ai._get_state_representation(it), use_exploration=False), None) for it in self.benchmark_items]

            novel_idxs = []
            reference_idxs = []
            futures = []
            
            for i in range(num_items):
                act_idx = action_results[i][0]
                act_name = ai.action_names.get(act_idx, "NoCompression")
                if act_name == "NovelMethod":
                    novel_idxs.append(i); futures.append(None)
                elif act_name == "ReferenceMethod":
                    reference_idxs.append(i); futures.append(None)
                elif act_name == "RLE" and CUPY_AVAILABLE and getattr(ai, 'use_gpu_acceleration', False):
                    # Route ALL simple-RLE work to the GPU.  Every agent's RLE
                    # items are coalesced into one big batched kernel launch,
                    # which is the whole reason the GPU path exists — sending
                    # only half here left the GPU largely idle.  The CPU pool
                    # still runs concurrently on NoCompression / AdvancedRLE /
                    # Novel / Reference items, so we keep heterogeneous overlap.
                    global_gpu_texts.append(self.benchmark_items[i])
                    global_gpu_minruns.append(rle_min_run)
                    global_gpu_agent_item_refs.append((aidx, i))
                    futures.append(None) # GPU slot
                else:
                    futures.append(cpu_pool.submit(_compress_single_item, (i, act_name, rle_min_run)))
                    
            pending.append((agent, ai, action_results, futures, novel_idxs, reference_idxs))

        # ── Phase 2 Execute GPU Batch concurrently with CPU Pool ──
        global_gpu_results = {}
        if global_gpu_texts:
            try:
                gpu_id = getattr(population[0].puffin_ai, 'gpu_id', 0) if population and population[0].puffin_ai else 0
                _t0_gpu = time.perf_counter_ns()
                
                comp_batch = gpu_accelerated_rle_compress_batch(global_gpu_texts, global_gpu_minruns, gpu_id)
                exp_lens = [len(t) for t in global_gpu_texts]
                decomp_batch = gpu_accelerated_rle_decompress_batch(comp_batch, exp_lens, gpu_id)
                
                gpu_time_ms = (time.perf_counter_ns() - _t0_gpu) / 1_000_000
                per_item_ms = gpu_time_ms / len(global_gpu_texts)

                for idx, (aidx, item_idx) in enumerate(global_gpu_agent_item_refs):
                    orig_text = global_gpu_texts[idx]
                    comp_text = comp_batch[idx]
                    decomp_text = decomp_batch[idx]
                    
                    orig_size = len(orig_text)
                    comp_size = len(comp_text) if comp_text else 0
                    ok = (decomp_text == orig_text)
                    rle_err = decomp_text if decomp_text in _RLE_DECOMPRESSION_ERRORS_set else None

                    reward = _calculate_reward_func(orig_text, comp_text, decomp_text, "RLE", per_item_ms, rle_err)
                    if _calculate_size_scaled_reward_func and orig_size > 0:
                        ws = (comp_size < orig_size and ok and rle_err is None)
                        reward = _calculate_size_scaled_reward_func(reward, orig_size, comp_size, ws)

                    global_gpu_results[(aidx, item_idx)] = {
                        "fallback": False, "item_idx": item_idx, "reward": reward,
                        "original_size": orig_size, "compressed_size": comp_size,
                        "was_success": (comp_size < orig_size and ok and rle_err is None),
                        "decompression_ok": ok, "rle_error": rle_err is not None,
                        "action_name": "RLE", "proc_ms": per_item_ms,
                    }
            except Exception as e:
                self.logger.error(f"Global GPU Batch failed: {e}")
                for idx, (aidx, item_idx) in enumerate(global_gpu_agent_item_refs):
                    global_gpu_results[(aidx, item_idx)] = {
                        "fallback": False, "item_idx": item_idx, "reward": EVALUATION_FAIL_REWARD,
                        "original_size": len(global_gpu_texts[idx]), "compressed_size": len(global_gpu_texts[idx]),
                        "was_success": False, "decompression_ok": False, "rle_error": True,
                        "action_name": "RLE", "proc_ms": 0.0,
                    }

        # ── Phase 3: Collect everything and aggregate ──
        all_results = []
        for aidx, (agent, ai, action_results, futures, novel_idxs, reference_idxs) in enumerate(pending):
            if ai is None:
                all_results.append((EVALUATION_FAIL_REWARD, {"notes": "MissingCoreAI"}))
                continue

            stats = {
                "total_reward": 0.0, "items_evaluated": 0, "successful_rle": 0, "rle_expansion": 0, "rle_no_change": 0,
                "chose_nocompression": 0, "chose_adv_rle": 0, "chose_novel_method": 0, "chose_reference_method": 0,
                "sum_compression_ratios_rle_success": 0.0, "sum_expansion_ratios_rle_fail": 0.0,
                "decomp_failures_mismatch": 0, "rle_errors_returned": 0, "total_processing_time_ms": 0.0,
                "total_original_bytes": 0, "total_compressed_bytes": 0,
                "method_bytes_saved": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                "method_attempts": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                "method_successes": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
            }

            item_results: list[Any] = [None] * num_items
            for i in range(num_items):
                if futures[i] is None:
                    if (aidx, i) in global_gpu_results:
                        item_results[i] = global_gpu_results[(aidx, i)]
                    continue
                try: item_results[i] = futures[i].result(timeout=120)
                except Exception:
                    item_results[i] = { "fallback": False, "item_idx": i, "reward": EVALUATION_FAIL_REWARD, "original_size": len(self.benchmark_items[i]), "compressed_size": len(self.benchmark_items[i]), "was_success": False, "decompression_ok": False, "rle_error": True, "action_name": "Unknown", "proc_ms": 0.0, }

            for i in novel_idxs:
                item_text = self.benchmark_items[i]
                novel_c = getattr(ai, "_novel_compress_fn", None); novel_d = getattr(ai, "_novel_decompress_fn", None)
                if novel_c and novel_d:
                    try:
                        _t0 = time.perf_counter_ns(); comp = novel_c(item_text); decomp = novel_d(comp)
                        ms = (time.perf_counter_ns() - _t0) / 1_000_000
                        osiz = len(item_text); csiz = len(comp) if comp else 0; ok = decomp == item_text; ws = csiz < osiz and ok
                        rw = _calculate_reward_func(item_text, comp, decomp, "NovelMethod", ms, None)
                        if _calculate_size_scaled_reward_func and osiz > 0: rw = _calculate_size_scaled_reward_func(rw, osiz, csiz, ws)
                        item_results[i] = { "fallback": False, "item_idx": i, "reward": rw, "original_size": osiz, "compressed_size": csiz, "was_success": ws, "decompression_ok": ok, "rle_error": False, "action_name": "NovelMethod", "proc_ms": ms, }
                    except Exception: item_results[i] = { "fallback": False, "item_idx": i, "reward": EVALUATION_FAIL_REWARD, "original_size": len(item_text), "compressed_size": len(item_text), "was_success": False, "decompression_ok": False, "rle_error": True, "action_name": "NovelMethod", "proc_ms": 0.0, }
                else: item_results[i] = { "fallback": False, "item_idx": i, "reward": EVALUATION_FAIL_REWARD * 0.5, "original_size": len(self.benchmark_items[i]), "compressed_size": len(self.benchmark_items[i]), "was_success": False, "decompression_ok": True, "rle_error": False, "action_name": "NovelMethod", "proc_ms": 0.0, }

            for i in reference_idxs:
                item_text = self.benchmark_items[i]
                ref_c = getattr(ai, "_reference_compress_fn", None); ref_d = getattr(ai, "_reference_decompress_fn", None)
                if ref_c and ref_d:
                    try:
                        item_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
                        _t0 = time.perf_counter_ns(); comp_bytes = ref_c(item_bytes); decomp_result = ref_d(comp_bytes)
                        ms = (time.perf_counter_ns() - _t0) / 1_000_000
                        decomp_text = decomp_result.decode('utf-8') if isinstance(decomp_result, bytes) else decomp_result
                        osiz = len(item_text); csiz = len(comp_bytes) if comp_bytes else 0; ok = decomp_text == item_text; ws = csiz < osiz and ok
                        comp_placeholder = "X" * csiz
                        rw = _calculate_reward_func(item_text, comp_placeholder, decomp_text, "ReferenceMethod", ms, None)
                        if _calculate_size_scaled_reward_func and osiz > 0: rw = _calculate_size_scaled_reward_func(rw, osiz, csiz, ws)
                        item_results[i] = { "fallback": False, "item_idx": i, "reward": rw, "original_size": osiz, "compressed_size": csiz, "was_success": ws, "decompression_ok": ok, "rle_error": False, "action_name": "ReferenceMethod", "proc_ms": ms, }
                    except Exception: item_results[i] = { "fallback": False, "item_idx": i, "reward": EVALUATION_FAIL_REWARD, "original_size": len(item_text), "compressed_size": len(item_text), "was_success": False, "decompression_ok": False, "rle_error": True, "action_name": "ReferenceMethod", "proc_ms": 0.0, }
                else: item_results[i] = { "fallback": False, "item_idx": i, "reward": EVALUATION_FAIL_REWARD * 0.5, "original_size": len(self.benchmark_items[i]), "compressed_size": len(self.benchmark_items[i]), "was_success": False, "decompression_ok": True, "rle_error": False, "action_name": "ReferenceMethod", "proc_ms": 0.0, }

            total_reward = 0.0
            experiences = []
            for i, r in enumerate(item_results):
                if r is None or r.get("fallback"): continue
                stats["items_evaluated"] += 1; total_reward += r["reward"]; stats["total_processing_time_ms"] += r.get("proc_ms", 0)
                stats["total_original_bytes"] += r["original_size"]; stats["total_compressed_bytes"] += r["compressed_size"]
                an = r.get("action_name", "")
                if an in ("RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"):
                    stats["method_attempts"][an] = stats["method_attempts"].get(an, 0) + 1
                    stats["method_bytes_saved"][an] = stats["method_bytes_saved"].get(an, 0) + (r["original_size"] - r["compressed_size"])
                    if r["was_success"]: stats["method_successes"][an] = stats["method_successes"].get(an, 0) + 1
                if an == "NoCompression": stats["chose_nocompression"] += 1
                elif an == "AdvancedRLE": stats["chose_adv_rle"] += 1
                elif an == "NovelMethod": stats["chose_novel_method"] += 1
                elif an == "ReferenceMethod": stats["chose_reference_method"] += 1

                if r["was_success"]:
                    stats["successful_rle"] += 1
                    if r["compressed_size"] > 0: stats["sum_compression_ratios_rle_success"] += (r["original_size"] / r["compressed_size"])
                elif an in ("RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"):
                    if r["compressed_size"] > r["original_size"]:
                        stats["rle_expansion"] += 1
                        if r["compressed_size"] > 0: stats["sum_expansion_ratios_rle_fail"] += (r["original_size"] / r["compressed_size"])
                    elif r["compressed_size"] == r["original_size"]: stats["rle_no_change"] += 1

                if r.get("rle_error"): stats["rle_errors_returned"] += 1
                elif not r.get("decompression_ok", True) and an != "NoCompression": stats["decomp_failures_mismatch"] += 1

                # Learn from this experience!
                feat = action_results[i][1] if len(action_results[i]) > 1 else None
                if feat is not None:
                    experiences.append((feat, action_results[i][0], r["reward"]))
                elif hasattr(ai, '_update_q_table'):
                    # Fallback for Tabular Agents so they still learn in pipelined eval
                    try:
                        state_idx = ai._get_state_representation(self.benchmark_items[i])
                        ai._update_q_table(state_idx, action_results[i][0], r["reward"], next_state_idx=state_idx)
                    except Exception: pass

            # Push NN batches
            if hasattr(ai, "batch_push_experiences") and experiences: 
                ai.batch_push_experiences(experiences)

            n = max(stats["items_evaluated"], 1)
            raw_fit = total_reward / n
            fit = raw_fit - baseline
            rle_count = n - stats["chose_nocompression"] - stats["chose_adv_rle"] - stats["chose_novel_method"] - stats["chose_reference_method"]
            mc = { "RLE": max(0, rle_count), "NoCompression": stats["chose_nocompression"], "AdvancedRLE": stats["chose_adv_rle"], "NovelMethod": stats["chose_novel_method"], "ReferenceMethod": stats["chose_reference_method"], }
            div_adj = 0.0
            if _calculate_method_diversity_adjustment_func and n > 0: div_adj = _calculate_method_diversity_adjustment_func(mc, n)
            fit += div_adj
            stats["total_reward"] = total_reward; stats["method_profile"] = {k: v / n for k, v in mc.items()} if n > 0 else {}
            stats["diversity_adjustment"] = div_adj
            all_results.append((fit, stats))

        return all_results