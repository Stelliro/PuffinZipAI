# PuffinZipAI_Project/puffinzip_ai/utils/benchmark_evaluator.py
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
    _fallback_logger_be.critical(
        f"CRITICAL ERROR (benchmark_evaluator.py): Failed to import core components. Error: {e_be_imp}", exc_info=True)

DEFAULT_BENCHMARK_REPETITIONS = 1
DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET = 20

def _is_debug():
    """Check debug flag dynamically from config module."""
    return _config_module and getattr(_config_module, 'DEBUG_LOG_CONSOLE_OUTPUT_ENABLED', False)

EVALUATION_FAIL_REWARD = -100.0
EVALUATION_TIMEOUT_REWARD_PENALTY = -50.0
MAX_ITEM_PROCESS_TIME_SEC = 30.0

# Optimized default throttle settings — GPU mode should NOT throttle
AGENTS_PER_THROTTLE_CHECK = 50
ITEMS_PER_THROTTLE_CHECK = 500
THROTTLE_SLEEP_DURATION_BENCH_EVAL = 0.0  # No sleep — let GPU saturate


# ---------------------------------------------------------------------------
# GPU+CPU Pipeline: ProcessPoolExecutor compression workers
# ---------------------------------------------------------------------------
# These module-level functions are picklable and run in child processes,
# bypassing the GIL for true CPU parallelism across all cores.

_pipeline_ctx: dict = {}  # Per-process context set by initialiser


def _pipeline_worker_init(benchmark_items_list):
    """Initialise a pipeline worker process.

    Called once per child process by ``ProcessPoolExecutor(initializer=...)``.
    Stores benchmark items and imports compression functions so each work
    item only transfers a small integer index over IPC.

    On Windows the ``spawn`` start-method causes Ctrl+C (SIGINT) to
    propagate to every child process, producing ugly ``KeyboardInterrupt``
    tracebacks.  We suppress SIGINT here so only the parent handles it and
    can shut the pool down cleanly.
    """
    import signal, sys
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    _pipeline_ctx["items"] = benchmark_items_list
    # Absolute imports — safe in spawned child processes
    from puffinzip_ai.rle_utils import rle_compress, rle_decompress
    from puffinzip_ai.reward_system import calculate_reward, calculate_size_scaled_reward
    from puffinzip_ai.rle_constants import RLE_DECOMPRESSION_ERRORS
    _pipeline_ctx["compress"] = rle_compress
    _pipeline_ctx["decompress"] = rle_decompress
    _pipeline_ctx["reward_fn"] = calculate_reward
    _pipeline_ctx["size_reward_fn"] = calculate_size_scaled_reward
    _pipeline_ctx["rle_errors"] = RLE_DECOMPRESSION_ERRORS


def _compress_single_item(args):
    """Compress + decompress + reward for one benchmark item.

    Runs inside a ``ProcessPoolExecutor`` child process so it truly bypasses
    the GIL.  Returns a compact result dict — full compressed text is NOT
    returned to minimise IPC payload.
    """
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
            # Unknown action — signal caller to handle in-process
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


# Legacy fitness thresholds — NO LONGER USED for gating decisions.
# Kept for reference / logging only.  All difficulty scaling now uses
# compression ratio (%) exclusively, per v0.9.7 design.
COMPLEXITY_FITNESS_THRESHOLDS = {
    DataComplexity.SIMPLE: -3.0,
    DataComplexity.MODERATE: 2.0,
    DataComplexity.COMPLEX: 6.0,
    DataComplexity.VERY_COMPLEX: 12.0
}

# --- COMPRESSION-RATIO-GATED COMPLEXITY (PRIMARY GATE) ---
# Per-tier minimum compression ratio (% reduction) required before the AI can
# advance TO that tier.  This is the SOLE gate for complexity advancement —
# the AI must prove it can compress well at the current scale before facing
# harder data patterns.  Fitness score is NOT used for gating.
# 70% means compressed output is ≤ 30% of original size (70% reduction).
COMPLEXITY_RATIO_GATES = {
    DataComplexity.SIMPLE: 25.0,       # 25% reduction to advance from VERY_SIMPLE
    DataComplexity.MODERATE: 45.0,     # 45% reduction to advance from SIMPLE
    DataComplexity.COMPLEX: 60.0,      # 60% reduction to advance from MODERATE
    DataComplexity.VERY_COMPLEX: 70.0  # 70% reduction to advance from COMPLEX
}

# --- GOLD STANDARD GATES (COMPLEXITY) ---
# Minimum fraction of benchmark items (0.0-1.0) that the AI must BEAT
# all baseline compressors on before advancing to the next complexity tier.
# Advancement requires ratio gate AND gold standard gate.
# The AI must prove it can compete with standard compressors before
# difficulty increases.  Even the first gate (SIMPLE) requires a small
# minimum so the AI doesn't advance on raw ratio alone while losing
# every head-to-head comparison.
COMPLEXITY_GOLD_STANDARD_GATES = {
    DataComplexity.SIMPLE: 0.10,       # Beat baselines on ≥10% of items to leave VERY_SIMPLE
    DataComplexity.MODERATE: 0.3,      # Beat baselines on ≥30% of items
    DataComplexity.COMPLEX: 0.5,       # Beat baselines on ≥50% of items
    DataComplexity.VERY_COMPLEX: 0.7,  # Beat baselines on ≥70% of items
}

# Resource-aware data sizes - kept small enough to avoid OOM on average PCs
# These are per-item sizes; with 30 items, SIMPLE = ~15-150MB total which is reasonable
COMPLEXITY_LENGTH_RANGES_BYTES = {
    DataComplexity.VERY_SIMPLE: (4 * 1024, 32 * 1024),          # 4KB - 32KB per item
    DataComplexity.SIMPLE: (32 * 1024, 256 * 1024),             # 32KB - 256KB per item
    DataComplexity.MODERATE: (256 * 1024, 1 * 1024 * 1024),     # 256KB - 1MB per item
    DataComplexity.COMPLEX: (1 * 1024 * 1024, 4 * 1024 * 1024), # 1MB - 4MB per item
    DataComplexity.VERY_COMPLEX: (4 * 1024 * 1024, 10 * 1024 * 1024)  # 4MB - 10MB per item
}

# --- CONTINUOUS SIZE SCALING (RATIO-BASED) ---
# Continuously scales item sizes based on the best compression ratio (%)
# rather than fitness score.  This ensures the AI must actually compress
# well at the current scale before sizes grow — not just accumulate a
# high fitness from bonuses.  Ratio 0% → smallest sizes, 100% → largest.
CONTINUOUS_SIZE_FLOOR_BYTES = 64 * 1024    # 64KB minimum — must be above SIZE_BONUS_BASE_THRESHOLD (32KB) so size scaling works
CONTINUOUS_SIZE_CEILING_BYTES = 10 * 1024 * 1024  # 10MB maximum per item at highest generation tier

# --- GENERATION-AWARE SIZE TIERS ---
# Each tier defines the total benchmark budget and per-item ceiling that
# unlocks at a given generation.  Advancement to a higher tier requires
# meeting the generation threshold AND the compression ratio gate from
# SIZE_TIER_RATIO_GATES.  The legacy min_fitness field is retained for
# backward compatibility but is NO LONGER checked.
# Format: (gen_threshold, total_budget_bytes, per_item_ceiling_bytes, _legacy_min_fitness)
GENERATION_SIZE_TIERS = [
    (0,   5 * 1024 * 1024,     256 * 1024,           -10.0),  # Gen 0-4:  warm-up, ~5MB total, 256KB/item
    (5,   10 * 1024 * 1024,    512 * 1024,             0.0),  # Gen 5-9:  ~10MB total, 512KB/item
    (10,  20 * 1024 * 1024,    1 * 1024 * 1024,        2.0),  # Gen 10-14: ~20MB total, 1MB/item
    (15,  40 * 1024 * 1024,    2 * 1024 * 1024,        4.0),  # Gen 15-19: ~40MB total, 2MB/item
    (20,  70 * 1024 * 1024,    4 * 1024 * 1024,        7.0),  # Gen 20-24: ~70MB total, 4MB/item
    (25,  120 * 1024 * 1024,   7 * 1024 * 1024,       10.0),  # Gen 25+:   ~120MB total, 7MB/item
]

# --- SIZE TIER RATIO GATES ---
# Minimum compression ratio (%) the AI must achieve on the CURRENT tier's
# data before the size tier can advance.  This mirrors COMPLEXITY_RATIO_GATES
# but for benchmark SIZE: the AI must prove it can compress at the current
# scale before data gets bigger.  Index matches GENERATION_SIZE_TIERS.
# Tier 0 has no gate (always unlocked).  Higher tiers require progressively
# better compression ratios.
SIZE_TIER_RATIO_GATES = [
    0.0,    # Tier 0: always unlocked (warm-up)
    20.0,   # Tier 1: need 20% compression to move from 256KB to 512KB items
    35.0,   # Tier 2: need 35% compression to move to 1MB items
    50.0,   # Tier 3: need 50% compression to move to 2MB items
    60.0,   # Tier 4: need 60% compression to move to 4MB items
    70.0,   # Tier 5: need 70% compression to move to 7MB items
]

# --- SIZE TIER GOLD STANDARD GATES ---
# Minimum fraction of benchmark items (0.0-1.0) the AI must beat all
# baselines on before advancing to the next size tier.  Mirrors
# COMPLEXITY_GOLD_STANDARD_GATES but for benchmark SIZE tiers.
# Index matches GENERATION_SIZE_TIERS / SIZE_TIER_RATIO_GATES.
SIZE_TIER_GOLD_STANDARD_GATES = [
    0.0,    # Tier 0: always unlocked
    0.1,    # Tier 1: beat baselines on ≥10% of items for 512KB items
    0.2,    # Tier 2: beat baselines on ≥20% of items for 1MB items
    0.3,    # Tier 3: ≥30% for 2MB items
    0.5,    # Tier 4: ≥50% for 4MB items
    0.6,    # Tier 5: ≥60% for 7MB items
]

# Fallback total benchmark budget (used when generation-aware tiers don't apply)
TOTAL_BENCHMARK_BUDGET_BYTES = 20 * 1024 * 1024  # 20 MB fallback

# Maximum growth multiplier per refresh — items can grow at most Nx between refreshes
# Kept conservative (2x) to prevent runaway size growth that tanks fitness.
MAX_SIZE_GROWTH_FACTOR = 2.0

# Minimum shrink factor per refresh — items can shrink at most to this fraction
# of their previous avg size.  Prevents the "crash cycle" where a size spike
# tanks fitness, sizes freefall, AI recovers, sizes spike again.
# 0.5 means items can halve at most per refresh.
MIN_SIZE_SHRINK_FACTOR = 0.5

# Legacy fitness hysteresis margin — NO LONGER USED.  Ratio-based gating
# uses the 0.5× ratio-drop threshold for hysteresis instead.
TIER_HYSTERESIS_MARGIN = 2.0


def get_generation_size_limits(generation: int, best_fitness: float = 0.0,
                               previous_tier_index: int = -1,
                               best_compression_ratio: float = 0.0,
                               refreshes_at_tier: int = 0,
                               gold_standard_win_rate: float = -1.0) -> tuple:
    """
    Determine the active total benchmark budget and per-item ceiling based on
    the current generation AND compression ratio.

    Walks the GENERATION_SIZE_TIERS list and selects the highest tier whose
    generation threshold is met AND whose SIZE_TIER_RATIO_GATES compression
    ratio gate is satisfied.  Fitness score is **not** used for gating —
    all difficulty decisions are ratio-based.

    Advancement to a new tier additionally requires:
      - at most **one tier** beyond the previous index (no skipping)
      - a minimum of ``_MIN_SIZE_TIER_REFRESHES`` refreshes at the current
        tier (dwell time)
      - gold standard win rate ≥ ``SIZE_TIER_GOLD_STANDARD_GATES[idx]``
        (if ``gold_standard_win_rate`` ≥ 0; -1 = no gold-standard data yet,
        which is treated as "gate not applicable").

    Previously-retained tiers can drop if the ratio falls below 50 % of
    their gate.

    Args:
        generation: Current generation number (1-based)
        best_fitness: (legacy, unused for gating — kept for API compat)
        previous_tier_index: Index (into GENERATION_SIZE_TIERS) of the tier
            that was active on the last refresh.  Pass -1 on the first call.
        best_compression_ratio: Best agent's compression ratio (0-100%).
            Tier advancement requires meeting SIZE_TIER_RATIO_GATES.
        refreshes_at_tier: Number of refreshes spent at the current tier.
            Must meet ``_MIN_SIZE_TIER_REFRESHES`` before advancing.
        gold_standard_win_rate: Fraction (0.0-1.0) of benchmark items where
            the AI beat ALL baseline compressors in the most recent gold
            standard benchmark.  -1.0 means no gold-standard data is
            available yet (gate is bypassed).

    Returns:
        (total_budget_bytes, per_item_ceiling_bytes, active_tier_index)
    """
    _SIZE_DROP_HYSTERESIS = 0.50  # 50 % of gate → drop retained tier
    # Minimum refreshes at a size tier before advancing.  Mirrors the
    # complexity dwell requirement so sizes don't outpace complexity.
    _MIN_SIZE_TIER_REFRESHES = 2

    active_budget = GENERATION_SIZE_TIERS[0][1]
    active_ceiling = GENERATION_SIZE_TIERS[0][2]
    active_index = 0

    for idx, (gen_threshold, budget, ceiling, _legacy_fitness) in enumerate(GENERATION_SIZE_TIERS):
        if generation < gen_threshold:
            break  # haven't reached this generation yet

        ratio_gate = SIZE_TIER_RATIO_GATES[idx] if idx < len(SIZE_TIER_RATIO_GATES) else 0.0
        gs_gate = SIZE_TIER_GOLD_STANDARD_GATES[idx] if idx < len(SIZE_TIER_GOLD_STANDARD_GATES) else 0.0

        if idx > previous_tier_index:
            # NEW tier — must meet the full ratio gate to advance.
            if best_compression_ratio < ratio_gate:
                continue  # ratio too low to advance to this tier
            # Gold standard gate — must beat baselines on enough items.
            # When gs_gate > 0, BLOCK advancement if no gold standard data
            # exists yet (win_rate < 0) OR the win rate is below the gate.
            # The old code bypassed the gate when win_rate < 0, which let
            # tier advancement happen before the AI proved it can beat
            # baselines at any scale.
            if gs_gate > 0 and (gold_standard_win_rate < 0 or gold_standard_win_rate < gs_gate):
                continue  # gold standard win rate too low or no data yet
            # Also enforce the dwell requirement — only advance ONE tier
            # at a time and only if the AI has spent enough refreshes at
            # the current tier.  ``refreshes_at_tier`` is passed in.
            if idx > previous_tier_index + 1:
                continue  # can only advance one tier at a time
            if refreshes_at_tier < _MIN_SIZE_TIER_REFRESHES:
                continue  # dwell requirement not met
        elif idx > 0 and best_compression_ratio > 0:
            # RETAINED tier — allow it to DROP if the ratio has fallen
            # well below the gate that unlocked it (50 % hysteresis).
            # Tier 0 is always retained (warm-up baseline).
            drop_threshold = ratio_gate * _SIZE_DROP_HYSTERESIS
            if best_compression_ratio < drop_threshold:
                continue  # ratio too low — revoke this tier

        active_budget = budget
        active_ceiling = ceiling
        active_index = idx

    return (active_budget, active_ceiling, active_index)


def compute_continuous_benchmark_size(best_fitness: float = 0.0, previous_avg_size: int = 0,
                                      current_generation: int = 0,
                                      previous_tier_index: int = -1,
                                      best_compression_ratio: float = 0.0,
                                      refreshes_at_tier: int = 0,
                                      gold_standard_win_rate: float = -1.0) -> tuple:
    """
    Compute target benchmark item size range as a continuous function of
    **compression ratio**, with generation-aware ceiling scaling and
    **bidirectional growth limiting**.

    The interpolation parameter ``t`` is derived directly from the
    compression ratio (0% → smallest sizes, 100% → largest).  Fitness score
    is **not** used — all size scaling is ratio-based.

    Growth is capped at ``MAX_SIZE_GROWTH_FACTOR`` (2×) and shrinkage is
    floored at ``MIN_SIZE_SHRINK_FACTOR`` (0.5×) per refresh to prevent
    destructive oscillation.

    Args:
        best_fitness: (legacy, unused for sizing — kept for API compat)
        previous_avg_size: Average item size from previous benchmark set (0 if first)
        current_generation: Current generation number (1-based)
        previous_tier_index: Tier index from last refresh (for hysteresis)
        best_compression_ratio: Best agent's compression ratio (0-100%).
            This is the PRIMARY driver of size scaling.
        gold_standard_win_rate: Fraction of items where AI beat all baselines
            (-1.0 = no data yet).

    Returns:
        (min_size_bytes, max_size_bytes, active_tier_index)
    """
    import math

    # Determine the active per-item ceiling from generation tiers (with hysteresis)
    _, tier_ceiling, active_tier_index = get_generation_size_limits(
        current_generation, best_fitness, previous_tier_index=previous_tier_index,
        best_compression_ratio=best_compression_ratio,
        refreshes_at_tier=refreshes_at_tier,
        gold_standard_win_rate=gold_standard_win_rate)
    effective_ceiling = min(tier_ceiling, CONTINUOUS_SIZE_CEILING_BYTES)

    # --- RATIO-BASED INTERPOLATION ---
    # Compression ratio (0-100%) directly drives size scaling.
    # 0% ratio  → t = 0 → smallest sizes (floor)
    # 50% ratio → t = 0.5 → midpoint
    # 100% ratio → t = 1.0 → largest sizes (ceiling)
    ratio_clamped = max(0.0, min(100.0, best_compression_ratio))
    t = ratio_clamped / 100.0

    # --- GOLD STANDARD DAMPENING ---
    # When gold standard data is available and the AI isn't beating
    # baselines, dampen the interpolation to prevent sizes from
    # growing beyond what the AI can actually handle competitively.
    # A high raw ratio on easy data doesn't mean the AI is doing well
    # if standard compressors achieve the same or better ratio.
    #   gs_win_rate=0.0 → gs_dampen=0.10 → t capped at 0.10 (tiny items)
    #   gs_win_rate=0.5 → gs_dampen=0.55 → moderate items
    #   gs_win_rate=1.0 → gs_dampen=1.0  → full range
    #   gs_win_rate<0   → warm-up cap at 0.35 (conservative until proven)
    if gold_standard_win_rate >= 0:
        gs_dampen = 0.10 + 0.90 * max(0.0, min(1.0, gold_standard_win_rate))
        t = min(t, gs_dampen)
    else:
        # No gold standard data yet (first 1-2 gens).  Use a conservative
        # warm-up cap so sizes don't start large before the AI has proven
        # it can compete with standard compressors.
        t = min(t, 0.35)

    # Exponential interpolation between floor and the tier-based ceiling
    log_floor = math.log(max(1, CONTINUOUS_SIZE_FLOOR_BYTES))
    log_ceiling = math.log(max(2, effective_ceiling))
    
    log_target = log_floor + t * (log_ceiling - log_floor)
    target_center = int(math.exp(log_target))
    
    # --- BIDIRECTIONAL GROWTH RATE LIMITER ---
    # Cap upward growth at MAX_SIZE_GROWTH_FACTOR (2x) AND
    # cap downward shrinkage at MIN_SIZE_SHRINK_FACTOR (0.5x).
    # This prevents the destructive cycle:
    #   high score → size spike → AI crashes → size freefalls → recovers → spike again
    if previous_avg_size > 0:
        max_allowed = int(previous_avg_size * MAX_SIZE_GROWTH_FACTOR)
        min_allowed = int(previous_avg_size * MIN_SIZE_SHRINK_FACTOR)
        # Floor the minimum so we never go below CONTINUOUS_SIZE_FLOOR_BYTES
        min_allowed = max(min_allowed, CONTINUOUS_SIZE_FLOOR_BYTES)
        target_center = min(target_center, max_allowed)
        target_center = max(target_center, min_allowed)

    # Hard-cap target_center to the per-item ceiling so the growth
    # limiter can never push sizes above the tier's ceiling.
    target_center = min(target_center, effective_ceiling)

    # Create a range around the target (±15%)
    # Kept tight to prevent random walks upward — a wide range (±50%)
    # causes benchmark sizes to drift up across refreshes because
    # random.randint picks skew toward upper values when combined with
    # per-item ±30% variance.
    min_size = max(CONTINUOUS_SIZE_FLOOR_BYTES, int(target_center * 0.85))
    max_size = min(effective_ceiling, int(target_center * 1.15))

    # Ensure min < max
    if min_size >= max_size:
        max_size = min_size + 1024

    return (min_size, max_size, active_tier_index)


class BenchmarkItemEvaluator:
    def __init__(self, benchmark_dataset_path=None, logger_instance=None, tuned_params=None, dynamic_benchmarking=True):
        self.benchmark_dataset_path = benchmark_dataset_path
        self.benchmark_items = []
        self.logger = logger_instance if logger_instance else _setup_logger_func_val("BenchmarkEvaluator",
                                                                                     log_level=logging.INFO)
        self.tuned_params = tuned_params if tuned_params is not None else {}
        from .performance_tuner import get_tuned_parameters
        tier_params = get_tuned_parameters("BALANCED")  # Or suggest_performance_tier()
        self.tuned_params.update(tier_params)
        self.items_per_throttle_check = self.tuned_params.get("ITEMS_PER_THROTTLE_CHECK", 50)
        self.agents_per_throttle_check = self.tuned_params.get("AGENTS_PER_THROTTLE_CHECK", AGENTS_PER_THROTTLE_CHECK)
        self.items_per_throttle_check = self.tuned_params.get("ITEMS_PER_THROTTLE_CHECK", ITEMS_PER_THROTTLE_CHECK)
        self.throttle_sleep_duration = self.tuned_params.get("THROTTLE_SLEEP_DURATION_BENCH_EVAL",
                                                             THROTTLE_SLEEP_DURATION_BENCH_EVAL)
        self.dynamic_benchmarking_enabled = dynamic_benchmarking
        self._temp_agent_for_generation = None
        self._previous_tier_index = -1  # tracks last active tier for hysteresis
        self._previous_size_tier_refreshes = 0  # how many refreshes spent at current size tier
        self._last_best_compression_ratio = 0.0  # best compression ratio (%) from last eval — gates complexity advancement
        # Progression-locked complexity tier.  Can only advance ONE step at a
        # time, and only when the next tier's ratio gate is met while training
        # on the CURRENT tier's data.  Prevents jumping from VERY_SIMPLE to
        # VERY_COMPLEX in a single generation.
        self._current_complexity_tier = DataComplexity.VERY_SIMPLE
        self._refreshes_at_current_tier = 0  # dwell counter — min refreshes before advancement
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
                rle_min_encodable_run=random.randint(8, 64),
                )
            except Exception as e_temp_agent:
                self.logger.error(f"Failed to create temp PuffinZipAI for item generation: {e_temp_agent}")
        return self._temp_agent_for_generation

    def _generate_one_dynamic_item(self, complexity_level: DataComplexity,
                                   target_size_bytes_override: int | None = None) -> str:
        """Generate a single benchmark item using fast bulk chunk generation.

        Always uses a direct chunk-based path (no agent delegation) for
        consistent, high-throughput data generation.  Produces MB-scale
        items in milliseconds instead of seconds.
        """
        min_l, max_l = COMPLEXITY_LENGTH_RANGES_BYTES.get(
            complexity_level, COMPLEXITY_LENGTH_RANGES_BYTES[DataComplexity.SIMPLE])

        # --- 1. Target size ---
        if target_size_bytes_override is not None and target_size_bytes_override > 0:
            variance_factor = random.uniform(0.8, 1.2)
            length = max(1, int(target_size_bytes_override * variance_factor))
        else:
            length = max(1, random.randint(min_l, max_l))

        # --- 2. Complexity → generation parameters ---
        # run_likelihood : probability each chunk is a repeated run
        # unique_focus   : controls character-pool breadth (higher = more unique)
        # max_run_cap    : hard ceiling on any single run length
        run_likelihood = 0.3
        unique_focus = 0.5
        max_run_cap = length  # no cap by default

        if complexity_level == DataComplexity.VERY_SIMPLE:
            run_likelihood = random.uniform(0.5, 0.7)
            unique_focus = random.uniform(0.2, 0.4)
            max_run_cap = random.randint(80, 150)   # cap runs so data isn't trivially compressible
        elif complexity_level == DataComplexity.SIMPLE:
            run_likelihood = random.uniform(0.4, 0.6)
            unique_focus = random.uniform(0.3, 0.5)
            max_run_cap = random.randint(40, 80)    # tighter cap than VERY_SIMPLE
        elif complexity_level == DataComplexity.MODERATE:
            run_likelihood = random.uniform(0.2, 0.4)
            unique_focus = random.uniform(0.5, 0.7)
            max_run_cap = 50
        elif complexity_level == DataComplexity.COMPLEX:
            run_likelihood = random.uniform(0.1, 0.25)
            unique_focus = random.uniform(0.65, 0.85)
            max_run_cap = 10
        elif complexity_level == DataComplexity.VERY_COMPLEX:
            run_likelihood = random.uniform(0.05, 0.15)
            unique_focus = random.uniform(0.8, 0.95)
            max_run_cap = 4

        # --- 3. Character pool ---
        alpha_num_sym = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{};':\",./<>? "
        pool_size = max(2, int(len(alpha_num_sym) * (0.2 + 0.8 * unique_focus)))
        char_pool = list(alpha_num_sym[:pool_size])

        # --- 4. Pre-compute run-length bounds ---
        base_max_run = max(2, int(length * 0.05 + (length * 0.2 * run_likelihood)))
        effective_max_run = min(base_max_run, max_run_cap)
        min_run = 2
        max_random_seg = max(1, int(5 * (1.0 - run_likelihood)))  # wider random segs at low run likelihood

        if length > 10 * 1024 * 1024:
            self.logger.info(
                f"Generating large item: complexity={getattr(complexity_level, 'name', 'UNKNOWN')}, "
                f"~{length / (1024 * 1024):.2f}MB")

        # --- 5. Bulk chunk generation (fast path for ALL complexity levels) ---
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
            # Ultra-fallback: fill remaining with random.choices
            if remaining > 0:
                pool = string.ascii_letters + string.digits + " ._-\n"
                chunks.append(''.join(random.choices(pool, k=remaining)))

        item_content = ''.join(chunks)
        # Ensure exact length
        if len(item_content) < length:
            item_content += ''.join(random.choices(char_pool, k=length - len(item_content)))
        item_content = item_content[:length]

        if length > 10 * 1024 * 1024:
            self.logger.info(
                f"Finished large item. Actual size: {len(item_content) / (1024 * 1024):.2f}MB")
        return item_content

    # ------------------------------------------------------------------
    #  CORRUPTION INJECTION (v0.9.7)
    #  Used to train anti-corruption agents.  Creates corrupted variants
    #  of benchmark data so agents learn to handle garbled inputs.
    # ------------------------------------------------------------------

    @staticmethod
    def corrupt_compressed_data(compressed_text: str, corruption_level: float = 0.05) -> str:
        """Inject random corruption into already-compressed text.

        Simulates partially corrupted compressed files (bit flips, byte
        insertion, byte deletion) that anti-corruption agents must
        decompress gracefully.

        All injected characters are restricted to the printable ASCII range
        (0x20-0x7E) so downstream decompression in novel methods is never
        broken by null bytes or control characters.

        Args:
            compressed_text: The compressed string (may contain control chars).
            corruption_level: Fraction of characters to corrupt (0.0 = none, 1.0 = all).

        Returns:
            Corrupted version of the compressed text (printable ASCII only).
        """
        if not compressed_text or corruption_level <= 0.0:
            return compressed_text

        chars = list(compressed_text)
        n = len(chars)
        num_corruptions = max(1, int(n * corruption_level))

        for _ in range(num_corruptions):
            pos = random.randint(0, n - 1)
            op = random.random()
            if op < 0.5:
                # Bit flip: replace with random printable char
                chars[pos] = chr(random.randint(32, 126))
            elif op < 0.75:
                # Byte insertion: insert printable garbage at position
                chars.insert(pos, chr(random.randint(32, 126)))
                n += 1
            else:
                # Byte deletion: remove character
                if n > 1:
                    chars.pop(pos)
                    n -= 1

        return ''.join(chars)

    @staticmethod
    def _sanitize_to_printable(text: str) -> str:
        """Replace any non-printable-ASCII characters with printable substitutes.

        Ensures downstream decompression in novel methods never encounters
        null bytes, control characters, or high-byte Unicode that could
        break string-based compression pipelines.
        """
        if not text:
            return text
        return ''.join(
            ch if 32 <= ord(ch) <= 126 else chr(32 + (ord(ch) % 95))
            for ch in text
        )

    def generate_corrupted_benchmark_items(self, clean_items: list | None = None,
                                           corruption_level: float = 0.05,
                                           garbage_fraction: float = 0.0) -> list:
        """Generate a set of corrupted benchmark items from existing clean items.

        Each clean item is optionally pre-corrupted with garbage injection
        (``inject_garbage_into_clean_data``), then compressed with simple RLE,
        then corrupted at the specified level.  Anti-corruption agents are
        evaluated on their ability to decompress these garbled files.

        Args:
            clean_items: List of clean text strings.  Defaults to current
                benchmark_items if None.
            corruption_level: Fraction of characters to corrupt (0.01-0.20).
            garbage_fraction: If > 0, inject this fraction of garbage bytes
                into the clean data *before* compression.  Used in later
                corruption stages to make the input itself noisy (default 0.0).

        Returns:
            List of (corrupted_compressed_text, original_text) tuples.
        """
        source_items = clean_items if clean_items is not None else self.benchmark_items
        if not source_items:
            return []

        corrupted_pairs = []
        for item_text in source_items:
            if not item_text:
                continue
            try:
                # Optionally inject garbage into clean data before compression
                if garbage_fraction > 0.0:
                    item_text = self.inject_garbage_into_clean_data(
                        item_text, garbage_fraction=garbage_fraction
                    )

                # Compress the (possibly garbage-injected) item first
                if _rle_compress_func:
                    compressed = _rle_compress_func(item_text, method="simple", min_run_len_override=3)
                else:
                    compressed = item_text
                # Inject corruption into the compressed output
                corrupted = self.corrupt_compressed_data(compressed, corruption_level)
                # Sanitize: RLE output may contain control chars; ensure
                # the final item is entirely printable ASCII.
                corrupted = self._sanitize_to_printable(corrupted)
                corrupted_pairs.append((corrupted, item_text))
            except Exception:
                # If compression fails, corrupt the raw text instead
                corrupted = self.corrupt_compressed_data(item_text, corruption_level)
                corrupted = self._sanitize_to_printable(corrupted)
                corrupted_pairs.append((corrupted, item_text))

        return corrupted_pairs

    def inject_garbage_into_clean_data(self, item_text: str,
                                       garbage_fraction: float = 0.05) -> str:
        """Inject random garbage bytes into clean (uncompressed) data.

        Used in later stages of corruption training (phase 2+): the agent
        must learn to compress data that already has embedded noise without
        crashing.  Garbage characters are restricted to printable ASCII
        (0x20-0x7E) to avoid null bytes / control characters that break
        downstream decompression in novel methods.

        Args:
            item_text: Clean input text.
            garbage_fraction: Fraction of text length to add as garbage.

        Returns:
            Text with injected garbage characters (printable ASCII only).
        """
        if not item_text or garbage_fraction <= 0.0:
            return item_text

        n = len(item_text)
        num_garbage = max(1, int(n * garbage_fraction))
        chars = list(item_text)

        for _ in range(num_garbage):
            pos = random.randint(0, len(chars))
            garbage_char = chr(random.randint(32, 126))
            chars.insert(pos, garbage_char)

        return ''.join(chars)

    # ------------------------------------------------------------------
    #  CENTRAL PHASED TRAINING API (v0.9.7)
    #  Single-source-of-truth for building the anti-corruption benchmark
    #  set for a given generation / phase / agent_type.  Eliminates duplication
    #  between evolutionary_optimizer.py and any other caller.
    # ------------------------------------------------------------------

    def get_anti_corruption_benchmark_items(
        self,
        generation_num: int,
        clean_items: list[str] | None = None,
        github_items: list[str] | None = None,
        *,
        phased_enabled: bool = True,
        phase1_end: int = 10,
        phase2_end: int = 30,
        phase3_github_ratio: float = 0.80,
    ) -> tuple[list[str], str, float]:
        """Build the anti-corruption benchmark item set for a generation.

        Centralises all phased-training decision logic so the optimizer
        (and any future caller) doesn't need to duplicate ratio/blend/label
        computation.

        Phases:
          1 (gen 0 → phase1_end):       100% corrupted synthetic data
          2 (phase1_end+1 → phase2_end): blend of corrupted + real files
          3 (phase2_end+1 onward):       mostly real-world files

        From phase 2 onward ``inject_garbage_into_clean_data`` is used to
        make the clean input itself noisy (garbage_fraction scales with
        generation, 0 → 0.03), so agents learn to handle both corrupted
        compressed AND noisy source data.

        Args:
            generation_num: Current generation number (1-based).
            clean_items: Clean benchmark items to corrupt.  Defaults to
                ``self.benchmark_items``.
            github_items: Pre-fetched GitHub real-world file contents.
                Pass ``[]`` or ``None`` if unavailable (the method will
                adjust the ratio and label accordingly).
            phased_enabled: Master switch for phased training.
            phase1_end: Last generation of corruption-only phase.
            phase2_end: Last generation of blended phase.
            phase3_github_ratio: Target fraction of GitHub items in phase 3.

        Returns:
            ``(items, phase_label, corruption_level)`` where *items* is the
            final list of benchmark-item strings, *phase_label* is a
            human-readable description of the active phase, and
            *corruption_level* is the corruption fraction applied.
        """
        clean_items = clean_items if clean_items is not None else list(self.benchmark_items)

        # -- Phase determination --
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

        # -- Corruption level scales with generation --
        corruption_level = min(0.15, 0.03 + generation_num * 0.002)

        # -- Garbage fraction: 0 in phase 1, ramps to 0.03 in phase 2+ --
        if phased_enabled and generation_num > phase1_end:
            progress_since_p1 = min(1.0, (generation_num - phase1_end) / max(1, phase2_end - phase1_end))
            garbage_fraction = 0.03 * progress_since_p1
        else:
            garbage_fraction = 0.0

        # -- PART A: corrupted synthetic items --
        corrupted_pairs = self.generate_corrupted_benchmark_items(
            clean_items=clean_items,
            corruption_level=corruption_level,
            garbage_fraction=garbage_fraction,
        )
        corrupted_items = [pair[0] for pair in corrupted_pairs] if corrupted_pairs else []

        # -- PART B: GitHub real-world items --
        if github_items is None:
            github_items = []
        if not github_items and github_ratio > 0.0:
            github_ratio = 0.0
            phase_label += " [DEGRADED: no GitHub data]"

        # -- MIX --
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

        # Attach metadata for logging by callers
        self._last_anti_corr_github_used = github_items_used

        return anti_corr_items, phase_label, corruption_level

    # Minimum number of benchmark refreshes the AI must spend at the
    # current complexity tier BEFORE it can advance to the next one.
    # This prevents the AI from sprinting through all tiers in < 15 gens
    # on trivially-compressible synthetic data.
    _MIN_REFRESHES_BEFORE_ADVANCE = 2

    def determine_target_complexity(self, population_average_fitness: float,
                                     best_compression_ratio: float = 0.0,
                                     gold_standard_win_rate: float = -1.0) -> DataComplexity:
        """Determine the complexity tier, enforcing single-step advancement
        with minimum dwell time and gold-standard gating, but allowing
        **multi-step drops**.

        Complexity can only advance **one tier at a time**, AND the AI must
        have spent at least ``_MIN_REFRESHES_BEFORE_ADVANCE`` refreshes at
        the current tier before advancement is allowed.  To advance from
        tier N to tier N+1:
          - the compression ratio must meet the next tier's ratio gate
            (``COMPLEXITY_RATIO_GATES``), AND
          - the gold standard win rate must meet the next tier's gold
            standard gate (``COMPLEXITY_GOLD_STANDARD_GATES``), unless
            no gold-standard data is available (win_rate < 0).

        This prevents the AI from jumping from VERY_SIMPLE to VERY_COMPLEX in
        a few generations just because synthetic data is easy to compress.

        Complexity can **DROP multiple tiers in one refresh** proportionally to
        how far the ratio has fallen.  The drop uses 75 % of each tier's gate
        as hysteresis.

        The result is stored in ``self._current_complexity_tier`` and returned.

        Args:
            population_average_fitness: (legacy, unused for gating — kept for
                API compat and logging)
            best_compression_ratio: Best agent's compression ratio as a
                percentage (0-100).  0 means no data / first gen.
            gold_standard_win_rate: Fraction (0.0-1.0) of benchmark items
                where the AI beat ALL baseline compressors.  -1.0 means no
                gold-standard data is available yet (gate bypassed).

        Returns:
            DataComplexity enum member.
        """
        if not DataComplexity:
            return type('MockDataComplexity', (),
                        {'VERY_SIMPLE': 0, 'SIMPLE': 1, 'MODERATE': 2, 'COMPLEX': 3,
                         'VERY_COMPLEX': 4})()

        # Increment the dwell counter each time this is called (= each refresh).
        self._refreshes_at_current_tier += 1

        current = self._current_complexity_tier
        current_val = getattr(current, 'value', 0)
        original_name = getattr(current, 'name', 'UNKNOWN')

        # --- TRY TO ADVANCE one tier ---
        # Only consider the immediately next tier (no skipping).
        # Must meet BOTH the ratio gate AND the minimum dwell time.
        next_val = current_val + 1
        max_val = getattr(DataComplexity.VERY_COMPLEX, 'value', 4)
        if next_val <= max_val:
            try:
                next_tier = DataComplexity(next_val)
            except ValueError:
                next_tier = None

            if next_tier is not None:
                ratio_gate = COMPLEXITY_RATIO_GATES.get(next_tier, 100.0)
                gs_gate = COMPLEXITY_GOLD_STANDARD_GATES.get(next_tier, 0.0)
                dwell_ok = self._refreshes_at_current_tier >= self._MIN_REFRESHES_BEFORE_ADVANCE
                # Gold standard gate: if a gate > 0 exists, the AI must
                # beat baselines on enough items.  When no gold standard
                # data is available yet (win_rate < 0), BLOCK advancement
                # instead of bypassing — the AI must prove itself first.
                gs_ok = (gs_gate <= 0
                         or (gold_standard_win_rate >= 0
                             and gold_standard_win_rate >= gs_gate))
                ratio_ok = best_compression_ratio >= ratio_gate
                if ratio_ok and dwell_ok and gs_ok:
                    self._current_complexity_tier = next_tier
                    self._refreshes_at_current_tier = 0  # reset dwell counter
                    gs_str = (f", gs_win_rate {gold_standard_win_rate:.0%} ≥ {gs_gate:.0%}"
                              if gs_gate > 0 and gold_standard_win_rate >= 0 else "")
                    self.logger.info(
                        f"Complexity advanced: {current.name} → {next_tier.name} "
                        f"(ratio {best_compression_ratio:.1f}% ≥ {ratio_gate:.0f}%, "
                        f"dwell satisfied{gs_str})")
                    return self._current_complexity_tier
                else:
                    # Log why advancement was deferred
                    reasons = []
                    if not ratio_ok:
                        reasons.append(f"ratio {best_compression_ratio:.1f}% < {ratio_gate:.0f}%")
                    if not dwell_ok:
                        reasons.append(f"dwell {self._refreshes_at_current_tier}/{self._MIN_REFRESHES_BEFORE_ADVANCE}")
                    if not gs_ok:
                        reasons.append(f"gs_win_rate {gold_standard_win_rate:.0%} < {gs_gate:.0%}")
                    if ratio_ok:  # only log when ratio is met but something else blocks
                        self.logger.info(
                            f"Complexity advancement deferred: {current.name} → {next_tier.name} "
                            f"({', '.join(reasons)})")

        # --- MULTI-STEP DROP ---
        # Keep dropping tiers as long as the compression ratio is below the
        # retention threshold for the current tier.  The retention threshold
        # is 75 % of the tier's advancement gate.
        #
        # This means difficulty always stays proportional to real performance:
        #   COMPLEX requires 60 % to advance → drops if ratio < 45 %
        #   MODERATE requires 45 % to advance → drops if ratio < 33.75 %
        #   SIMPLE requires 25 % to advance → drops if ratio < 18.75 %
        #
        # We skip the drop when best_compression_ratio == 0 (first gen or no
        # data) to avoid false drops on empty results.
        _DROP_HYSTERESIS = 0.75  # 75 % of advancement gate
        if best_compression_ratio > 0 and current_val > 0:
            target_val = current_val
            while target_val > 0:
                try:
                    tier_at = DataComplexity(target_val)
                except ValueError:
                    break
                gate = COMPLEXITY_RATIO_GATES.get(tier_at, 0.0)
                drop_threshold = gate * _DROP_HYSTERESIS
                if best_compression_ratio < drop_threshold:
                    target_val -= 1
                else:
                    break  # ratio is high enough to stay at this tier

            if target_val < current_val:
                try:
                    new_tier = DataComplexity(target_val)
                except ValueError:
                    new_tier = DataComplexity.VERY_SIMPLE
                self._current_complexity_tier = new_tier
                self._refreshes_at_current_tier = 0  # reset dwell counter on drop too
                self.logger.info(
                    f"Complexity dropped: {original_name} → {new_tier.name} "
                    f"(ratio {best_compression_ratio:.1f}% — multi-step, "
                    f"75 % hysteresis)")
                return self._current_complexity_tier

        # --- NO CHANGE ---
        return self._current_complexity_tier

    def generate_and_set_dynamic_benchmark_items(self, num_items_to_generate: int = DEFAULT_MAX_ITEMS_FOR_DYNAMIC_SET,
                                                 population_average_fitness: float = -100.0,
                                                 current_generation: int = 0,
                                                 target_item_size_mb_override: float | None = None,
                                                 fixed_complexity_override_name: str | None = None,
                                                 best_compression_ratio: float = 0.0,
                                                 gold_standard_win_rate: float = -1.0):
        if not self.dynamic_benchmarking_enabled:
            self.logger.info("Dynamic benchmarking is disabled. No new items generated.")
            if not self.benchmark_items and self.benchmark_dataset_path: self.load_benchmark_data()
            if not self.benchmark_items: self.benchmark_items = ["Fallback AAA", "Fallback BBBCCC",
                                                                 "Fallback DDDDEEEEFFFF"]
            return bool(self.benchmark_items)

        if best_compression_ratio > 0:
            self._last_best_compression_ratio = best_compression_ratio

        target_size_bytes_final = None
        target_complexity_for_generation: DataComplexity = DataComplexity.SIMPLE

        if target_item_size_mb_override is not None and target_item_size_mb_override > 0:
            target_size_bytes_final = int(target_item_size_mb_override * 1024 * 1024)
            target_complexity_for_generation = DataComplexity.USER_DEFINED_LARGE
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
                target_complexity_for_generation = self.determine_target_complexity(
                    population_average_fitness, best_compression_ratio=best_compression_ratio,
                    gold_standard_win_rate=gold_standard_win_rate)
        else:
            target_complexity_for_generation = self.determine_target_complexity(
                population_average_fitness, best_compression_ratio=best_compression_ratio,
                gold_standard_win_rate=gold_standard_win_rate)
            
            # --- CONTINUOUS SIZE SCALING (with hysteresis & bidirectional limits) ---
            # Uses the continuous fitness-to-size function with tier hysteresis
            # to prevent the destructive oscillation cycle.
            _active_item_ceiling = None  # per-item ceiling for capping below
            try:
                # Track previous avg size for bidirectional growth limiting
                prev_avg_size = 0
                if self.benchmark_items:
                    prev_avg_size = sum(len(it) for it in self.benchmark_items if it) // max(1, len(self.benchmark_items))
                # Get generation-aware budget and per-item ceiling (with hysteresis + ratio gate)
                tier_budget, tier_ceiling, new_tier_index = get_generation_size_limits(
                    current_generation, population_average_fitness,
                    previous_tier_index=self._previous_tier_index,
                    best_compression_ratio=best_compression_ratio,
                    refreshes_at_tier=self._previous_size_tier_refreshes,
                    gold_standard_win_rate=gold_standard_win_rate)
                _active_item_ceiling = tier_ceiling  # remember for per-item cap
                continuous_min, continuous_max, _ = compute_continuous_benchmark_size(
                    population_average_fitness, previous_avg_size=prev_avg_size,
                    current_generation=current_generation,
                    previous_tier_index=self._previous_tier_index,
                    best_compression_ratio=best_compression_ratio,
                    refreshes_at_tier=self._previous_size_tier_refreshes,
                    gold_standard_win_rate=gold_standard_win_rate)
                # Update dwell counter for size tier
                if new_tier_index != self._previous_tier_index:
                    self._previous_size_tier_refreshes = 0  # reset on tier change
                else:
                    self._previous_size_tier_refreshes += 1
                # Remember the active tier for next refresh's hysteresis
                self._previous_tier_index = new_tier_index
                # Use the continuous size as the target, but keep the complexity tier
                # for controlling data PATTERNS (run likelihood, unique chars, etc.)
                target_size_bytes_final = random.randint(continuous_min, continuous_max)
                self.logger.info(
                    f"Gen {current_generation}: Generation-aware size scaling active. "
                    f"Fitness={population_average_fitness:.3f} -> Target size: "
                    f"{continuous_min / 1024:.0f}KB - {continuous_max / 1024:.0f}KB per item "
                    f"(prev avg: {prev_avg_size / 1024:.0f}KB, "
                    f"tier budget: {tier_budget / (1024*1024):.0f}MB, "
                    f"tier ceiling: {tier_ceiling / 1024:.0f}KB/item, "
                    f"tier idx: {new_tier_index}). "
                    f"Complexity tier (for patterns): {getattr(target_complexity_for_generation, 'name', 'UNKNOWN')}")
            except Exception as e_cont:
                self.logger.warning(f"Continuous size scaling failed, using tier-based: {e_cont}")
                target_size_bytes_final = None

            # NOTE: The old "nudge" logic (bump complexity every N gens) has been
            # removed.  Complexity advancement is now fully handled by
            # determine_target_complexity() which enforces single-step
            # progression gated by BOTH fitness thresholds AND compression
            # ratio gates.  The nudge was redundant and could bypass the
            # single-step lock.

            self.logger.info(
                f"Generating {num_items_to_generate} new dynamic benchmark items. "
                f"Target Complexity: {getattr(target_complexity_for_generation, 'name', 'UNKNOWN')} "
                f"(based on AvgFit: {population_average_fitness:.3f}, "
                f"Ratio: {best_compression_ratio:.1f}%, Gen: {current_generation}).")

        # Generate items — each item gets a slightly randomized size from the range
        new_items = []
        for _ in range(num_items_to_generate):
            if target_size_bytes_final is not None:
                # Vary each item's size by ±30% for diversity
                item_target = int(target_size_bytes_final * random.uniform(0.7, 1.3))
                item_target = max(CONTINUOUS_SIZE_FLOOR_BYTES, item_target)
                # Hard-cap to the active per-item ceiling so variance
                # never produces items above the tier’s ceiling.
                if _active_item_ceiling is not None:
                    item_target = min(item_target, _active_item_ceiling)
            else:
                item_target = None
            new_items.append(self._generate_one_dynamic_item(target_complexity_for_generation, item_target))
        self.benchmark_items = new_items
        
        # --- TOTAL BUDGET ENFORCEMENT ---
        # Use generation-aware budget if available, otherwise fall back to static budget
        try:
            active_budget, _, _ = get_generation_size_limits(
                current_generation, population_average_fitness,
                previous_tier_index=self._previous_tier_index,
                best_compression_ratio=best_compression_ratio,
                refreshes_at_tier=self._previous_size_tier_refreshes,
                gold_standard_win_rate=gold_standard_win_rate)
        except Exception:
            active_budget = TOTAL_BENCHMARK_BUDGET_BYTES
        
        # If total benchmark size exceeds the active budget, trim items proportionally
        total_bytes = sum(len(item) for item in self.benchmark_items)
        if total_bytes > active_budget and self.benchmark_items:
            scale_factor = active_budget / total_bytes
            trimmed_items = []
            for item in self.benchmark_items:
                trimmed_len = max(CONTINUOUS_SIZE_FLOOR_BYTES, int(len(item) * scale_factor))
                trimmed_items.append(item[:trimmed_len])
            self.benchmark_items = trimmed_items
            new_total = sum(len(item) for item in self.benchmark_items)
            self.logger.info(
                f"Budget enforcement: {total_bytes / (1024*1024):.1f}MB exceeded "
                f"{active_budget / (1024*1024):.0f}MB budget -> trimmed to {new_total / (1024*1024):.1f}MB")
        
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
                      "chose_nocompression": 0, "chose_adv_rle": 0, "chose_novel_method": 0, "chose_reference_method": 0,
                      "sum_compression_ratios_rle_success": 0.0,
                      "sum_expansion_ratios_rle_fail": 0.0, "decomp_failures_mismatch": 0, "rle_errors_returned": 0,
                      "total_processing_time_ms": 0.0,
                      "total_original_bytes": 0, "total_compressed_bytes": 0,
                      # Per-method compression tracking: bytes saved (positive = good)
                      "method_bytes_saved": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                      "method_attempts": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                      "method_successes": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                      }

        # --- BASELINE COMPUTATION ---
        # Compute what plain simple RLE achieves on this benchmark so we only reward improvements
        if not hasattr(self, '_cached_baseline_reward') or self._cached_baseline_generation != id(self.benchmark_items):
            _bl_t0 = time.perf_counter()
            baseline_total = 0.0
            baseline_count = 0
            for item_text_bl in self.benchmark_items:
                if not item_text_bl:
                    continue
                try:
                    bl_compressed = _rle_compress_func(item_text_bl, method="simple", min_run_len_override=6)
                    bl_decompressed = _rle_decompress_func(bl_compressed, method="simple", min_run_len_override=6)
                    bl_reward = _calculate_reward_func(item_text_bl, bl_compressed, bl_decompressed, "RLE", 0.5, None)
                    baseline_total += bl_reward
                    baseline_count += 1
                except Exception:
                    pass
            self._cached_baseline_reward = baseline_total / max(baseline_count, 1)
            self._cached_baseline_generation = id(self.benchmark_items)
            _bl_elapsed = (time.perf_counter() - _bl_t0) * 1000
            total_bl_kb = sum(len(it) for it in self.benchmark_items if it) / 1024
            avg_item_kb = total_bl_kb / max(baseline_count, 1)
            if _is_debug():
                print(f"DEBUG-TIMING: [baseline_compute] {_bl_elapsed:.0f}ms for {baseline_count} items "
                      f"({total_bl_kb:.0f}KB total, avg {avg_item_kb:.0f}KB/item) "
                      f"baseline_avg_reward={self._cached_baseline_reward:.4f}")
        baseline_avg_reward = self._cached_baseline_reward

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
                _t_compress_ns = 0
                _t_decompress_ns = 0
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
                        _tc0 = time.perf_counter_ns()
                        compressed_text_item_rep = _rle_compress_func(item_text, method="simple",
                                                                      min_run_len_override=rle_min_run)
                        _tc1 = time.perf_counter_ns()
                        _t_compress_ns = _tc1 - _tc0
                        decompressed_text_item_rep = _rle_decompress_func(compressed_text_item_rep, method="simple",
                                                                          min_run_len_override=rle_min_run)
                        _t_decompress_ns = time.perf_counter_ns() - _tc1
                    elif action_name == "NoCompression":
                        compressed_text_item_rep = item_text
                        decompressed_text_item_rep = item_text
                        eval_stats['chose_nocompression'] += 1
                    elif action_name == "AdvancedRLE":
                        _tc0 = time.perf_counter_ns()
                        compressed_text_item_rep = _rle_compress_func(item_text, method="advanced")
                        _tc1 = time.perf_counter_ns()
                        _t_compress_ns = _tc1 - _tc0
                        decompressed_text_item_rep = _rle_decompress_func(compressed_text_item_rep, method="advanced")
                        _t_decompress_ns = time.perf_counter_ns() - _tc1
                        eval_stats['chose_adv_rle'] += 1
                    elif action_name == "NovelMethod":
                        eval_stats['chose_novel_method'] += 1
                        novel_compress = getattr(agent_ai, '_novel_compress_fn', None)
                        novel_decompress = getattr(agent_ai, '_novel_decompress_fn', None)
                        if novel_compress and novel_decompress:
                            try:
                                _tc0 = time.perf_counter_ns()
                                compressed_text_item_rep = novel_compress(item_text)
                                _tc1 = time.perf_counter_ns()
                                _t_compress_ns = _tc1 - _tc0
                                decompressed_text_item_rep = novel_decompress(compressed_text_item_rep)
                                _t_decompress_ns = time.perf_counter_ns() - _tc1
                            except Exception as e_novel:
                                compressed_text_item_rep = item_text
                                decompressed_text_item_rep = "ERROR_NOVEL_METHOD_FAILED"
                                self.logger.debug(f"Agent {agent_id_str}: NovelMethod failed: {e_novel}")
                        else:
                            compressed_text_item_rep = item_text
                            decompressed_text_item_rep = item_text
                    elif action_name == "ReferenceMethod":
                        eval_stats['chose_reference_method'] += 1
                        ref_compress = getattr(agent_ai, '_reference_compress_fn', None)
                        ref_decompress = getattr(agent_ai, '_reference_decompress_fn', None)
                        if ref_compress and ref_decompress:
                            try:
                                item_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
                                _tc0 = time.perf_counter_ns()
                                compressed_bytes = ref_compress(item_bytes)
                                _tc1 = time.perf_counter_ns()
                                _t_compress_ns = _tc1 - _tc0
                                decompressed_bytes = ref_decompress(compressed_bytes)
                                _t_decompress_ns = time.perf_counter_ns() - _tc1
                                # Use placeholder string of compressed length for ratio calc
                                compressed_text_item_rep = "X" * len(compressed_bytes)
                                decompressed_text_item_rep = decompressed_bytes.decode('utf-8') if isinstance(decompressed_bytes, bytes) else decompressed_bytes
                            except Exception as e_ref:
                                compressed_text_item_rep = item_text
                                decompressed_text_item_rep = "ERROR_REFERENCE_METHOD_FAILED"
                                self.logger.debug(f"Agent {agent_id_str}: ReferenceMethod failed: {e_ref}")
                        else:
                            compressed_text_item_rep = item_text
                            decompressed_text_item_rep = item_text
                    else:
                        self.logger.error(
                            f"Agent {agent_id_str} chose unknown action: {action_name} for item {item_idx}.")
                        decompressed_text_item_rep = "ERROR_UNKNOWN_ACTION_IN_EVAL"

                    if action_name in ["RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"]:
                        if original_size > 0:
                            compressed_size = len(compressed_text_item_rep) if compressed_text_item_rep else 0
                            eval_stats['total_original_bytes'] += original_size
                            eval_stats['total_compressed_bytes'] += compressed_size
                            # --- Per-method compression tracking ---
                            _method_key = action_name
                            eval_stats['method_attempts'][_method_key] = eval_stats['method_attempts'].get(_method_key, 0) + 1
                            _bytes_saved = original_size - compressed_size
                            eval_stats['method_bytes_saved'][_method_key] = eval_stats['method_bytes_saved'].get(_method_key, 0) + _bytes_saved
                            if compressed_size < original_size and decompressed_text_item_rep == item_text:
                                rle_chosen_and_successful = True
                                eval_stats['sum_compression_ratios_rle_success'] += original_size / (
                                    compressed_size if compressed_size > 0 else 1)
                                eval_stats['method_successes'][_method_key] = eval_stats['method_successes'].get(_method_key, 0) + 1
                            elif compressed_size > original_size:
                                eval_stats['rle_expansion'] += 1
                                eval_stats['sum_expansion_ratios_rle_fail'] += original_size / (
                                    compressed_size if compressed_size > 0 else 1)
                            elif compressed_size == original_size:
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
                    
                    # --- SIZE-SCALED REWARD ---
                    # Apply size bonus: compressing larger files well = much higher reward
                    if _calculate_size_scaled_reward_func and original_size > 0:
                        was_success = (rle_chosen_and_successful or 
                                      (action_name in ["RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"] and 
                                       len(compressed_text_item_rep) < original_size and 
                                       decompressed_text_item_rep == item_text))
                        reward_rep = _calculate_size_scaled_reward_func(
                            reward_rep, original_size, len(compressed_text_item_rep), was_success
                        )
                    
                    sum_reward_for_item += reward_rep
                    
                    # --- ONLINE Q-LEARNING DURING EVALUATION ---
                    # Feed benchmark experience back into the agent's Q-table
                    # so it learns from real data, not just random self-play
                    _t_qupdate_ns = 0
                    if hasattr(agent_ai, '_update_q_table'):
                        try:
                            _tq0 = time.perf_counter_ns()
                            next_state = agent_ai._get_state_representation(compressed_text_item_rep) if compressed_text_item_rep else state_idx
                            agent_ai._update_q_table(state_idx, action_idx, reward_rep, next_state_idx=next_state)
                            _t_qupdate_ns = time.perf_counter_ns() - _tq0
                        except Exception:
                            pass  # Non-fatal: evaluation scoring still works even if Q-update fails
                    
                    # --- PER-ITEM TIMING (only log slow items > 200ms) ---
                    _item_total_ms = processing_time_ms_rep
                    if _item_total_ms > 200:
                        if _is_debug():
                            print(f"DEBUG-TIMING: [slow_item] agent={agent_id_str} item={item_idx} size={original_size}B "
                                  f"action={action_name} total={_item_total_ms:.0f}ms "
                                  f"compress={_t_compress_ns/1e6:.0f}ms decompress={_t_decompress_ns/1e6:.0f}ms "
                                  f"q_update={_t_qupdate_ns/1e6:.0f}ms")

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
        raw_fitness = total_reward_for_agent if items_evaluated_count == 0 else total_reward_for_agent / items_evaluated_count
        
        # Subtract baseline: agents that just use built-in RLE score ~0
        # Only genuine improvements get positive fitness
        fitness_after_baseline = raw_fitness - baseline_avg_reward

        # --- METHOD DIVERSITY ADJUSTMENT ---
        # Penalize agents that spam one method, reward diverse strategy use
        method_counts = {
            'RLE': eval_stats.get('successful_rle', 0) + eval_stats.get('rle_expansion', 0) + eval_stats.get('rle_no_change', 0),
            'NoCompression': eval_stats.get('chose_nocompression', 0),
            'AdvancedRLE': eval_stats.get('chose_adv_rle', 0),
            'NovelMethod': eval_stats.get('chose_novel_method', 0),
            'ReferenceMethod': eval_stats.get('chose_reference_method', 0),
        }
        # RLE count needs to include all RLE attempts (not just successes)
        # Recalculate: total items minus other methods = RLE attempts
        rle_attempts = items_evaluated_count - method_counts['NoCompression'] - method_counts['AdvancedRLE'] - method_counts['NovelMethod'] - method_counts['ReferenceMethod']
        method_counts['RLE'] = max(0, rle_attempts)
        
        diversity_adjustment = 0.0
        if _calculate_method_diversity_adjustment_func and items_evaluated_count > 0:
            diversity_adjustment = _calculate_method_diversity_adjustment_func(method_counts, items_evaluated_count)
        
        final_fitness_score = fitness_after_baseline + diversity_adjustment
        
        # Store method profile for population-level novelty scoring
        eval_stats['method_profile'] = {}
        if items_evaluated_count > 0:
            for method_name, count in method_counts.items():
                eval_stats['method_profile'][method_name] = count / items_evaluated_count
        eval_stats['diversity_adjustment'] = diversity_adjustment

        log_msg = (
            f"Agent {agent_id_str} - Fitness Eval Complete. Items: {items_evaluated_count}, RawFit: {raw_fitness:.4f}, "
            f"Baseline: {baseline_avg_reward:.4f}, DiversityAdj: {diversity_adjustment:+.4f}, FinalFit: {final_fitness_score:.4f}. "
            f"Methods: RLE:{method_counts['RLE']}, NC:{method_counts['NoCompression']}, Adv:{method_counts['AdvancedRLE']}, Nov:{method_counts['NovelMethod']}. "
            f"Stats: SRLE:{eval_stats['successful_rle']}, Exp:{eval_stats['rle_expansion']}, "
            f"MM:{eval_stats['decomp_failures_mismatch']}, RLErr:{eval_stats['rle_errors_returned']}")
        self.logger.info(log_msg)
        return final_fitness_score, eval_stats

    def evaluate_population_batch(self, population: list, repetitions_per_item: int = DEFAULT_BENCHMARK_REPETITIONS,
                                  gui_stop_event=None):
        if not self.benchmark_items:
            self.logger.warning("No benchmark items loaded/generated. Cannot evaluate population.")
            if self.dynamic_benchmarking_enabled:
                self.generate_and_set_dynamic_benchmark_items(
                    num_items_to_generate=10, population_average_fitness=-1000)
            if not self.benchmark_items:
                return [(EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkDataAvailable"}) for _ in population]

        results = []
        num_agents_processed_since_throttle_check = 0
        _batch_t0 = time.perf_counter()
        _agent_times_ms = []
        num_items = len(self.benchmark_items)
        total_data_kb = sum(len(it) for it in self.benchmark_items if it) / 1024
        self.logger.info(f"Starting batch evaluation for population of {len(population)} agents. ({num_items} items, {total_data_kb:.0f}KB total benchmark data)")
        if _is_debug():
            print(f"DEBUG-TIMING: [batch_start] {len(population)} agents x {num_items} items ({total_data_kb:.0f}KB)")
        for agent_idx, evolving_agent_instance in enumerate(population):
            if gui_stop_event and gui_stop_event.is_set():
                self.logger.info(f"Population evaluation stopped by GUI at agent {agent_idx}/{len(population)}.")
                results.extend([(EVALUATION_FAIL_REWARD, {"notes": "EvaluationInterrupted"}) for _ in
                                range(len(population) - agent_idx)])
                break

            # Agent throttling - keep GUI responsive during heavy loads
            if num_agents_processed_since_throttle_check >= self.agents_per_throttle_check:
                if self.throttle_sleep_duration > 0:
                    time.sleep(self.throttle_sleep_duration)
                num_agents_processed_since_throttle_check = 0

            if not (hasattr(evolving_agent_instance, 'puffin_ai') and evolving_agent_instance.puffin_ai is not None):
                self.logger.error(
                    f"Agent {getattr(evolving_agent_instance, 'agent_id', f'Idx_{agent_idx}')} missing PuffinZipAI core. Assigning fail reward.")
                results.append((EVALUATION_FAIL_REWARD, {"notes": "MissingCoreAI"}))
                continue

            _agent_t0 = time.perf_counter()
            agent_fitness, agent_stats_dict = self.evaluate_agent_fitness(evolving_agent_instance.puffin_ai,
                                                                          repetitions=repetitions_per_item,
                                                                          gui_stop_event=gui_stop_event)
            _agent_ms = (time.perf_counter() - _agent_t0) * 1000
            _agent_times_ms.append(_agent_ms)
            results.append((agent_fitness, agent_stats_dict))
            num_agents_processed_since_throttle_check += 1

            # Per-agent timing (log every agent for visibility)
            agent_id_log = getattr(evolving_agent_instance, 'agent_id', f'Idx_{agent_idx}')
            if _is_debug():
                print(f"DEBUG-TIMING: [agent_{agent_idx}] {agent_id_log} -> {_agent_ms:.0f}ms  fit={agent_fitness:.4f}  proc_ms={agent_stats_dict.get('total_processing_time_ms', 0):.0f}ms")

        _batch_total = (time.perf_counter() - _batch_t0) * 1000
        _avg_agent = sum(_agent_times_ms) / len(_agent_times_ms) if _agent_times_ms else 0
        _max_agent = max(_agent_times_ms) if _agent_times_ms else 0
        _slowest_idx = _agent_times_ms.index(_max_agent) if _agent_times_ms else -1
        if _is_debug():
            print(f"DEBUG-TIMING: [batch_done] {_batch_total:.0f}ms total | {len(results)} agents | avg={_avg_agent:.0f}ms | max={_max_agent:.0f}ms (agent {_slowest_idx})")
        self.logger.info(f"Batch evaluation finished. Processed {len(results)} agent results out of {len(population)}.")
        return results

    # --- GPU+CPU Pipelined Evaluation -------------------------------------------
    def evaluate_population_pipelined(self, population, cpu_pool, gui_stop_event=None):
        """GPU-batched inference → CPU-parallel compression pipeline.

        **Phase 1 (GPU):**  For each agent, batch ALL benchmark items into a
        single forward pass (``batch_choose_actions``).
        **Phase 2 (CPU):**  Submit per-item compression jobs to *cpu_pool*
        (``ProcessPoolExecutor``).  Because these run in child processes they
        bypass the GIL for true multi-core parallelism.
        GPU work for agent N+1 overlaps with CPU work for agent N.
        **Phase 3 (GPU):**  Collect rewards, push to replay buffer, run batch
        DQN training steps.

        Returns
        -------
        list[tuple[float, dict]]
            ``(fitness, eval_stats)`` per agent, same format as
            :meth:`evaluate_population_batch`.
        """
        if not self.benchmark_items:
            return [
                (EVALUATION_FAIL_REWARD, {"notes": "NoBenchmarkData"})
                for _ in population
            ]

        # Ensure baseline is computed (cached across agents)
        if (
            not hasattr(self, "_cached_baseline_reward")
            or self._cached_baseline_generation != id(self.benchmark_items)
        ):
            bl_total, bl_count = 0.0, 0
            for it in self.benchmark_items:
                if not it:
                    continue
                try:
                    bc = _rle_compress_func(it, method="simple", min_run_len_override=6)
                    bd = _rle_decompress_func(bc, method="simple", min_run_len_override=6)
                    bl_total += _calculate_reward_func(it, bc, bd, "RLE", 0.5, None)
                    bl_count += 1
                except Exception:
                    pass
            self._cached_baseline_reward = bl_total / max(bl_count, 1)
            self._cached_baseline_generation = id(self.benchmark_items)
        baseline = self._cached_baseline_reward
        num_items = len(self.benchmark_items)

        # ── Phase 1 + 2: GPU batch-infer → CPU submit (pipelined) ──
        pending = []  # (agent, ai, action_results, futures, novel_idxs, reference_idxs)
        for aidx, agent in enumerate(population):
            if gui_stop_event and gui_stop_event.is_set():
                remaining = len(population) - len(pending)
                pad = [(EVALUATION_FAIL_REWARD, {"notes": "Interrupted"})] * remaining
                # Still need to collect already-submitted agents below
                break

            if not (hasattr(agent, "puffin_ai") and agent.puffin_ai):
                pending.append((agent, None, None, None, None, None))
                continue

            ai = agent.puffin_ai
            rle_min_run = getattr(ai, "rle_min_encodable_run_length", 2)

            # Phase 1 (GPU): batched forward pass
            if hasattr(ai, "batch_choose_actions"):
                action_results = ai.batch_choose_actions(
                    self.benchmark_items, use_exploration=False
                )
            else:
                # Fallback for tabular agents
                action_results = []
                for item in self.benchmark_items:
                    s = ai._get_state_representation(item)
                    a = ai._choose_action(s, use_exploration=False)
                    action_results.append((a, None))

            # Phase 2 (CPU): submit compression jobs — overlap with next GPU batch
            novel_idxs = []
            reference_idxs = []
            futures = []
            for i in range(num_items):
                act_idx = action_results[i][0]
                act_name = ai.action_names.get(act_idx, "NoCompression")
                if act_name == "NovelMethod":
                    novel_idxs.append(i)
                    futures.append(None)  # handled in-process later
                elif act_name == "ReferenceMethod":
                    reference_idxs.append(i)
                    futures.append(None)  # handled in-process later (closures not picklable)
                else:
                    futures.append(
                        cpu_pool.submit(
                            _compress_single_item, (i, act_name, rle_min_run)
                        )
                    )
            pending.append((agent, ai, action_results, futures, novel_idxs, reference_idxs))

        # ── Phase 3: collect results + batch Q-learning ──
        all_results = []
        for agent, ai, action_results, futures, novel_idxs, reference_idxs in pending:
            if ai is None:
                all_results.append(
                    (EVALUATION_FAIL_REWARD, {"notes": "MissingCoreAI"})
                )
                continue

            stats = {
                "total_reward": 0.0,
                "items_evaluated": 0,
                "successful_rle": 0,
                "rle_expansion": 0,
                "rle_no_change": 0,
                "chose_nocompression": 0,
                "chose_adv_rle": 0,
                "chose_novel_method": 0,
                "chose_reference_method": 0,
                "sum_compression_ratios_rle_success": 0.0,
                "sum_expansion_ratios_rle_fail": 0.0,
                "decomp_failures_mismatch": 0,
                "rle_errors_returned": 0,
                "total_processing_time_ms": 0.0,
                "total_original_bytes": 0,
                "total_compressed_bytes": 0,
                # Per-method compression tracking (must match sequential path)
                "method_bytes_saved": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                "method_attempts": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
                "method_successes": {"RLE": 0, "AdvancedRLE": 0, "NovelMethod": 0, "ReferenceMethod": 0},
            }

            item_results: list[Any] = [None] * num_items

            # Collect ProcessPoolExecutor futures
            for i in range(num_items):
                if futures[i] is None:
                    continue
                try:
                    item_results[i] = futures[i].result(timeout=120)
                except Exception:
                    item_results[i] = {
                        "fallback": False,
                        "item_idx": i,
                        "reward": EVALUATION_FAIL_REWARD,
                        "original_size": len(self.benchmark_items[i]),
                        "compressed_size": len(self.benchmark_items[i]),
                        "was_success": False,
                        "decompression_ok": False,
                        "rle_error": True,
                        "action_name": "Unknown",
                        "proc_ms": 0.0,
                    }

            # Handle NovelMethod items in main process (closures not picklable)
            for i in novel_idxs:
                item_text = self.benchmark_items[i]
                novel_c = getattr(ai, "_novel_compress_fn", None)
                novel_d = getattr(ai, "_novel_decompress_fn", None)
                if novel_c and novel_d:
                    try:
                        _t0 = time.perf_counter_ns()
                        comp = novel_c(item_text)
                        decomp = novel_d(comp)
                        ms = (time.perf_counter_ns() - _t0) / 1_000_000
                        osiz = len(item_text)
                        csiz = len(comp) if comp else 0
                        ok = decomp == item_text
                        ws = csiz < osiz and ok
                        rw = _calculate_reward_func(
                            item_text, comp, decomp, "NovelMethod", ms, None
                        )
                        if _calculate_size_scaled_reward_func and osiz > 0:
                            rw = _calculate_size_scaled_reward_func(
                                rw, osiz, csiz, ws
                            )
                        item_results[i] = {
                            "fallback": False,
                            "item_idx": i,
                            "reward": rw,
                            "original_size": osiz,
                            "compressed_size": csiz,
                            "was_success": ws,
                            "decompression_ok": ok,
                            "rle_error": False,
                            "action_name": "NovelMethod",
                            "proc_ms": ms,
                        }
                    except Exception:
                        item_results[i] = {
                            "fallback": False,
                            "item_idx": i,
                            "reward": EVALUATION_FAIL_REWARD,
                            "original_size": len(item_text),
                            "compressed_size": len(item_text),
                            "was_success": False,
                            "decompression_ok": False,
                            "rle_error": True,
                            "action_name": "NovelMethod",
                            "proc_ms": 0.0,
                        }
                else:
                    # No novel funcs attached — treat as no-compression
                    item_results[i] = {
                        "fallback": False,
                        "item_idx": i,
                        "reward": EVALUATION_FAIL_REWARD * 0.5,
                        "original_size": len(self.benchmark_items[i]),
                        "compressed_size": len(self.benchmark_items[i]),
                        "was_success": False,
                        "decompression_ok": True,
                        "rle_error": False,
                        "action_name": "NovelMethod",
                        "proc_ms": 0.0,
                    }

            # Handle ReferenceMethod items in main process (closures not picklable)
            for i in reference_idxs:
                item_text = self.benchmark_items[i]
                ref_c = getattr(ai, "_reference_compress_fn", None)
                ref_d = getattr(ai, "_reference_decompress_fn", None)
                if ref_c and ref_d:
                    try:
                        item_bytes = item_text.encode('utf-8') if isinstance(item_text, str) else item_text
                        _t0 = time.perf_counter_ns()
                        comp_bytes = ref_c(item_bytes)
                        decomp_result = ref_d(comp_bytes)
                        ms = (time.perf_counter_ns() - _t0) / 1_000_000
                        decomp_text = decomp_result.decode('utf-8') if isinstance(decomp_result, bytes) else decomp_result
                        osiz = len(item_text)
                        csiz = len(comp_bytes) if comp_bytes else 0
                        ok = decomp_text == item_text
                        ws = csiz < osiz and ok
                        # Use placeholder string of compressed length for reward
                        comp_placeholder = "X" * csiz
                        rw = _calculate_reward_func(
                            item_text, comp_placeholder, decomp_text, "ReferenceMethod", ms, None
                        )
                        if _calculate_size_scaled_reward_func and osiz > 0:
                            rw = _calculate_size_scaled_reward_func(
                                rw, osiz, csiz, ws
                            )
                        item_results[i] = {
                            "fallback": False,
                            "item_idx": i,
                            "reward": rw,
                            "original_size": osiz,
                            "compressed_size": csiz,
                            "was_success": ws,
                            "decompression_ok": ok,
                            "rle_error": False,
                            "action_name": "ReferenceMethod",
                            "proc_ms": ms,
                        }
                    except Exception:
                        item_results[i] = {
                            "fallback": False,
                            "item_idx": i,
                            "reward": EVALUATION_FAIL_REWARD,
                            "original_size": len(item_text),
                            "compressed_size": len(item_text),
                            "was_success": False,
                            "decompression_ok": False,
                            "rle_error": True,
                            "action_name": "ReferenceMethod",
                            "proc_ms": 0.0,
                        }
                else:
                    # No reference funcs attached — treat as no-compression
                    item_results[i] = {
                        "fallback": False,
                        "item_idx": i,
                        "reward": EVALUATION_FAIL_REWARD * 0.5,
                        "original_size": len(self.benchmark_items[i]),
                        "compressed_size": len(self.benchmark_items[i]),
                        "was_success": False,
                        "decompression_ok": True,
                        "rle_error": False,
                        "action_name": "ReferenceMethod",
                        "proc_ms": 0.0,
                    }

            # Aggregate stats + collect Q-learning experiences
            total_reward = 0.0
            experiences = []
            for i, r in enumerate(item_results):
                if r is None or r.get("fallback"):
                    continue
                stats["items_evaluated"] += 1
                total_reward += r["reward"]
                stats["total_processing_time_ms"] += r.get("proc_ms", 0)
                stats["total_original_bytes"] += r["original_size"]
                stats["total_compressed_bytes"] += r["compressed_size"]

                an = r.get("action_name", "")

                # --- Per-method compression tracking ---
                if an in ("RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"):
                    stats["method_attempts"][an] = stats["method_attempts"].get(an, 0) + 1
                    _bytes_saved = r["original_size"] - r["compressed_size"]
                    stats["method_bytes_saved"][an] = stats["method_bytes_saved"].get(an, 0) + _bytes_saved
                    if r["was_success"]:
                        stats["method_successes"][an] = stats["method_successes"].get(an, 0) + 1

                if an == "NoCompression":
                    stats["chose_nocompression"] += 1
                elif an == "AdvancedRLE":
                    stats["chose_adv_rle"] += 1
                elif an == "NovelMethod":
                    stats["chose_novel_method"] += 1
                elif an == "ReferenceMethod":
                    stats["chose_reference_method"] += 1

                if r["was_success"]:
                    stats["successful_rle"] += 1
                    if r["compressed_size"] > 0:
                        stats["sum_compression_ratios_rle_success"] += (
                            r["original_size"] / r["compressed_size"]
                        )
                elif an in ("RLE", "AdvancedRLE", "NovelMethod", "ReferenceMethod"):
                    if r["compressed_size"] > r["original_size"]:
                        stats["rle_expansion"] += 1
                        if r["compressed_size"] > 0:
                            stats["sum_expansion_ratios_rle_fail"] += (
                                r["original_size"] / r["compressed_size"]
                            )
                    elif r["compressed_size"] == r["original_size"]:
                        stats["rle_no_change"] += 1

                if r.get("rle_error"):
                    stats["rle_errors_returned"] += 1
                elif (
                    not r.get("decompression_ok", True)
                    and an != "NoCompression"
                ):
                    stats["decomp_failures_mismatch"] += 1

                # Collect experience for batch Q-learning
                feat = action_results[i][1] if len(action_results[i]) > 1 else None
                if feat is not None:
                    experiences.append((feat, action_results[i][0], r["reward"]))

            # Phase 3b: batch Q-learning on GPU
            if hasattr(ai, "batch_push_experiences") and experiences:
                ai.batch_push_experiences(experiences)

            # Fitness computation (mirrors evaluate_agent_fitness)
            n = max(stats["items_evaluated"], 1)
            raw_fit = total_reward / n
            fit = raw_fit - baseline

            rle_count = (
                n
                - stats["chose_nocompression"]
                - stats["chose_adv_rle"]
                - stats["chose_novel_method"]
                - stats["chose_reference_method"]
            )
            mc = {
                "RLE": max(0, rle_count),
                "NoCompression": stats["chose_nocompression"],
                "AdvancedRLE": stats["chose_adv_rle"],
                "NovelMethod": stats["chose_novel_method"],
                "ReferenceMethod": stats["chose_reference_method"],
            }
            div_adj = 0.0
            if _calculate_method_diversity_adjustment_func and n > 0:
                div_adj = _calculate_method_diversity_adjustment_func(mc, n)
            fit += div_adj

            stats["total_reward"] = total_reward
            stats["method_profile"] = (
                {k: v / n for k, v in mc.items()} if n > 0 else {}
            )
            stats["diversity_adjustment"] = div_adj

            all_results.append((fit, stats))

        return all_results
