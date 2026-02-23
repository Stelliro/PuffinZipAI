# PuffinZipAI_Project/puffinzip_ai/reward_system.py
"""
**Hyper-Complex Adaptive Reward & Punishment System for PuffinZipAI.**

Multi-signal reward calculation with:
    * **Compression ratio reward** — primary signal scaled by space saved
    * **Size-scaled rewards** — larger files successfully compressed earn more
    * **Speed bonuses/penalties** — fast compression is rewarded, slow is punished
    * **Data-awareness rewards** — choosing the RIGHT method for data type
    * **Consistency bonuses** — maintaining compression quality over time
    * **Exploration incentives** — discovering useful new compression strategies
    * **Method diversity scoring** — punishes monotony, rewards variety
    * **Population novelty scoring** — rewards unique strategies in the population
    * **Generation repetition penalties** — prevents population-wide stagnation
    * **Progressive difficulty scaling** — adapts reward magnitude to training stage
    * **Decompression verification** — strict penalties for lossy compression
    * **Competitive benchmarking rewards** — bonus for beating baseline methods
"""

from typing import Optional
import math

# ==========================================================================
# PRIMARY REWARD/PENALTY CONSTANTS
# ==========================================================================

# --- Core penalties ---
PENALTY_MISMATCH = -10.0                    # Decompression produced wrong output
PENALTY_RLE_PROCESSING_ERROR = -12.0        # RLE internal error during processing
PENALTY_FOR_USELESS_RLE_ATTEMPT = -0.2      # RLE chosen but no compression achieved
PENALTY_SCALER_EXPANSION = 3.0              # Multiplier for data expansion penalty
PENALTY_CATASTROPHIC_EXPANSION = -8.0       # Data expanded by >50%
PENALTY_WRONG_METHOD_FOR_DATA = -1.5        # Chose RLE for random data, etc.
PENALTY_REPEATED_FAILURE = -2.0             # Same method failing repeatedly
PENALTY_REFERENCE_METHOD_BASE = -0.5        # Base penalty when using a reference (scaffolded) method

# --- Core rewards ---
REWARD_SCALER_COMPRESSION_SUCCESS = 10.0    # Base reward multiplier for compression
REWARD_EMPTY_INPUT_HANDLED_CORRECTLY = 0.1  # Empty input handled gracefully
BASE_REWARD_NOCOMPRESSION_CORRECT = 0.0     # NoCompression is neutral by default
REWARD_EXCEPTIONAL_COMPRESSION = 5.0        # Bonus for >50% compression ratio
REWARD_SPEED_BONUS = 0.5                    # Bonus for fast compression
REWARD_CORRECT_METHOD_CHOICE = 1.0          # Bonus for choosing optimal method
REWARD_BEAT_BASELINE = 3.0                  # Bonus for beating gzip/bz2/lzma
REWARD_CONSISTENCY_BONUS = 0.5              # Bonus for consistently good results

# --- Speed thresholds ---
MAX_TIME_MS_KB_PENALTY_THRESHOLD = 150.0    # ms/KB above which penalty applies
TIME_PENALTY_VALUE = -0.3                   # Base time penalty
SPEED_BONUS_MS_KB_THRESHOLD = 20.0          # ms/KB below which speed bonus applies

# --- FILE SIZE SCALING ---
SIZE_BONUS_BASE_THRESHOLD_BYTES = 32 * 1024     # 32KB — files below get no size bonus
SIZE_BONUS_MAX_MULTIPLIER = 5.0                 # Max additional multiplier for large files
SIZE_BONUS_SCALING_FACTOR = 0.5                 # Controls how fast bonus grows (log scale)

# --- PROGRESSIVE DIFFICULTY SCALING ---
# Rewards scale based on the agent's generation/experience level
PROGRESSIVE_BASE_GENERATION = 0             # Starting generation (full reward)
PROGRESSIVE_MATURE_GENERATION = 50          # Generation where expectations ramp up
PROGRESSIVE_EXPERT_GENERATION = 200         # Generation where max expectations apply
PROGRESSIVE_EXPERT_NOCOMP_PENALTY = -0.5    # Expert agents penalised for NoCompression
PROGRESSIVE_EXPERT_EXPANSION_MULT = 2.0     # Expert agents penalised more for expansion

# --- CONSISTENCY TRACKING ---
CONSISTENCY_WINDOW = 20                     # Number of recent results to track
CONSISTENCY_HIGH_THRESHOLD = 0.7            # Fraction of recent successes for bonus
CONSISTENCY_LOW_THRESHOLD = 0.3             # Below this = penalty for inconsistency
CONSISTENCY_PENALTY = -0.5                  # Penalty for inconsistent performance

# --- NOVELTY & DIVERSITY SCORING ---
METHOD_MONOTONY_PENALTY_SCALER = -2.0       # Scales with method dominance
METHOD_DIVERSITY_BONUS = 1.5                # Bonus for diverse method usage
POPULATION_CONFORMITY_PENALTY = -1.0        # Penalty for being too similar to population
POPULATION_NOVELTY_BONUS = 2.0              # Bonus for unique strategy
GENERATION_HISTORY_DECAY = 0.85             # Decay for method history across generations
GENERATION_REPETITION_MAX_PENALTY = -3.0    # Max penalty from generation repetition

# --- DATA-AWARE REWARD SIGNALS ---
# Thresholds for determining data type characteristics
HIGH_ENTROPY_THRESHOLD = 0.85               # Byte entropy above this = random/encrypted
LOW_ENTROPY_THRESHOLD = 0.3                 # Byte entropy below this = very compressible
HIGH_RUN_RATIO_THRESHOLD = 0.2              # Max-run/len above this = RLE-friendly


# ==========================================================================
# CORE REWARD CALCULATION
# ==========================================================================

def calculate_reward(
        original_text: str,
        compressed_text: str,
        decompressed_text: str,
        action_taken: str,
        processing_time_ms: float,
        rle_error_code: Optional[str] = None,
        generation: int = 0,
        recent_results: Optional[list] = None,
) -> float:
    """Calculate the multi-signal reward for a compression action.

    This is the primary reward function combining:
    1. Compression ratio (primary signal)
    2. Speed bonus/penalty
    3. Progressive difficulty scaling
    4. Data-awareness adjustments
    5. Consistency tracking

    Parameters
    ----------
    original_text : str
        The original uncompressed text.
    compressed_text : str
        The text after compression.
    decompressed_text : str
        The text after decompression (should match original).
    action_taken : str
        Name of the compression method used.
    processing_time_ms : float
        Time taken for compress+decompress in milliseconds.
    rle_error_code : str, optional
        Error code from RLE processing (None = no error).
    generation : int
        Current evolution generation (for progressive scaling).
    recent_results : list, optional
        List of recent (success: bool) results for consistency tracking.

    Returns
    -------
    float
        The calculated reward value.
    """
    original_size = len(original_text)
    compressed_size = len(compressed_text)
    reward = 0.0

    # --- Handle empty input ---
    if original_size == 0:
        if action_taken in ["RLE", "AdvancedRLE", "NovelMethod"]:
            return REWARD_EMPTY_INPUT_HANDLED_CORRECTLY if (
                compressed_text == "" and decompressed_text == "" and rle_error_code is None
            ) else PENALTY_MISMATCH
        elif action_taken == "NoCompression":
            return REWARD_EMPTY_INPUT_HANDLED_CORRECTLY if (
                compressed_text == "" and decompressed_text == ""
            ) else PENALTY_MISMATCH
        else:
            return PENALTY_MISMATCH

    # --- Compression method evaluation ---
    if action_taken in ["RLE", "AdvancedRLE", "NovelMethod"]:
        # Check for processing errors
        if rle_error_code is not None and rle_error_code in _get_rle_errors():
            return PENALTY_RLE_PROCESSING_ERROR

        # CRITICAL: Decompression must be lossless
        if decompressed_text != original_text:
            return PENALTY_MISMATCH

        if compressed_size < original_size:
            # SUCCESS: Compression achieved
            space_saved_ratio = (original_size - compressed_size) / original_size
            reward = REWARD_SCALER_COMPRESSION_SUCCESS * space_saved_ratio

            # Exceptional compression bonus (>50% reduction)
            if space_saved_ratio > 0.5:
                reward += REWARD_EXCEPTIONAL_COMPRESSION * (space_saved_ratio - 0.5) * 2.0

            # Tiered compression quality bonuses
            if space_saved_ratio > 0.8:
                reward += 3.0  # Outstanding compression
            elif space_saved_ratio > 0.6:
                reward += 1.5  # Excellent compression

        elif compressed_size > original_size:
            # FAILURE: Data expanded — this is bad
            expansion_factor = compressed_size / original_size
            reward = -PENALTY_SCALER_EXPANSION * (expansion_factor - 1)

            # Catastrophic expansion penalty (>50% bigger)
            if expansion_factor > 1.5:
                reward += PENALTY_CATASTROPHIC_EXPANSION

        else:
            # No change — useless attempt, mild penalty
            reward = PENALTY_FOR_USELESS_RLE_ATTEMPT

    elif action_taken == "NoCompression":
        # Verify correctness
        if compressed_text == original_text and decompressed_text == original_text:
            reward = BASE_REWARD_NOCOMPRESSION_CORRECT

            # Data-awareness: if data is truly random, NoCompression is smart
            entropy = _quick_entropy(original_text)
            if entropy > HIGH_ENTROPY_THRESHOLD:
                reward += REWARD_CORRECT_METHOD_CHOICE * 0.5  # Smart choice for random data
            elif entropy < LOW_ENTROPY_THRESHOLD:
                # Missed opportunity — this data was very compressible!
                reward += PENALTY_WRONG_METHOD_FOR_DATA * 0.5
        else:
            return PENALTY_MISMATCH * 1.5
    else:
        reward = -20.0  # Unknown action

    # --- Speed bonus/penalty ---
    if reward > PENALTY_MISMATCH and original_size > 0:
        kb_size = original_size / 1024.0
        if kb_size > 0.001:
            time_ms_per_kb = processing_time_ms / kb_size

            # Speed penalty for slow compression
            if time_ms_per_kb > MAX_TIME_MS_KB_PENALTY_THRESHOLD:
                slowness = min((time_ms_per_kb - MAX_TIME_MS_KB_PENALTY_THRESHOLD) / 
                              MAX_TIME_MS_KB_PENALTY_THRESHOLD, 3.0)
                reward += TIME_PENALTY_VALUE * (1.0 + slowness * 0.5)

            # Speed bonus for fast compression
            elif time_ms_per_kb < SPEED_BONUS_MS_KB_THRESHOLD and compressed_size < original_size:
                speedup = 1.0 - (time_ms_per_kb / SPEED_BONUS_MS_KB_THRESHOLD)
                reward += REWARD_SPEED_BONUS * speedup

        elif processing_time_ms > (MAX_TIME_MS_KB_PENALTY_THRESHOLD * 0.05):
            reward += TIME_PENALTY_VALUE / 2

    # --- Progressive difficulty scaling ---
    if generation > 0:
        reward = _apply_progressive_scaling(reward, generation, action_taken, 
                                            compressed_size < original_size if original_size > 0 else False)

    # --- Consistency tracking ---
    if recent_results is not None and len(recent_results) >= 5:
        consistency_adj = _calculate_consistency_adjustment(recent_results)
        reward += consistency_adj

    return reward


# ==========================================================================
# PROGRESSIVE DIFFICULTY SCALING
# ==========================================================================

def _apply_progressive_scaling(
    reward: float,
    generation: int,
    action_taken: str,
    was_successful: bool,
) -> float:
    """Scale reward based on the agent's evolution generation.

    Early generations get full reward for any success.
    Later generations are expected to perform better — NoCompression
    becomes penalised, expansion penalties increase.
    """
    # Maturity factor: 0.0 (newborn) → 1.0 (expert)
    if generation <= PROGRESSIVE_BASE_GENERATION:
        maturity = 0.0
    elif generation >= PROGRESSIVE_EXPERT_GENERATION:
        maturity = 1.0
    else:
        maturity = (generation - PROGRESSIVE_BASE_GENERATION) / (
            PROGRESSIVE_EXPERT_GENERATION - PROGRESSIVE_BASE_GENERATION
        )

    # Mature agents: penalise lazy NoCompression choices more
    if action_taken == "NoCompression" and maturity > 0.3:
        reward += PROGRESSIVE_EXPERT_NOCOMP_PENALTY * maturity

    # Mature agents: harsher expansion penalties
    if reward < 0 and not was_successful and maturity > 0.5:
        reward *= (1.0 + (PROGRESSIVE_EXPERT_EXPANSION_MULT - 1.0) * maturity)

    # Slight reward boost for early success (encourage learning)
    if was_successful and maturity < 0.3:
        reward *= (1.0 + 0.5 * (1.0 - maturity / 0.3))

    return reward


# ==========================================================================
# SIZE-SCALED REWARDS
# ==========================================================================

def calculate_size_scaled_reward(
    base_reward: float,
    original_size_bytes: int,
    compressed_size_bytes: int,
    was_successful_compression: bool
) -> float:
    """Scale the reward based on the size of the file being compressed.

    Larger files that are successfully compressed get a multiplicative bonus.
    Larger files that FAIL to compress get a harsher penalty.

    Args:
        base_reward: the raw reward from calculate_reward()
        original_size_bytes: size of the original uncompressed data
        compressed_size_bytes: size after compression
        was_successful_compression: True if compression reduced size + verified

    Returns:
        float: the size-adjusted reward
    """
    if original_size_bytes <= 0:
        return base_reward

    # Size factor using log scale
    if original_size_bytes <= SIZE_BONUS_BASE_THRESHOLD_BYTES:
        size_factor = 1.0
    else:
        log_ratio = math.log2(original_size_bytes / SIZE_BONUS_BASE_THRESHOLD_BYTES)
        size_factor = 1.0 + min(
            SIZE_BONUS_MAX_MULTIPLIER - 1.0,
            log_ratio * SIZE_BONUS_SCALING_FACTOR
        )

    if was_successful_compression:
        # Bonus: larger files compressed well = much higher reward
        compression_ratio = 1.0 - (compressed_size_bytes / original_size_bytes)
        ratio_bonus = 1.0 + compression_ratio * (size_factor - 1.0)
        return base_reward * size_factor * ratio_bonus
    elif base_reward < 0:
        # Penalty: failing on larger files is worse (capped at 2x)
        penalty_factor = min(2.0, size_factor)
        return base_reward * penalty_factor
    else:
        # Neutral — mild size influence
        return base_reward * size_factor


# ==========================================================================
# COMPETITIVE BENCHMARKING REWARD
# ==========================================================================

def calculate_competitive_reward(
    base_reward: float,
    original_size: int,
    ai_compressed_size: int,
    baseline_compressed_size: int,
    baseline_method: str = "gzip",
) -> float:
    """Calculate bonus/penalty for performance relative to baseline methods.

    Parameters
    ----------
    base_reward : float
        The raw reward from calculate_reward().
    original_size : int
        Size of original data.
    ai_compressed_size : int
        AI's compressed size.
    baseline_compressed_size : int
        Best baseline method's compressed size.
    baseline_method : str
        Name of the baseline method for logging.

    Returns
    -------
    float
        Adjusted reward with competitive bonus/penalty.
    """
    if original_size <= 0 or ai_compressed_size <= 0:
        return base_reward

    ai_ratio = 1.0 - (ai_compressed_size / original_size)
    baseline_ratio = 1.0 - (baseline_compressed_size / original_size)
    improvement = ai_ratio - baseline_ratio

    if improvement > 0:
        # Beat the baseline! Scale bonus by margin
        bonus = REWARD_BEAT_BASELINE * min(improvement / 0.1, 3.0)
        return base_reward + bonus
    elif improvement > -0.05:
        # Within 5% of baseline — acceptable, small bonus
        return base_reward + 0.5
    else:
        # Significantly worse than baseline — mild penalty to learn
        return base_reward - 0.5 * min(abs(improvement) / 0.1, 2.0)


# ==========================================================================
# DATA-AWARENESS REWARD ADJUSTMENTS
# ==========================================================================

def calculate_data_aware_adjustment(
    action_taken: str,
    original_text: str,
    was_successful: bool,
) -> float:
    """Calculate reward adjustment based on how well the chosen method
    matches the data characteristics.

    Parameters
    ----------
    action_taken : str
        The compression method chosen.
    original_text : str
        The original text data.
    was_successful : bool
        Whether compression reduced size.

    Returns
    -------
    float
        Adjustment to add to the base reward.
    """
    if not original_text:
        return 0.0

    n = len(original_text)
    entropy = _quick_entropy(original_text)
    max_run_ratio = _quick_max_run_ratio(original_text)
    adjustment = 0.0

    if action_taken in ["RLE", "AdvancedRLE"]:
        if max_run_ratio > HIGH_RUN_RATIO_THRESHOLD and was_successful:
            # RLE on run-heavy data = smart choice
            adjustment += REWARD_CORRECT_METHOD_CHOICE
        elif entropy > HIGH_ENTROPY_THRESHOLD and not was_successful:
            # RLE on random data = predictably bad, but penalise the choice
            adjustment += PENALTY_WRONG_METHOD_FOR_DATA
        elif max_run_ratio < 0.02 and not was_successful:
            # No runs at all — should have known better
            adjustment += PENALTY_WRONG_METHOD_FOR_DATA * 0.7

    elif action_taken == "NovelMethod":
        if was_successful:
            # Novel methods that work get extra exploration reward
            adjustment += REWARD_CORRECT_METHOD_CHOICE * 1.5
        elif entropy > HIGH_ENTROPY_THRESHOLD:
            # Random data defeats all methods — no penalty
            pass
        else:
            # Mild penalty — novel method should be better by now
            adjustment += PENALTY_WRONG_METHOD_FOR_DATA * 0.3

    elif action_taken == "NoCompression":
        if entropy > HIGH_ENTROPY_THRESHOLD:
            # Smart: random data can't be compressed
            adjustment += REWARD_CORRECT_METHOD_CHOICE * 0.3
        elif entropy < LOW_ENTROPY_THRESHOLD:
            # Missed opportunity! This data was very compressible
            adjustment += PENALTY_WRONG_METHOD_FOR_DATA * 0.8

    return adjustment


# ==========================================================================
# CONSISTENCY TRACKING
# ==========================================================================

def _calculate_consistency_adjustment(recent_results: list) -> float:
    """Calculate reward adjustment based on recent performance consistency.

    Args:
        recent_results: list of bool (True = successful compression)

    Returns:
        float: adjustment value
    """
    if not recent_results:
        return 0.0

    window = recent_results[-CONSISTENCY_WINDOW:]
    success_rate = sum(1 for r in window if r) / max(len(window), 1)

    if success_rate >= CONSISTENCY_HIGH_THRESHOLD:
        return CONSISTENCY_BONUS * (success_rate - CONSISTENCY_HIGH_THRESHOLD) / (
            1.0 - CONSISTENCY_HIGH_THRESHOLD
        )
    elif success_rate <= CONSISTENCY_LOW_THRESHOLD:
        return CONSISTENCY_PENALTY * (CONSISTENCY_LOW_THRESHOLD - success_rate) / CONSISTENCY_LOW_THRESHOLD
    
    return 0.0

CONSISTENCY_BONUS = REWARD_CONSISTENCY_BONUS


# ==========================================================================
# METHOD DIVERSITY SCORING
# ==========================================================================

def calculate_method_diversity_adjustment(method_counts: dict, total_items: int) -> float:
    """Calculate a fitness adjustment based on how diverse an agent's method choices are.

    Agents that spam a single method get penalized.
    Agents that use multiple methods effectively get a bonus.

    Args:
        method_counts: dict mapping method name -> count of times chosen
        total_items: total benchmark items evaluated

    Returns:
        float: adjustment to add to fitness (can be negative)
    """
    if total_items <= 0:
        return 0.0

    adjustment = 0.0
    methods_used = {k: v for k, v in method_counts.items() if v > 0}
    num_methods_used = len(methods_used)

    if num_methods_used == 0:
        return 0.0

    # Calculate dominance: what fraction of items used the most common method
    max_count = max(methods_used.values())
    dominance_ratio = max_count / total_items

    # PUNISHMENT: If one method dominates > 70% of choices
    if dominance_ratio > 0.7:
        penalty_strength = (dominance_ratio - 0.7) / 0.3
        adjustment += METHOD_MONOTONY_PENALTY_SCALER * penalty_strength

    # REWARD: Using 3+ methods effectively gets a diversity bonus
    if num_methods_used >= 3:
        entropy = 0.0
        for count in methods_used.values():
            p = count / total_items
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(max(num_methods_used, 2))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        adjustment += METHOD_DIVERSITY_BONUS * normalized_entropy

    # Extra punishment: NoCompression chosen > 50% is lazy
    nocomp_ratio = method_counts.get('NoCompression', 0) / total_items
    if nocomp_ratio > 0.5:
        lazy_penalty = -1.5 * (nocomp_ratio - 0.5) / 0.5
        adjustment += lazy_penalty

    # Bonus for agents that use NovelMethod successfully
    novel_ratio = method_counts.get('NovelMethod', 0) / total_items
    if novel_ratio > 0.1:
        adjustment += 0.5 * min(novel_ratio / 0.3, 1.0)

    return adjustment


# ==========================================================================
# POPULATION NOVELTY SCORING
# ==========================================================================

def calculate_population_novelty_scores(population_method_profiles: list) -> list:
    """Calculate novelty scores for each agent based on strategy uniqueness.

    Args:
        population_method_profiles: list of dicts, each mapping method_name -> usage_fraction

    Returns:
        list of float: novelty adjustment per agent
    """
    if not population_method_profiles or len(population_method_profiles) < 2:
        return [0.0] * len(population_method_profiles) if population_method_profiles else []

    n = len(population_method_profiles)
    all_methods = set()
    for profile in population_method_profiles:
        all_methods.update(profile.keys())

    # Calculate mean strategy profile
    mean_profile = {}
    for method in all_methods:
        mean_profile[method] = sum(p.get(method, 0.0) for p in population_method_profiles) / n

    # Euclidean distance from mean for each agent
    distances = []
    for profile in population_method_profiles:
        dist_sq = 0.0
        for method in all_methods:
            diff = profile.get(method, 0.0) - mean_profile[method]
            dist_sq += diff * diff
        distances.append(math.sqrt(dist_sq))

    # Normalize distances to [0, 1]
    max_dist = max(distances) if distances else 1.0
    if max_dist < 1e-9:
        return [POPULATION_CONFORMITY_PENALTY] * n

    novelty_scores = []
    for dist in distances:
        normalized = dist / max_dist
        if normalized < 0.2:
            score = POPULATION_CONFORMITY_PENALTY * (1.0 - normalized / 0.2)
        else:
            score = POPULATION_NOVELTY_BONUS * ((normalized - 0.2) / 0.8)
        novelty_scores.append(score)

    return novelty_scores


# ==========================================================================
# GENERATION REPETITION PENALTY
# ==========================================================================

def calculate_generation_repetition_penalty(
    current_method_profile: dict,
    generation_method_history: list,
    decay: float = GENERATION_HISTORY_DECAY
) -> float:
    """Calculate penalty for methods overused across previous generations.

    Args:
        current_method_profile: dict mapping method_name -> usage_fraction
        generation_method_history: list of dicts (one per past generation)
        decay: how much older generations matter (0.85 = recent gens matter more)

    Returns:
        float: penalty (negative or zero)
    """
    if not generation_method_history or not current_method_profile:
        return 0.0

    # Build weighted historical usage profile
    historical_usage = {}
    total_weight = 0.0
    for gen_idx, gen_profile in enumerate(reversed(generation_method_history)):
        weight = decay ** gen_idx
        total_weight += weight
        for method, fraction in gen_profile.items():
            historical_usage[method] = historical_usage.get(method, 0.0) + fraction * weight

    if total_weight > 0:
        for method in historical_usage:
            historical_usage[method] /= total_weight

    # Penalty for heavily using historically overused methods
    penalty = 0.0
    for method, current_fraction in current_method_profile.items():
        historical_fraction = historical_usage.get(method, 0.0)
        if historical_fraction > 0.5 and current_fraction > 0.5:
            overlap = current_fraction * historical_fraction
            penalty += overlap

    penalty = max(GENERATION_REPETITION_MAX_PENALTY, -penalty * 2.0)
    return penalty


# ==========================================================================
# COMPOSITE FITNESS CALCULATOR
# ==========================================================================

def calculate_composite_fitness(
    compression_rewards: list,
    method_counts: dict,
    total_items: int,
    generation: int = 0,
    population_method_profiles: Optional[list] = None,
    generation_method_history: Optional[list] = None,
    agent_index: int = 0,
    baseline_results: Optional[dict] = None,
) -> dict:
    """Calculate a comprehensive composite fitness score for an agent.

    Combines:
    1. Mean compression reward
    2. Method diversity adjustment
    3. Population novelty score
    4. Generation repetition penalty
    5. Consistency bonus/penalty
    6. Progressive generation scaling

    Parameters
    ----------
    compression_rewards : list
        List of raw rewards from evaluate_benchmark_item calls.
    method_counts : dict
        Method name -> usage count.
    total_items : int
        Total benchmark items evaluated.
    generation : int
        Current evolution generation.
    population_method_profiles : list, optional
        All agents' method profiles for novelty scoring.
    generation_method_history : list, optional
        Historical method profiles per generation.
    agent_index : int
        Index of this agent in the population.
    baseline_results : dict, optional
        Baseline compression results for competitive scoring.

    Returns
    -------
    dict with keys:
        'fitness': float — the composite fitness score
        'components': dict — breakdown of each fitness component
    """
    if not compression_rewards:
        return {"fitness": 0.0, "components": {}}

    # 1. Mean compression reward (primary signal)
    mean_reward = sum(compression_rewards) / len(compression_rewards)

    # 2. Method diversity
    diversity_adj = calculate_method_diversity_adjustment(method_counts, total_items)

    # 3. Population novelty
    novelty_adj = 0.0
    if population_method_profiles and agent_index < len(population_method_profiles):
        novelty_scores = calculate_population_novelty_scores(population_method_profiles)
        if agent_index < len(novelty_scores):
            novelty_adj = novelty_scores[agent_index]

    # 4. Generation repetition
    gen_rep_penalty = 0.0
    if generation_method_history and method_counts:
        total = max(sum(method_counts.values()), 1)
        current_profile = {k: v / total for k, v in method_counts.items()}
        gen_rep_penalty = calculate_generation_repetition_penalty(
            current_profile, generation_method_history
        )

    # 5. Consistency
    positive_results = [r > 0 for r in compression_rewards]
    consistency_adj = _calculate_consistency_adjustment(positive_results)

    # 6. Progressive scaling factor
    if generation <= PROGRESSIVE_BASE_GENERATION:
        gen_factor = 1.0
    elif generation >= PROGRESSIVE_EXPERT_GENERATION:
        gen_factor = 1.5
    else:
        gen_factor = 1.0 + 0.5 * (generation - PROGRESSIVE_BASE_GENERATION) / (
            PROGRESSIVE_EXPERT_GENERATION - PROGRESSIVE_BASE_GENERATION
        )

    # Composite
    fitness = (
        mean_reward * gen_factor
        + diversity_adj
        + novelty_adj
        + gen_rep_penalty
        + consistency_adj
    )

    components = {
        "mean_reward": mean_reward,
        "diversity_adjustment": diversity_adj,
        "novelty_adjustment": novelty_adj,
        "generation_repetition_penalty": gen_rep_penalty,
        "consistency_adjustment": consistency_adj,
        "generation_factor": gen_factor,
        "total_items": total_items,
        "methods_used": len([v for v in method_counts.values() if v > 0]),
    }

    return {"fitness": fitness, "components": components}


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def _quick_entropy(data: str) -> float:
    """Fast Shannon entropy calculation normalised to [0, 1]."""
    if not data:
        return 0.0
    n = len(data)
    freq = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return min(entropy / 8.0, 1.0)


def _quick_max_run_ratio(data: str) -> float:
    """Fast max run length / total length ratio."""
    if not data:
        return 0.0
    max_run = 1
    current_run = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run / len(data)


def _get_rle_errors():
    """Get the set of known RLE error codes."""
    try:
        from .rle_constants import RLE_DECOMPRESSION_ERRORS
        return RLE_DECOMPRESSION_ERRORS
    except ImportError:
        return {
            "ERROR_INVALID_RLE_FORMAT_NO_COUNT",
            "ERROR_INVALID_RLE_FORMAT_BAD_COUNT",
            "ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT",
            "ERROR_MALFORMED_RLE_STRING",
            "ERROR_COUNT_TOO_LARGE_FOR_SAFETY",
            "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED",
            "ERROR_MEMORY_DURING_CHUNK_ALLOCATION",
            "ERROR_MEMORY_DURING_FINAL_JOIN",
        }


# ==========================================================================
# SCAFFOLDING-AWARE REWARD ADJUSTMENT
# ==========================================================================

def calculate_scaffolded_reward(
    base_reward: float,
    action_taken: str,
    agent_id: str,
    original_text: str,
    compressed_size: int,
    original_size: int,
    generation: int = 0,
    scaffolding_manager=None,
) -> float:
    """Apply scaffolding adjustments when an agent uses a reference method.

    If the agent used a reference method (action_taken == "ReferenceMethod"),
    the base reward is multiplied by a decaying factor based on reliance.
    If the agent used its OWN method, a bonus is added if it beats/matches
    the reference methods.

    Parameters
    ----------
    base_reward : float
        Raw reward from calculate_reward().
    action_taken : str
        The compression method name ("ReferenceMethod" or own method).
    agent_id : str
        Unique identifier for the agent.
    original_text : str
        The original text being compressed.
    compressed_size : int
        Size after compression.
    original_size : int
        Size before compression.
    generation : int
        Current evolution generation.
    scaffolding_manager : ScaffoldingManager, optional
        The scaffolding manager instance. If None, uses global singleton.

    Returns
    -------
    float
        Adjusted reward.
    """
    if scaffolding_manager is None:
        try:
            from .compression_scaffolding import get_scaffolding_manager
            scaffolding_manager = get_scaffolding_manager()
        except ImportError:
            return base_reward

    used_reference = (action_taken == "ReferenceMethod")

    if used_reference:
        # Apply decaying multiplier based on reliance + generation
        multiplier = scaffolding_manager.calculate_reward_multiplier(agent_id, generation)
        adjusted = base_reward * multiplier + PENALTY_REFERENCE_METHOD_BASE
        # Record usage and check for ban trigger
        scaffolding_manager.record_and_check(agent_id, True, "reference", generation)
        return adjusted
    else:
        # Own method — check for bonus if it beats reference
        scaffolding_manager.record_and_check(agent_id, False, "", generation)
        if original_size > 0 and compressed_size < original_size:
            own_ratio = compressed_size / original_size
            bonus = scaffolding_manager.calculate_own_method_bonus(
                agent_id, own_ratio, original_text, generation
            )
            return base_reward + bonus
        return base_reward


# ==========================================================================
# MODULE-LEVEL BACKWARD COMPATIBILITY
# ==========================================================================

try:
    from .rle_constants import RLE_DECOMPRESSION_ERRORS, RLE_ERROR_NO_CHAR
except ImportError:
    RLE_ERROR_NO_CHAR = "ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT"
    RLE_DECOMPRESSION_ERRORS = {
        "ERROR_INVALID_RLE_FORMAT_NO_COUNT",
        "ERROR_INVALID_RLE_FORMAT_BAD_COUNT",
        RLE_ERROR_NO_CHAR,
        "ERROR_MALFORMED_RLE_STRING",
        "ERROR_COUNT_TOO_LARGE_FOR_SAFETY",
        "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED",
        "ERROR_MEMORY_DURING_CHUNK_ALLOCATION",
        "ERROR_MEMORY_DURING_FINAL_JOIN",
    }


# ==========================================================================
# SELF-TEST
# ==========================================================================

if __name__ == '__main__':
    try:
        from .rle_utils import rle_compress
    except ImportError:
        print("Warning: rle_utils not found for reward_system __main__ test.")
        rle_compress = lambda x, **kwargs: x

    print("--- Testing Hyper-Complex Reward System ---")

    reward1 = calculate_reward("AAAAABBBCC", "5A3B2C", "AAAAABBBCC", "RLE", 0.1)
    print(f"Scenario 1 (Good RLE): {reward1:.4f} (Expected ~4.0)")

    reward2 = calculate_reward("ABCDE", "AABBCCDDEE", "ABCDE", "RLE", 0.1)
    print(f"Scenario 2 (RLE Expansion): {reward2:.4f} (Expected negative)")

    reward3 = calculate_reward("AAAAA", "5A", "AAAAB", "RLE", 0.1)
    print(f"Scenario 3 (Mismatch): {reward3:.4f} (Expected {PENALTY_MISMATCH})")

    reward4 = calculate_reward("AAAAA", "5", RLE_ERROR_NO_CHAR, "RLE", 0.1,
                               rle_error_code=RLE_ERROR_NO_CHAR)
    print(f"Scenario 4 (RLE Error): {reward4:.4f} (Expected {PENALTY_RLE_PROCESSING_ERROR})")

    reward5 = calculate_reward("ABCDE", "ABCDE", "ABCDE", "NoCompression", 0.1)
    print(f"Scenario 5 (NoCompression): {reward5:.4f}")

    # Progressive scaling test
    reward6a = calculate_reward("ABCDE", "ABCDE", "ABCDE", "NoCompression", 0.1, generation=0)
    reward6b = calculate_reward("ABCDE", "ABCDE", "ABCDE", "NoCompression", 0.1, generation=200)
    print(f"Scenario 6 (NoComp gen=0): {reward6a:.4f}")
    print(f"Scenario 6 (NoComp gen=200): {reward6b:.4f} (should be more negative)")

    # Diversity test
    div_adj = calculate_method_diversity_adjustment(
        {"RLE": 80, "NoCompression": 10, "AdvancedRLE": 10}, 100
    )
    print(f"Diversity (70% one method): {div_adj:.4f}")

    div_adj2 = calculate_method_diversity_adjustment(
        {"RLE": 30, "NoCompression": 20, "AdvancedRLE": 30, "NovelMethod": 20}, 100
    )
    print(f"Diversity (even spread): {div_adj2:.4f}")

    # Composite fitness
    rewards = [5.0, 3.0, -1.0, 7.0, 4.0, 2.0, 6.0, 1.0, 3.0, 5.0]
    composite = calculate_composite_fitness(
        rewards,
        {"RLE": 5, "NoCompression": 2, "AdvancedRLE": 3},
        10,
        generation=50,
    )
    print(f"Composite fitness: {composite['fitness']:.4f}")
    print(f"Components: {composite['components']}")

    print("\n--- Reward System Test Complete ---")
