# PuffinZipAI_Project/puffinzip_ai/evolution_core/mutation_methods.py
import random
import numpy as np


def mutate_parameter(param_value, min_val, max_val, mutation_strength, is_int):
    if not isinstance(param_value, (int, float)):
        return param_value

    mutation_amount = (random.random() - 0.5) * 2 * mutation_strength

    if is_int:
        effective_range_for_strength = 10
        if max_val is not None and min_val is not None and max_val > min_val:
            effective_range_for_strength = max_val - min_val
        mutated_value = param_value + int(round(mutation_amount * max(1, effective_range_for_strength * 0.1)))
    else:
        mutated_value = param_value + mutation_amount

    if min_val is not None:
        mutated_value = max(min_val, mutated_value)
    if max_val is not None:
        mutated_value = min(max_val, mutated_value)

    return int(round(mutated_value)) if is_int else mutated_value


def mutate_thresholds(thresholds: list, mutation_rate: float,
                      min_thresh_val=1, max_thresh_val=5000,
                      max_thresh_count_config=7, min_thresh_count_config=1,
                      prob_add_remove_thresh=0.1):
    new_thresholds = list(thresholds)

    if random.random() < prob_add_remove_thresh:
        if random.random() < 0.5 and len(new_thresholds) < max_thresh_count_config:
            new_val_base = new_thresholds[-1] if new_thresholds else random.randint(5, 50)
            new_thresh = new_val_base + random.randint(int(new_val_base * 0.05) + 1, int(new_val_base * 0.2) + 20)
            new_thresh = max(min_thresh_val, min(max_thresh_val, new_thresh))
            new_thresholds.append(new_thresh)
        elif len(new_thresholds) > min_thresh_count_config and new_thresholds:
            new_thresholds.pop(random.randrange(len(new_thresholds)))

    for i in range(len(new_thresholds)):
        if random.random() < mutation_rate:
            mutation_strength_thresh = (max_thresh_val - min_thresh_val) * 0.05 * random.random()
            change = int(round((random.random() - 0.5) * 2 * mutation_strength_thresh))
            new_thresholds[i] = max(min_thresh_val, min(max_thresh_val, new_thresholds[i] + change))

    current_thresholds_unique_sorted = sorted(list(set(t for t in new_thresholds if t > 0)))

    while len(current_thresholds_unique_sorted) < min_thresh_count_config:
        fallback_val = current_thresholds_unique_sorted[-1] + random.randint(10,
                                                                             50) if current_thresholds_unique_sorted else random.randint(
            10, 100)
        current_thresholds_unique_sorted.append(max(min_thresh_val, min(max_thresh_val, fallback_val)))
        current_thresholds_unique_sorted = sorted(list(set(current_thresholds_unique_sorted)))

    return current_thresholds_unique_sorted[:max_thresh_count_config]


def apply_mutations(evolving_agent, mutation_rate_config: dict) -> bool:
    ai_core = evolving_agent.puffin_ai
    mutated = False

    base_mutation_rate = mutation_rate_config.get('base_rate', 0.1)
    param_mutation_prob_factor = mutation_rate_config.get('param_factor', 1.0)
    threshold_mutation_prob_factor = mutation_rate_config.get('threshold_factor', 1.0)
    rle_run_mutation_prob_override = mutation_rate_config.get('rle_min_run_prob', 0.15)  # Default from PMA init

    param_configs_local = [
        {'name': 'learning_rate', 'min': 0.00001, 'max': 0.7, 'is_int': False, 'strength': 0.1},
        {'name': 'discount_factor', 'min': 0.5, 'max': 0.99999, 'is_int': False, 'strength': 0.05},
        {'name': 'exploration_rate', 'min': 0.005, 'max': 1.0, 'is_int': False, 'strength': 0.15},
        {'name': 'exploration_decay_rate', 'min': 0.97, 'max': 0.99999, 'is_int': False, 'strength': 0.005},
        {'name': 'min_exploration_rate', 'min': 0.00001, 'max': 0.25, 'is_int': False, 'strength': 0.05},
    ]

    for p_conf in param_configs_local:
        if hasattr(ai_core, p_conf['name']) and random.random() < base_mutation_rate * param_mutation_prob_factor:
            current_val = getattr(ai_core, p_conf['name'])
            mutated_val = mutate_parameter(current_val, p_conf['min'], p_conf['max'], p_conf['strength'],
                                           p_conf['is_int'])
            setattr(ai_core, p_conf['name'], mutated_val)
            mutated = True

    if hasattr(ai_core, 'len_thresholds') and random.random() < base_mutation_rate * threshold_mutation_prob_factor:
        original_thresholds = tuple(ai_core.len_thresholds)
        max_thresh_count = getattr(ai_core, 'MAX_THRESHOLDS_COUNT', 7)
        min_thresh_count = getattr(ai_core, 'MIN_THRESHOLDS_COUNT', 1)

        ai_core.len_thresholds = mutate_thresholds(
            thresholds=ai_core.len_thresholds,
            mutation_rate=base_mutation_rate * 0.5,
            min_thresh_val=1,
            max_thresh_val=5000,
            max_thresh_count_config=max_thresh_count,
            min_thresh_count_config=min_thresh_count,
            prob_add_remove_thresh=0.2
        )
        if tuple(ai_core.len_thresholds) != original_thresholds:
            if hasattr(ai_core, '_reinitialize_state_dependent_vars'):
                # Preserve Q-values across threshold change by interpolating
                old_q = ai_core.q_table.copy() if hasattr(ai_core, 'q_table') and ai_core.q_table is not None else None
                old_state_size = ai_core.state_space_size if hasattr(ai_core, 'state_space_size') else 0
                ai_core._reinitialize_state_dependent_vars()
                # Carry over Q-values for states that still exist
                if old_q is not None and old_state_size > 0:
                    copy_rows = min(old_q.shape[0], ai_core.q_table.shape[0])
                    copy_cols = min(old_q.shape[1], ai_core.q_table.shape[1])
                    ai_core.q_table[:copy_rows, :copy_cols] = old_q[:copy_rows, :copy_cols]
            mutated = True

    rle_min_run_bounds_min_cfg = getattr(ai_core, 'RLE_MIN_RUN_BOUNDS_MIN', 2)
    rle_min_run_bounds_max_cfg = getattr(ai_core, 'RLE_MIN_RUN_BOUNDS_MAX', 7)

    if hasattr(ai_core, 'rle_min_encodable_run_length') and random.random() < rle_run_mutation_prob_override:
        current_rle_min = ai_core.rle_min_encodable_run_length
        change = random.choice([-1, 1]) if random.random() > 0.2 else random.choice([-2, -1, 1, 2])
        new_rle_min = current_rle_min + change
        ai_core.rle_min_encodable_run_length = max(rle_min_run_bounds_min_cfg,
                                                   min(rle_min_run_bounds_max_cfg, new_rle_min))
        if ai_core.rle_min_encodable_run_length != current_rle_min:
            mutated = True

    # --- NOVEL METHOD MUTATION ---
    # Mutate the novel compression pipeline parameters so genetic diversity
    # isn't limited to Q-learning params alone.  Changes are verified for
    # invertibility before being committed — if verification fails, the
    # original method is kept.
    novel_mutation_prob = mutation_rate_config.get('novel_method_mutation_prob', 0.15)
    novel = getattr(ai_core, 'novel_method', None)
    if novel is not None and hasattr(novel, 'metadata') and random.random() < novel_mutation_prob:
        try:
            from ..novel_compression_generator import NovelCompressionGenerator
            _gen = NovelCompressionGenerator()

            meta = dict(novel.metadata)  # shallow copy
            old_pipeline = meta.get('pipeline', 'rle_only')
            old_seed = meta.get('discovery_seed', None)
            old_rle_min = meta.get('rle_min_run', 3)
            new_pipeline = old_pipeline
            new_seed = old_seed
            new_rle_min = old_rle_min
            changed = False

            # 1. rle_min_run perturbation (40% chance)
            if random.random() < 0.40:
                delta = random.choice([-1, 1])
                new_rle_min = max(2, min(7, old_rle_min + delta))
                if new_rle_min != old_rle_min:
                    changed = True

            # 2. Discovery seed toggle/change (15% chance)
            if random.random() < 0.15:
                if old_seed is not None:
                    # 50% chance to remove discovery, 50% to change seed
                    if random.random() < 0.5:
                        new_seed = None
                    else:
                        new_seed = random.randint(0, 2**31)
                else:
                    # Add a discovery transform
                    new_seed = random.randint(0, 2**31)
                changed = True

            # 3. Pipeline swap (8% chance)
            if random.random() < 0.08:
                pipelines = list(NovelCompressionGenerator.PIPELINES.keys())
                pipelines_filtered = [p for p in pipelines if p != old_pipeline]
                if pipelines_filtered:
                    new_pipeline = random.choice(pipelines_filtered)
                    changed = True

            if changed:
                cfn, dfn = _gen._build_pipeline(new_pipeline, discovery_seed=new_seed, rle_min_run=new_rle_min)
                if _gen._verify_invertibility(cfn, dfn):
                    # Update the method's metadata and functions
                    novel.metadata['pipeline'] = new_pipeline
                    novel.metadata['discovery_seed'] = new_seed
                    novel.metadata['rle_min_run'] = new_rle_min
                    novel.metadata['steps'] = list(NovelCompressionGenerator.PIPELINES.get(new_pipeline, ['rle']))
                    novel.compress_fn = cfn
                    novel.decompress_fn = dfn
                    ai_core._novel_compress_fn = cfn
                    ai_core._novel_decompress_fn = dfn
                    # Update description
                    disc_str = f" + discovery(seed={new_seed})" if new_seed else ""
                    novel.description = f"Pipeline: {new_pipeline}{disc_str}, min_run={new_rle_min}"
                    mutated = True
                # else: verification failed — keep original method
        except Exception:
            pass  # Non-fatal: keep original method

    # --- NN weight mutation (auto-applied if agent is a DQN agent) ---
    nn_mutated = mutate_nn_weights(evolving_agent, mutation_rate_config)
    if nn_mutated:
        mutated = True

    return mutated


def apply_hypermutation(evolving_agent, hyper_config: dict = None):
    ai_core = evolving_agent.puffin_ai

    if hyper_config is None: hyper_config = {}

    param_strength_factor = hyper_config.get('HYPERMUTATION_PARAM_STRENGTH_FACTOR', 2.5)
    threshold_change_prob = hyper_config.get('HYPERMUTATION_THRESHOLD_COUNT_CHANGE_PROB', 0.4)

    param_configs_local = [
        {'name': 'learning_rate', 'min': 0.00001, 'max': 0.7, 'is_int': False, 'strength': 0.1 * param_strength_factor},
        {'name': 'discount_factor', 'min': 0.5, 'max': 0.99999, 'is_int': False,
         'strength': 0.05 * param_strength_factor},
        {'name': 'exploration_rate', 'min': 0.005, 'max': 1.0, 'is_int': False,
         'strength': 0.15 * param_strength_factor},
        {'name': 'exploration_decay_rate', 'min': 0.97, 'max': 0.99999, 'is_int': False,
         'strength': 0.005 * param_strength_factor},
        {'name': 'min_exploration_rate', 'min': 0.00001, 'max': 0.25, 'is_int': False,
         'strength': 0.05 * param_strength_factor},
    ]

    for p_conf in param_configs_local:
        if hasattr(ai_core, p_conf['name']):
            current_val = getattr(ai_core, p_conf['name'])
            mutated_val = mutate_parameter(current_val, p_conf['min'], p_conf['max'], p_conf['strength'],
                                           p_conf['is_int'])
            setattr(ai_core, p_conf['name'], mutated_val)

    if hasattr(ai_core, 'len_thresholds'):
        original_thresholds_hyper = tuple(ai_core.len_thresholds)
        max_thresh_count_cfg = getattr(ai_core, 'MAX_THRESHOLDS_COUNT', 7)
        min_thresh_count_cfg = getattr(ai_core, 'MIN_THRESHOLDS_COUNT', 1)
        ai_core.len_thresholds = mutate_thresholds(
            thresholds=ai_core.len_thresholds,
            mutation_rate=0.4,
            min_thresh_val=1, max_thresh_val=5000,
            max_thresh_count_config=max_thresh_count_cfg,
            min_thresh_count_config=min_thresh_count_cfg,
            prob_add_remove_thresh=threshold_change_prob
        )
        if tuple(ai_core.len_thresholds) != original_thresholds_hyper:
            if hasattr(ai_core, '_reinitialize_state_dependent_vars'):
                # Preserve Q-values across threshold change during hypermutation
                old_q = ai_core.q_table.copy() if hasattr(ai_core, 'q_table') and ai_core.q_table is not None else None
                old_state_size = ai_core.state_space_size if hasattr(ai_core, 'state_space_size') else 0
                ai_core._reinitialize_state_dependent_vars()
                if old_q is not None and old_state_size > 0:
                    copy_rows = min(old_q.shape[0], ai_core.q_table.shape[0])
                    copy_cols = min(old_q.shape[1], ai_core.q_table.shape[1])
                    ai_core.q_table[:copy_rows, :copy_cols] = old_q[:copy_rows, :copy_cols]

    if hasattr(ai_core, 'rle_min_encodable_run_length'):
        rle_min_run_b_min = getattr(ai_core, 'RLE_MIN_RUN_BOUNDS_MIN', 2)
        rle_min_run_b_max = getattr(ai_core, 'RLE_MIN_RUN_BOUNDS_MAX', 7)
        change_hyper = random.choice([-2, -1, -1, 1, 1, 2, 0])  # Allow no change sometimes even in hyper
        new_rle_min_hyper = ai_core.rle_min_encodable_run_length + change_hyper
        ai_core.rle_min_encodable_run_length = max(rle_min_run_b_min, min(rle_min_run_b_max, new_rle_min_hyper))

    # --- NN hypermutation (auto-applied if agent is a DQN agent) ---
    hypermutate_nn_weights(evolving_agent, hyper_config)

    # The ELS framework is responsible for ensuring the agent has the mutated ai_core,
    # usually by working on the ai_core directly obtained from evolving_agent.puffin_ai


# ---------------------------------------------------------------------------
# Neural-network weight mutation
# ---------------------------------------------------------------------------

def mutate_nn_weights(evolving_agent, mutation_rate_config: dict = None) -> bool:
    """Apply Gaussian noise to neural-network weights of a PuffinZipAI_NN agent.

    For each parameter tensor in the policy network, with probability
    ``NN_MUTATION_WEIGHT_PROB`` (from config), additive Gaussian noise with
    std ``NN_MUTATION_WEIGHT_NOISE_STD`` is applied.

    After mutation the target network is synced to the mutated policy net.

    Returns True if any weights were mutated, False otherwise.
    """
    try:
        from ..nn_core.nn_agent import PuffinZipAI_NN
    except ImportError:
        return False

    ai_core = evolving_agent.puffin_ai
    if not isinstance(ai_core, PuffinZipAI_NN):
        return False

    try:
        import torch
        from ..config import NN_MUTATION_WEIGHT_NOISE_STD, NN_MUTATION_WEIGHT_PROB
    except ImportError:
        return False

    if mutation_rate_config is None:
        mutation_rate_config = {}

    # Allow per-call overrides
    noise_std = mutation_rate_config.get('nn_noise_std', NN_MUTATION_WEIGHT_NOISE_STD)
    weight_prob = mutation_rate_config.get('nn_weight_prob', NN_MUTATION_WEIGHT_PROB)

    mutated = False
    with torch.no_grad():
        for name, param in ai_core._policy_net.named_parameters():
            if random.random() < weight_prob:
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)
                mutated = True

    if mutated:
        # Re-sync target network after weight mutation
        ai_core._target_net.hard_sync_from(ai_core._policy_net)

    return mutated


def hypermutate_nn_weights(evolving_agent, hyper_config: dict = None) -> bool:
    """Aggressive neural-network weight mutation (hypermutation).

    Uses 3× the normal noise standard deviation and mutates ALL parameter
    layers unconditionally.  Called when the population is stagnating.
    """
    try:
        from ..nn_core.nn_agent import PuffinZipAI_NN
    except ImportError:
        return False

    ai_core = evolving_agent.puffin_ai
    if not isinstance(ai_core, PuffinZipAI_NN):
        return False

    try:
        import torch
        from ..config import NN_MUTATION_WEIGHT_NOISE_STD
    except ImportError:
        return False

    if hyper_config is None:
        hyper_config = {}

    strength_factor = hyper_config.get('HYPERMUTATION_PARAM_STRENGTH_FACTOR', 3.0)
    noise_std = NN_MUTATION_WEIGHT_NOISE_STD * strength_factor

    with torch.no_grad():
        for param in ai_core._policy_net.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)

    # Re-sync target network
    ai_core._target_net.hard_sync_from(ai_core._policy_net)
    return True