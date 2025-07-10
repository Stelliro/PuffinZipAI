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
                ai_core._reinitialize_state_dependent_vars()
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
                ai_core._reinitialize_state_dependent_vars()

    if hasattr(ai_core, 'rle_min_encodable_run_length'):
        rle_min_run_b_min = getattr(ai_core, 'RLE_MIN_RUN_BOUNDS_MIN', 2)
        rle_min_run_b_max = getattr(ai_core, 'RLE_MIN_RUN_BOUNDS_MAX', 7)
        change_hyper = random.choice([-2, -1, -1, 1, 1, 2, 0])  # Allow no change sometimes even in hyper
        new_rle_min_hyper = ai_core.rle_min_encodable_run_length + change_hyper
        ai_core.rle_min_encodable_run_length = max(rle_min_run_b_min, min(rle_min_run_b_max, new_rle_min_hyper))

    # The ELS framework is responsible for ensuring the agent has the mutated ai_core,
    # usually by working on the ai_core directly obtained from evolving_agent.puffin_ai