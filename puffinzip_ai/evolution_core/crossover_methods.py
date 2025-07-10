# PuffinZipAI_Project/puffinzip_ai/evolution_core/crossover_methods.py
import random
import numpy as np
from ..config import (
    ADVANCED_CROSSOVER_PROBABILITY,
    RLE_MIN_RUN_BOUNDS_MIN,
    RLE_MIN_RUN_BOUNDS_MAX
)
from . import breeding_methods

def q_table_single_point_crossover(q_table1: np.ndarray, q_table2: np.ndarray):
    if q_table1 is None or q_table2 is None or q_table1.shape != q_table2.shape:
        c1q = np.copy(q_table1) if q_table1 is not None else (np.copy(q_table2) if q_table2 is not None else None)
        c2q = np.copy(q_table2) if q_table2 is not None else (np.copy(q_table1) if q_table1 is not None else None)
        return c1q, c2q

    rows, cols = q_table1.shape
    child1_q = np.copy(q_table1)
    child2_q = np.copy(q_table2)

    if rows > 1:
        crossover_point_row = random.randint(1, rows - 1)
        child1_q[crossover_point_row:] = q_table2[crossover_point_row:]
        child2_q[crossover_point_row:] = q_table1[crossover_point_row:]
    elif cols > 1:
        crossover_point_col = random.randint(1, cols - 1)
        child1_q[:, crossover_point_col:] = q_table2[:, crossover_point_col:]
        child2_q[:, crossover_point_col:] = q_table1[:, crossover_point_col:]
    return child1_q, child2_q

def q_table_average_crossover(q_table1: np.ndarray, q_table2: np.ndarray):
    if q_table1 is None or q_table2 is None or q_table1.shape != q_table2.shape:
        c1q = np.copy(q_table1) if q_table1 is not None else (np.copy(q_table2) if q_table2 is not None else None)
        c2q = np.copy(q_table2) if q_table2 is not None else (np.copy(q_table1) if q_table1 is not None else None)
        return c1q, c2q
    child1_q = (q_table1 + q_table2) / 2.0
    child2_q = (q_table1 + q_table2) / 2.0
    return child1_q, child2_q

def q_table_uniform_crossover(q_table1: np.ndarray, q_table2: np.ndarray, p: float = 0.5):
    if q_table1 is None or q_table2 is None or q_table1.shape != q_table2.shape:
        c1q = np.copy(q_table1) if q_table1 is not None else (np.copy(q_table2) if q_table2 is not None else None)
        c2q = np.copy(q_table2) if q_table2 is not None else (np.copy(q_table1) if q_table1 is not None else None)
        return c1q, c2q

    child1_q = np.copy(q_table1)
    child2_q = np.copy(q_table2)
    mask = np.random.rand(*q_table1.shape) < p
    temp_child1_masked_values = np.copy(child1_q[mask])
    child1_q[mask] = child2_q[mask]
    child2_q[mask] = temp_child1_masked_values
    return child1_q, child2_q

def parameter_blend_crossover(param_list1: list, param_list2: list, alpha: float = None):
    use_random_alpha_per_param = alpha is None
    default_alpha_if_not_random = 0.5

    if not param_list1 or not param_list2 or len(param_list1) != len(param_list2):
        return list(param_list1) if param_list1 else [], list(param_list2) if param_list2 else []

    child1_params = []
    child2_params = []
    for p1_val, p2_val in zip(param_list1, param_list2):
        current_alpha = random.random() if use_random_alpha_per_param else \
                        (alpha if alpha is not None else default_alpha_if_not_random)

        if isinstance(p1_val, (int, float)) and isinstance(p2_val, (int, float)):
            blend1 = p1_val * current_alpha + p2_val * (1 - current_alpha)
            blend2 = p2_val * current_alpha + p1_val * (1 - current_alpha)
            child1_params.append(int(round(blend1)) if isinstance(p1_val, int) and isinstance(p2_val, int) else blend1)
            child2_params.append(int(round(blend2)) if isinstance(p1_val, int) and isinstance(p2_val, int) else blend2)
        else:
            if random.random() < 0.5:
                child1_params.append(p1_val)
                child2_params.append(p2_val)
            else:
                child1_params.append(p2_val)
                child2_params.append(p1_val)
    return child1_params, child2_params

def parameter_single_point_crossover(param_list1: list, param_list2: list):
    if not param_list1 or not param_list2 or len(param_list1) != len(param_list2) or len(param_list1) < 2:
        return list(param_list1) if param_list1 else [], list(param_list2) if param_list2 else []
    n = len(param_list1)
    crossover_point = random.randint(1, n - 1)
    child1_params = param_list1[:crossover_point] + param_list2[crossover_point:]
    child2_params = param_list2[:crossover_point] + param_list1[crossover_point:]
    return child1_params, child2_params

def apply_crossover(parent1_ai, parent2_ai, parent1_fitness, parent2_fitness, els_logger, els_config):
    child1_ai_instance = parent1_ai.clone_core_model()
    child2_ai_instance = parent2_ai.clone_core_model()

    use_advanced_breeding_strategies = random.random() < ADVANCED_CROSSOVER_PROBABILITY

    if hasattr(parent1_ai, 'q_table') and parent1_ai.q_table is not None and \
       hasattr(parent2_ai, 'q_table') and parent2_ai.q_table is not None:

        if parent1_ai.q_table.shape == parent2_ai.q_table.shape:
            if use_advanced_breeding_strategies and hasattr(breeding_methods, 'fitness_weighted_q_table_crossover'):
                child1_q_new, child2_q_new = breeding_methods.fitness_weighted_q_table_crossover(
                    parent1_ai.q_table, parent2_ai.q_table, parent1_fitness, parent2_fitness
                )
            else:
                q_crossover_function_standard = random.choice(
                    [q_table_uniform_crossover, q_table_average_crossover, q_table_single_point_crossover]
                )
                child1_q_new, child2_q_new = q_crossover_function_standard(parent1_ai.q_table, parent2_ai.q_table)

            if child1_ai_instance.q_table is not None and child1_ai_instance.q_table.shape == child1_q_new.shape:
                child1_ai_instance.q_table = child1_q_new
            if child2_ai_instance.q_table is not None and child2_ai_instance.q_table.shape == child2_q_new.shape:
                child2_ai_instance.q_table = child2_q_new
        else:
            if els_logger: els_logger.warning("Q-Table shape mismatch between parents. Q-Table crossover skipped (cloned).")

    original_child1_thresholds = tuple(child1_ai_instance.len_thresholds)
    original_child2_thresholds = tuple(child2_ai_instance.len_thresholds)

    if hasattr(parent1_ai, 'len_thresholds') and hasattr(parent2_ai, 'len_thresholds'):
        if use_advanced_breeding_strategies and hasattr(breeding_methods, 'complex_threshold_crossover'):
            child1_thresh_new, child2_thresh_new = breeding_methods.complex_threshold_crossover(
                list(parent1_ai.len_thresholds), list(parent2_ai.len_thresholds)
            )
        else:
            child1_thresh_new, child2_thresh_new = parameter_single_point_crossover(
                list(parent1_ai.len_thresholds), list(parent2_ai.len_thresholds)
            )
        child1_ai_instance.len_thresholds = child1_thresh_new if child1_thresh_new else list(parent1_ai.len_thresholds)
        child2_ai_instance.len_thresholds = child2_thresh_new if child2_thresh_new else list(parent2_ai.len_thresholds)

    if tuple(child1_ai_instance.len_thresholds) != original_child1_thresholds:
        child1_ai_instance._reinitialize_state_dependent_vars()
    if tuple(child2_ai_instance.len_thresholds) != original_child2_thresholds:
        child2_ai_instance._reinitialize_state_dependent_vars()

    params_to_cross_config = [
        {'name': 'learning_rate', 'min': 0.00001, 'max': 0.7, 'is_int': False},
        {'name': 'discount_factor', 'min': 0.5, 'max': 0.99999, 'is_int': False},
        {'name': 'exploration_rate', 'min': 0.005, 'max': 1.0, 'is_int': False},
        {'name': 'exploration_decay_rate', 'min': 0.97, 'max': 0.99999, 'is_int': False},
        {'name': 'min_exploration_rate', 'min': 0.00001, 'max': 0.25, 'is_int': False},
        {'name': 'rle_min_encodable_run_length',
         'min': RLE_MIN_RUN_BOUNDS_MIN, 'max': RLE_MIN_RUN_BOUNDS_MAX, 'is_int': True}
    ]

    parent1_param_values = [getattr(parent1_ai, p_conf['name'], None) for p_conf in params_to_cross_config]
    parent2_param_values = [getattr(parent2_ai, p_conf['name'], None) for p_conf in params_to_cross_config]
    param_meta_for_crossover = [{'min': p_conf['min'], 'max': p_conf['max'], 'is_int': p_conf['is_int']} for p_conf in params_to_cross_config]

    if all(v is not None for v in parent1_param_values) and all(v is not None for v in parent2_param_values):
        if use_advanced_breeding_strategies and hasattr(breeding_methods, 'fitness_weighted_parameter_crossover'):
            child1_crossed_params, child2_crossed_params = breeding_methods.fitness_weighted_parameter_crossover(
                parent1_param_values, parent2_param_values, parent1_fitness, parent2_fitness, param_meta_for_crossover
            )
        else:
            child1_crossed_params, child2_crossed_params = parameter_blend_crossover(
                parent1_param_values, parent2_param_values, alpha=None
            )

        for i, param_config_item in enumerate(params_to_cross_config):
            param_name_to_set = param_config_item['name']
            val_for_child1 = child1_crossed_params[i]
            val_for_child2 = child2_crossed_params[i]

            if param_config_item['min'] is not None:
                val_for_child1 = max(param_config_item['min'], val_for_child1)
                val_for_child2 = max(param_config_item['min'], val_for_child2)
            if param_config_item['max'] is not None:
                val_for_child1 = min(param_config_item['max'], val_for_child1)
                val_for_child2 = min(param_config_item['max'], val_for_child2)

            setattr(child1_ai_instance, param_name_to_set, val_for_child1)
            setattr(child2_ai_instance, param_name_to_set, val_for_child2)

        child1_ai_instance.min_exploration_rate = min(child1_ai_instance.min_exploration_rate, child1_ai_instance.exploration_rate * 0.9)
        child2_ai_instance.min_exploration_rate = min(child2_ai_instance.min_exploration_rate, child2_ai_instance.exploration_rate * 0.9)

    return child1_ai_instance, child2_ai_instance