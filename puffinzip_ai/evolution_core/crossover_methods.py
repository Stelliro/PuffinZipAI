# PuffinZipAI_Project/puffinzip_ai/evolution_core/crossover_methods.py
import random
import numpy as np
from ..config import (
    ADVANCED_CROSSOVER_PROBABILITY,
    RLE_MIN_RUN_BOUNDS_MIN,
    RLE_MIN_RUN_BOUNDS_MAX
)
from . import breeding_methods


def _resize_q_table(q_table, target_rows):
    """Resize Q-table state dimension via linear interpolation, preserving learned knowledge."""
    if q_table is None:
        return None
    current_rows, cols = q_table.shape
    if current_rows == target_rows:
        return np.copy(q_table)
    if current_rows == 0 or target_rows == 0:
        return np.zeros((target_rows, cols))
    result = np.zeros((target_rows, cols))
    for col in range(cols):
        result[:, col] = np.interp(
            np.linspace(0, 1, target_rows),
            np.linspace(0, 1, current_rows),
            q_table[:, col]
        )
    return result

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

    # --- STEP 1: Q-TABLE CROSSOVER (resize to common shape if needed) ---
    child1_q_crossed = None
    child2_q_crossed = None

    if hasattr(parent1_ai, 'q_table') and parent1_ai.q_table is not None and \
       hasattr(parent2_ai, 'q_table') and parent2_ai.q_table is not None:

        q1 = parent1_ai.q_table
        q2 = parent2_ai.q_table

        # Resize smaller Q-table to match the larger one for meaningful crossover
        if q1.shape != q2.shape:
            target_rows = max(q1.shape[0], q2.shape[0])
            q1 = _resize_q_table(q1, target_rows)
            q2 = _resize_q_table(q2, target_rows)

        # Now shapes match — apply crossover
        if use_advanced_breeding_strategies and hasattr(breeding_methods, 'fitness_weighted_q_table_crossover'):
            child1_q_crossed, child2_q_crossed = breeding_methods.fitness_weighted_q_table_crossover(
                q1, q2, parent1_fitness, parent2_fitness
            )
        else:
            q_crossover_function_standard = random.choice(
                [q_table_uniform_crossover, q_table_average_crossover, q_table_single_point_crossover]
            )
            child1_q_crossed, child2_q_crossed = q_crossover_function_standard(q1, q2)

    # --- STEP 2: THRESHOLD CROSSOVER ---
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

    # Reinitialize state-dependent vars (recomputes state_space_size + zeros Q-table for new shape)
    child1_ai_instance._reinitialize_state_dependent_vars()
    child2_ai_instance._reinitialize_state_dependent_vars()

    # --- STEP 3: Restore crossed Q-tables, resized to match the child's new state space ---
    if child1_q_crossed is not None and child1_ai_instance.q_table is not None:
        child1_ai_instance.q_table = _resize_q_table(child1_q_crossed, child1_ai_instance.q_table.shape[0])
    if child2_q_crossed is not None and child2_ai_instance.q_table is not None:
        child2_ai_instance.q_table = _resize_q_table(child2_q_crossed, child2_ai_instance.q_table.shape[0])

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

    # --- NN WEIGHT CROSSOVER ---
    _apply_nn_crossover(parent1_ai, parent2_ai, child1_ai_instance, child2_ai_instance)

    return child1_ai_instance, child2_ai_instance


# ---------------------------------------------------------------------------
# Neural-network weight crossover
# ---------------------------------------------------------------------------

def _apply_nn_crossover(parent1_ai, parent2_ai, child1_ai, child2_ai):
    """If both parents are NN agents, perform layer-wise weight crossover.

    For each named parameter tensor in the policy network, with probability
    ``NN_CROSSOVER_LAYER_SWAP_PROB`` the children swap that layer's weights.
    This is analogous to uniform crossover but on NN weight tensors.

    If only one parent is an NN agent, the child inherits that parent's
    weights without crossover.  If neither parent is NN, this is a no-op.
    """
    try:
        from ..nn_core.nn_agent import PuffinZipAI_NN
    except ImportError:
        return  # NN module not available

    p1_is_nn = isinstance(parent1_ai, PuffinZipAI_NN)
    p2_is_nn = isinstance(parent2_ai, PuffinZipAI_NN)
    c1_is_nn = isinstance(child1_ai, PuffinZipAI_NN)
    c2_is_nn = isinstance(child2_ai, PuffinZipAI_NN)

    if not (c1_is_nn and c2_is_nn):
        return  # children are not NN agents — nothing to do

    if not (p1_is_nn and p2_is_nn):
        # Mixed crossover: children already got weights from clone — keep them
        return

    try:
        import torch
        from ..config import NN_CROSSOVER_LAYER_SWAP_PROB
    except ImportError:
        return

    # Layer-wise uniform crossover on policy_net weights
    p1_sd = parent1_ai._policy_net.state_dict()
    p2_sd = parent2_ai._policy_net.state_dict()

    c1_sd = {}
    c2_sd = {}
    for key in p1_sd:
        if key not in p2_sd:
            c1_sd[key] = p1_sd[key].clone()
            c2_sd[key] = p1_sd[key].clone()
            continue
        if random.random() < NN_CROSSOVER_LAYER_SWAP_PROB:
            # Swap: child1 gets parent2's weights, child2 gets parent1's
            c1_sd[key] = p2_sd[key].clone()
            c2_sd[key] = p1_sd[key].clone()
        else:
            # No swap: child1 keeps parent1, child2 keeps parent2
            c1_sd[key] = p1_sd[key].clone()
            c2_sd[key] = p2_sd[key].clone()

    child1_ai._policy_net.load_state_dict(c1_sd)
    child2_ai._policy_net.load_state_dict(c2_sd)

    # Sync target networks to the newly-crossed policy networks
    child1_ai._target_net.hard_sync_from(child1_ai._policy_net)
    child2_ai._target_net.hard_sync_from(child2_ai._policy_net)