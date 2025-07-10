# PuffinZipAI_Project/puffinzip_ai/evolution_core/breeding_methods.py
import random
import numpy as np
from ..config import MAX_THRESHOLDS_COUNT_MERGED, MIN_THRESHOLDS_COUNT

if 'MAX_THRESHOLDS_COUNT_MERGED' not in globals():
    MAX_THRESHOLDS_COUNT_MERGED = 8
if 'MIN_THRESHOLDS_COUNT' not in globals():
    MIN_THRESHOLDS_COUNT = 1


def complex_threshold_crossover(thresholds1: list, thresholds2: list):
    combined_thresholds = sorted(list(set(thresholds1 + thresholds2)))

    if len(combined_thresholds) > MAX_THRESHOLDS_COUNT_MERGED:
        to_remove_count = len(combined_thresholds) - MAX_THRESHOLDS_COUNT_MERGED
        temp_thresholds = list(combined_thresholds)

        for _ in range(to_remove_count):
            if len(temp_thresholds) <= MIN_THRESHOLDS_COUNT: break

            min_interval = float('inf')
            removal_candidate_idx = -1

            if len(temp_thresholds) == 1: break

            current_intervals = []
            for i in range(len(temp_thresholds) - 1):
                current_intervals.append(temp_thresholds[i + 1] - temp_thresholds[i])

            if not current_intervals: break

            min_interval_val = min(current_intervals)

            indices_of_min_interval = [i for i, interval in enumerate(current_intervals) if
                                       interval == min_interval_val]

            if not indices_of_min_interval: break

            chosen_min_interval_start_idx = random.choice(indices_of_min_interval)

            if random.random() < 0.5:
                removal_candidate_idx = chosen_min_interval_start_idx
            else:
                removal_candidate_idx = chosen_min_interval_start_idx + 1

            if 0 <= removal_candidate_idx < len(temp_thresholds):
                temp_thresholds.pop(removal_candidate_idx)
            elif temp_thresholds and len(temp_thresholds) > MAX_THRESHOLDS_COUNT_MERGED:
                temp_thresholds.pop(random.randrange(len(temp_thresholds)))

        combined_thresholds = sorted(list(set(temp_thresholds)))

    if not combined_thresholds:
        fallback_base = thresholds1 if thresholds1 else (thresholds2 if thresholds2 else [])
        if not fallback_base: fallback_base = [random.randint(10, 100)]
        combined_thresholds = sorted(list(set(fallback_base)))
        if len(combined_thresholds) > MAX_THRESHOLDS_COUNT_MERGED:
            combined_thresholds = combined_thresholds[:MAX_THRESHOLDS_COUNT_MERGED]
        elif len(combined_thresholds) < MIN_THRESHOLDS_COUNT:
            while len(combined_thresholds) < MIN_THRESHOLDS_COUNT:
                new_val_base = combined_thresholds[-1] if combined_thresholds else random.randint(5, 50)
                combined_thresholds.append(new_val_base + random.randint(5, 50))
                combined_thresholds = sorted(list(set(combined_thresholds)))

    child1_thresholds = [max(1, int(t)) for t in combined_thresholds]
    child2_thresholds = [max(1, int(t)) for t in combined_thresholds]

    if not child1_thresholds: child1_thresholds = [random.randint(10, 50) for _ in range(MIN_THRESHOLDS_COUNT)]
    if not child2_thresholds: child2_thresholds = [random.randint(10, 50) for _ in range(MIN_THRESHOLDS_COUNT)]

    child1_thresholds = sorted(list(set(child1_thresholds)))
    child2_thresholds = sorted(list(set(child2_thresholds)))

    return child1_thresholds, child2_thresholds


def fitness_weighted_parameter_crossover(params1: list, params2: list, fitness1: float, fitness2: float,
                                         param_configs: list):
    f1_norm, f2_norm = fitness1, fitness2

    min_fitness_floor = -100.0
    f1_norm = max(f1_norm, min_fitness_floor)
    f2_norm = max(f2_norm, min_fitness_floor)

    shift = 0
    if f1_norm < 1e-9 or f2_norm < 1e-9:
        shift = abs(min(f1_norm, f2_norm, 0.0)) + 1e-6
        f1_norm += shift
        f2_norm += shift

    total_fit = f1_norm + f2_norm
    w1, w2 = 0.5, 0.5
    if total_fit > 1e-9:
        w1 = f1_norm / total_fit
        w2 = f2_norm / total_fit

    child1_params, child2_params = [], []

    for i, (p1_val, p2_val) in enumerate(zip(params1, params2)):
        config = param_configs[i] if i < len(param_configs) else {}
        min_val, max_val = config.get('min'), config.get('max')
        is_int = config.get('is_int', isinstance(p1_val, int) and isinstance(p2_val, int))

        if isinstance(p1_val, (int, float)) and isinstance(p2_val, (int, float)):
            c1_val = p1_val * w1 + p2_val * w2
            c2_val = p1_val * w2 + p2_val * w1

            if min_val is not None:
                c1_val = max(min_val, c1_val)
                c2_val = max(min_val, c2_val)
            if max_val is not None:
                c1_val = min(max_val, c1_val)
                c2_val = min(max_val, c2_val)

            child1_params.append(int(round(c1_val)) if is_int else c1_val)
            child2_params.append(int(round(c2_val)) if is_int else c2_val)
        else:
            child1_params.append(p1_val if random.random() < w1 else p2_val)
            child2_params.append(p2_val if random.random() < w2 else p1_val)

    return child1_params, child2_params


def fitness_weighted_q_table_crossover(q1: np.ndarray, q2: np.ndarray, fitness1: float, fitness2: float):
    if q1 is None or q2 is None or q1.shape != q2.shape:
        c1q = np.copy(q1) if q1 is not None else (np.copy(q2) if q2 is not None else None)
        c2q = np.copy(q2) if q2 is not None else (np.copy(q1) if q1 is not None else None)
        return c1q, c2q

    f1_norm, f2_norm = fitness1, fitness2
    min_fitness_floor = -100.0
    f1_norm = max(f1_norm, min_fitness_floor)
    f2_norm = max(f2_norm, min_fitness_floor)

    shift = 0
    if f1_norm < 1e-9 or f2_norm < 1e-9:
        shift = abs(min(f1_norm, f2_norm, 0.0)) + 1e-6
        f1_norm += shift
        f2_norm += shift

    total_fit = f1_norm + f2_norm
    w1 = 0.5
    if total_fit > 1e-9:
        w1 = f1_norm / total_fit
    w2 = 1.0 - w1

    child1_q = w1 * q1 + w2 * q2
    child2_q = w2 * q1 + w1 * q2

    return child1_q, child2_q