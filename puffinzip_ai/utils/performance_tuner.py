# PuffinZipAI_Project/puffinzip_ai/utils/performance_tuner.py
import sys
import time
import platform
import psutil
import logging
import random

tuner_logger = logging.getLogger("puffinzip_ai.performance_tuner")
if not tuner_logger.handlers:
    tuner_logger.setLevel(logging.INFO)
    tuner_logger.addHandler(logging.NullHandler())

DEFAULT_TUNABLE_PARAMS = {
    "AGENTS_PER_THROTTLE_CHECK": 5,
    "ITEMS_PER_THROTTLE_CHECK": 10,
    "THROTTLE_SLEEP_DURATION_BENCH_EVAL": 0.001,
    "RLE_THROTTLE_RUN_LENGTH_THRESHOLD": 2 * 1024 * 1024,
    "RLE_THROTTLE_CHUNK_SIZE": 512 * 1024,
    "RLE_THROTTLE_SLEEP_DURATION": 0.001,
}

PERFORMANCE_TIERS = {
    "LOW_END": {
        "AGENTS_PER_THROTTLE_CHECK": 1,
        "ITEMS_PER_THROTTLE_CHECK": 3,
        "THROTTLE_SLEEP_DURATION_BENCH_EVAL": 0.002,
        "RLE_THROTTLE_RUN_LENGTH_THRESHOLD": 1 * 1024 * 1024,
        "RLE_THROTTLE_CHUNK_SIZE": 128 * 1024,
        "RLE_THROTTLE_SLEEP_DURATION": 0.002,
    },
    "BALANCED": DEFAULT_TUNABLE_PARAMS.copy(),
    "HIGH_END": {
        "AGENTS_PER_THROTTLE_CHECK": 10,
        "ITEMS_PER_THROTTLE_CHECK": 15,
        "THROTTLE_SLEEP_DURATION_BENCH_EVAL": 0.0005,
        "RLE_THROTTLE_RUN_LENGTH_THRESHOLD": 8 * 1024 * 1024,
        "RLE_THROTTLE_CHUNK_SIZE": 1 * 1024 * 1024,
        "RLE_THROTTLE_SLEEP_DURATION": 0.0005,
    }
}

def _simple_cpu_benchmark(iterations=5 * 10 ** 5):
    start_time = time.perf_counter()
    text = "benchmark_string" * 50
    val = 0
    for i in range(iterations):
        s = text + str(i % 1000)
        if "500" in s: val += 1
        val = (val * i) % 999999
    end_time = time.perf_counter()
    return end_time - start_time

def get_system_specs():
    specs = {"cpu_cores_physical": 1, "total_ram_gb": 4}
    try:
        specs["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        specs["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        specs["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        tuner_logger.info(
            f"System Specs: CPU Cores (L/P): {specs['cpu_cores_logical']}/{specs['cpu_cores_physical']}, RAM: {specs['total_ram_gb']}GB")
    except Exception as e:
        tuner_logger.warning(f"Could not get detailed system specs via psutil: {e}. Using conservative defaults.")
    return specs

def suggest_performance_tier() -> str:
    tuner_logger.info("Suggesting performance tier...")
    try:
        _simple_cpu_benchmark(10 ** 4)
        time.sleep(0.05)

        benchmark_duration = _simple_cpu_benchmark()
        tuner_logger.info(f"Simple CPU benchmark duration: {benchmark_duration:.4f} seconds.")

        specs = get_system_specs()
        cpu_cores = specs.get("cpu_cores_physical", 1)
        ram_gb = specs.get("total_ram_gb", 4)

        tier = "BALANCED"
        if benchmark_duration > 0.8:
            tier = "LOW_END"
        elif benchmark_duration < 0.25:
            if cpu_cores >= 8 and ram_gb >= 16:
                tier = "HIGH_END"
            elif cpu_cores >= 4 and ram_gb >= 8:
                tier = "HIGH_END"
            else:
                tier = "BALANCED"
        elif benchmark_duration < 0.5:
            if cpu_cores >= 6 and ram_gb >= 8:
                tier = "BALANCED"
            else:
                tier = "LOW_END"
        else:
            if cpu_cores < 4 or ram_gb < 6: tier = "LOW_END"

        tuner_logger.info(
            f"Suggested performance tier: {tier} (Benchmark: {benchmark_duration:.2f}s, Cores: {cpu_cores}, RAM: {ram_gb}GB)")
        return tier
    except Exception as e_suggest:
        tuner_logger.error(f"Error during performance tier suggestion: {e_suggest}. Defaulting to BALANCED.",
                           exc_info=True)
        return "BALANCED"

def get_tuned_parameters(tier_name: str = None) -> dict:
    if tier_name is None:
        tier_name = suggest_performance_tier()

    if tier_name not in PERFORMANCE_TIERS:
        tuner_logger.warning(f"Unknown performance tier '{tier_name}'. Using BALANCED tier parameters.")
        tier_name = "BALANCED"

    tuned_values = DEFAULT_TUNABLE_PARAMS.copy()
    tuned_values.update(PERFORMANCE_TIERS[tier_name])

    tuner_logger.info(f"Final tuned parameters for tier '{tier_name}': {tuned_values}")
    return tuned_values

if __name__ == '__main__':
    if not tuner_logger.handlers or isinstance(tuner_logger.handlers[0], logging.NullHandler):
        tuner_logger.handlers.clear()
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        tuner_logger.addHandler(ch)
        tuner_logger.setLevel(logging.INFO)

    print("--- Testing Performance Tuner Standalone ---")
    final_params = get_tuned_parameters()
    print("\nAuto-Suggested Tuned Parameters:")
    for key, value in final_params.items(): print(f"  {key}: {value}")

    print("\n--- Testing with Forced 'LOW_END' Tier ---")
    low_params = get_tuned_parameters("LOW_END")
    for key, value in low_params.items(): print(f"  {key}: {value}")

    print("\n--- Testing with Forced 'HIGH_END' Tier ---")
    high_params = get_tuned_parameters("HIGH_END")
    for key, value in high_params.items(): print(f"  {key}: {value}")