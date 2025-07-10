# PuffinZipAI_Project/puffinzip_ai/reward_system.py
PENALTY_MISMATCH = -10.0
PENALTY_RLE_PROCESSING_ERROR = -12.0
PENALTY_FOR_USELESS_RLE_ATTEMPT = -0.2
PENALTY_SCALER_EXPANSION = 3.0

REWARD_SCALER_COMPRESSION_SUCCESS = 10.0
REWARD_EMPTY_INPUT_HANDLED_CORRECTLY = 0.1
BASE_REWARD_NOCOMPRESSION_CORRECT = 0.0

MAX_TIME_MS_KB_PENALTY_THRESHOLD = 150.0
TIME_PENALTY_VALUE = -0.3

try:
    from .rle_constants import RLE_DECOMPRESSION_ERRORS, ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT
    RLE_ERROR_NO_CHAR = ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT
except ImportError:
    RLE_ERROR_NO_CHAR = "ERROR_INVALID_RLE_FORMAT_NO_CHAR_AFTER_COUNT_RS_FB"
    RLE_DECOMPRESSION_ERRORS = {"ERROR_INVALID_RLE_FORMAT_NO_COUNT", "ERROR_INVALID_RLE_FORMAT_BAD_COUNT",
                                RLE_ERROR_NO_CHAR, "ERROR_MALFORMED_RLE_STRING", "ERROR_COUNT_TOO_LARGE_FOR_SAFETY",
                                "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED", "ERROR_MEMORY_DURING_CHUNK_ALLOCATION",
                                "ERROR_MEMORY_DURING_FINAL_JOIN"}


def calculate_reward(
        original_text: str,
        compressed_text: str,
        decompressed_text: str,
        action_taken: str,
        processing_time_ms: float,
        rle_error_code: str = None
) -> float:
    original_size = len(original_text)
    compressed_size = len(compressed_text)
    reward = 0.0

    if original_size == 0:
        if action_taken in ["RLE", "AdvancedRLE"]:
            return REWARD_EMPTY_INPUT_HANDLED_CORRECTLY if (
                        compressed_text == "" and decompressed_text == "" and rle_error_code is None) else PENALTY_MISMATCH
        elif action_taken == "NoCompression":
            return REWARD_EMPTY_INPUT_HANDLED_CORRECTLY if (
                        compressed_text == "" and decompressed_text == "") else PENALTY_MISMATCH
        else:
            return PENALTY_MISMATCH

    if action_taken in ["RLE", "AdvancedRLE"]:
        if rle_error_code is not None and rle_error_code in RLE_DECOMPRESSION_ERRORS:
            return PENALTY_RLE_PROCESSING_ERROR
        if decompressed_text != original_text:
            return PENALTY_MISMATCH

        if compressed_size < original_size:
            space_saved_ratio = (original_size - compressed_size) / original_size
            reward = REWARD_SCALER_COMPRESSION_SUCCESS * space_saved_ratio
        elif compressed_size > original_size:
            expansion_factor = compressed_size / original_size
            reward = -PENALTY_SCALER_EXPANSION * (expansion_factor - 1)
        else:
            reward = PENALTY_FOR_USELESS_RLE_ATTEMPT

    elif action_taken == "NoCompression":
        if compressed_text == original_text and decompressed_text == original_text:
            reward = BASE_REWARD_NOCOMPRESSION_CORRECT
        else:
            return PENALTY_MISMATCH * 1.5
    else:
        reward = -20.0

    if reward > PENALTY_MISMATCH:
        if original_size > 0:
            kb_size = original_size / 1024.0
            if kb_size > 0.001:
                time_ms_per_kb = processing_time_ms / kb_size
                if time_ms_per_kb > MAX_TIME_MS_KB_PENALTY_THRESHOLD:
                    reward += TIME_PENALTY_VALUE
            elif processing_time_ms > (MAX_TIME_MS_KB_PENALTY_THRESHOLD * 0.05):
                reward += TIME_PENALTY_VALUE / 2
    return reward


if __name__ == '__main__':
    try:
        from .rle_utils import rle_compress
    except ImportError:
        print("Warning: rle_utils not found for reward_system __main__ test. Test limited.")
        rle_compress = lambda x, **kwargs: x

    print("--- Testing Reward System (v2 values & logic, main block corrected) ---")

    reward1 = calculate_reward("AAAAABBBCC", "5A3B2C", "AAAAABBBCC", "RLE", 0.1)
    print(f"Scenario 1 (Good RLE): {reward1:.4f} (Expected ~4.0)")

    reward2 = calculate_reward("ABCDE", "AABBCCDDEE", "ABCDE", "RLE", 0.1)
    print(f"Scenario 2 (RLE Expansion): {reward2:.4f} (Expected -3.0 for 2x expansion)")

    reward3 = calculate_reward("AAAAA", "5A", "AAAAB", "RLE", 0.1)
    print(f"Scenario 3 (RLE Decomp Mismatch): {reward3:.4f} (Expected {PENALTY_MISMATCH})")

    reward4 = calculate_reward("AAAAA", "5", RLE_ERROR_NO_CHAR, "RLE", 0.1, rle_error_code=RLE_ERROR_NO_CHAR)
    print(f"Scenario 4 (RLE Error Code): {reward4:.4f} (Expected {PENALTY_RLE_PROCESSING_ERROR})")

    reward5 = calculate_reward("ABCDE", "ABCDE", "ABCDE", "NoCompression", 0.1)
    print(f"Scenario 5 (NoCompression chosen): {reward5:.4f} (Expected {BASE_REWARD_NOCOMPRESSION_CORRECT})")

    reward6 = calculate_reward("", "", "", "RLE", 0.1)
    print(f"Scenario 6 (Empty Input, RLE): {reward6:.4f} (Expected {REWARD_EMPTY_INPUT_HANDLED_CORRECTLY})")

    reward7 = calculate_reward("", "", "", "NoCompression", 0.1)
    print(f"Scenario 7 (Empty Input, NoComp): {reward7:.4f} (Expected {REWARD_EMPTY_INPUT_HANDLED_CORRECTLY})")

    original_long = "A" * 1000
    comp_long_eff = "1000A"
    reward8_raw_eff = REWARD_SCALER_COMPRESSION_SUCCESS * (
                (len(original_long) - len(comp_long_eff)) / len(original_long)) if len(original_long) > 0 else 0
    print(f"Scenario 8 (RAW Good RLE, very high ratio): {reward8_raw_eff:.4f}")
    reward8 = calculate_reward(original_long, comp_long_eff, original_long, "RLE", 1.0)
    print(f"Scenario 8 (Good RLE, Fast): {reward8:.4f} (Expected approx {reward8_raw_eff:.4f})")

    reward9 = calculate_reward(original_long, comp_long_eff, original_long, "RLE",
                               200.0 * (len(original_long) / 1024.0) + 1)
    print(f"Scenario 9 (Good RLE, Slow): {reward9:.4f} (Expected approx {reward8_raw_eff + TIME_PENALTY_VALUE:.4f})")

    reward10 = calculate_reward("ABCDEF", "ABCDEF", "ABCDEF", "RLE", 1.0)
    print(f"Scenario 10 (RLE No Change): {reward10:.4f} (Expected {PENALTY_FOR_USELESS_RLE_ATTEMPT})")

    print(
        f"\nWith new rewards, always choosing 'NoCompression' correctly would yield an average of: {BASE_REWARD_NOCOMPRESSION_CORRECT:.4f}")

    difficult_data = "a83HskW0&!kAp"
    rle_result_difficult_simple_min2 = "a83HskW0&!kAp";
    rle_result_difficult_adv_min2 = "a83HskW0&!kAp"
    reward_rle_difficult = calculate_reward(difficult_data, rle_result_difficult_simple_min2, difficult_data, "RLE",
                                            1.0)
    print(f"Difficult data, chose RLE, no change: {reward_rle_difficult:.4f}")
    reward_adv_rle_difficult = calculate_reward(difficult_data, rle_result_difficult_adv_min2, difficult_data,
                                                "AdvancedRLE", 1.0)
    print(f"Difficult data, chose AdvRLE, no change: {reward_adv_rle_difficult:.4f}")
    reward_nocomp_difficult = calculate_reward(difficult_data, difficult_data, difficult_data, "NoCompression", 1.0)
    print(f"Difficult data, chose NoComp: {reward_nocomp_difficult:.4f}")

    compressible_data = "A" * 300
    rle_compressible_simple_min2 = "3A"
    if callable(rle_compress):
        rle_compressible_simple_min2 = rle_compress(compressible_data, method="simple",
                                                    min_run_len_override=2)

    reward_rle_compressible = calculate_reward(compressible_data, rle_compressible_simple_min2, compressible_data,
                                               "RLE", 1.0)
    exp_compr_rew = REWARD_SCALER_COMPRESSION_SUCCESS * (
                (len(compressible_data) - len(rle_compressible_simple_min2)) / len(compressible_data)) if len(
        compressible_data) > 0 and len(rle_compressible_simple_min2) < len(
        compressible_data) else PENALTY_FOR_USELESS_RLE_ATTEMPT
    print(f"Compressible data, chose RLE, compressed: {reward_rle_compressible:.4f} (Expected {exp_compr_rew:.4f})")

    reward_nocomp_compressible = calculate_reward(compressible_data, compressible_data, compressible_data,
                                                  "NoCompression", 1.0)
    print(
        f"Compressible data, chose NoComp: {reward_nocomp_compressible:.4f} (Expected {BASE_REWARD_NOCOMPRESSION_CORRECT})")