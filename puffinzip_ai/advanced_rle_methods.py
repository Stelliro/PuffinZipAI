# PuffinZipAI_Project/puffinzip_ai/advanced_rle_methods.py
import logging
import time

RLE_ERROR_MALFORMED = "ERROR_MALFORMED_RLE_STRING"
RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY = "ERROR_COUNT_TOO_LARGE_FOR_SAFETY"
RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED = "ERROR_TOTAL_SIZE_LIMIT_EXCEEDED"
RLE_ERROR_MEMORY_ON_CHUNK = "ERROR_MEMORY_DURING_CHUNK_ALLOCATION"
RLE_ERROR_MEMORY_ON_JOIN = "ERROR_MEMORY_DURING_FINAL_JOIN"
RLE_DECOMPRESSION_ERRORS = {
    RLE_ERROR_MALFORMED, RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY,
    RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED, RLE_ERROR_MEMORY_ON_CHUNK,
    RLE_ERROR_MEMORY_ON_JOIN
}
ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = 200 * 1024 * 1024
ABSOLUTE_MAX_PARSED_COUNT = 100 * 1024 * 1024
MAX_COUNT_STRING_DIGITS = 9

_adv_constants_imported_successfully = False
_constants_source_log_message = "Unknown"

try:
    from .rle_constants import (
        RLE_DECOMPRESSION_ERRORS as RC_RLE_DECOMPRESSION_ERRORS,
        RLE_ERROR_MALFORMED as RC_RLE_ERROR_MALFORMED,
        RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY as RC_RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY,
        RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED as RC_RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED,
        RLE_ERROR_MEMORY_ON_CHUNK as RC_RLE_ERROR_MEMORY_ON_CHUNK,
        RLE_ERROR_MEMORY_ON_JOIN as RC_RLE_ERROR_MEMORY_ON_JOIN,
        ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE as RC_ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE,
        ABSOLUTE_MAX_PARSED_COUNT as RC_ABSOLUTE_MAX_PARSED_COUNT,
        MAX_COUNT_STRING_DIGITS as RC_MAX_COUNT_STRING_DIGITS
    )

    RLE_ERROR_MALFORMED = RC_RLE_ERROR_MALFORMED
    RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY = RC_RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY
    RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED = RC_RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
    RLE_ERROR_MEMORY_ON_CHUNK = RC_RLE_ERROR_MEMORY_ON_CHUNK
    RLE_ERROR_MEMORY_ON_JOIN = RC_RLE_ERROR_MEMORY_ON_JOIN
    RLE_DECOMPRESSION_ERRORS = RC_RLE_DECOMPRESSION_ERRORS
    ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = RC_ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE
    ABSOLUTE_MAX_PARSED_COUNT = RC_ABSOLUTE_MAX_PARSED_COUNT
    MAX_COUNT_STRING_DIGITS = RC_MAX_COUNT_STRING_DIGITS
    _adv_constants_imported_successfully = True
    _constants_source_log_message = "Relative Import (.rle_constants)"
except ImportError:
    _pza_rle_constants_module = None
    _temp_logger_adv_init = logging.getLogger("advanced_rle_methods_init_import_fallback")
    _temp_logger_adv_init.warning("Relative import '.rle_constants' failed. Trying 'puffinzip_ai.rle_constants'.")
    try:
        from puffinzip_ai import rle_constants as pza_rc

        _pza_rle_constants_module = pza_rc
    except ImportError:
        _temp_logger_adv_init.warning("'puffinzip_ai.rle_constants' import also failed.")

    if _pza_rle_constants_module:
        try:
            RLE_ERROR_MALFORMED = _pza_rle_constants_module.RLE_ERROR_MALFORMED
            RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY = _pza_rle_constants_module.RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY
            RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED = _pza_rle_constants_module.RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
            RLE_ERROR_MEMORY_ON_CHUNK = _pza_rle_constants_module.RLE_ERROR_MEMORY_ON_CHUNK
            RLE_ERROR_MEMORY_ON_JOIN = _pza_rle_constants_module.RLE_ERROR_MEMORY_ON_JOIN
            RLE_DECOMPRESSION_ERRORS = _pza_rle_constants_module.RLE_DECOMPRESSION_ERRORS
            ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE = _pza_rle_constants_module.ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE
            ABSOLUTE_MAX_PARSED_COUNT = _pza_rle_constants_module.ABSOLUTE_MAX_PARSED_COUNT
            MAX_COUNT_STRING_DIGITS = _pza_rle_constants_module.MAX_COUNT_STRING_DIGITS
            _adv_constants_imported_successfully = True
            _constants_source_log_message = "Package Absolute Import (puffinzip_ai.rle_constants)"
        except AttributeError as e_attr:
            _temp_logger_adv_init.critical(
                f"CRITICAL (advanced_rle_methods.py): Attrib err from puffinzip_ai.rle_constants ({e_attr}). Using hardcoded fallbacks.")
            _constants_source_log_message = "Hardcoded Fallback (AttributeError during package import)"
    else:
        _temp_logger_adv_init.critical(
            "CRITICAL (advanced_rle_methods.py): ALL imports for rle_constants failed. Using internal hardcoded fallbacks.")
        _constants_source_log_message = "Hardcoded Fallback (All imports failed)"

if not _adv_constants_imported_successfully:
    logging.getLogger("advanced_rle_methods_init").warning(f"Constants from: {_constants_source_log_message}")
else:
    logging.getLogger("advanced_rle_methods_init").info(f"Constants from: {_constants_source_log_message}")

adv_rle_logger = logging.getLogger("puffinzip_ai.advanced_rle_methods")
if not adv_rle_logger.handlers:
    adv_rle_logger.setLevel(logging.WARNING)
    adv_rle_logger.addHandler(logging.NullHandler())

MIN_ENCODABLE_RUN_LENGTH_ADVANCED = 2
ADV_THROTTLE_RUN_LENGTH_THRESHOLD = 1 * 1024 * 1024
ADV_THROTTLE_CHUNK_SIZE = 256 * 1024
ADV_THROTTLE_SLEEP_DURATION = 0.001


def advanced_rle_compress(text_data: str) -> str:
    if not isinstance(text_data, str):
        adv_rle_logger.error("Input data for advanced RLE compression must be a string.")
        raise TypeError("Input data for RLE compression must be a string.")
    if not text_data:
        return ""

    current_min_run = MIN_ENCODABLE_RUN_LENGTH_ADVANCED
    n = len(text_data)
    result_parts = []
    i = 0
    while i < n:
        current_char = text_data[i]
        count = 1
        i += 1
        while i < n and text_data[i] == current_char:
            count += 1
            i += 1

        if count >= current_min_run:
            result_parts.append(str(count))
            result_parts.append(current_char)
        else:
            result_parts.append(current_char * count)

    return "".join(result_parts)


def advanced_rle_decompress(compressed_text_data: str) -> str:
    if not isinstance(compressed_text_data, str):
        adv_rle_logger.error("Input data for advanced RLE decompression must be a string.")
        raise TypeError("Input data for RLE decompression must be a string.")
    if not compressed_text_data:
        return ""

    current_min_run = MIN_ENCODABLE_RUN_LENGTH_ADVANCED
    result_parts = []
    i = 0
    n = len(compressed_text_data)
    total_decompressed_size = 0
    max_result_parts_heuristic = max(n * 3, 20000)

    while i < n:
        if len(result_parts) > max_result_parts_heuristic:
            adv_rle_logger.error(
                f"Advanced RLE Decomp loop protection. Parts: {len(result_parts)}. Input: '{compressed_text_data[:100]}'")
            return RLE_ERROR_MALFORMED

        char = compressed_text_data[i]
        if char.isdigit():
            count_str = ""
            start_of_count_idx = i
            digit_read_count = 0
            while i < n and compressed_text_data[i].isdigit() and digit_read_count < MAX_COUNT_STRING_DIGITS:
                count_str += compressed_text_data[i]
                i += 1
                digit_read_count += 1

            if digit_read_count == MAX_COUNT_STRING_DIGITS and i < n and compressed_text_data[i].isdigit():
                context = compressed_text_data[max(0, start_of_count_idx - 5): min(n, i + 15)]
                adv_rle_logger.warning(
                    f"Adv RLE Decomp: Count str >{MAX_COUNT_STRING_DIGITS} digits, treating as literal. Near: '{context}'")
                while i < n and compressed_text_data[i].isdigit():
                    count_str += compressed_text_data[i];
                    i += 1
                result_parts.append(count_str)
                total_decompressed_size += len(count_str)
                if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
                    adv_rle_logger.error(f"Adv RLE Decomp: Total size {total_decompressed_size} exceeded limit.")
                    return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
                continue

            if not count_str:
                result_parts.append(char);
                total_decompressed_size += 1
                if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE: return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
                i += 1
                continue

            parsed_count = 0
            try:
                parsed_count = int(count_str)
            except ValueError:
                result_parts.append(count_str);
                total_decompressed_size += len(count_str)
                if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE: return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
                continue

            if parsed_count > ABSOLUTE_MAX_PARSED_COUNT:
                context = compressed_text_data[max(0, start_of_count_idx - 10):min(n, i + 10)]
                adv_rle_logger.error(
                    f"Adv RLE Decomp: parsed_count {parsed_count} EXCEEDS ABSOLUTE_MAX_PARSED_COUNT. Error. Near: '{context}'")
                return RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY

            if parsed_count >= current_min_run and i < n:
                char_to_repeat = compressed_text_data[i]

                if total_decompressed_size + parsed_count > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
                    adv_rle_logger.error(
                        f"Adv RLE Decomp: Total size would exceed limit. Run: {parsed_count}{char_to_repeat}")
                    return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED

                if parsed_count > ADV_THROTTLE_RUN_LENGTH_THRESHOLD:
                    adv_rle_logger.info(
                        f"Adv RLE Decomp: Throttling large run. Count: {parsed_count}, Char: '{char_to_repeat}'.")
                    remaining_count = parsed_count
                    while remaining_count > 0:
                        chunk_len = min(remaining_count, ADV_THROTTLE_CHUNK_SIZE)
                        try:
                            result_parts.append(char_to_repeat * chunk_len)
                        except MemoryError:
                            adv_rle_logger.error(f"MemoryError during Adv RLE throttled chunk append.")
                            return RLE_ERROR_MEMORY_ON_CHUNK
                        total_decompressed_size += chunk_len
                        remaining_count -= chunk_len
                        if remaining_count > 0:
                            time.sleep(ADV_THROTTLE_SLEEP_DURATION)
                else:
                    try:
                        result_parts.append(char_to_repeat * parsed_count)
                    except MemoryError:
                        adv_rle_logger.error(f"MemoryError during Adv RLE normal append.")
                        return RLE_ERROR_MEMORY_ON_CHUNK
                    total_decompressed_size += parsed_count
                i += 1
            else:
                result_parts.append(count_str)
                total_decompressed_size += len(count_str)
                if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE: return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
        else:
            result_parts.append(char)
            total_decompressed_size += 1
            if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE: return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
            i += 1

    try:
        final_result = "".join(result_parts)
    except MemoryError:
        adv_rle_logger.error(
            f"MemoryError during final join in Adv RLE. Parts: {len(result_parts)}, Calc total_size: {total_decompressed_size}.")
        return RLE_ERROR_MEMORY_ON_JOIN

    if len(final_result) > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
        adv_rle_logger.error(
            f"Adv RLE Decomp: Final string length {len(final_result)} after join exceeds ABSOLUTE_MAX.")
        return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
    return final_result


if __name__ == '__main__':
    adv_rle_logger_local_main = logging.getLogger("puffinzip_ai.advanced_rle_methods")
    adv_init_logger_local_main = logging.getLogger("advanced_rle_methods_init")

    if not adv_rle_logger_local_main.handlers or isinstance(adv_rle_logger_local_main.handlers[0], logging.NullHandler):
        adv_rle_logger_local_main.handlers.clear()
        adv_rle_logger_local_main.setLevel(logging.INFO)
        ch_main_rle = logging.StreamHandler()
        ch_main_rle.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        adv_rle_logger_local_main.addHandler(ch_main_rle)

    if not adv_init_logger_local_main.handlers or isinstance(adv_init_logger_local_main.handlers[0],
                                                             logging.NullHandler):
        adv_init_logger_local_main.handlers.clear()
        adv_init_logger_local_main.setLevel(logging.INFO)
        ch_main_init = logging.StreamHandler()
        ch_main_init.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        adv_init_logger_local_main.addHandler(ch_main_init)
        adv_init_logger_local_main.info(
            f"Test Main: Configured logger for advanced_rle_methods_init. Constants source message in main log should now be visible.")

    print(f"--- Advanced RLE Methods Tests ---")
    print(
        f"  (Internal MIN_RUN = {MIN_ENCODABLE_RUN_LENGTH_ADVANCED}, Constants source: {_constants_source_log_message})")
    print(
        f"  (ABSOLUTE_MAX_PARSED_COUNT = {ABSOLUTE_MAX_PARSED_COUNT}, MAX_COUNT_STRING_DIGITS = {MAX_COUNT_STRING_DIGITS})")

    test_cases = [
        ("A", "A"), ("AA", "2A"), ("AAA", "3A"),
        ("AAAAABBBCCCD", "5A3B3CD"), ("ABC", "ABC"), ("AABBCC", "2A2B2C"),
        ("11122", "3122"), ("A11A", "A21A"), ("TEST111END", "TEST31END")
    ]

    all_passed = True
    for i, (original, expected_compressed) in enumerate(test_cases):
        compressed = advanced_rle_compress(original)
        decompressed = ""
        try:
            decompressed = advanced_rle_decompress(compressed)
        except Exception as e_decomp:
            decompressed = f"DECOMP_ERROR: {e_decomp}"

        print(f"\nTest Case AD-{i + 1}: Orig='{original}'")
        print(f"  Compr: '{compressed}' (Exp: '{expected_compressed}')")
        print(f"  Decompr: '{decompressed}'")

        if original == decompressed and compressed == expected_compressed:
            print(f"  Status: PASS")
        else:
            all_passed = False
            print(f"  Status: FAIL!!!")
            if original != decompressed:
                print(f"    Original '{original}' != Decompressed '{decompressed}'")
            if compressed != expected_compressed:
                print(f"    Compressed '{compressed}' != Expected '{expected_compressed}'")

    print("\n--- Malformed/Edge Case Decompression (Advanced RLE) ---")
    malformed_tests_advanced = [
        ("123456789X", "123456789X"),
        ("9" * (MAX_COUNT_STRING_DIGITS + 2) + "Y", "9" * (MAX_COUNT_STRING_DIGITS + 2) + "Y"),
        (str(ABSOLUTE_MAX_PARSED_COUNT + 1) + "A", RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY),
        ("1" * MAX_COUNT_STRING_DIGITS + "A", RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY),
        # Default test (may be overridden below)
        ("1" * (MAX_COUNT_STRING_DIGITS + 1) + "B", "1" * (MAX_COUNT_STRING_DIGITS + 1) + "B")
        # Number too long, treat literal + B
    ]

    try:
        one_char_count = int("1" * MAX_COUNT_STRING_DIGITS)
        if MAX_COUNT_STRING_DIGITS > 0 and one_char_count <= ABSOLUTE_MAX_PARSED_COUNT:
            malformed_tests_advanced[3] = ("1" * MAX_COUNT_STRING_DIGITS + "A", "A" * one_char_count)
    except ValueError:
        adv_rle_logger_local_main.error("Malformed test setup error: MAX_COUNT_STRING_DIGITS invalid for int()")

    for i, (compressed_input, expected_output) in enumerate(malformed_tests_advanced):
        decompressed_output_adv = advanced_rle_decompress(compressed_input)
        print(f"\nMalformed/Edge Adv Case {i + 1}: Input='{compressed_input[:50]}...'")
        print(f"  Decompressed (Adv): '{str(decompressed_output_adv)[:50]}...' (Exp: '{str(expected_output)[:50]}...')")
        if decompressed_output_adv == expected_output:
            print(f"  Status: PASS")
        else:
            all_passed = False
            print(f"  Status: FAIL!!! Expected '{expected_output}', Got '{decompressed_output_adv}'")

    print("\n--- Summary ---")
    if all_passed:
        print("All advanced_rle_methods tests PASSED.")
    else:
        print("!!! SOME advanced_rle_methods TESTS FAILED. !!!")

if not _adv_constants_imported_successfully:
    pass
else:
    pass

if not adv_rle_logger.handlers or isinstance(adv_rle_logger.handlers[0], logging.NullHandler):
    pass
else:
    handler_names = [h.__class__.__name__ for h in adv_rle_logger.handlers]
    if 'StreamHandler' in handler_names:
        pass