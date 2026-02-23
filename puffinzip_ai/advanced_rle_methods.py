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
# SOH control char used as escape/control prefix.  Format is completely unambiguous:
#   Run  (count >= min_run):  DELIM + count_digits + DELIM + char
#   Literal DELIM in data:    DELIM + DELIM
#   Everything else:          literal char
ADV_RLE_DELIMITER = '\x01'


def advanced_rle_compress(text_data: str) -> str:
    """Compress text using delimiter-framed RLE.

    Encoding rules:
      - A run of `count` copies of `char` (count >= MIN_ENCODABLE_RUN_LENGTH_ADVANCED)
        is encoded as  DELIM + str(count) + DELIM + char.
      - A literal DELIM character is escaped as  DELIM + DELIM.
      - All other characters (including digits) are emitted literally.
    """
    if not isinstance(text_data, str):
        adv_rle_logger.error("Input data for advanced RLE compression must be a string.")
        raise TypeError("Input data for RLE compression must be a string.")
    if not text_data:
        return ""

    current_min_run = MIN_ENCODABLE_RUN_LENGTH_ADVANCED
    DELIM = ADV_RLE_DELIMITER
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
            # Framed run: DELIM + count + DELIM + char
            result_parts.append(DELIM)
            result_parts.append(str(count))
            result_parts.append(DELIM)
            result_parts.append(current_char)
        else:
            # Literal character(s) — escape any DELIM occurrences
            for _ in range(count):
                if current_char == DELIM:
                    result_parts.append(DELIM)
                    result_parts.append(DELIM)
                else:
                    result_parts.append(current_char)

    return "".join(result_parts)


def advanced_rle_decompress(compressed_text_data: str) -> str:
    """Decompress delimiter-framed RLE.

    Decoding rules:
      - DELIM + digits + DELIM + char  →  repeat char `digits` times.
      - DELIM + DELIM                  →  literal DELIM character.
      - Any other character             →  literal.
    """
    if not isinstance(compressed_text_data, str):
        adv_rle_logger.error("Input data for advanced RLE decompression must be a string.")
        raise TypeError("Input data for RLE decompression must be a string.")
    if not compressed_text_data:
        return ""

    DELIM = ADV_RLE_DELIMITER
    result_parts = []
    i = 0
    n = len(compressed_text_data)
    total_decompressed_size = 0

    while i < n:
        if len(result_parts) > max(n * 3, 20000):
            adv_rle_logger.error(
                f"Advanced RLE Decomp loop protection. Parts: {len(result_parts)}. Input len: {n}")
            return RLE_ERROR_MALFORMED

        char = compressed_text_data[i]

        if char == DELIM:
            i += 1
            if i >= n:
                # Trailing orphan DELIM — malformed but tolerate as literal
                result_parts.append(DELIM)
                total_decompressed_size += 1
                break

            next_char = compressed_text_data[i]

            if next_char == DELIM:
                # Escaped DELIM → literal DELIM
                result_parts.append(DELIM)
                total_decompressed_size += 1
                i += 1

            elif next_char.isdigit():
                # Run encoding: DELIM + count_digits + DELIM + char
                count_str = ""
                digit_read_count = 0
                while i < n and compressed_text_data[i].isdigit() and digit_read_count < MAX_COUNT_STRING_DIGITS:
                    count_str += compressed_text_data[i]
                    i += 1
                    digit_read_count += 1

                # Too many digits — safety limit
                if digit_read_count == MAX_COUNT_STRING_DIGITS and i < n and compressed_text_data[i].isdigit():
                    adv_rle_logger.error(
                        f"Adv RLE Decomp: Count string exceeds {MAX_COUNT_STRING_DIGITS} digits.")
                    return RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY

                # Expect closing DELIM
                if i >= n or compressed_text_data[i] != DELIM:
                    adv_rle_logger.error("Adv RLE Decomp: Missing closing delimiter after count.")
                    return RLE_ERROR_MALFORMED
                i += 1  # consume closing DELIM

                # Expect the character to repeat
                if i >= n:
                    adv_rle_logger.error("Adv RLE Decomp: Missing char after run header.")
                    return RLE_ERROR_MALFORMED

                char_to_repeat = compressed_text_data[i]
                i += 1

                try:
                    parsed_count = int(count_str)
                except ValueError:
                    return RLE_ERROR_MALFORMED

                if parsed_count > ABSOLUTE_MAX_PARSED_COUNT:
                    adv_rle_logger.error(
                        f"Adv RLE Decomp: parsed_count {parsed_count} EXCEEDS ABSOLUTE_MAX_PARSED_COUNT.")
                    return RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY

                if total_decompressed_size + parsed_count > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
                    return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED

                if parsed_count > ADV_THROTTLE_RUN_LENGTH_THRESHOLD:
                    remaining_count = parsed_count
                    while remaining_count > 0:
                        chunk_len = min(remaining_count, ADV_THROTTLE_CHUNK_SIZE)
                        try:
                            result_parts.append(char_to_repeat * chunk_len)
                        except MemoryError:
                            return RLE_ERROR_MEMORY_ON_CHUNK
                        total_decompressed_size += chunk_len
                        remaining_count -= chunk_len
                        if remaining_count > 0:
                            time.sleep(ADV_THROTTLE_SLEEP_DURATION)
                else:
                    try:
                        result_parts.append(char_to_repeat * parsed_count)
                    except MemoryError:
                        return RLE_ERROR_MEMORY_ON_CHUNK
                    total_decompressed_size += parsed_count
            else:
                # DELIM followed by non-digit, non-DELIM — malformed, tolerate
                result_parts.append(DELIM)
                total_decompressed_size += 1
                # Don't consume next_char, it will be processed on next iteration
        else:
            # Literal character (digits, letters, anything — all literal outside DELIM)
            result_parts.append(char)
            total_decompressed_size += 1
            if total_decompressed_size > ABSOLUTE_MAX_TOTAL_DECOMPRESSED_SIZE:
                return RLE_ERROR_TOTAL_SIZE_LIMIT_EXCEEDED
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

    D = ADV_RLE_DELIMITER  # shorthand for expected values
    test_cases = [
        ("A", "A"), ("AA", f"{D}2{D}A"), ("AAA", f"{D}3{D}A"),
        ("AAAAABBBCCCD", f"{D}5{D}A{D}3{D}B{D}3{D}CD"), ("ABC", "ABC"),
        ("AABBCC", f"{D}2{D}A{D}2{D}B{D}2{D}C"),
        ("11122", f"{D}3{D}1{D}2{D}2"), ("A11A", f"A{D}2{D}1A"),
        ("TEST111END", f"TEST{D}3{D}1END")
    ]

    all_passed = True
    for i, (original, expected_compressed) in enumerate(test_cases):
        compressed = advanced_rle_compress(original)
        decompressed = ""
        try:
            decompressed = advanced_rle_decompress(compressed)
        except Exception as e_decomp:
            decompressed = f"DECOMP_ERROR: {e_decomp}"

        roundtrip_ok = (original == decompressed)
        compress_ok = (compressed == expected_compressed)
        print(f"\nTest Case AD-{i + 1}: Orig='{original}'")
        print(f"  Compr: {repr(compressed)} (Exp: {repr(expected_compressed)})")
        print(f"  Decompr: '{decompressed}'")

        if roundtrip_ok and compress_ok:
            print(f"  Status: PASS")
        elif roundtrip_ok and not compress_ok:
            print(f"  Status: PASS (roundtrip OK, compressed format differs)")
        else:
            all_passed = False
            print(f"  Status: FAIL!!!")
            if not roundtrip_ok:
                print(f"    Original '{original}' != Decompressed '{decompressed}'")

    print("\n--- Malformed/Edge Case Decompression (Advanced RLE) ---")
    # In the new delimiter-framed format, strings without DELIM are all-literal
    malformed_tests_advanced = [
        # Plain digits: no DELIM, so all literal passthrough
        ("123456789X", "123456789X"),
        ("9" * (MAX_COUNT_STRING_DIGITS + 2) + "Y", "9" * (MAX_COUNT_STRING_DIGITS + 2) + "Y"),
        # Proper run encoding roundtrip
        (f"{D}{ABSOLUTE_MAX_PARSED_COUNT + 1}{D}A", RLE_ERROR_COUNT_TOO_LARGE_FOR_SAFETY),
        # Valid large run within limits
        (f"{D}100{D}Z", "Z" * 100),
        # Escaped DELIM roundtrip
        (f"{D}{D}", D),
    ]

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